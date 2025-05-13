import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchmetrics

from dataset import BilingualDataset, casual_mask

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from config import get_weights_file_path, get_config, latest_weights_file_path, latest_weights_file_path

from model import build_transformer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import warnings
# import torchmetrics


def greedy_decode(
    model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device
):
    """
    Performs greedy decoding for the transformer model.

    Args:
        model (torch.nn.Module): The transformer model.
        source (torch.Tensor): The input source sequence tensor.
        source_mask (torch.Tensor): The source mask tensor.
        tokenizer_src: The source tokenizer object.
        tokenizer_tgt: The target tokenizer object.
        max_len (int): The maximum length of the target sequence.
        device (torch.device): The device to run the computation on.

    Returns:
        torch.Tensor: The decoded target sequence as a tensor.
    """
    sos_idx = tokenizer_src.token_to_id("[SOS]")
    eos_idx = tokenizer_src.token_to_id("[EOS]")

    # Precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device=device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for the target(decoder input)
        decoder_mask = (
            casual_mask(decoder_input.size(1)).type_as(source_mask).to(device=device)
        )

        # Calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get the next token
        prob = model.project(out[:, -1])  # Take the -1 last token generated
        _, next_word = torch.max(prob, dim=1)  # ignore the value, take the index only

        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1)
                .type_as(source)
                .fill_(next_word.item())
                .to(device=device),
            ],
            dim=1,
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(
    model,
    validation_ds,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    device,
    print_msg,
    global_step,
    writer,
    num_examples=2,
):
    """
    Run validation on the provided dataset.

    Args:
        model (nn.Module): The model to validate.
        validation_ds (Dataset): The validation dataset.
        tokenizer_src: Tokenizer for the source language.
        tokenizer_tgt: Tokenizer for the target language.
        max_len (int): Maximum length for decoding.
        device (torch.device): Device to perform validation on.
        print_msg (callable): Function to print messages.
        global_step (int): The current global training step.
        writer (SummaryWriter): TensorBoard writer for logging metrics.
        num_examples (int): Number of examples to display during validation.
    """
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    # Determine console width
    try:
        with os.popen("stty size", "r") as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except Exception:
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (b, 1, 1, seq_len)

            # Ensure batch size is 1 for validation
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            # Decode model output
            model_out = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_src,
                tokenizer_tgt,
                max_len,
                device,
            )

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print source, target, and predicted texts
            print_msg("-" * console_width)
            print_msg(f"{'SOURCE:':>12} {source_text}")
            print_msg(f"{'TARGET:':>12} {target_text}")
            print_msg(f"{'PREDICTED:':>12} {model_out_text}")

            if count == num_examples:
                print_msg("-" * console_width)
                break

    # TensorBoard Logging
    if writer:
        metric_cer = torchmetrics.CharErrorRate()
        cer = metric_cer(predicted, expected)
        writer.add_scalar("validation_cer", cer, global_step)
        writer.flush()

        metric_wer = torchmetrics.WordErrorRate()
        wer = metric_wer(predicted, expected)
        writer.add_scalar("validation_wer", wer, global_step)
        writer.flush()

        metric_bleu = torchmetrics.BLEUScore()
        bleu = metric_bleu(predicted, expected)
        writer.add_scalar("validation_bleu", bleu, global_step)
        writer.flush()


def get_all_sentences(ds, lang):
    """
    Generator that yields all sentences in the specified language
    from the dataset.

    Args:
        ds (iterable): Dataset containing translation pairs.
        lang (str): Language key to extract sentences.

    Yields:
        str: Sentence in the specified language.
    """
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, ds, lang):
    """
    Creates or loads a tokenizer for a specified language.

    Args:
        config (dict): Configuration dictionary containing tokenizer file path.
        ds (iterable): Dataset containing translation pairs.
        lang (str): Language key to extract sentences.

    Returns:
        Tokenizer: A tokenizer object for the specified language.
    """
    tokenizer_path = Path(config["tokenizer_file"].format(lang))

    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2,
        )

        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_ds(config):
    """
    Load and prepare the training and validation datasets.

    Args:
        config (dict): Configuration dictionary containing language settings, sequence length, batch size, etc.

    Returns:
        tuple: Tuple containing train DataLoader, validation DataLoader, source tokenizer, and target tokenizer.
    """
    src_lang = config["lang_src"]
    tgt_lang = config["lang_tgt"]
    seq_len = config["seq_len"]

    # Load dataset
    ds_raw = load_dataset("opus_books", f"{src_lang}-{tgt_lang}", split="train")

    # Build Tokenizers
    tokenizer_src = get_or_build_tokenizer(config=config, ds=ds_raw, lang=src_lang)
    tokenizer_tgt = get_or_build_tokenizer(config=config, ds=ds_raw, lang=tgt_lang)

    # Divide dataset: 90% training, 10% validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # Create Bilingual Datasets
    train_ds = BilingualDataset(
        ds=train_ds_raw,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        seq_len=seq_len,
    )

    val_ds = BilingualDataset(
        ds=val_ds_raw,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        seq_len=seq_len,
    )

    # Calculate max sentence lengths for source and target
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][src_lang]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][tgt_lang]).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(
        val_ds, batch_size=1, shuffle=True
    )

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    """
    Build and return a Transformer model.

    Args:
        config (dict): Configuration dictionary with model parameters.
        vocab_src_len (int): Size of the source vocabulary.
        vocab_tgt_len (int): Size of the target vocabulary.

    Returns:
        nn.Module: Transformer model instance.
    """
    seq_len = config["seq_len"]

    model = build_transformer(
        src_vocab_size=vocab_src_len,
        tgt_vocab_size=vocab_tgt_len,
        src_seq_len=seq_len,
        tgt_seq_len=seq_len,
        d_model=config["d_model"],
    )

    return model



def train_model(config):
    """
    Train the model using the specified configuration.

    Args:
        config (dict): Configuration dictionary containing training parameters.
    """
    # Define the device
    device = (
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.has_mps or torch.backends.mps.is_available() 
        else "cpu"
    )
    print("Using device:", device)
    
    if device == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3:.2f} GB")
    elif device == "mps":
        print("Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    
    device = torch.device(device)

    # Ensure weights folder exists
    weights_dir = Path(f"{config['datasource']}_{config['model_folder']}")
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset and model
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard writer
    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    # Load pre-trained model if specified
    initial_epoch = 0
    global_step = 0
    preload = config.get("preload")
    model_filename = (
        latest_weights_file_path(config) if preload == "latest" 
        else get_weights_file_path(config, preload) if preload 
        else None
    )

    if model_filename:
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    else:
        print("No model to preload, starting from scratch")

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), 
        label_smoothing=0.1
    ).to(device)

    # Training loop
    for epoch in range(initial_epoch, config["num_epochs"]):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)

            # Forward pass
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            # Compute loss
            label = batch["label"].to(device)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():.3f}"})

            # Log the loss
            writer.add_scalar("train_loss", loss.item(), global_step)
            writer.flush()

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Validation
        run_validation(
            model, val_dataloader, tokenizer_src, tokenizer_tgt, 
            config["seq_len"], device, 
            lambda msg: batch_iterator.write(msg), 
            global_step, writer
        )

        # Save model checkpoint
        checkpoint_path = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            checkpoint_path,
        )


if __name__ == "__main__":
    """
    Entry point for the script. Configures settings and starts training the model.
    """
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config=config)