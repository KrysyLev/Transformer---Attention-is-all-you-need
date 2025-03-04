import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualDataset, casual_mask

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from config import get_weight_file_path, get_config

from model import build_transformer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import warnings
from torchmetrics.text import BLEUScore, WordErrorRate, CharErrorRate


def greedy_decode(
    model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device
):
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
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

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

            # Print the source, target and model output
            print_msg("-" * console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg("-" * console_width)
                break
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate
        metric = CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar("validation cer", cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar("validation wer", wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar("validation BLEU", bleu, global_step)
        writer.flush()


def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, ds, lang):
    # config['tokenzier_file'].format(lang) = '../tokenizers/tokenizer_(0).json'
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    src_lang = config["lang_src"]
    tgt_lang = config["lang_tgt"]
    seq_len = config["seq_len"]

    ds_raw = load_dataset("opus_books", f"{src_lang}-{tgt_lang}", split="train")

    # Build Tokenizers
    tokenizer_src = get_or_build_tokenizer(config=config, ds=ds_raw, lang=src_lang)
    tokenizer_tgt = get_or_build_tokenizer(config=config, ds=ds_raw, lang=tgt_lang)

    # Divide 90% for training and 10% for validation

    train_ds_size = int(0.9 * len(ds_raw))
    validation_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, validation_ds_size])

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

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][src_lang]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][tgt_lang]).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(
        val_ds, batch_size=1, shuffle=True
    )  # Set batch_size = 1 due to we want to go 1-1 validation

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
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
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(
        config=config
    )

    model = get_model(
        config=config,
        vocab_src_len=tokenizer_src.get_vocab_size(),
        vocab_tgt_len=tokenizer_tgt.get_vocab_size(),
    ).to(device=device)
    # Define the tensorboard
    writer = SummaryWriter(
        config["experiment_name"]
    )  # Visualizing the training process

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if config["preload"]:
        model_filename = get_weight_file_path(config, config["preload"])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device=device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        batch_iter = tqdm(
            train_dataloader, desc=f"Processing epoch: {epoch:02d}"
        )  # Progress bar
        for batch in batch_iter:
            model.train()

            encoder_input = batch["encoder_input"].to(device)  # (Batch, seq_len)
            decoder_input = batch["decoder_input"].to(device)  # (Batch, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (Batch, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(
                device
            )  # (Batch, 1, seq_len, seq_len)

            # Run through the transformer

            encoder_output = model.encode(
                encoder_input, encoder_mask
            )  # (Batch, seq_len, d_model)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )  # (Batch, seq_len, d_model)
            proj_output = model.project(
                decoder_output
            )  # (Batch, seq_len, tgt_vocab_size)

            label = batch["label"].to(device)  # (Batch, seq_len)

            # (Batch, seq_len, tgt_vocab_size) --> #(Batch * seq_len, tgt_vocab_size)
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)
            )
            batch_iter.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar("train loss:", loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        run_validation(
            model,
            val_dataloader,
            tokenizer_src,
            tokenizer_tgt,
            max_len=config["seq_len"],
            device=device,
            print_msg=lambda msg: batch_iter.write(msg),
            global_step=global_step,
            writer=writer,
        )

        # Save the model at the end of every epoch
        model_filename = get_weight_file_path(config, f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config=config)
