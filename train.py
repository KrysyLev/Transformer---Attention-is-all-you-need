import torch
import torch.nn as nn
from torch.utils.data import Dataset, Dataloader, random_split

from dataset import BilingualDataset, casual_mask
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

from model import build_transformer


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

    train_dataloader = Dataloader(
        train_ds, batch_size=config["batch_size"], shuffle=True
    )
    val_dataloader = Dataloader(
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
