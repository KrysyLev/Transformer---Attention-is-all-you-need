from typing import Any
import torch
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    """
    A PyTorch Dataset class for bilingual translation data.

    Args:
        ds (list): A list of translation pairs in the form of dictionaries.
        tokenizer_src: Tokenizer for the source language.
        tokenizer_tgt: Tokenizer for the target language.
        src_lang (str): Source language code.
        tgt_lang (str): Target language code.
        seq_len (int): Maximum sequence length.

    Attributes:
        sos_token (torch.Tensor): Start of sentence token for target language.
        eos_token (torch.Tensor): End of sentence token for target language.
        pad_token (torch.Tensor): Padding token for target language.
    """

    def __init__(
        self,
        ds: list,
        tokenizer_src,
        tokenizer_tgt,
        src_lang: str,
        tgt_lang: str,
        seq_len: int,
    ) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self) -> int:
        """Returns the size of the dataset."""
        return len(self.ds)

    def __getitem__(self, index: Any) -> dict:
        """
        Returns the processed data for the given index.

        Args:
            index (Any): Index of the data item.

        Returns:
            dict: A dictionary containing encoder input, decoder input, masks, and labels.
        """
        target_pair = self.ds[index]
        src_text = target_pair["translation"][self.src_lang]
        tgt_text = target_pair["translation"][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_padding = self.seq_len - len(enc_input_tokens) - 2  # [SOS], [EOS]
        dec_padding = self.seq_len - len(dec_input_tokens) - 1  # [SOS]

        if enc_padding < 0 or dec_padding < 0:
            raise ValueError("Sentence is too long.")

        # Encoder input with [SOS] and [EOS]
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_padding, dtype=torch.int64),
        ])

        # Decoder input with only [SOS]
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_padding, dtype=torch.int64),
        ])

        # Label with only [EOS]
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_padding, dtype=torch.int64),
        ])

        assert encoder_input.size(0) == self.seq_len, "Encoder input size mismatch."
        assert decoder_input.size(0) == self.seq_len, "Decoder input size mismatch."
        assert label.size(0) == self.seq_len, "Label size mismatch."

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

def casual_mask(size: int) -> torch.Tensor:
    """
    Generates a causal mask to prevent the decoder from attending to future tokens.

    Args:
        size (int): The size of the mask, typically the sequence length.

    Returns:
        torch.Tensor: A binary mask tensor of shape (1, size, size) where
                      positions that should not be attended to are marked as 0.
    """
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).int()
    return mask == 0


