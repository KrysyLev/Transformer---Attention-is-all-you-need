import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)  # d_model = 512 in paper

    def forward(self, x):
        return (
            self.embedding(x) * math.sqrt(self.d_model)
        )  # Scaling This follows the Transformer model from the original "Attention Is All You Need" paper (Vaswani et al., 2017).


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a positional matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a single vector of shape (seq_len, 1)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        denominator = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Instead of computing 10000^(2i/d_model) directly (which could cause numerical instability),
        # We efficiently compute the inverse using exponentiation.

        # Apply sin to even pos
        pe[:, 0::2] = torch.sin(pos * denominator)
        pe[:, 1::2] = torch.cos(pos * denominator)

        # add the batch dim to first dim

        pe = pe.unsqueeze(0)  # (1, seq_len. d_model)

        self.register_buffer("pe", pe)  # Save the tensor

    def forward(self, x):
        x = (
            x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        )  # This ensures that the positional encoding does not update during backpropagation.
        return self.dropout(x)  # prevent overfitting


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # mutiplied
        self.bias = nn.Parameter(torch.zeros(1))  # Added

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1 & b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # W2 & b2

    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiheadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.dk = d_model // h
        self.w_q = nn.Linear(d_model, d_model)  # Wq
        self.w_k = nn.Linear(d_model, d_model)  # Wk
        self.w_v = nn.Linear(d_model, d_model)  # Wv

        self.w_o = nn.Linear(d_model, d_model)  # Wo
        self.dropout = nn.Dropout(dropout)
        
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        
        #(Batch, h, Seq_len, d_k) --> (Batch, h, Seq_len, Seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) #(Batch, h, seq_len, seq_len) due to matmul

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return(attention_scores @ value) , attention_scores
    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # (Batch, seq_len, d_model) ==> (Batch, seq_len, d_model)
        key = self.w_k(k)  # (Batch, seq_len, d_model) ==> (Batch, seq_len, d_model)
        value = self.w_v(v)  # (Batch, seq_len, d_model) ==> (Batch, seq_len, d_model)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)  
        # Split the query into small pieces to feed to each head 
        # (Batch, seq_len, d_model) --> (Batch, seq_len, h, d_k) --> (Batch, h, seq_len, d_k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        x, self.attention_scores = MultiheadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        #(Batch, h, seq_len, d_k) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, d_model)
        x = x.tranpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k) #contiguous force the pytorch to take the transpose into memory
        
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.w_o(x)