import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    """
    Implements input embeddings for the Transformer model.

    The embeddings map input token indices to dense vectors of size `d_model`.
    The resulting embeddings are scaled by the square root of `d_model` to stabilize training,
    as suggested in the Transformer model ("Attention Is All You Need", Vaswani et al., 2017).

    Attributes:
        embedding (torch.nn.Embedding): The embedding layer mapping input tokens to `d_model`-dimensional vectors.
        d_model (int): Dimensionality of the embeddings.
        vocab_size (int): Size of the vocabulary.

    Args:
        d_model (int): The dimensionality of the embeddings.
        vocab_size (int): The size of the input vocabulary.
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        Initializes the InputEmbeddings module.

        Args:
            d_model (int): The dimensionality of the embeddings.
            vocab_size (int): The size of the input vocabulary.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the embedding layer to the input tensor and scales the embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len), containing token indices.

        Returns:
            torch.Tensor: Scaled embeddings of shape (batch_size, seq_len, d_model).
        """
        # Apply embedding and scale by sqrt(d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Implements positional encoding as described in the Transformer model ("Attention Is All You Need", Vaswani et al., 2017).

    This module generates fixed positional encodings for input sequences, adding information about
    the relative position of tokens in a sequence to the embeddings.

    Attributes:
        d_model (int): Dimensionality of the model.
        seq_len (int): Maximum sequence length.
        dropout (torch.nn.Dropout): Dropout layer for regularization.
        pe (torch.Tensor): Precomputed positional encodings of shape (1, seq_len, d_model).

    Args:
        d_model (int): The dimensionality of the embeddings.
        seq_len (int): The maximum length of the input sequences.
        dropout (float): The dropout probability applied after adding positional encodings.
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """
        Initializes the PositionalEncoding module.

        Args:
            d_model (int): The dimensionality of the embeddings.
            seq_len (int): The maximum length of the input sequences.
            dropout (float): The dropout probability applied after adding positional encodings.
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a positional matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # Create a single vector of shape (seq_len, 1)
        denominator = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Instead of computing 10000^(2i/d_model) directly (which could cause numerical instability),
        # We efficiently compute the inverse using exponentiation.

        # Apply sin to even pos
        pe[:, 0::2] = torch.sin(pos * denominator)
        pe[:, 1::2] = torch.cos(pos * denominator)

        # Add the batch dim to first dim
        pe = pe.unsqueeze(0)  # (1, seq_len. d_model)

        # Register as a buffer to prevent updates during training
        self.register_buffer("pe", pe)  # Save the tensor

    def forward(self, x):
        """
        Applies positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The tensor with positional encoding applied, same shape as input.
        """
        x = (
            x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        )  # This ensures that the positional encoding does not update during backpropagation.
        return self.dropout(x)  # Prevent overfitting


class LayerNormalization(nn.Module):
    """
    Applies Layer Normalization to the input tensor.

    Layer Normalization normalizes each element of the input tensor across the last dimension
    to have zero mean and unit variance, and then applies learnable scaling and bias parameters.

    Attributes:
        alpha (torch.nn.Parameter): Learnable scaling parameter of shape (features,).
        bias (torch.nn.Parameter): Learnable bias parameter of shape (features,).
        eps (float): Small value to prevent division by zero during normalization. Default is 1e-6.

    Args:
        features (int): The number of features in the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
    """

    def __init__(self, features: int, eps: float = 1e-6) -> None:
        """
        Initializes the LayerNormalization module.

        Args:
            features (int): The number of features in the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
        """
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))  # Scaling parameter
        self.bias = nn.Parameter(torch.zeros(features))  # Bias parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies layer normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (..., features).

        Returns:
            torch.Tensor: The normalized tensor of the same shape as input.
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    """
    Implements a FeedForward Network block as used in transformer models.

    The block consists of two linear transformations with a ReLU activation and dropout in between.
    It projects the input tensor to a higher-dimensional space (`d_ff`), applies activation and dropout,
    and then projects it back to the original dimension (`d_model`).

    Attributes:
        linear_1 (torch.nn.Linear): First linear layer projecting from `d_model` to `d_ff`.
        dropout (torch.nn.Dropout): Dropout layer with specified dropout probability.
        linear_2 (torch.nn.Linear): Second linear layer projecting back to `d_model`.

    Args:
        d_model (int): The dimensionality of the input tensor.
        d_ff (int): The dimensionality of the feedforward layer.
        dropout (float): The dropout probability applied between the linear layers.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        """
        Initializes the FeedForwardBlock.

        Args:
            d_model (int): The dimensionality of the input tensor.
            d_ff (int): The dimensionality of the feedforward layer.
            dropout (float): The dropout probability applied between the linear layers.
        """
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # First linear layer
        self.dropout = nn.Dropout(dropout)  # Dropout layer
        self.linear_2 = nn.Linear(d_ff, d_model)  # Second linear layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the feedforward network to the input tensor.

        The sequence of operations is:
        1. Linear transformation to `d_ff` dimensions.
        2. ReLU activation.
        3. Dropout.
        4. Linear transformation back to `d_model` dimensions.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class MultiheadAttentionBlock(nn.Module):
    """
    Implements a multi-head attention block as described in the Transformer model
    ("Attention Is All You Need", Vaswani et al., 2017).

    This module applies attention across multiple heads, allowing the model to jointly attend
    to information from different representation subspaces.

    Attributes:
        d_model (int): Dimensionality of the model.
        h (int): Number of attention heads.
        d_k (int): Dimensionality of each head (d_model // h).
        w_q (torch.nn.Linear): Linear layer for query projection.
        w_k (torch.nn.Linear): Linear layer for key projection.
        w_v (torch.nn.Linear): Linear layer for value projection.
        w_o (torch.nn.Linear): Linear layer for output projection.
        dropout (torch.nn.Dropout): Dropout layer for regularization.
        attention_scores (torch.Tensor): Attention scores from the last forward pass.

    Args:
        d_model (int): The dimensionality of the input tensor.
        h (int): The number of attention heads.
        dropout (float): The dropout probability applied after attention.
    """

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        """
        Initializes the MultiheadAttentionBlock.

        Args:
            d_model (int): The dimensionality of the model.
            h (int): The number of attention heads.
            dropout (float): The dropout probability.

        Raises:
            AssertionError: If d_model is not divisible by h.
        """
        super().__init__()
        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, "d_model must be divisible by h."

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        dropout: nn.Dropout,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes scaled dot-product attention.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, h, seq_len, d_k).
            key (torch.Tensor): Key tensor of shape (batch_size, h, seq_len, d_k).
            value (torch.Tensor): Value tensor of shape (batch_size, h, seq_len, d_k).
            mask (torch.Tensor): Mask tensor of shape (batch_size, 1, seq_len, seq_len), with 0s masking positions.
            dropout (torch.nn.Dropout): Dropout layer applied to the attention scores.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output tensor of shape (batch_size, h, seq_len, d_k)
            and attention scores of shape (batch_size, h, seq_len, seq_len).
        """
        d_k = query.size(-1)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        attention_scores = torch.softmax(attention_scores, dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        output = torch.matmul(attention_scores, value)
        return output, attention_scores

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies multi-head attention to the input tensors.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, seq_len, d_model).
            k (torch.Tensor): Key tensor of shape (batch_size, seq_len, d_model).
            v (torch.Tensor): Value tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor): Mask tensor of shape (batch_size, 1, seq_len, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Linear projections
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Reshape and split for multiple heads
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        # Apply attention
        x, self.attention_scores = self.attention(query, key, value, mask, self.dropout)

        # Concatenate heads and project
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    """
    Implements a residual connection followed by a dropout and layer normalization.

    This structure is used in the Transformer model to facilitate gradient flow through the network
    and stabilize training by normalizing the output of the sublayer.

    Attributes:
        dropout (torch.nn.Dropout): Dropout layer to prevent overfitting.
        norm (LayerNormalization): Layer normalization applied before the sublayer.

    Args:
        dropout (float): The dropout probability applied after the sublayer operation.
    """

    def __init__(self, features: int, dropout: float) -> None:
        """
        Initializes the ResidualConnection module.

        Args:
            dropout (float): The dropout probability applied after the sublayer operation.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        Applies the residual connection to the input tensor.

        The sequence of operations is:
        1. Apply layer normalization to the input.
        2. Apply the sublayer (e.g., feedforward or attention).
        3. Apply dropout to the output of the sublayer.
        4. Add the input to the output of the sublayer (residual connection).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            sublayer (nn.Module): The sublayer to apply, such as a feedforward network or attention block.

        Returns:
            torch.Tensor: The tensor after applying the residual connection, same shape as input.
        """
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    """
    Implements a single Encoder block consisting of:
    1. Multi-head self-attention block.
    2. Feed-forward network block.
    3. Two residual connections with dropout and layer normalization.

    This follows the structure of the original Transformer model as described in
    "Attention Is All You Need" (Vaswani et al., 2017).

    Attributes:
        self_attention_block (MultiheadAttentionBlock): Multi-head self-attention block.
        feed_forward_block (FeedForwardBlock): Feed-forward block for processing the output of self-attention.
        residual_connections (nn.ModuleList): List of two ResidualConnection layers.

    Args:
        self_attention_block (MultiheadAttentionBlock): The self-attention block instance.
        feed_forward_block (FeedForwardBlock): The feed-forward block instance.
        dropout (float): The dropout probability applied in residual connections.
    """

    def __init__(
        self,
        features: int,
        self_attention_block: MultiheadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        """
        Initializes the EncoderBlock module.

        Args:
            self_attention_block (MultiheadAttentionBlock): The self-attention block instance.
            feed_forward_block (FeedForwardBlock): The feed-forward block instance.
            dropout (float): The dropout probability.
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)]
        )

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the EncoderBlock.

        The sequence of operations:
        1. Apply self-attention block followed by a residual connection.
        2. Apply feed-forward block followed by another residual connection.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            src_mask (torch.Tensor): Source mask tensor of shape (batch_size, 1, seq_len, seq_len).

        Returns:
            torch.Tensor: Output tensor of the same shape as the input.
        """
        # Apply the first residual connection with self-attention
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )

        # Apply the second residual connection with feed-forward block
        x = self.residual_connections[1](x, self.feed_forward_block)

        return x


class Encoder(nn.Module):
    """
    Implements the Encoder module consisting of multiple EncoderBlocks.

    This module applies a sequence of EncoderBlocks followed by a final LayerNormalization.
    The number of layers is determined by the length of the provided layers list.

    Attributes:
        layers (nn.ModuleList): A list of EncoderBlock instances.
        norm (LayerNormalization): Layer normalization applied after the final EncoderBlock.

    Args:
        features (int): The number of features (d_model) used in LayerNormalization.
        layers (nn.ModuleList): A list of EncoderBlock instances to be applied sequentially.
    """

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        """
        Initializes the Encoder module.

        Args:
            features (int): The number of features (d_model) for LayerNormalization.
            layers (nn.ModuleList): A list of EncoderBlock instances.
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Encoder module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor): Mask tensor of shape (batch_size, 1, seq_len, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model) after
            applying all EncoderBlocks and final LayerNormalization.
        """
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


#     The original Transformer paper (Attention is All You Need) does not apply normalization at the end.
# But modern implementations (like fairseq, Hugging Face Transformers) do, because it improves stability.


class DecoderBlock(nn.Module):
    """
    Decoder block for the transformer architecture, which consists of:
    - Self-attention mechanism
    - Cross-attention mechanism (between the encoder and decoder)
    - Feed-forward block

    This block applies residual connections around each sub-layer (self-attention,
    cross-attention, and feed-forward block) with layer normalization.

    Args:
        features (int): The number of features (i.e., the size of the model).
        self_attention_block (MultiheadAttentionBlock): The self-attention block used for self-attention.
        cross_attention_block (MultiheadAttentionBlock): The cross-attention block used for attending to encoder outputs.
        feed_forward_block (FeedForwardBlock): The feed-forward network block.
        dropout (float): The dropout rate applied after each sub-layer.

    """

    def __init__(
        self,
        features: int,
        self_attention_block: MultiheadAttentionBlock,
        cross_attention_block: MultiheadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass through the decoder block.

        Args:
            x (Tensor): The input tensor to the decoder block.
            encoder_output (Tensor): The output from the encoder (used in cross-attention).
            src_mask (Tensor): The source mask (for encoder input).
            tgt_mask (Tensor): The target mask (for decoder input).

        Returns:
            Tensor: The output after applying self-attention, cross-attention, and feed-forward network.
        """
        # Apply self-attention
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        # Apply cross-attention (using encoder output)
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask
            ),
        )
        # Apply feed-forward block
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    """
    Decoder for the transformer architecture. It consists of multiple decoder blocks.

    The decoder processes the input sequence by passing it through multiple decoder blocks
    and applies layer normalization at the end.

    Args:
        features (int): The number of features (i.e., the size of the model).
        layers (nn.ModuleList): The list of decoder blocks to be applied sequentially.

    """

    def __init__(self, features: int, layers: nn.ModuleList):
        """
        Initializes the Decoder with the provided number of features and decoder blocks.

        Args:
            features (int): The number of features (i.e., the size of the model).
            layers (nn.ModuleList): The list of decoder blocks to apply sequentially.
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass through the decoder.

        Args:
            x (Tensor): The input tensor to the decoder.
            encoder_output (Tensor): The output from the encoder (used in cross-attention).
            src_mask (Tensor): The source mask (for encoder input).
            tgt_mask (Tensor): The target mask (for decoder input).

        Returns:
            Tensor: The output after applying all decoder blocks followed by layer normalization.
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    """
    Projection layer that maps the model output to vocabulary space.

    This layer projects the input tensor (which has shape (Batch, seq_len, d_model))
    to the vocabulary size, resulting in a tensor with shape (Batch, seq_len, vocab_size).

    Args:
        d_model (int): The size of the model (i.e., the number of features in the model output).
        vocab_size (int): The size of the vocabulary (i.e., the number of classes for prediction).
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        Initializes the projection layer with a linear transformation to map from d_model to vocab_size.

        Args:
            d_model (int): The size of the model output.
            vocab_size (int): The size of the vocabulary.
        """
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        Forward pass through the projection layer.

        Args:
            x (Tensor): The input tensor with shape (Batch, seq_len, d_model).

        Returns:
            Tensor: The output tensor with shape (Batch, seq_len, vocab_size), representing
                    the predicted probabilities over the vocabulary.
        """
        return self.proj(x)


class Transformer(nn.Module):
    """
    The Transformer model that consists of an encoder and a decoder with positional encoding
    and input embeddings. It also includes a projection layer for mapping the output
    of the decoder to vocabulary space.

    Args:
        encoder (Encoder): The encoder module of the Transformer.
        decoder (Decoder): The decoder module of the Transformer.
        src_embed (InputEmbeddings): Embedding layer for the source language.
        tgt_embed (InputEmbeddings): Embedding layer for the target language.
        src_pos (PositionalEncoding): Positional encoding for the source language.
        tgt_pos (PositionalEncoding): Positional encoding for the target language.
        projection_layer (ProjectionLayer): A projection layer to map the decoder output to vocabulary size.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ) -> None:
        """
        Initializes the Transformer model with the given encoder, decoder, embeddings,
        positional encodings, and projection layer.

        Args:
            encoder (Encoder): The encoder module of the Transformer.
            decoder (Decoder): The decoder module of the Transformer.
            src_embed (InputEmbeddings): Embedding layer for the source language.
            tgt_embed (InputEmbeddings): Embedding layer for the target language.
            src_pos (PositionalEncoding): Positional encoding for the source language.
            tgt_pos (PositionalEncoding): Positional encoding for the target language.
            projection_layer (ProjectionLayer): A projection layer to map the decoder output to vocabulary size.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Encodes the source sequence using the encoder with embeddings and positional encoding.

        Args:
            src (Tensor): The source sequence with shape (batch, seq_len).
            src_mask (Tensor): The mask for the source sequence with shape (batch, seq_len).

        Returns:
            Tensor: The encoded source sequence with shape (batch, seq_len, d_model).
        """
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(
        self,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decodes the target sequence using the decoder, based on encoder output and target mask.

        Args:
            encoder_output (Tensor): The encoded source sequence with shape (batch, seq_len, d_model).
            src_mask (Tensor): The mask for the source sequence with shape (batch, seq_len).
            tgt (Tensor): The target sequence with shape (batch, seq_len).
            tgt_mask (Tensor): The mask for the target sequence with shape (batch, seq_len).

        Returns:
            Tensor: The decoded target sequence with shape (batch, seq_len, d_model).
        """
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects the output of the decoder to the vocabulary space using the projection layer.

        Args:
            x (Tensor): The input tensor with shape (batch, seq_len, d_model).

        Returns:
            Tensor: The projected tensor with shape (batch, seq_len, vocab_size).
        """
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> Transformer:
    """
    Builds a Transformer model.

    Args:
        src_vocab_size (int): Size of the source vocabulary.
        tgt_vocab_size (int): Size of the target vocabulary.
        src_seq_len (int): Length of the source sequences.
        tgt_seq_len (int): Length of the target sequences.
        d_model (int, optional): The number of expected features in the input. Defaults to 512.
        N (int, optional): The number of layers in the encoder and decoder. Defaults to 6.
        h (int, optional): The number of heads in the multiheadattention models. Defaults to 8.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        d_ff (int, optional): The dimension of the feedforward network. Defaults to 2048.

    Returns:
        Transformer: The constructed Transformer model.
    """
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiheadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(
            d_model, encoder_self_attention_block, feed_forward_block, dropout
        )
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiheadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiheadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            d_model,
            decoder_self_attention_block,
            decoder_cross_attention_block,
            feed_forward_block,
            dropout,
        )
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer
    )

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
