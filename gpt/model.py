import torch
import lightning.pytorch as pl

import torch.nn as nn
import torch.nn.functional as F

from functorch.einops import rearrange


class CausalSelfAttention(nn.Module):
    def __init__(self, num_heads=12, embed_dim=768, dropout=0.1):
        super().__init__()

        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads"

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.attn = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # c = embed_dim
        H = self.num_heads

        # Self-attention
        qkv = self.attn(x)  # b t c -> b t 3c
        q, k, v = qkv.view(3, C, B, T)  # q, k, v are each (c b t)
        q, k, v = map(
            lambda t: rearrange(t, "(d h) b t -> b h t d", h=H), (q, k, v)
        )  # where channels C = dims D * heads H

        # Scale dot product attention, attend over T dim (2nd last)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )  # L = t, S = t, E = d, Ev = d, so output is (b h t d)

        # Project back to embedding dimension
        y = rearrange(y, "b h t d -> b t (h d)")

        # output projection
        y = self.proj(y)
        y = self.proj_dropout(y)

        return y


class MLP(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, num_heads=12, embed_dim=768, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(
            num_heads=num_heads, embed_dim=embed_dim, dropout=dropout
        )
        self.mlp = MLP(embed_dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(pl.LightningModule):
    def __init__(
        self,
        num_heads=12,
        embed_dim=768,
        dropout=0.1,
        num_blocks=6,
        sequence_length=1024,
        vocab_size=25152,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size

        self.token_embeddings = nn.Embedding(
            vocab_size, embed_dim
        )  # from word to embedding dim
        self.pos_embeddings = nn.Embedding(
            sequence_length, embed_dim
        )  # from position to embedding dim
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[Block(num_heads, embed_dim, dropout) for _ in range(num_blocks)]
        )  # transformer blocks

        self.ln = nn.LayerNorm(embed_dim)
        self.output_head = nn.Linear(
            embed_dim, vocab_size
        )  # from embedding dim, what word to predict next?

    def forward(self, x):
        # x is (batch, sequence_length) with dtype torch.int (indices to vocab words)
        B, T = x.shape

        # get token embeddings
        te = self.token_embeddings(x)  # (batch, sequence_length, embed_dim)
        # get positional embeddings
        pe = self.pos_embeddings(torch.arange(T, dtype=torch.int))[
            None, :, :
        ]  # (1, sequence_length, embed_dim)

        # sum token and positional embeddings
        x = self.dropout(te + pe)  # (batch, sequence_length, embed_dim)

        # pass through transformer blocks
        x = self.blocks(x)  # (batch, sequence_length, embed_dim)
        x = self.ln(x)  # (batch, sequence_length, embed_dim)

        if self.training:
            # output logits
            logits = self.output_head(x)  # (batch, sequence_length, vocab_size)
        else:
            logits = self.output_head(x[:, [-1], :])  # (batch, 1, vocab_size)

        return logits


if __name__ == "__main__":
    model = GPT()
    x = torch.randint(0, 25152, (1, 1024))
    y = model(x)
    assert y.shape == (1, 1024, 25152)
