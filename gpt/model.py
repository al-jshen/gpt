import numpy as np
import torch
import lightning.pytorch as pl
import torch.nn as nn
import torch.nn.functional as F
from functorch.einops import rearrange
from deepspeed.ops.adam import FusedAdam
from einops.layers.torch import Rearrange


class Attention(nn.Module):
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
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )  # L = t, S = t, E = d, Ev = d, so output is (b h t d)

        # Project back to embedding dimension
        y = rearrange(y, "b h t d -> b t (h d)")

        # output projection
        y = self.proj(y)
        y = self.proj_dropout(y)

        return y


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


# make a lightning module mlp which is just an mlp but it takes in a 2d input and flattens it
# then it has a linear layer at the end to output the right number of classes
class L_MLP(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        return FusedAdam(self.parameters(), lr=1e-3)

    def _step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )
        return loss

    def testing_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log(
            "test_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )
        return loss


class TransformerBlock(nn.Module):
    def __init__(self, num_heads=12, embed_dim=768, mlp_dim=768 * 4, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = Attention(num_heads=num_heads, embed_dim=embed_dim, dropout=dropout)
        self.mlp = MLP(embed_dim, mlp_dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ViT(pl.LightningModule):
    def __init__(
        self,
        num_heads=12,
        embed_dim=768,
        mlp_dim=768 * 4,
        dropout=0.1,
        num_blocks=6,
        image_channels=3,
        image_size=(256, 256),
        patch_size=(16, 16),
        output_dim=10,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.spatial_dims = len(image_size)
        assert len(patch_size) == self.spatial_dims
        for d in range(self.spatial_dims):
            assert (
                image_size[d] % patch_size[d] == 0
            ), f"image size must be divisible by patch size in all spatial dims, mismatch in dim {d+1}"
        self.n_patches: tuple[int, ...] = tuple(
            [image_size[d] // patch_size[d] for d in range(self.spatial_dims)]
        )

        # embedding
        self.patch_embed = nn.Conv2d(
            in_channels=image_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.positional_encoding = nn.Parameter(
            torch.randn(
                1, np.prod(self.n_patches), embed_dim
            )  # extra dim at front for batch, extra dim in middle for class token
        )

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(num_heads, embed_dim, mlp_dim, dropout)
                for _ in range(num_blocks)
            ]
        )  # transformer blocks

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        # output head here ...
        self.patch_debed = nn.ConvTranspose2d(
            in_channels=embed_dim,
            out_channels=image_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )
        # self.output_head = nn.Linear(embed_dim, output_dim)

    def forward(self, x):

        # B, C, H, W = x.shape

        # embed, and add positional encodings
        x = self.patch_embed(x)  # (batch, embed_dim, H // patch_size, W // patch_size)
        x = rearrange(
            x, "b c nh nw -> b (nh nw) c", nh=self.n_patches[0], nw=self.n_patches[1]
        )  # (batch, n_patch_h * n_patch_w, embed_dim)
        # normalize
        x = self.ln1(x)

        # add positional encoding and do dropout
        x = self.dropout(x + self.positional_encoding)  # (batch, n_patches, embed_dim)

        # pass through transformer blocks
        x = self.blocks(x)  # (batch, sequence_length, embed_dim)
        # normalize again
        x = self.ln2(x)  # (batch, sequence_length, embed_dim)

        # output head, reverse the patch embedding
        x = rearrange(
            x, "b (nh nw) c -> b c nh nw", nh=self.n_patches[0], nw=self.n_patches[1]
        )
        x = self.patch_debed(x)  # (batch, image_channels, H, W)

        # x = x[:, 0, :]  # pull out the class token  (batch, embed_dim)
        # # send to output dimensions
        # x = self.output_head(x)  # (batch, output_dim)

        return x

    def configure_optimizers(self):
        return FusedAdam(self.parameters(), lr=1e-2)

    def _step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )
        return loss

    def testing_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log(
            "test_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )
        return loss

    @property
    def nparams(self):
        return sum([p.numel() for p in self.parameters()])
