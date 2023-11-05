import numpy as np
import torch
import lightning.pytorch as pl
import torch.nn as nn
import torch.nn.functional as F
from functorch.einops import rearrange
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
        B, T, C = x.shape  # c = embed_dim, T = sequences (patches)
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


class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


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


class InstanceNormXd(nn.Module):
    def __init__(self, dim, eps=1e-7):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # n c h w d
        n_spatial_dims = len(x.shape) - 2
        assert n_spatial_dims >= 1
        std_dims = tuple(range(2, n_spatial_dims + 2))
        std = torch.std(x, dim=std_dims, keepdim=True)
        x = (x) / (std + self.eps)
        x = (
            x * self.weight[None, :, *(None,) * n_spatial_dims]
            + self.bias[None, :, *(None,) * n_spatial_dims]
        )
        return x


class ViT_torch(nn.Module):
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
        output_head=None,
    ):
        super().__init__()
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
        # self.patch_embed = nn.Conv2d(
        #     in_channels=image_channels,
        #     out_channels=embed_dim,
        #     kernel_size=patch_size,
        #     stride=patch_size,
        # )
        self.patch_embed = nn.Sequential(
            Rearrange(
                "b c (nh p1) (nw p2) -> b (nh nw) (p1 p2 c)",
                p1=patch_size[0],
                p2=patch_size[1],
                nh=self.n_patches[0],
                nw=self.n_patches[1],
            ),
            nn.LayerNorm(patch_size[0] * patch_size[1] * image_channels),
            nn.Linear(patch_size[0] * patch_size[1] * image_channels, embed_dim),
            nn.LayerNorm(embed_dim),
        )  # (batch, im_channels, h, w) -> (batch, n_patches, embed_dim)

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

        self.in1 = InstanceNormXd(image_channels)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        # # output head here ...
        # self.patch_debed = nn.ConvTranspose2d(
        #     in_channels=embed_dim,
        #     out_channels=image_channels,
        #     kernel_size=patch_size,
        #     stride=patch_size,
        # )

        if output_head is None:
            self.output_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, patch_size[0] * patch_size[1] * image_channels),
                nn.LayerNorm(patch_size[0] * patch_size[1] * image_channels),
                Rearrange(
                    "b (nh nw) (p1 p2 c) -> b c (nh p1) (nw p2)",
                    p1=patch_size[0],
                    p2=patch_size[1],
                    nh=self.n_patches[0],
                    nw=self.n_patches[1],
                ),
            )  # (batch, n_patches, embed_dim) -> (batch, im_channels, h, w)
            # self.output_head = nn.Linear(embed_dim, output_dim)
        else:
            self.output_head = output_head

    def forward(self, x):
        # B, C, H, W = x.shape

        # normalize
        x = self.in1(x)

        # embed, and add positional encodings
        x = self.patch_embed(x)  # (batch, embed_dim, H // patch_size, W // patch_size)

        # normalize
        x = self.ln1(x)

        # add positional encoding and do dropout
        x = self.dropout(x + self.positional_encoding)  # (batch, n_patches, embed_dim)

        # pass through transformer blocks
        x = self.blocks(x)  # (batch, sequence_length, embed_dim)

        # normalize again
        x = self.ln2(x)  # (batch, sequence_length, embed_dim)

        # output head, reverse the patch embedding
        # x = self.patch_debed(x)  # (batch, image_channels, H, W)

        x = self.output_head(x)

        return x

    @property
    def nparams(self):
        return sum([p.numel() for p in self.parameters()])


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
        output_head=None,
        lr=1e-3,
        loss_fn=F.mse_loss,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = ViT_torch(
            num_heads=num_heads,
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            num_blocks=num_blocks,
            image_channels=image_channels,
            image_size=image_size,
            patch_size=patch_size,
            output_head=output_head,
        )

        self.loss_fn = loss_fn
        self.lr = lr

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
