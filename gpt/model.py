import numpy as np
import torch
import lightning.pytorch as pl
import torch.nn as nn
import torch.nn.functional as F
from functorch.einops import rearrange
from einops.layers.torch import Rearrange
from einops import repeat


class Attention(nn.Module):
    def __init__(self, num_heads=12, embed_dim=768, dropout=0.1):
        super().__init__()

        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads"

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.to_qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # c = embed_dim, T = sequences (patches)
        H = self.num_heads

        # Self-attention
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # b t c -> b t 3c -> 3 b t c
        q, k, v = map(
            lambda t: rearrange(t, "b t (h d) -> b h t d", h=H), qkv
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


class ClassificationHead(nn.Module):
    def __init__(self, in_dims, num_classes, pool="cls"):
        assert pool in {
            "cls",
            "mean",
        }, f"pool must be one of 'cls' or 'mean', got {pool}"
        super().__init__()
        self.pool = pool
        self.proj = nn.Linear(in_dims, num_classes)

    def forward(self, x):
        if self.pool == "cls":
            x = x[:, -1]
        elif self.pool == "mean":
            x = x.mean(dim=1)
        return self.proj(x)


class DebedHead(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, patch_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size
        self.patch_size = patch_size
        assert len(image_size) == len(patch_size)
        self.spatial_dims = len(image_size)
        assert self.spatial_dims <= 3, "spatial dims must be 1, 2, or 3"
        self.n_patches = [
            image_size[d] // patch_size[d] for d in range(self.spatial_dims)
        ]

        self.output_head = nn.Sequential(
            Rearrange(
                "b (nh nw) c -> b c nh nw",
                nh=self.n_patches[0],
                nw=self.n_patches[1],
            ),
            getattr(nn, f"ConvTranspose{self.spatial_dims}d")(
                in_channels=self.embed_dim,
                out_channels=self.image_channels,
                kernel_size=patch_size,
                stride=patch_size,
            ),
        )


class Parallel(nn.Module):
    def __init__(self, fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        return sum([fn(x) for fn in self.fns])


class TransformerBlock(nn.Module):
    def __init__(
        self,
        num_heads=12,
        embed_dim=768,
        mlp_dim=768 * 4,
        dropout=0.1,
        parallel_paths=2,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        # self.ln = nn.LayerNorm(embed_dim)
        self.attns = Parallel(
            [
                Attention(num_heads=num_heads, embed_dim=embed_dim, dropout=dropout)
                for _ in range(parallel_paths)
            ]
        )
        self.mlps = Parallel(
            [MLP(embed_dim, mlp_dim, dropout=dropout) for _ in range(parallel_paths)]
        )

    def forward(self, x):
        x = x + self.attns(self.ln1(x))
        x = x + self.mlps(self.ln2(x))
        return x
        # x = x + self.attns(x)
        # x = x + self.mlps(x)
        # return self.ln(x)


class InstanceNormXd(nn.Module):
    def __init__(self, dim, eps=1e-7):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # n c n1, ..., nd
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


class ViT(nn.Module):
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

        # # embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            Rearrange("b c h w -> b (h w) c"),
        )

        self.positional_encoding = nn.Parameter(
            torch.randn(
                1, np.prod(self.n_patches), embed_dim
            )  # extra dim at front for batch
        )

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(num_heads, embed_dim, mlp_dim, dropout)
                for _ in range(num_blocks)
            ]
        )  # transformer blocks

        self.norm = nn.LayerNorm(embed_dim)

        # # output head here ...
        if output_head is None:
            self.output_head = DebedHead(
                embed_dim, image_channels, image_size, patch_size
            )

        else:
            self.output_head = output_head

    def forward(self, x):
        B, C, H, W = x.shape

        # embed
        x = self.patch_embed(x)

        # add positional encoding and do dropout
        x = self.dropout(x + self.positional_encoding)  # (batch, n_patches, embed_dim)

        # pass through transformer blocks
        x = self.blocks(x)  # (batch, sequence_length, embed_dim)

        # normalize
        x = self.norm(x)  # (batch, sequence_length, embed_dim)

        x = self.output_head(x)

        return x

    @property
    def nparams(self):
        return sum([p.numel() for p in self.parameters()])

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True


class LightningWrapper(pl.LightningModule):
    """
    Turns a torch.nn.Module into a pl.LightningModule
    """

    def __init__(self, modelclass, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        assert "lr" in kwargs, "must have lr"
        assert "loss_fn" in kwargs, "must have loss_fn"
        for k, v in kwargs.items():
            setattr(self, k, v)
        # remove lr and loss_fn from kwargs
        del kwargs["lr"]
        del kwargs["loss_fn"]
        # build model
        self.model = modelclass(**kwargs)

    def forward(self, x):
        return self.model(x)

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

    def testing_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,  # momentum=0.9, nesterov=True
        )
