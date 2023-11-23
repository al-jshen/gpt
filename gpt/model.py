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
        class_token=False,
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

        # this seems to have less patch artifacts vs conv
        # patch_dim = patch_size[0] * patch_size[1] * image_channels
        # self.patch_embed = nn.Sequential(
        #     Rearrange(
        #         "b c (nh p1) (nw p2) -> b (nh nw) (p1 p2 c)",
        #         p1=patch_size[0],
        #         p2=patch_size[1],
        #         nh=self.n_patches[0],
        #         nw=self.n_patches[1],
        #     ),
        #     nn.LayerNorm(patch_dim),
        #     nn.Linear(patch_dim, embed_dim),
        #     nn.LayerNorm(embed_dim),
        # )

        self.positional_encoding = nn.Parameter(
            torch.randn(
                1, np.prod(self.n_patches) + (1 if class_token else 0), embed_dim
            )  # extra dim at front for batch, extra dim in middle for class token
        )

        if class_token:
            self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        else:
            self.class_token = None

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
            self.output_head = nn.Sequential(
                Rearrange(
                    "b (nh nw) c -> b c nh nw",
                    nh=self.n_patches[0],
                    nw=self.n_patches[1],
                ),
                nn.ConvTranspose2d(
                    in_channels=embed_dim,
                    out_channels=image_channels,
                    kernel_size=patch_size,
                    stride=patch_size,
                ),
            )

            # self.output_head = nn.Sequential(
            #     nn.LayerNorm(embed_dim),
            #     nn.Linear(embed_dim, patch_size[0] * patch_size[1] * image_channels),
            #     nn.LayerNorm(patch_size[0] * patch_size[1] * image_channels),
            #     Rearrange(
            #         "b (nh nw) (p1 p2 c) -> b c (nh p1) (nw p2)",
            #         p1=patch_size[0],
            #         p2=patch_size[1],
            #         nh=self.n_patches[0],
            #         nw=self.n_patches[1],
            #     ),
            # )
        else:
            self.output_head = output_head

    def forward(self, x):
        B, C, H, W = x.shape

        # embed
        x = self.patch_embed(x)

        # add class token if necessary
        if self.class_token is not None:
            class_token = repeat(self.class_token, "1 1 d -> b 1 d", b=B)
            x = torch.cat((x, class_token), dim=1)

        # add positional encoding and do dropout
        x = self.dropout(x + self.positional_encoding)  # (batch, n_patches, embed_dim)

        # pass through transformer blocks
        x = self.blocks(x)  # (batch, sequence_length, embed_dim)

        # normalize
        x = self.norm(x)  # (batch, sequence_length, embed_dim)

        # # extract class token if necessary
        # if self.class_token is not None:
        #     x = x[:, -1, :]
        # else:
        #     x = x.mean(dim=1)

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
