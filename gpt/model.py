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
    def __init__(self, in_dims, num_classes):
        super().__init__()
        self.proj = nn.Linear(in_dims, num_classes)

    def forward(self, x):
        x = x.mean(dim=1)
        return self.proj(x)


class DebedHead(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, patch_size):
        super().__init__()
        assert len(image_size) == len(patch_size)
        spatial_dims = len(image_size)
        assert spatial_dims <= 3, "spatial dims must be 1, 2, or 3"
        n_patches = [image_size[d] // patch_size[d] for d in range(spatial_dims)]

        self.debed = nn.Sequential(
            Rearrange(
                "b (nh nw) c -> b c nh nw",
                nh=n_patches[0],
                nw=n_patches[1],
            ),
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=patch_size,
                stride=patch_size,
            ),
        )

    def forward(self, x):
        return self.debed(x)


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
        std = torch.std(x, dim=std_dims, unmaskdim=True)
        x = (x) / (std + self.eps)
        x = (
            x * self.weight[None, :, *(None,) * n_spatial_dims]
            + self.bias[None, :, *(None,) * n_spatial_dims]
        )
        return x


class ViT(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        mlp_dim=768 * 4,
        num_heads=12,
        dropout=0.1,
        num_blocks=6,
        parallel_paths=2,
        image_channels=3,
        image_size=(256, 256),
        patch_size=(16, 16),
        output_head=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.parallel_paths = parallel_paths
        self.image_channels = image_channels
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

        patch_dim = np.prod(patch_size) * image_channels

        self.to_patch = Rearrange(
            "b c (nh p1) (nw p2) -> b (nh nw) (p1 p2 c)",
            p1=patch_size[0],
            p2=patch_size[1],
            nh=self.n_patches[0],
            nw=self.n_patches[1],
        )

        self.patch_embed = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.positional_encoding = nn.Parameter(
            torch.randn(
                1, np.prod(self.n_patches), embed_dim
            )  # extra dim at front for batch
        )

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(num_heads, embed_dim, mlp_dim, dropout, parallel_paths)
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

        # split into patches
        x = self.to_patch(x)  # (b, n_patches, pixels_per_patch)

        # embed patches
        x = self.patch_embed(x)  # (b, n_patches, embed_dim)

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


class LightningMAE(pl.LightningModule):
    def __init__(self, **kwargs):
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
        self.model = MAE(**kwargs)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, batch_idx):
        mask_patches, out_mask_patches, mask_idx = self.model(batch)
        batch_indexer = torch.arange(batch.shape[0]).unsqueeze(-1)
        loss = self.loss_fn(
            out_mask_patches[batch_indexer, mask_idx],
            mask_patches[batch_indexer, mask_idx],
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
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
        loss = self._step(batch, batch_idx)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class MAE(nn.Module):
    def __init__(
        self,
        vit,
        masking_fraction,
        decoder_dim=256,
        decoder_num_heads=8,
        decoder_num_blocks=4,
        decoder_dropout=0.1,
    ):
        super().__init__()
        self.vit = vit
        num_patches = np.prod(vit.n_patches)
        self.masking_fraction = masking_fraction
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.positional_encoding = nn.Parameter(torch.randn(num_patches, decoder_dim))
        # self.positional_encoding = nn.Embedding(num_patches, decoder_dim)
        self.decoder_dim = decoder_dim
        self.decoder_num_heads = decoder_num_heads
        self.decoder_num_blocks = decoder_num_blocks
        self.decoder_dropout = decoder_dropout
        self.encoder_to_decoder = nn.Linear(self.vit.embed_dim, self.decoder_dim)
        self.decoder = nn.Sequential(
            *[
                TransformerBlock(
                    decoder_num_heads, decoder_dim, decoder_dim * 4, decoder_dropout
                )
                for _ in range(decoder_num_blocks)
            ]
        )  # transformer blocks

        n_pixels_per_patch = np.prod(vit.patch_size) * vit.image_channels
        self.debed = nn.Linear(decoder_dim, n_pixels_per_patch)

    def forward(self, x):
        B, C, H, W = x.shape

        patches = self.vit.to_patch(x)  # (b n_patches patch_dim)

        tokens = self.vit.patch_embed(patches)  # (b n_patches embed_dim)

        n_patches = tokens.shape[1]

        # add positional encoding
        tokens = tokens + self.vit.positional_encoding  # (batch, n_patches, embed_dim)

        # make random indices to determine what gets mask/kept
        rand_idx = torch.rand(B, n_patches)
        shuf_idx = torch.argsort(rand_idx, dim=1)
        unshuf_idx = torch.argsort(shuf_idx, dim=1)

        num_mask = int(self.masking_fraction * n_patches)

        mask_idx = shuf_idx[:, :num_mask]  # front part is mask (removed)
        unmask_idx = shuf_idx[
            :, num_mask:
        ]  # back part is kept unmasked and passed through encoder

        batch_indexer = torch.arange(B).unsqueeze(-1)

        # for mask parts, just unmask the image patches (not embeddings)
        mask_patches = patches[
            batch_indexer, mask_idx
        ]  # (batch, n_mask, n_pixels_per_patch)

        # for unmask parts, unmask the tokens
        unmask_tokens = tokens[
            batch_indexer, unmask_idx
        ]  # (batch, n_unmask, embed_dim)

        # encode
        encoded_tokens = self.vit.blocks(unmask_tokens)  # (batch, n_unmask, embed_dim)
        encoded_tokens = self.vit.norm(unmask_tokens)  # (batch, n_unmask, embed_dim)

        # project to decoder dim
        unmask_tokens = self.encoder_to_decoder(
            encoded_tokens
        )  # (batch, n_unmask, decoder_dim)

        # add positional encoding
        unmask_tokens = (
            unmask_tokens + self.positional_encoding[unmask_idx, :]
        )  # self.positional_encoding(unmask_idx) for nn.Embedding

        # repeat the mask token
        mask_tokens = repeat(
            self.mask_token, "d -> b n d", b=B, n=num_mask
        )  # (batch, n_mask, decoder_dim)

        # add positional encoding
        mask_tokens = (
            mask_tokens + self.positional_encoding[mask_idx, :]
        )  # self.positional_encoding(mask_idx) for nn.Embedding

        # full_tokens = torch.zeros(B, n_patches, self.decoder_dim, device=x.device)

        # full_tokens[batch_indexer, mask_idx] = mask_tokens
        # full_tokens[batch_indexer, unmask_idx] = unmask_tokens

        # join the mask tokens back up with the unmask/encoded tokens
        full_tokens = torch.cat(
            (mask_tokens, unmask_tokens), axis=1
        )  # (batch, n_patches, decoder_dim)

        # unshuffle
        full_tokens = full_tokens[
            batch_indexer, unshuf_idx
        ]  # (batch, n_patches, decoder_dim)

        # # add positional embeddings
        # full_tokens = (
        #     full_tokens + self.positional_encoding
        # )  # (batch, n_patches, decoder_dim)

        # decode
        decoded_tokens = self.decoder(full_tokens)  # (batch, n_patches, decoder_dim)

        # # extract the mask tokens
        # mask_tokens = decoded_tokens[
        #     batch_indexer, mask_idx
        # ]  # (batch, n_mask, decoder_dim)

        # project back to image patches
        out = self.debed(decoded_tokens)  # (batch, n_patches, n_pixels_per_patch)

        return patches, out, mask_idx


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
