import numpy as np
import torch
import lightning.pytorch as pl
import torch.nn as nn
import torch.nn.functional as F
from functorch.einops import rearrange
from einops.layers.torch import Rearrange
from einops import repeat
from functools import cached_property
from deepspeed.ops.adam import DeepSpeedCPUAdam
from lightning.pytorch.strategies import DeepSpeedStrategy
from gpt.utils import patchify
from rotary_embedding_torch import RotaryEmbedding


class Attention(nn.Module):
    def __init__(self, num_heads=12, embed_dim=768, dropout=0.1, reattention=False):
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

        self.qnorm = nn.LayerNorm(embed_dim // num_heads)
        self.knorm = nn.LayerNorm(embed_dim // num_heads)

        self.rope = RotaryEmbedding(dim=embed_dim // num_heads // 2, use_xpos=True)

        self.reattention = reattention
        if reattention:
            self.reattn_weight = nn.Parameter(torch.randn(num_heads, num_heads))
            self.reattn_norm = nn.Sequential(
                Rearrange("b h i j -> b i j h"),
                nn.LayerNorm(num_heads),
                Rearrange("b i j h -> b h i j"),
            )
            self.v_ones = None

    def scaled_dot_product_reattention(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
    ):
        B, H, N, C = query.shape
        if self.v_ones is None:
            self.v_ones = torch.ones((N, N)).expand(H, N, N).to(query.device)

        attn_weight = F.scaled_dot_product_attention(
            query,
            key,
            self.v_ones.expand(B, H, N, N),
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

        attn_weight = torch.einsum(
            "b h i j, h g -> b g i j", attn_weight, self.reattn_weight
        )
        attn_weight = self.reattn_norm(attn_weight)

        return attn_weight @ value

    def forward(self, x):
        # B, N, C = x.shape  # c = embed_dim, N = sequences (patches)
        H = self.num_heads

        # Self-attention
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # b n c -> b n 3c -> 3 b n c
        q, k, v = map(
            lambda x: rearrange(x, "b n (he d) -> b he n d", he=H), qkv
        )  # where channels C = dims D * heads H

        q = self.qnorm(q)
        k = self.knorm(k)

        q, v = self.rope.rotate_queries_and_keys(q, v)

        # Scale dot product attention, attend over T dim (2nd last)
        if self.reattention:
            y = self.scaled_dot_product_reattention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0
            )

        else:
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0
            )  # L = t, S = t, E = d, Ev = d, so output is (b h t d)

        # Project back to embedding dimension
        y = rearrange(y, "b h t d -> b t (h d)")

        # output projection
        y = self.proj(y)
        y = self.proj_dropout(y)

        return y


_kernel_sizes = {
    1: (1, 1, 1),
    2: (2, 1, 1),
    4: (2, 2, 1),
    8: (2, 2, 2),
    16: (4, 2, 2),
    32: (4, 4, 2),
    64: (4, 4, 4),
}


def build_kernel_size(patch_size):
    patch_size = tuple(patch_size)
    # back pad with 1 until length 3
    patch_size = patch_size + (1,) * (3 - len(patch_size))
    return list(zip(*[_kernel_sizes[s] for s in patch_size]))


class hMLP_stem(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        patch_size=(16, 16, 16),
        in_chans=3,
        embed_dim=768,
        spatial_dims=3,
    ):
        super().__init__()
        assert (
            len(patch_size) == spatial_dims
        ), "Must have one patch size for each spatial dimension"
        self.kernel_sizes = build_kernel_size(patch_size)
        self.patch_size = patch_size
        self.spatial_dims = spatial_dims
        self.in_proj = torch.nn.Sequential(
            *[
                nn.Conv3d(
                    in_chans,
                    embed_dim // 4,
                    kernel_size=self.kernel_sizes[0],
                    stride=self.kernel_sizes[0],
                    bias=False,
                    padding=0,
                ),
                nn.InstanceNorm3d(embed_dim // 4, affine=True),
                nn.GELU(),
                nn.Conv3d(
                    embed_dim // 4,
                    embed_dim // 4,
                    kernel_size=self.kernel_sizes[1],
                    stride=self.kernel_sizes[1],
                    bias=False,
                    padding=0,
                ),
                nn.InstanceNorm3d(embed_dim // 4, affine=True),
                nn.GELU(),
                nn.Conv3d(
                    embed_dim // 4,
                    embed_dim,
                    kernel_size=self.kernel_sizes[2],
                    stride=self.kernel_sizes[2],
                    bias=False,
                    padding=0,
                ),
                nn.InstanceNorm3d(embed_dim, affine=True),
            ]
        )

    def forward(self, x):
        # b c h w
        if self.spatial_dims == 2:
            return self.in_proj(x.unsqueeze(-1)).squeeze(-1)
        # b c h w d
        return self.in_proj(x)


class hMLP_output(nn.Module):
    """Patch to Image Debedding"""

    def __init__(
        self,
        embed_dim=768,
        out_chans=3,
        patch_size=(16, 16, 16),
        spatial_dims=3,
    ):
        super().__init__()
        assert (
            len(patch_size) == spatial_dims
        ), "Must have one patch size for each spatial dimension"
        self.kernel_sizes = build_kernel_size(patch_size)
        self.patch_size = patch_size
        self.spatial_dims = spatial_dims

        self.out_proj = torch.nn.Sequential(
            *[
                nn.ConvTranspose3d(
                    embed_dim,
                    embed_dim // 4,
                    kernel_size=self.kernel_sizes[0],
                    stride=self.kernel_sizes[0],
                    bias=False,
                ),
                nn.InstanceNorm3d(embed_dim // 4, affine=True),
                nn.GELU(),
                nn.ConvTranspose3d(
                    embed_dim // 4,
                    embed_dim // 4,
                    kernel_size=self.kernel_sizes[1],
                    stride=self.kernel_sizes[1],
                    bias=False,
                ),
                nn.InstanceNorm3d(embed_dim // 4, affine=True),
                nn.GELU(),
                nn.ConvTranspose3d(
                    embed_dim // 4,
                    out_chans,
                    kernel_size=self.kernel_sizes[2],
                    stride=self.kernel_sizes[2],
                ),
            ]
        )

    def forward(self, x):
        if self.spatial_dims == 2:
            return self.out_proj(x.unsqueeze(-1)).squeeze(-1)
        return self.out_proj(x)


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
        self.spatial_dims = len(image_size)
        assert self.spatial_dims <= 3, "spatial dims must be 1, 2, or 3"
        self.num_patches = tuple(
            [image_size[d] // patch_size[d] for d in range(self.spatial_dims)]
        )

        # self.debed = nn.ConvTranspose2d(
        #     in_channels=in_channels,
        #     out_channels=out_channels,
        #     kernel_size=patch_size,
        #     stride=patch_size,
        # )

        self.debed = hMLP_output(
            embed_dim=in_channels,
            out_chans=out_channels,
            patch_size=patch_size,
            spatial_dims=self.spatial_dims,
        )

    def forward(self, x):
        x = rearrange(
            x,
            "b (nh nw) c -> b c nh nw",
            nh=self.num_patches[0],
            nw=self.num_patches[1],
        )  # (b, embed_dim, nh, nw)
        x = self.debed(x)
        return x


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
        mlp_ratio=4,
        dropout=0.1,
        parallel_paths=2,
        layer_scale=1e-2,
        reattention=True,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ls1 = nn.Parameter(torch.ones(embed_dim) * layer_scale)
        self.ls2 = nn.Parameter(torch.ones(embed_dim) * layer_scale)
        self.attns = Parallel(
            [
                Attention(
                    num_heads=num_heads,
                    embed_dim=embed_dim,
                    dropout=dropout,
                    reattention=reattention,
                )
                for _ in range(parallel_paths)
            ]
        )
        self.mlps = Parallel(
            [
                MLP(embed_dim, embed_dim * mlp_ratio, dropout=dropout)
                for _ in range(parallel_paths)
            ]
        )

    def forward(self, x):
        # x has shape (batch, sequence_length, embed_dim)
        x = x + self.ls1[None, None, :] * self.attns(self.ln1(x))
        x = x + self.ls2[None, None, :] * self.mlps(self.ln2(x))
        return x


class PEG(nn.Module):
    """
    Positional Encoding Generator from https://arxiv.org/pdf/2102.10882.pdf
    """

    def __init__(self, in_chans, embed_dim=768, kernel_size=3, stride=1):
        super().__init__()
        assert kernel_size >= 3, "kernel size must be at least 3"
        padding = (kernel_size - 1) // 2
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
            groups=embed_dim,
        )
        self.stride = stride

    def forward(self, x, H, W, D):
        cnn_feat = rearrange(x, "b (h w d) c -> b c h w d", h=H, w=W, d=D)
        if self.stride == 1:
            out = self.proj(cnn_feat) + cnn_feat
        else:
            out = self.proj(cnn_feat)
        out = rearrange(out, "b c h w d -> b (h w d) c")
        return out


class ViT(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        mlp_ratio=4,
        num_heads=12,
        dropout=0.1,
        num_blocks=6,
        parallel_paths=2,
        image_channels=3,
        image_size=(256, 256),
        patch_size=(16, 16),
        reattention=True,
        output_head=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
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

        # patch_dim = np.prod(patch_size) * image_channels

        # self.to_patch = Rearrange(
        #     "b c (nh p1) (nw p2) -> b (nh nw) (p1 p2 c)",
        #     p1=patch_size[0],
        #     p2=patch_size[1],
        #     nh=self.n_patches[0],
        #     nw=self.n_patches[1],
        # )

        # self.patch_embed = nn.Sequential(
        #     nn.LayerNorm(patch_dim),
        #     nn.Linear(patch_dim, embed_dim),
        #     nn.LayerNorm(embed_dim),
        # )

        self.patch_embed = hMLP_stem(
            patch_size=patch_size,
            in_chans=image_channels,
            embed_dim=embed_dim,
            spatial_dims=self.spatial_dims,
        )

        # self.positional_encoding = nn.Parameter(
        #     torch.randn(
        #         1, np.prod(self.n_patches), embed_dim
        #     )  # extra dim at front for batch
        # )
        self.positional_encoding = PEG(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    num_heads,
                    embed_dim,
                    mlp_ratio,
                    dropout,
                    parallel_paths,
                    layer_scale=0.1 if i < 9 else 1e-5 if i < 12 else 1e-6,
                    reattention=reattention,
                )
                for i in range(num_blocks)
            ]
        )  # transformer blocks

        self.norm = nn.LayerNorm(embed_dim)

        self.output_head = output_head

    def forward(self, x):
        # B, C, H, W = x.shape

        # # split into patches
        # x = self.to_patch(x)  # (b, n_patches, pixels_per_patch)

        # embed patches
        x = self.patch_embed(x)  # (b, embed_dim, nh, nw)

        x = rearrange(x, "b c h w -> b (h w) c")  # (b, n_patches, embed_dim)

        # # add positional encoding and do dropout
        # x = self.dropout(
        #     x + self.positional_encoding
        # )  # (batch, n_patches, embed_dim)

        # # pass through transformer blocks
        # x = self.blocks(x)  # (batch, sequence_length, embed_dim)

        # pass through first transformer block
        x = self.blocks[0](x)

        # add PEG encoding
        x = x + self.positional_encoding(x, *self.n_patches, 1)

        # pass through remaining transformer blocks
        x = self.blocks[1:](x)

        # normalize
        x = self.norm(x)  # (batch, sequence_length, embed_dim)

        if self.output_head is not None:
            x = self.output_head(x)

        return x

    @cached_property
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
        if "logging" not in kwargs:
            kwargs["logging"] = True
        for k, v in kwargs.items():
            setattr(self, k, v)
        # remove lr and loss_fn from kwargs
        for k in ["lr", "loss_fn", "logging"]:
            del kwargs[k]
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

    def _log(self, log_name, loss):
        if self.logging:
            self.log(
                log_name,
                loss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self._log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self._log("val_loss", loss)
        return loss

    def testing_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self._log("test_loss", loss)
        return loss

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def configure_optimizers(self):
        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(self.parameters(), lr=self.lr)
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            fused=True,
        )


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
                    num_heads=decoder_num_heads,
                    embed_dim=decoder_dim,
                    mlp_ratio=4,
                    dropout=decoder_dropout,
                    parallel_paths=2,
                    layer_scale=0.1,
                    reattention=True,
                )
                for _ in range(decoder_num_blocks)
            ]
        )  # transformer blocks

        n_pixels_per_patch = np.prod(vit.patch_size) * vit.image_channels
        self.debed = nn.Linear(decoder_dim, n_pixels_per_patch)

    def forward(self, x):
        B, C, H, W = x.shape

        # patches = self.vit.to_patch(x)  # (b n_patches patch_dim)

        tokens = self.vit.patch_embed(x)  # (b embed_dim, nh, nw)

        tokens = rearrange(tokens, "b c h w -> b (h w) c")  # (b n_patches embed_dim)

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

        # mask_patches = patches[
        #     batch_indexer, mask_idx
        # ]  # (batch, n_mask, n_pixels_per_patch)

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

        # join the mask tokens back up with the unmask/encoded tokens
        full_tokens = torch.cat(
            (mask_tokens, unmask_tokens), axis=1
        )  # (batch, n_patches, decoder_dim)

        # unshuffle
        full_tokens = full_tokens[
            batch_indexer, unshuf_idx
        ]  # (batch, n_patches, decoder_dim)

        # decode
        decoded_tokens = self.decoder(full_tokens)  # (batch, n_patches, decoder_dim)

        # project back to image patches
        out = self.debed(decoded_tokens)  # (batch, n_patches, n_pixels_per_patch)

        return patchify(x, self.vit.patch_size), out, mask_idx


class LightningWrapper(pl.LightningModule):
    """
    Turns a torch.nn.Module into a pl.LightningModule
    """

    def __init__(self, modelclass, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # self._build_steps()

        assert "lr" in kwargs, "must have lr"
        assert "loss_fn" in kwargs, "must have loss_fn"
        if "logging" not in kwargs:
            kwargs["logging"] = True
        for k, v in kwargs.items():
            if not k.startswith("inner_"):
                setattr(self, k, v)

        self.modelclass = modelclass
        if issubclass(modelclass, pl.LightningModule):
            self.wraps_lightning = True
            inner_kwargs = {}

            for k, v in kwargs.items():
                # if there is the same key but with inner_, use that
                if f"inner_{k}" in kwargs:
                    inner_kwargs[k] = kwargs[f"inner_{k}"]
                else:
                    if not k.startswith("inner_"):
                        inner_kwargs[k] = v
                    else:
                        inner_kwargs[k.replace("inner_", "")] = v

            # logging handled by outer wrapper
            inner_kwargs["logging"] = False

        else:
            self.wraps_lightning = False
            # remove lr and loss_fn from kwargs
            inner_kwargs = kwargs.copy()
            del_keys = ["lr", "loss_fn", "logging"]
            for k in del_keys:
                if k in inner_kwargs:
                    del inner_kwargs[k]

        # build model
        if "checkpoint" in kwargs:
            self.model = kwargs["checkpoint"]
        else:
            self.model = modelclass(**inner_kwargs)

        if self.wraps_lightning:
            # if the model is a lightning module, we need to make sure it's not
            # trying to log anything
            self.model.logging = False

    @property
    def nparams(self):
        return sum([p.numel() for p in self.parameters()])

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def _log(self, log_name, loss):
        if self.logging:
            self.log(
                log_name,
                loss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

    def training_step(self, batch, batch_idx):
        if self.wraps_lightning:
            loss = self.model.training_step(batch, batch_idx)
        else:
            loss = self._step(batch, batch_idx)
        self._log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.wraps_lightning:
            loss = self.model.validation_step(batch, batch_idx)
        else:
            loss = self._step(batch, batch_idx)
        self._log("val_loss", loss)
        return loss

    def testing_step(self, batch, batch_idx):
        if self.wraps_lightning:
            loss = self.model.validation_step(batch, batch_idx)
        else:
            loss = self._step(batch, batch_idx)
        self._log("test_loss", loss)
        return loss

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def configure_optimizers(self):
        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(self.parameters(), lr=self.lr)
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            fused=True,
        )
