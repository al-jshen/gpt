from functools import cached_property
from functools import partial

from einops import einsum, pack, rearrange, repeat, unpack
from einops.layers.torch import Rearrange
import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

from gpt.utils import patchify
from jaxtyping import Float
from rotary_embedding_torch import RotaryEmbedding


def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.1, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class AdaLayerNorm(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.t_emb_proj = zero_init(nn.Linear(embedding_dim, input_dim * 2))

    def forward(
        self,
        x: Float[torch.Tensor, "b ... c"],
        t_emb: Float[torch.Tensor, "b d"],
    ) -> Float[torch.Tensor, "b ... c"]:
        c = x.shape[-1]
        num_spatial_dims = len(x.shape) - 2
        assert c == self.input_dim, "input_dim must match the last dimension of x"

        t_emb = self.t_emb_proj(t_emb)[:, *((None,) * num_spatial_dims), :]
        scale, shift = t_emb.chunk(2, dim=-1)

        x = F.layer_norm(x, [c])

        x = x * (1 + scale) + shift

        return x


def rms_norm(x, scale, eps):
    dtype = torch.promote_types(x.dtype, torch.float32)
    mean_sq = torch.mean(x.to(dtype) ** 2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, input_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(input_dim))

    def forward(
        self, x: Float[torch.Tensor, "b ... c"]
    ) -> Float[torch.Tensor, "b ... c"]:
        return rms_norm(x, self.scale, self.eps)


class AdaRMSNorm(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.linear = zero_init(nn.Linear(embedding_dim, input_dim, bias=False))

    def forward(
        self, x: Float[torch.Tensor, "b ... c"], cond: Float[torch.Tensor, "b d"]
    ) -> Float[torch.Tensor, "b ... c"]:
        num_spatial_dims = len(x.shape) - 2
        scale = 1.0 + self.linear(cond)[:, *((None,) * num_spatial_dims), :]
        return rms_norm(x, scale, self.eps)


class Attention(nn.Module):
    def __init__(
        self,
        num_heads=12,
        embed_dim=768,
        dropout=0.1,
        reattention=False,
        is_causal=False,
        use_rope=True,
    ):
        super().__init__()

        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads"

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

        self.qnorm = RMSNorm(embed_dim // num_heads)
        self.knorm = RMSNorm(embed_dim // num_heads)

        if use_rope:
            self.rope = RotaryEmbedding(dim=embed_dim // num_heads // 2, use_xpos=True)
            self.rope_fn = self.rope.rotate_queries_and_keys
        else:
            self.rope_fn = lambda q, k: (q, k)

        self.reattention = reattention
        if reattention:
            self.reattn_weight = nn.Parameter(torch.randn(num_heads, num_heads))
            self.reattn_norm = nn.Sequential(
                Rearrange("b h i j -> b i j h"),
                Lambda(lambda x: x.contiguous()),
                RMSNorm(num_heads),
                Rearrange("b i j h -> b h i j"),
                Lambda(lambda x: x.contiguous()),
            )
            self.v_ones = None

        self.is_causal = is_causal

    def scaled_dot_product_reattention(
        self,
        query,
        key,
        value,
        dropout_p=0.0,
        scale=None,
        attn_mask=None,
    ):
        B, H, N, C = query.shape
        if self.v_ones is None or self.v_ones.shape[1] != N:
            self.v_ones = (
                torch.ones((N, N))
                .expand(H, N, N)
                .to(device=query.device, dtype=query.dtype)
            )

        attn_weight = F.scaled_dot_product_attention(
            query,
            key,
            self.v_ones.expand(B, H, N, N),
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=self.is_causal,
            scale=scale,
        )

        attn_weight = einsum(attn_weight, self.reattn_weight, "b h i j, h g -> b g i j")
        attn_weight = self.reattn_norm(attn_weight)

        return attn_weight @ value

    def forward(self, q, k, v, attn_mask=None):
        # B, N, C = x.shape  # c = embed_dim, N = sequences (patches)
        H = self.num_heads

        # # Self-attention
        # qkv = self.to_qkv(x).chunk(3, dim=-1)  # b n c -> b n 3c -> 3 b n c

        q, k, v = map(
            lambda x: rearrange(x, "b n (he d) -> b he n d", he=H).contiguous(),
            (q, k, v),
        )  # where channels C = dims D * heads H

        q = self.qnorm(q)
        k = self.knorm(k)

        q, k = self.rope_fn(q, k)

        # Scale dot product attention, attend over T dim (2nd last)
        if self.reattention:
            attn_fn = self.scaled_dot_product_reattention
        else:
            attn_fn = partial(F.scaled_dot_product_attention, is_causal=self.is_causal)

        y = attn_fn(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            attn_mask=attn_mask,
        )  # L = t, S = t, E = d, Ev = d, so output is (b h t d)

        # Project back to embedding dimension
        y = rearrange(y, "b h t d -> b t (h d)").contiguous()

        # output projection
        y = self.proj(y)
        y = self.proj_dropout(y)

        return y


class SelfAttention(Attention):
    def forward(self, x, attn_mask=None):
        return super().forward(x, x, x, attn_mask=attn_mask)


class CrossAttention(Attention):
    def forward(self, x, y, attn_mask=None):
        return super().forward(x, y, y, attn_mask=attn_mask)


class MMAttention(SelfAttention):
    def forward(self, x, y, attn_mask=None):
        xy = torch.cat((x, y), dim=1)  # along sequence dim
        return super().forward(xy, attn_mask=attn_mask)


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
    ):
        super().__init__()
        spatial_dims = len(patch_size)
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
    ):
        super().__init__()
        spatial_dims = len(patch_size)
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
    def __init__(self, in_dim, out_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


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
            # spatial_dims=self.spatial_dims,
        )

    def forward(self, x):
        x = rearrange(
            x,
            "b (nh nw) c -> b c nh nw",
            nh=self.num_patches[0],
            nw=self.num_patches[1],
        ).contiguous()  # (b, embed_dim, nh, nw)
        x = self.debed(x)
        return x


class Parallel(nn.Module):
    def __init__(self, fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        return sum([fn(x) for fn in self.fns])


class ConvolutionalSpatialGatingUnit(torch.nn.Module):
    """Convolutional Spatial Gating Unit (CSGU)."""

    def __init__(
        self,
        size: int,
        kernel_size: int,
        dropout_rate: float,
        use_linear_after_conv: bool,
    ):
        super().__init__()

        n_channels = size // 2  # split input channels
        self.norm = nn.LayerNorm(n_channels)
        self.dwconv = torch.nn.Conv1d(
            n_channels,
            n_channels,
            kernel_size,
            1,
            (kernel_size - 1) // 2,
            groups=n_channels,
        )
        if use_linear_after_conv:
            self.linear = torch.nn.Linear(n_channels, n_channels)
        else:
            self.linear = None

        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        """Forward method

        Args:
            x (torch.Tensor): (N, T, D)

        Returns:
            out (torch.Tensor): (N, T, D/2)
        """

        x_r, x_g = x.chunk(2, dim=-1)

        x_g = self.norm(x_g)  # (N, T, D/2)
        x_g = (
            self.dwconv(x_g.transpose(1, 2)).transpose(1, 2).contiguous()
        )  # (N, T, D/2)
        if self.linear is not None:
            x_g = self.linear(x_g)
        out = x_r * x_g  # (N, T, D/2)
        out = self.dropout(out)
        return out


class ConvolutionalGatingMLP(torch.nn.Module):
    """Convolutional Gating MLP (cgMLP)."""

    def __init__(
        self,
        size: int,
        linear_units: int,
        kernel_size: int,
        dropout_rate: float,
        use_linear_after_conv: bool,
    ):
        super().__init__()

        self.channel_proj1 = torch.nn.Sequential(
            torch.nn.Linear(size, linear_units), torch.nn.GELU()
        )
        self.csgu = ConvolutionalSpatialGatingUnit(
            size=linear_units,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            use_linear_after_conv=use_linear_after_conv,
        )
        self.channel_proj2 = torch.nn.Linear(linear_units // 2, size)

    def forward(self, x):
        x = self.channel_proj1(x)  # size -> linear_units
        x = self.csgu(x)  # linear_units -> linear_units/2
        x = self.channel_proj2(x)  # linear_units/2 -> size

        return x


class AxialAttention(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        dropout=0.1,
        reattention=False,
        is_causal=False,
        use_rope=True,
    ):
        super().__init__()

        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.norm1 = nn.InstanceNorm3d(embed_dim)
        self.norm2 = nn.InstanceNorm3d(embed_dim)

        self.to_qkv = nn.Conv3d(embed_dim, embed_dim * 3, 1)
        self.proj = nn.Conv3d(embed_dim, embed_dim, 1)
        self.proj_dropout = nn.Dropout(dropout)

        self.qnorm = RMSNorm(embed_dim // num_heads)
        self.knorm = RMSNorm(embed_dim // num_heads)

        if use_rope:
            self.rope = RotaryEmbedding(dim=embed_dim // num_heads // 2, use_xpos=True)
            self.rope_fn = self.rope.rotate_queries_and_keys
        else:
            self.rope_fn = lambda q, k: (q, k)

        self.reattention = reattention

        if reattention:
            self.reattn_weight = nn.Parameter(torch.randn(num_heads, num_heads))
            self.reattn_norm = nn.Sequential(
                Rearrange("b h i j -> b i j h"),
                Lambda(lambda x: x.contiguous()),
                nn.LayerNorm(num_heads),
                Rearrange("b i j h -> b h i j"),
                Lambda(lambda x: x.contiguous()),
            )
            self.v_ones = None

        self.is_causal = is_causal

    def scaled_dot_product_reattention(
        self,
        query,
        key,
        value,
        dropout_p=0.0,
        scale=None,
    ):
        B, H, N, C = query.shape
        if self.v_ones is None or self.v_ones.shape[1] != N:
            self.v_ones = (
                torch.ones((N, N))
                .expand(H, N, N)
                .to(device=query.device, dtype=query.dtype)
            )

        attn_weight = F.scaled_dot_product_attention(
            query,
            key,
            self.v_ones.expand(B, H, N, N),
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=self.is_causal,
            scale=scale,
        )

        attn_weight = einsum(attn_weight, self.reattn_weight, "b h i j, h g -> b g i j")
        attn_weight = self.reattn_norm(attn_weight)

        return attn_weight @ value

    def forward(self, x):
        b, c, h, w, d = x.shape
        he = self.num_heads

        x = self.norm1(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)  # b c h w d -> b 3c h w d -> 3 b c h w d
        q, k, v = map(
            lambda x: rearrange(
                x, "b (he c) h w d -> b he h w d c", he=he
            ).contiguous(),
            qkv,
        )  # where channels C = dims D * heads H

        q = self.qnorm(q)
        k = self.knorm(k)

        q, k = self.rope_fn(q, k)

        # Scale dot product attention, attend over T dim (2nd last)
        if self.reattention:
            attn_fn = self.scaled_dot_product_reattention
        else:
            attn_fn = partial(F.scaled_dot_product_attention, is_causal=self.is_causal)

        attn_fn = partial(attn_fn, dropout_p=self.dropout if self.training else 0.0)

        qh, kh, vh = map(
            lambda x: rearrange(x, "b he h w d c -> (b w d) he h c").contiguous(),
            (q, k, v),
        )
        yh = attn_fn(qh, kh, vh)
        yh = rearrange(
            yh, "(b w d) he h c -> b (he c) h w d", he=he, h=h, w=w, d=d
        ).contiguous()

        qw, kw, vw = map(
            lambda x: rearrange(x, "b he h w d c -> (b h d) he w c").contiguous(),
            (q, k, v),
        )
        yw = attn_fn(qw, kw, vw)
        yw = rearrange(
            yw, "(b h d) he w c -> b (he c) h w d", he=he, h=h, w=w, d=d
        ).contiguous()

        qd, kd, vd = map(
            lambda x: rearrange(x, "b he h w d c -> (b h w) he d c").contiguous(),
            (q, k, v),
        )
        yd = attn_fn(qd, kd, vd)
        yd = rearrange(
            yd, "(b h w) he d c -> b (he c) h w d", he=he, h=h, w=w, d=d
        ).contiguous()

        y = (yh + yw + yd) / 3

        y = self.norm2(y)

        # output linear projection
        y = self.proj(y)
        y = self.proj_dropout(y)

        return y


class ComplicatedTransformerBlock(nn.Module):
    """Transformer block.

    Additional tweaks:
    - layerscale (https://arxiv.org/pdf/2103.17239.pdf)
    - reattention (https://arxiv.org/pdf/2103.11886v4.pdf)
    - branching and convolutional gating for global+local info (https://arxiv.org/pdf/2207.02971.pdf)
    - depth-wise convolutional merging and macaron attention (https://arxiv.org/pdf/2210.00077.pdf, https://arxiv.org/pdf/1906.02762.pdf)
    - parallel layers (https://arxiv.org/abs/2203.09795)
    - rotary positional encoding (https://arxiv.org/pdf/2104.09864.pdf)
    - drop path (https://arxiv.org/pdf/2106.09681.pdf)
    """

    def __init__(
        self,
        num_heads=12,
        embed_dim=768,
        mlp_ratio=4,
        conv_kernel_size=31,
        dropout=0.1,
        parallel_paths=2,
        layer_scale=1e-5,
        reattention=True,
        axial_3d=False,
        is_causal=False,
    ):
        super().__init__()
        if axial_3d:
            raise NotImplementedError("Axial 3D is not implemented yet.")
        self.axial_3d = axial_3d
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.ls1 = nn.Parameter(torch.ones(embed_dim) * layer_scale, requires_grad=True)
        self.ls2 = nn.Parameter(torch.ones(embed_dim) * layer_scale, requires_grad=True)
        self.ls3 = nn.Parameter(torch.ones(embed_dim) * layer_scale, requires_grad=True)
        self.attns = Parallel(
            [
                Attention(
                    num_heads=num_heads,
                    embed_dim=embed_dim,
                    dropout=dropout,
                    reattention=reattention,
                    is_causal=is_causal,
                )
                if not axial_3d
                else AxialAttention(
                    num_heads=num_heads,
                    embed_dim=embed_dim,
                    dropout=dropout,
                    reattention=reattention,
                    is_causal=is_causal,
                )
                for _ in range(parallel_paths)
            ]
        )
        self.mlps = Parallel(
            [
                MLP(embed_dim, embed_dim, embed_dim * mlp_ratio, dropout=dropout)
                for _ in range(parallel_paths)
            ]
        )
        self.cgmlps = Parallel(
            [
                ConvolutionalGatingMLP(
                    embed_dim,
                    embed_dim * mlp_ratio,
                    kernel_size=conv_kernel_size,
                    dropout_rate=dropout,
                    use_linear_after_conv=False,
                )
                for _ in range(parallel_paths)
            ]
        )
        self.mlp1 = MLP(embed_dim, embed_dim, embed_dim * 2, dropout=dropout)
        self.mlp2 = MLP(embed_dim, embed_dim, embed_dim * 2, dropout=dropout)
        self.dwconv = nn.Conv1d(
            embed_dim * 2,
            embed_dim * 2,
            kernel_size=conv_kernel_size,
            stride=1,
            padding=(conv_kernel_size - 1) // 2,
            groups=embed_dim * 2,
        )
        self.proj = nn.Linear(embed_dim * 2, embed_dim)
        self.drop_path = DropPath(dropout)

    def channel_last(self, fn, x):
        x = rearrange(x, "b c ... -> b ... c")
        x = fn(x)
        x = rearrange(x, "b ... c -> b c ...")
        return x

    def forward(self, x):
        if not self.axial_3d:
            # x has shape (batch, sequence_length, embed_dim)
            x = 0.5 * self.mlp1(x) + x  # (batch, sequence_length, embed_dim)
            x1 = x2 = x  # split branches

            # branch 1
            x1 = x1 + self.drop_path(
                self.ls1[None, None, :]
                * self.attns(self.ln1(x1), self.ln1(x1), self.ln1(x1))
            )
            x1 = x1 + self.drop_path(self.ls2[None, None, :] * self.mlps(self.ln2(x1)))

            # branch 2
            x2 = x2 + self.drop_path(
                self.ls3[None, None, :] * self.cgmlps(self.ln3(x2))
            )

            # join branches, e-branchformer style merge
            x_out = torch.cat(
                (x1, x2), dim=-1
            )  # (batch, sequence_length, embed_dim * 2)
            x_out = (
                self.dwconv(x_out.transpose(1, 2)).transpose(1, 2).contiguous() + x_out
            )  # (batch, sequence_length, embed_dim * 2)

            x_out = self.proj(x_out) + x  # (batch, sequence_length, embed_dim)
            x_out = 0.5 * self.mlp2(x_out) + x_out
            return x_out

        if self.axial_3d:
            b, c, h, w, d = x.shape
            x = rearrange(x, "b c h w d -> b (h w d) c")
            x = 0.5 * self.mlp1(x) + x  # (batch, sequence_length, embed_dim)
            x1 = x2 = x  # split branches
            x1 = (
                x1
                + self.drop_path(
                    self.ls1[None, None, :]
                    * rearrange(
                        self.attns(
                            rearrange(
                                self.ln1(x1), "b (h w d) c -> b c h w d", h=h, w=w, d=d
                            ).contiguous()
                        ),
                        "b c h w d -> b (h w d) c",
                    ).contiguous()
                )
                # + self.drop_path(self.ls2[None, None, :] * self.mlps(self.ln2(x1)))
            )
            x2 = x2 + self.drop_path(
                self.ls3[None, None, :] * self.cgmlps(self.ln3(x2))
            )
            x_out = torch.cat(
                (x1, x2), dim=-1
            )  # (batch, sequence_length, embed_dim * 2)
            # e-branchformer style merge
            x_out = (
                self.dwconv(x_out.transpose(1, 2)).transpose(1, 2).contiguous() + x_out
            )  # (batch, sequence_length, embed_dim * 2)
            x_out = self.proj(x_out) + x  # (batch, sequence_length, embed_dim)
            x_out = 0.5 * self.mlp2(x_out) + x_out
            x_out = rearrange(x_out, "b (h w d) c -> b c h w d", h=h, w=w, d=d)
            return x_out


class SinusoidalEmbedding(nn.Module):
    def __init__(self, hidden_dim, min_period, max_period, dtype=torch.float32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.min_period = min_period
        self.max_period = max_period
        self.register_buffer(
            "freqs",
            2
            * math.pi
            / torch.logspace(
                math.log10(min_period),
                math.log10(max_period),
                hidden_dim // 2,
                dtype=dtype,
            ),
        )

    def forward(self, x):
        triarg = self.freqs * x.unsqueeze(1)
        # interleave sin and cos
        pe = torch.zeros(x.size(0), self.hidden_dim, device=x.device, dtype=x.dtype)
        pe[:, 0::2] = torch.sin(triarg)
        pe[:, 1::2] = torch.cos(triarg)
        return pe.unsqueeze(0)


class RevIN(nn.Module):
    """
    Affine reversible instance norm.
    """

    def __init__(self, channels):
        super().__init__()
        self.dims = channels
        self.norm_scale = nn.Parameter(torch.ones(1, channels))  # extra dim for batch
        self.norm_shift = nn.Parameter(torch.zeros(1, channels))
        self.unnorm_scale = nn.Parameter(torch.ones(1, channels))
        self.unnorm_shift = nn.Parameter(torch.zeros(1, channels))

    def normalize(self, x):
        rest = tuple(range(2, x.ndim))  # everything but batch and channels
        assert x.shape[1] == self.dims  # proper number of channels
        std, mean = torch.std_mean(x, axis=rest, keepdim=True)
        return ((x - mean) / std) * self.norm_scale + self.norm_shift, mean, std

    def forward(self, x):
        return self.normalize(x)

    def unnormalize(self, x, mean, std):
        x = (x - self.norm_shift) / self.norm_scale
        x = x * std + mean
        x = x * self.unnorm_scale + self.unnorm_shift
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
        cnn_feat = rearrange(x, "b (h w d) c -> b c h w d", h=H, w=W, d=D).contiguous()
        if self.stride == 1:
            out = self.proj(cnn_feat) + cnn_feat
        else:
            out = self.proj(cnn_feat)
        out = rearrange(out, "b c h w d -> b (h w d) c").contiguous()
        return out


class ViT(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        mlp_ratio=4,
        conv_kernel_size=31,
        num_heads=12,
        dropout=0.0,
        num_blocks=6,
        parallel_paths=2,
        image_channels=3,
        image_size=(256, 256),
        patch_size=(16, 16),
        num_registers=4,
        reattention=True,
        output_head=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.conv_kernel_size = conv_kernel_size
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.parallel_paths = parallel_paths
        self.image_channels = image_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_registers = num_registers
        self.spatial_dims = len(patch_size)
        assert len(patch_size) == self.spatial_dims
        self.n_patches: tuple[int, ...] = tuple(
            [image_size[d] // patch_size[d] for d in range(self.spatial_dims)]
        )

        self.input_head = nn.Conv2d(
            image_channels,
            embed_dim // 4,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        self.patch_embed = hMLP_stem(
            patch_size=patch_size,
            in_chans=embed_dim // 4,
            embed_dim=embed_dim,
            # spatial_dims=self.spatial_dims,
        )

        self.positional_encoding = nn.Parameter(
            torch.randn(
                1, np.prod(self.n_patches), embed_dim
            )  # extra dim at front for batch
        )
        # self.positional_encoding = PEG(embed_dim, embed_dim)

        self.register_tokens = nn.Parameter(torch.randn(num_registers, embed_dim))

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[
                ComplicatedTransformerBlock(
                    num_heads,
                    embed_dim,
                    mlp_ratio,
                    conv_kernel_size,
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
        B, C, H, W = x.shape

        # n_patches = tuple(
        #     [x.shape[2 + d] // self.patch_size[d] for d in range(self.spatial_dims)]
        # )

        # # split into patches
        # x = self.to_patch(x)  # (b, n_patches, pixels_per_patch)

        # x = einsum(x, self.input_head_weights, "b c1 h w, c1 c2 -> b c2 h w")
        x = self.input_head(x)

        # embed patches
        x = self.patch_embed(x)  # (b, embed_dim, nh, nw)

        x = rearrange(
            x, "b c h w -> b (h w) c"
        ).contiguous()  # (b, n_patches, embed_dim)

        # # add positional encoding and do dropout
        x = self.dropout(x + self.positional_encoding)  # (batch, n_patches, embed_dim)

        # add register tokens
        r = repeat(self.register_tokens, "n d -> b n d", b=B)

        x, pack_info = pack(
            [x, r], "b * d"
        )  # (batch, n_patches + n_registers, embed_dim)

        # pass through transformer blocks
        x = self.blocks(x)  # (batch, sequence_length, embed_dim)

        x, _ = unpack(x, pack_info, "b * d")

        # # pass through first transformer block
        # x = self.blocks[0](x)

        # # # add PEG encoding
        # # x = x + self.positional_encoding(x, *n_patches, 1)

        # # pass through remaining transformer blocks
        # x = self.blocks[1:](x)

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


class MAE(nn.Module):
    def __init__(
        self,
        vit,
        masking_fraction,
        decoder_dim=256,
        decoder_num_heads=8,
        decoder_num_blocks=2,
        decoder_dropout=0.0,
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
                ComplicatedTransformerBlock(
                    num_heads=decoder_num_heads,
                    embed_dim=decoder_dim,
                    mlp_ratio=4,
                    dropout=decoder_dropout,
                    parallel_paths=2,
                    layer_scale=1e-4,
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

        tokens = rearrange(
            tokens, "b c h w -> b (h w) c"
        ).contiguous()  # (b n_patches embed_dim)

        n_patches = tokens.shape[1]

        # add positional encoding
        tokens = tokens + self.vit.positional_encoding

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

        # encode, adding register tokens
        unmask_tokens, pack_info = pack(
            [unmask_tokens, self.vit.register_tokens], "b * d"
        )
        encoded_tokens = self.vit.blocks(unmask_tokens)  # (batch, n_unmask, embed_dim)
        encoded_tokens, _ = unpack(encoded_tokens, pack_info, "b * d")
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

    def step(self, batch, batch_idx):
        x = batch
        mask_patches, out_mask_patches, mask_idx = self(x)
        batch_indexer = torch.arange(batch.shape[0]).unsqueeze(-1)
        loss = F.mse_loss(
            out_mask_patches[batch_indexer, mask_idx],
            mask_patches[batch_indexer, mask_idx],
        )
        return loss


class LightningWrapper(pl.LightningModule):
    """
    Turns a torch.nn.Module into a pl.LightningModule
    """

    def __init__(self, modelclass, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        if "logging" not in kwargs:
            kwargs["logging"] = True

        if "custom_step" in kwargs:
            self.custom_step = kwargs["custom_step"]
        else:
            self.custom_step = None

        for k, v in kwargs.items():
            if not k.startswith("inner_"):
                setattr(self, k, v)

        if hasattr(self, "loss_mod") and hasattr(self, "loss_fn"):
            raise ValueError("cannot have both loss_mod and loss_fn")
        elif hasattr(self, "loss_mod"):
            assert hasattr(
                self, "loss_mod_args"
            ), "must have loss_mod_args if loss_mod is present"
            self.loss_fn = self.loss_mod(**self.loss_mod_args)

        self.modelclass = modelclass

        del_keys = ["lr", "loss_fn", "loss_mod", "loss_mod_args", "logging"]
        inner_kwargs = {k: v for k, v in kwargs.items() if k not in del_keys}

        # build model
        if "checkpoint" in kwargs:
            self.model = kwargs["checkpoint"]
        else:
            self.model = modelclass(**inner_kwargs)

        self.custom_step = hasattr(self.model, "step")

        assert (
            self.custom_step or "loss_fn" or "loss_mod" in kwargs
        ), "must have loss_fn, loss_mod or custom step (step method in modelclass)"
        assert "lr" in kwargs, "must have lr"

    @property
    def nparams(self):
        return sum([p.numel() for p in self.parameters()])

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, batch_idx):
        if self.custom_step:
            return self.model.step(batch, batch_idx)

        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def _log(self, log_name, loss):
        if self.logging:
            if not isinstance(loss, dict):
                self.log(
                    f"{log_name}_loss",
                    loss,
                    prog_bar=True,
                    logger=True,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                    rank_zero_only=True,
                )
                return loss
            else:
                log_dict = {f"{log_name}_{k}": v for k, v in loss.items()}
                self.log_dict(
                    log_dict,
                    prog_bar=True,
                    logger=True,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                    rank_zero_only=True,
                )
                return log_dict[f"{log_name}_loss"]

    def training_step(self, batch, batch_idx):
        loss_aux = self._step(batch, batch_idx)
        loss = self._log("train", loss_aux)
        return loss

    def validation_step(self, batch, batch_idx):
        loss_aux = self._step(batch, batch_idx)
        loss = self._log("val", loss_aux)
        return loss

    def testing_step(self, batch, batch_idx):
        loss_aux = self._step(batch, batch_idx)
        loss = self._log("test", loss_aux)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1e-3, total_iters=self.warmup_steps
                ),
                optim.lr_scheduler.ConstantLR(optimizer, factor=1),
            ],
            milestones=[self.warmup_steps],
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": True,
        }
        return [optimizer], [scheduler]
