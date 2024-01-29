import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from gpt.model import AxialAttention, hMLP_output, hMLP_stem, TransformerBlock


class UNetBlock(nn.Module):
    def __init__(
        self, in_dims, out_dims, patch_size=(1, 1, 1), num_heads=8, attention=True
    ):
        super().__init__()

        self.hmlp_stem = (
            hMLP_stem(patch_size=patch_size, in_chans=in_dims, embed_dim=out_dims)
            if out_dims > in_dims
            else hMLP_output(
                patch_size=patch_size, embed_dim=in_dims, out_chans=out_dims
            )
        )
        self.norm = nn.LayerNorm(out_dims)

        self.conv1 = nn.Conv3d(
            in_channels=out_dims, out_channels=out_dims, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv3d(
            in_channels=out_dims, out_channels=out_dims, kernel_size=3, padding=1
        )

        if attention:
            self.norm1 = nn.LayerNorm(out_dims)
            self.norm2 = nn.LayerNorm(out_dims)

            self.attn1 = AxialAttention(
                embed_dim=out_dims, num_heads=num_heads, reattention=True
            )
            self.attn2 = AxialAttention(
                embed_dim=out_dims, num_heads=num_heads, reattention=True
            )

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.attention = attention

    def forward(self, x):
        x = self.hmlp_stem(x)
        b, c, h, w, d = x.shape

        x = rearrange(x, "b c h w d -> b (h w d) c")
        x = self.norm(x)
        x = rearrange(x, "b (h w d) c -> b c h w d", h=h, w=w, d=d)

        x = self.conv1(x)
        x = F.gelu(x)

        if self.attention:
            x = rearrange(x, "b c h w d -> b (h w d) c")
            x = self.norm1(x)
            x = rearrange(x, "b (h w d) c -> b c h w d", h=h, w=w, d=d)
            x = self.attn1(x)

        x = self.conv2(x)
        x = F.gelu(x)

        if self.attention:
            x = rearrange(x, "b c h w d -> b (h w d) c")
            x = self.norm2(x)
            x = rearrange(x, "b (h w d) c -> b c h w d", h=h, w=w, d=d)
            x = self.attn2(x)

        return x


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv3d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_dims,
        hidden_dims=(64, 128),
        patch_size=(1, 1, 1),
        upsample_res=1,
        num_heads=4,
        attention=True,
    ):
        super().__init__()
        self.in_dims = in_dims
        self.hidden_dims = hidden_dims
        self.patch_size = patch_size
        self.n_hidden = len(hidden_dims)
        self.upsample_res = upsample_res

        self.down = [
            UNetBlock(
                in_dims, hidden_dims[0], patch_size=patch_size, num_heads=num_heads, attention=attention
            ),
            Downsample(hidden_dims[0]),
        ]
        for i in range(self.n_hidden - 1):
            self.down.append(
                UNetBlock(hidden_dims[i], hidden_dims[i + 1], num_heads=num_heads, attention=attention)
            )
            self.down.append(Downsample(hidden_dims[i + 1]))
        self.down = nn.ModuleList(self.down)

        self.mid = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=hidden_dims[-1], mlp_ratio=2, num_heads=8, axial_3d=True
                )
                for _ in range(2)
            ]
        )

        self.up = []
        for i in range(1, self.n_hidden):
            self.up.append(Upsample(hidden_dims[self.n_hidden - i]))
            self.up.append(
                UNetBlock(
                    hidden_dims[self.n_hidden - i],
                    hidden_dims[self.n_hidden - (i + 1)],
                    num_heads=num_heads,
                    attention=attention
                )
            )
        self.up.append(Upsample(hidden_dims[0]))
        self.up.append(
            UNetBlock(hidden_dims[0], in_dims, patch_size=patch_size, num_heads=in_dims, attention=attention)
        )

        self.up = nn.ModuleList(self.up)

        self.final = []

        if upsample_res > 1:
            assert upsample_res == 2, "upsample res must be 1 or 2 right now"
            self.final = nn.Sequential(
                Upsample(in_dims), nn.Conv3d(in_dims, in_dims, 1, 1)
            )
        else:
            self.final = nn.Identity()

        self.dwconvs1 = nn.ModuleList(
            [nn.Conv3d(d, d, 1, 1, groups=d) for d in (*(hidden_dims[::-1]), in_dims)]
        )
        self.dwconvs2 = nn.ModuleList(
            [nn.Conv3d(d, d, 1, 1, groups=d) for d in (*(hidden_dims[::-1]), in_dims)]
        )

    def forward(self, x):
        res_x = [x]

        for m in self.down:
            x = m(x)
            if isinstance(m, UNetBlock):
                res_x.append(x)

        for m in self.mid:
            x = m(x)

        res_x = res_x[::-1]
        res_ctr = 0

        for m in self.up:
            if isinstance(m, UNetBlock):
                w = res_x[res_ctr]
                w = self.dwconvs1[res_ctr](w) + x
                w = F.gelu(w)
                w = self.dwconvs2[res_ctr](w)
                res_ctr += 1
                w_h = torch.softmax(w, dim=-3)
                attn_h = w_h * x
                w_w = torch.softmax(w, dim=-2)
                attn_w = w_w * x
                w_d = torch.softmax(w, dim=-1)
                attn_d = w_d * x
                attn = (attn_h + attn_w + attn_d) / 3
                x = m(attn)
            else:
                x = m(x)

        x = self.final(x)
        return x
