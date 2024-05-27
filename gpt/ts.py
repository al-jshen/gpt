import torch
import torch.nn as nn
from gpt.model import MLP, SelfAttention, CrossAttention, SinusoidalEmbedding, RevIN
from einops import rearrange


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        num_layers,
        num_heads,
        mlp_ratio,
        dropout,
        min_period,
        max_period,
    ):
        super().__init__()

        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = SinusoidalEmbedding(
            embed_dim, min_period, max_period
        )
        self.attns = nn.Sequential(
            *[
                SelfAttention(num_heads=num_heads, embed_dim=embed_dim, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.mlps = nn.Sequential(
            *[
                MLP(embed_dim, embed_dim * mlp_ratio, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.attn_norms = nn.Sequential(
            *[nn.LayerNorm(embed_dim) for _ in range(num_layers)]
        )
        self.mlp_norms = nn.Sequential(
            *[nn.LayerNorm(embed_dim) for _ in range(num_layers)]
        )

    def forward(self, t, x):
        # x is b n c_in and normalized
        temb = self.positional_encoding(t)  # 1 n c

        x = self.embedding(x)  # b n c_in -> b n c
        x = x + temb  # b n c

        for attn, attn_norm, mlp, mlp_norm in zip(
            self.attns, self.attn_norms, self.mlps, self.mlp_norms
        ):
            x = x + attn(attn_norm(x))
            x = x + mlp(mlp_norm(x))

        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        num_layers,
        num_heads,
        mlp_ratio,
        dropout,
        min_period,
        max_period,
    ):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = SinusoidalEmbedding(
            embed_dim, min_period, max_period
        )
        self.sattns = nn.Sequential(
            *[
                SelfAttention(
                    num_heads=num_heads,
                    embed_dim=embed_dim,
                    dropout=dropout,
                    is_causal=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.cattns = nn.Sequential(
            *[
                CrossAttention(
                    num_heads=num_heads,
                    embed_dim=embed_dim,
                    dropout=dropout,
                    use_rope=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.mlps = nn.Sequential(
            *[
                MLP(embed_dim, embed_dim * mlp_ratio, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.sattn_norms = nn.Sequential(
            *[nn.LayerNorm(embed_dim) for _ in range(num_layers)]
        )
        self.cattn_norms = nn.Sequential(
            *[nn.LayerNorm(embed_dim) for _ in range(num_layers)]
        )
        self.mlp_norms = nn.Sequential(
            *[nn.LayerNorm(embed_dim) for _ in range(num_layers)]
        )

    def forward(self, t, x, enc_out):
        temb = self.positional_encoding(t)  # 1 n c

        x = self.embedding(x)  # b n c_in -> b n c
        x = x + temb  # b n c

        for sattn, sattn_norm, cattn, cattn_norm, mlp, mlp_norm in zip(
            self.sattns,
            self.sattn_norms,
            self.cattns,
            self.cattn_norms,
            self.mlps,
            self.mlp_norms,
        ):
            x = x + sattn(sattn_norm(x))
            x = x + cattn(cattn_norm(x), enc_out)
            x = x + mlp(mlp_norm(x))

        return x


class TSTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        num_layers,
        num_heads,
        mlp_ratio,
        dropout,
        min_period,
        max_period,
    ):
        super().__init__()
        args = dict(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            min_period=min_period,
            max_period=max_period,
        )
        self.encoder = TransformerEncoder(**args)
        self.decoder = TransformerDecoder(**args)
        self.revin = RevIN(input_dim)
        self.output_head = nn.Linear(embed_dim, input_dim)

    def forward(self, t, x):
        x = rearrange(x, "b n c -> b c n")
        x_norm, mean, std = self.revin.normalize(
            x
        )  # b c_in n, mean and std are b c_in 1
        x_norm = rearrange(x_norm, "b c n -> b n c")  # b n c_in
        enc_out = self.encoder(t, x_norm)  # b n c_emb
        dec_out = self.decoder(t, x_norm, enc_out)  # b n c_emb
        out = self.output_head(dec_out)  # b n c_in
        out = rearrange(out, "b n c -> b c n")
        out = self.revin.unnormalize(out, mean, std)
        out = rearrange(out, "b c n -> b n c")
        return out

    def forecast(self, t, x):
        """
        For t with shape n_t and x with shape b n c with n < n_t, forecast the future n_t - n steps.
        """
        x = rearrange(x, "b n c -> b c n")
        x_norm, mean, std = self.revin.normalize(x)
        x_norm = rearrange(x_norm, "b c n -> b n c")
        n = x_norm.shape[1]
        enc_out = self.encoder(t[:n], x_norm)
        for i in range(n, len(t)):
            dec_out = self.decoder(t[:i], x_norm, enc_out)[..., -1:, :]
            out = self.output_head(dec_out)
            x_norm = torch.cat([x_norm, out], dim=1)

        out = rearrange(x_norm, "b n c -> b c n")
        out = self.revin.unnormalize(out, mean, std)
        out = rearrange(out, "b c n -> b n c")

        return out
