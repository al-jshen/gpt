import torch
from einops import rearrange


def unpatchify(x, channels=1):
    if x.ndim == 2:
        return _unpatchify(x, channels)
    if x.ndim == 3:
        return torch.vmap(_unpatchify)(x, channels=channels)
    else:
        return NotImplementedError


def _unpatchify(x, channels=1):
    """Unpatchify a tensor."""
    N, D = x.shape
    n = int(N**0.5)
    ps = int((D // channels) ** 0.5)
    return (
        rearrange(
            x,
            "(nh nw) (ph pw c) -> c nh ph nw pw",
            nh=n,
            nw=n,
            ph=ps,
            pw=ps,
            c=channels,
        )
        .contiguous()
        .view(channels, n * ps, n * ps)
    )
