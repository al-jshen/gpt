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


def patchify(x, patch_size):
    if x.ndim == 3:
        return _patchify(x, patch_size)
    if x.ndim == 4:
        return torch.vmap(_patchify)(x, patch_size=patch_size)
    else:
        return NotImplementedError


def _patchify(x, patch_size):
    """Patchify an image."""
    C, H, W = x.shape
    ph, pw = patch_size
    nh, nw = H // ph, W // pw
    return (
        rearrange(
            x,
            "c (nh ph) (nw pw) -> (nh nw) (ph pw c)",
            nh=nh,
            nw=nw,
            ph=ph,
            pw=pw,
            c=C,
        )
        .contiguous()
        .view(nh * nw, ph * pw * C)
    )
