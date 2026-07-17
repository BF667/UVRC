# coding: utf-8
"""
Shared utilities for audio separation models.

Extracts duplicated STFT, get_act, get_norm, Upscale, Downscale, and
subband-conversion helpers that were previously copy-pasted across
mdx23c_tfc_tdf_v3, segm_models, torchseg_models, and upernet_swin_transformers.
"""

from functools import partial

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# STFT / iSTFT helper
# ---------------------------------------------------------------------------

class STFT:
    """Short-Time Fourier Transform wrapper used by most separator models.

    Parameters
    ----------
    config : object
        Must expose ``n_fft``, ``hop_length``, and ``dim_f`` attributes.
    """

    def __init__(self, config):
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True)
        self.dim_f = config.dim_f

    def __call__(self, x):
        window = self.window.to(x.device)
        batch_dims = x.shape[:-2]
        c, t = x.shape[-2:]
        x = x.reshape([-1, t])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
            return_complex=True,
        )
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([*batch_dims, c, 2, -1, x.shape[-1]]).reshape(
            [*batch_dims, c * 2, -1, x.shape[-1]]
        )
        return x[..., : self.dim_f, :]

    def inverse(self, x):
        window = self.window.to(x.device)
        batch_dims = x.shape[:-3]
        c, f, t = x.shape[-3:]
        n = self.n_fft // 2 + 1
        f_pad = torch.zeros([*batch_dims, c, n - f, t]).to(x.device)
        x = torch.cat([x, f_pad], -2)
        x = x.reshape([*batch_dims, c // 2, 2, n, t]).reshape([-1, 2, n, t])
        x = x.permute([0, 2, 3, 1])
        x = x[..., 0] + x[..., 1] * 1j
        x = torch.istft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
        )
        x = x.reshape([*batch_dims, 2, -1])
        return x


# ---------------------------------------------------------------------------
# Activation / Normalisation factories
# ---------------------------------------------------------------------------

def get_act(act_type: str) -> nn.Module:
    """Return an activation module by name."""
    if act_type == "gelu":
        return nn.GELU()
    elif act_type == "relu":
        return nn.ReLU()
    elif act_type[:3] == "elu":
        alpha = float(act_type.replace("elu", ""))
        return nn.ELU(alpha)
    else:
        raise ValueError(f"Unknown activation type: {act_type}")


def get_norm(norm_type: str):
    """Return a norm constructor partially applied with *norm_type*."""

    def norm(c, norm_type):
        if norm_type == "BatchNorm":
            return nn.BatchNorm2d(c)
        elif norm_type == "InstanceNorm":
            return nn.InstanceNorm2d(c, affine=True)
        elif "GroupNorm" in norm_type:
            g = int(norm_type.replace("GroupNorm", ""))
            return nn.GroupNorm(num_groups=g, num_channels=c)
        else:
            return nn.Identity()

    return partial(norm, norm_type=norm_type)


# ---------------------------------------------------------------------------
# Small building blocks
# ---------------------------------------------------------------------------

class Upscale(nn.Module):
    def __init__(self, in_c, out_c, scale, norm, act):
        super().__init__()
        self.conv = nn.Sequential(
            norm(in_c),
            act,
            nn.ConvTranspose2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=scale,
                stride=scale,
                bias=False,
            ),
        )

    def forward(self, x):
        return self.conv(x)


class Downscale(nn.Module):
    def __init__(self, in_c, out_c, scale, norm, act):
        super().__init__()
        self.conv = nn.Sequential(
            norm(in_c),
            act,
            nn.Conv2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=scale,
                stride=scale,
                bias=False,
            ),
        )

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# Subband conversion helpers
# ---------------------------------------------------------------------------

def cac2cws(x: torch.Tensor, num_subbands: int) -> torch.Tensor:
    """Channel-axis-channels -> channel-width-subbands reshape."""
    k = num_subbands
    b, c, f, t = x.shape
    x = x.reshape(b, c, k, f // k, t)
    x = x.reshape(b, c * k, f // k, t)
    return x


def cws2cac(x: torch.Tensor, num_subbands: int) -> torch.Tensor:
    """Channel-width-subbands -> channel-axis-channels reshape."""
    k = num_subbands
    b, c, f, t = x.shape
    x = x.reshape(b, c // k, k, f, t)
    x = x.reshape(b, c // k, f * k, t)
    return x
