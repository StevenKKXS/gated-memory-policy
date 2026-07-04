from collections import OrderedDict
from pathlib import Path

import einops
import torch
import torch.nn.functional as F
from torch import nn

from imitation_learning.models.encoders.image_encoders import BaseImageEncoder


class _QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class _ResidualAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(width, width * 4)),
                    ("gelu", _QuickGELU()),
                    ("c_proj", nn.Linear(width * 4, width)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm(width)

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class _Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[_ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resblocks(x)


class _LongCLIPVisionTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int = 224,
        patch_size: int = 14,
        width: int = 1024,
        layers: int = 24,
        heads: int = 16,
        output_dim: int = 768,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        scale = width**-0.5
        grid_size = input_resolution // patch_size
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(grid_size * grid_size + 1, width)
        )
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = _Transformer(width, layers, heads)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        cls = self.class_embedding.to(dtype=x.dtype)
        cls = cls + torch.zeros(
            x.shape[0],
            1,
            x.shape[-1],
            dtype=x.dtype,
            device=x.device,
        )
        x = torch.cat([cls, x], dim=1)
        x = x + self.positional_embedding.to(dtype=x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        return x @ self.proj


class LongCLIPImageEncoder(BaseImageEncoder):
    def __init__(
        self,
        weights_path: str,
        feature_aggregation: str = "map",
        apply_image_norm: bool = True,
        input_resolution: int = 224,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if feature_aggregation != "map":
            raise ValueError("LongCLIPImageEncoder only supports map aggregation")

        self.feature_aggregation = feature_aggregation
        self.apply_image_norm = apply_image_norm
        self.input_resolution = input_resolution
        self.feature_dim = 768
        self.token_num = 1
        self.img_mean = nn.Parameter(torch.tensor([0.48145466, 0.4578275, 0.40821073]))
        self.img_std = nn.Parameter(torch.tensor([0.26862954, 0.26130258, 0.27577711]))
        self.img_mean.requires_grad = False
        self.img_std.requires_grad = False

        self.model = _LongCLIPVisionTransformer(input_resolution=input_resolution)
        state_dict = torch.load(Path(weights_path), map_location="cpu")
        visual_state_dict = {
            f"model.{key.removeprefix('visual.')}": value
            for key, value in state_dict.items()
            if key.startswith("visual.")
        }
        load_result = self.load_state_dict(visual_state_dict, strict=False)
        missing = [key for key in load_result.missing_keys if key.startswith("model.")]
        if missing:
            raise RuntimeError(f"Missing LongCLIP visual keys: {missing[:8]}")
        if self.frozen:
            for param in self.parameters():
                param.requires_grad = False

    def _prepare_images(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.input_resolution, self.input_resolution):
            x = F.interpolate(
                x,
                size=(self.input_resolution, self.input_resolution),
                mode="bicubic",
                align_corners=False,
            )
        if self.apply_image_norm:
            x = (x - self.img_mean[None, :, None, None]) / self.img_std[
                None, :, None, None
            ]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) in (5, 6), f"x.shape: {x.shape}"
        batch_size = x.shape[0]
        has_traj_dim = len(x.shape) == 6
        if has_traj_dim:
            x = einops.rearrange(x, "b t l c h w -> (b t) l c h w")

        current_batch, obs_len, *image_shape = x.shape
        assert obs_len == self.image_meta.length
        assert tuple(image_shape) == self.image_shape
        x = einops.rearrange(x, "b l c h w -> (b l) c h w")
        x = self._prepare_images(x)
        with torch.set_grad_enabled(not self.frozen):
            features = self.model(x)
        features = einops.rearrange(
            features,
            "(b l) f -> b l 1 f",
            b=current_batch,
            l=obs_len,
        )
        if has_traj_dim:
            features = einops.rearrange(
                features,
                "(b t) l n f -> b t l n f",
                b=batch_size,
            )
        return features
