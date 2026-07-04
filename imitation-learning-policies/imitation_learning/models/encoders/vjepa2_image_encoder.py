import einops
import torch
import torch.nn.functional as F
from torch import nn

from imitation_learning.models.encoders.image_encoders import BaseImageEncoder


class VJEPA2ImageEncoder(BaseImageEncoder):
    """Frozen V-JEPA 2 frame encoder for QFrame evidence ranking.

    The HF VJEPA2 model is a video model, so each input image is treated as a
    one-frame clip. The model repeats short clips internally when needed for
    the tubelet size. We expose one pooled token per input frame to keep the
    QFrame evidence feature scale comparable to the LongCLIP variant.
    """

    def __init__(
        self,
        weights_path: str,
        feature_aggregation: str = "mean",
        apply_image_norm: bool = True,
        input_resolution: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if feature_aggregation != "mean":
            raise ValueError("VJEPA2ImageEncoder only supports mean aggregation")

        from transformers import VJEPA2Model

        self.feature_aggregation = feature_aggregation
        self.apply_image_norm = apply_image_norm
        self.input_resolution = input_resolution
        self.model = VJEPA2Model.from_pretrained(weights_path, local_files_only=True)
        self.model.eval()

        hidden_size = int(getattr(self.model.config, "hidden_size", 1024))
        self.feature_dim = hidden_size
        self.token_num = 1
        self.img_mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]))
        self.img_std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]))
        self.img_mean.requires_grad = False
        self.img_std.requires_grad = False

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
        pixel_values_videos = einops.rearrange(x, "b c h w -> b 1 c h w")
        with torch.set_grad_enabled(not self.frozen):
            outputs = self.model(
                pixel_values_videos=pixel_values_videos,
                skip_predictor=True,
            )
            features = outputs.last_hidden_state.mean(dim=1)

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
