from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import CLIPTokenizer

from imitation_learning.models.encoders.longclip_image_encoder import _Transformer


class LongCLIPTextEncoder(nn.Module):
    def __init__(
        self,
        weights_path: str,
        tokenizer_name: str = "openai/clip-vit-large-patch14",
        context_length: int = 248,
        frozen: bool = True,
    ):
        super().__init__()
        self.context_length = context_length
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)

        state_dict = torch.load(Path(weights_path), map_location="cpu")
        token_weight = state_dict["token_embedding.weight"]
        width = int(token_weight.shape[1])
        layers = len(
            {
                int(key.split(".")[2])
                for key in state_dict
                if key.startswith("transformer.resblocks.")
            }
        )
        heads = width // 64

        self.token_embedding = nn.Embedding(int(token_weight.shape[0]), width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        self.transformer = _Transformer(width, layers, heads)
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Parameter(torch.empty(width, width))

        text_state = {
            key: value
            for key, value in state_dict.items()
            if key.startswith("token_embedding.")
            or key.startswith("transformer.")
            or key.startswith("ln_final.")
            or key == "text_projection"
        }
        text_state["positional_embedding"] = state_dict["positional_embedding"][
            :context_length
        ]
        load_result = self.load_state_dict(text_state, strict=False)
        unexpected = list(load_result.unexpected_keys)
        missing = list(load_result.missing_keys)
        if unexpected or missing:
            raise RuntimeError(
                "LongCLIP text key mismatch: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        if frozen:
            for param in self.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def encode_text(self, texts: list[str] | tuple[str, ...]) -> torch.Tensor:
        device = next(self.parameters()).device
        tokens = self.tokenizer(
            list(texts),
            padding="max_length",
            max_length=self.context_length,
            truncation=True,
            return_tensors="pt",
        )["input_ids"].to(device)
        return self.forward(tokens)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(token_ids)
        x = x + self.positional_embedding.to(dtype=x.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        eot_positions = token_ids.argmax(dim=-1)
        x = x[torch.arange(x.shape[0], device=x.device), eot_positions]
        return x @ self.text_projection
