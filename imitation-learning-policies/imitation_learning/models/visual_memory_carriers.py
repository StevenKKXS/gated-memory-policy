import torch
import torch.nn as nn


class _NoCudnnFlattenGRU(nn.GRU):
    """GRU variant that avoids cuDNN flatten/version checks on shared hosts."""

    def flatten_parameters(self) -> None:
        return None


class VisualGRUMemoryCarrier(nn.Module):
    """GRU visual memory manager that emits late-cue carrier tokens."""

    def __init__(
        self,
        feature_dim: int | None = None,
        hidden_dim: int | None = None,
        memory_token_num: int = 1,
        num_layers: int = 1,
        dropout: float = 0.0,
        input_dim: int | None = None,
        output_dim: int | None = None,
        token_num: int | None = None,
        max_len: int | None = None,
    ):
        super().__init__()
        self.sequence_mode = input_dim is not None or output_dim is not None
        if self.sequence_mode:
            assert input_dim is not None and output_dim is not None
            feature_dim = input_dim
            self.output_dim = output_dim
            memory_token_num = token_num or memory_token_num
        else:
            assert feature_dim is not None
            self.output_dim = feature_dim
        assert feature_dim > 0
        assert memory_token_num > 0
        assert num_layers > 0
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim or feature_dim
        self.memory_token_num = memory_token_num
        self.max_len = max_len
        self.input_norm = nn.LayerNorm(feature_dim)
        self.gru = _NoCudnnFlattenGRU(
            input_size=feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Linear(
            self.hidden_dim,
            memory_token_num * self.output_dim,
        )
        self.output_norm = nn.LayerNorm(self.output_dim)

    def forward(
        self,
        history_img_features: torch.Tensor,
        history_mask: torch.Tensor | None = None,
        force_zero: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if history_img_features.dim() != 4:
            raise ValueError(
                "history_img_features must have shape [B, T, token_num, D]"
            )
        batch_size, history_len, _token_num, feature_dim = history_img_features.shape
        if feature_dim != self.feature_dim:
            raise ValueError(
                f"Expected feature dim {self.feature_dim}, got {feature_dim}"
            )
        if self.max_len is not None and history_len > self.max_len:
            history_img_features = history_img_features[:, : self.max_len]
            if history_mask is not None:
                history_mask = history_mask[:, : self.max_len]
            history_len = self.max_len
        valid_mask = _coerce_history_mask(
            history_mask,
            batch_size,
            history_len,
            history_img_features.device,
        )
        if self.sequence_mode:
            return self._forward_sequence(history_img_features, valid_mask, force_zero)
        return self._forward_carrier(history_img_features, valid_mask, force_zero)

    def _forward_sequence(
        self,
        history_img_features: torch.Tensor,
        valid_mask: torch.Tensor,
        force_zero: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, history_len = valid_mask.shape
        if force_zero:
            out = history_img_features.new_zeros(
                batch_size,
                history_len,
                self.memory_token_num,
                self.output_dim,
            )
            return out, valid_mask
        frame_features = self.input_norm(history_img_features.mean(dim=2))
        frame_features = frame_features * valid_mask[:, :, None]
        with torch.backends.cudnn.flags(enabled=False):
            output, _hidden = self.gru(frame_features)
        out = self.output(output).view(
            batch_size,
            history_len,
            self.memory_token_num,
            self.output_dim,
        )
        out = self.output_norm(out)
        out = out.masked_fill(~valid_mask[:, :, None, None], 0.0)
        return out, valid_mask

    def _forward_carrier(
        self,
        history_img_features: torch.Tensor,
        valid_mask: torch.Tensor,
        force_zero: bool,
    ) -> torch.Tensor:
        batch_size, history_len = valid_mask.shape
        if force_zero:
            return history_img_features.new_zeros(
                batch_size,
                self.memory_token_num,
                1,
                self.output_dim,
            )
        frame_features = self.input_norm(history_img_features.mean(dim=2))
        frame_features = frame_features * valid_mask[:, :, None]
        lengths = valid_mask.long().sum(dim=1)
        has_valid = lengths > 0
        safe_lengths = lengths.clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            frame_features,
            safe_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        with torch.backends.cudnn.flags(enabled=False):
            _output, hidden = self.gru(packed)
        last_hidden = hidden[-1]
        carrier = self.output(last_hidden).view(
            batch_size,
            self.memory_token_num,
            self.output_dim,
        )
        carrier = self.output_norm(carrier)
        carrier = carrier.masked_fill(~has_valid[:, None, None], 0.0)
        return carrier[:, :, None, :]


class LearnedLateCueSelector(nn.Module):
    """PMA-style selector that maps visual history features to carrier tokens."""

    def __init__(
        self,
        feature_dim: int | None = None,
        memory_token_num: int = 1,
        max_history_len: int = 64,
        num_heads: int = 8,
        dropout: float = 0.0,
        input_dim: int | None = None,
        output_dim: int | None = None,
        token_num: int | None = None,
        max_len: int | None = None,
    ):
        super().__init__()
        self.sequence_mode = input_dim is not None or output_dim is not None
        if self.sequence_mode:
            assert input_dim is not None and output_dim is not None
            feature_dim = input_dim
            self.output_dim = output_dim
            memory_token_num = token_num or memory_token_num
            max_history_len = max_len or max_history_len
        else:
            assert feature_dim is not None
            self.output_dim = feature_dim
        assert feature_dim > 0
        assert memory_token_num > 0
        assert max_history_len > 0
        assert num_heads > 0
        assert feature_dim % num_heads == 0
        self.feature_dim = feature_dim
        self.memory_token_num = memory_token_num
        self.max_history_len = max_history_len
        self.query_tokens = nn.Parameter(
            torch.empty(1, memory_token_num, feature_dim)
        )
        self.temporal_embedding = nn.Parameter(
            torch.zeros(1, max_history_len, feature_dim)
        )
        self.input_norm = nn.LayerNorm(feature_dim)
        self.query_norm = nn.LayerNorm(feature_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output = nn.Linear(feature_dim, self.output_dim)
        self.output_norm = nn.LayerNorm(self.output_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.temporal_embedding, mean=0.0, std=0.01)

    def forward(
        self,
        history_img_features: torch.Tensor,
        history_mask: torch.Tensor | None = None,
        force_zero: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if history_img_features.dim() != 4:
            raise ValueError(
                "history_img_features must have shape [B, T, token_num, D]"
            )
        batch_size, history_len, _token_num, feature_dim = history_img_features.shape
        if feature_dim != self.feature_dim:
            raise ValueError(
                f"Expected feature dim {self.feature_dim}, got {feature_dim}"
            )
        if history_len > self.max_history_len:
            history_img_features = history_img_features[:, : self.max_history_len]
            if history_mask is not None:
                history_mask = history_mask[:, : self.max_history_len]
            history_len = self.max_history_len
        valid_mask = _coerce_history_mask(
            history_mask,
            batch_size,
            history_len,
            history_img_features.device,
        )
        if self.sequence_mode:
            return self._forward_sequence(history_img_features, valid_mask, force_zero)
        return self._forward_carrier(history_img_features, valid_mask, force_zero)

    def _forward_sequence(
        self,
        history_img_features: torch.Tensor,
        valid_mask: torch.Tensor,
        force_zero: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, history_len = valid_mask.shape
        if force_zero:
            out = history_img_features.new_zeros(
                batch_size,
                history_len,
                self.memory_token_num,
                self.output_dim,
            )
            return out, valid_mask
        frame_features = self.input_norm(history_img_features.mean(dim=2))
        frame_features = frame_features + self.temporal_embedding[:, :history_len]
        selected = self.output(frame_features)[:, :, None, :].expand(
            -1,
            -1,
            self.memory_token_num,
            -1,
        )
        selected = self.output_norm(selected)
        selected = selected.masked_fill(~valid_mask[:, :, None, None], 0.0)
        return selected, valid_mask

    def _forward_carrier(
        self,
        history_img_features: torch.Tensor,
        valid_mask: torch.Tensor,
        force_zero: bool,
    ) -> torch.Tensor:
        batch_size, history_len = valid_mask.shape
        if force_zero:
            return history_img_features.new_zeros(
                batch_size,
                self.memory_token_num,
                1,
                self.output_dim,
            )
        frame_features = history_img_features.mean(dim=2)
        frame_features = frame_features + self.temporal_embedding[:, :history_len]
        frame_features = self.input_norm(frame_features)
        has_valid = valid_mask.any(dim=1)
        safe_key_padding_mask = ~valid_mask
        if not bool(has_valid.all()):
            safe_key_padding_mask = safe_key_padding_mask.clone()
            safe_key_padding_mask[~has_valid] = False
            frame_features = frame_features.clone()
            frame_features[~has_valid] = 0.0
        query = self.query_norm(self.query_tokens.expand(batch_size, -1, -1))
        carrier, _weights = self.cross_attn(
            query=query,
            key=frame_features,
            value=frame_features,
            key_padding_mask=safe_key_padding_mask,
            need_weights=False,
        )
        carrier = self.output_norm(self.output(carrier))
        carrier = carrier.masked_fill(~has_valid[:, None, None], 0.0)
        return carrier[:, :, None, :]


def _coerce_history_mask(
    history_mask: torch.Tensor | None,
    batch_size: int,
    history_len: int,
    device: torch.device,
) -> torch.Tensor:
    if history_mask is None:
        return torch.ones(
            batch_size,
            history_len,
            device=device,
            dtype=torch.bool,
        )
    valid_mask = history_mask.to(device=device, dtype=torch.bool)
    if valid_mask.shape != (batch_size, history_len):
        raise ValueError("history_mask must have shape [B, T] matching features")
    return valid_mask
