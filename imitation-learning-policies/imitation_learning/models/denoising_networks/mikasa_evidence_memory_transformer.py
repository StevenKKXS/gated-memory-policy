import torch
import torch.nn as nn
from timm.models.vision_transformer import RmsNorm

from imitation_learning.models.denoising_networks.mikasa_memory_transformer import (
    MikasaMemoryTransformer,
)
from imitation_learning.models.denoising_networks.modules import HistoryCrossAttention
from imitation_learning.models.evidence_selection import (
    select_qframe_causal_evidence_masks,
)


class MikasaEvidenceMemoryTransformer(MikasaMemoryTransformer):
    """
    Mikasa-only evidence memory branch on top of the validated memory transformer.

    The base class still owns DiT blocks, history projection, train parallel_forward,
    and eval rolling-history inputs. This subclass only reads selected evidence after
    selected late blocks.
    """

    def __init__(
        self,
        evidence_memory_enabled: bool = True,
        evidence_memory_num_last_blocks: int = 2,
        evidence_candidate_max_num: int = 8,
        evidence_high_topk: int = 2,
        evidence_low_topk: int = 2,
        evidence_memory_query_source: str = "global_cond",
        evidence_memory_gate_init_bias: float = -4.0,
        evidence_memory_out_init_gain: float = 0.0,
        evidence_memory_residual_scale: float = 0.1,
        **kwargs,
    ):
        history_attention_type = kwargs.get("history_attention_type", "token_wise")
        ssmax_scaling_param = kwargs.get("ssmax_scaling_param")
        super().__init__(**kwargs)

        self.evidence_memory_enabled: bool = evidence_memory_enabled
        self.evidence_memory_num_last_blocks: int = evidence_memory_num_last_blocks
        self.evidence_candidate_max_num: int = evidence_candidate_max_num
        self.evidence_high_topk: int = evidence_high_topk
        self.evidence_low_topk: int = evidence_low_topk
        self.evidence_memory_query_source: str = evidence_memory_query_source
        self.evidence_memory_gate_init_bias: float = evidence_memory_gate_init_bias
        self.evidence_memory_out_init_gain: float = evidence_memory_out_init_gain
        self.evidence_memory_residual_scale: float = evidence_memory_residual_scale

        assert self.evidence_memory_num_last_blocks >= 0
        assert self.evidence_candidate_max_num >= 0
        assert self.evidence_high_topk >= 0
        assert self.evidence_low_topk >= 0
        assert self.evidence_memory_query_source in ("denoising", "global_cond")
        assert self.evidence_memory_residual_scale >= 0

        if self.evidence_memory_enabled and self.evidence_memory_num_last_blocks > 0:
            start_idx = max(self.layer_num - self.evidence_memory_num_last_blocks, 0)
            self.evidence_memory_block_indices: tuple[int, ...] = tuple(
                range(start_idx, self.layer_num)
            )
        else:
            self.evidence_memory_block_indices = tuple()

        self.evidence_memory_norms: nn.ModuleDict = nn.ModuleDict()
        self.evidence_high_attns: nn.ModuleDict = nn.ModuleDict()
        self.evidence_low_attns: nn.ModuleDict = nn.ModuleDict()
        self.evidence_memory_outs: nn.ModuleDict = nn.ModuleDict()
        self.evidence_memory_gates: nn.ModuleDict = nn.ModuleDict()

        for block_idx in self.evidence_memory_block_indices:
            key = str(block_idx)
            self.evidence_memory_norms[key] = RmsNorm(self.hidden_dim, eps=1e-6)
            self.evidence_high_attns[key] = HistoryCrossAttention(
                dim=self.hidden_dim,
                head_num=self.head_num,
                qkv_bias=True,
                qk_norm=True,
                norm_layer=RmsNorm,
                attention_type=history_attention_type,
                ssmax_scaling_param=ssmax_scaling_param,
            )
            self.evidence_low_attns[key] = HistoryCrossAttention(
                dim=self.hidden_dim,
                head_num=self.head_num,
                qkv_bias=True,
                qk_norm=True,
                norm_layer=RmsNorm,
                attention_type=history_attention_type,
                ssmax_scaling_param=ssmax_scaling_param,
            )
            self.evidence_memory_outs[key] = nn.Linear(
                self.hidden_dim,
                self.hidden_dim,
                bias=False,
            )
            self.evidence_memory_gates[key] = nn.Sequential(
                nn.Linear(self.hidden_dim * 2 + 1, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 1),
            )

        self.initialize_evidence_memory_weights()

    def _normalize_evidence_query_features(
        self,
        evidence_query_features: torch.Tensor,
    ) -> torch.Tensor:
        if evidence_query_features.dim() == 2:
            return evidence_query_features
        if evidence_query_features.dim() == 3:
            return evidence_query_features.mean(dim=1)
        raise ValueError(
            "evidence_query_features must be (batch, dim) or "
            f"(batch, token_num, dim), got {evidence_query_features.shape}"
        )

    def _normalize_history_evidence_features(
        self,
        history_evidence_features: torch.Tensor,
    ) -> torch.Tensor:
        if history_evidence_features.dim() == 3:
            return history_evidence_features
        if history_evidence_features.dim() == 4:
            return history_evidence_features.mean(dim=2)
        raise ValueError(
            "history_evidence_features must be (batch, history_len, dim) or "
            "(batch, history_len, token_num, dim), got "
            f"{history_evidence_features.shape}"
        )

    def _project_to_latent_space(
        self,
        data_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        evidence_query_features = data_dict.pop("evidence_query_features", None)
        history_evidence_features = data_dict.pop("history_evidence_features", None)
        input_dict = super()._project_to_latent_space(data_dict)

        if evidence_query_features is not None:
            input_dict["evidence_query_features"] = (
                self._normalize_evidence_query_features(evidence_query_features)
            )
        if history_evidence_features is not None:
            input_dict["history_evidence_features"] = (
                self._normalize_history_evidence_features(history_evidence_features)
            )
        return input_dict

    def _project_to_latent_space_multi_traj(
        self,
        data_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        evidence_query_features = data_dict.pop("evidence_query_features", None)
        history_evidence_features = data_dict.pop("history_evidence_features", None)
        traj_num = data_dict["noisy_action"].shape[1]
        batch_size = data_dict["noisy_action"].shape[0]
        device = data_dict["noisy_action"].device

        if "training_traj_indices" in data_dict:
            traj_indices = data_dict["training_traj_indices"].to(
                device=device,
                dtype=torch.long,
            )
        else:
            traj_indices = torch.arange(traj_num, device=device).unsqueeze(0)
            traj_indices = traj_indices.expand(batch_size, -1)

        input_dict = super()._project_to_latent_space_multi_traj(data_dict)

        if evidence_query_features is not None:
            if evidence_query_features.dim() == 4:
                evidence_query_features = evidence_query_features.mean(dim=2)
            if evidence_query_features.dim() != 3:
                raise ValueError(
                    "multi-traj evidence_query_features must be "
                    "(batch, traj_num, dim) or (batch, traj_num, token_num, dim), "
                    f"got {evidence_query_features.shape}"
                )
            batch_idx_2d = torch.arange(batch_size, device=device)[:, None]
            gathered_query = evidence_query_features[batch_idx_2d, traj_indices]
            input_dict["evidence_query_features"] = gathered_query.reshape(
                batch_size * traj_indices.shape[1],
                gathered_query.shape[-1],
            )

        if history_evidence_features is not None:
            if history_evidence_features.dim() == 4:
                history_evidence_features = history_evidence_features.mean(dim=2)
            if history_evidence_features.dim() != 3:
                raise ValueError(
                    "multi-traj history_evidence_features must be "
                    "(batch, traj_num, dim) or (batch, traj_num, token_num, dim), "
                    f"got {history_evidence_features.shape}"
                )
            window_offsets = torch.arange(self.max_history_len, device=device)
            source_indices = (
                traj_indices[:, :, None]
                - self.max_history_len
                + window_offsets[None, None, :]
            )
            valid_mask = source_indices >= 0
            source_indices_clamped = source_indices.clamp(min=0)
            batch_idx_3d = torch.arange(batch_size, device=device)[:, None, None]
            gathered_history = history_evidence_features[
                batch_idx_3d,
                source_indices_clamped,
            ]
            gathered_history = gathered_history * valid_mask[:, :, :, None].to(
                dtype=gathered_history.dtype
            )
            input_dict["history_evidence_features"] = gathered_history.reshape(
                batch_size * traj_indices.shape[1],
                self.max_history_len,
                gathered_history.shape[-1],
            )
        return input_dict

    def initialize_evidence_memory_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.evidence_memory_norms.apply(_basic_init)
        self.evidence_high_attns.apply(_basic_init)
        self.evidence_low_attns.apply(_basic_init)
        self.evidence_memory_outs.apply(_basic_init)
        self.evidence_memory_gates.apply(_basic_init)

        for key in self.evidence_memory_block_indices:
            str_key = str(key)
            out = self.evidence_memory_outs[str_key]
            if self.evidence_memory_out_init_gain > 0:
                nn.init.xavier_uniform_(
                    out.weight,
                    gain=self.evidence_memory_out_init_gain,
                )
            else:
                nn.init.zeros_(out.weight)

            gate_head = self.evidence_memory_gates[str_key][-1]
            assert isinstance(gate_head, nn.Linear)
            nn.init.zeros_(gate_head.weight)
            nn.init.constant_(gate_head.bias, self.evidence_memory_gate_init_bias)

    def _run_blocks(self, input_dict: dict[str, torch.Tensor]):
        if len(self.record_data_entries) > 0:
            record_data_dict: dict[str, list[torch.Tensor]] | None = {
                entry: [] for entry in self.record_data_entries
            }
            if "memory_gate_val" in input_dict and "memory_gate_val" in record_data_dict:
                record_data_dict["memory_gate_val"].append(
                    input_dict["memory_gate_val"].clone()
                )
        else:
            record_data_dict = None

        for block_idx, block in enumerate(self.blocks):
            block_input_dict = {
                key: value
                for key, value in input_dict.items()
                if key
                not in (
                    "evidence_query_features",
                    "history_evidence_features",
                )
            }
            input_dict["x"] = block(
                record_data_dict=record_data_dict,
                **block_input_dict,
            )
            if block_idx in self.evidence_memory_block_indices:
                input_dict["x"] = self._apply_evidence_memory_adapter(
                    str(block_idx),
                    input_dict,
                    record_data_dict,
                )

        if record_data_dict is not None:
            self.recorded_data_dict = record_data_dict

        return input_dict

    def _apply_evidence_memory_adapter(
        self,
        block_key: str,
        input_dict: dict[str, torch.Tensor],
        record_data_dict: dict[str, list[torch.Tensor]] | None,
    ) -> torch.Tensor:
        x = input_dict["x"]
        history_latents = input_dict.get("history_latents")
        if (
            not self.evidence_memory_enabled
            or history_latents is None
            or history_latents.shape[1] == 0
        ):
            return x

        history_mask = input_dict.get("history_mask")
        query = self._evidence_query(
            x,
            input_dict["global_cond"],
            input_dict.get("global_cond_mask"),
        )
        history_evidence_features = None
        if (
            "evidence_query_features" in input_dict
            and "history_evidence_features" in input_dict
        ):
            query = input_dict["evidence_query_features"].to(dtype=x.dtype)
            history_evidence_features = input_dict["history_evidence_features"].to(
                dtype=x.dtype
            )
        high_mask, low_mask, candidate_mask = select_qframe_causal_evidence_masks(
            query,
            history_latents,
            history_mask,
            max_candidates=self.evidence_candidate_max_num,
            high_topk=self.evidence_high_topk,
            low_topk=self.evidence_low_topk,
            history_evidence_features=history_evidence_features,
        )

        evidence_query = self.evidence_memory_norms[block_key](x)
        high_attention = self._read_evidence_branch(
            evidence_query,
            history_latents,
            high_mask,
            self.evidence_high_attns[block_key],
        )
        low_latents = history_latents.mean(dim=2, keepdim=True)
        low_attention = self._read_evidence_branch(
            evidence_query,
            low_latents,
            low_mask,
            self.evidence_low_attns[block_key],
        )
        evidence_attention = high_attention + low_attention

        selected_mask = high_mask | low_mask
        selected_ratio = selected_mask.float().mean(dim=1, keepdim=True).to(
            dtype=x.dtype
        )
        gate_input = torch.cat(
            [
                x.mean(dim=1),
                evidence_attention.mean(dim=1),
                selected_ratio,
            ],
            dim=-1,
        )
        gate = torch.sigmoid(self.evidence_memory_gates[block_key](gate_input))
        delta = self.evidence_memory_outs[block_key](evidence_attention)
        x = x + self.evidence_memory_residual_scale * gate[:, None, :] * delta

        self._record_evidence_diagnostics(
            record_data_dict,
            gate,
            delta,
            candidate_mask,
            high_mask,
            low_mask,
        )
        return x

    def _read_evidence_branch(
        self,
        query: torch.Tensor,
        memory_latents: torch.Tensor,
        memory_mask: torch.Tensor,
        attention: HistoryCrossAttention,
    ) -> torch.Tensor:
        safe_latents = memory_latents
        safe_mask = memory_mask.bool()
        has_valid = safe_mask.any(dim=1, keepdim=True)
        if has_valid.logical_not().any():
            safe_latents = memory_latents.clone()
            safe_mask = safe_mask.clone()
            empty_rows = has_valid.squeeze(1).logical_not()
            safe_latents[empty_rows] = 0
            safe_mask[empty_rows, 0] = True

        read = attention(query, safe_latents, safe_mask, None)
        return read * has_valid.to(dtype=query.dtype)[:, :, None]

    def _evidence_query(
        self,
        x: torch.Tensor,
        global_cond: torch.Tensor,
        global_cond_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.evidence_memory_query_source == "global_cond":
            return self._masked_token_mean(global_cond, global_cond_mask)
        return x.mean(dim=1)

    def _masked_token_mean(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if mask is None:
            return tokens.mean(dim=1)
        mask = mask.bool()
        weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (tokens * weights).sum(dim=1) / denom

    def _record_evidence_diagnostics(
        self,
        record_data_dict: dict[str, list[torch.Tensor]] | None,
        gate: torch.Tensor,
        delta: torch.Tensor,
        candidate_mask: torch.Tensor,
        high_mask: torch.Tensor,
        low_mask: torch.Tensor,
    ):
        if record_data_dict is None:
            return
        if "qframe_evidence_memory_gate" in record_data_dict:
            record_data_dict["qframe_evidence_memory_gate"].append(
                gate.detach().clone()
            )
        if "qframe_evidence_candidate_ratio" in record_data_dict:
            record_data_dict["qframe_evidence_candidate_ratio"].append(
                candidate_mask.float().mean(dim=1, keepdim=True).detach().clone()
            )
        if "qframe_evidence_high_selected_ratio" in record_data_dict:
            record_data_dict["qframe_evidence_high_selected_ratio"].append(
                high_mask.float().mean(dim=1, keepdim=True).detach().clone()
            )
        if "qframe_evidence_low_selected_ratio" in record_data_dict:
            record_data_dict["qframe_evidence_low_selected_ratio"].append(
                low_mask.float().mean(dim=1, keepdim=True).detach().clone()
            )
        if "qframe_evidence_delta_norm" in record_data_dict:
            record_data_dict["qframe_evidence_delta_norm"].append(
                delta.detach().norm(dim=-1).mean(dim=1, keepdim=True).clone()
            )
