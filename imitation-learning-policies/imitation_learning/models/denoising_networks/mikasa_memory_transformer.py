import copy
import einops
import math
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, RmsNorm

from imitation_learning.models.common.modules import Projector
from imitation_learning.models.common.pos_embeddings import \
    get_1d_sincos_pos_embed_from_grid
from imitation_learning.models.denoising_networks.conditional_transformer import (
    ConditionalTransformer, ConditionalTransformerBlock)
from imitation_learning.models.denoising_networks.modules import \
    CrossAttention, HistoryCrossAttention


class BinaryGatingSTEv1(torch.autograd.Function):
    """
    Straight-through estimator for binary gating.
    Backward assumes y = x * gate
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, gate)
        # Return 1.0 if x > 0.5, else 0.0
        return x * (gate > 0.5).float()


    @staticmethod
    def backward(ctx, grad_outputs):
        # Identity gradient: let the MLP think it's continuous
        x, gate = ctx.saved_tensors
        grad_x = grad_outputs * gate
        grad_gate = grad_outputs * x
        return grad_x, grad_gate


class BinaryGatingSTEv2(torch.autograd.Function):
    """
    Straight-through estimator for binary gating.
    Backward uses binary mask to compute gradient of x
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        mask = (gate > 0.5).float()
        ctx.save_for_backward(x, mask)
        # Return 1.0 if x > 0.5, else 0.0
        return x * mask


    @staticmethod
    def backward(ctx, grad_outputs):
        # Identity gradient: let the MLP think it's continuous
        x, mask = ctx.saved_tensors
        grad_x = grad_outputs * mask
        grad_gate = grad_outputs * x
        return grad_x, grad_gate

class BinaryGatingSTEv3(torch.autograd.Function):
    """
    Straight-through estimator for binary gating.
    Backward uses binary mask
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        mask = (gate > 0.5).float()
        ctx.save_for_backward(x, mask)
        return x * mask


    @staticmethod
    def backward(ctx, grad_outputs):
        # Identity gradient: let the MLP think it's continuous
        x, mask = ctx.saved_tensors
        grad_x = grad_outputs * mask
        grad_gate = grad_outputs
        return grad_x, grad_gate


class MemoryTransformerBlock(ConditionalTransformerBlock):
    def __init__(
        self,
        hidden_dim: int,
        head_num: int,
        history_attention_type: str,
        input_token_num: int,
        skip_history_attn: bool,  # For ablation study
        add_additional_self_attn: bool,
        ssmax_scaling_param: float | None,
        binary_gating: bool = True, # To be compatible with previous checkpoints. Should be overridden in the future configs
        straight_through: str = "", # Whether to use straight-through estimator for binary gating
        history_retrieval_topk: int | None = None,
        history_retrieval_query_source: str = "denoising",
        main_history_selector: str = "",
        main_history_first_token_num: int = 0,
        main_history_recent_token_num: int = 0,
        main_history_delta_token_num: int = 0,
        main_history_persistent_anchor_enabled: bool = False,
        cue_event_residual_memory_enabled: bool = False,
        cue_event_memory_first_token_num: int = 4,
        cue_event_memory_recent_token_num: int = 2,
        cue_event_memory_delta_token_num: int = 4,
        cue_event_memory_gate_init_bias: float = -4.0,
        cue_event_memory_out_init_gain: float = 0.0,
        late_cue_adapter_enabled: bool = False,
        late_cue_adapter_first_token_num: int = 1,
        late_cue_adapter_gate_init_bias: float = -6.0,
        late_cue_adapter_out_init_gain: float = 0.0,
        late_cue_adapter_residual_scale: float = 0.05,
        anchor_current_binding_enabled: bool = False,
        anchor_current_binding_gate_init_bias: float = -4.0,
        anchor_current_binding_out_init_gain: float = 0.0,
        anchor_current_binding_residual_scale: float = 0.1,
        history_summary_memory_enabled: bool = False,
        history_summary_memory_token_num: int = 2,
        history_summary_memory_gate_init_bias: float = -4.0,
        history_summary_memory_out_init_gain: float = 0.0,
        history_summary_memory_residual_scale: float = 0.1,
        state_token_memory_enabled: bool = False,
        state_token_memory_gate_init_bias: float = -4.0,
        state_token_memory_out_init_gain: float = 0.0,
        state_token_memory_residual_scale: float = 0.1,
        state_token_read_prefix_num: int = 0,
    ):
        super().__init__(hidden_dim, head_num)

        self.binary_gating: bool = binary_gating

        self.history_cross_attn: HistoryCrossAttention = HistoryCrossAttention(
            dim=hidden_dim,
            head_num=head_num,
            qkv_bias=True,
            qk_norm=True,
            norm_layer=RmsNorm,
            attention_type=history_attention_type,
            ssmax_scaling_param=ssmax_scaling_param,
        )
        self.history_norm1: nn.Module = RmsNorm(hidden_dim, eps=1e-6)
        self.history_norm2: nn.Module = RmsNorm(hidden_dim, eps=1e-6)
        self.add_additional_self_attn:bool = add_additional_self_attn
        if self.add_additional_self_attn:
            self.history_norm3: nn.Module = RmsNorm(hidden_dim, eps=1e-6)
            self.history_self_attn: Attention = Attention(
                dim=hidden_dim,
                num_heads=head_num,
                qkv_bias=True,
                qk_norm=True,
                norm_layer=RmsNorm,
            )
        self.input_token_num: int = input_token_num
        self.straight_through: str = straight_through
        if self.straight_through != "":
            assert self.binary_gating, "Straight-through estimator for binary gating is only supported when binary gating is enabled"

        self.skip_history_attn: bool = skip_history_attn
        self.history_retrieval_topk: int | None = history_retrieval_topk
        if self.history_retrieval_topk is not None:
            assert self.history_retrieval_topk >= 0
        self.history_retrieval_query_source: str = history_retrieval_query_source
        assert self.history_retrieval_query_source in (
            "denoising",
            "global_cond",
        ), f"Unknown history_retrieval_query_source: {self.history_retrieval_query_source}"
        self.main_history_selector: str = main_history_selector
        assert self.main_history_selector in (
            "",
            "stratified",
            "stratified_retrieval_union",
        ), f"Unknown main_history_selector: {self.main_history_selector}"
        self.main_history_first_token_num: int = main_history_first_token_num
        self.main_history_recent_token_num: int = main_history_recent_token_num
        self.main_history_delta_token_num: int = main_history_delta_token_num
        assert self.main_history_first_token_num >= 0
        assert self.main_history_recent_token_num >= 0
        assert self.main_history_delta_token_num >= 0
        self.main_history_persistent_anchor_enabled: bool = (
            main_history_persistent_anchor_enabled
        )
        self.cue_event_residual_memory_enabled: bool = cue_event_residual_memory_enabled
        self.cue_event_memory_first_token_num: int = cue_event_memory_first_token_num
        self.cue_event_memory_recent_token_num: int = cue_event_memory_recent_token_num
        self.cue_event_memory_delta_token_num: int = cue_event_memory_delta_token_num
        self.cue_event_memory_gate_init_bias: float = cue_event_memory_gate_init_bias
        self.cue_event_memory_out_init_gain: float = cue_event_memory_out_init_gain
        if self.cue_event_residual_memory_enabled:
            assert self.cue_event_memory_first_token_num >= 0
            assert self.cue_event_memory_recent_token_num >= 0
            assert self.cue_event_memory_delta_token_num >= 0
            self.history_cue_event_out: nn.Linear = nn.Linear(
                hidden_dim, hidden_dim, bias=False
            )
            self.history_cue_event_gate: nn.Sequential = nn.Sequential(
                nn.Linear(hidden_dim * 2 + 1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )

        self.late_cue_adapter_enabled: bool = late_cue_adapter_enabled
        self.late_cue_adapter_first_token_num: int = late_cue_adapter_first_token_num
        self.late_cue_adapter_gate_init_bias: float = late_cue_adapter_gate_init_bias
        self.late_cue_adapter_out_init_gain: float = late_cue_adapter_out_init_gain
        self.late_cue_adapter_residual_scale: float = late_cue_adapter_residual_scale
        if self.late_cue_adapter_enabled:
            assert self.late_cue_adapter_first_token_num > 0
            assert self.late_cue_adapter_residual_scale >= 0
            self.late_cue_adapter_norm: nn.Module = RmsNorm(hidden_dim, eps=1e-6)
            self.late_cue_adapter_attn: HistoryCrossAttention = HistoryCrossAttention(
                dim=hidden_dim,
                head_num=head_num,
                qkv_bias=True,
                qk_norm=True,
                norm_layer=RmsNorm,
                attention_type=history_attention_type,
                ssmax_scaling_param=ssmax_scaling_param,
            )
            self.late_cue_adapter_out: nn.Linear = nn.Linear(
                hidden_dim, hidden_dim, bias=False
            )
            self.late_cue_adapter_gate: nn.Sequential = nn.Sequential(
                nn.Linear(hidden_dim * 2 + 1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )

        self.anchor_current_binding_enabled: bool = anchor_current_binding_enabled
        self.anchor_current_binding_gate_init_bias: float = (
            anchor_current_binding_gate_init_bias
        )
        self.anchor_current_binding_out_init_gain: float = (
            anchor_current_binding_out_init_gain
        )
        self.anchor_current_binding_residual_scale: float = (
            anchor_current_binding_residual_scale
        )
        if self.anchor_current_binding_enabled:
            assert self.anchor_current_binding_residual_scale >= 0
            self.anchor_current_binding_norm: nn.Module = RmsNorm(
                hidden_dim, eps=1e-6
            )
            self.anchor_current_binding_attn: HistoryCrossAttention = HistoryCrossAttention(
                dim=hidden_dim,
                head_num=head_num,
                qkv_bias=True,
                qk_norm=True,
                norm_layer=RmsNorm,
                attention_type=history_attention_type,
                ssmax_scaling_param=ssmax_scaling_param,
            )
            self.anchor_current_binding_out: nn.Linear = nn.Linear(
                hidden_dim, hidden_dim, bias=False
            )
            self.anchor_current_binding_gate: nn.Sequential = nn.Sequential(
                nn.Linear(hidden_dim * 3 + 1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )

        self.history_summary_memory_enabled: bool = history_summary_memory_enabled
        self.history_summary_memory_token_num: int = history_summary_memory_token_num
        self.history_summary_memory_gate_init_bias: float = (
            history_summary_memory_gate_init_bias
        )
        self.history_summary_memory_out_init_gain: float = (
            history_summary_memory_out_init_gain
        )
        self.history_summary_memory_residual_scale: float = (
            history_summary_memory_residual_scale
        )
        if self.history_summary_memory_enabled:
            assert self.history_summary_memory_token_num > 0
            assert self.history_summary_memory_residual_scale >= 0
            self.history_summary_query_tokens: nn.Parameter = nn.Parameter(
                torch.empty(1, self.history_summary_memory_token_num, hidden_dim)
            )
            self.history_summary_write_norm: nn.Module = RmsNorm(
                hidden_dim, eps=1e-6
            )
            self.history_summary_write_attn: CrossAttention = CrossAttention(
                dim=hidden_dim,
                head_num=head_num,
                qkv_bias=True,
                qk_norm=True,
                norm_layer=RmsNorm,
            )
            self.history_summary_read_norm: nn.Module = RmsNorm(
                hidden_dim, eps=1e-6
            )
            self.history_summary_read_attn: CrossAttention = CrossAttention(
                dim=hidden_dim,
                head_num=head_num,
                qkv_bias=True,
                qk_norm=True,
                norm_layer=RmsNorm,
            )
            self.history_summary_out: nn.Linear = nn.Linear(
                hidden_dim, hidden_dim, bias=False
            )
            self.history_summary_gate: nn.Sequential = nn.Sequential(
                nn.Linear(hidden_dim * 3 + 1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )

        self.state_token_memory_enabled: bool = state_token_memory_enabled
        self.state_token_memory_gate_init_bias: float = (
            state_token_memory_gate_init_bias
        )
        self.state_token_memory_out_init_gain: float = (
            state_token_memory_out_init_gain
        )
        self.state_token_memory_residual_scale: float = (
            state_token_memory_residual_scale
        )
        self.state_token_read_prefix_num: int = state_token_read_prefix_num
        assert self.state_token_read_prefix_num >= 0
        if self.state_token_memory_enabled:
            assert self.state_token_memory_residual_scale >= 0
            self.state_token_read_norm: nn.Module = RmsNorm(hidden_dim, eps=1e-6)
            self.state_token_read_attn: CrossAttention = CrossAttention(
                dim=hidden_dim,
                head_num=head_num,
                qkv_bias=True,
                qk_norm=True,
                norm_layer=RmsNorm,
            )
            self.state_token_out: nn.Linear = nn.Linear(
                hidden_dim, hidden_dim, bias=False
            )
            self.state_token_gate: nn.Sequential = nn.Sequential(
                nn.Linear(hidden_dim * 3 + 1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )

    def reset_cue_event_memory_parameters(self):
        if not self.cue_event_residual_memory_enabled:
            return
        if self.cue_event_memory_out_init_gain > 0:
            nn.init.xavier_uniform_(
                self.history_cue_event_out.weight,
                gain=self.cue_event_memory_out_init_gain,
            )
        else:
            nn.init.zeros_(self.history_cue_event_out.weight)
        gate_head = self.history_cue_event_gate[-1]
        assert isinstance(gate_head, nn.Linear)
        nn.init.zeros_(gate_head.weight)
        nn.init.constant_(gate_head.bias, self.cue_event_memory_gate_init_bias)

    def reset_late_cue_adapter_parameters(self):
        if not self.late_cue_adapter_enabled:
            return
        if self.late_cue_adapter_out_init_gain > 0:
            nn.init.xavier_uniform_(
                self.late_cue_adapter_out.weight,
                gain=self.late_cue_adapter_out_init_gain,
            )
        else:
            nn.init.zeros_(self.late_cue_adapter_out.weight)
        gate_head = self.late_cue_adapter_gate[-1]
        assert isinstance(gate_head, nn.Linear)
        nn.init.zeros_(gate_head.weight)
        nn.init.constant_(gate_head.bias, self.late_cue_adapter_gate_init_bias)

    def reset_anchor_current_binding_parameters(self):
        if not self.anchor_current_binding_enabled:
            return
        if self.anchor_current_binding_out_init_gain > 0:
            nn.init.xavier_uniform_(
                self.anchor_current_binding_out.weight,
                gain=self.anchor_current_binding_out_init_gain,
            )
        else:
            nn.init.zeros_(self.anchor_current_binding_out.weight)
        gate_head = self.anchor_current_binding_gate[-1]
        assert isinstance(gate_head, nn.Linear)
        nn.init.zeros_(gate_head.weight)
        nn.init.constant_(
            gate_head.bias, self.anchor_current_binding_gate_init_bias
        )

    def reset_history_summary_memory_parameters(self):
        if not self.history_summary_memory_enabled:
            return
        nn.init.normal_(self.history_summary_query_tokens, std=0.02)
        if self.history_summary_memory_out_init_gain > 0:
            nn.init.xavier_uniform_(
                self.history_summary_out.weight,
                gain=self.history_summary_memory_out_init_gain,
            )
        else:
            nn.init.zeros_(self.history_summary_out.weight)
        gate_head = self.history_summary_gate[-1]
        assert isinstance(gate_head, nn.Linear)
        nn.init.zeros_(gate_head.weight)
        nn.init.constant_(
            gate_head.bias, self.history_summary_memory_gate_init_bias
        )

    def reset_state_token_memory_parameters(self):
        if not self.state_token_memory_enabled:
            return
        if self.state_token_memory_out_init_gain > 0:
            nn.init.xavier_uniform_(
                self.state_token_out.weight,
                gain=self.state_token_memory_out_init_gain,
            )
        else:
            nn.init.zeros_(self.state_token_out.weight)
        gate_head = self.state_token_gate[-1]
        assert isinstance(gate_head, nn.Linear)
        nn.init.zeros_(gate_head.weight)
        nn.init.constant_(gate_head.bias, self.state_token_memory_gate_init_bias)

    def forward(
        self,
        x: torch.Tensor,
        global_cond: torch.Tensor,
        global_cond_mask: torch.Tensor | None = None,
        history_latents: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
        late_cue_anchor_latents: torch.Tensor | None = None,
        late_cue_anchor_mask: torch.Tensor | None = None,
        late_cue_anchor_action_latents: torch.Tensor | None = None,
        late_cue_anchor_action_mask: torch.Tensor | None = None,
        state_token_latents: torch.Tensor | None = None,
        state_token_mask: torch.Tensor | None = None,
        memory_gate_val: torch.Tensor | None = None,
        step: torch.Tensor | None = None,
        disable_direct_history_paths: bool = False,
        record_data_dict: dict[str, list[torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """
        x: (batch_size, token_num, hidden_dim)
        global_cond: (batch_size, cond_token_num, hidden_dim)
        global_cond_mask: (batch_size, cond_token_num)
        history_latents: (batch_size, history_len, action_token_num, hidden_dim)
        history_mask: (batch_size, history_len)
        late_cue_anchor_latents: (batch_size, anchor_len, token_num, hidden_dim)
        late_cue_anchor_mask: (batch_size, anchor_len)
        state_token_latents: (batch_size, state_token_num, hidden_dim)
        state_token_mask: (batch_size, state_token_num)
        memory_gate_val: (batch_size,)
        step: (batch_size,)
        """
        assert (
            x.shape[1] == self.input_token_num
        ), f"x.shape[1] ({x.shape[1]}) must be equal to input_token_num ({self.input_token_num})"
        attn: torch.Tensor = self.attn(self.norm1(x))
        x = x + attn
        cross_attn: torch.Tensor = self.cross_attn(
            self.norm2(x), global_cond, global_cond_mask
        )
        x = x + cross_attn

        # print(f"{memory_gate_val=}")

        if history_latents is not None and not self.skip_history_attn:
            # TODO: automatically skip history calculation if memory gate val is 0
            main_history_latents = history_latents
            main_history_mask = history_mask
            retrieval_query = self._history_retrieval_query(
                x, global_cond, global_cond_mask
            )
            stratified_mask = None
            retrieval_mask = None
            if self.main_history_selector in ("stratified", "stratified_retrieval_union"):
                stratified_mask = self._select_stratified_history_mask(
                    history_latents, history_mask
                )
                if self.main_history_selector == "stratified_retrieval_union":
                    retrieval_mask = self._select_retrieval_history_mask(
                        retrieval_query, history_latents, history_mask
                    )
                    main_history_mask = stratified_mask | retrieval_mask
                else:
                    main_history_mask = stratified_mask
            elif (
                self.history_retrieval_topk is not None
                and self.history_retrieval_topk > 0
            ):
                main_history_mask = self._select_retrieval_history_mask(
                    retrieval_query, history_latents, history_mask
                )
                retrieval_mask = main_history_mask
            main_history_mask_pre_anchor = main_history_mask
            if (
                self.main_history_persistent_anchor_enabled
                and late_cue_anchor_latents is not None
            ):
                main_history_latents, main_history_mask = (
                    self._prepend_anchor_to_main_history(
                        late_cue_anchor_latents,
                        late_cue_anchor_mask,
                        main_history_latents,
                        main_history_mask,
                    )
                )

            if (
                record_data_dict is not None
                and "stratified_history_selected_ratio" in record_data_dict
                and stratified_mask is not None
            ):
                record_data_dict["stratified_history_selected_ratio"].append(
                    self._mask_valid_ratio(
                        stratified_mask,
                        history_latents.shape[:2],
                        history_latents.device,
                        history_latents.dtype,
                    ).detach().clone()
                )
            if (
                record_data_dict is not None
                and "retrieval_history_selected_ratio" in record_data_dict
            ):
                retrieval_ratio_mask = retrieval_mask
                if retrieval_ratio_mask is None:
                    retrieval_ratio_mask = torch.zeros(
                        history_latents.shape[:2],
                        device=history_latents.device,
                        dtype=torch.bool,
                    )
                record_data_dict["retrieval_history_selected_ratio"].append(
                    self._mask_valid_ratio(
                        retrieval_ratio_mask,
                        history_latents.shape[:2],
                        history_latents.device,
                        history_latents.dtype,
                    ).detach().clone()
                )
            if (
                record_data_dict is not None
                and "main_history_selector_ratio_pre_anchor" in record_data_dict
            ):
                record_data_dict["main_history_selector_ratio_pre_anchor"].append(
                    self._mask_valid_ratio(
                        main_history_mask_pre_anchor,
                        history_latents.shape[:2],
                        history_latents.device,
                        history_latents.dtype,
                    ).detach().clone()
                )
            if (
                record_data_dict is not None
                and "main_history_merged_ratio_post_anchor" in record_data_dict
            ):
                record_data_dict["main_history_merged_ratio_post_anchor"].append(
                    self._mask_valid_ratio(
                        main_history_mask,
                        main_history_latents.shape[:2],
                        main_history_latents.device,
                        main_history_latents.dtype,
                    ).detach().clone()
                )

            history_query = self.history_norm1(x)
            if not disable_direct_history_paths:
                history_attention: torch.Tensor = self.history_cross_attn(
                    history_query,
                    main_history_latents,
                    main_history_mask,
                    record_data_dict,
                )  # (batch_size, token_num, hidden_dim)

                if memory_gate_val is not None:
                    # history_attention = history_attention * memory_gate_val[:, None, None]
                    if self.binary_gating:
                        if self.straight_through == "v1":
                            # Only this version works for the memory gate training
                            gated_history_attention = BinaryGatingSTEv1.apply(history_attention, memory_gate_val[:, None, None])
                        elif self.straight_through == "v2":
                            gated_history_attention = BinaryGatingSTEv2.apply(history_attention, memory_gate_val[:, None, None])
                        elif self.straight_through == "v3":
                            gated_history_attention = BinaryGatingSTEv3.apply(history_attention, memory_gate_val[:, None, None])
                        else:
                            gated_history_attention = history_attention * (memory_gate_val[:, None, None] > 0.5).float()
                        assert isinstance(gated_history_attention, torch.Tensor)
                        assert gated_history_attention.shape == history_attention.shape
                        history_attention = gated_history_attention
                    else:
                        history_attention = history_attention * memory_gate_val[:, None, None]

                    # if record_data_dict is not None and "memory_gate_val" in record_data_dict:
                    #     record_data_dict["memory_gate_val"].append(memory_gate_val.clone()) # Should not be detached for regularization

                x = x + history_attention

            if (
                record_data_dict is not None
                and "main_history_selected_ratio" in record_data_dict
            ):
                # Backward-compatible diagnostic name: post-anchor merged ratio.
                record_data_dict["main_history_selected_ratio"].append(
                    self._mask_valid_ratio(
                        main_history_mask,
                        main_history_latents.shape[:2],
                        main_history_latents.device,
                        main_history_latents.dtype,
                    ).detach().clone()
                )
            if (
                record_data_dict is not None
                and "main_history_anchor_valid_ratio" in record_data_dict
                and self.main_history_persistent_anchor_enabled
                and late_cue_anchor_latents is not None
            ):
                record_data_dict["main_history_anchor_valid_ratio"].append(
                    self._mask_valid_ratio(
                        late_cue_anchor_mask,
                        late_cue_anchor_latents.shape[:2],
                        late_cue_anchor_latents.device,
                        late_cue_anchor_latents.dtype,
                    ).detach().clone()
                )

            if (
                not disable_direct_history_paths
                and self.anchor_current_binding_enabled
                and late_cue_anchor_latents is not None
            ):
                if late_cue_anchor_mask is None:
                    anchor_binding_mask = torch.ones(
                        late_cue_anchor_latents.shape[:2],
                        device=late_cue_anchor_latents.device,
                        dtype=torch.bool,
                    )
                else:
                    anchor_binding_mask = late_cue_anchor_mask.bool()

                anchor_binding_query = self.anchor_current_binding_norm(x)
                anchor_binding_attention: torch.Tensor = (
                    self.anchor_current_binding_attn(
                        anchor_binding_query,
                        late_cue_anchor_latents,
                        anchor_binding_mask,
                        None,
                    )
                )
                valid_ratio = anchor_binding_mask.float().mean(
                    dim=1, keepdim=True
                )
                current_cond_summary = self._masked_token_mean(
                    global_cond, global_cond_mask
                )
                gate_input = torch.cat(
                    [
                        x.mean(dim=1),
                        current_cond_summary,
                        anchor_binding_attention.mean(dim=1),
                        valid_ratio.to(dtype=x.dtype),
                    ],
                    dim=-1,
                )
                anchor_binding_gate = torch.sigmoid(
                    self.anchor_current_binding_gate(gate_input)
                )
                anchor_binding_residual = self.anchor_current_binding_out(
                    anchor_binding_attention
                )
                x = (
                    x
                    + self.anchor_current_binding_residual_scale
                    * anchor_binding_gate[:, None, :]
                    * anchor_binding_residual
                )

                if (
                    record_data_dict is not None
                    and "anchor_current_binding_gate" in record_data_dict
                ):
                    record_data_dict["anchor_current_binding_gate"].append(
                        anchor_binding_gate.detach().clone()
                    )
                if (
                    record_data_dict is not None
                    and "anchor_current_binding_selected_ratio" in record_data_dict
                ):
                    record_data_dict[
                        "anchor_current_binding_selected_ratio"
                    ].append(valid_ratio.detach().clone())

            if self.history_summary_memory_enabled:
                (
                    history_summary_attention,
                    history_summary_gate,
                    history_summary_valid_ratio,
                    history_summary_token_norm,
                ) = self._read_history_summary_memory(
                    x,
                    main_history_latents,
                    main_history_mask,
                )
                history_summary_residual = self.history_summary_out(
                    history_summary_attention
                )
                x = (
                    x
                    + self.history_summary_memory_residual_scale
                    * history_summary_gate[:, None, :]
                    * history_summary_residual
                )

                if (
                    record_data_dict is not None
                    and "history_summary_memory_gate" in record_data_dict
                ):
                    record_data_dict["history_summary_memory_gate"].append(
                        history_summary_gate.detach().clone()
                    )
                if (
                    record_data_dict is not None
                    and "history_summary_memory_valid_ratio" in record_data_dict
                ):
                    record_data_dict["history_summary_memory_valid_ratio"].append(
                        history_summary_valid_ratio.detach().clone()
                    )
                if (
                    record_data_dict is not None
                    and "history_summary_memory_token_norm" in record_data_dict
                ):
                    record_data_dict["history_summary_memory_token_norm"].append(
                        history_summary_token_norm.detach().clone()
                    )
                if (
                    record_data_dict is not None
                    and "history_summary_memory_read_norm" in record_data_dict
                ):
                    record_data_dict["history_summary_memory_read_norm"].append(
                        history_summary_attention.detach().norm(dim=-1).mean(
                            dim=1, keepdim=True
                        ).clone()
                    )

            if (
                not disable_direct_history_paths
                and self.cue_event_residual_memory_enabled
            ):
                cue_event_mask = self._select_cue_event_history_mask(
                    history_latents, history_mask
                )
                cue_event_attention: torch.Tensor = self.history_cross_attn(
                    history_query,
                    history_latents,
                    cue_event_mask,
                    record_data_dict,
                )
                if cue_event_mask is None:
                    valid_ratio = torch.ones(
                        x.shape[0], 1, device=x.device, dtype=x.dtype
                    )
                else:
                    valid_ratio = cue_event_mask.float().mean(dim=1, keepdim=True)
                gate_input = torch.cat(
                    [
                        x.mean(dim=1),
                        cue_event_attention.mean(dim=1),
                        valid_ratio.to(dtype=x.dtype),
                    ],
                    dim=-1,
                )
                cue_event_gate = torch.sigmoid(self.history_cue_event_gate(gate_input))
                cue_event_residual = self.history_cue_event_out(cue_event_attention)
                x = x + cue_event_gate[:, None, :] * cue_event_residual

                if (
                    record_data_dict is not None
                    and "cue_event_memory_gate" in record_data_dict
                ):
                    record_data_dict["cue_event_memory_gate"].append(
                        cue_event_gate.detach().clone()
                    )
                if (
                    record_data_dict is not None
                    and "cue_event_memory_selected_ratio" in record_data_dict
                ):
                    record_data_dict["cue_event_memory_selected_ratio"].append(
                        valid_ratio.detach().clone()
                    )

            if (
                not disable_direct_history_paths
                and self.late_cue_adapter_enabled
            ):
                if late_cue_anchor_latents is not None:
                    fallback_mask = self._select_late_cue_history_mask(
                        history_latents, history_mask
                    )
                    if fallback_mask is None:
                        fallback_mask = torch.ones(
                            history_latents.shape[:2],
                            device=history_latents.device,
                            dtype=torch.bool,
                        )
                    else:
                        fallback_mask = fallback_mask.bool()

                    if late_cue_anchor_mask is None:
                        late_cue_anchor_mask = torch.ones(
                            late_cue_anchor_latents.shape[:2],
                            device=late_cue_anchor_latents.device,
                            dtype=torch.bool,
                        )
                    else:
                        late_cue_anchor_mask = late_cue_anchor_mask.bool()

                    anchor_valid = late_cue_anchor_mask.any(dim=1, keepdim=True)
                    late_cue_latents = torch.cat(
                        [late_cue_anchor_latents, history_latents], dim=1
                    )
                    late_cue_mask = torch.cat(
                        [
                            late_cue_anchor_mask,
                            fallback_mask & anchor_valid.logical_not(),
                        ],
                        dim=1,
                    )
                else:
                    late_cue_latents = history_latents
                    late_cue_mask = self._select_late_cue_history_mask(
                        history_latents, history_mask
                    )
                late_cue_query = self.late_cue_adapter_norm(x)
                late_cue_attention: torch.Tensor = self.late_cue_adapter_attn(
                    late_cue_query,
                    late_cue_latents,
                    late_cue_mask,
                    None,
                )
                if late_cue_mask is None:
                    valid_ratio = torch.ones(
                        x.shape[0], 1, device=x.device, dtype=x.dtype
                    )
                else:
                    valid_ratio = late_cue_mask.float().mean(dim=1, keepdim=True)
                gate_input = torch.cat(
                    [
                        x.mean(dim=1),
                        late_cue_attention.mean(dim=1),
                        valid_ratio.to(dtype=x.dtype),
                    ],
                    dim=-1,
                )
                late_cue_gate = torch.sigmoid(self.late_cue_adapter_gate(gate_input))
                late_cue_residual = self.late_cue_adapter_out(late_cue_attention)
                x = (
                    x
                    + self.late_cue_adapter_residual_scale
                    * late_cue_gate[:, None, :]
                    * late_cue_residual
                )

                if (
                    record_data_dict is not None
                    and "late_cue_adapter_gate" in record_data_dict
                ):
                    record_data_dict["late_cue_adapter_gate"].append(
                        late_cue_gate.detach().clone()
                    )
                if (
                    record_data_dict is not None
                    and "late_cue_adapter_selected_ratio" in record_data_dict
                ):
                    record_data_dict["late_cue_adapter_selected_ratio"].append(
                        valid_ratio.detach().clone()
                    )

            # if memory_gate_val is None:
            #     history_attention: torch.Tensor = self.history_cross_attn(
            #         self.history_norm1(x),
            #         history_latents,
            #         history_mask,
            #         record_data_dict,
            #     )  # (batch_size, token_num, hidden_dim)
            #     x = x + history_attention

            # else:
            #     # history_attention = history_attention * memory_gate_val[:, None, None]
            #     if self.binary_gating:

            #         # In training, calculate history attention normally for gate training
            #         if torch.torch.is_grad_enabled() and self.straight_through != "":
            #             history_attention: torch.Tensor = self.history_cross_attn(
            #                 self.history_norm1(x),
            #                 history_latents,
            #                 history_mask,
            #                 record_data_dict,
            #             )  # (batch_size, token_num, hidden_dim)
            #             if self.straight_through == "v1":
            #                 # Only this version works for the memory gate training
            #                 gated_history_attention = BinaryGatingSTEv1.apply(history_attention, memory_gate_val[:, None, None])
            #             elif self.straight_through == "v2":
            #                 gated_history_attention = BinaryGatingSTEv2.apply(history_attention, memory_gate_val[:, None, None])
            #             elif self.straight_through == "v3":
            #                 gated_history_attention = BinaryGatingSTEv3.apply(history_attention, memory_gate_val[:, None, None])
            #             else:
            #                 gated_history_attention = history_attention * (memory_gate_val[:, None, None] > 0.5).float()
            #             assert isinstance(gated_history_attention, torch.Tensor)
            #             assert gated_history_attention.shape == history_attention.shape
            #             history_attention = gated_history_attention
            #             x = x + history_attention
            #         else:
            #             # In inference mode, skip history attention calculation if not necessary
            #             binarized_memory_gate_val = (memory_gate_val > 0.5).bool()
            #             # if sum(binarized_memory_gate_val) > 0:
            #             #     gated_history_attention: torch.Tensor = self.history_cross_attn(
            #             #         self.history_norm1(x[binarized_memory_gate_val]),
            #             #         history_latents[binarized_memory_gate_val],
            #             #         history_mask[binarized_memory_gate_val] if history_mask is not None else None,
            #             #         record_data_dict,
            #             #     )
            #             #     x = x + gated_history_attention
            #             # else:
            #             #     # all memory gate values are 0, skip history attention calculation
            #             #     pass

            #     else:
            #         history_attention: torch.Tensor = self.history_cross_attn(
            #             self.history_norm1(x),
            #             history_latents,
            #             history_mask,
            #             record_data_dict,
            #         )  # (batch_size, token_num, hidden_dim)

            #         history_attention = history_attention * memory_gate_val[:, None, None]
            #         x = x + history_attention

            #     if record_data_dict is not None and "memory_gate_val" in record_data_dict:
            #         record_data_dict["memory_gate_val"].append(memory_gate_val.clone()) # Should not be detached for regularization

        if self.state_token_memory_enabled and state_token_latents is not None:
            state_attention, state_gate, state_valid_ratio = (
                self._read_state_token_memory(
                    x,
                    state_token_latents,
                    state_token_mask,
                )
            )
            state_residual = self.state_token_out(state_attention)
            x = (
                x
                + self.state_token_memory_residual_scale
                * state_gate[:, None, :]
                * state_residual
            )

            if (
                record_data_dict is not None
                and "state_token_memory_gate" in record_data_dict
            ):
                record_data_dict["state_token_memory_gate"].append(
                    state_gate.detach().clone()
                )
            if (
                record_data_dict is not None
                and "state_token_memory_valid_ratio" in record_data_dict
            ):
                record_data_dict["state_token_memory_valid_ratio"].append(
                    state_valid_ratio.detach().clone()
                )
            if (
                record_data_dict is not None
                and "state_token_memory_token_norm" in record_data_dict
            ):
                record_data_dict["state_token_memory_token_norm"].append(
                    state_token_latents.detach().norm(dim=-1).mean(
                        dim=1, keepdim=True
                    ).clone()
                )
            if (
                record_data_dict is not None
                and "state_token_memory_read_norm" in record_data_dict
            ):
                record_data_dict["state_token_memory_read_norm"].append(
                    state_attention.detach().norm(dim=-1).mean(
                        dim=1, keepdim=True
                    ).clone()
                )

        if self.add_additional_self_attn:
            x = x + self.history_self_attn(self.history_norm3(x))

        x = x + self.ff(self.norm3(x))

        return x

    def _read_state_token_memory(
        self,
        x: torch.Tensor,
        state_token_latents: torch.Tensor,
        state_token_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state_token_mask is None:
            state_token_mask = torch.ones(
                state_token_latents.shape[:2],
                device=state_token_latents.device,
                dtype=torch.bool,
            )
        else:
            state_token_mask = state_token_mask.bool()
        if self.state_token_read_prefix_num > 0:
            prefix = min(self.state_token_read_prefix_num, state_token_mask.shape[1])
            prefix_mask = torch.zeros_like(state_token_mask)
            prefix_mask[:, :prefix] = state_token_mask[:, :prefix]
            state_token_mask = prefix_mask

        safe_state_tokens = state_token_latents
        safe_mask = state_token_mask
        has_valid = state_token_mask.any(dim=1, keepdim=True)
        if has_valid.logical_not().any():
            safe_state_tokens = state_token_latents.clone()
            safe_mask = state_token_mask.clone()
            empty_rows = has_valid.squeeze(1).logical_not()
            safe_state_tokens[empty_rows] = 0
            safe_mask[empty_rows, 0] = True

        valid_ratio = state_token_mask.float().mean(dim=1, keepdim=True).to(
            dtype=x.dtype
        )
        valid_scale = has_valid.to(dtype=x.dtype)[:, :, None]
        state_attention = self.state_token_read_attn(
            self.state_token_read_norm(x),
            safe_state_tokens,
            safe_mask,
        )
        state_attention = state_attention * valid_scale
        state_weights = state_token_mask.to(dtype=x.dtype).unsqueeze(-1)
        state_count = state_weights.sum(dim=1).clamp_min(1.0)
        state_summary = (safe_state_tokens * state_weights).sum(dim=1) / state_count
        state_summary = state_summary * has_valid.to(dtype=x.dtype)
        gate_input = torch.cat(
            [
                x.mean(dim=1),
                state_summary,
                state_attention.mean(dim=1),
                valid_ratio,
            ],
            dim=-1,
        )
        state_gate = torch.sigmoid(self.state_token_gate(gate_input))
        return state_attention, state_gate, valid_ratio

    def _mask_valid_ratio(
        self,
        mask: torch.Tensor | None,
        shape: torch.Size | tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if mask is None:
            return torch.ones(shape[0], 1, device=device, dtype=dtype)
        return mask.float().mean(dim=1, keepdim=True).to(dtype=dtype)

    def _prepend_anchor_to_main_history(
        self,
        anchor_latents: torch.Tensor,
        anchor_mask: torch.Tensor | None,
        history_latents: torch.Tensor,
        history_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if anchor_mask is None:
            anchor_mask = torch.ones(
                anchor_latents.shape[:2],
                device=anchor_latents.device,
                dtype=torch.bool,
            )
        else:
            anchor_mask = anchor_mask.bool()
        if history_mask is None:
            history_mask = torch.ones(
                history_latents.shape[:2],
                device=history_latents.device,
                dtype=torch.bool,
            )
        else:
            history_mask = history_mask.bool()

        merged_latents = torch.cat([anchor_latents, history_latents], dim=1)
        merged_mask = torch.cat([anchor_mask, history_mask], dim=1)
        return merged_latents, merged_mask

    def _read_history_summary_memory(
        self,
        x: torch.Tensor,
        memory_latents: torch.Tensor,
        memory_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, memory_len, token_num, hidden_dim = memory_latents.shape
        flat_memory = memory_latents.reshape(batch_size, memory_len * token_num, hidden_dim)
        if memory_mask is None:
            flat_mask = torch.ones(
                batch_size,
                memory_len * token_num,
                device=memory_latents.device,
                dtype=torch.bool,
            )
            valid_ratio = torch.ones(batch_size, 1, device=x.device, dtype=x.dtype)
            has_valid = torch.ones(batch_size, 1, device=x.device, dtype=torch.bool)
        else:
            memory_mask = memory_mask.bool()
            flat_mask = (
                memory_mask[:, :, None]
                .expand(batch_size, memory_len, token_num)
                .reshape(batch_size, memory_len * token_num)
            )
            valid_ratio = memory_mask.float().mean(dim=1, keepdim=True).to(dtype=x.dtype)
            has_valid = memory_mask.any(dim=1, keepdim=True)

        if flat_memory.shape[1] == 0:
            flat_memory = x.new_zeros(batch_size, 1, hidden_dim)
            flat_mask = torch.ones(batch_size, 1, device=x.device, dtype=torch.bool)
            has_valid = torch.zeros(batch_size, 1, device=x.device, dtype=torch.bool)
            valid_ratio = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
        else:
            empty_rows = flat_mask.logical_not().all(dim=1)
            if empty_rows.any():
                flat_memory = flat_memory.clone()
                flat_mask = flat_mask.clone()
                flat_memory[empty_rows] = 0
                flat_mask[empty_rows, 0] = True

        summary_queries = self.history_summary_query_tokens.expand(batch_size, -1, -1)
        summary_tokens = self.history_summary_write_attn(
            self.history_summary_write_norm(summary_queries),
            flat_memory,
            flat_mask,
        )
        valid_scale = has_valid.to(dtype=x.dtype)[:, :, None]
        summary_tokens = summary_tokens * valid_scale

        summary_attention = self.history_summary_read_attn(
            self.history_summary_read_norm(x),
            summary_tokens,
            None,
        )
        summary_attention = summary_attention * valid_scale
        gate_input = torch.cat(
            [
                x.mean(dim=1),
                summary_tokens.mean(dim=1),
                summary_attention.mean(dim=1),
                valid_ratio,
            ],
            dim=-1,
        )
        summary_gate = torch.sigmoid(self.history_summary_gate(gate_input))
        summary_token_norm = summary_tokens.norm(dim=-1).mean(dim=1, keepdim=True)
        return summary_attention, summary_gate, valid_ratio, summary_token_norm

    def _select_stratified_history_mask(
        self,
        history_latents: torch.Tensor,
        history_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, history_len = history_latents.shape[:2]
        if history_mask is None:
            valid_mask = torch.ones(
                batch_size, history_len, device=history_latents.device, dtype=torch.bool
            )
        else:
            valid_mask = history_mask.bool()

        selected = torch.zeros_like(valid_mask)

        if self.main_history_first_token_num > 0:
            first_rank = valid_mask.long().cumsum(dim=1)
            selected = selected | (
                valid_mask & (first_rank <= self.main_history_first_token_num)
            )

        if self.main_history_recent_token_num > 0:
            recent_rank = torch.flip(
                torch.flip(valid_mask.long(), dims=[1]).cumsum(dim=1), dims=[1]
            )
            selected = selected | (
                valid_mask & (recent_rank <= self.main_history_recent_token_num)
            )

        delta_topk = min(self.main_history_delta_token_num, history_len)
        if delta_topk > 0:
            pooled_history = torch.nn.functional.normalize(
                history_latents.mean(dim=2), dim=-1
            )
            delta = torch.zeros(
                batch_size,
                history_len,
                device=history_latents.device,
                dtype=history_latents.dtype,
            )
            if history_len > 1:
                delta[:, 1:] = (pooled_history[:, 1:] - pooled_history[:, :-1]).norm(
                    dim=-1
                )
            delta = delta.masked_fill(valid_mask.logical_not(), -float("inf"))
            topk_indices = torch.topk(delta, k=delta_topk, dim=1).indices
            delta_selected = torch.zeros_like(valid_mask)
            delta_selected.scatter_(1, topk_indices, True)
            selected = selected | (delta_selected & valid_mask)

        selected = selected & valid_mask
        empty_rows = valid_mask.any(dim=1) & selected.logical_not().all(dim=1)
        if empty_rows.any():
            recent_rank = torch.flip(
                torch.flip(valid_mask.long(), dims=[1]).cumsum(dim=1), dims=[1]
            )
            selected = selected | (empty_rows[:, None] & valid_mask & (recent_rank == 1))
        return selected

    def _history_retrieval_query(
        self,
        x: torch.Tensor,
        global_cond: torch.Tensor,
        global_cond_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.history_retrieval_query_source == "global_cond":
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

    def _select_cue_event_history_mask(
        self,
        history_latents: torch.Tensor,
        history_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        batch_size, history_len = history_latents.shape[:2]
        if history_len == 0:
            return history_mask

        if history_mask is None:
            valid_mask = torch.ones(
                batch_size, history_len, device=history_latents.device, dtype=torch.bool
            )
        else:
            valid_mask = history_mask.bool()

        selected = torch.zeros_like(valid_mask)

        if self.cue_event_memory_first_token_num > 0:
            first_rank = valid_mask.long().cumsum(dim=1)
            selected = selected | (
                valid_mask & (first_rank <= self.cue_event_memory_first_token_num)
            )

        if self.cue_event_memory_recent_token_num > 0:
            recent_rank = torch.flip(
                torch.flip(valid_mask.long(), dims=[1]).cumsum(dim=1), dims=[1]
            )
            selected = selected | (
                valid_mask & (recent_rank <= self.cue_event_memory_recent_token_num)
            )

        delta_topk = min(self.cue_event_memory_delta_token_num, history_len)
        if delta_topk > 0:
            pooled_history = torch.nn.functional.normalize(
                history_latents.mean(dim=2), dim=-1
            )
            delta = torch.zeros(
                batch_size,
                history_len,
                device=history_latents.device,
                dtype=history_latents.dtype,
            )
            if history_len > 1:
                delta[:, 1:] = (pooled_history[:, 1:] - pooled_history[:, :-1]).norm(
                    dim=-1
                )
            delta = delta.masked_fill(valid_mask.logical_not(), -float("inf"))
            topk_indices = torch.topk(delta, k=delta_topk, dim=1).indices
            delta_selected = torch.zeros_like(valid_mask)
            delta_selected.scatter_(1, topk_indices, True)
            selected = selected | (delta_selected & valid_mask)

        selected = selected & valid_mask
        return selected

    def _select_late_cue_history_mask(
        self,
        history_latents: torch.Tensor,
        history_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        batch_size, history_len = history_latents.shape[:2]
        if history_len == 0:
            return history_mask

        if history_mask is None:
            valid_mask = torch.ones(
                batch_size, history_len, device=history_latents.device, dtype=torch.bool
            )
        else:
            valid_mask = history_mask.bool()

        first_rank = valid_mask.long().cumsum(dim=1)
        selected = valid_mask & (first_rank <= self.late_cue_adapter_first_token_num)
        return selected

    def _select_retrieval_history_mask(
        self,
        query_source: torch.Tensor,
        history_latents: torch.Tensor,
        history_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Keep only the top-k history chunks most similar to the retrieval query.
        This is a parameter-free retrieval gate over history chunks; setting top-k to
        None or 0 in config restores dense GMP history attention.
        """
        batch_size, history_len = history_latents.shape[:2]
        if history_mask is None:
            valid_mask = torch.ones(
                (batch_size, history_len), device=query_source.device, dtype=torch.bool
            )
        else:
            valid_mask = history_mask.bool()

        topk = min(int(self.history_retrieval_topk or 0), history_len)
        if topk <= 0 or topk >= history_len:
            return valid_mask

        query = torch.nn.functional.normalize(query_source, dim=-1)
        keys = torch.nn.functional.normalize(history_latents.mean(dim=2), dim=-1)
        scores = torch.einsum("bd,bhd->bh", query, keys) / math.sqrt(
            query_source.shape[-1]
        )

        scores = scores.masked_fill(valid_mask.logical_not(), -float("inf"))

        selected = torch.zeros(
            (batch_size, history_len), device=query_source.device, dtype=torch.bool
        )
        topk_indices = torch.topk(scores, k=topk, dim=1).indices
        selected.scatter_(1, topk_indices, True)
        return selected & valid_mask


class MikasaMemoryTransformer(ConditionalTransformer):
    def __init__(
        self,
        max_history_len: int,
        freeze_non_history_modules: bool,
        history_attention_type: str,
        record_data_entries: list[str],  # For debugging
        ssmax_scaling_param: float | None,
        include_action_history: bool,
        history_action_num_per_chunk: int,
        skip_history_attn: bool,  # For ablation study
        add_memory_gate_token: bool = False, # For compatibility with previous checkpoints
        binary_gating: bool = True, # For compatibility with previous checkpoints
        straight_through: str = "", # "v1", "v2", "v3", or ""
        add_additional_self_attn: bool = True,
        history_img_features_dim: int = 0,
        history_img_features_token_num: int = 0,
        history_retrieval_topk: int | None = None,
        history_retrieval_query_source: str = "denoising",
        main_history_selector: str = "",
        main_history_first_token_num: int = 0,
        main_history_recent_token_num: int = 0,
        main_history_delta_token_num: int = 0,
        main_history_persistent_anchor_enabled: bool = False,
        cue_event_residual_memory_enabled: bool = False,
        cue_event_memory_first_token_num: int = 4,
        cue_event_memory_recent_token_num: int = 2,
        cue_event_memory_delta_token_num: int = 4,
        cue_event_memory_gate_init_bias: float = -4.0,
        cue_event_memory_out_init_gain: float = 0.0,
        late_cue_adapter_enabled: bool = False,
        late_cue_adapter_num_last_blocks: int = 2,
        late_cue_adapter_first_token_num: int = 1,
        late_cue_adapter_gate_init_bias: float = -6.0,
        late_cue_adapter_out_init_gain: float = 0.0,
        late_cue_adapter_residual_scale: float = 0.05,
        late_cue_anchor_enabled: bool = False,
        late_cue_anchor_len: int = 1,
        late_cue_anchor_causal_mask: bool = False,
        anchor_current_binding_enabled: bool = False,
        anchor_current_binding_num_last_blocks: int = 2,
        anchor_current_binding_gate_init_bias: float = -4.0,
        anchor_current_binding_out_init_gain: float = 0.0,
        anchor_current_binding_residual_scale: float = 0.1,
        history_summary_memory_enabled: bool = False,
        history_summary_memory_num_last_blocks: int = 2,
        history_summary_memory_token_num: int = 2,
        history_summary_memory_gate_init_bias: float = -4.0,
        history_summary_memory_out_init_gain: float = 0.0,
        history_summary_memory_residual_scale: float = 0.1,
        state_token_memory_enabled: bool = False,
        state_token_memory_num_last_blocks: int = 2,
        state_token_num: int = 2,
        state_token_initial_std: float = 0.02,
        state_token_update_gate_init_bias: float = -2.0,
        state_token_update_out_init_gain: float = 0.001,
        state_token_update_residual_scale: float = 1.0,
        state_token_update_include_action_history: bool = True,
        state_token_protected_prefix_num: int = 0,
        state_token_memory_gate_init_bias: float = -4.0,
        state_token_memory_out_init_gain: float = 0.0,
        state_token_memory_residual_scale: float = 0.1,
        state_token_read_prefix_num: int = 0,
        state_token_action_adapter_enabled: bool = False,
        state_token_action_adapter_gate_init_bias: float = -4.0,
        state_token_action_adapter_out_init_gain: float = 0.0,
        state_token_action_adapter_residual_scale: float = 0.05,
        state_token_action_prehead_adapter_enabled: bool = False,
        state_token_action_prehead_adapter_gate_init_bias: float = -4.0,
        state_token_action_prehead_adapter_out_init_gain: float = 0.0,
        state_token_action_prehead_adapter_residual_scale: float = 0.05,
        late_cue_action_prehead_adapter_enabled: bool = False,
        late_cue_action_prehead_adapter_gate_init_bias: float = -4.0,
        late_cue_action_prehead_adapter_out_init_gain: float = 0.0,
        late_cue_action_prehead_adapter_residual_scale: float = 0.05,
        visual_memory_carrier_type: str = "",
        visual_memory_carrier_token_num: int = 1,
        visual_memory_carrier_max_len: int = 64,
        visual_memory_carrier_hidden_dim: int | None = None,
        visual_memory_carrier_num_layers: int = 1,
        visual_memory_carrier_num_heads: int = 8,
        visual_memory_carrier_dropout: float = 0.0,
        visual_memory_carrier_force_zero: bool = False,
        state_token_pre_action_obs_update_in_training_enabled: bool = False,
        **kwargs,
    ):
        kwargs["name"] = "mikasa_memory_transformer"

        super().__init__(**kwargs)

        self.binary_gating: bool = binary_gating
        self.skip_history_attn: bool = skip_history_attn
        assert late_cue_adapter_num_last_blocks >= 0
        late_cue_adapter_start_idx = max(
            self.layer_num - late_cue_adapter_num_last_blocks, 0
        )
        assert anchor_current_binding_num_last_blocks >= 0
        anchor_current_binding_start_idx = max(
            self.layer_num - anchor_current_binding_num_last_blocks, 0
        )
        assert history_summary_memory_num_last_blocks >= 0
        history_summary_memory_start_idx = max(
            self.layer_num - history_summary_memory_num_last_blocks, 0
        )
        assert state_token_memory_num_last_blocks >= 0
        state_token_memory_start_idx = max(
            self.layer_num - state_token_memory_num_last_blocks, 0
        )
        assert state_token_read_prefix_num >= 0

        self.blocks: nn.ModuleList = nn.ModuleList(
            [
                MemoryTransformerBlock(
                    hidden_dim=self.hidden_dim,
                    head_num=self.head_num,
                    history_attention_type=history_attention_type,
                    input_token_num=self.input_pos_embedding.shape[1] + (1 if add_memory_gate_token else 0),
                    skip_history_attn=skip_history_attn,
                    add_additional_self_attn=add_additional_self_attn,
                    ssmax_scaling_param=ssmax_scaling_param,
                    binary_gating=binary_gating,
                    straight_through=straight_through,
                    history_retrieval_topk=history_retrieval_topk,
                    history_retrieval_query_source=history_retrieval_query_source,
                    main_history_selector=main_history_selector,
                    main_history_first_token_num=main_history_first_token_num,
                    main_history_recent_token_num=main_history_recent_token_num,
                    main_history_delta_token_num=main_history_delta_token_num,
                    main_history_persistent_anchor_enabled=(
                        main_history_persistent_anchor_enabled
                    ),
                    cue_event_residual_memory_enabled=cue_event_residual_memory_enabled,
                    cue_event_memory_first_token_num=cue_event_memory_first_token_num,
                    cue_event_memory_recent_token_num=cue_event_memory_recent_token_num,
                    cue_event_memory_delta_token_num=cue_event_memory_delta_token_num,
                    cue_event_memory_gate_init_bias=cue_event_memory_gate_init_bias,
                    cue_event_memory_out_init_gain=cue_event_memory_out_init_gain,
                    late_cue_adapter_enabled=(
                        late_cue_adapter_enabled and i >= late_cue_adapter_start_idx
                    ),
                    late_cue_adapter_first_token_num=late_cue_adapter_first_token_num,
                    late_cue_adapter_gate_init_bias=late_cue_adapter_gate_init_bias,
                    late_cue_adapter_out_init_gain=late_cue_adapter_out_init_gain,
                    late_cue_adapter_residual_scale=late_cue_adapter_residual_scale,
                    anchor_current_binding_enabled=(
                        anchor_current_binding_enabled
                        and i >= anchor_current_binding_start_idx
                    ),
                    anchor_current_binding_gate_init_bias=(
                        anchor_current_binding_gate_init_bias
                    ),
                    anchor_current_binding_out_init_gain=(
                        anchor_current_binding_out_init_gain
                    ),
                    anchor_current_binding_residual_scale=(
                        anchor_current_binding_residual_scale
                    ),
                    history_summary_memory_enabled=(
                        history_summary_memory_enabled
                        and i >= history_summary_memory_start_idx
                    ),
                    history_summary_memory_token_num=history_summary_memory_token_num,
                    history_summary_memory_gate_init_bias=(
                        history_summary_memory_gate_init_bias
                    ),
                    history_summary_memory_out_init_gain=(
                        history_summary_memory_out_init_gain
                    ),
                    history_summary_memory_residual_scale=(
                        history_summary_memory_residual_scale
                    ),
                    state_token_memory_enabled=(
                        state_token_memory_enabled
                        and i >= state_token_memory_start_idx
                    ),
                    state_token_memory_gate_init_bias=(
                        state_token_memory_gate_init_bias
                    ),
                    state_token_memory_out_init_gain=(
                        state_token_memory_out_init_gain
                    ),
                    state_token_memory_residual_scale=(
                        state_token_memory_residual_scale
                    ),
                    state_token_read_prefix_num=state_token_read_prefix_num,
                )
                for i in range(self.layer_num)
            ]
        )

        self.max_history_len: int = max_history_len
        self.state_token_memory_enabled: bool = state_token_memory_enabled
        self.state_token_num: int = state_token_num
        self.state_token_initial_std: float = state_token_initial_std
        self.state_token_update_gate_init_bias: float = state_token_update_gate_init_bias
        self.state_token_update_out_init_gain: float = state_token_update_out_init_gain
        self.state_token_update_residual_scale: float = state_token_update_residual_scale
        self.state_token_update_include_action_history: bool = (
            state_token_update_include_action_history
        )
        self.state_token_protected_prefix_num: int = state_token_protected_prefix_num
        self.state_token_read_prefix_num: int = state_token_read_prefix_num
        self.state_token_action_adapter_enabled: bool = (
            state_token_action_adapter_enabled
        )
        self.state_token_action_adapter_gate_init_bias: float = (
            state_token_action_adapter_gate_init_bias
        )
        self.state_token_action_adapter_out_init_gain: float = (
            state_token_action_adapter_out_init_gain
        )
        self.state_token_action_adapter_residual_scale: float = (
            state_token_action_adapter_residual_scale
        )
        self.state_token_action_prehead_adapter_enabled: bool = (
            state_token_action_prehead_adapter_enabled
        )
        self.state_token_action_prehead_adapter_gate_init_bias: float = (
            state_token_action_prehead_adapter_gate_init_bias
        )
        self.state_token_action_prehead_adapter_out_init_gain: float = (
            state_token_action_prehead_adapter_out_init_gain
        )
        self.state_token_action_prehead_adapter_residual_scale: float = (
            state_token_action_prehead_adapter_residual_scale
        )
        self.late_cue_action_prehead_adapter_enabled: bool = (
            late_cue_action_prehead_adapter_enabled
        )
        self.late_cue_action_prehead_adapter_gate_init_bias: float = (
            late_cue_action_prehead_adapter_gate_init_bias
        )
        self.late_cue_action_prehead_adapter_out_init_gain: float = (
            late_cue_action_prehead_adapter_out_init_gain
        )
        self.late_cue_action_prehead_adapter_residual_scale: float = (
            late_cue_action_prehead_adapter_residual_scale
        )
        self.state_token_pre_action_obs_update_in_training_enabled: bool = (
            state_token_pre_action_obs_update_in_training_enabled
        )
        if self.state_token_memory_enabled:
            assert self.state_token_num > 0
            assert self.state_token_initial_std >= 0
            assert self.state_token_update_residual_scale >= 0
            assert 0 <= self.state_token_protected_prefix_num <= self.state_token_num
            assert self.state_token_read_prefix_num <= self.state_token_num
            self.state_token_initial: nn.Parameter = nn.Parameter(
                torch.empty(1, self.state_token_num, self.hidden_dim)
            )
            self.state_token_update_norm: nn.Module = RmsNorm(
                self.hidden_dim, eps=1e-6
            )
            self.state_token_update_attn: CrossAttention = CrossAttention(
                dim=self.hidden_dim,
                head_num=self.head_num,
                qkv_bias=True,
                qk_norm=True,
                norm_layer=RmsNorm,
            )
            self.state_token_update_out: nn.Linear = nn.Linear(
                self.hidden_dim, self.hidden_dim, bias=False
            )
            self.state_token_update_gate: nn.Sequential = nn.Sequential(
                nn.Linear(self.hidden_dim * 3 + 1, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 1),
            )
        if self.state_token_action_adapter_enabled:
            assert self.state_token_memory_enabled, (
                "state-token action adapter requires state-token memory"
            )
            assert self.state_token_action_adapter_residual_scale >= 0
            self.state_token_action_adapter_norm: nn.Module = RmsNorm(
                self.hidden_dim, eps=1e-6
            )
            self.state_token_action_adapter_attn: CrossAttention = CrossAttention(
                dim=self.hidden_dim,
                head_num=self.head_num,
                qkv_bias=True,
                qk_norm=True,
                norm_layer=RmsNorm,
            )
            self.state_token_action_adapter_out: nn.Linear = nn.Linear(
                self.hidden_dim, self.action_dim, bias=False
            )
            self.state_token_action_adapter_gate: nn.Sequential = nn.Sequential(
                nn.Linear(self.hidden_dim * 3 + 1, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 1),
            )
        if self.state_token_action_prehead_adapter_enabled:
            assert self.state_token_memory_enabled, (
                "state-token prehead action adapter requires state-token memory"
            )
            assert self.state_token_action_prehead_adapter_residual_scale >= 0
            self.state_token_action_prehead_adapter_norm: nn.Module = RmsNorm(
                self.hidden_dim, eps=1e-6
            )
            self.state_token_action_prehead_adapter_attn: CrossAttention = CrossAttention(
                dim=self.hidden_dim,
                head_num=self.head_num,
                qkv_bias=True,
                qk_norm=True,
                norm_layer=RmsNorm,
            )
            self.state_token_action_prehead_adapter_out: nn.Linear = nn.Linear(
                self.hidden_dim, self.hidden_dim, bias=False
            )
            self.state_token_action_prehead_adapter_gate: nn.Sequential = nn.Sequential(
                nn.Linear(self.hidden_dim * 3 + 1, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 1),
            )
        if self.late_cue_action_prehead_adapter_enabled:
            assert self.late_cue_action_prehead_adapter_residual_scale >= 0
            self.late_cue_action_prehead_adapter_norm: nn.Module = RmsNorm(
                self.hidden_dim, eps=1e-6
            )
            self.late_cue_action_prehead_adapter_attn: CrossAttention = CrossAttention(
                dim=self.hidden_dim,
                head_num=self.head_num,
                qkv_bias=True,
                qk_norm=True,
                norm_layer=RmsNorm,
            )
            self.late_cue_action_prehead_adapter_out: nn.Linear = nn.Linear(
                self.hidden_dim, self.hidden_dim, bias=False
            )
            self.late_cue_action_prehead_adapter_gate: nn.Sequential = nn.Sequential(
                nn.Linear(self.hidden_dim * 3 + 1, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 1),
            )
        self.late_cue_anchor_enabled: bool = late_cue_anchor_enabled
        assert late_cue_anchor_len >= 1, "late_cue_anchor_len must be >= 1"
        if self.late_cue_anchor_enabled:
            assert (
                late_cue_anchor_len <= max_history_len
            ), "late_cue_anchor_len must be <= max_history_len"
        self.late_cue_anchor_len: int = late_cue_anchor_len
        self.late_cue_anchor_causal_mask: bool = late_cue_anchor_causal_mask
        self.visual_memory_carrier_type: str = visual_memory_carrier_type
        assert self.visual_memory_carrier_type in (
            "",
            "selector",
            "gru",
        ), f"Unknown visual_memory_carrier_type: {self.visual_memory_carrier_type}"
        self.visual_memory_carrier_enabled: bool = (
            self.visual_memory_carrier_type != ""
        )
        self.visual_memory_carrier_token_num: int = visual_memory_carrier_token_num
        self.visual_memory_carrier_max_len: int = visual_memory_carrier_max_len
        self.visual_memory_carrier_force_zero: bool = visual_memory_carrier_force_zero
        assert self.visual_memory_carrier_token_num > 0
        assert self.visual_memory_carrier_max_len > 0
        assert visual_memory_carrier_num_layers > 0

        self.record_data_entries: list[str] = record_data_entries
        self.recorded_data_dict: dict[str, list[torch.Tensor]] = {}

        self.history_time_embedding: nn.Parameter = nn.Parameter(
            torch.zeros(
                1, max_history_len, self.hidden_dim
            )  # (batch, max_history_len, hidden_dim)
        )

        assert history_action_num_per_chunk > 0, "history_action_num_per_chunk must be greater than 0"
        assert history_action_num_per_chunk <= self.action_token_num, "history_action_num_per_chunk must be less than or equal to action_token_num"

        self.history_action_num_per_chunk: int = history_action_num_per_chunk

        self.initialize_memory_weights()

        if freeze_non_history_modules:
            for name, param in self.named_parameters():
                if "history" not in name:
                    param.requires_grad = False

        self.history_img_features_dim: int = history_img_features_dim
        self.history_img_features_token_num: int = history_img_features_token_num
        print(
            f"history_img_features_dim: {self.history_img_features_dim}, history_img_features_token_num: {self.history_img_features_token_num}"
        )
        self.history_img_features_projector: nn.Module | None = None
        if (
            self.history_img_features_dim > 0
            and self.history_img_features_token_num > 0
        ):
            self.history_img_features_projector = Projector(
                self.projector_type, self.history_img_features_dim, self.hidden_dim
            )
        self.visual_memory_carrier: nn.Module | None = None
        if self.visual_memory_carrier_enabled:
            from imitation_learning.models.visual_memory_carriers import (
                LearnedLateCueSelector,
                VisualGRUMemoryCarrier,
            )

            assert self.history_img_features_projector is not None, (
                "visual memory carrier requires history image features"
            )
            if self.visual_memory_carrier_type == "selector":
                self.visual_memory_carrier = LearnedLateCueSelector(
                    feature_dim=self.history_img_features_dim,
                    memory_token_num=self.visual_memory_carrier_token_num,
                    max_history_len=self.visual_memory_carrier_max_len,
                    num_heads=visual_memory_carrier_num_heads,
                    dropout=visual_memory_carrier_dropout,
                )
            elif self.visual_memory_carrier_type == "gru":
                self.visual_memory_carrier = VisualGRUMemoryCarrier(
                    feature_dim=self.history_img_features_dim,
                    hidden_dim=visual_memory_carrier_hidden_dim,
                    memory_token_num=self.visual_memory_carrier_token_num,
                    num_layers=visual_memory_carrier_num_layers,
                    dropout=visual_memory_carrier_dropout,
                )
        if (
            self.state_token_memory_enabled
            and not self.state_token_update_include_action_history
        ):
            assert self.history_img_features_projector is not None, (
                "state-token visual-only updates require history image features"
            )

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(
            f"MemoryTransformer trainable parameters: {trainable_params}, total parameters: {total_params}"
        )
        self.include_action_history: bool = include_action_history
        assert (
            self.include_action_history or self.history_img_features_dim > 0
        ), "At least one of include_action_history or history_img_features_dim must be True"

        if add_memory_gate_token:
            assert binary_gating, "Memory gate token is only supported when binary gating is enabled"
            self.memory_gate_tokens: nn.Parameter | None = nn.Parameter(
                torch.randn(2, self.hidden_dim)
            ) # Two tokens: [0] when memory gate == 0, [1] when memory gate == 1
        else:
            self.memory_gate_tokens = None

    def _build_visual_memory_carrier_latents(
        self,
        history_img_features: torch.Tensor,
        history_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.visual_memory_carrier is not None
        assert self.history_img_features_projector is not None
        carrier_features = self.visual_memory_carrier(
            history_img_features,
            history_mask,
            force_zero=self.visual_memory_carrier_force_zero,
        )
        carrier_mask = torch.ones(
            carrier_features.shape[:2],
            device=carrier_features.device,
            dtype=torch.bool,
        )
        if history_mask is not None:
            carrier_mask = carrier_mask & history_mask.bool().any(dim=1, keepdim=True)
        if self.visual_memory_carrier_force_zero:
            carrier_mask = torch.zeros_like(carrier_mask)
        carrier_latents = self.history_img_features_projector(carrier_features)
        carrier_len = carrier_latents.shape[1]
        carrier_latents = (
            carrier_latents
            + self.history_time_embedding[:, :carrier_len, None, :]
        )
        return carrier_latents, carrier_mask

    def set_skip_history_attn(self, skip_history_attn: bool):
        self.skip_history_attn = skip_history_attn
        for block in self.blocks:
            assert isinstance(block, MemoryTransformerBlock)
            block.skip_history_attn = skip_history_attn

    def initialize_memory_weights(self):
        # Initialize transformer
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.blocks.apply(_basic_init)
        for block in self.blocks:
            assert isinstance(block, MemoryTransformerBlock)
            block.reset_cue_event_memory_parameters()
            block.reset_late_cue_adapter_parameters()
            block.reset_anchor_current_binding_parameters()
            block.reset_history_summary_memory_parameters()
            block.reset_state_token_memory_parameters()
        self.reset_state_token_update_parameters()
        self.reset_state_token_action_adapter_parameters()
        self.reset_state_token_action_prehead_adapter_parameters()
        self.reset_late_cue_action_prehead_adapter_parameters()

        history_time_embedding = get_1d_sincos_pos_embed_from_grid(
            embed_dim=self.hidden_dim,
            pos=np.arange(self.max_history_len),
        )  # (max_history_len, hidden_dim)
        self.history_time_embedding.data.copy_(
            torch.from_numpy(history_time_embedding).unsqueeze(0)
        )  # (1, max_history_len, hidden_dim)

    def reset_state_token_update_parameters(self):
        if not self.state_token_memory_enabled:
            return
        nn.init.normal_(
            self.state_token_initial,
            std=self.state_token_initial_std,
        )
        if self.state_token_update_out_init_gain > 0:
            nn.init.xavier_uniform_(
                self.state_token_update_out.weight,
                gain=self.state_token_update_out_init_gain,
            )
        else:
            nn.init.zeros_(self.state_token_update_out.weight)
        gate_head = self.state_token_update_gate[-1]
        assert isinstance(gate_head, nn.Linear)
        nn.init.zeros_(gate_head.weight)
        nn.init.constant_(gate_head.bias, self.state_token_update_gate_init_bias)

    def reset_state_token_action_adapter_parameters(self):
        if not self.state_token_action_adapter_enabled:
            return
        if self.state_token_action_adapter_out_init_gain > 0:
            nn.init.xavier_uniform_(
                self.state_token_action_adapter_out.weight,
                gain=self.state_token_action_adapter_out_init_gain,
            )
        else:
            nn.init.zeros_(self.state_token_action_adapter_out.weight)
        gate_head = self.state_token_action_adapter_gate[-1]
        assert isinstance(gate_head, nn.Linear)
        nn.init.zeros_(gate_head.weight)
        nn.init.constant_(
            gate_head.bias, self.state_token_action_adapter_gate_init_bias
        )

    def reset_state_token_action_prehead_adapter_parameters(self):
        if not self.state_token_action_prehead_adapter_enabled:
            return
        if self.state_token_action_prehead_adapter_out_init_gain > 0:
            nn.init.xavier_uniform_(
                self.state_token_action_prehead_adapter_out.weight,
                gain=self.state_token_action_prehead_adapter_out_init_gain,
            )
        else:
            nn.init.zeros_(self.state_token_action_prehead_adapter_out.weight)
        gate_head = self.state_token_action_prehead_adapter_gate[-1]
        assert isinstance(gate_head, nn.Linear)
        nn.init.zeros_(gate_head.weight)
        nn.init.constant_(
            gate_head.bias, self.state_token_action_prehead_adapter_gate_init_bias
        )

    def reset_late_cue_action_prehead_adapter_parameters(self):
        if not self.late_cue_action_prehead_adapter_enabled:
            return
        if self.late_cue_action_prehead_adapter_out_init_gain > 0:
            nn.init.xavier_uniform_(
                self.late_cue_action_prehead_adapter_out.weight,
                gain=self.late_cue_action_prehead_adapter_out_init_gain,
            )
        else:
            nn.init.zeros_(self.late_cue_action_prehead_adapter_out.weight)
        gate_head = self.late_cue_action_prehead_adapter_gate[-1]
        assert isinstance(gate_head, nn.Linear)
        nn.init.zeros_(gate_head.weight)
        nn.init.constant_(
            gate_head.bias, self.late_cue_action_prehead_adapter_gate_init_bias
        )

    def initial_state_tokens(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        assert self.state_token_memory_enabled
        state_tokens = self.state_token_initial.to(device=device)
        if dtype is not None:
            state_tokens = state_tokens.to(dtype=dtype)
        return state_tokens.expand(batch_size, -1, -1).clone()

    def build_state_token_update_latents(
        self,
        update_actions: torch.Tensor | None,
        history_img_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        update_latents_list: list[torch.Tensor] = []
        if update_actions is not None and self.state_token_update_include_action_history:
            update_action_latents = self.action_projector(update_actions)
            start_idx = self.input_pos_embedding.shape[1] - self.action_token_num
            end_idx = start_idx + update_actions.shape[-2]
            action_pos_embedding = self.input_pos_embedding[:, start_idx:end_idx, :]
            while action_pos_embedding.dim() < update_action_latents.dim():
                action_pos_embedding = action_pos_embedding.unsqueeze(1)
            update_latents_list.append(update_action_latents + action_pos_embedding)

        if (
            history_img_features is not None
            and self.history_img_features_projector is not None
        ):
            update_latents_list.append(
                self.history_img_features_projector(history_img_features)
            )

        assert len(update_latents_list) > 0, (
            "state-token update requires action history or image feature history"
        )
        return torch.cat(update_latents_list, dim=-2)

    def update_state_tokens(
        self,
        state_tokens: torch.Tensor,
        update_latents: torch.Tensor,
        update_mask: torch.Tensor | None = None,
        state_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.state_token_memory_enabled
        batch_size, update_token_num, hidden_dim = update_latents.shape
        assert state_tokens.shape == (
            batch_size,
            self.state_token_num,
            hidden_dim,
        )
        if update_mask is None:
            update_mask = torch.ones(
                batch_size, device=update_latents.device, dtype=torch.bool
            )
        else:
            update_mask = update_mask.bool()
        if state_token_mask is not None:
            assert state_token_mask.shape == (batch_size, self.state_token_num)
            state_token_mask = state_token_mask.bool()

        token_mask = update_mask[:, None].expand(batch_size, update_token_num)
        safe_update_latents = update_latents
        safe_token_mask = token_mask
        if update_mask.logical_not().any():
            safe_update_latents = update_latents.clone()
            safe_token_mask = token_mask.clone()
            empty_rows = update_mask.logical_not()
            safe_update_latents[empty_rows] = 0
            safe_token_mask[empty_rows, 0] = True

        valid_scale = update_mask.to(dtype=state_tokens.dtype)[:, None, None]
        update_attention = self.state_token_update_attn(
            self.state_token_update_norm(state_tokens),
            safe_update_latents,
            safe_token_mask,
        )
        update_attention = update_attention * valid_scale
        update_summary = safe_update_latents.mean(dim=1) * update_mask.to(
            dtype=state_tokens.dtype
        )[:, None]
        valid_ratio = update_mask.to(dtype=state_tokens.dtype)[:, None]
        gate_input = torch.cat(
            [
                state_tokens.mean(dim=1),
                update_summary,
                update_attention.mean(dim=1),
                valid_ratio,
            ],
            dim=-1,
        )
        update_gate = torch.sigmoid(self.state_token_update_gate(gate_input))
        update_delta = self.state_token_update_out(update_attention)
        updated_state_tokens = (
            state_tokens
            + self.state_token_update_residual_scale
            * update_gate[:, None, :]
            * update_delta
        )
        prefix_num = self.state_token_protected_prefix_num
        if prefix_num <= 0 or state_token_mask is None:
            return updated_state_tokens

        protected_prefix = torch.where(
            state_token_mask[:, :prefix_num, None],
            state_tokens[:, :prefix_num],
            updated_state_tokens[:, :prefix_num],
        )
        if prefix_num == self.state_token_num:
            return protected_prefix
        return torch.cat(
            [protected_prefix, updated_state_tokens[:, prefix_num:]],
            dim=1,
        )

    def scan_state_tokens(
        self,
        update_latents: torch.Tensor,
        update_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.state_token_memory_enabled
        batch_size, traj_num = update_latents.shape[:2]
        state_tokens = self.initial_state_tokens(
            batch_size,
            update_latents.device,
            update_latents.dtype,
        )
        if update_mask is None:
            update_mask = torch.ones(
                batch_size,
                traj_num,
                device=update_latents.device,
                dtype=torch.bool,
            )
        else:
            update_mask = update_mask.bool()

        seen_valid = torch.zeros(
            batch_size, device=update_latents.device, dtype=torch.bool
        )
        state_tokens_before: list[torch.Tensor] = []
        state_masks_before: list[torch.Tensor] = []
        for traj_idx in range(traj_num):
            state_mask_before = seen_valid[:, None].expand(
                batch_size, self.state_token_num
            )
            state_tokens_before.append(state_tokens)
            state_masks_before.append(state_mask_before)
            step_valid = update_mask[:, traj_idx]
            state_tokens = self.update_state_tokens(
                state_tokens,
                update_latents[:, traj_idx],
                step_valid,
                state_token_mask=state_mask_before,
            )
            seen_valid = seen_valid | step_valid

        return (
            torch.stack(state_tokens_before, dim=1),
            torch.stack(state_masks_before, dim=1),
        )

    def scan_state_tokens_pre_action_obs(
        self,
        obs_update_latents: torch.Tensor,
        action_update_latents: torch.Tensor | None = None,
        update_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.state_token_memory_enabled
        batch_size, traj_num = obs_update_latents.shape[:2]
        state_tokens = self.initial_state_tokens(
            batch_size,
            obs_update_latents.device,
            obs_update_latents.dtype,
        )
        if action_update_latents is not None:
            assert action_update_latents.shape[:2] == (batch_size, traj_num)
            assert action_update_latents.shape[-1] == obs_update_latents.shape[-1]
        if update_mask is None:
            update_mask = torch.ones(
                batch_size,
                traj_num,
                device=obs_update_latents.device,
                dtype=torch.bool,
            )
        else:
            update_mask = update_mask.bool()

        seen_valid = torch.zeros(
            batch_size, device=obs_update_latents.device, dtype=torch.bool
        )
        state_tokens_for_read: list[torch.Tensor] = []
        state_masks_for_read: list[torch.Tensor] = []
        for traj_idx in range(traj_num):
            state_mask_before = seen_valid[:, None].expand(
                batch_size, self.state_token_num
            )
            step_valid = update_mask[:, traj_idx]
            state_tokens = self.update_state_tokens(
                state_tokens,
                obs_update_latents[:, traj_idx],
                step_valid,
                state_token_mask=state_mask_before,
            )
            read_mask = (seen_valid | step_valid)[:, None].expand(
                batch_size,
                self.state_token_num,
            )
            state_tokens_for_read.append(state_tokens)
            state_masks_for_read.append(read_mask)
            if action_update_latents is not None:
                state_tokens = self.update_state_tokens(
                    state_tokens,
                    action_update_latents[:, traj_idx],
                    step_valid,
                    state_token_mask=read_mask,
                )
            seen_valid = seen_valid | step_valid

        return (
            torch.stack(state_tokens_for_read, dim=1),
            torch.stack(state_masks_for_read, dim=1),
        )

    def _project_to_latent_space(
        self, data_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        This function will manage all the history related data obtain history latents
        Will also call the super class's _process_data_dict to process the rest of the data
        data_dict: {
            "noisy_action": (batch, traj_length, action_dim),
            "step": (batch,),
            "global_cond": (batch, global_cond_token_num, global_cond_dim),
            "local_cond": (batch, local_cond_token_num, local_cond_dim),
            "global_cond_mask": (batch, global_cond_token_num),
            "local_cond_mask": (batch, local_cond_token_num),
            "history_noisy_actions": (batch, history_len, history_action_num_per_chunk, action_dim),
            "history_mask": (batch, history_len),
            "history_img_features": (batch, history_len, history_img_features_token_num, history_img_features_dim),
            "memory_gate_val": (batch,),
        }

        input_dict: {
            "x": (batch, 1(memory gate token, optional) + 1(denoising step) + local_cond_token_num + action_token_num, hidden_dim),
            "global_cond": (batch, global_cond_token_num, hidden_dim),
            "global_cond_mask": (batch, global_cond_token_num), # optional
            "history_latents": (batch, history_len, action_token_num, hidden_dim),
            "history_mask": (batch, history_len), # optional
            "memory_gate_val": (batch,), # optional
            "step": (batch,),
        }
        """
        state_token_latents = data_dict.pop("state_token_latents", None)
        state_token_mask = data_dict.pop("state_token_mask", None)
        if "history_noisy_actions" in data_dict:
            history_noisy_actions = data_dict.pop("history_noisy_actions")
            late_cue_anchor_img_features = data_dict.pop(
                "late_cue_anchor_img_features", None
            )
            late_cue_anchor_mask = data_dict.pop("late_cue_anchor_mask", None)
            visual_memory_carrier_img_features = data_dict.pop(
                "visual_memory_carrier_img_features", None
            )
            visual_memory_carrier_mask = data_dict.pop(
                "visual_memory_carrier_mask", None
            )

            history_len = history_noisy_actions.shape[1]
            assert (
                history_len <= self.max_history_len
            ), "history_len must be less than or equal to max_history_len"
            # Apply relative time embeddings to the history latents
            history_latents_list: list[torch.Tensor] = []
            if self.include_action_history:
                action_latents: torch.Tensor = self.action_projector(
                    history_noisy_actions
                )  # (batch, history_len, history_action_num_per_chunk, hidden_dim)
                start_idx = (
                    self.input_pos_embedding.shape[1]
                    - self.action_token_num
                )
                end_idx = (
                    self.input_pos_embedding.shape[1]
                    - self.action_token_num
                    + self.history_action_num_per_chunk
                )
                action_latents = (
                    action_latents
                    + self.input_pos_embedding[:, None, start_idx:end_idx, :]
                )
                history_latents_list.append(action_latents)

            if self.history_img_features_projector is not None and self.max_history_len > 0:
                history_img_features = data_dict.pop("history_img_features")
                # print(f"{torch.allclose(history_img_features, history_img_features[0])=}")
                history_img_features_latent = self.history_img_features_projector(
                    history_img_features
                )  # (batch, history_len, history_img_features_token_num, hidden_dim)
                history_latents_list.append(history_img_features_latent)
                # (batch, history_len, action_token_num + history_img_features_token_num, hidden_dim)

            history_latents = torch.cat(history_latents_list, dim=2)
            history_latents = (
                history_latents + self.history_time_embedding[:, -history_len:, None, :]
            )
            input_dict = {}
            if "history_mask" in data_dict:
                input_dict["history_mask"] = data_dict.pop("history_mask")
            if "memory_gate_val" in data_dict:
                input_dict["memory_gate_val"] = data_dict.pop("memory_gate_val")
            input_dict["step"] = data_dict["step"]
            input_dict["history_latents"] = history_latents
            if (
                self.late_cue_anchor_enabled
                and not self.visual_memory_carrier_enabled
                and late_cue_anchor_img_features is not None
                and self.history_img_features_projector is not None
            ):
                if late_cue_anchor_img_features.dim() == 3:
                    late_cue_anchor_img_features = late_cue_anchor_img_features[:, None]
                late_cue_anchor_latents = self.history_img_features_projector(
                    late_cue_anchor_img_features
                )
                late_cue_anchor_latents = (
                    late_cue_anchor_latents
                    + self.history_time_embedding[:, : late_cue_anchor_latents.shape[1], None, :]
                )
                input_dict["late_cue_anchor_latents"] = late_cue_anchor_latents
                input_dict["late_cue_anchor_action_latents"] = late_cue_anchor_latents
                if late_cue_anchor_mask is None:
                    late_cue_anchor_mask = torch.ones(
                        late_cue_anchor_latents.shape[:2],
                        device=late_cue_anchor_latents.device,
                        dtype=torch.bool,
                    )
                input_dict["late_cue_anchor_mask"] = late_cue_anchor_mask.bool()
                input_dict["late_cue_anchor_action_mask"] = late_cue_anchor_mask.bool()
            if (
                self.late_cue_anchor_enabled
                and self.visual_memory_carrier_enabled
                and visual_memory_carrier_img_features is not None
            ):
                carrier_latents, carrier_mask = (
                    self._build_visual_memory_carrier_latents(
                        visual_memory_carrier_img_features,
                        visual_memory_carrier_mask,
                    )
                )
                input_dict["late_cue_anchor_latents"] = carrier_latents
                input_dict["late_cue_anchor_action_latents"] = carrier_latents
                input_dict["late_cue_anchor_mask"] = carrier_mask
                input_dict["late_cue_anchor_action_mask"] = carrier_mask
            if state_token_latents is not None:
                input_dict["state_token_latents"] = state_token_latents
                if state_token_mask is not None:
                    input_dict["state_token_mask"] = state_token_mask.bool()
            input_dict.update(super()._project_to_latent_space(data_dict))

            if self.memory_gate_tokens is not None:
                assert "memory_gate_val" in input_dict, "memory_gate_val must be in input_dict when memory gate tokens are used"
                binary_gate_val = (input_dict["memory_gate_val"] > 0.5).int()
                batch_memory_gate_tokens = self.memory_gate_tokens[binary_gate_val, None, :] # (batch, 1, hidden_dim)
                input_dict["x"] = torch.cat([batch_memory_gate_tokens, input_dict["x"]], dim=1)
                # print(f"{input_dict['x'].shape=}")

            return input_dict

        else:
            input_dict = super()._project_to_latent_space(data_dict)
            if state_token_latents is not None:
                input_dict["state_token_latents"] = state_token_latents
                if state_token_mask is not None:
                    input_dict["state_token_mask"] = state_token_mask.bool()

        return input_dict

    def _project_to_latent_space_multi_traj(
        self, data_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Will generate history latents from noisy actions
        Will also call the super class's _process_data_dict to process the rest of the data
        This function is only called during training when multiple supervised trajectories are provided at the same time
        data_dict: {
            # Same as no-history version
            "noisy_action": (batch, traj_num, data_length, action_dim),
            "step": (batch,),
            "global_cond": (batch, traj_num, global_cond_token_num, global_cond_dim),
            "local_cond": (batch, traj_num, local_cond_token_num, local_cond_dim),
            "global_cond_mask": (batch, traj_num, global_cond_token_num),
            "local_cond_mask": (batch, traj_num, local_cond_token_num),
            "noisy_traj_mask": (batch, traj_num, data_length),
            "step_mask": (batch, traj_num),
            # History-specific
            "history_noisy_actions": (batch, traj_num, history_action_num_per_chunk, action_dim), # Optional
            "history_img_features": (batch, traj_num, data_length, feature_dim), # Optional
            "history_mask": (batch, traj_num),
            "memory_gate_val": (batch, traj_num), # Optional
            "training_traj_indices": (batch, max_training_traj_num), # Optional
        }

        return: {
            "x": (batch*traj_num, local_cond_token_num + action_token_num + 1(denoising step) + 1(memory gate token, optional), hidden_dim),
            "global_cond": (batch*traj_num, global_cond_token_num, hidden_dim),
            "global_cond_mask": (batch*traj_num, global_cond_token_num),

            "step": (batch*traj_num,),
            "history_latents": (batch*traj_num, max_history_len, token_num, hidden_dim),
            "history_mask": (batch*traj_num, max_history_len),
            "memory_gate_val": (batch*traj_num,),
        }
        if self.max_training_traj_num > 0, traj_num will be replaced by self.max_training_traj_num if exceeds
        """

        traj_num = data_dict["noisy_action"].shape[1]
        batch_size = data_dict["noisy_action"].shape[0]
        device = data_dict["noisy_action"].device

        assert (
            traj_num >= self.max_history_len + 1
        ), f"To make sure all the history time embeddings are trained,traj_num ({traj_num}) must be at least equal to max_history_len + 1 ({self.max_history_len + 1}) in each training step"
        assert (
            traj_num == data_dict["history_noisy_actions"].shape[1]
        ), f"history_noisy_actions ({data_dict['history_noisy_actions'].shape}) must have the same trajectory number (dimension 1) as noisy_action ({traj_num})"

        # Step 1: Project history features on ALL trajectories (needed for history window construction)
        history_latents_list: list[torch.Tensor] = []
        projected_history_img_features_latent = None

        if self.include_action_history:
            projected_history_actions = self.action_projector(
                data_dict["history_noisy_actions"]
            )  # (batch, traj_num, history_action_num_per_chunk, hidden_dim)
            start_idx = (
                self.input_pos_embedding.shape[1]
                - self.action_token_num
            )
            end_idx = (
                self.input_pos_embedding.shape[1]
                - self.action_token_num
                + self.history_action_num_per_chunk
            )
            projected_history_actions = (
                projected_history_actions
                + self.input_pos_embedding[:, None, start_idx:end_idx, :]
            )
            history_latents_list.append(projected_history_actions)

        if self.history_img_features_projector is not None:
            projected_history_img_features_latent = self.history_img_features_projector(
                data_dict["history_img_features"]
            )  # (batch, traj_num, traj_length, hidden_dim)
            history_latents_list.append(projected_history_img_features_latent)

        history_latents = torch.cat(history_latents_list, dim=2)
        # (batch, traj_num, token_num, hidden_dim)
        all_state_token_latents = None
        all_state_token_masks = None
        if self.state_token_memory_enabled:
            if self.state_token_update_include_action_history:
                state_update_actions = data_dict.get(
                    "state_token_update_actions",
                    data_dict["history_noisy_actions"],
                )
            else:
                state_update_actions = None
            state_update_img_features = data_dict.get("history_img_features")
            state_update_mask = None
            if "entire_traj_is_padding" in data_dict:
                state_update_mask = ~data_dict["entire_traj_is_padding"].bool()
            if self.state_token_pre_action_obs_update_in_training_enabled:
                assert state_update_img_features is not None, (
                    "pre-action obs state-token training update requires "
                    "history_img_features"
                )
                obs_update_latents = self.build_state_token_update_latents(
                    None,
                    state_update_img_features,
                )
                action_update_latents = None
                if state_update_actions is not None:
                    action_update_latents = self.build_state_token_update_latents(
                        state_update_actions,
                        None,
                    )
                (
                    all_state_token_latents,
                    all_state_token_masks,
                ) = self.scan_state_tokens_pre_action_obs(
                    obs_update_latents,
                    action_update_latents,
                    state_update_mask,
                )
            else:
                state_update_latents = self.build_state_token_update_latents(
                    state_update_actions,
                    state_update_img_features,
                )
                (
                    all_state_token_latents,
                    all_state_token_masks,
                ) = self.scan_state_tokens(
                    state_update_latents,
                    state_update_mask,
                )
            if "state_token_source_indices" in data_dict:
                source_indices = data_dict["state_token_source_indices"].to(
                    device=device,
                    dtype=torch.long,
                )
                assert source_indices.shape == (batch_size,)
                all_state_token_latents = all_state_token_latents[source_indices]
                all_state_token_masks = all_state_token_masks[source_indices]
        late_cue_anchor_latents = None
        late_cue_anchor_action_latents = None
        late_cue_anchor_len = 0
        if self.late_cue_anchor_enabled and not self.visual_memory_carrier_enabled:
            late_cue_anchor_len = min(self.late_cue_anchor_len, history_latents.shape[1])
            late_cue_anchor_latents = (
                history_latents[:, :late_cue_anchor_len]
                + self.history_time_embedding[:, :late_cue_anchor_len, None, :]
            )
            if projected_history_img_features_latent is not None:
                late_cue_anchor_action_latents = (
                    projected_history_img_features_latent[:, :late_cue_anchor_len]
                    + self.history_time_embedding[:, :late_cue_anchor_len, None, :]
                )

        # Step 2: Determine effective trajectories and sample non-history tensors early
        has_training_traj_indices = "training_traj_indices" in data_dict
        if has_training_traj_indices:
            traj_indices = data_dict["training_traj_indices"]  # (batch, effective_traj_num)
            effective_traj_num = traj_indices.shape[1]
        else:
            traj_indices = torch.arange(traj_num, device=device).unsqueeze(0).expand(batch_size, -1)
            effective_traj_num = traj_num

        reshaped_data_dict: dict[str, torch.Tensor] = {}
        non_history_keys = [
            "noisy_action",
            "global_cond",
            "local_cond",
            "global_cond_mask",
            "local_cond_mask",
            "noisy_traj_mask",
            "step_mask",
        ]
        for key in non_history_keys:
            if key in data_dict:
                tensor = data_dict[key]
                if has_training_traj_indices:
                    batch_idx_2d = torch.arange(batch_size, device=device)[:, None]
                    tensor = tensor[batch_idx_2d, traj_indices]
                reshaped_data_dict[key] = einops.rearrange(
                    tensor, "batch traj_num ... -> (batch traj_num) ..."
                )

        reshaped_data_dict["step"] = (
            data_dict["step"].repeat(1, effective_traj_num).view(-1)
        )

        # Step 3: Project non-history tensors (only effective trajectories go through the projectors)
        input_dict = super()._project_to_latent_space(reshaped_data_dict)
        input_dict["step"] = reshaped_data_dict["step"]

        # Step 4: Construct history windows via vectorized gather (replaces Python for-loop)
        # For trajectory index i, history window position k (0..max_history_len-1)
        # maps to source index: i - max_history_len + k
        window_offsets = torch.arange(self.max_history_len, device=device)
        # source_indices[b, j, k] = traj_indices[b, j] - max_history_len + k
        source_indices = traj_indices[:, :, None] - self.max_history_len + window_offsets[None, None, :]
        # (batch, effective_traj_num, max_history_len)
        valid_mask = source_indices >= 0
        source_indices_clamped = source_indices.clamp(min=0)

        batch_idx_3d = torch.arange(batch_size, device=device)[:, None, None]
        merged_history_latents = history_latents[batch_idx_3d, source_indices_clamped]
        # (batch, effective_traj_num, max_history_len, token_num, hidden_dim)
        merged_history_latents = merged_history_latents * valid_mask[:, :, :, None, None]

        if "history_mask" in data_dict:
            full_history_mask = data_dict["history_mask"]  # (batch, traj_num)
            merged_history_masks = full_history_mask[batch_idx_3d, source_indices_clamped]
            # (batch, effective_traj_num, max_history_len)
            merged_history_masks = merged_history_masks & valid_mask
        else:
            merged_history_masks = valid_mask

        input_dict["history_latents"] = einops.rearrange(
            merged_history_latents, "batch traj_num ... -> (batch traj_num) ... "
        )  # (batch*effective_traj_num, max_history_len, token_num, hidden_dim)

        input_dict["history_latents"] = (
            input_dict["history_latents"] + self.history_time_embedding[:, :, None, :]
        )
        input_dict["history_mask"] = einops.rearrange(
            merged_history_masks, "batch traj_num ... -> (batch traj_num) ... "
        )
        if (
            self.late_cue_anchor_enabled
            and self.visual_memory_carrier_enabled
            and self.history_img_features_projector is not None
        ):
            carrier_history_len = min(
                self.visual_memory_carrier_max_len,
                data_dict["history_img_features"].shape[1],
            )
            source_history_img_features = data_dict["history_img_features"][
                :, :carrier_history_len
            ]
            prefix_features = source_history_img_features[:, None].expand(
                -1,
                effective_traj_num,
                -1,
                -1,
                -1,
            )
            prefix_features = einops.rearrange(
                prefix_features,
                "batch traj_num history_len token_num feature_dim -> (batch traj_num) history_len token_num feature_dim",
            )
            prefix_positions = torch.arange(
                carrier_history_len,
                device=device,
            )[None, None, :]
            prefix_mask = prefix_positions < traj_indices[:, :, None]
            if "history_mask" in data_dict:
                prefix_mask = prefix_mask & data_dict["history_mask"][
                    :, None, :carrier_history_len
                ].bool()
            prefix_mask = einops.rearrange(
                prefix_mask,
                "batch traj_num history_len -> (batch traj_num) history_len",
            )
            carrier_latents, carrier_mask = self._build_visual_memory_carrier_latents(
                prefix_features,
                prefix_mask,
            )
            input_dict["late_cue_anchor_latents"] = carrier_latents
            input_dict["late_cue_anchor_action_latents"] = carrier_latents
            input_dict["late_cue_anchor_mask"] = carrier_mask
            input_dict["late_cue_anchor_action_mask"] = carrier_mask
        if all_state_token_latents is not None and all_state_token_masks is not None:
            batch_idx_2d = torch.arange(batch_size, device=device)[:, None]
            gathered_state_token_latents = all_state_token_latents[
                batch_idx_2d, traj_indices
            ]
            gathered_state_token_masks = all_state_token_masks[
                batch_idx_2d, traj_indices
            ]
            input_dict["state_token_latents"] = einops.rearrange(
                gathered_state_token_latents,
                "batch traj_num state_token_num hidden_dim -> (batch traj_num) state_token_num hidden_dim",
            )
            input_dict["state_token_mask"] = einops.rearrange(
                gathered_state_token_masks,
                "batch traj_num state_token_num -> (batch traj_num) state_token_num",
            )
        if late_cue_anchor_latents is not None:
            expanded_anchor_latents = late_cue_anchor_latents[:, None].expand(
                -1, effective_traj_num, -1, -1, -1
            )
            input_dict["late_cue_anchor_latents"] = einops.rearrange(
                expanded_anchor_latents,
                "batch traj_num anchor_len token_num hidden_dim -> (batch traj_num) anchor_len token_num hidden_dim",
            )
            if self.late_cue_anchor_causal_mask:
                anchor_positions = torch.arange(
                    late_cue_anchor_len, device=device
                )[None, None, :]
                expanded_anchor_mask = anchor_positions < traj_indices[:, :, None]
            else:
                expanded_anchor_mask = torch.ones(
                    (batch_size, effective_traj_num, late_cue_anchor_len),
                    device=device,
                    dtype=torch.bool,
                )
            input_dict["late_cue_anchor_mask"] = einops.rearrange(
                expanded_anchor_mask,
                "batch traj_num anchor_len -> (batch traj_num) anchor_len",
            )
            if late_cue_anchor_action_latents is not None:
                expanded_action_anchor_latents = (
                    late_cue_anchor_action_latents[:, None].expand(
                        -1, effective_traj_num, -1, -1, -1
                    )
                )
                input_dict["late_cue_anchor_action_latents"] = einops.rearrange(
                    expanded_action_anchor_latents,
                    "batch traj_num anchor_len token_num hidden_dim -> (batch traj_num) anchor_len token_num hidden_dim",
                )
                input_dict["late_cue_anchor_action_mask"] = input_dict[
                    "late_cue_anchor_mask"
                ]

        # Step 5: Memory gate (sample if needed)
        if "memory_gate_val" in data_dict:
            memory_gate_val = data_dict["memory_gate_val"]
            if has_training_traj_indices:
                batch_idx_2d = torch.arange(batch_size, device=device)[:, None]
                memory_gate_val = memory_gate_val[batch_idx_2d, traj_indices]
            input_dict["memory_gate_val"] = einops.rearrange(
                memory_gate_val, "batch traj_num ... -> (batch traj_num) ..."
            )
        if self.memory_gate_tokens is not None:
            assert "memory_gate_val" in input_dict, "memory_gate_val must be in input_dict when memory gate tokens are used"
            binary_gate_val = (input_dict["memory_gate_val"] > 0.5).int()
            batch_memory_gate_tokens = self.memory_gate_tokens[binary_gate_val, None, :] # (batch, 1, hidden_dim)
            input_dict["x"] = torch.cat([batch_memory_gate_tokens, input_dict["x"]], dim=1)

        return input_dict

    def parallel_forward(
        self,
        data_dict: dict[str, torch.Tensor],
        disable_direct_history_paths: bool = False,
        disable_state_token_action_adapters: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Only used when training with multiple ground-truth trajectories
        data_dict: {
            "noisy_action": (batch, traj_num, traj_length, action_dim),
            "step": (batch,),
            "global_cond": (batch, traj_num, global_cond_token_num, global_cond_dim),
            "local_cond": (batch, traj_num, local_cond_token_num, local_cond_dim),
            "history_noisy_actions": (batch, traj_num, history_action_num_per_chunk, action_dim),
            "history_mask": (batch, traj_num),
            "memory_gate_val": (batch, traj_num),
            "training_traj_indices": (batch, max_training_traj_num), # Optional, if provided, will only use the indexed trajectories for training
        }
        return: {
            "action": (batch, traj_num, traj_length, action_dim),
        }
        if "training_traj_indices" in data_dict, traj_num will be replaced by max_training_traj_num
        """

        input_dict = self._project_to_latent_space_multi_traj(data_dict)
        input_dict["disable_direct_history_paths"] = disable_direct_history_paths
        parallel_results = self._get_results(
            self._run_blocks(input_dict),
            disable_state_token_action_adapters=disable_state_token_action_adapters,
        )

        batch_size = data_dict["noisy_action"].shape[0]
        if "training_traj_indices" in data_dict:
            traj_num = data_dict["training_traj_indices"].shape[1]
        else:
            # raise RuntimeError("training_traj_indices is not provided")
            traj_num = data_dict["noisy_action"].shape[1]
        results: dict[str, torch.Tensor] = {
            "action": einops.rearrange(
                parallel_results["action"],
                "(batch traj_num) ... -> batch traj_num ...",
                batch=batch_size,
                traj_num=traj_num,
            ),
        }
        return results

    def _run_blocks(self, input_dict: dict[str, torch.Tensor]):


        if len(self.record_data_entries) > 0:
            record_data_dict: dict[str, list[torch.Tensor]] | None = {
                entry: [] for entry in self.record_data_entries
            }
            if "memory_gate_val" in input_dict and "memory_gate_val" in record_data_dict:
                record_data_dict["memory_gate_val"].append(input_dict["memory_gate_val"].clone())
        else:
            record_data_dict = None

        for i, block in enumerate(self.blocks):
            input_dict["x"] = block(record_data_dict=record_data_dict, **input_dict)

        if record_data_dict is not None:
            self.recorded_data_dict = record_data_dict

        return input_dict

    def _read_state_token_action_adapter(
        self,
        action_tokens: torch.Tensor,
        state_token_latents: torch.Tensor,
        state_token_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state_token_mask is None:
            state_token_mask = torch.ones(
                state_token_latents.shape[:2],
                device=state_token_latents.device,
                dtype=torch.bool,
            )
        else:
            state_token_mask = state_token_mask.bool()
        if self.state_token_read_prefix_num > 0:
            prefix = min(self.state_token_read_prefix_num, state_token_mask.shape[1])
            prefix_mask = torch.zeros_like(state_token_mask)
            prefix_mask[:, :prefix] = state_token_mask[:, :prefix]
            state_token_mask = prefix_mask

        safe_state_tokens = state_token_latents
        safe_mask = state_token_mask
        has_valid = state_token_mask.any(dim=1, keepdim=True)
        if has_valid.logical_not().any():
            safe_state_tokens = state_token_latents.clone()
            safe_mask = state_token_mask.clone()
            empty_rows = has_valid.squeeze(1).logical_not()
            safe_state_tokens[empty_rows] = 0
            safe_mask[empty_rows, 0] = True

        valid_ratio = state_token_mask.float().mean(dim=1, keepdim=True).to(
            dtype=action_tokens.dtype
        )
        valid_scale = has_valid.to(dtype=action_tokens.dtype)[:, :, None]
        state_attention = self.state_token_action_adapter_attn(
            self.state_token_action_adapter_norm(action_tokens),
            safe_state_tokens,
            safe_mask,
        )
        state_attention = state_attention * valid_scale
        state_summary = safe_state_tokens.mean(dim=1) * has_valid.to(
            dtype=action_tokens.dtype
        )
        gate_input = torch.cat(
            [
                action_tokens.mean(dim=1),
                state_summary,
                state_attention.mean(dim=1),
                valid_ratio,
            ],
            dim=-1,
        )
        state_gate = torch.sigmoid(self.state_token_action_adapter_gate(gate_input))
        return state_attention, state_gate, valid_ratio

    def _read_state_token_action_prehead_adapter(
        self,
        action_tokens: torch.Tensor,
        state_token_latents: torch.Tensor,
        state_token_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state_token_mask is None:
            state_token_mask = torch.ones(
                state_token_latents.shape[:2],
                device=state_token_latents.device,
                dtype=torch.bool,
            )
        else:
            state_token_mask = state_token_mask.bool()
        if self.state_token_read_prefix_num > 0:
            prefix = min(self.state_token_read_prefix_num, state_token_mask.shape[1])
            prefix_mask = torch.zeros_like(state_token_mask)
            prefix_mask[:, :prefix] = state_token_mask[:, :prefix]
            state_token_mask = prefix_mask

        safe_state_tokens = state_token_latents
        safe_mask = state_token_mask
        has_valid = state_token_mask.any(dim=1, keepdim=True)
        if has_valid.logical_not().any():
            safe_state_tokens = state_token_latents.clone()
            safe_mask = state_token_mask.clone()
            empty_rows = has_valid.squeeze(1).logical_not()
            safe_state_tokens[empty_rows] = 0
            safe_mask[empty_rows, 0] = True

        valid_ratio = state_token_mask.float().mean(dim=1, keepdim=True).to(
            dtype=action_tokens.dtype
        )
        valid_scale = has_valid.to(dtype=action_tokens.dtype)[:, :, None]
        state_attention = self.state_token_action_prehead_adapter_attn(
            self.state_token_action_prehead_adapter_norm(action_tokens),
            safe_state_tokens,
            safe_mask,
        )
        state_attention = state_attention * valid_scale
        state_summary = safe_state_tokens.mean(dim=1) * has_valid.to(
            dtype=action_tokens.dtype
        )
        gate_input = torch.cat(
            [
                action_tokens.mean(dim=1),
                state_summary,
                state_attention.mean(dim=1),
                valid_ratio,
            ],
            dim=-1,
        )
        state_gate = torch.sigmoid(
            self.state_token_action_prehead_adapter_gate(gate_input)
        )
        return state_attention, state_gate, valid_ratio

    def _read_late_cue_action_prehead_adapter(
        self,
        action_tokens: torch.Tensor,
        late_cue_anchor_latents: torch.Tensor,
        late_cue_anchor_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, anchor_len, anchor_token_num, hidden_dim = (
            late_cue_anchor_latents.shape
        )
        anchor_tokens = einops.rearrange(
            late_cue_anchor_latents,
            "b anchor_len token_num hidden_dim -> b (anchor_len token_num) hidden_dim",
        )
        if late_cue_anchor_mask is None:
            anchor_mask = torch.ones(
                (batch_size, anchor_len),
                device=late_cue_anchor_latents.device,
                dtype=torch.bool,
            )
        else:
            anchor_mask = late_cue_anchor_mask.bool()
        token_mask = (
            anchor_mask[:, :, None]
            .expand(batch_size, anchor_len, anchor_token_num)
            .reshape(batch_size, anchor_len * anchor_token_num)
        )

        safe_anchor_tokens = anchor_tokens
        safe_mask = token_mask
        has_valid = token_mask.any(dim=1, keepdim=True)
        if has_valid.logical_not().any():
            safe_anchor_tokens = anchor_tokens.clone()
            safe_mask = token_mask.clone()
            empty_rows = has_valid.squeeze(1).logical_not()
            safe_anchor_tokens[empty_rows] = 0
            safe_mask[empty_rows, 0] = True

        valid_ratio = token_mask.float().mean(dim=1, keepdim=True).to(
            dtype=action_tokens.dtype
        )
        valid_scale = has_valid.to(dtype=action_tokens.dtype)[:, :, None]
        cue_attention = self.late_cue_action_prehead_adapter_attn(
            self.late_cue_action_prehead_adapter_norm(action_tokens),
            safe_anchor_tokens,
            safe_mask,
        )
        cue_attention = cue_attention * valid_scale
        masked_anchor_tokens = safe_anchor_tokens * safe_mask[:, :, None].to(
            dtype=action_tokens.dtype
        )
        anchor_count = safe_mask.sum(dim=1, keepdim=True).clamp(min=1).to(
            dtype=action_tokens.dtype
        )
        cue_summary = masked_anchor_tokens.sum(dim=1) / anchor_count
        cue_summary = cue_summary * has_valid.to(dtype=action_tokens.dtype)
        gate_input = torch.cat(
            [
                action_tokens.mean(dim=1),
                cue_summary,
                cue_attention.mean(dim=1),
                valid_ratio,
            ],
            dim=-1,
        )
        cue_gate = torch.sigmoid(self.late_cue_action_prehead_adapter_gate(gate_input))
        return cue_attention, cue_gate, valid_ratio

    def _get_results(
        self,
        input_dict: dict[str, torch.Tensor],
        disable_state_token_action_adapters: bool = False,
    ):
        token_num = input_dict["x"].shape[1]
        action_token_start_idx = token_num - self.action_token_num
        action_tokens = input_dict["x"][:, action_token_start_idx:token_num, :]

        if (
            self.late_cue_action_prehead_adapter_enabled
            and "late_cue_anchor_action_latents" in input_dict
            and not disable_state_token_action_adapters
        ):
            cue_attention, cue_gate, cue_valid_ratio = (
                self._read_late_cue_action_prehead_adapter(
                    action_tokens,
                    input_dict["late_cue_anchor_action_latents"],
                    input_dict.get("late_cue_anchor_action_mask"),
                )
            )
            cue_token_delta = self.late_cue_action_prehead_adapter_out(cue_attention)
            action_tokens = (
                action_tokens
                + self.late_cue_action_prehead_adapter_residual_scale
                * cue_gate[:, None, :]
                * cue_token_delta
            )

            if len(self.record_data_entries) > 0:
                record_data_dict = self.recorded_data_dict
                if "late_cue_action_prehead_adapter_gate" in record_data_dict:
                    record_data_dict[
                        "late_cue_action_prehead_adapter_gate"
                    ].append(cue_gate.detach().clone())
                if (
                    "late_cue_action_prehead_adapter_valid_ratio"
                    in record_data_dict
                ):
                    record_data_dict[
                        "late_cue_action_prehead_adapter_valid_ratio"
                    ].append(cue_valid_ratio.detach().clone())
                if "late_cue_action_prehead_adapter_delta_norm" in record_data_dict:
                    record_data_dict[
                        "late_cue_action_prehead_adapter_delta_norm"
                    ].append(
                        cue_token_delta.detach().norm(dim=-1).mean(
                            dim=1, keepdim=True
                        ).clone()
                    )

        if (
            self.state_token_action_prehead_adapter_enabled
            and "state_token_latents" in input_dict
            and not disable_state_token_action_adapters
        ):
            state_attention, state_gate, state_valid_ratio = (
                self._read_state_token_action_prehead_adapter(
                    action_tokens,
                    input_dict["state_token_latents"],
                    input_dict.get("state_token_mask"),
                )
            )
            action_token_delta = self.state_token_action_prehead_adapter_out(
                state_attention
            )
            action_tokens = (
                action_tokens
                + self.state_token_action_prehead_adapter_residual_scale
                * state_gate[:, None, :]
                * action_token_delta
            )

            if len(self.record_data_entries) > 0:
                record_data_dict = self.recorded_data_dict
                if "state_token_action_prehead_adapter_gate" in record_data_dict:
                    record_data_dict[
                        "state_token_action_prehead_adapter_gate"
                    ].append(state_gate.detach().clone())
                if (
                    "state_token_action_prehead_adapter_valid_ratio"
                    in record_data_dict
                ):
                    record_data_dict[
                        "state_token_action_prehead_adapter_valid_ratio"
                    ].append(state_valid_ratio.detach().clone())
                if "state_token_action_prehead_adapter_delta_norm" in record_data_dict:
                    record_data_dict[
                        "state_token_action_prehead_adapter_delta_norm"
                    ].append(
                        action_token_delta.detach().norm(dim=-1).mean(
                            dim=1, keepdim=True
                        ).clone()
                    )

        action = self.action_final_layer.forward(action_tokens)

        if (
            self.state_token_action_adapter_enabled
            and "state_token_latents" in input_dict
            and not disable_state_token_action_adapters
        ):
            state_attention, state_gate, state_valid_ratio = (
                self._read_state_token_action_adapter(
                    action_tokens,
                    input_dict["state_token_latents"],
                    input_dict.get("state_token_mask"),
                )
            )
            action_delta = self.state_token_action_adapter_out(state_attention)
            action = (
                action
                + self.state_token_action_adapter_residual_scale
                * state_gate[:, None, :]
                * action_delta
            )

            if len(self.record_data_entries) > 0:
                record_data_dict = self.recorded_data_dict
                if "state_token_action_adapter_gate" in record_data_dict:
                    record_data_dict["state_token_action_adapter_gate"].append(
                        state_gate.detach().clone()
                    )
                if "state_token_action_adapter_valid_ratio" in record_data_dict:
                    record_data_dict[
                        "state_token_action_adapter_valid_ratio"
                    ].append(state_valid_ratio.detach().clone())
                if "state_token_action_adapter_delta_norm" in record_data_dict:
                    record_data_dict["state_token_action_adapter_delta_norm"].append(
                        action_delta.detach().norm(dim=-1).mean(
                            dim=1, keepdim=True
                        ).clone()
                    )

        return {"action": action}

    def forward(
        self,
        data_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        data_dict = copy.deepcopy(data_dict)  # Avoid modifying the original data_dict

        input_dict = self._project_to_latent_space(data_dict)
        input_dict = self._run_blocks(input_dict)
        return self._get_results(input_dict)
