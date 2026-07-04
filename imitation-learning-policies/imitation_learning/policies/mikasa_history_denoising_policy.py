from __future__ import annotations

import copy
import os
from functools import partial
from typing import Any, cast

import einops
import torch
import torch.nn.functional as F

try:
    from imitation_learning.common.datatypes import batch_type
    from imitation_learning.models.common.memory_gate import MemoryGate
    from imitation_learning.models.denoising_networks.mikasa_memory_transformer import (
        MikasaMemoryTransformer,
    )
    from imitation_learning.models.encoders.multi_token_encoder import MultiTokenEncoder
    from imitation_learning.policies.base_denoising_policy import BaseDenoisingPolicy
    from torch._dynamo.eval_frame import OptimizedModule
    from robot_utils.data_utils import dict_apply
    from robot_utils.torch_utils import aggregate_batch, split_batch
    import cv2
except ModuleNotFoundError as _IMPORT_ERROR:
    batch_type = dict

    class _UnavailableDependency:
        pass

    class BaseDenoisingPolicy:
        def __init__(self, *args, **kwargs):
            raise _IMPORT_ERROR

    MemoryGate = _UnavailableDependency
    MikasaMemoryTransformer = _UnavailableDependency
    MultiTokenEncoder = _UnavailableDependency
    OptimizedModule = _UnavailableDependency

    def dict_apply(*args, **kwargs):
        raise _IMPORT_ERROR

    def aggregate_batch(*args, **kwargs):
        raise _IMPORT_ERROR

    def split_batch(*args, **kwargs):
        raise _IMPORT_ERROR

    cv2 = None
import numpy as np
import time


def _stack_recorded_data_dict(
    recorded_data_dict: dict[str, list[torch.Tensor]],
) -> dict[str, torch.Tensor] | None:
    """Stack complete diagnostic records and skip incomplete no-history steps."""
    if any(len(values) == 0 for values in recorded_data_dict.values()):
        return None
    return {
        key: torch.stack(values, dim=1).detach()
        for key, values in recorded_data_dict.items()
    }


def _validate_burn_in_loss_traj_num(burn_in_loss_traj_num: int) -> int:
    if burn_in_loss_traj_num < 0:
        raise ValueError("burn_in_loss_traj_num must be non-negative")
    return burn_in_loss_traj_num


def _validate_burn_in_start_id(burn_in_start_id: int) -> int:
    if burn_in_start_id < 0:
        raise ValueError("burn_in_start_id must be non-negative")
    return burn_in_start_id


def _apply_burn_in_loss_mask(
    loss_valid_mask: torch.Tensor,
    traj_idx: torch.Tensor,
    burn_in_start_id: int,
    burn_in_loss_traj_num: int,
    training_traj_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    if burn_in_loss_traj_num <= 0:
        return loss_valid_mask

    if traj_idx is None:
        raise ValueError("conditional burn-in requires traj_idx or start index")

    sample_start_indices = traj_idx.to(device=loss_valid_mask.device)
    if sample_start_indices.dim() == 1:
        sample_start_indices = sample_start_indices[:, None]
    else:
        sample_start_indices = sample_start_indices.reshape(
            sample_start_indices.shape[0], -1
        )[:, :1]

    if training_traj_indices is None:
        traj_indices = torch.arange(
            loss_valid_mask.shape[1],
            device=loss_valid_mask.device,
        )[None, :].expand(loss_valid_mask.shape[0], -1)
    else:
        traj_indices = training_traj_indices.to(device=loss_valid_mask.device)
        if traj_indices.dim() == 1:
            traj_indices = traj_indices[None, :].expand(loss_valid_mask.shape[0], -1)

    late_start_mask = sample_start_indices > burn_in_start_id
    burn_in_valid_mask = (~late_start_mask) | (
        traj_indices >= burn_in_loss_traj_num
    )
    return loss_valid_mask & burn_in_valid_mask


class MikasaHistoryDenoisingPolicy(BaseDenoisingPolicy):
    def __init__(
        self,
        skip_memory: bool,
        history_mask_max_prob: float,
        history_img_feature_encoder: MultiTokenEncoder | None,
        action_no_error_range: tuple[int, int],
        train_history_action_noise_level: str,
        eval_history_action_noise_level: str,
        history_action_num_per_chunk: int,
        history_evidence_feature_encoder: MultiTokenEncoder | None = None,
        # add_noise_to_history_img_features: bool,
        memory_gate: MemoryGate | None = None,
        max_training_traj_num: int = -1,
        training_traj_sampling_strategy: str = "random",
        burn_in_loss_traj_num: int = 0,
        burn_in_start_id: int = 0,
        history_contrastive_loss_weight: float = 0.0,
        history_contrastive_margin: float = 0.0,
        history_contrastive_mode: str = "all_history",
        history_contrastive_detach_correct_loss: bool = True,
        summary_only_aux_loss_weight: float = 0.0,
        summary_contrastive_loss_weight: float = 0.0,
        summary_contrastive_margin: float = 0.0,
        summary_contrastive_detach_correct_loss: bool = True,
        state_token_contrastive_loss_weight: float = 0.0,
        state_token_contrastive_margin: float = 0.0,
        state_token_contrastive_detach_correct_loss: bool = True,
        state_token_action_grounding_loss_weight: float = 0.0,
        state_token_action_grounding_aux_loss_weight: float = 0.0,
        state_token_action_grounding_margin: float = 0.0,
        state_token_action_grounding_detach_correct_loss: bool = True,
        state_token_action_grounding_action_delta_quantile: float = 0.0,
        state_token_action_grounding_action_delta_min: float = 0.0,
        state_token_action_grounding_min_traj_index: int = 1,
        state_token_action_grounding_include_critical: bool = False,
        anchor_action_delta_loss_weight: float = 0.0,
        anchor_action_delta_cosine_loss_weight: float = 0.0,
        anchor_action_delta_action_delta_quantile: float = 0.0,
        anchor_action_delta_action_delta_min: float = 0.0,
        anchor_action_delta_min_traj_index: int = 1,
        anchor_action_delta_include_critical: bool = False,
        decision_window_contrastive_loss_weight: float = 0.0,
        decision_window_contrastive_margin: float = 0.0,
        decision_window_contrastive_detach_correct_loss: bool = True,
        decision_window_invariance_loss_weight: float = 0.0,
        decision_window_invariance_detach_correct: bool = True,
        functional_preservation_loss_weight: float = 0.0,
        functional_preservation_detach_reference: bool = True,
        teacher_preservation_loss_weight: float = 0.0,
        teacher_preservation_disable_action_adapters: bool = True,
        decision_window_contrastive_memory_source: str = "state_token",
        decision_window_action_delta_quantile: float = 0.0,
        decision_window_action_delta_min: float = 0.0,
        decision_window_min_traj_index: int = 1,
        decision_window_include_critical: bool = False,
        state_token_pre_action_obs_update_enabled: bool = False,
        **kwargs,
    ):

        if history_img_feature_encoder is not None:
            kwargs["denoising_network_partial"] = partial(
                kwargs["denoising_network_partial"],
                history_img_features_dim=history_img_feature_encoder.feature_dim,
                history_img_features_token_num=history_img_feature_encoder.token_num,
                history_action_num_per_chunk=history_action_num_per_chunk,
            )
        if memory_gate is not None and isinstance(memory_gate, MemoryGate):
            kwargs["denoising_network_partial"] = partial(
                kwargs["denoising_network_partial"],
            )

        super().__init__(**kwargs)

        self.action_no_error_range: tuple[int, ...] = tuple(action_no_error_range)
        """
        If any training action between the two values is error, will not backprop the loss
        """
        assert self.action_no_error_range[1] > self.action_no_error_range[0] >= 0


        self.history_noisy_actions_dict: dict[int, list[list[torch.Tensor]]] = {}
        """
        episode_idx -> list [num_history] of lists [num_inference_steps] of tensors [noisy_history_action, shape: (action_length, action_dim)]
        history_mask_max_prob: [0, 1], higher means more history latents will be masked
        Since the action diffusion is not in latent space, we directly store the noisy action as latents
        In the future, we can also store image features/latents in the buffer
        """
        self.history_img_features_dict: dict[int, list[torch.Tensor]] = {}
        """
        episode_idx -> list [num_history] of tensors [history_img_features, shape: (history_img_features_length, img_length*history_img_features_token_num, history_img_features_dim)]
        """
        self.history_evidence_features_dict: dict[int, list[torch.Tensor]] = {}
        """
        episode_idx -> list [num_history] of tensors used only for evidence
        selection, e.g. frozen LongCLIP frame embeddings.
        """
        self.late_cue_anchor_img_features_dict: dict[int, list[torch.Tensor]] = {}
        """
        episode_idx -> first K observed image feature tensors for the late-cue
        adapter anchor. This buffer is not pruned with the rolling history.
        """
        self.visual_memory_carrier_img_features_dict: dict[int, list[torch.Tensor]] = {}
        """
        episode_idx -> prefix image feature tensors for learned visual memory
        carriers. This buffer is capped by the carrier max length, not by the
        rolling history window.
        """
        self.state_token_dict: dict[int, torch.Tensor] = {}
        """
        episode_idx -> recurrent state tokens. By default they are updated after
        the current action chunk has been generated; optionally, the current
        observation can be written before action generation.
        """
        self.state_token_seen_dict: dict[int, bool] = {}
        self._state_token_read_mask_override_announced = False
        self.state_token_pre_action_obs_update_enabled: bool = (
            state_token_pre_action_obs_update_enabled
        )
        assert (
            0 <= history_mask_max_prob <= 1
        ), f"history_mask_max_prob must be in [0, 1], but got {history_mask_max_prob}"
        self.history_mask_max_prob: float = history_mask_max_prob

        if not isinstance(memory_gate, MemoryGate):
            self.memory_gate: MemoryGate | None = None
            print(f"No memory gate provided. Got {memory_gate}")
        else:
            self.memory_gate: MemoryGate | None = memory_gate

        self.skip_memory: bool = skip_memory
        """
        If True, will not pass history latents to the denoising network
        """
        if skip_memory:
            self.enable_skip_memory()

        self.recorded_data_dicts: dict[int, list[dict[str, torch.Tensor]]] = {}
        """
        episode_idx -> list of dicts
        each item in the list:
            "history_cross_attention": (diffusion_step_num, transformer_layer_num, head_num, token_num, history_len*token_num)
            "memory_gate_val": (diffusion_step_num, transformer_layer_num, input_token_num)
        """

        self.history_img_feature_encoder: MultiTokenEncoder | None = (
            history_img_feature_encoder
        )
        self.history_evidence_feature_encoder: MultiTokenEncoder | None = (
            history_evidence_feature_encoder
        )
        self.qframe_query_mode = os.environ.get(
            "MIKASA_EVAL_QFRAME_QUERY_MODE",
            "image_only",
        ).strip().lower()
        if self.qframe_query_mode in {"", "off"}:
            self.qframe_query_mode = "image_only"
        if self.qframe_query_mode not in {
            "image_only",
            "text_only",
            "image_text_fused",
        }:
            raise ValueError(
                "MIKASA_EVAL_QFRAME_QUERY_MODE must be image_only, text_only, "
                f"or image_text_fused, got {self.qframe_query_mode!r}"
            )
        self.qframe_text_instruction = os.environ.get(
            "MIKASA_EVAL_QFRAME_TEXT_INSTRUCTION",
            "Observe the cube's color, wait, then touch the cube of the same color.",
        ).strip()
        self.qframe_text_alpha = float(
            os.environ.get("MIKASA_EVAL_QFRAME_TEXT_ALPHA", "0.5")
        )
        if not 0.0 <= self.qframe_text_alpha <= 1.0:
            raise ValueError("MIKASA_EVAL_QFRAME_TEXT_ALPHA must be in [0, 1]")
        self.qframe_longclip_weights_path = os.environ.get(
            "MIKASA_EVAL_QFRAME_LONGCLIP_WEIGHTS",
            "/mnt/3fs1/data/tingwen.du/icra_method_dev/deps/Long-CLIP/longclip-L.pt",
        ).strip()
        self._qframe_text_encoder_holder: list[object | None] = [None]

        self.history_action_num_per_chunk: int = history_action_num_per_chunk
        """
        Number of history actions to be stored in the buffer. Should be the number of executed actions in one chunk.
        """

        assert train_history_action_noise_level in ["last_step", "none", "random"]
        assert eval_history_action_noise_level in ["last_step", "none", "random"]
        self.train_history_action_noise_level: str = train_history_action_noise_level
        self.eval_history_action_noise_level: str = eval_history_action_noise_level
        """
        last_step: use one-step less noisy history action as condition
        none: use clean history action as condition
        random: use random noise levey history action as condition
        """


        self.max_training_traj_num: int = max_training_traj_num
        """
        Maximum number of trajectories to be used for training. If -1, will use all the trajectories. Otherwise, will sample a subset of trajectories.
        This is used when there are too many trajectories in the dataloader (say 150+) to save memory.
        """
        self.training_traj_sampling_strategy: str = training_traj_sampling_strategy
        assert self.training_traj_sampling_strategy in ("random", "tail"), (
            "training_traj_sampling_strategy must be 'random' or 'tail', "
            f"got {self.training_traj_sampling_strategy}"
        )
        self.burn_in_loss_traj_num: int = _validate_burn_in_loss_traj_num(
            burn_in_loss_traj_num
        )
        self.burn_in_start_id: int = _validate_burn_in_start_id(burn_in_start_id)
        self.history_contrastive_loss_weight: float = history_contrastive_loss_weight
        assert (
            self.history_contrastive_loss_weight >= 0.0
        ), "history_contrastive_loss_weight must be non-negative"
        self.history_contrastive_margin: float = history_contrastive_margin
        assert (
            self.history_contrastive_margin >= 0.0
        ), "history_contrastive_margin must be non-negative"
        self.history_contrastive_mode: str = history_contrastive_mode
        assert self.history_contrastive_mode in (
            "anchor_only",
            "rolling_history_only",
            "all_history",
        ), (
            "history_contrastive_mode must be one of "
            "'anchor_only', 'rolling_history_only', or 'all_history', "
            f"got {self.history_contrastive_mode}"
        )
        self.history_contrastive_detach_correct_loss: bool = (
            history_contrastive_detach_correct_loss
        )
        self.summary_only_aux_loss_weight: float = summary_only_aux_loss_weight
        assert (
            self.summary_only_aux_loss_weight >= 0.0
        ), "summary_only_aux_loss_weight must be non-negative"
        self.summary_contrastive_loss_weight: float = summary_contrastive_loss_weight
        assert (
            self.summary_contrastive_loss_weight >= 0.0
        ), "summary_contrastive_loss_weight must be non-negative"
        self.summary_contrastive_margin: float = summary_contrastive_margin
        assert (
            self.summary_contrastive_margin >= 0.0
        ), "summary_contrastive_margin must be non-negative"
        self.summary_contrastive_detach_correct_loss: bool = (
            summary_contrastive_detach_correct_loss
        )
        self.state_token_contrastive_loss_weight: float = (
            state_token_contrastive_loss_weight
        )
        assert (
            self.state_token_contrastive_loss_weight >= 0.0
        ), "state_token_contrastive_loss_weight must be non-negative"
        self.state_token_contrastive_margin: float = state_token_contrastive_margin
        assert (
            self.state_token_contrastive_margin >= 0.0
        ), "state_token_contrastive_margin must be non-negative"
        self.state_token_contrastive_detach_correct_loss: bool = (
            state_token_contrastive_detach_correct_loss
        )
        self.state_token_action_grounding_loss_weight: float = (
            state_token_action_grounding_loss_weight
        )
        assert (
            self.state_token_action_grounding_loss_weight >= 0.0
        ), "state_token_action_grounding_loss_weight must be non-negative"
        self.state_token_action_grounding_aux_loss_weight: float = (
            state_token_action_grounding_aux_loss_weight
        )
        assert (
            self.state_token_action_grounding_aux_loss_weight >= 0.0
        ), "state_token_action_grounding_aux_loss_weight must be non-negative"
        self.state_token_action_grounding_margin: float = (
            state_token_action_grounding_margin
        )
        assert (
            self.state_token_action_grounding_margin >= 0.0
        ), "state_token_action_grounding_margin must be non-negative"
        self.state_token_action_grounding_detach_correct_loss: bool = (
            state_token_action_grounding_detach_correct_loss
        )
        self.state_token_action_grounding_action_delta_quantile: float = (
            state_token_action_grounding_action_delta_quantile
        )
        assert 0.0 <= self.state_token_action_grounding_action_delta_quantile < 1.0, (
            "state_token_action_grounding_action_delta_quantile must be in [0, 1)"
        )
        self.state_token_action_grounding_action_delta_min: float = (
            state_token_action_grounding_action_delta_min
        )
        assert (
            self.state_token_action_grounding_action_delta_min >= 0.0
        ), "state_token_action_grounding_action_delta_min must be non-negative"
        self.state_token_action_grounding_min_traj_index: int = (
            state_token_action_grounding_min_traj_index
        )
        assert (
            self.state_token_action_grounding_min_traj_index >= 0
        ), "state_token_action_grounding_min_traj_index must be non-negative"
        self.state_token_action_grounding_include_critical: bool = (
            state_token_action_grounding_include_critical
        )
        self.anchor_action_delta_loss_weight: float = (
            anchor_action_delta_loss_weight
        )
        assert (
            self.anchor_action_delta_loss_weight >= 0.0
        ), "anchor_action_delta_loss_weight must be non-negative"
        self.anchor_action_delta_cosine_loss_weight: float = (
            anchor_action_delta_cosine_loss_weight
        )
        assert (
            self.anchor_action_delta_cosine_loss_weight >= 0.0
        ), "anchor_action_delta_cosine_loss_weight must be non-negative"
        self.anchor_action_delta_action_delta_quantile: float = (
            anchor_action_delta_action_delta_quantile
        )
        assert 0.0 <= self.anchor_action_delta_action_delta_quantile < 1.0, (
            "anchor_action_delta_action_delta_quantile must be in [0, 1)"
        )
        self.anchor_action_delta_action_delta_min: float = (
            anchor_action_delta_action_delta_min
        )
        assert (
            self.anchor_action_delta_action_delta_min >= 0.0
        ), "anchor_action_delta_action_delta_min must be non-negative"
        self.anchor_action_delta_min_traj_index: int = (
            anchor_action_delta_min_traj_index
        )
        assert (
            self.anchor_action_delta_min_traj_index >= 0
        ), "anchor_action_delta_min_traj_index must be non-negative"
        self.anchor_action_delta_include_critical: bool = (
            anchor_action_delta_include_critical
        )
        self.decision_window_contrastive_loss_weight: float = (
            decision_window_contrastive_loss_weight
        )
        assert (
            self.decision_window_contrastive_loss_weight >= 0.0
        ), "decision_window_contrastive_loss_weight must be non-negative"
        self.decision_window_contrastive_margin: float = (
            decision_window_contrastive_margin
        )
        assert (
            self.decision_window_contrastive_margin >= 0.0
        ), "decision_window_contrastive_margin must be non-negative"
        self.decision_window_contrastive_detach_correct_loss: bool = (
            decision_window_contrastive_detach_correct_loss
        )
        self.decision_window_invariance_loss_weight: float = (
            decision_window_invariance_loss_weight
        )
        assert (
            self.decision_window_invariance_loss_weight >= 0.0
        ), "decision_window_invariance_loss_weight must be non-negative"
        self.decision_window_invariance_detach_correct: bool = (
            decision_window_invariance_detach_correct
        )
        self.functional_preservation_loss_weight: float = (
            functional_preservation_loss_weight
        )
        assert (
            self.functional_preservation_loss_weight >= 0.0
        ), "functional_preservation_loss_weight must be non-negative"
        self.functional_preservation_detach_reference: bool = (
            functional_preservation_detach_reference
        )
        self.teacher_preservation_loss_weight: float = (
            teacher_preservation_loss_weight
        )
        assert (
            self.teacher_preservation_loss_weight >= 0.0
        ), "teacher_preservation_loss_weight must be non-negative"
        self.teacher_preservation_disable_action_adapters: bool = (
            teacher_preservation_disable_action_adapters
        )
        self.__dict__["_teacher_preservation_denoising_network"] = None
        self.decision_window_contrastive_memory_source: str = (
            decision_window_contrastive_memory_source
        )
        assert self.decision_window_contrastive_memory_source in (
            "history",
            "state_token",
            "history_and_state_token",
            "state_token_history",
        ), (
            "decision_window_contrastive_memory_source must be one of "
            "'history', 'state_token', 'history_and_state_token', or "
            f"'state_token_history', got {self.decision_window_contrastive_memory_source}"
        )
        self.decision_window_action_delta_quantile: float = (
            decision_window_action_delta_quantile
        )
        assert 0.0 <= self.decision_window_action_delta_quantile < 1.0, (
            "decision_window_action_delta_quantile must be in [0, 1)"
        )
        self.decision_window_action_delta_min: float = decision_window_action_delta_min
        assert (
            self.decision_window_action_delta_min >= 0.0
        ), "decision_window_action_delta_min must be non-negative"
        self.decision_window_min_traj_index: int = decision_window_min_traj_index
        assert (
            self.decision_window_min_traj_index >= 0
        ), "decision_window_min_traj_index must be non-negative"
        self.decision_window_include_critical: bool = decision_window_include_critical

    def enable_skip_memory(self):
        self.skip_memory = True
        print(f"Setting skip_memory to {self.skip_memory}")
        for key, params in self.denoising_network.named_parameters():
            if "history" in key:
                if self.skip_memory:
                    params.requires_grad = False

        if self.memory_gate is not None:
            for key, params in self.memory_gate.named_parameters():
                if self.skip_memory:
                    params.requires_grad = False

    # ================================ Inference =================================

    def predict_action(
        self,
        normalized_batch: batch_type,
    ) -> batch_type:
        """
        Single-trajectory input:
            normalized_batch:
                "robot0_wrist_camera": (batch_size, traj_length, 3, image_size, image_size)
                "robot0_10d": (batch_size, traj_length, 8)
                "episode_idx": (batch_size,)
            return:
                "action0_10d": (batch_size, traj_length, 8)

        Multi-trajectory input: (only available when skip_memory is False)
            normalized_batch:
                "robot0_wrist_camera": (batch_size, traj_num, traj_length, 3, image_size, image_size)
                "robot0_10d": (batch_size, traj_num, traj_length, 8)
            return:
                "action0_10d": (batch_size, traj_num, traj_length, 8)
        """

        meta = next(iter(self.global_cond_encoder.cond_meta.values()))
        if meta.name not in normalized_batch:
            input_shape = normalized_batch[f"{meta.name}_feature"].shape
            expected_shape = torch.Size([self.global_cond_encoder.feature_dim])
        else:
            input_shape = normalized_batch[meta.name].shape
            expected_shape = meta.shape

        if self.skip_memory:
            assert (
                len(input_shape) - len(expected_shape) == 2
            ), "Please make sure you are using single-trajectory dataset when skip_memory is True"
            return super().predict_action(normalized_batch)

        if len(input_shape) - len(expected_shape) == 2:
            return self.predict_single_traj(normalized_batch)

        elif len(input_shape) - len(expected_shape) == 3:
            return self.predict_multi_traj(normalized_batch)

        else:
            raise ValueError(
                f"Unexpected input shape: {input_shape}, expected shape: {expected_shape}"
            )

    def _apply_eval_state_token_read_mask_override(
        self,
        state_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        mode = os.environ.get("MIKASA_EVAL_STATE_TOKEN_READ_MASK", "").strip().lower()
        if mode in {"", "none", "off", "false", "0"}:
            return state_token_mask

        if not self._state_token_read_mask_override_announced:
            print(f"[mikasa-eval-overrides] state_token_read_mask={mode}")
            self._state_token_read_mask_override_announced = True

        if mode == "empty":
            return torch.zeros_like(state_token_mask)
        if mode.startswith("prefix"):
            raw_prefix = mode.removeprefix("prefix")
            if raw_prefix == "":
                raise ValueError("MIKASA_EVAL_STATE_TOKEN_READ_MASK=prefix requires N")
            prefix = int(raw_prefix)
            if prefix <= 0:
                raise ValueError("state token read-mask prefix must be positive")
            masked = torch.zeros_like(state_token_mask)
            prefix = min(prefix, state_token_mask.shape[1])
            masked[:, :prefix] = state_token_mask[:, :prefix]
            return masked
        if mode.startswith("drop_prefix"):
            raw_prefix = mode.removeprefix("drop_prefix")
            if raw_prefix == "":
                raise ValueError(
                    "MIKASA_EVAL_STATE_TOKEN_READ_MASK=drop_prefix requires N"
                )
            prefix = int(raw_prefix)
            if prefix <= 0:
                raise ValueError("state token read-mask prefix must be positive")
            masked = state_token_mask.clone()
            prefix = min(prefix, state_token_mask.shape[1])
            masked[:, :prefix] = False
            return masked
        raise ValueError(
            "MIKASA_EVAL_STATE_TOKEN_READ_MASK must be empty/off/prefixN/"
            f"drop_prefixN/empty, got {mode!r}"
        )

    def _encode_current_history_img_features(
        self,
        normalized_batch: batch_type,
        batch_size: int,
    ) -> torch.Tensor:
        assert self.history_img_feature_encoder is not None
        img_dict = {}
        for k in self.history_img_feature_encoder.data_entry_names:
            if k in normalized_batch:
                img_dict[k] = normalized_batch[k]
            elif f"{k}_feature" in normalized_batch:
                img_dict[f"{k}_feature"] = normalized_batch[f"{k}_feature"]
            else:
                raise ValueError(f"Key {k} not found in normalized_batch")
        img_features = self.history_img_feature_encoder.forward(img_dict)
        return img_features.reshape(
            batch_size,
            -1,
            self.history_img_feature_encoder.feature_dim,
        )

    def _raw_image_dict_for_evidence_encoder(
        self,
        normalized_batch: batch_type,
    ) -> dict[str, torch.Tensor]:
        assert self.history_evidence_feature_encoder is not None
        img_dict = {}
        for key in self.history_evidence_feature_encoder.data_entry_names:
            if key not in normalized_batch:
                raise ValueError(
                    f"Raw image key {key} not found in normalized_batch for "
                    "history_evidence_feature_encoder"
                )
            img_dict[key] = normalized_batch[key]
        return img_dict

    def _encode_current_history_evidence_features(
        self,
        normalized_batch: batch_type,
        batch_size: int,
    ) -> torch.Tensor:
        assert self.history_evidence_feature_encoder is not None
        img_features = self.history_evidence_feature_encoder.forward(
            self._raw_image_dict_for_evidence_encoder(normalized_batch)
        )
        img_features = img_features.reshape(
            batch_size,
            -1,
            self.history_evidence_feature_encoder.feature_dim,
        )
        return img_features.mean(dim=1)

    def _qframe_text_query_features(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.qframe_text_instruction == "":
            raise ValueError(
                "MIKASA_EVAL_QFRAME_TEXT_INSTRUCTION must be non-empty when "
                f"MIKASA_EVAL_QFRAME_QUERY_MODE={self.qframe_query_mode}"
            )
        encoder = self._qframe_text_encoder_holder[0]
        if encoder is None:
            from imitation_learning.models.encoders.longclip_text_encoder import (
                LongCLIPTextEncoder,
            )

            encoder = LongCLIPTextEncoder(
                weights_path=self.qframe_longclip_weights_path,
                frozen=True,
            ).to(device)
            encoder.eval()
            self._qframe_text_encoder_holder[0] = encoder
            print(
                "[mikasa-qframe] using text query "
                f"mode={self.qframe_query_mode} alpha={self.qframe_text_alpha} "
                f"instruction={self.qframe_text_instruction!r}",
                flush=True,
            )
        else:
            encoder = encoder.to(device)
        features = encoder.encode_text([self.qframe_text_instruction] * batch_size)
        return features.to(device=device, dtype=dtype)

    def _qframe_evidence_query_features(
        self,
        image_query_features: torch.Tensor,
    ) -> torch.Tensor:
        if self.qframe_query_mode == "image_only":
            return image_query_features
        text_query_features = self._qframe_text_query_features(
            batch_size=image_query_features.shape[0],
            device=image_query_features.device,
            dtype=image_query_features.dtype,
        )
        if self.qframe_query_mode == "text_only":
            return text_query_features
        image_query = F.normalize(image_query_features, dim=-1, eps=1e-6)
        text_query = F.normalize(text_query_features, dim=-1, eps=1e-6)
        return (
            self.qframe_text_alpha * image_query
            + (1.0 - self.qframe_text_alpha) * text_query
        )

    def _encode_training_history_evidence_features(
        self,
        normalized_batch: batch_type,
    ) -> torch.Tensor:
        assert self.history_evidence_feature_encoder is not None
        img_features = self.history_evidence_feature_encoder.forward(
            self._raw_image_dict_for_evidence_encoder(normalized_batch)
        )
        if img_features.dim() == 5:
            img_features = img_features.flatten(start_dim=2, end_dim=3)
        if img_features.dim() != 4:
            raise ValueError(
                "training evidence features must be "
                f"(batch, traj_num, token_num, dim), got {img_features.shape}"
            )
        return img_features.mean(dim=2)

    def predict_single_traj(
        self,
        normalized_batch: batch_type,
    ) -> batch_type:
        """
        normalized_batch:
            "robot0_wrist_camera": (batch_size, traj_length, 3, image_size, image_size)
            "robot0_10d": (batch_size, traj_length, 8)
            "third_person_camera": (batch_size, traj_length, 3, image_size, image_size) # For table-bin scenario
            "episode_idx": (batch_size,)
        return:
            "action0_10d": (batch_size, traj_length, 8)
        """

        assert (
            "action" not in normalized_batch
        ), "Please exclude the batch `action` for evaluation"

        data_dict, _ = self._encode_input_add_noise(normalized_batch, mode="eval")

        if self.memory_gate is not None:
            memory_gate_val = (
                self.memory_gate.get_gate_value(normalized_batch)
            ) # (batch_size, )
            bs = memory_gate_val.shape[0]
            binarized_memory_gate_val = (memory_gate_val > 0.5).bool()

            assert isinstance(self.denoising_network, MikasaMemoryTransformer)

            if not torch.torch.torch.is_grad_enabled() \
                and self.denoising_network.binary_gating \
                and bs == 1 \
                and sum(binarized_memory_gate_val) == 0 \
                and "history_cross_attention" not in self.denoising_network.record_data_entries:
                self.denoising_network.set_skip_history_attn(True)
            else:
                self.denoising_network.set_skip_history_attn(False)

        else:
            memory_gate_val = None

        new_history_action_dict: dict[int, list[torch.Tensor]] = {}
        batch_size, traj_length, action_dim = data_dict["noisy_action"].shape
        assert batch_size == len(
            normalized_batch["episode_idx"]
        ), f"Please make sure the batch size {batch_size} in data_dict['trajectory'].shape: {data_dict['trajectory'].shape}, is the same as the number of episodes {len(normalized_batch['episode_idx'])}"


        if isinstance(self.denoising_network, OptimizedModule):
            # After torch compile: Just fix the type of the denoising network for type checking.
            self.denoising_network = cast(MikasaMemoryTransformer, cast(Any, self.denoising_network))
        else:
            assert isinstance(self.denoising_network, MikasaMemoryTransformer)
        max_history_len: int = self.denoising_network.max_history_len
        use_late_cue_anchor = bool(
            getattr(self.denoising_network, "late_cue_anchor_enabled", False)
        )
        use_visual_memory_carrier = bool(
            getattr(self.denoising_network, "visual_memory_carrier_enabled", False)
        )
        late_cue_anchor_len = int(
            getattr(self.denoising_network, "late_cue_anchor_len", 1)
        )
        visual_memory_carrier_max_len = int(
            getattr(
                self.denoising_network,
                "visual_memory_carrier_max_len",
                max_history_len,
            )
        )
        assert late_cue_anchor_len >= 1, "late_cue_anchor_len must be >= 1"
        assert visual_memory_carrier_max_len >= 1
        use_state_tokens = bool(
            getattr(self.denoising_network, "state_token_memory_enabled", False)
        )

        # history_img_features is invariant to the diffusion step
        if self.history_img_feature_encoder is not None and max_history_len > 0:
            history_img_features = torch.zeros(
                (
                    batch_size,
                    max_history_len,
                    self.history_img_feature_encoder.token_num,
                    self.history_img_feature_encoder.feature_dim,
                ),
                device=self.device,
            )
            for idx, episode_idx in enumerate(normalized_batch["episode_idx"]):
                if int(episode_idx) in self.history_img_features_dict.keys():
                    history_len = len(self.history_img_features_dict[int(episode_idx)])
                    history_img_features[idx, -history_len:] = torch.stack(
                        self.history_img_features_dict[int(episode_idx)], dim=0
                    )
        history_evidence_features = None
        new_history_evidence_features = None
        if self.history_evidence_feature_encoder is not None and max_history_len > 0:
            history_evidence_features = torch.zeros(
                (
                    batch_size,
                    max_history_len,
                    self.history_evidence_feature_encoder.feature_dim,
                ),
                device=self.device,
            )
            for idx, episode_idx in enumerate(normalized_batch["episode_idx"]):
                episode_idx = int(episode_idx)
                if episode_idx in self.history_evidence_features_dict:
                    history_len = len(self.history_evidence_features_dict[episode_idx])
                    history_evidence_features[idx, -history_len:] = torch.stack(
                        self.history_evidence_features_dict[episode_idx],
                        dim=0,
                    )
            new_history_evidence_features = (
                self._encode_current_history_evidence_features(
                    normalized_batch,
                    batch_size,
                )
            )
        late_cue_anchor_img_features = None
        late_cue_anchor_mask = None
        if (
            use_late_cue_anchor
            and self.history_img_feature_encoder is not None
            and max_history_len > 0
        ):
            late_cue_anchor_img_features = torch.zeros(
                (
                    batch_size,
                    late_cue_anchor_len,
                    self.history_img_feature_encoder.token_num,
                    self.history_img_feature_encoder.feature_dim,
                ),
                device=self.device,
            )
            late_cue_anchor_mask = torch.zeros(
                (batch_size, late_cue_anchor_len),
                device=self.device,
                dtype=torch.bool,
            )
            for idx, episode_idx in enumerate(normalized_batch["episode_idx"]):
                episode_idx = int(episode_idx)
                if episode_idx in self.late_cue_anchor_img_features_dict:
                    anchor_features = self.late_cue_anchor_img_features_dict[
                        episode_idx
                    ][:late_cue_anchor_len]
                    anchor_count = len(anchor_features)
                    if anchor_count > 0:
                        late_cue_anchor_img_features[idx, :anchor_count] = (
                            torch.stack(anchor_features, dim=0)
                        )
                        late_cue_anchor_mask[idx, :anchor_count] = True
        visual_memory_carrier_img_features = None
        visual_memory_carrier_mask = None
        if (
            use_late_cue_anchor
            and use_visual_memory_carrier
            and self.history_img_feature_encoder is not None
            and max_history_len > 0
        ):
            visual_memory_carrier_img_features = torch.zeros(
                (
                    batch_size,
                    visual_memory_carrier_max_len,
                    self.history_img_feature_encoder.token_num,
                    self.history_img_feature_encoder.feature_dim,
                ),
                device=self.device,
            )
            visual_memory_carrier_mask = torch.zeros(
                (batch_size, visual_memory_carrier_max_len),
                device=self.device,
                dtype=torch.bool,
            )
            for idx, episode_idx in enumerate(normalized_batch["episode_idx"]):
                episode_idx = int(episode_idx)
                prefix_features = self.visual_memory_carrier_img_features_dict.get(
                    episode_idx,
                    [],
                )[:visual_memory_carrier_max_len]
                prefix_count = len(prefix_features)
                if prefix_count > 0:
                    visual_memory_carrier_img_features[idx, :prefix_count] = (
                        torch.stack(prefix_features, dim=0)
                    )
                    visual_memory_carrier_mask[idx, :prefix_count] = True
        state_token_latents = None
        state_token_mask = None
        state_token_read_mask = None
        if use_state_tokens:
            state_token_latents = self.denoising_network.initial_state_tokens(
                batch_size,
                self.device,
                data_dict["noisy_action"].dtype,
            )
            state_token_mask = torch.zeros(
                (
                    batch_size,
                    self.denoising_network.state_token_num,
                ),
                device=self.device,
                dtype=torch.bool,
            )
            for idx, episode_idx in enumerate(normalized_batch["episode_idx"]):
                episode_idx = int(episode_idx)
                if episode_idx in self.state_token_dict:
                    state_token_latents[idx] = self.state_token_dict[episode_idx].to(
                        device=self.device,
                        dtype=state_token_latents.dtype,
                    )
                    if self.state_token_seen_dict.get(episode_idx, False):
                        state_token_mask[idx] = True
            state_token_read_mask = self._apply_eval_state_token_read_mask_override(
                state_token_mask
            )

        new_history_img_features = None
        if (
            self.state_token_pre_action_obs_update_enabled
            and use_state_tokens
            and state_token_latents is not None
            and self.history_img_feature_encoder is not None
        ):
            new_history_img_features = self._encode_current_history_img_features(
                normalized_batch,
                batch_size,
            )
            pre_action_update_latents = (
                self.denoising_network.build_state_token_update_latents(
                    None,
                    new_history_img_features,
                )
            )
            pre_action_update_mask = torch.ones(
                batch_size,
                device=self.device,
                dtype=torch.bool,
            )
            state_token_latents = self.denoising_network.update_state_tokens(
                state_token_latents,
                pre_action_update_latents,
                pre_action_update_mask,
                state_token_mask=state_token_mask,
            )
            state_token_mask = state_token_mask | pre_action_update_mask[:, None]
            state_token_read_mask = self._apply_eval_state_token_read_mask_override(
                state_token_mask
            )

        recorded_data_dicts: list[dict[str, torch.Tensor]] = []

        for k, t in enumerate(self.noise_scheduler.get_inference_timesteps()):
            if max_history_len > 0:
                history_noisy_actions = torch.zeros(
                    (
                        batch_size,
                        max_history_len,
                        self.history_action_num_per_chunk,
                        action_dim
                    ),
                    device=self.device,
                ) # (batch_size, max_history_len, history_action_num_per_chunk, action_dim)

                history_mask = torch.zeros(
                    (batch_size, max_history_len),
                    device=self.device,
                    dtype=torch.bool,
                )

                # print(f"{self.history_noisy_actions_dict.keys()=}")
                # print(f"{normalized_batch['episode_idx']=}")

                for l, episode_idx in enumerate(normalized_batch["episode_idx"]):
                    if int(episode_idx) in self.history_noisy_actions_dict.keys():

                        if self.eval_history_action_noise_level == "none":
                            diffusion_step_idx = -1
                        elif self.eval_history_action_noise_level == "random":
                            rand_idx = int(torch.randint(0, len(self.noise_scheduler.get_inference_timesteps()), (1,)).item())
                            diffusion_step_idx = rand_idx
                        elif self.eval_history_action_noise_level == "last_step":
                            diffusion_step_idx = k
                        else:
                            raise ValueError(f"Invalid history action noise level: {self.eval_history_action_noise_level}")

                        history_len = len(self.history_noisy_actions_dict[int(episode_idx)])
                        stacked_history_noisy_action = torch.stack(
                            [
                                self.history_noisy_actions_dict[int(episode_idx)][i][
                                    diffusion_step_idx
                                ]
                                for i in range(history_len)
                            ],
                            dim=0,
                        )  # (history_len, token_num, hidden_dim)

                        history_noisy_actions[l, -history_len:] = (
                            stacked_history_noisy_action
                        )

                        history_mask[l, -history_len:] = 1

                # These keys need to be overridden every time before the denoising network is called
                # Since the denoising network will pop the keys after the forward pass
                data_dict["history_noisy_actions"] = history_noisy_actions
                data_dict["history_mask"] = history_mask
            else:
                data_dict.pop("history_noisy_actions", None)
                data_dict.pop("history_mask", None)
            if memory_gate_val is not None:
                data_dict["memory_gate_val"] = memory_gate_val


            data_dict["step"] = (
                torch.ones((batch_size,), device=self.device) * t
            )

            if self.history_img_feature_encoder is not None and max_history_len > 0:
                noise_ratio = t / self.noise_scheduler.train_step_num
                data_dict["history_img_features"] = history_img_features
            if history_evidence_features is not None:
                data_dict["history_evidence_features"] = history_evidence_features
                assert new_history_evidence_features is not None
                data_dict["evidence_query_features"] = (
                    self._qframe_evidence_query_features(new_history_evidence_features)
                )
            if late_cue_anchor_img_features is not None:
                data_dict["late_cue_anchor_img_features"] = late_cue_anchor_img_features
                data_dict["late_cue_anchor_mask"] = late_cue_anchor_mask
            if visual_memory_carrier_img_features is not None:
                data_dict["visual_memory_carrier_img_features"] = (
                    visual_memory_carrier_img_features
                )
                data_dict["visual_memory_carrier_mask"] = visual_memory_carrier_mask
            if state_token_latents is not None:
                data_dict["state_token_latents"] = state_token_latents
                data_dict["state_token_mask"] = (
                    state_token_read_mask
                    if state_token_read_mask is not None
                    else state_token_mask
                )

            # if self.mask_in_eval:
            #     self._add_random_masks(data_dict)

            model_output = self.denoising_network.forward(data_dict)
            if len(self.denoising_network.record_data_entries) > 0:
                # print(f"{self.denoising_network.record_data_entries=}, {self.denoising_network.recorded_data_dict=}")
                merged_data_dict = _stack_recorded_data_dict(
                    self.denoising_network.recorded_data_dict
                )
                if merged_data_dict is not None:
                    recorded_data_dicts.append(copy.deepcopy(merged_data_dict))

            data_dict["noisy_action"] = self.noise_scheduler.step(
                model_output["action"],
                int(t),
                data_dict["noisy_action"],
            )

            for l, episode_idx in enumerate(normalized_batch["episode_idx"]):
                if int(episode_idx) not in new_history_action_dict.keys():
                    new_history_action_dict[int(episode_idx)] = []
                new_history_action_dict[int(episode_idx)].append(
                    data_dict["noisy_action"][l, :self.history_action_num_per_chunk].detach().clone()
                )

        if max_history_len == 0 and not use_state_tokens: # For ablation study
            output = self.action_decoder.forward(data_dict["noisy_action"])
            return output  # (batch_size, traj_length, action_dim)

        # ================================ Update history buffer ================================

        if max_history_len > 0:
            for episode_idx, history_action in new_history_action_dict.items():
                # History: list [num_inference_steps] of tensors [noisy_history_action, shape: (action_length, action_dim)]
                if episode_idx not in self.history_noisy_actions_dict.keys():
                    self.history_noisy_actions_dict[episode_idx] = []
                self.history_noisy_actions_dict[episode_idx].append(history_action)

                while (
                    len(self.history_noisy_actions_dict[episode_idx])
                    > self.denoising_network.max_history_len
                ):
                    self.history_noisy_actions_dict[episode_idx].pop(0)

        if self.history_img_feature_encoder is not None:
            if new_history_img_features is None:
                new_history_img_features = self._encode_current_history_img_features(
                    normalized_batch,
                    batch_size,
                )

            for episode_idx, history_img_features in zip(
                normalized_batch["episode_idx"], new_history_img_features
            ):
                episode_idx = int(episode_idx)
                if (
                    max_history_len > 0
                    and
                    use_late_cue_anchor
                    and len(
                        self.late_cue_anchor_img_features_dict.setdefault(
                            episode_idx, []
                        )
                    )
                    < late_cue_anchor_len
                ):
                    self.late_cue_anchor_img_features_dict[episode_idx].append(
                        history_img_features.detach().clone()
                    )
                if (
                    max_history_len > 0
                    and use_late_cue_anchor
                    and use_visual_memory_carrier
                    and len(
                        self.visual_memory_carrier_img_features_dict.setdefault(
                            episode_idx,
                            [],
                        )
                    )
                    < visual_memory_carrier_max_len
                ):
                    self.visual_memory_carrier_img_features_dict[episode_idx].append(
                        history_img_features.detach().clone()
                    )
                if max_history_len > 0:
                    if episode_idx not in self.history_img_features_dict.keys():
                        self.history_img_features_dict[episode_idx] = []
                    self.history_img_features_dict[episode_idx].append(
                        history_img_features.detach().clone()
                    )

                    while (
                        len(self.history_img_features_dict[episode_idx])
                        > self.denoising_network.max_history_len
                    ):
                        self.history_img_features_dict[episode_idx].pop(0)
        if (
            self.history_evidence_feature_encoder is not None
            and max_history_len > 0
        ):
            if new_history_evidence_features is None:
                new_history_evidence_features = (
                    self._encode_current_history_evidence_features(
                        normalized_batch,
                        batch_size,
                    )
                )
            for episode_idx, evidence_features in zip(
                normalized_batch["episode_idx"],
                new_history_evidence_features,
            ):
                episode_idx = int(episode_idx)
                if episode_idx not in self.history_evidence_features_dict:
                    self.history_evidence_features_dict[episode_idx] = []
                self.history_evidence_features_dict[episode_idx].append(
                    evidence_features.detach().clone()
                )
                while (
                    len(self.history_evidence_features_dict[episode_idx])
                    > self.denoising_network.max_history_len
                ):
                    self.history_evidence_features_dict[episode_idx].pop(0)

        if use_state_tokens and state_token_latents is not None:
            state_update_actions = None
            if getattr(
                self.denoising_network,
                "state_token_update_include_action_history",
                True,
            ):
                state_update_actions = torch.stack(
                    [
                        new_history_action_dict[int(episode_idx)][-1]
                        for episode_idx in normalized_batch["episode_idx"]
                    ],
                    dim=0,
                )
            state_update_img_features = (
                None
                if self.state_token_pre_action_obs_update_enabled
                else new_history_img_features
            )
            updated_state_tokens = state_token_latents
            if state_update_actions is not None or state_update_img_features is not None:
                state_update_latents = (
                    self.denoising_network.build_state_token_update_latents(
                        state_update_actions,
                        state_update_img_features,
                    )
                )
                state_update_mask = torch.ones(
                    batch_size,
                    device=self.device,
                    dtype=torch.bool,
                )
                updated_state_tokens = self.denoising_network.update_state_tokens(
                    state_token_latents,
                    state_update_latents,
                    state_update_mask,
                    state_token_mask=state_token_mask,
                )
            for idx, episode_idx in enumerate(normalized_batch["episode_idx"]):
                episode_idx = int(episode_idx)
                self.state_token_dict[episode_idx] = (
                    updated_state_tokens[idx].detach().clone()
                )
                self.state_token_seen_dict[episode_idx] = True


        if len(recorded_data_dicts) > 0:
            merged_data_dict: dict[str, torch.Tensor] = {}
            merged_data_dict = aggregate_batch(
                recorded_data_dicts, partial(torch.stack, dim=1)
            )
            # "history_cross_attention": (batch_size, diffusion_step_num, transformer_layer_num, head_num, token_num, history_len*token_num)
            splitted_data_dicts = split_batch(
                merged_data_dict, partial(torch.unbind, dim=0)
            )
            for k, splitted_data_dict in enumerate(splitted_data_dicts):
                episode_idx = normalized_batch["episode_idx"][k]
                splitted_data_dict = dict_apply(
                    splitted_data_dict, lambda x: x.detach().clone().cpu()
                )
                if int(episode_idx) not in self.recorded_data_dicts.keys():
                    self.recorded_data_dicts[int(episode_idx)] = []
                self.recorded_data_dicts[int(episode_idx)].append(splitted_data_dict)

        output = self.action_decoder.forward(data_dict["noisy_action"])

        return output  # (batch_size, traj_length, action_dim)

    def predict_multi_traj(
        self,
        normalized_batch: batch_type,
    ) -> batch_type:
        """
        Used when the batch contains multiple trajectories in the same episode.
        This function is only used when running validation with multiple ground-truth trajectories.

        normalized_batch:
            "robot0_wrist_camera": (batch_size, traj_num, traj_length, 3, image_size, image_size)
            "robot0_10d": (batch_size, traj_num, traj_length, 8)
            "third_person_camera": (batch_size, traj_num, traj_length, 3, image_size, image_size) # For table-bin scenario
            "episode_idx": (batch_size) # Need to be overridden
        return:
            "action0_10d": (batch_size, traj_num, traj_length, 8)
        """
        traj_num_dim_idx = 1  # batch_size is 0

        # Use single trajectory prediction to iteratively predict all trajectories
        self.reset()
        actions: list[dict[str, torch.Tensor]] = []
        if "variance_temperature" in normalized_batch:
            normalized_batch.pop(
                "variance_temperature"
            )  # Remove variance_temperature from meta

        batch_size = normalized_batch["episode_idx"].shape[0]
        traj_num = normalized_batch["episode_idx"].shape[1]


        for batch in split_batch(
            normalized_batch,
            partial(torch.unbind, dim=traj_num_dim_idx),
        ):  # Along the traj_num dimension
            """
            batch:
                "robot0_wrist_camera": (batch_size, traj_length, 3, image_size, image_size)
                "robot0_10d": (batch_size, traj_length, 8)
                "third_person_camera": (batch_size, traj_length, 3, image_size, image_size) # For table-bin scenario
                "episode_idx": (batch_size, )
            """
            batch["episode_idx"] = torch.arange(batch_size, device=self.device) # Override episode idx so the history can be correctly recorded
            actions.append(
                self.predict_single_traj(batch)
            )  # (batch_size, traj_length, action_dim)

        return aggregate_batch(
            actions, partial(torch.stack, dim=traj_num_dim_idx)
        )  # (batch_size, traj_num, traj_length, action_dim)


    # ================================ Training =================================

    def _encode_input_multi_traj(
        self, normalized_batch: batch_type
    ) -> tuple[batch_type, batch_type]:
        """
        Should be called only when training history cross-attention modules
        args:
            normalized_batch:
                "robot0_wrist_camera": (batch_size, traj_num, data_length, 3, image_size, image_size)
                "robot0_wrist_camera_feature": (batch_size, traj_num, data_length, 768) [Optional]
                "robot0_10d": (batch_size, traj_num, data_length, 10)
                "action0_10d": (batch_size, traj_num, data_length, 10)
                "future_0_wrist_camera": (batch_size, traj_num, data_length, 3, image_size, image_size)
                "third_person_camera": (batch_size, traj_num, data_length, 3, image_size, image_size) # For table-bin scenario
        return:
            data_dict:
                "global_cond": (batch_size, traj_num, token_num, global_cond_dim)
                "local_cond": (batch_size, traj_num, token_num, local_cond_dim)
                "noisy_action": (batch_size, traj_num, data_length, action_dim) # Noisy action latents
                "history_noisy_actions": (batch_size, traj_num, history_action_num_per_chunk, action_dim) # History latents, the noise will be 1-inference-step less than "noisy_action", to match the inference scenarios
                "history_img_features": (batch_size, traj_num, token_num, history_img_features_dim) # History image features
                "history_noisy_future_img_features": (batch_size, traj_num, token_num, feature_dim)
                "memory_gate_val": (batch_size, traj_num)
                "step": (batch_size,)
            target:
                "action": (batch_size, traj_num, data_length, 8)
        """

        batch_size = next(iter(normalized_batch.values())).shape[0]
        traj_num = next(iter(normalized_batch.values())).shape[1]
        data_dict: dict[str, torch.Tensor] = {}

        global_cond_dict = {
            k: v
            for k, v in normalized_batch.items()
            if k in self.global_cond_encoder.data_entry_names
        }

        global_cond_dict_feature = {
            k: v
            for k, v in normalized_batch.items()
            if "feature" in k and k.replace("_feature", "") in self.global_cond_encoder.data_entry_names
        }
        global_cond_dict.update(global_cond_dict_feature)

        data_dict["global_cond"] = einops.rearrange(
            self.global_cond_encoder.forward(
                dict_apply(
                    global_cond_dict,
                    lambda x: einops.rearrange(x, "b t ... -> (b t) ..."),
                )
            ),
            "(b t) ... -> b t ...",
            b=batch_size,
        )

        target: dict[str, torch.Tensor] = {}

        if self.local_cond_encoder is not None:
            local_cond_dict = {
                k: v
                for k, v in normalized_batch.items()
                if k in self.local_cond_encoder.data_entry_names
            }
            data_dict["local_cond"] = einops.rearrange(
                self.local_cond_encoder.forward(
                    dict_apply(
                        local_cond_dict,
                        lambda x: einops.rearrange(x, "b t ... -> (b t) ..."),
                    )
                ),
                "(b t) ... -> b t ...",
                b=batch_size,
            )

        train_timesteps: int = self.noise_scheduler.train_step_num
        inference_timesteps = self.noise_scheduler.inference_step_num
        step_ratio = train_timesteps // inference_timesteps

        data_dict["step"] = self.noise_scheduler.sample_training_timesteps(
            batch_size=batch_size,
            device=self.device,
            generator=self.torch_rng,
        )

        trajectory = einops.rearrange(
            self.action_decoder.encode(
                dict_apply(
                    {
                        k: normalized_batch[k]
                        for k in self.action_decoder.data_entry_names
                    },
                    lambda x: einops.rearrange(x, "b t ... -> (b t) ..."),
                )
            ),
            "(b t) ... -> b t ...",
            b=batch_size,
        ) # (batch_size, traj_num, traj_length, action_dim)

        action_noise = torch.randn_like(
            trajectory,
        )  # (batch_size, traj_num, traj_length, action_dim)
        data_dict["noisy_action"], target["action"] = self.noise_scheduler.get_noisy_action_and_target(
            trajectory,
            action_noise,
            data_dict["step"],
        )

        history_latent_diffusion_step = self.noise_scheduler.get_less_noisy_timesteps(data_dict["step"])

        if self.train_history_action_noise_level == "none":
            data_dict["history_noisy_actions"] = trajectory
        elif self.train_history_action_noise_level == "random":

            flattened_traj = einops.rearrange(
                trajectory,
                "b t ... -> (b t) ...",
            )
            history_action_noise = torch.randn_like(
                flattened_traj,
            )
            rand_timesteps = torch.randint(
                0,
                train_timesteps,
                (batch_size * traj_num,),
                device=self.device,
                generator=self.torch_rng,
            )
            data_dict["history_noisy_actions"] = einops.rearrange(
                self.noise_scheduler.get_noisy_action_and_target(
                    flattened_traj,
                    history_action_noise,
                    cast(torch.IntTensor, rand_timesteps),
                )[0],
                "(b t) ... -> b t ...",
                b=batch_size,
            )

        elif self.train_history_action_noise_level == "last_step":
            history_action_noise = torch.randn_like(
                trajectory,
            )
            data_dict["history_noisy_actions"], _ = self.noise_scheduler.get_noisy_action_and_target(
                trajectory,
                history_action_noise,
                cast(torch.IntTensor, history_latent_diffusion_step),
            )
        else:
            raise ValueError(f"Unknown history action noise level: {self.train_history_action_noise_level}")

        # Truncate the history noisy actions to the number of history actions per chunk
        data_dict["history_noisy_actions"] = data_dict["history_noisy_actions"][:, :, :self.history_action_num_per_chunk]
        if (
            getattr(self.denoising_network, "state_token_memory_enabled", False)
            and getattr(
                self.denoising_network,
                "state_token_update_include_action_history",
                True,
            )
        ):
            data_dict["state_token_update_actions"] = trajectory[
                :, :, : self.history_action_num_per_chunk
            ]

        if self.memory_gate is not None:
            normalized_batch_without_text = {
                k: v
                for k, v in normalized_batch.items()
                if not isinstance(v[0], str)
            }
            flattened_data_dict = dict_apply(
                normalized_batch_without_text,
                lambda x: einops.rearrange(x, "b t ... -> (b t) ..."),
            )
            val = self.memory_gate.get_gate_value(
                flattened_data_dict
            )
            data_dict["memory_gate_val"] = einops.rearrange(
                val,
                "(b t) ... -> b t ...",
                b=batch_size,
            ) # (batch_size, traj_num)
            # print(f"Memory gate val: {data_dict['memory_gate_val']}, {data_dict['memory_gate_val'].shape}, \ntraj_idx: {normalized_batch['traj_idx']}, {normalized_batch['traj_idx'].shape}")

        if self.history_img_feature_encoder is not None:
            img_dict = {
                k: normalized_batch[k]
                for k in self.history_img_feature_encoder.data_entry_names if k in normalized_batch
            }
            img_feature_dict = {
                f"{k}_feature": normalized_batch[f"{k}_feature"]
                for k in self.history_img_feature_encoder.data_entry_names if f"{k}_feature" in normalized_batch
            }
            img_dict.update(img_feature_dict)
            data_dict["history_img_features"] = (
                self.history_img_feature_encoder.forward(img_dict)
            )  # (batch_size, traj_num, img_num*img_feature_token_num, history_img_features_dim)

        if self.history_evidence_feature_encoder is not None:
            evidence_features = self._encode_training_history_evidence_features(
                normalized_batch
            )
            data_dict["history_evidence_features"] = evidence_features
            data_dict["evidence_query_features"] = evidence_features

        data_dict["entire_traj_is_padding"] = normalized_batch["entire_traj_is_padding"]

        if self.max_training_traj_num > 0:
            valid_traj_indices: list[torch.Tensor] = []
            valid_traj_masks: list[torch.Tensor] = []
            for i in range(batch_size):
                valid_traj_num = int(torch.sum(~data_dict["entire_traj_is_padding"][i]))
                assert valid_traj_num > 0, "At least one trajectory must be valid"
                assert not torch.any(data_dict["entire_traj_is_padding"][i, :valid_traj_num]), f"entire_traj_is_padding must be False for the first few trajectories, but got {data_dict['entire_traj_is_padding'][i, :valid_traj_num]}"
                if self.training_traj_sampling_strategy == "random":
                    sampled_traj_indices = torch.randint(
                        0,
                        valid_traj_num,
                        (self.max_training_traj_num,),
                        device=data_dict["entire_traj_is_padding"].device,
                    )
                    sampled_traj_mask = torch.ones(
                        self.max_training_traj_num,
                        device=data_dict["entire_traj_is_padding"].device,
                        dtype=torch.bool,
                    )
                elif self.training_traj_sampling_strategy == "tail":
                    tail_start = max(valid_traj_num - self.max_training_traj_num, 0)
                    sampled_traj_indices = torch.arange(
                        tail_start,
                        valid_traj_num,
                        device=data_dict["entire_traj_is_padding"].device,
                    )
                    sampled_traj_mask = torch.ones(
                        sampled_traj_indices.numel(),
                        device=data_dict["entire_traj_is_padding"].device,
                        dtype=torch.bool,
                    )
                    if sampled_traj_indices.numel() < self.max_training_traj_num:
                        pad_num = self.max_training_traj_num - sampled_traj_indices.numel()
                        sampled_traj_indices = torch.cat(
                            [
                                torch.zeros(
                                    pad_num,
                                    device=sampled_traj_indices.device,
                                    dtype=sampled_traj_indices.dtype,
                                ),
                                sampled_traj_indices,
                            ],
                            dim=0,
                        )
                        sampled_traj_mask = torch.cat(
                            [
                                torch.zeros(
                                    pad_num,
                                    device=sampled_traj_mask.device,
                                    dtype=torch.bool,
                                ),
                                sampled_traj_mask,
                            ],
                            dim=0,
                        )
                else:
                    raise ValueError(
                        "Unknown training_traj_sampling_strategy: "
                        f"{self.training_traj_sampling_strategy}"
                    )
                # aggregated_indices = sampled_traj_indices + i * self.max_training_traj_num
                valid_traj_indices.append(sampled_traj_indices)
                valid_traj_masks.append(sampled_traj_mask)
            all_valid_traj_indices = torch.stack(valid_traj_indices, dim=0)
            all_valid_traj_masks = torch.stack(valid_traj_masks, dim=0)
            # print(f"{all_valid_traj_indices.shape=}")
            data_dict["training_traj_indices"] = all_valid_traj_indices # (batch_size, max_training_traj_num)
            data_dict["training_traj_valid_mask"] = all_valid_traj_masks

            batch_idx = torch.arange(batch_size, device=self.device)[:, None]
            for k, v in target.items():
                target[k] = target[k][batch_idx, all_valid_traj_indices]
                # print(f"{k}: {target[k].shape}")
            # data_dict will be processed in MikasaMemoryTransformer._project_to_latent_space_multi_traj

        return data_dict, target

    def _get_history_contrastive_source_indices(
        self,
        normalized_batch: batch_type,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if batch_size < 2:
            return None

        episode_idx = normalized_batch.get("episode_idx")
        if not torch.is_tensor(episode_idx):
            return torch.roll(torch.arange(batch_size, device=device), shifts=1)

        episode_idx = episode_idx.to(device)
        if episode_idx.dim() >= 2:
            episode_idx = episode_idx[:, 0]
        elif episode_idx.dim() == 0:
            return None
        if episode_idx.dim() > 1:
            episode_idx = episode_idx.reshape(batch_size, -1)[:, 0]

        source_indices = torch.empty(batch_size, dtype=torch.long, device=device)
        for batch_idx in range(batch_size):
            candidates = torch.nonzero(
                episode_idx != episode_idx[batch_idx],
                as_tuple=False,
            ).flatten()
            if candidates.numel() == 0:
                return None
            cyclic_distance = (candidates - batch_idx) % batch_size
            source_indices[batch_idx] = candidates[torch.argmin(cyclic_distance)]

        return source_indices

    def _get_history_contrastive_replacement_masks(
        self,
        data_dict: batch_type,
        batch_size: int,
        traj_num: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "training_traj_indices" in data_dict:
            target_indices = data_dict["training_traj_indices"]
        else:
            target_indices = torch.arange(traj_num, device=device).unsqueeze(0)
            target_indices = target_indices.expand(batch_size, -1)

        replacement_mask = torch.zeros(
            (batch_size, traj_num),
            dtype=torch.bool,
            device=device,
        )
        target_has_replacement = torch.zeros(
            target_indices.shape,
            dtype=torch.bool,
            device=device,
        )
        max_history_len = int(getattr(self.denoising_network, "max_history_len", 0))
        anchor_len = int(getattr(self.denoising_network, "late_cue_anchor_len", 1))

        for batch_idx in range(batch_size):
            for target_pos in range(target_indices.shape[1]):
                target_idx = int(target_indices[batch_idx, target_pos].item())
                if self.history_contrastive_mode == "anchor_only":
                    start_idx = 0
                    end_idx = min(anchor_len, target_idx)
                elif self.history_contrastive_mode == "rolling_history_only":
                    start_idx = max(target_idx - max_history_len, 0)
                    end_idx = target_idx
                elif self.history_contrastive_mode == "all_history":
                    start_idx = 0
                    end_idx = target_idx
                else:
                    raise ValueError(
                        "Unknown history_contrastive_mode: "
                        f"{self.history_contrastive_mode}"
                    )

                start_idx = max(start_idx, 0)
                end_idx = min(end_idx, traj_num)
                if end_idx <= start_idx:
                    continue
                replacement_mask[batch_idx, start_idx:end_idx] = True
                target_has_replacement[batch_idx, target_pos] = True

        return replacement_mask, target_has_replacement

    def _build_history_contrastive_data_dict(
        self,
        normalized_batch: batch_type,
        data_dict: batch_type,
    ) -> tuple[batch_type | None, torch.Tensor | None]:
        history_keys = (
            "history_noisy_actions",
            "history_img_features",
            "history_mask",
        )
        if not any(key in data_dict for key in history_keys):
            return None, None

        first_history_key = next(key for key in history_keys if key in data_dict)
        batch_size, traj_num = data_dict[first_history_key].shape[:2]
        device = data_dict[first_history_key].device
        source_indices = self._get_history_contrastive_source_indices(
            normalized_batch,
            batch_size,
            device,
        )
        if source_indices is None:
            return None, None

        replacement_mask, target_has_replacement = (
            self._get_history_contrastive_replacement_masks(
                data_dict,
                batch_size,
                traj_num,
                device,
            )
        )
        if not torch.any(replacement_mask):
            return None, None

        mismatched_data_dict = dict(data_dict)
        for key in history_keys:
            if key not in data_dict:
                continue
            value = data_dict[key]
            source_value = value[source_indices]
            mask = replacement_mask
            while mask.dim() < value.dim():
                mask = mask.unsqueeze(-1)
            mismatched_data_dict[key] = torch.where(mask, source_value, value)

        return mismatched_data_dict, target_has_replacement

    def _get_effective_target_traj_indices(
        self,
        data_dict: batch_type,
        batch_size: int,
        traj_num: int,
        device: torch.device,
    ) -> torch.Tensor:
        if "training_traj_indices" in data_dict:
            return data_dict["training_traj_indices"].to(
                device=device,
                dtype=torch.long,
            )
        target_traj_indices = torch.arange(traj_num, device=device).unsqueeze(0)
        return target_traj_indices.expand(batch_size, -1)

    def _build_decision_window_contrastive_data_dict(
        self,
        normalized_batch: batch_type,
        data_dict: batch_type,
    ) -> tuple[batch_type | None, torch.Tensor | None, torch.Tensor | None]:
        first_key = "noisy_action"
        if first_key not in data_dict:
            return None, None, None
        batch_size, traj_num = data_dict[first_key].shape[:2]
        device = data_dict[first_key].device
        source_indices = self._get_history_contrastive_source_indices(
            normalized_batch,
            batch_size,
            device,
        )
        if source_indices is None:
            return None, None, None

        target_traj_indices = self._get_effective_target_traj_indices(
            data_dict,
            batch_size,
            traj_num,
            device,
        )
        mismatched_data_dict = dict(data_dict)
        target_has_replacement = torch.zeros(
            target_traj_indices.shape,
            dtype=torch.bool,
            device=device,
        )
        source_mode = self.decision_window_contrastive_memory_source
        use_history = source_mode in ("history", "history_and_state_token", "state_token_history")
        use_state_token = source_mode in (
            "state_token",
            "history_and_state_token",
            "state_token_history",
        )

        if use_history:
            history_keys = (
                "history_noisy_actions",
                "history_img_features",
                "history_mask",
            )
            if any(key in data_dict for key in history_keys):
                replacement_mask, history_target_mask = (
                    self._get_history_contrastive_replacement_masks(
                        data_dict,
                        batch_size,
                        traj_num,
                        device,
                    )
                )
                if torch.any(replacement_mask):
                    for key in history_keys:
                        if key not in data_dict:
                            continue
                        value = data_dict[key]
                        source_value = value[source_indices]
                        mask = replacement_mask
                        while mask.dim() < value.dim():
                            mask = mask.unsqueeze(-1)
                        mismatched_data_dict[key] = torch.where(
                            mask,
                            source_value,
                            value,
                        )
                    target_has_replacement = (
                        target_has_replacement | history_target_mask
                    )

        if use_state_token:
            if bool(getattr(self.denoising_network, "state_token_memory_enabled", False)):
                mismatched_data_dict["state_token_source_indices"] = source_indices
                target_has_replacement = target_has_replacement | (
                    target_traj_indices > 0
                )

        if not torch.any(target_has_replacement):
            return None, None, None
        return mismatched_data_dict, target_has_replacement, source_indices

    def _build_decision_window_mask(
        self,
        target_action: torch.Tensor,
        source_indices: torch.Tensor,
        base_valid_mask: torch.Tensor,
        target_traj_indices: torch.Tensor,
        critical_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_delta = F.mse_loss(
            target_action,
            target_action[source_indices],
            reduction="none",
        )
        action_delta = einops.reduce(action_delta, "b t ... -> b t", "mean")
        window_mask = (
            base_valid_mask
            & (target_traj_indices >= self.decision_window_min_traj_index)
            & (action_delta >= self.decision_window_action_delta_min)
        )
        threshold = action_delta.new_tensor(self.decision_window_action_delta_min)
        quantile = self.decision_window_action_delta_quantile
        if quantile > 0.0 and torch.any(window_mask):
            threshold = torch.quantile(action_delta[window_mask].detach(), quantile)
            window_mask = window_mask & (action_delta >= threshold)
        if (
            self.decision_window_include_critical
            and critical_mask is not None
        ):
            window_mask = window_mask | (critical_mask & base_valid_mask)
        return window_mask, action_delta.detach(), threshold.detach()

    def _get_teacher_preservation_network(
        self,
        device: torch.device,
    ) -> MikasaMemoryTransformer:
        teacher = self.__dict__.get("_teacher_preservation_denoising_network")
        if teacher is None:
            teacher = copy.deepcopy(self.denoising_network)
            teacher.requires_grad_(False)
            teacher.eval()
            teacher.to(device)
            # Avoid registering the teacher as a child module; checkpoints should
            # contain only the trainable policy, not the frozen reference copy.
            self.__dict__["_teacher_preservation_denoising_network"] = teacher
            print("Captured frozen teacher denoising network for preservation loss")
        else:
            teacher.eval()
            try:
                teacher_device = next(teacher.parameters()).device
            except StopIteration:
                teacher_device = device
            if teacher_device != device:
                teacher.to(device)
        assert isinstance(teacher, MikasaMemoryTransformer)
        return teacher

    def _apply_burn_in_loss_mask(
        self,
        action_loss: torch.Tensor,
        loss_valid_mask: torch.Tensor,
        training_traj_indices: torch.Tensor | None = None,
        sample_start_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loss_valid_mask = _apply_burn_in_loss_mask(
            loss_valid_mask=loss_valid_mask,
            traj_idx=sample_start_indices,
            burn_in_start_id=self.burn_in_start_id,
            burn_in_loss_traj_num=self.burn_in_loss_traj_num,
            training_traj_indices=training_traj_indices,
        )
        action_loss = action_loss * loss_valid_mask
        return action_loss, loss_valid_mask


    def compute_loss(
        self,
        normalized_batch: batch_type,
    ) -> batch_type:
        """
        If self.skip_memory, will directly call the method in the superclass:
            normalized_batch:
                "robot0_wrist_camera": (batch_size, traj_length, 3, image_size, image_size)
                "robot0_wrist_camera_feature": (batch_size, traj_length, 768) [Optional]
                "robot0_10d": (batch_size, traj_length, 8)
                "action0_10d": (batch_size, traj_length, 8)
                "future_0_wrist_camera": (batch_size, traj_length, 3, image_size, image_size)
                "third_person_camera": (batch_size, traj_length, 3, image_size, image_size) # For table-bin scenario
                "action_is_error": (batch_size, traj_length)
                "action_is_critical": (batch_size, traj_length)

        If not self.skip_memory:
            normalized_batch:
                "robot0_wrist_camera": (batch_size, traj_num, traj_length, 3, image_size, image_size)
                "robot0_wrist_camera_feature": (batch_size, traj_num, traj_length, 768) [Optional]
                "robot0_10d": (batch_size, traj_num, traj_length, 8)
                "action0_10d": (batch_size, traj_num, traj_length, 8)
                "future_0_wrist_camera": (batch_size, traj_num, traj_length, 3, image_size, image_size)
                "third_person_camera": (batch_size, traj_num, traj_length, 3, image_size, image_size) # For table-bin scenario
                "entire_traj_is_padding": (batch_size, traj_num)
                "action_is_error": (batch_size, traj_num, traj_length)
                "action_is_critical": (batch_size, traj_num, traj_length) # Optional
        """
        if self.skip_memory:
            return super().compute_loss(normalized_batch)

        # print(f"normalized_batch keys: {normalized_batch.keys()}")

        loss = {}

        self.shared_model_manager.clear_cache() # Clear the cache before every forward pass


        # print(normalized_batch["robot0_wrist_camera"][0,0].min(), normalized_batch["robot0_wrist_camera"][0,0].max())
        # img = normalized_batch["robot0_wrist_camera"][0,0].cpu().numpy()
        # cv2_img = img.squeeze(0).transpose(1, 2, 0)  # [image_size, image_size, 3]
        # cv2_img = (cv2_img) * 255
        # cv2_img = cv2.cvtColor(cv2_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        # cv2.imwrite(f"robot0_wrist_camera.png", cv2_img)

        # third_person_camera = normalized_batch["third_person_camera"][0,0].cpu().numpy()
        # print(third_person_camera.min(), third_person_camera.max())
        # cv2_img = third_person_camera.squeeze(0).transpose(1, 2, 0)  # [image_size, image_size, 3]
        # cv2_img = (cv2_img) * 255
        # cv2_img = cv2.cvtColor(cv2_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        # cv2.imwrite(f"third_person_camera.png", cv2_img)

        # exit()

        action_key_names = self.action_decoder.data_entry_names
        action_traj_length = normalized_batch[action_key_names[0]].shape[2]

        global_cond_key_names = self.global_cond_encoder.data_entry_names
        global_cond_valid_key_name = global_cond_key_names[0]
        if f"{global_cond_valid_key_name}_feature" in normalized_batch:
            global_cond_valid_key_name = f"{global_cond_valid_key_name}_feature"

        traj_num = normalized_batch[global_cond_valid_key_name].shape[1]
        batch_size = normalized_batch[global_cond_valid_key_name].shape[0]

        if "local_cond" in normalized_batch and len(normalized_batch["local_cond"]) > 0:
            assert (batch_size, traj_num) == next(
                iter(normalized_batch["local_cond"].values())
            ).shape[:2], "Please make sure you are using multi-trajectory dataset"

        assert (batch_size, traj_num) == normalized_batch[action_key_names[0]].shape[
            :2
        ], f"Please make sure you are using multi-trajectory dataset. (batch_size: {batch_size}, traj_num: {traj_num}, action_shape: {normalized_batch[action_key_names[0]].shape})"

        data_dict, target = self._encode_input_multi_traj(normalized_batch)
        if "training_traj_indices" in data_dict:
            loss["diagnostic/training_traj_index_mean"] = (
                data_dict["training_traj_indices"].detach().float().mean()
            )
            loss["diagnostic/training_traj_index_max"] = (
                data_dict["training_traj_indices"].detach().float().max()
            )

        # self._add_random_masks(data_dict)

        if isinstance(self.denoising_network, OptimizedModule):
            # After torch compile: Just fix the type of the denoising network for type checking.
            self.denoising_network = cast(MikasaMemoryTransformer, cast(Any, self.denoising_network))
        else:
            assert isinstance(
                self.denoising_network, MikasaMemoryTransformer
            ), "MikasaMemoryTransformer is required for memory-based policy"

        teacher_denoising_network = None
        if self.teacher_preservation_loss_weight > 0.0:
            teacher_denoising_network = self._get_teacher_preservation_network(
                data_dict["noisy_action"].device
            )

        model_output: dict[str, torch.Tensor] = self.denoising_network.parallel_forward(
            data_dict
        )
        for key, values in self.denoising_network.recorded_data_dict.items():
            if key in ("memory_gate_val", "history_cross_attention") or len(values) == 0:
                continue
            if key.endswith("_ratio") or key.endswith("_gate") or key.endswith("_norm"):
                loss[f"diagnostic/{key}"] = torch.stack(
                    [value.detach().float().mean() for value in values]
                ).mean()

        if "memory_gate_val" in self.denoising_network.recorded_data_dict:
            memory_gate_val = self.denoising_network.recorded_data_dict[
                "memory_gate_val"
            ]  # (batch_size, traj_num, transformer_layer_num, input_token_num)

            if len(memory_gate_val) > 0:
                effective_traj_num = (
                    data_dict["training_traj_indices"].shape[1]
                    if "training_traj_indices" in data_dict
                    else traj_num
                )
                memory_gate_val = einops.reduce(
                    memory_gate_val,
                    "l (b t)-> b t",
                    "mean",
                    b=batch_size,
                    t=effective_traj_num,
                )  # (batch_size, traj_num)
                if "training_traj_valid_mask" in data_dict:
                    memory_gate_val = memory_gate_val * data_dict[
                        "training_traj_valid_mask"
                    ]
                else:
                    memory_gate_val = memory_gate_val * (
                        ~normalized_batch["entire_traj_is_padding"]
                    )  # Do not compute loss for padding trajectories
            else:
                memory_gate_val = None
        else:
            memory_gate_val = None

        critical_memory_gate_val = None

        action_loss = F.mse_loss(
            model_output["action"], target["action"], reduction="none"
        )

        action_loss = einops.reduce(
            action_loss, "b t ... -> b t", "mean"
        )  # mean over all dimensions except batch and traj_num # (batch_size, traj_num)
        loss_valid_mask = torch.ones_like(action_loss, dtype=torch.bool)
        if self.max_training_traj_num <= 0:
            loss_valid_mask = ~normalized_batch["entire_traj_is_padding"]
        elif "training_traj_valid_mask" in data_dict:
            loss_valid_mask = data_dict["training_traj_valid_mask"].bool()
        action_loss, loss_valid_mask = self._apply_burn_in_loss_mask(
            action_loss,
            loss_valid_mask,
            data_dict.get("training_traj_indices"),
            normalized_batch.get("traj_idx"),
        )
        if memory_gate_val is not None:
            memory_gate_val = memory_gate_val * loss_valid_mask

        # img_feature_loss = None

        if "action_is_error" in normalized_batch:
            traj_error_mask = torch.zeros(action_traj_length, device=self.device)
            traj_error_mask[
                self.action_no_error_range[0] : self.action_no_error_range[1]
            ] = 1
            action_is_error = normalized_batch["action_is_error"]
            if self.max_training_traj_num > 0 and "training_traj_indices" in data_dict:
                batch_idx = torch.arange(batch_size, device=self.device)[:, None]
                action_is_error = action_is_error[
                    batch_idx,
                    data_dict["training_traj_indices"],
                ]
            traj_is_error = (
                action_is_error * traj_error_mask[None, None, :]
            )  # (batch_size, traj_num, traj_length)
            traj_is_error = einops.reduce(
                traj_is_error, "b t ... -> b t", "any"
            )  # (batch_size, traj_num)
            loss_valid_mask = loss_valid_mask & (~traj_is_error)
            action_loss = action_loss * (~traj_is_error)

            # if img_feature_loss is not None:
            #     img_feature_loss = img_feature_loss * (~traj_is_error)
            if memory_gate_val is not None:
                memory_gate_val = memory_gate_val * (~traj_is_error)

        summary_only_action_loss = None
        if (
            self.summary_only_aux_loss_weight > 0.0
            or self.summary_contrastive_loss_weight > 0.0
        ):
            summary_only_model_output = self.denoising_network.parallel_forward(
                data_dict,
                disable_direct_history_paths=True,
            )
            summary_only_action_loss = F.mse_loss(
                summary_only_model_output["action"],
                target["action"],
                reduction="none",
            )
            summary_only_action_loss = einops.reduce(
                summary_only_action_loss,
                "b t ... -> b t",
                "mean",
            )
            summary_only_action_loss = summary_only_action_loss * loss_valid_mask
            if self.summary_only_aux_loss_weight > 0.0:
                if loss_valid_mask.sum() == 0:
                    summary_only_aux_raw = (
                        summary_only_action_loss.sum()
                        / summary_only_action_loss.numel()
                    )
                else:
                    summary_only_aux_raw = (
                        summary_only_action_loss.sum() / loss_valid_mask.sum()
                    )
                loss["summary_only_aux"] = (
                    summary_only_aux_raw * self.summary_only_aux_loss_weight
                )
                loss["diagnostic/summary_only_aux_raw"] = summary_only_aux_raw.detach()
                loss["diagnostic/summary_only_aux_weight"] = action_loss.new_tensor(
                    self.summary_only_aux_loss_weight
                )

        if self.summary_contrastive_loss_weight > 0.0:
            summary_contrastive_loss = action_loss.new_zeros(())
            summary_contrastive_raw = action_loss.new_zeros(())
            summary_contrastive_correct = action_loss.new_zeros(())
            summary_contrastive_mismatched = action_loss.new_zeros(())
            summary_contrastive_active_ratio = action_loss.new_zeros(())
            summary_contrastive_valid_ratio = action_loss.new_zeros(())

            mismatched_data_dict, summary_contrastive_target_mask = (
                self._build_history_contrastive_data_dict(
                    normalized_batch,
                    data_dict,
                )
            )
            if (
                mismatched_data_dict is not None
                and summary_contrastive_target_mask is not None
                and summary_only_action_loss is not None
            ):
                mismatched_summary_output = self.denoising_network.parallel_forward(
                    mismatched_data_dict,
                    disable_direct_history_paths=True,
                )
                mismatched_summary_action_loss = F.mse_loss(
                    mismatched_summary_output["action"],
                    target["action"],
                    reduction="none",
                )
                mismatched_summary_action_loss = einops.reduce(
                    mismatched_summary_action_loss,
                    "b t ... -> b t",
                    "mean",
                )
                summary_contrastive_valid_mask = (
                    summary_contrastive_target_mask & loss_valid_mask
                )
                summary_contrastive_valid_ratio = (
                    summary_contrastive_valid_mask.detach().float().mean()
                )
                if torch.any(summary_contrastive_valid_mask):
                    correct_for_rank = (
                        summary_only_action_loss.detach()
                        if self.summary_contrastive_detach_correct_loss
                        else summary_only_action_loss
                    )
                    rank_loss = F.relu(
                        self.summary_contrastive_margin
                        + correct_for_rank
                        - mismatched_summary_action_loss
                    )
                    valid_rank_loss = rank_loss[summary_contrastive_valid_mask]
                    summary_contrastive_raw = valid_rank_loss.mean()
                    summary_contrastive_loss = (
                        summary_contrastive_raw
                        * self.summary_contrastive_loss_weight
                    )
                    summary_contrastive_correct = (
                        summary_only_action_loss[summary_contrastive_valid_mask]
                        .detach()
                        .mean()
                    )
                    summary_contrastive_mismatched = (
                        mismatched_summary_action_loss[
                            summary_contrastive_valid_mask
                        ]
                        .detach()
                        .mean()
                    )
                    summary_contrastive_active_ratio = (
                        valid_rank_loss.detach().gt(0).float().mean()
                    )

            loss["summary_contrastive"] = summary_contrastive_loss
            loss["diagnostic/summary_contrastive_raw"] = (
                summary_contrastive_raw.detach()
            )
            loss["diagnostic/summary_contrastive_correct_loss"] = (
                summary_contrastive_correct.detach()
            )
            loss["diagnostic/summary_contrastive_mismatched_loss"] = (
                summary_contrastive_mismatched.detach()
            )
            loss["diagnostic/summary_contrastive_active_ratio"] = (
                summary_contrastive_active_ratio.detach()
            )
            loss["diagnostic/summary_contrastive_valid_ratio"] = (
                summary_contrastive_valid_ratio.detach()
            )
            loss["diagnostic/summary_contrastive_weight"] = action_loss.new_tensor(
                self.summary_contrastive_loss_weight
            )
            loss["diagnostic/summary_contrastive_margin"] = action_loss.new_tensor(
                self.summary_contrastive_margin
            )

        critical_action_loss = None
        traj_action_is_critical = None
        if "action_is_critical" in normalized_batch:
            # Is based on the previous filtered loss (action_is_error and entire_traj_is_padding)
            single_action_is_critical = normalized_batch["action_is_critical"]
            if self.max_training_traj_num > 0 and "training_traj_indices" in data_dict:
                batch_idx = torch.arange(batch_size, device=self.device)[:, None]
                single_action_is_critical = single_action_is_critical[
                    batch_idx,
                    data_dict["training_traj_indices"],
                ]
            # (batch_size, traj_num, traj_length)
            traj_action_is_critical = torch.any(
                single_action_is_critical, dim=2
            ).squeeze(
                -1
            )  # (batch_size, traj_num)
            critical_action_loss = action_loss * traj_action_is_critical

            critical_valid_mask = traj_action_is_critical & loss_valid_mask
            if critical_valid_mask.sum() > 0:
                loss["critical_action"] = (
                    critical_action_loss.sum() / critical_valid_mask.sum()
                )
            else:
                loss["critical_action"] = critical_action_loss.sum()

            if memory_gate_val is not None:
                critical_memory_gate_val = memory_gate_val * traj_action_is_critical
                valid_mask = critical_valid_mask
                if valid_mask.sum() > 0:
                    loss["critical_memory_gate_val"] = (
                        critical_memory_gate_val.sum()
                        / valid_mask.sum()
                    )
                    loss["critical_binary_memory_gate_val"] = (critical_memory_gate_val > 0.5).sum() / valid_mask.sum()
                else:
                    loss["critical_memory_gate_val"] = critical_memory_gate_val.sum()
                    loss["critical_binary_memory_gate_val"] = (critical_memory_gate_val > 0.5).sum()

        if memory_gate_val is not None:
            valid_num = loss_valid_mask.sum()
            if valid_num > 0:
                loss["memory_gate_val"] = memory_gate_val.sum() / valid_num
                loss["binary_memory_gate_val"] = (
                    (memory_gate_val > 0.5).sum() / valid_num
                )
            else:
                loss["memory_gate_val"] = memory_gate_val.sum()
                loss["binary_memory_gate_val"] = (memory_gate_val > 0.5).sum()

        if self.history_contrastive_loss_weight > 0.0:
            contrastive_loss = action_loss.new_zeros(())
            contrastive_raw = action_loss.new_zeros(())
            contrastive_correct = action_loss.new_zeros(())
            contrastive_mismatched = action_loss.new_zeros(())
            contrastive_active_ratio = action_loss.new_zeros(())
            contrastive_valid_ratio = action_loss.new_zeros(())

            mismatched_data_dict, contrastive_target_mask = (
                self._build_history_contrastive_data_dict(
                    normalized_batch,
                    data_dict,
                )
            )
            if (
                mismatched_data_dict is not None
                and contrastive_target_mask is not None
            ):
                mismatched_model_output = self.denoising_network.parallel_forward(
                    mismatched_data_dict
                )
                mismatched_action_loss = F.mse_loss(
                    mismatched_model_output["action"],
                    target["action"],
                    reduction="none",
                )
                mismatched_action_loss = einops.reduce(
                    mismatched_action_loss,
                    "b t ... -> b t",
                    "mean",
                )
                contrastive_valid_mask = contrastive_target_mask & (
                    loss_valid_mask
                )
                contrastive_valid_ratio = (
                    contrastive_valid_mask.detach().float().mean()
                )
                if torch.any(contrastive_valid_mask):
                    correct_for_rank = (
                        action_loss.detach()
                        if self.history_contrastive_detach_correct_loss
                        else action_loss
                    )
                    rank_loss = F.relu(
                        self.history_contrastive_margin
                        + correct_for_rank
                        - mismatched_action_loss
                    )
                    valid_rank_loss = rank_loss[contrastive_valid_mask]
                    contrastive_raw = valid_rank_loss.mean()
                    contrastive_loss = (
                        contrastive_raw * self.history_contrastive_loss_weight
                    )
                    contrastive_correct = (
                        action_loss[contrastive_valid_mask].detach().mean()
                    )
                    contrastive_mismatched = (
                        mismatched_action_loss[contrastive_valid_mask]
                        .detach()
                        .mean()
                    )
                    contrastive_active_ratio = (
                        valid_rank_loss.detach().gt(0).float().mean()
                    )

            loss["history_contrastive"] = contrastive_loss
            loss["diagnostic/history_contrastive_raw"] = contrastive_raw.detach()
            loss["diagnostic/history_contrastive_correct_loss"] = (
                contrastive_correct.detach()
            )
            loss["diagnostic/history_contrastive_mismatched_loss"] = (
                contrastive_mismatched.detach()
            )
            loss["diagnostic/history_contrastive_active_ratio"] = (
                contrastive_active_ratio.detach()
            )
            loss["diagnostic/history_contrastive_valid_ratio"] = (
                contrastive_valid_ratio.detach()
            )

        if self.state_token_contrastive_loss_weight > 0.0:
            state_token_contrastive_loss = action_loss.new_zeros(())
            state_token_contrastive_raw = action_loss.new_zeros(())
            state_token_contrastive_correct = action_loss.new_zeros(())
            state_token_contrastive_mismatched = action_loss.new_zeros(())
            state_token_contrastive_active_ratio = action_loss.new_zeros(())
            state_token_contrastive_valid_ratio = action_loss.new_zeros(())

            use_state_tokens = bool(
                getattr(self.denoising_network, "state_token_memory_enabled", False)
            )
            source_indices = None
            if use_state_tokens:
                source_indices = self._get_history_contrastive_source_indices(
                    normalized_batch,
                    batch_size,
                    data_dict["noisy_action"].device,
                )
            if source_indices is not None:
                mismatched_data_dict = dict(data_dict)
                mismatched_data_dict["state_token_source_indices"] = source_indices
                mismatched_model_output = self.denoising_network.parallel_forward(
                    mismatched_data_dict
                )
                mismatched_action_loss = F.mse_loss(
                    mismatched_model_output["action"],
                    target["action"],
                    reduction="none",
                )
                mismatched_action_loss = einops.reduce(
                    mismatched_action_loss,
                    "b t ... -> b t",
                    "mean",
                )
                if "training_traj_indices" in data_dict:
                    target_traj_indices = data_dict["training_traj_indices"]
                else:
                    target_traj_indices = torch.arange(
                        traj_num, device=data_dict["noisy_action"].device
                    ).unsqueeze(0)
                    target_traj_indices = target_traj_indices.expand(batch_size, -1)
                # State-token memory is only available after at least one prior
                # in-episode update, so exclude first decision windows.
                state_token_valid_mask = target_traj_indices > 0
                state_token_valid_mask = state_token_valid_mask & loss_valid_mask
                state_token_contrastive_valid_ratio = (
                    state_token_valid_mask.detach().float().mean()
                )
                if torch.any(state_token_valid_mask):
                    correct_for_rank = (
                        action_loss.detach()
                        if self.state_token_contrastive_detach_correct_loss
                        else action_loss
                    )
                    rank_loss = F.relu(
                        self.state_token_contrastive_margin
                        + correct_for_rank
                        - mismatched_action_loss
                    )
                    valid_rank_loss = rank_loss[state_token_valid_mask]
                    state_token_contrastive_raw = valid_rank_loss.mean()
                    state_token_contrastive_loss = (
                        state_token_contrastive_raw
                        * self.state_token_contrastive_loss_weight
                    )
                    state_token_contrastive_correct = (
                        action_loss[state_token_valid_mask].detach().mean()
                    )
                    state_token_contrastive_mismatched = (
                        mismatched_action_loss[state_token_valid_mask]
                        .detach()
                        .mean()
                    )
                    state_token_contrastive_active_ratio = (
                        valid_rank_loss.detach().gt(0).float().mean()
                    )

            loss["state_token_contrastive"] = state_token_contrastive_loss
            loss["diagnostic/state_token_contrastive_raw"] = (
                state_token_contrastive_raw.detach()
            )
            loss["diagnostic/state_token_contrastive_correct_loss"] = (
                state_token_contrastive_correct.detach()
            )
            loss["diagnostic/state_token_contrastive_mismatched_loss"] = (
                state_token_contrastive_mismatched.detach()
            )
            loss["diagnostic/state_token_contrastive_active_ratio"] = (
                state_token_contrastive_active_ratio.detach()
            )
            loss["diagnostic/state_token_contrastive_valid_ratio"] = (
                state_token_contrastive_valid_ratio.detach()
            )
            loss["diagnostic/state_token_contrastive_weight"] = (
                action_loss.new_tensor(self.state_token_contrastive_loss_weight)
            )
            loss["diagnostic/state_token_contrastive_margin"] = (
                action_loss.new_tensor(self.state_token_contrastive_margin)
            )

        if (
            self.state_token_action_grounding_loss_weight > 0.0
            or self.state_token_action_grounding_aux_loss_weight > 0.0
        ):
            grounding_loss = action_loss.new_zeros(())
            grounding_aux_loss = action_loss.new_zeros(())
            grounding_raw = action_loss.new_zeros(())
            grounding_aux_raw = action_loss.new_zeros(())
            grounding_correct = action_loss.new_zeros(())
            grounding_mismatched = action_loss.new_zeros(())
            grounding_active_ratio = action_loss.new_zeros(())
            grounding_valid_ratio = action_loss.new_zeros(())
            grounding_delta_mean = action_loss.new_zeros(())
            grounding_delta_threshold = action_loss.new_tensor(
                self.state_token_action_grounding_action_delta_min
            )

            use_state_tokens = bool(
                getattr(self.denoising_network, "state_token_memory_enabled", False)
            )
            source_indices = None
            if use_state_tokens:
                source_indices = self._get_history_contrastive_source_indices(
                    normalized_batch,
                    batch_size,
                    data_dict["noisy_action"].device,
                )
            if source_indices is not None:
                target_traj_indices = self._get_effective_target_traj_indices(
                    data_dict,
                    batch_size,
                    traj_num,
                    data_dict["noisy_action"].device,
                )
                base_grounding_mask = (
                    loss_valid_mask
                    & (
                        target_traj_indices
                        >= self.state_token_action_grounding_min_traj_index
                    )
                )
                action_delta = F.mse_loss(
                    target["action"].detach(),
                    target["action"].detach()[source_indices],
                    reduction="none",
                )
                action_delta = einops.reduce(action_delta, "b t ... -> b t", "mean")
                grounding_delta_threshold = action_delta.new_tensor(
                    self.state_token_action_grounding_action_delta_min
                )
                grounding_mask = base_grounding_mask & (
                    action_delta >= self.state_token_action_grounding_action_delta_min
                )
                quantile = self.state_token_action_grounding_action_delta_quantile
                if quantile > 0.0 and torch.any(grounding_mask):
                    grounding_delta_threshold = torch.quantile(
                        action_delta[grounding_mask].detach(),
                        quantile,
                    )
                    grounding_mask = grounding_mask & (
                        action_delta >= grounding_delta_threshold
                    )
                if (
                    self.state_token_action_grounding_include_critical
                    and traj_action_is_critical is not None
                ):
                    grounding_mask = grounding_mask | (
                        traj_action_is_critical & loss_valid_mask
                    )
                grounding_valid_ratio = grounding_mask.detach().float().mean()
                if torch.any(base_grounding_mask):
                    grounding_delta_mean = action_delta[base_grounding_mask].mean()

                correct_state_only_output = self.denoising_network.parallel_forward(
                    data_dict,
                    disable_direct_history_paths=True,
                )
                mismatched_data_dict = dict(data_dict)
                mismatched_data_dict["state_token_source_indices"] = source_indices
                mismatched_state_only_output = self.denoising_network.parallel_forward(
                    mismatched_data_dict,
                    disable_direct_history_paths=True,
                )
                correct_state_only_loss = F.mse_loss(
                    correct_state_only_output["action"],
                    target["action"],
                    reduction="none",
                )
                correct_state_only_loss = einops.reduce(
                    correct_state_only_loss,
                    "b t ... -> b t",
                    "mean",
                )
                mismatched_state_only_loss = F.mse_loss(
                    mismatched_state_only_output["action"],
                    target["action"],
                    reduction="none",
                )
                mismatched_state_only_loss = einops.reduce(
                    mismatched_state_only_loss,
                    "b t ... -> b t",
                    "mean",
                )
                if torch.any(grounding_mask):
                    grounding_aux_raw = correct_state_only_loss[grounding_mask].mean()
                    grounding_aux_loss = (
                        grounding_aux_raw
                        * self.state_token_action_grounding_aux_loss_weight
                    )
                    correct_for_rank = (
                        correct_state_only_loss.detach()
                        if self.state_token_action_grounding_detach_correct_loss
                        else correct_state_only_loss
                    )
                    rank_loss = F.relu(
                        self.state_token_action_grounding_margin
                        + correct_for_rank
                        - mismatched_state_only_loss
                    )
                    valid_rank_loss = rank_loss[grounding_mask]
                    grounding_raw = valid_rank_loss.mean()
                    grounding_loss = (
                        grounding_raw
                        * self.state_token_action_grounding_loss_weight
                    )
                    grounding_correct = (
                        correct_state_only_loss[grounding_mask].detach().mean()
                    )
                    grounding_mismatched = (
                        mismatched_state_only_loss[grounding_mask].detach().mean()
                    )
                    grounding_active_ratio = (
                        valid_rank_loss.detach().gt(0).float().mean()
                    )

            loss["state_token_action_grounding"] = grounding_loss
            if self.state_token_action_grounding_aux_loss_weight > 0.0:
                loss["state_token_action_grounding_aux"] = grounding_aux_loss
            loss["diagnostic/state_token_action_grounding_raw"] = (
                grounding_raw.detach()
            )
            loss["diagnostic/state_token_action_grounding_aux_raw"] = (
                grounding_aux_raw.detach()
            )
            loss["diagnostic/state_token_action_grounding_correct_loss"] = (
                grounding_correct.detach()
            )
            loss["diagnostic/state_token_action_grounding_mismatched_loss"] = (
                grounding_mismatched.detach()
            )
            loss["diagnostic/state_token_action_grounding_active_ratio"] = (
                grounding_active_ratio.detach()
            )
            loss["diagnostic/state_token_action_grounding_valid_ratio"] = (
                grounding_valid_ratio.detach()
            )
            loss["diagnostic/state_token_action_grounding_action_delta_mean"] = (
                grounding_delta_mean.detach()
            )
            loss["diagnostic/state_token_action_grounding_action_delta_threshold"] = (
                grounding_delta_threshold.detach()
            )
            loss["diagnostic/state_token_action_grounding_weight"] = (
                action_loss.new_tensor(self.state_token_action_grounding_loss_weight)
            )
            loss["diagnostic/state_token_action_grounding_aux_weight"] = (
                action_loss.new_tensor(
                    self.state_token_action_grounding_aux_loss_weight
                )
            )
            loss["diagnostic/state_token_action_grounding_margin"] = (
                action_loss.new_tensor(self.state_token_action_grounding_margin)
            )

        if (
            self.anchor_action_delta_loss_weight > 0.0
            or self.anchor_action_delta_cosine_loss_weight > 0.0
        ):
            anchor_delta_loss = action_loss.new_zeros(())
            anchor_delta_raw = action_loss.new_zeros(())
            anchor_delta_cosine_raw = action_loss.new_zeros(())
            anchor_delta_valid_ratio = action_loss.new_zeros(())
            anchor_delta_behavior_mean = action_loss.new_zeros(())
            anchor_delta_threshold = action_loss.new_tensor(
                self.anchor_action_delta_action_delta_min
            )
            anchor_delta_model_rmse = action_loss.new_zeros(())
            anchor_delta_target_rmse = action_loss.new_zeros(())
            anchor_delta_swap_rmse = action_loss.new_zeros(())
            anchor_delta_cosine = action_loss.new_zeros(())

            use_anchor_prehead = bool(
                getattr(
                    self.denoising_network,
                    "late_cue_action_prehead_adapter_enabled",
                    False,
                )
            )
            source_indices = None
            if use_anchor_prehead and "history_img_features" in data_dict:
                source_indices = self._get_history_contrastive_source_indices(
                    normalized_batch,
                    batch_size,
                    data_dict["noisy_action"].device,
                )
            if source_indices is not None:
                anchor_len = int(
                    getattr(self.denoising_network, "late_cue_anchor_len", 1)
                )
                history_img_features = data_dict["history_img_features"]
                anchor_len = min(anchor_len, history_img_features.shape[1])
                if anchor_len > 0:
                    swapped_data_dict = dict(data_dict)
                    swapped_history_img_features = history_img_features.clone()
                    swapped_history_img_features[:, :anchor_len] = (
                        history_img_features[source_indices, :anchor_len]
                    )
                    swapped_data_dict[
                        "history_img_features"
                    ] = swapped_history_img_features

                    correct_anchor_output = self.denoising_network.parallel_forward(
                        data_dict,
                        disable_direct_history_paths=True,
                    )
                    swapped_anchor_output = self.denoising_network.parallel_forward(
                        swapped_data_dict,
                        disable_direct_history_paths=True,
                    )

                    target_traj_indices = self._get_effective_target_traj_indices(
                        data_dict,
                        batch_size,
                        traj_num,
                        data_dict["noisy_action"].device,
                    )
                    target_delta = (
                        target["action"].detach()
                        - target["action"].detach()[source_indices]
                    )
                    behavior_delta = F.mse_loss(
                        target["action"].detach(),
                        target["action"].detach()[source_indices],
                        reduction="none",
                    )
                    behavior_delta = einops.reduce(
                        behavior_delta,
                        "b t ... -> b t",
                        "mean",
                    )
                    anchor_mask = (
                        loss_valid_mask
                        & (
                            target_traj_indices
                            >= self.anchor_action_delta_min_traj_index
                        )
                        & (
                            behavior_delta
                            >= self.anchor_action_delta_action_delta_min
                        )
                    )
                    quantile = self.anchor_action_delta_action_delta_quantile
                    if quantile > 0.0 and torch.any(anchor_mask):
                        anchor_delta_threshold = torch.quantile(
                            behavior_delta[anchor_mask].detach(),
                            quantile,
                        )
                        anchor_mask = anchor_mask & (
                            behavior_delta >= anchor_delta_threshold
                        )
                    if (
                        self.anchor_action_delta_include_critical
                        and traj_action_is_critical is not None
                    ):
                        anchor_mask = anchor_mask | (
                            traj_action_is_critical & loss_valid_mask
                        )

                    anchor_delta_valid_ratio = anchor_mask.detach().float().mean()
                    if torch.any(loss_valid_mask):
                        anchor_delta_behavior_mean = behavior_delta[
                            loss_valid_mask
                        ].mean()
                    if torch.any(anchor_mask):
                        model_delta = (
                            correct_anchor_output["action"]
                            - swapped_anchor_output["action"]
                        )
                        signed_delta_loss = F.mse_loss(
                            model_delta,
                            target_delta,
                            reduction="none",
                        )
                        signed_delta_loss = einops.reduce(
                            signed_delta_loss,
                            "b t ... -> b t",
                            "mean",
                        )
                        anchor_delta_raw = signed_delta_loss[anchor_mask].mean()

                        flat_model_delta = model_delta.flatten(start_dim=2)
                        flat_target_delta = target_delta.flatten(start_dim=2)
                        cosine = F.cosine_similarity(
                            flat_model_delta,
                            flat_target_delta,
                            dim=-1,
                        )
                        anchor_delta_cosine = cosine[anchor_mask].mean()
                        anchor_delta_cosine_raw = (
                            1.0 - anchor_delta_cosine
                        )
                        anchor_delta_loss = (
                            anchor_delta_raw
                            * self.anchor_action_delta_loss_weight
                            + anchor_delta_cosine_raw
                            * self.anchor_action_delta_cosine_loss_weight
                        )
                        anchor_delta_model_rmse = (
                            model_delta[anchor_mask].detach().pow(2).mean().sqrt()
                        )
                        anchor_delta_target_rmse = (
                            target_delta[anchor_mask].detach().pow(2).mean().sqrt()
                        )
                        anchor_delta_swap_rmse = (
                            F.mse_loss(
                                correct_anchor_output["action"],
                                swapped_anchor_output["action"],
                                reduction="none",
                            )[anchor_mask]
                            .detach()
                            .mean()
                            .sqrt()
                        )

            loss["anchor_action_delta"] = anchor_delta_loss
            loss["diagnostic/anchor_action_delta_raw"] = (
                anchor_delta_raw.detach()
            )
            loss["diagnostic/anchor_action_delta_cosine_raw"] = (
                anchor_delta_cosine_raw.detach()
            )
            loss["diagnostic/anchor_action_delta_valid_ratio"] = (
                anchor_delta_valid_ratio.detach()
            )
            loss["diagnostic/anchor_action_delta_action_delta_mean"] = (
                anchor_delta_behavior_mean.detach()
            )
            loss["diagnostic/anchor_action_delta_action_delta_threshold"] = (
                anchor_delta_threshold.detach()
            )
            loss["diagnostic/anchor_action_delta_model_rmse"] = (
                anchor_delta_model_rmse.detach()
            )
            loss["diagnostic/anchor_action_delta_target_rmse"] = (
                anchor_delta_target_rmse.detach()
            )
            loss["diagnostic/anchor_action_delta_swap_rmse"] = (
                anchor_delta_swap_rmse.detach()
            )
            loss["diagnostic/anchor_action_delta_cosine"] = (
                anchor_delta_cosine.detach()
            )
            loss["diagnostic/anchor_action_delta_weight"] = (
                action_loss.new_tensor(self.anchor_action_delta_loss_weight)
            )
            loss["diagnostic/anchor_action_delta_cosine_weight"] = (
                action_loss.new_tensor(
                    self.anchor_action_delta_cosine_loss_weight
                )
            )

        if (
            self.decision_window_contrastive_loss_weight > 0.0
            or self.decision_window_invariance_loss_weight > 0.0
            or self.functional_preservation_loss_weight > 0.0
            or self.teacher_preservation_loss_weight > 0.0
        ):
            decision_loss = action_loss.new_zeros(())
            decision_raw = action_loss.new_zeros(())
            decision_correct = action_loss.new_zeros(())
            decision_mismatched = action_loss.new_zeros(())
            decision_active_ratio = action_loss.new_zeros(())
            decision_valid_ratio = action_loss.new_zeros(())
            decision_invariance_loss = action_loss.new_zeros(())
            decision_invariance_raw = action_loss.new_zeros(())
            decision_invariance_valid_ratio = action_loss.new_zeros(())
            functional_preservation_loss = action_loss.new_zeros(())
            functional_preservation_raw = action_loss.new_zeros(())
            functional_preservation_valid_ratio = action_loss.new_zeros(())
            teacher_preservation_loss = action_loss.new_zeros(())
            teacher_preservation_raw = action_loss.new_zeros(())
            teacher_preservation_valid_ratio = action_loss.new_zeros(())
            decision_delta_mean = action_loss.new_zeros(())
            decision_delta_threshold = action_loss.new_tensor(
                self.decision_window_action_delta_min
            )

            (
                mismatched_data_dict,
                decision_target_mask,
                source_indices,
            ) = self._build_decision_window_contrastive_data_dict(
                normalized_batch,
                data_dict,
            )
            if (
                mismatched_data_dict is not None
                and decision_target_mask is not None
                and source_indices is not None
            ):
                mismatched_model_output = self.denoising_network.parallel_forward(
                    mismatched_data_dict
                )
                mismatched_action_loss = F.mse_loss(
                    mismatched_model_output["action"],
                    target["action"],
                    reduction="none",
                )
                mismatched_action_loss = einops.reduce(
                    mismatched_action_loss,
                    "b t ... -> b t",
                    "mean",
                )
                target_traj_indices = self._get_effective_target_traj_indices(
                    data_dict,
                    batch_size,
                    traj_num,
                    data_dict["noisy_action"].device,
                )
                behavior_window_mask, behavior_delta, behavior_threshold = (
                    self._build_decision_window_mask(
                        target["action"].detach(),
                        source_indices,
                        decision_target_mask & loss_valid_mask,
                        target_traj_indices,
                        traj_action_is_critical,
                    )
                )
                decision_valid_ratio = behavior_window_mask.detach().float().mean()
                if torch.any(decision_target_mask & loss_valid_mask):
                    decision_delta_mean = behavior_delta[
                        decision_target_mask & loss_valid_mask
                    ].mean()
                decision_delta_threshold = behavior_threshold
                decision_candidate_mask = (
                    decision_target_mask
                    & loss_valid_mask
                    & (target_traj_indices >= self.decision_window_min_traj_index)
                    & (behavior_delta >= self.decision_window_action_delta_min)
                )
                if torch.any(behavior_window_mask):
                    correct_for_rank = (
                        action_loss.detach()
                        if self.decision_window_contrastive_detach_correct_loss
                        else action_loss
                    )
                    rank_loss = F.relu(
                        self.decision_window_contrastive_margin
                        + correct_for_rank
                        - mismatched_action_loss
                    )
                    valid_rank_loss = rank_loss[behavior_window_mask]
                    decision_raw = valid_rank_loss.mean()
                    decision_loss = (
                        decision_raw * self.decision_window_contrastive_loss_weight
                    )
                    decision_correct = (
                        action_loss[behavior_window_mask].detach().mean()
                    )
                    decision_mismatched = (
                        mismatched_action_loss[behavior_window_mask].detach().mean()
                    )
                    decision_active_ratio = (
                        valid_rank_loss.detach().gt(0).float().mean()
                    )
                if self.decision_window_invariance_loss_weight > 0.0:
                    invariance_window_mask = (
                        decision_candidate_mask & (~behavior_window_mask)
                    )
                    decision_invariance_valid_ratio = (
                        invariance_window_mask.detach().float().mean()
                    )
                    if torch.any(invariance_window_mask):
                        correct_action = (
                            model_output["action"].detach()
                            if self.decision_window_invariance_detach_correct
                            else model_output["action"]
                        )
                        invariance_action_loss = F.mse_loss(
                            mismatched_model_output["action"],
                            correct_action,
                            reduction="none",
                        )
                        invariance_action_loss = einops.reduce(
                            invariance_action_loss,
                            "b t ... -> b t",
                            "mean",
                        )
                        decision_invariance_raw = invariance_action_loss[
                            invariance_window_mask
                        ].mean()
                        decision_invariance_loss = (
                            decision_invariance_raw
                            * self.decision_window_invariance_loss_weight
                        )
                if self.functional_preservation_loss_weight > 0.0:
                    preservation_window_mask = (
                        decision_candidate_mask & (~behavior_window_mask)
                    )
                    functional_preservation_valid_ratio = (
                        preservation_window_mask.detach().float().mean()
                    )
                    if torch.any(preservation_window_mask):
                        if self.functional_preservation_detach_reference:
                            with torch.no_grad():
                                reference_model_output = (
                                    self.denoising_network.parallel_forward(
                                        data_dict,
                                        disable_state_token_action_adapters=True,
                                    )
                                )
                            reference_action = reference_model_output["action"].detach()
                        else:
                            reference_model_output = (
                                self.denoising_network.parallel_forward(
                                    data_dict,
                                    disable_state_token_action_adapters=True,
                                )
                            )
                            reference_action = reference_model_output["action"]
                        preservation_action_loss = F.mse_loss(
                            model_output["action"],
                            reference_action,
                            reduction="none",
                        )
                        preservation_action_loss = einops.reduce(
                            preservation_action_loss,
                            "b t ... -> b t",
                            "mean",
                        )
                        functional_preservation_raw = preservation_action_loss[
                            preservation_window_mask
                        ].mean()
                        functional_preservation_loss = (
                            functional_preservation_raw
                            * self.functional_preservation_loss_weight
                        )
                if self.teacher_preservation_loss_weight > 0.0:
                    teacher_window_mask = (
                        decision_candidate_mask & (~behavior_window_mask)
                    )
                    teacher_preservation_valid_ratio = (
                        teacher_window_mask.detach().float().mean()
                    )
                    if torch.any(teacher_window_mask):
                        assert teacher_denoising_network is not None
                        with torch.no_grad():
                            teacher_model_output = (
                                teacher_denoising_network.parallel_forward(
                                    data_dict,
                                    disable_state_token_action_adapters=(
                                        self.teacher_preservation_disable_action_adapters
                                    ),
                                )
                            )
                        teacher_action_loss = F.mse_loss(
                            model_output["action"],
                            teacher_model_output["action"].detach(),
                            reduction="none",
                        )
                        teacher_action_loss = einops.reduce(
                            teacher_action_loss,
                            "b t ... -> b t",
                            "mean",
                        )
                        teacher_preservation_raw = teacher_action_loss[
                            teacher_window_mask
                        ].mean()
                        teacher_preservation_loss = (
                            teacher_preservation_raw
                            * self.teacher_preservation_loss_weight
                        )

            loss["decision_window_contrastive"] = decision_loss
            if self.decision_window_invariance_loss_weight > 0.0:
                loss["decision_window_invariance"] = decision_invariance_loss
            if self.functional_preservation_loss_weight > 0.0:
                loss["functional_preservation"] = functional_preservation_loss
            if self.teacher_preservation_loss_weight > 0.0:
                loss["teacher_preservation"] = teacher_preservation_loss
            loss["diagnostic/decision_window_contrastive_raw"] = (
                decision_raw.detach()
            )
            loss["diagnostic/decision_window_contrastive_correct_loss"] = (
                decision_correct.detach()
            )
            loss["diagnostic/decision_window_contrastive_mismatched_loss"] = (
                decision_mismatched.detach()
            )
            loss["diagnostic/decision_window_contrastive_active_ratio"] = (
                decision_active_ratio.detach()
            )
            loss["diagnostic/decision_window_contrastive_valid_ratio"] = (
                decision_valid_ratio.detach()
            )
            loss["diagnostic/decision_window_invariance_raw"] = (
                decision_invariance_raw.detach()
            )
            loss["diagnostic/decision_window_invariance_valid_ratio"] = (
                decision_invariance_valid_ratio.detach()
            )
            loss["diagnostic/decision_window_invariance_weight"] = (
                action_loss.new_tensor(self.decision_window_invariance_loss_weight)
            )
            loss["diagnostic/functional_preservation_raw"] = (
                functional_preservation_raw.detach()
            )
            loss["diagnostic/functional_preservation_valid_ratio"] = (
                functional_preservation_valid_ratio.detach()
            )
            loss["diagnostic/functional_preservation_weight"] = (
                action_loss.new_tensor(self.functional_preservation_loss_weight)
            )
            loss["diagnostic/teacher_preservation_raw"] = (
                teacher_preservation_raw.detach()
            )
            loss["diagnostic/teacher_preservation_valid_ratio"] = (
                teacher_preservation_valid_ratio.detach()
            )
            loss["diagnostic/teacher_preservation_weight"] = (
                action_loss.new_tensor(self.teacher_preservation_loss_weight)
            )
            loss["diagnostic/decision_window_action_delta_mean"] = (
                decision_delta_mean.detach()
            )
            loss["diagnostic/decision_window_action_delta_threshold"] = (
                decision_delta_threshold.detach()
            )
            loss["diagnostic/decision_window_contrastive_weight"] = (
                action_loss.new_tensor(self.decision_window_contrastive_loss_weight)
            )
            loss["diagnostic/decision_window_contrastive_margin"] = (
                action_loss.new_tensor(self.decision_window_contrastive_margin)
            )

        if loss_valid_mask.sum() == 0:
            loss["action"] = action_loss.sum() / action_loss.numel()
        else:
            loss["action"] = action_loss.sum() / loss_valid_mask.sum()



        self._add_base_anchor_loss(loss)
        return loss

    def reset(self):
        super().reset()
        self.history_noisy_actions_dict = {}
        self.history_img_features_dict = {}
        self.history_evidence_features_dict = {}
        self.late_cue_anchor_img_features_dict = {}
        self.visual_memory_carrier_img_features_dict = {}
        self.state_token_dict = {}
        self.state_token_seen_dict = {}
        self.recorded_data_dicts = {}
