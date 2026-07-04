from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from imitation_learning.models.evidence_selection import (
    select_qframe_causal_evidence_masks,
)
from imitation_learning.models.denoising_networks.mikasa_evidence_memory_transformer import (
    MikasaEvidenceMemoryTransformer,
)


def test_qframe_selector_limits_candidates_to_valid_causal_history():
    query = torch.tensor([[1.0, 0.0]])
    history_latents = torch.randn(1, 6, 1, 2)
    history_mask = torch.tensor([[False, True, True, True, True, True]])

    high_mask, low_mask, candidate_mask = select_qframe_causal_evidence_masks(
        query,
        history_latents,
        history_mask,
        max_candidates=3,
        high_topk=1,
        low_topk=1,
    )

    assert candidate_mask.shape == (1, 6)
    assert candidate_mask.sum().item() <= 3
    assert not candidate_mask[0, 0]
    assert torch.equal(high_mask & ~candidate_mask, torch.zeros_like(high_mask))
    assert torch.equal(low_mask & ~candidate_mask, torch.zeros_like(low_mask))
    assert not (high_mask & low_mask).any()


def test_qframe_selector_splits_high_and_low_resolution_topk_by_query_score():
    query = torch.tensor([[1.0, 0.0]])
    history_latents = torch.tensor(
        [
            [
                [[0.0, 1.0]],
                [[1.0, 0.0]],
                [[0.8, 0.6]],
                [[-1.0, 0.0]],
                [[0.6, 0.8]],
            ]
        ]
    )
    history_mask = torch.ones(1, 5, dtype=torch.bool)

    high_mask, low_mask, candidate_mask = select_qframe_causal_evidence_masks(
        query,
        history_latents,
        history_mask,
        max_candidates=5,
        high_topk=2,
        low_topk=2,
    )

    assert torch.equal(candidate_mask, history_mask)
    assert torch.equal(
        high_mask,
        torch.tensor([[False, True, True, False, False]]),
    )
    assert torch.equal(
        low_mask,
        torch.tensor([[True, False, False, False, True]]),
    )


def test_qframe_selector_can_rank_with_external_evidence_features():
    query = torch.tensor([[1.0, 0.0]])
    history_latents = torch.zeros(1, 4, 1, 3)
    history_mask = torch.ones(1, 4, dtype=torch.bool)
    history_evidence_features = torch.tensor(
        [
            [
                [0.0, 1.0],
                [0.2, 0.8],
                [1.0, 0.0],
                [0.9, 0.1],
            ]
        ]
    )

    high_mask, low_mask, candidate_mask = select_qframe_causal_evidence_masks(
        query,
        history_latents,
        history_mask,
        max_candidates=4,
        high_topk=1,
        low_topk=1,
        history_evidence_features=history_evidence_features,
    )

    assert torch.equal(candidate_mask, history_mask)
    assert torch.equal(
        high_mask,
        torch.tensor([[False, False, True, False]]),
    )
    assert torch.equal(
        low_mask,
        torch.tensor([[False, False, False, True]]),
    )


def test_qframe_selector_handles_empty_history_without_selecting_dummy_tokens():
    query = torch.randn(2, 4)
    history_latents = torch.randn(2, 3, 2, 4)
    history_mask = torch.zeros(2, 3, dtype=torch.bool)

    high_mask, low_mask, candidate_mask = select_qframe_causal_evidence_masks(
        query,
        history_latents,
        history_mask,
        max_candidates=8,
        high_topk=2,
        low_topk=2,
    )

    assert not high_mask.any()
    assert not low_mask.any()
    assert not candidate_mask.any()


def _small_evidence_transformer() -> MikasaEvidenceMemoryTransformer:
    return MikasaEvidenceMemoryTransformer(
        action_dim=3,
        action_token_num=2,
        global_cond_dim=5,
        global_cond_token_num=1,
        local_cond_dim=4,
        local_cond_token_num=1,
        seed=7,
        head_num=4,
        layer_num=2,
        hidden_dim=16,
        projector_type="linear",
        global_cond_pos_emb_type="1d",
        max_history_len=4,
        freeze_non_history_modules=False,
        history_attention_type="token_wise",
        record_data_entries=[
            "qframe_evidence_memory_gate",
            "qframe_evidence_high_selected_ratio",
            "qframe_evidence_low_selected_ratio",
        ],
        ssmax_scaling_param=None,
        include_action_history=True,
        history_action_num_per_chunk=2,
        skip_history_attn=False,
        history_img_features_dim=6,
        history_img_features_token_num=1,
        evidence_memory_enabled=True,
        evidence_memory_num_last_blocks=1,
        evidence_candidate_max_num=3,
        evidence_high_topk=1,
        evidence_low_topk=1,
        evidence_memory_out_init_gain=0.0,
    )


def test_mikasa_evidence_transformer_forward_uses_eval_history_path():
    model = _small_evidence_transformer()
    data = {
        "noisy_action": torch.randn(2, 2, 3),
        "step": torch.tensor([1.0, 2.0]),
        "global_cond": torch.randn(2, 1, 5),
        "local_cond": torch.randn(2, 1, 4),
        "history_noisy_actions": torch.randn(2, 4, 2, 3),
        "history_img_features": torch.randn(2, 4, 1, 6),
        "history_mask": torch.tensor(
            [
                [False, True, True, True],
                [False, False, True, True],
            ]
        ),
        "evidence_query_features": torch.randn(2, 5),
        "history_evidence_features": torch.randn(2, 4, 5),
    }

    output = model(data)

    assert output["action"].shape == (2, 2, 3)
    assert "qframe_evidence_memory_gate" in model.recorded_data_dict
    assert len(model.recorded_data_dict["qframe_evidence_memory_gate"]) == 1


def test_mikasa_evidence_transformer_parallel_forward_uses_training_history_path():
    model = _small_evidence_transformer()
    batch_size = 2
    traj_num = 5
    data = {
        "noisy_action": torch.randn(batch_size, traj_num, 2, 3),
        "step": torch.tensor([[3.0], [5.0]]),
        "global_cond": torch.randn(batch_size, traj_num, 1, 5),
        "local_cond": torch.randn(batch_size, traj_num, 1, 4),
        "history_noisy_actions": torch.randn(batch_size, traj_num, 2, 3),
        "history_img_features": torch.randn(batch_size, traj_num, 1, 6),
        "history_mask": torch.ones(batch_size, traj_num, dtype=torch.bool),
        "evidence_query_features": torch.randn(batch_size, traj_num, 5),
        "history_evidence_features": torch.randn(batch_size, traj_num, 5),
    }

    output = model.parallel_forward(data)

    assert output["action"].shape == (batch_size, traj_num, 2, 3)
    assert "qframe_evidence_high_selected_ratio" in model.recorded_data_dict
