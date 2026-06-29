from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from imitation_learning.policies.mikasa_history_denoising_policy import (
    _apply_burn_in_loss_mask,
)


def test_burn_in_zero_returns_original_mask():
    mask = torch.tensor([[True, True, False]])
    traj_idx = torch.tensor([[9, 10, 11]])
    out = _apply_burn_in_loss_mask(
        loss_valid_mask=mask,
        traj_idx=traj_idx,
        burn_in_start_id=8,
        burn_in_loss_traj_num=0,
        training_traj_indices=None,
    )
    assert torch.equal(out, mask)


def test_early_start_keeps_all_valid_slots():
    mask = torch.tensor([[True, True, True, False]])
    traj_idx = torch.tensor([[8, 9, 10, 11]])
    out = _apply_burn_in_loss_mask(
        loss_valid_mask=mask,
        traj_idx=traj_idx,
        burn_in_start_id=8,
        burn_in_loss_traj_num=2,
        training_traj_indices=None,
    )
    assert torch.equal(out, mask)


def test_late_start_masks_prefix_slots_only():
    mask = torch.tensor([[True, True, True, True]])
    traj_idx = torch.tensor([[9, 10, 11, 12]])
    out = _apply_burn_in_loss_mask(
        loss_valid_mask=mask,
        traj_idx=traj_idx,
        burn_in_start_id=8,
        burn_in_loss_traj_num=2,
        training_traj_indices=None,
    )
    expected = torch.tensor([[False, False, True, True]])
    assert torch.equal(out, expected)


def test_invalid_padding_remains_invalid_after_burn_in():
    mask = torch.tensor([[True, False, True, True]])
    traj_idx = torch.tensor([[20, 21, 22, 23]])
    out = _apply_burn_in_loss_mask(
        loss_valid_mask=mask,
        traj_idx=traj_idx,
        burn_in_start_id=8,
        burn_in_loss_traj_num=2,
        training_traj_indices=None,
    )
    expected = torch.tensor([[False, False, True, True]])
    assert torch.equal(out, expected)


def test_training_traj_indices_use_original_slot_indices():
    mask = torch.tensor([[True, True, True]])
    traj_idx = torch.tensor([[20, 21, 22, 23, 24]])
    out = _apply_burn_in_loss_mask(
        loss_valid_mask=mask,
        traj_idx=traj_idx,
        burn_in_start_id=8,
        burn_in_loss_traj_num=2,
        training_traj_indices=torch.tensor([0, 2, 4]),
    )
    expected = torch.tensor([[False, True, True]])
    assert torch.equal(out, expected)
