from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from imitation_learning.models.visual_memory_carriers import (
    LearnedLateCueSelector,
    VisualGRUMemoryCarrier,
)


def test_visual_gru_carrier_preserves_batch_and_token_dims():
    carrier = VisualGRUMemoryCarrier(
        input_dim=16,
        output_dim=32,
        token_num=1,
        hidden_dim=24,
        num_layers=1,
        dropout=0.0,
        max_len=8,
    )
    features = torch.randn(3, 5, 2, 16)
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, False, False, False, False],
            [True, True, True, True, True],
        ]
    )
    out, out_mask = carrier(features, mask)
    assert out.shape == (3, 5, 1, 32)
    assert out_mask.shape == (3, 5)
    assert torch.equal(out_mask, mask)


def test_selector_carrier_outputs_configured_token_count():
    selector = LearnedLateCueSelector(
        input_dim=16,
        output_dim=32,
        token_num=2,
        max_len=8,
    )
    features = torch.randn(4, 6, 3, 16)
    mask = torch.ones(4, 6, dtype=torch.bool)
    out, out_mask = selector(features, mask)
    assert out.shape == (4, 6, 2, 32)
    assert out_mask.shape == (4, 6)
