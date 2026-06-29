from pathlib import Path
import os
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
POLICY_DIR = ROOT / "imitation_learning" / "configs" / "workspace" / "policy"
TRAINING_PY = Path(
    "/mnt/3fs1/data/tingwen.du/icra_method_dev/envs/"
    "imitation-py310-h200-headless/bin/python"
)


def _load_yaml(path: Path):
    return yaml.safe_load(path.read_text())


def test_original_gmp_policy_target_is_unchanged():
    cfg = _load_yaml(POLICY_DIR / "diffusion_memory_transformer.yaml")
    assert cfg["_target_"] == (
        "imitation_learning.policies.history_denoising_policy.HistoryDenoisingPolicy"
    )
    assert cfg["defaults"][2] == {
        "override denoising_network@denoising_network_partial": "memory_transformer"
    }


def test_original_gmp_network_target_is_unchanged():
    cfg = _load_yaml(POLICY_DIR / "denoising_network" / "memory_transformer.yaml")
    assert cfg["_target_"] == (
        "imitation_learning.models.denoising_networks.memory_transformer."
        "MemoryTransformer"
    )
    assert "history_retrieval_topk" not in cfg
    assert "late_cue_anchor_enabled" not in cfg
    assert "visual_memory_carrier_type" not in cfg


def test_original_gmp_import_paths_still_resolve():
    code = """
from imitation_learning.models.denoising_networks.memory_transformer import MemoryTransformer
from imitation_learning.policies.history_denoising_policy import HistoryDenoisingPolicy
print(MemoryTransformer.__name__)
print(HistoryDenoisingPolicy.__name__)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    result = subprocess.run(
        [str(TRAINING_PY), "-c", code],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "MemoryTransformer",
        "HistoryDenoisingPolicy",
    ]
