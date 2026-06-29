from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "imitation_learning" / "configs" / "workspace" / "policy"
MIKASA_NETWORK_CFG = POLICY_DIR / "denoising_network" / "mikasa_memory_transformer.yaml"


def _load_yaml(path: Path):
    return yaml.safe_load(path.read_text())


def test_mikasa_memory_transformer_exposes_retrieval_and_late_cue_defaults():
    cfg = _load_yaml(MIKASA_NETWORK_CFG)
    assert cfg["_target_"] == (
        "imitation_learning.models.denoising_networks.mikasa_memory_transformer."
        "MikasaMemoryTransformer"
    )
    assert "history_retrieval_topk" in cfg
    assert "late_cue_anchor_enabled" in cfg
    assert "late_cue_anchor_len" in cfg
    assert "late_cue_anchor_causal_mask" in cfg


def test_mikasa_base_policy_uses_mikasa_targets():
    cfg = _load_yaml(POLICY_DIR / "diffusion_mikasa_memory_transformer.yaml")
    assert cfg["_target_"] == (
        "imitation_learning.policies.mikasa_history_denoising_policy."
        "MikasaHistoryDenoisingPolicy"
    )
    assert {
        "override denoising_network@denoising_network_partial": "mikasa_memory_transformer"
    } in cfg["defaults"]


def test_retrieval_config_uses_memory_policy_and_topk_override_path():
    cfg = _load_yaml(POLICY_DIR / "diffusion_mikasa_retrieval_memory_transformer.yaml")
    assert "diffusion_mikasa_memory_transformer" in cfg["defaults"]
    assert "denoising_network_partial" in cfg
    network = cfg["denoising_network_partial"]
    assert network.get("history_retrieval_topk") in (1, 4)


def test_direct_anchor_config_enables_late_cue_anchor():
    cfg = _load_yaml(
        POLICY_DIR / "diffusion_mikasa_start_anchored_direct_anchor_memory.yaml"
    )
    assert "diffusion_mikasa_memory_transformer" in cfg["defaults"]
    network = cfg["denoising_network_partial"]
    assert network["late_cue_anchor_enabled"] is True
    assert network["late_cue_anchor_len"] >= 1


def test_selector_config_uses_mikasa_base_and_selector_carrier():
    cfg = _load_yaml(POLICY_DIR / "diffusion_mikasa_visual_selector_late_anchor_memory.yaml")
    assert "diffusion_mikasa_memory_transformer" in cfg["defaults"]
    network = cfg["denoising_network_partial"]
    assert network["visual_memory_carrier_type"] == "selector"
    assert network["late_cue_anchor_enabled"] is True


def test_gru_config_uses_mikasa_base_and_gru_carrier():
    cfg = _load_yaml(POLICY_DIR / "diffusion_mikasa_visual_gru_late_anchor_memory.yaml")
    assert "diffusion_mikasa_memory_transformer" in cfg["defaults"]
    network = cfg["denoising_network_partial"]
    assert network["visual_memory_carrier_type"] == "gru"
    assert network["late_cue_anchor_enabled"] is True
