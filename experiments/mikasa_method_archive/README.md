# Mikasa Memory Method Archive

## Scope

This archive preserves code paths and reproduction metadata for:

- no-memory, compact mean 0.408
- top1 retrieval, compact mean 0.564
- selector, compact mean 0.654
- clean GRU, compact mean 0.668
- late cue / direct anchor, historical high-score line above 0.70 and repaired 2026-06-19 seed42 line 0.666
- conditional GRU burn-in follow-up experiments

## What Is In Git

- Mikasa-only repo-local implementation under `imitation-learning-policies/`
- GMP compatibility guard tests proving original GMP `_target_` strings and import paths remain unchanged
- Small result TSVs under `results/`
- Provenance TSVs under `provenance/`
- Historical launchers under `legacy_launchers/`
- Repo-local launchers under `launchers/`
- A five-task method matrix under `manifests/`

## What Is Not In Git

Large checkpoints, train logs, rollout videos, eval outputs, datasets, and generated manifests remain under:

`/mnt/3fs1/data/tingwen.du/icra_method_dev`

The archive records those artifacts by absolute path instead of vendoring them.

## Repo-Local Entry Points

The normalized launchers use this checkout by exporting `PYTHONPATH` to the repo-local `imitation-learning-policies` tree:

- `launchers/run_no_memory_5task.sh`
- `launchers/run_top1_retrieval_5task.sh`
- `launchers/run_selector_5task.sh`
- `launchers/run_gru_5task.sh`
- `launchers/run_gru_conditional_burnin_5task.sh`
- `launchers/run_late_cue_direct_anchor_5task.sh`

Each launcher supports `--dry-run` to print the training commands without starting training. Outputs default to:

- `$ICRA_BASE/runs/mikasa_method_archive`
- `$ICRA_BASE/logs/mikasa_method_archive`

## GMP Compatibility Boundary

Original GMP baseline files are intentionally left untouched so existing checkpoints, original training configs, original `_target_` strings, and import paths remain reproducible:

- `imitation-learning-policies/imitation_learning/models/denoising_networks/memory_transformer.py`
- `imitation-learning-policies/imitation_learning/policies/history_denoising_policy.py`
- `imitation-learning-policies/imitation_learning/configs/workspace/policy/denoising_network/memory_transformer.yaml`
- `imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_memory_transformer.yaml`

Mikasa exploration code enters through new bypass files and configs:

- `imitation_learning.models.denoising_networks.mikasa_memory_transformer.MikasaMemoryTransformer`
- `imitation_learning.policies.mikasa_history_denoising_policy.MikasaHistoryDenoisingPolicy`
- `imitation_learning.models.visual_memory_carriers`
- `diffusion_mikasa_*` policy configs

## Recommended Future Baseline

Use the clean GRU line as the primary method-development baseline because it has the strongest compact mean among no-memory, top1, selector, and GRU. Keep selector and late-cue/direct-anchor as comparison lines and idea sources.

## Late-Cue Caveat

The direct-anchor high-score line and repaired 2026-06-19 late-direct line are recorded separately. Treat the historical high-score direct-anchor line as exploratory and protocol-sensitive until it is reproduced under the same strict evaluation protocol as the compact table.
