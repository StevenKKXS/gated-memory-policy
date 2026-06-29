# Mikasa Memory Method Archive

Date: 2026-06-29

## Summary

This branch consolidates the Mikasa memory-method exploration into a reviewable archive while preserving the original GMP baseline as a stable reproducible target. The main design choice is a bypass implementation: Mikasa methods use new transformer, policy, visual-carrier, and config entrypoints, while existing GMP files and config defaults remain unchanged.

## Method Results

| Method | Compact mean | Repo-local policy config | Notes |
|---|---:|---|---|
| no-memory | 0.408 | `diffusion_transformer` | Base diffusion no-history line. |
| top1 retrieval | 0.564 | `diffusion_mikasa_retrieval_memory_transformer` | Uses `history_retrieval_topk=1`; some historical notes round/report this as 0.565. |
| selector | 0.654 | `diffusion_mikasa_visual_selector_late_anchor_memory` | Visual selector carrier plus late cue anchor. |
| clean GRU | 0.668 | `diffusion_mikasa_visual_gru_late_anchor_memory` | Recommended future method-development baseline from this archive. |
| conditional GRU burn-in | follow-up | `diffusion_mikasa_visual_gru_late_anchor_memory` | Adds conditional loss masking; kept as follow-up rather than replacing clean GRU. |
| late cue / direct anchor | 0.666 repaired; 0.736-0.738 historical | `diffusion_mikasa_start_anchored_direct_anchor_memory` | Historical high-score line is protocol-sensitive; repaired 2026-06-19 line is the strict comparison row. |

Task-level result tables are stored in:

- `experiments/mikasa_method_archive/results/20260620_nomem_top1_selector_gru_compact.tsv`
- `experiments/mikasa_method_archive/results/20260620_selector_gru_seed42_summary.tsv`
- `experiments/mikasa_method_archive/results/late_cue_direct_anchor_notes.tsv`

The five-task launcher matrix is stored in:

- `experiments/mikasa_method_archive/manifests/mikasa_5task_method_matrix.tsv`

## Source Code Provenance

| Method | Historical source code path |
|---|---|
| no-memory | `/mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/code/imitation-learning-policies_anchor_carrier_clear_20260619` |
| top1 retrieval | `/mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/code/imitation-learning-policies_remdp` |
| selector | `/mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/code/imitation-learning-policies_visual_selector_carrier_20260619` |
| clean GRU | `/mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/code/imitation-learning-policies_visual_gru_carrier_clean_20260619` |
| conditional GRU burn-in | `/mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/code/imitation-learning-policies_visual_gru_carrier_clean_burnin_20260620` |
| late cue / direct anchor | `/mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/code/imitation-learning-policies_remdp_direct_anchor_20260611` |
| repaired late-direct / no-memory | `/mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/code/imitation-learning-policies_anchor_carrier_clear_20260619` |

The machine-readable provenance tables are:

- `experiments/mikasa_method_archive/provenance/method_code_sources.tsv`
- `experiments/mikasa_method_archive/provenance/result_sources.tsv`
- `experiments/mikasa_method_archive/provenance/checkpoint_sources.tsv`

## Unified Implementation

The code landed in the repo-local `imitation-learning-policies/` tree through Mikasa-specific entrypoints:

- `imitation-learning-policies/imitation_learning/models/denoising_networks/mikasa_memory_transformer.py`
  - `MikasaMemoryTransformer`
  - top1 retrieval controls
  - late-cue/direct-anchor latents and action prehead adapter wiring
  - visual selector and GRU carrier hooks
- `imitation-learning-policies/imitation_learning/policies/mikasa_history_denoising_policy.py`
  - `MikasaHistoryDenoisingPolicy`
  - Mikasa history cache plumbing
  - conditional burn-in loss mask
- `imitation-learning-policies/imitation_learning/models/visual_memory_carriers.py`
  - `LearnedLateCueSelector`
  - `VisualGRUMemoryCarrier`
  - clean GRU wrapper that avoids cuDNN flattening side effects
- `imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_mikasa_*.yaml`
  - Mikasa-only configs targeting the new policy and transformer classes

## GMP Files Left Untouched

These original GMP compatibility files were intentionally not modified:

- `imitation-learning-policies/imitation_learning/models/denoising_networks/memory_transformer.py`
- `imitation-learning-policies/imitation_learning/policies/history_denoising_policy.py`
- `imitation-learning-policies/imitation_learning/configs/workspace/policy/denoising_network/memory_transformer.yaml`
- `imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_memory_transformer.yaml`
- `imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_gated_transformer.yaml`
- `imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_binary_gated_transformer.yaml`
- `imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_continuous_gated_transformer.yaml`
- `imitation-learning-policies/imitation_learning/configs/workspace/policy/flow_memory_transformer.yaml`
- `imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_memory_transformer_large.yaml`

Compatibility is guarded by:

- `imitation-learning-policies/tests/test_gmp_baseline_compatibility.py`

## Artifacts Kept By Reference

Large artifacts are intentionally kept out of Git and referenced by path under:

`/mnt/3fs1/data/tingwen.du/icra_method_dev`

This includes checkpoints, eval summaries, training logs, generated manifests, rollout videos, and datasets. Historical baseline paths under `/mnt/3fs1/data/tingwen.du/gated-memory-policy-*` remain read-only references.

## Launchers

Historical launchers are copied unmodified under:

- `experiments/mikasa_method_archive/legacy_launchers/`

Repo-local normalized launchers are under:

- `experiments/mikasa_method_archive/launchers/`

The normalized launchers export `PYTHONPATH` to this checkout and default outputs to the ICRA method-development base. The no-memory launcher stays on `diffusion_transformer`; Mikasa memory methods use `diffusion_mikasa_*` configs.

## Known Caveats

- Top1 InterceptMedium has historical rerun variance in handoff notes; the compact archive pins the strict table row at 0.810 and the five-task mean at 0.564.
- The late-cue/direct-anchor high-score line above 0.70 is kept as an exploratory, protocol-sensitive idea source. The repaired 2026-06-19 seed42 line is the stricter comparison row at 0.666 mean.
- Conditional GRU burn-in is archived because it is useful implementation work, but it is not promoted over clean GRU as the primary baseline.
- The archive records absolute ICRA-base paths for large artifacts; it does not guarantee those artifacts are portable outside this storage layout.

## Verification

Focused tests cover:

- original GMP config targets and import paths
- Mikasa config targets
- provenance table schema
- visual selector and GRU carrier behavior
- conditional burn-in loss masking
- repo-local launcher dry-run boundary checks

Task 8 smoke also imports original GMP and Mikasa classes with:

`/mnt/3fs1/data/tingwen.du/icra_method_dev/envs/imitation-py310-h200-headless/bin/python`
