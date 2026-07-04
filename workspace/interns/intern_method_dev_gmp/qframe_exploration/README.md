# Q-frame Exploration

This branch packages the Mikasa Q-frame evidence-memory exploration code for
review. The branch keeps the original GMP policy/network targets intact and adds
Mikasa-specific bypass implementations for the experimental methods.

## Code Layout

- `imitation_learning/models/denoising_networks/mikasa_memory_transformer.py`
  is the Mikasa method-development copy of the memory transformer.
- `imitation_learning/policies/mikasa_history_denoising_policy.py` is the
  matching Mikasa policy wrapper.
- `imitation_learning/models/denoising_networks/mikasa_evidence_memory_transformer.py`
  adds Q-frame evidence memory on top of the Mikasa transformer.
- `imitation_learning/models/evidence_selection.py` implements causal candidate
  subsampling and top-k evidence selection.
- `imitation_learning/models/encoders/longclip_image_encoder.py`,
  `longclip_text_encoder.py`, and `vjepa2_image_encoder.py` provide frozen
  evidence encoders for LongCLIP and VJEPA2 variants.
- `imitation_learning/utils/qframe_query_modes.py` contains the offline query
  scoring helpers used by the debug reranker.
- `scripts/rerank_qframe_debug_with_text.py` reranks saved Q-frame debug grids
  with image-only, text-only, or image-text fused LongCLIP queries.

The original GMP files
`imitation_learning/models/denoising_networks/memory_transformer.py`,
`imitation_learning/policies/history_denoising_policy.py`, and the original GMP
configs are not changed by this branch.

## Method Settings

The reviewed Q-frame setting uses bounded causal history only:

| task group | max candidates | high-res selected | low-res selected |
|---|---:|---:|---:|
| RememberColor series | 8 | top 2 | top 2 |
| InterceptMedium | 12 | top 2 | top 2 |
| ShellGameTouch | 8 | top 2 | top 2 |

Selection first samples an evenly spaced causal candidate set from already
observed history rows, then ranks candidates by cosine similarity between the
current query and history evidence keys. High-res rows keep all visual tokens;
low-res rows are pooled to one token per row.

## Main Configs

- `diffusion_mikasa_qframe_evidence_memory_transformer.yaml`: Q-frame with GMP
  visual history features as keys.
- `diffusion_mikasa_qframe_evidence_memory_transformer_intercept_medium.yaml`:
  Q-frame with the Intercept candidate budget set to 12.
- `diffusion_mikasa_qframe_longclip_evidence_memory_transformer.yaml`: Q-frame
  with frozen LongCLIP image evidence keys.
- `diffusion_mikasa_qframe_vjepa2_evidence_memory_transformer.yaml`: Q-frame
  with frozen VJEPA2 image evidence keys.

LongCLIP and VJEPA2 weights are expected under the method-development base:

- `/mnt/3fs1/data/tingwen.du/icra_method_dev/deps/Long-CLIP/longclip-L.pt`
- `/mnt/3fs1/data/tingwen.du/icra_method_dev/deps/vjepa2-vitl-fpc64-256`

## Launchers

The copied launchers are the exact experiment helpers used during exploration.
They default to the shared experiment copy under
`/mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/qframe_evidence_memory_20260630`.
When running directly from this branch, set:

```bash
export CODE_DIR=/work-agents/intern_method_dev_gmp/gated-memory-policy/imitation-learning-policies
```

Smoke example:

```bash
cd /work-agents/intern_method_dev_gmp/gated-memory-policy
MODE=--dry-run RUN_SET=smoke \
  bash workspace/interns/intern_method_dev_gmp/qframe_exploration/launch_qframe_evidence_8lane_20260630.sh
```

The 5-seed rollout helpers live in
`workspace/interns/intern_method_dev_gmp/qframe_exploration/scripts/` and use
the shared ICRA experiment/runs/logs paths.

## Query Modes

The rollout/debug path can switch LongCLIP query behavior with environment
variables:

```bash
MIKASA_EVAL_QFRAME_QUERY_MODE=image_only
MIKASA_EVAL_QFRAME_QUERY_MODE=text_only
MIKASA_EVAL_QFRAME_QUERY_MODE=image_text_fused
MIKASA_EVAL_QFRAME_TEXT_INSTRUCTION="Observe the cube's color, wait, then touch the cube of the same color."
MIKASA_EVAL_QFRAME_TEXT_ALPHA=0.5
MIKASA_EVAL_QFRAME_LONGCLIP_WEIGHTS=/mnt/3fs1/data/tingwen.du/icra_method_dev/deps/Long-CLIP/longclip-L.pt
```

`image_text_fused` scores candidates with
`alpha * image_score + (1 - alpha) * text_score`.

## Current 5-seed Results

Metric is `success_once mean +/- std` over five rollout seeds unless noted.

| method | RC3 | RC5 | RC9 | Intercept | Shell | mean | RC mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| GMP release | 0.814 +/- 0.047 | 0.668 +/- 0.049 | 0.180 +/- 0.032 | 0.840 +/- 0.035 | 0.990 +/- 0.007 | 0.698 | 0.554 |
| Q-frame v1 | 0.570 +/- 0.160 | 0.660 +/- 0.072 | 0.330 +/- 0.025 | 0.776 +/- 0.035 | 0.860 +/- 0.071 | 0.639 | 0.520 |
| LongCLIP image-only | 0.634 +/- 0.100 | 0.668 +/- 0.076 | 0.318 +/- 0.025 | 0.782 +/- 0.034 | 0.862 +/- 0.068 | 0.653 | 0.540 |
| LongCLIP image-text fused | 0.634 +/- 0.103 | 0.670 +/- 0.079 | 0.318 +/- 0.022 | 0.780 +/- 0.038 | 0.860 +/- 0.071 | 0.652 | 0.541 |
| VJEPA2 | 0.562 +/- 0.158 | 0.652 +/- 0.056 | 0.346 +/- 0.030 | 0.786 +/- 0.048 | 0.864 +/- 0.063 | 0.642 | 0.520 |
| first-image pin | 0.834 +/- 0.032 | 0.684 +/- 0.059 | 0.170 +/- 0.010 | 0.786 +/- 0.030 | 0.474 +/- 0.117 | 0.590 | 0.563 |

These results are for exploration and are included to make the branch reviewable;
large checkpoints, rollout videos, wandb files, and generated debug images are
kept out of git.
