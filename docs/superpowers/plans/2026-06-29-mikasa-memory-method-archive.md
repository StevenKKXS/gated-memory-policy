# Mikasa Memory Method Archive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the scattered Mikasa no-memory, top1 retrieval, selector, clean GRU, conditional burn-in, and late-cue/direct-anchor method implementations into one reviewable Git branch with reproducible experiment provenance while preserving the original GMP baseline as a stable, reproducible training/eval target.

**Architecture:** Use one unified `imitation-learning-policies/` code tree as the future Mikasa development target, but add Mikasa-specific policy, transformer, and config entrypoints beside the existing GMP implementation. Existing GMP core files, `_target_` strings, config defaults, import paths, and checkpoint load paths are compatibility boundaries and must remain unchanged. The archive stores shell launchers, small result tables, method manifests, and absolute ICRA-base paths to large checkpoints/eval summaries; it does not vendor large run directories or write outside the ICRA base.

**Tech Stack:** Python 3.10, PyTorch, Hydra/OmegaConf configs, bash launchers, pytest, existing training environment `/mnt/3fs1/data/tingwen.du/icra_method_dev/envs/imitation-py310-h200-headless/bin/python`.

## Global Constraints

- Primary repo: `/work-agents/intern_method_dev_gmp/gated-memory-policy`.
- Primary code tree: `/work-agents/intern_method_dev_gmp/gated-memory-policy/imitation-learning-policies`.
- ICRA method-development base: `/mnt/3fs1/data/tingwen.du/icra_method_dev`.
- Experiment outputs, logs, manifests, datasets, checkpoints, and large generated files stay under `/mnt/3fs1/data/tingwen.du/icra_method_dev`.
- Do not write to any other `/mnt` path.
- Historical baseline paths under `/mnt/3fs1/data/tingwen.du/gated-memory-policy-*` are read-only references.
- BCRNN is out of scope; only consolidate clean diffusion/memory/GRU paths.
- Commit after each independently reviewable task.
- After committing, push the feature branch.
- Do not push `master` or `main` directly; use a PR.
- Do not overwrite existing dirty user changes in `workspace/interns/intern_method_dev_gmp/knowledge.md`.
- Preserve original GMP reproducibility: existing checkpoints must still load through their original eval path, original training configs must still resolve, original `_target_` strings must still point to the same classes, and original import paths must remain valid.
- Do not modify these GMP compatibility files unless the user explicitly approves a separate GMP migration: `imitation-learning-policies/imitation_learning/models/denoising_networks/memory_transformer.py`, `imitation-learning-policies/imitation_learning/policies/history_denoising_policy.py`, `imitation-learning-policies/imitation_learning/configs/workspace/policy/denoising_network/memory_transformer.yaml`, `imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_memory_transformer.yaml`, or existing GMP-derived policy configs such as `diffusion_gated_transformer.yaml`, `diffusion_binary_gated_transformer.yaml`, `diffusion_continuous_gated_transformer.yaml`, `flow_memory_transformer.yaml`, and `diffusion_memory_transformer_large.yaml`.
- All Mikasa archive methods that need retrieval, late-cue/direct-anchor, selector, GRU carrier, or burn-in behavior must use new Mikasa-specific `_target_` paths and new Mikasa-specific config files.
- If a tiny shared utility seems useful to both GMP and Mikasa, put it in a new file and import it from Mikasa-only code first; do not retrofit old GMP files during this archive pass.
- Every code task must include a compatibility guard that verifies the protected GMP files have no diff and that their `_target_` strings remain unchanged.

---

## Current Source Inventory

| Method | Target archived result | Historical implementation source | Primary launch/result sources |
|---|---:|---|---|
| no-memory | mean 0.408 | `/mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/code/imitation-learning-policies_anchor_carrier_clear_20260619` | `launch_no_memory_diffusion_5task_20260619.sh`, `20260620_nomem_top1_selector_gru_compact.tsv` |
| top1 retrieval | mean 0.564/0.565 | `/mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/code/imitation-learning-policies_remdp` | `train_mikasa_task_memory.sh`, compact result TSV |
| selector | mean 0.654 | `/mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/code/imitation-learning-policies_visual_selector_carrier_20260619` | `launch_visual_carrier_selector_gru_5task_20260619.sh`, `20260620_selector_gru_seed42_summary.tsv` |
| clean GRU | mean 0.668 | `/mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/code/imitation-learning-policies_visual_gru_carrier_clean_20260619` | `launch_visual_carrier_selector_gru_5task_20260619.sh`, `20260620_selector_gru_seed42_summary.tsv` |
| conditional GRU burn-in | follow-up experiment line | `/mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/code/imitation-learning-policies_visual_gru_carrier_clean_burnin_20260620` | `launch_visual_gru_conditional_burnin_8lane_20260620.sh`, `final_results_with_eval_retries.tsv` |
| late cue / direct anchor | historical mean > 0.70, repaired line mean 0.666 | `/mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/code/imitation-learning-policies_remdp_direct_anchor_20260611` and `imitation-learning-policies_anchor_carrier_clear_20260619` | `launch_late_direct_anchor_5task_20260619.sh`, `launch_rc3_direct_anchor_protocol_rerun_20260619.sh`, direct-anchor handoff docs |

## File Structure

### Unified Code Tree

- Do not modify: `imitation-learning-policies/imitation_learning/models/denoising_networks/memory_transformer.py`
  - Remains the original GMP `MemoryTransformer` checkpoint/import target.
- Do not modify: `imitation-learning-policies/imitation_learning/policies/history_denoising_policy.py`
  - Remains the original GMP `HistoryDenoisingPolicy` checkpoint/import target.
- Do not modify: `imitation-learning-policies/imitation_learning/configs/workspace/policy/denoising_network/memory_transformer.yaml`
  - Remains the original GMP denoising network config with `_target_: imitation_learning.models.denoising_networks.memory_transformer.MemoryTransformer`.
- Do not modify: existing GMP policy configs, especially `diffusion_memory_transformer.yaml` and configs that inherit from it.
- Create: `imitation-learning-policies/imitation_learning/models/denoising_networks/mikasa_memory_transformer.py`
  - Holds Mikasa-only memory transformer support for top1 retrieval, late-cue/direct-anchor latents, visual selector carrier, visual GRU carrier, and action prehead adapter wiring.
  - The class name must be `MikasaMemoryTransformer`, and the config `_target_` must be `imitation_learning.models.denoising_networks.mikasa_memory_transformer.MikasaMemoryTransformer`.
- Create: `imitation-learning-policies/imitation_learning/policies/mikasa_history_denoising_policy.py`
  - Holds Mikasa-only history cache construction, late-cue anchor feature cache, visual carrier input plumbing, and conditional burn-in loss mask.
  - The class name must be `MikasaHistoryDenoisingPolicy`, and the config `_target_` must be `imitation_learning.policies.mikasa_history_denoising_policy.MikasaHistoryDenoisingPolicy`.
- Create: `imitation-learning-policies/imitation_learning/models/visual_memory_carriers.py`
  - Holds `LearnedLateCueSelector`, `VisualGRUMemoryCarrier`, and the no-cuDNN-flatten GRU wrapper used by the clean GRU implementation.
- Create: `imitation-learning-policies/imitation_learning/configs/workspace/policy/denoising_network/mikasa_memory_transformer.yaml`
  - Adds Mikasa default-off fields for retrieval, late cue, visual carrier, and adapter flags.
- Create: `imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_mikasa_memory_transformer.yaml`
  - Mikasa-only base policy config with `_target_: imitation_learning.policies.mikasa_history_denoising_policy.MikasaHistoryDenoisingPolicy` and `override denoising_network@denoising_network_partial: mikasa_memory_transformer`.
- Create: `imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_mikasa_retrieval_memory_transformer.yaml`
  - Repo-local top1 retrieval policy recipe.
- Create: `imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_mikasa_start_anchored_direct_anchor_memory.yaml`
  - Repo-local direct-anchor / late-cue policy recipe.
- Create: `imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_mikasa_visual_selector_late_anchor_memory.yaml`
  - Repo-local selector policy recipe.
- Create: `imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_mikasa_visual_gru_late_anchor_memory.yaml`
  - Repo-local clean GRU and conditional burn-in policy recipe.
- Modify only if required: `imitation-learning-policies/imitation_learning/datasets/multi_traj_dataset.py`
  - Ensure `traj_idx` or equivalent start-index metadata is available for conditional burn-in. If the current data path already exposes `traj_idx[:, 0]`, do not change the dataset.
  - Dataset changes must be additive and backward-compatible. Existing sample keys and shapes used by GMP must remain valid.
- Modify only if required: `imitation-learning-policies/imitation_learning/trainers/policy_trainer.py`
  - Preserve direct-anchor auxiliary loss logging only if it is present in the source code copy and needed by the selected late-cue recipe.
  - Trainer changes must be additive and no-op unless a Mikasa-only policy exposes the relevant auxiliary outputs.

### Tests

- Create: `imitation-learning-policies/tests/test_visual_memory_carriers.py`
  - Unit tests for selector and GRU carrier shape/mask behavior.
- Create: `imitation-learning-policies/tests/test_clean_gru_burn_in_loss_mask.py`
  - Unit tests for conditional burn-in masking: no burn-in when `burn_in_loss_traj_num=0`, no mask when `start_idx <= burn_in_start_id`, prefix mask when `start_idx > burn_in_start_id`, and preservation of invalid/padded slots.
- Create: `imitation-learning-policies/tests/test_mikasa_memory_archive_configs.py`
  - Static config tests ensuring all archived policy configs instantiate via Hydra/OmegaConf and preserve required keys.
- Create: `imitation-learning-policies/tests/test_gmp_baseline_compatibility.py`
  - Static compatibility tests ensuring original GMP `_target_` strings, original config files, and original import paths remain unchanged.
- Create: `imitation-learning-policies/tests/test_mikasa_memory_archive_provenance.py`
  - Static archive tests ensuring each method has code source, launcher source, result source, and large-artifact path entries.

### Experiment Archive

- Create: `experiments/mikasa_method_archive/README.md`
  - Human-facing reproduction guide for the five method families.
- Create: `experiments/mikasa_method_archive/provenance/method_code_sources.tsv`
  - One row per method/source code copy, with source path, intended branch target, and notes.
- Create: `experiments/mikasa_method_archive/provenance/result_sources.tsv`
  - One row per task/method result, with value, source TSV/summary path, and protocol note.
- Create: `experiments/mikasa_method_archive/provenance/checkpoint_sources.tsv`
  - One row per method/task checkpoint used for the archived score.
- Create: `experiments/mikasa_method_archive/results/20260620_nomem_top1_selector_gru_compact.tsv`
  - Small copied result table for no-memory/top1/selector/GRU.
- Create: `experiments/mikasa_method_archive/results/20260620_selector_gru_seed42_summary.tsv`
  - Small copied result/source table for selector/GRU/no-memory/late-direct repaired seed42.
- Create: `experiments/mikasa_method_archive/results/late_cue_direct_anchor_notes.tsv`
  - Explicitly separates historical high-score direct-anchor line from repaired 6/19 late-direct line.
- Create: `experiments/mikasa_method_archive/legacy_launchers/*.sh`
  - Unmodified historical shell launchers copied from ICRA base for audit.
- Create: `experiments/mikasa_method_archive/launchers/*.sh`
  - Repo-local normalized launchers that set `PYTHONPATH` to this repo's `imitation-learning-policies`.
- Create: `experiments/mikasa_method_archive/manifests/mikasa_5task_method_matrix.tsv`
  - Stable hyperparameter matrix: task, env, traj_num, max_history_len, action_len, traj_interval, sampling strategy, base checkpoint path, policy config.
- Create: `docs/experiments/2026-06-29-mikasa-memory-method-archive.md`
  - Concise narrative: what is consolidated, what is historical-only, what is a recommended future baseline.

---

### Task 1: Branch And Safety Baseline

**Files:**
- Read: repository status and existing dirty files.
- No code files modified.

**Interfaces:**
- Consumes: current repo state.
- Produces: clean execution branch name `intern_method_dev_gmp/mikasa-memory-method-archive` or an isolated worktree branch with the same name.

- [ ] **Step 1: Verify current branch and dirty state**

Run:

```bash
git status --short --branch
```

Expected:

```text
## intern_method_dev_gmp/visual-gru-burnin-clean
 M workspace/interns/intern_method_dev_gmp/knowledge.md
?? .superpowers/
?? docs/
```

- [ ] **Step 2: Create an isolated branch for archive work**

Preferred if keeping the current workspace untouched:

```bash
git worktree add ../gmp-mikasa-memory-method-archive -b intern_method_dev_gmp/mikasa-memory-method-archive HEAD
cd ../gmp-mikasa-memory-method-archive
```

Alternative if the user approves continuing in-place:

```bash
git switch -c intern_method_dev_gmp/mikasa-memory-method-archive
```

Expected: subsequent commits are not on `main` or `master`.

- [ ] **Step 3: Record local environment in AGENTS.md if missing**

Add this only if the branch worktree does not already contain equivalent instructions:

```markdown
- Mikasa method-development training Python:
  `/mnt/3fs1/data/tingwen.du/icra_method_dev/envs/imitation-py310-h200-headless/bin/python`
- For Mikasa memory-method experiment outputs, logs, manifests, checkpoints, and generated reports, write only under:
  `/mnt/3fs1/data/tingwen.du/icra_method_dev`
```

- [ ] **Step 4: Commit branch setup only if AGENTS.md changed**

Run:

```bash
git add AGENTS.md
git commit -m "docs: record mikasa method environment"
```

Expected: either one docs commit exists, or no commit is made because the instructions were already present.

### Task 2: Archive Provenance Schema And Static Validator

**Files:**
- Create: `experiments/mikasa_method_archive/provenance/method_code_sources.tsv`
- Create: `experiments/mikasa_method_archive/provenance/result_sources.tsv`
- Create: `experiments/mikasa_method_archive/provenance/checkpoint_sources.tsv`
- Create: `imitation-learning-policies/tests/test_mikasa_memory_archive_provenance.py`

**Interfaces:**
- Consumes: method names `no_memory`, `top1_retrieval`, `selector`, `gru`, `gru_burnin`, `late_cue_direct_anchor`.
- Produces: TSV schema that later launchers/docs rely on.

- [ ] **Step 1: Write failing provenance tests**

Create `imitation-learning-policies/tests/test_mikasa_memory_archive_provenance.py`:

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "experiments" / "mikasa_method_archive" / "provenance"
REQUIRED_METHODS = {
    "no_memory",
    "top1_retrieval",
    "selector",
    "gru",
    "gru_burnin",
    "late_cue_direct_anchor",
}


def _read_tsv(path: Path):
    rows = path.read_text().strip().splitlines()
    header = rows[0].split("\t")
    return [dict(zip(header, row.split("\t"))) for row in rows[1:]]


def test_each_method_has_code_source():
    rows = _read_tsv(ARCHIVE / "method_code_sources.tsv")
    assert REQUIRED_METHODS <= {row["method"] for row in rows}
    for row in rows:
        assert row["source_code_path"].startswith("/mnt/3fs1/data/tingwen.du/icra_method_dev/")
        assert row["repo_target"].startswith("imitation-learning-policies/")


def test_each_core_method_has_result_source():
    rows = _read_tsv(ARCHIVE / "result_sources.tsv")
    methods = {row["method"] for row in rows}
    assert {"no_memory", "top1_retrieval", "selector", "gru", "late_cue_direct_anchor"} <= methods
    for row in rows:
        assert row["task"]
        assert row["score"]
        assert row["source_path"].startswith("/mnt/3fs1/data/tingwen.du/icra_method_dev/")


def test_checkpoint_sources_are_paths_not_large_files():
    rows = _read_tsv(ARCHIVE / "checkpoint_sources.tsv")
    assert {"top1_retrieval", "selector", "gru", "late_cue_direct_anchor"} <= {
        row["method"] for row in rows
    }
    for row in rows:
        assert row["checkpoint_path"].startswith("/mnt/3fs1/data/tingwen.du/icra_method_dev/")
        assert row["checkpoint_path"].endswith(".ckpt")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd imitation-learning-policies
pytest tests/test_mikasa_memory_archive_provenance.py -v
```

Expected: FAIL because archive TSV files do not exist.

- [ ] **Step 3: Create provenance TSVs**

Create `experiments/mikasa_method_archive/provenance/method_code_sources.tsv` with tab-separated rows:

```text
method	source_code_path	repo_target	notes
no_memory	/mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/code/imitation-learning-policies_anchor_carrier_clear_20260619	imitation-learning-policies/	Base diffusion transformer no-history line used for mean 0.408.
top1_retrieval	/mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/code/imitation-learning-policies_remdp	imitation-learning-policies/	Memory transformer retrieval line with history_retrieval_topk=1.
selector	/mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/code/imitation-learning-policies_visual_selector_carrier_20260619	imitation-learning-policies/	Visual selector carrier plus late cue anchor.
gru	/mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/code/imitation-learning-policies_visual_gru_carrier_clean_20260619	imitation-learning-policies/	Clean VisualGRUMemoryCarrier line used for mean 0.668.
gru_burnin	/mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/code/imitation-learning-policies_visual_gru_carrier_clean_burnin_20260620	imitation-learning-policies/	Latest clean GRU code copy with conditional burn-in helper and tests.
late_cue_direct_anchor	/mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/code/imitation-learning-policies_remdp_direct_anchor_20260611	imitation-learning-policies/	Historical high-score direct-anchor exploration line; preserve separately from repaired 6/19 line.
late_cue_direct_anchor_repaired	/mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/code/imitation-learning-policies_anchor_carrier_clear_20260619	imitation-learning-policies/	Repaired 6/19 late-direct/no-memory code copy.
```

Create `experiments/mikasa_method_archive/provenance/result_sources.tsv` from the verified compact and seed42 summary tables. Include at minimum columns:

```text
method	task	score	source_path	protocol_note
```

Create `experiments/mikasa_method_archive/provenance/checkpoint_sources.tsv` with at minimum columns:

```text
method	task	checkpoint_path	source_note
```

- [ ] **Step 4: Run provenance tests**

Run:

```bash
cd imitation-learning-policies
pytest tests/test_mikasa_memory_archive_provenance.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add experiments/mikasa_method_archive/provenance imitation-learning-policies/tests/test_mikasa_memory_archive_provenance.py
git commit -m "docs: add mikasa method provenance schema"
```

### Task 3: Copy Small Result Tables And Legacy Launchers

**Files:**
- Create: `experiments/mikasa_method_archive/results/20260620_nomem_top1_selector_gru_compact.tsv`
- Create: `experiments/mikasa_method_archive/results/20260620_selector_gru_seed42_summary.tsv`
- Create: `experiments/mikasa_method_archive/results/late_cue_direct_anchor_notes.tsv`
- Create: `experiments/mikasa_method_archive/legacy_launchers/launch_no_memory_diffusion_5task_20260619.sh`
- Create: `experiments/mikasa_method_archive/legacy_launchers/train_mikasa_task_memory.sh`
- Create: `experiments/mikasa_method_archive/legacy_launchers/launch_visual_carrier_selector_gru_5task_20260619.sh`
- Create: `experiments/mikasa_method_archive/legacy_launchers/launch_visual_gru_conditional_burnin_8lane_20260620.sh`
- Create: `experiments/mikasa_method_archive/legacy_launchers/launch_late_direct_anchor_5task_20260619.sh`
- Create: `experiments/mikasa_method_archive/legacy_launchers/launch_rc3_direct_anchor_protocol_rerun_20260619.sh`

**Interfaces:**
- Consumes: ICRA-base historical files.
- Produces: small, auditable reproduction artifacts inside the repo. Large outputs remain referenced by path.

- [ ] **Step 1: Copy result TSVs**

Run:

```bash
mkdir -p experiments/mikasa_method_archive/results
cp /mnt/3fs1/data/tingwen.du/icra_method_dev/logs/mikasa_method_dev/20260620_nomem_top1_selector_gru_compact.tsv experiments/mikasa_method_archive/results/
cp /mnt/3fs1/data/tingwen.du/icra_method_dev/logs/mikasa_method_dev/20260620_selector_gru_seed42_summary.tsv experiments/mikasa_method_archive/results/
```

Expected: both copied files are small TSVs.

- [ ] **Step 2: Add late-cue note table**

Create `experiments/mikasa_method_archive/results/late_cue_direct_anchor_notes.tsv`:

```text
line	mean	source_path	interpretation
historical_direct_anchor_strict_best	0.736-0.738	/mnt/3fs1/data/tingwen.du/icra_method_dev/context_handoff_20260611_direct_anchor.md	Historical high-score exploration line; strong seed42 result but not treated as robust final baseline without protocol caveats.
late_direct_anchor_repaired_seed42	0.666	/mnt/3fs1/data/tingwen.du/icra_method_dev/logs/mikasa_method_dev/20260620_selector_gru_seed42_summary.tsv	Repaired 6/19 seed42 line included for method comparison with no-memory/selector/GRU.
```

- [ ] **Step 3: Copy legacy launchers unmodified**

Run:

```bash
mkdir -p experiments/mikasa_method_archive/legacy_launchers
cp /mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/launch_no_memory_diffusion_5task_20260619.sh experiments/mikasa_method_archive/legacy_launchers/
cp /mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/train_mikasa_task_memory.sh experiments/mikasa_method_archive/legacy_launchers/
cp /mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/launch_visual_carrier_selector_gru_5task_20260619.sh experiments/mikasa_method_archive/legacy_launchers/
cp /mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/launch_visual_gru_conditional_burnin_8lane_20260620.sh experiments/mikasa_method_archive/legacy_launchers/
cp /mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/launch_late_direct_anchor_5task_20260619.sh experiments/mikasa_method_archive/legacy_launchers/
cp /mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/launch_rc3_direct_anchor_protocol_rerun_20260619.sh experiments/mikasa_method_archive/legacy_launchers/
```

Expected: copied launchers retain original absolute historical code-copy paths for audit.

- [ ] **Step 4: Verify no large artifacts entered git**

Run:

```bash
find experiments/mikasa_method_archive -type f -size +5M -print
```

Expected: no output.

- [ ] **Step 5: Commit**

Run:

```bash
git add experiments/mikasa_method_archive/results experiments/mikasa_method_archive/legacy_launchers
git commit -m "docs: archive mikasa result tables and legacy launchers"
```

### Task 4: Add Mikasa Retrieval And Late-Cue Bypass Transformer

**Files:**
- Do not modify: `imitation-learning-policies/imitation_learning/models/denoising_networks/memory_transformer.py`
- Do not modify: `imitation-learning-policies/imitation_learning/configs/workspace/policy/denoising_network/memory_transformer.yaml`
- Do not modify: `imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_memory_transformer.yaml`
- Create: `imitation-learning-policies/imitation_learning/models/denoising_networks/mikasa_memory_transformer.py`
- Create: `imitation-learning-policies/imitation_learning/configs/workspace/policy/denoising_network/mikasa_memory_transformer.yaml`
- Create: `imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_mikasa_memory_transformer.yaml`
- Create: `imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_mikasa_retrieval_memory_transformer.yaml`
- Create: `imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_mikasa_start_anchored_direct_anchor_memory.yaml`
- Create: `imitation-learning-policies/tests/test_mikasa_memory_archive_configs.py`
- Create: `imitation-learning-policies/tests/test_gmp_baseline_compatibility.py`

**Interfaces:**
- Consumes: `history_retrieval_topk`, `late_cue_anchor_enabled`, `late_cue_anchor_len`, `late_cue_anchor_causal_mask`, action prehead adapter config fields from historical code copies.
- Produces: repo-local `MikasaMemoryTransformer` and Mikasa configs that can represent no-memory, top1 retrieval, and direct-anchor methods without changing original GMP targets.

- [ ] **Step 1: Write GMP compatibility guard tests**

Create `imitation-learning-policies/tests/test_gmp_baseline_compatibility.py`:

```python
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "imitation_learning" / "configs" / "workspace" / "policy"


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
    from imitation_learning.models.denoising_networks.memory_transformer import (
        MemoryTransformer,
    )
    from imitation_learning.policies.history_denoising_policy import (
        HistoryDenoisingPolicy,
    )

    assert MemoryTransformer.__name__ == "MemoryTransformer"
    assert HistoryDenoisingPolicy.__name__ == "HistoryDenoisingPolicy"
```

- [ ] **Step 2: Run compatibility guard tests**

Run:

```bash
cd imitation-learning-policies
pytest tests/test_gmp_baseline_compatibility.py -v
```

Expected: PASS before any Mikasa code is added. If this fails, stop and investigate because the baseline is already not in the expected state.

- [ ] **Step 3: Write failing Mikasa config tests**

Create `imitation-learning-policies/tests/test_mikasa_memory_archive_configs.py`:

```python
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
    assert {"override denoising_network@denoising_network_partial": "mikasa_memory_transformer"} in cfg["defaults"]


def test_retrieval_config_uses_memory_policy_and_topk_override_path():
    cfg = _load_yaml(POLICY_DIR / "diffusion_mikasa_retrieval_memory_transformer.yaml")
    assert "diffusion_mikasa_memory_transformer" in cfg["defaults"]
    assert "denoising_network_partial" in cfg
    network = cfg["denoising_network_partial"]
    assert network.get("history_retrieval_topk") in (1, 4)


def test_direct_anchor_config_enables_late_cue_anchor():
    cfg = _load_yaml(POLICY_DIR / "diffusion_mikasa_start_anchored_direct_anchor_memory.yaml")
    assert "diffusion_mikasa_memory_transformer" in cfg["defaults"]
    network = cfg["denoising_network_partial"]
    assert network["late_cue_anchor_enabled"] is True
    assert network["late_cue_anchor_len"] >= 1
```

- [ ] **Step 4: Run Mikasa config tests to verify they fail**

Run:

```bash
cd imitation-learning-policies
pytest tests/test_mikasa_memory_archive_configs.py -v
```

Expected: FAIL because Mikasa-specific configs and `MikasaMemoryTransformer` are missing from the current repo tree.

- [ ] **Step 5: Create Mikasa transformer without touching GMP transformer**

Use the latest burn-in code copy as the primary donor for consolidated behavior:

```bash
diff -u imitation_learning/models/denoising_networks/memory_transformer.py /mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/code/imitation-learning-policies_visual_gru_carrier_clean_burnin_20260620/imitation_learning/models/denoising_networks/memory_transformer.py
```

Create `imitation_learning/models/denoising_networks/mikasa_memory_transformer.py` by copying the latest burn-in donor behavior into a new module, then renaming only the public class:

- Donor class `MemoryTransformer` becomes `MikasaMemoryTransformer`.
- Internal helper/block names may remain donor-compatible unless renaming is needed for clarity.
- Existing imports from `imitation_learning.models.denoising_networks.memory_transformer` are allowed only for stable base classes/helpers; the old module itself must not be edited.
- Do not add compatibility aliases from the old module to the Mikasa class.

Port only the retrieval and late-cue pieces in this task:

- `history_retrieval_topk` constructor/config field.
- Retrieval mask selection helper.
- `late_cue_anchor_latents`, `late_cue_anchor_mask`, `late_cue_anchor_action_latents`, `late_cue_anchor_action_mask` forward plumbing.
- Adapter fields required by the direct-anchor config.

- [ ] **Step 6: Add Mikasa base and method configs**

Create `imitation_learning/configs/workspace/policy/denoising_network/mikasa_memory_transformer.yaml`:

```yaml
defaults:
  - conditional_transformer
  - _self_

_target_: imitation_learning.models.denoising_networks.mikasa_memory_transformer.MikasaMemoryTransformer

max_history_len: 16
freeze_non_history_modules: False
history_attention_type: token_wise
record_data_entries: []
include_action_history: True

memory_gate_include_first_token: True
memory_gate_func: abs
ssmax_scaling_param: null
add_additional_self_attn: True
skip_history_attn: False
add_memory_gate_token: False
binary_gating: True
straight_through: ""

history_retrieval_topk: null
late_cue_adapter_enabled: false
late_cue_anchor_enabled: false
late_cue_anchor_len: 0
late_cue_anchor_causal_mask: false
late_cue_action_prehead_adapter_enabled: false
late_cue_action_prehead_adapter_gate_init_bias: 0.0
late_cue_action_prehead_adapter_out_init_gain: 0.001
late_cue_action_prehead_adapter_residual_scale: 0.1
visual_memory_carrier_type: null
visual_memory_carrier_token_num: 1
visual_memory_carrier_max_len: 64
visual_memory_carrier_hidden_dim: null
visual_memory_carrier_num_layers: 1
visual_memory_carrier_num_heads: 8
visual_memory_carrier_dropout: 0.0
visual_memory_carrier_force_zero: false
```

Create `imitation_learning/configs/workspace/policy/diffusion_mikasa_memory_transformer.yaml`:

```yaml
defaults:
  - diffusion_transformer
  - cond_encoder@history_img_feature_encoder: multi_token_encoder
  - override denoising_network@denoising_network_partial: mikasa_memory_transformer
  - _self_

_target_: imitation_learning.policies.mikasa_history_denoising_policy.MikasaHistoryDenoisingPolicy

skip_memory: False
history_mask_max_prob: 0.0
memory_gate: null
history_img_feature_encoder:
  image_encoder_partial:
    feature_aggregation: 'map'
  name: history_img_feature_encoder
  data_entry_names: ${workspace.train_dataset.image_keys}

train_history_action_noise_level: last_step
eval_history_action_noise_level: last_step
history_action_num_per_chunk: 8
action_no_error_range: [4, 8]
burn_in_loss_traj_num: 0
burn_in_start_id: 0
```

Copy donor method configs into Mikasa-prefixed filenames and normalize their first default to `diffusion_mikasa_memory_transformer`:

```bash
cp /mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/code/imitation-learning-policies_visual_gru_carrier_clean_burnin_20260620/imitation_learning/configs/workspace/policy/diffusion_retrieval_memory_transformer.yaml imitation_learning/configs/workspace/policy/diffusion_mikasa_retrieval_memory_transformer.yaml
cp /mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/code/imitation-learning-policies_visual_gru_carrier_clean_burnin_20260620/imitation_learning/configs/workspace/policy/diffusion_start_anchored_direct_anchor_memory.yaml imitation_learning/configs/workspace/policy/diffusion_mikasa_start_anchored_direct_anchor_memory.yaml
```

After copying, edit each copied file so the `defaults` block starts with:

```yaml
defaults:
  - diffusion_mikasa_memory_transformer
  - _self_
```

Ensure `history_retrieval_topk` remains overrideable by launcher, because archived top1 uses `history_retrieval_topk=1` even if the config default is 4.

- [ ] **Step 7: Run config and compatibility tests**

Run:

```bash
cd imitation-learning-policies
pytest tests/test_gmp_baseline_compatibility.py tests/test_mikasa_memory_archive_configs.py -v
```

Expected: PASS.

- [ ] **Step 8: Verify protected GMP files have no diff**

Run from the repository root:

```bash
git diff --exit-code -- \
  imitation-learning-policies/imitation_learning/models/denoising_networks/memory_transformer.py \
  imitation-learning-policies/imitation_learning/configs/workspace/policy/denoising_network/memory_transformer.yaml \
  imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_memory_transformer.yaml
```

Expected: no output and exit code 0.

- [ ] **Step 9: Commit**

Run:

```bash
git add imitation-learning-policies/imitation_learning/models/denoising_networks/mikasa_memory_transformer.py imitation-learning-policies/imitation_learning/configs/workspace/policy/denoising_network/mikasa_memory_transformer.yaml imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_mikasa_memory_transformer.yaml imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_mikasa_retrieval_memory_transformer.yaml imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_mikasa_start_anchored_direct_anchor_memory.yaml imitation-learning-policies/tests/test_mikasa_memory_archive_configs.py imitation-learning-policies/tests/test_gmp_baseline_compatibility.py
git commit -m "feat: add mikasa retrieval and late-cue bypass configs"
```

### Task 5: Add Visual Selector And Clean GRU Carriers To Mikasa Transformer

**Files:**
- Create: `imitation-learning-policies/imitation_learning/models/visual_memory_carriers.py`
- Modify: `imitation-learning-policies/imitation_learning/models/denoising_networks/mikasa_memory_transformer.py`
- Do not modify: `imitation-learning-policies/imitation_learning/models/denoising_networks/memory_transformer.py`
- Create: `imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_mikasa_visual_selector_late_anchor_memory.yaml`
- Create: `imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_mikasa_visual_gru_late_anchor_memory.yaml`
- Create: `imitation-learning-policies/tests/test_visual_memory_carriers.py`

**Interfaces:**
- Consumes: `visual_memory_carrier_type`, `visual_memory_carrier_token_num`, `visual_memory_carrier_max_len`, `visual_memory_carrier_hidden_dim`, `visual_memory_carrier_num_layers`, `visual_memory_carrier_dropout`.
- Produces: `LearnedLateCueSelector` and `VisualGRUMemoryCarrier` available to Mikasa-only transformer configs.

- [ ] **Step 1: Write failing visual carrier tests**

Create `imitation-learning-policies/tests/test_visual_memory_carriers.py`:

```python
import torch

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd imitation-learning-policies
pytest tests/test_visual_memory_carriers.py -v
```

Expected: FAIL because `visual_memory_carriers.py` is missing or incomplete.

- [ ] **Step 3: Copy visual carrier implementation from latest clean burn-in code**

Run:

```bash
cp /mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/code/imitation-learning-policies_visual_gru_carrier_clean_burnin_20260620/imitation_learning/models/visual_memory_carriers.py imitation_learning/models/visual_memory_carriers.py
```

Expected: file includes `LearnedLateCueSelector`, `VisualGRUMemoryCarrier`, and the clean GRU no-cuDNN-flatten wrapper.

- [ ] **Step 4: Wire visual carrier config fields into Mikasa transformer**

Port from the burn-in source:

- Import `VisualGRUMemoryCarrier` and `LearnedLateCueSelector`.
- Add constructor fields for visual carrier type and dimensions.
- Instantiate selector when `visual_memory_carrier_type=selector`.
- Instantiate GRU when `visual_memory_carrier_type=gru`.
- Preserve default-off behavior when `visual_memory_carrier_type` is empty or `null`.
- Keep all imports and wiring inside `mikasa_memory_transformer.py`; do not edit the original GMP `memory_transformer.py`.

- [ ] **Step 5: Add selector and GRU configs**

Run:

```bash
cp /mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/code/imitation-learning-policies_visual_gru_carrier_clean_burnin_20260620/imitation_learning/configs/workspace/policy/diffusion_visual_gru_late_anchor_memory.yaml imitation_learning/configs/workspace/policy/diffusion_mikasa_visual_gru_late_anchor_memory.yaml
cp /mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/code/imitation-learning-policies_visual_selector_carrier_20260619/imitation_learning/configs/workspace/policy/diffusion_visual_selector_late_anchor_memory.yaml imitation_learning/configs/workspace/policy/diffusion_mikasa_visual_selector_late_anchor_memory.yaml
```

Expected:

- Both copied files have `defaults: [diffusion_mikasa_memory_transformer, _self_]` in YAML block form.
- Selector config uses `visual_memory_carrier_type: selector`.
- GRU config uses `visual_memory_carrier_type: gru`.
- Both keep late-cue anchor enabled exactly as historical recipes require.

- [ ] **Step 6: Extend Mikasa config tests for visual carriers**

Append these tests to `imitation-learning-policies/tests/test_mikasa_memory_archive_configs.py`:

```python
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
```

- [ ] **Step 7: Run carrier and config tests**

Run:

```bash
cd imitation-learning-policies
pytest tests/test_gmp_baseline_compatibility.py tests/test_visual_memory_carriers.py tests/test_mikasa_memory_archive_configs.py -v
```

Expected: PASS.

- [ ] **Step 8: Verify protected GMP files have no diff**

Run from the repository root:

```bash
git diff --exit-code -- \
  imitation-learning-policies/imitation_learning/models/denoising_networks/memory_transformer.py \
  imitation-learning-policies/imitation_learning/policies/history_denoising_policy.py \
  imitation-learning-policies/imitation_learning/configs/workspace/policy/denoising_network/memory_transformer.yaml \
  imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_memory_transformer.yaml
```

Expected: no output and exit code 0.

- [ ] **Step 9: Commit**

Run:

```bash
git add imitation-learning-policies/imitation_learning/models/visual_memory_carriers.py imitation-learning-policies/imitation_learning/models/denoising_networks/mikasa_memory_transformer.py imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_mikasa_visual_selector_late_anchor_memory.yaml imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_mikasa_visual_gru_late_anchor_memory.yaml imitation-learning-policies/tests/test_visual_memory_carriers.py imitation-learning-policies/tests/test_mikasa_memory_archive_configs.py
git commit -m "feat: add visual selector and clean gru carriers"
```

### Task 6: Add Mikasa History Policy Plumbing And Conditional Burn-In

**Files:**
- Create: `imitation-learning-policies/imitation_learning/policies/mikasa_history_denoising_policy.py`
- Do not modify: `imitation-learning-policies/imitation_learning/policies/history_denoising_policy.py`
- Modify only if required: `imitation-learning-policies/imitation_learning/datasets/multi_traj_dataset.py`
- Create: `imitation-learning-policies/tests/test_clean_gru_burn_in_loss_mask.py`

**Interfaces:**
- Consumes: `burn_in_loss_traj_num`, `burn_in_start_id`, batch `traj_idx` or explicit `start_idx`.
- Produces: loss-valid mask helper that implements conditional burn-in.

- [ ] **Step 1: Write failing burn-in mask tests**

Create `imitation-learning-policies/tests/test_clean_gru_burn_in_loss_mask.py`:

```python
import torch

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd imitation-learning-policies
pytest tests/test_clean_gru_burn_in_loss_mask.py -v
```

Expected: FAIL because `_apply_burn_in_loss_mask` is missing or has old unconditional behavior.

- [ ] **Step 3: Create Mikasa history policy and port history policy changes**

Port from:

```text
/mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/code/imitation-learning-policies_visual_gru_carrier_clean_burnin_20260620/imitation_learning/policies/history_denoising_policy.py
```

Create `imitation_learning/policies/mikasa_history_denoising_policy.py` as a Mikasa-only policy module:

- The public class must be `MikasaHistoryDenoisingPolicy`.
- It may start as a copy of the donor `HistoryDenoisingPolicy`, but its imports and `isinstance` checks must reference `MikasaMemoryTransformer`.
- It must not replace or alias `imitation_learning.policies.history_denoising_policy.HistoryDenoisingPolicy`.
- It must expose `_apply_burn_in_loss_mask` for the focused unit tests.

Required behavior:

- `burn_in_loss_traj_num=0` returns the original `loss_valid_mask`.
- `start_idx <= burn_in_start_id` returns the original valid mask for that sample.
- `start_idx > burn_in_start_id` masks only the first `burn_in_loss_traj_num` original traj slots participating in action loss.
- Final mask equals `original_valid_mask AND conditional_burn_in_mask`.
- When `training_traj_indices` is active, prefix masking is based on original traj slot ids, not post-sampling tensor column positions.

- [ ] **Step 4: Verify whether dataset changes are needed**

Run:

```bash
cd imitation-learning-policies
rg -n "traj_idx|start_idx" imitation_learning/datasets imitation_learning/policies/mikasa_history_denoising_policy.py
```

Expected:

- If batch exposes `traj_idx`, do not change dataset code.
- If batch does not expose stable start index, add explicit `start_idx` to the dataset sample and update only the Mikasa helper call accordingly.
- Any dataset change must be additive: no existing GMP key, shape, or meaning may change.

- [ ] **Step 5: Run burn-in and compatibility tests**

Run:

```bash
cd imitation-learning-policies
pytest tests/test_gmp_baseline_compatibility.py tests/test_clean_gru_burn_in_loss_mask.py -v
```

Expected: PASS.

- [ ] **Step 6: Verify protected GMP files have no diff**

Run from the repository root:

```bash
git diff --exit-code -- \
  imitation-learning-policies/imitation_learning/models/denoising_networks/memory_transformer.py \
  imitation-learning-policies/imitation_learning/policies/history_denoising_policy.py \
  imitation-learning-policies/imitation_learning/configs/workspace/policy/denoising_network/memory_transformer.yaml \
  imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_memory_transformer.yaml
```

Expected: no output and exit code 0.

- [ ] **Step 7: Commit**

Run:

```bash
git add imitation-learning-policies/imitation_learning/policies/mikasa_history_denoising_policy.py imitation-learning-policies/imitation_learning/datasets/multi_traj_dataset.py imitation-learning-policies/tests/test_clean_gru_burn_in_loss_mask.py
git commit -m "feat: add mikasa conditional gru burn-in loss mask"
```

### Task 7: Add Repo-Local Reproduction Launchers And Method Matrix

**Files:**
- Create: `experiments/mikasa_method_archive/manifests/mikasa_5task_method_matrix.tsv`
- Create: `experiments/mikasa_method_archive/launchers/run_no_memory_5task.sh`
- Create: `experiments/mikasa_method_archive/launchers/run_top1_retrieval_5task.sh`
- Create: `experiments/mikasa_method_archive/launchers/run_selector_5task.sh`
- Create: `experiments/mikasa_method_archive/launchers/run_gru_5task.sh`
- Create: `experiments/mikasa_method_archive/launchers/run_gru_conditional_burnin_5task.sh`
- Create: `experiments/mikasa_method_archive/launchers/run_late_cue_direct_anchor_5task.sh`

**Interfaces:**
- Consumes: unified repo-local policy configs and ICRA-base large artifact paths.
- Produces: reproduction entrypoints that users can run without knowing historical code-copy paths.

- [ ] **Step 1: Create method matrix**

Create `experiments/mikasa_method_archive/manifests/mikasa_5task_method_matrix.tsv` with columns:

```text
method	task	env	traj_num	max_history_len	action_len	traj_interval	training_traj_sampling_strategy	base_ckpt_path	policy_config	result_note
```

Required rows:

- `no_memory` for RememberColor3-v0, RememberColor5-v0, RememberColor9-v0, InterceptMedium-v0, ShellGameTouch-v0 using `policy_config=diffusion_transformer`.
- `top1_retrieval` for the same five tasks with `history_retrieval_topk=1` using `policy_config=diffusion_mikasa_retrieval_memory_transformer`.
- `selector` for the same five tasks using `policy_config=diffusion_mikasa_visual_selector_late_anchor_memory`.
- `gru` for the same five tasks using `policy_config=diffusion_mikasa_visual_gru_late_anchor_memory`.
- `gru_burnin` for RC3, RC5, RC9, InterceptMedium, ShellGameTouch using `policy_config=diffusion_mikasa_visual_gru_late_anchor_memory` plus conditional burn-in settings from the 2026-06-20 burn-in experiment.
- `late_cue_direct_anchor` with separate notes for historical high-score direct-anchor and repaired 6/19 late-direct lines using `policy_config=diffusion_mikasa_start_anchored_direct_anchor_memory`.

- [ ] **Step 2: Create launcher header shared by all scripts**

Every repo-local launcher starts with:

```bash
#!/usr/bin/env bash
set -euo pipefail

ICRA_BASE="${ICRA_BASE:-/mnt/3fs1/data/tingwen.du/icra_method_dev}"
PY="${PY:-$ICRA_BASE/envs/imitation-py310-h200-headless/bin/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CODE_DIR="$REPO_ROOT/imitation-learning-policies"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"

RUN_ROOT="$ICRA_BASE/runs/mikasa_method_archive"
LOG_ROOT="$ICRA_BASE/logs/mikasa_method_archive"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
```

- [ ] **Step 3: Create no-memory launcher**

Create `experiments/mikasa_method_archive/launchers/run_no_memory_5task.sh` using repo-local `diffusion_transformer` and setting:

```bash
POLICY_NAME=diffusion_transformer
INCLUDE_ACTION_HISTORY=false
EPISODE_STARTING_IDX_MAX=1
```

Expected: all outputs go to `$ICRA_BASE/runs/mikasa_method_archive` and `$ICRA_BASE/logs/mikasa_method_archive`.

- [ ] **Step 4: Create top1 retrieval launcher**

Create `experiments/mikasa_method_archive/launchers/run_top1_retrieval_5task.sh` using:

```bash
POLICY_NAME=diffusion_mikasa_retrieval_memory_transformer
HISTORY_RETRIEVAL_TOPK=1
INCLUDE_ACTION_HISTORY=false
INDEX_POOL_SIZE_PER_EPISODE=4
```

- [ ] **Step 5: Create selector launcher**

Create `experiments/mikasa_method_archive/launchers/run_selector_5task.sh` using:

```bash
POLICY_NAME=diffusion_mikasa_visual_selector_late_anchor_memory
VISUAL_MEMORY_CARRIER_TYPE=selector
```

- [ ] **Step 6: Create GRU launcher**

Create `experiments/mikasa_method_archive/launchers/run_gru_5task.sh` using:

```bash
POLICY_NAME=diffusion_mikasa_visual_gru_late_anchor_memory
VISUAL_MEMORY_CARRIER_TYPE=gru
BURN_IN_LOSS_TRAJ_NUM=0
```

- [ ] **Step 7: Create conditional burn-in launcher**

Create `experiments/mikasa_method_archive/launchers/run_gru_conditional_burnin_5task.sh` with task settings:

```text
rc3_burn	RememberColor3-v0	traj_num=7	max_history_len=6	action_len=10	traj_interval=10	burn_in_start_id=8	burn_in_loss_traj_num=2	training_traj_sampling_strategy=tail
rc5_burn	RememberColor5-v0	traj_num=7	max_history_len=6	action_len=10	traj_interval=10	burn_in_start_id=8	burn_in_loss_traj_num=2	training_traj_sampling_strategy=tail
rc9_burn	RememberColor9-v0	traj_num=7	max_history_len=6	action_len=10	traj_interval=10	burn_in_start_id=8	burn_in_loss_traj_num=2	training_traj_sampling_strategy=tail
intercept_burn	InterceptMedium-v0	traj_num=13	max_history_len=12	action_len=8	traj_interval=8	burn_in_start_id=16	burn_in_loss_traj_num=4	training_traj_sampling_strategy=tail
shell_burn	ShellGameTouch-v0	traj_num=46	max_history_len=45	action_len=2	traj_interval=10	burn_in_start_id=8	burn_in_loss_traj_num=8	training_traj_sampling_strategy=tail
```

- [ ] **Step 8: Create late-cue/direct-anchor launcher**

Create `experiments/mikasa_method_archive/launchers/run_late_cue_direct_anchor_5task.sh` using:

```bash
POLICY_NAME=diffusion_mikasa_start_anchored_direct_anchor_memory
LATE_CUE_ANCHOR_ENABLED=true
```

The launcher must include a comment that historical mean > 0.70 came from the direct-anchor exploration/protocol line and should be interpreted with its protocol notes.

- [ ] **Step 9: Shell syntax check**

Run:

```bash
bash -n experiments/mikasa_method_archive/launchers/*.sh
bash -n experiments/mikasa_method_archive/legacy_launchers/*.sh
```

Expected: PASS for all shell files.

- [ ] **Step 10: Commit**

Run:

```bash
git add experiments/mikasa_method_archive/manifests experiments/mikasa_method_archive/launchers
git commit -m "feat: add mikasa method archive launchers"
```

### Task 8: Smoke Test Imports And Config Entry Points

**Files:**
- No new files unless a smoke failure requires a focused fix in the files changed above.

**Interfaces:**
- Consumes: unified code tree and repo-local launchers.
- Produces: evidence that archived configs import and basic tests pass under the known training Python.

- [ ] **Step 1: Run import smoke**

Run:

```bash
cd imitation-learning-policies
PY=/mnt/3fs1/data/tingwen.du/icra_method_dev/envs/imitation-py310-h200-headless/bin/python
PYTHONPATH="$PWD" "$PY" - <<'PY'
from imitation_learning.policies.history_denoising_policy import HistoryDenoisingPolicy
from imitation_learning.policies.mikasa_history_denoising_policy import MikasaHistoryDenoisingPolicy
from imitation_learning.models.denoising_networks.memory_transformer import MemoryTransformer
from imitation_learning.models.denoising_networks.mikasa_memory_transformer import MikasaMemoryTransformer
from imitation_learning.models.visual_memory_carriers import LearnedLateCueSelector, VisualGRUMemoryCarrier
print("HistoryDenoisingPolicy", HistoryDenoisingPolicy is not None)
print("MikasaHistoryDenoisingPolicy", MikasaHistoryDenoisingPolicy is not None)
print("MemoryTransformer", MemoryTransformer is not None)
print("MikasaMemoryTransformer", MikasaMemoryTransformer is not None)
print("LearnedLateCueSelector", LearnedLateCueSelector is not None)
print("VisualGRUMemoryCarrier", VisualGRUMemoryCarrier is not None)
PY
```

Expected:

```text
HistoryDenoisingPolicy True
MikasaHistoryDenoisingPolicy True
MemoryTransformer True
MikasaMemoryTransformer True
LearnedLateCueSelector True
VisualGRUMemoryCarrier True
```

- [ ] **Step 2: Run focused tests**

Run:

```bash
cd imitation-learning-policies
PY=/mnt/3fs1/data/tingwen.du/icra_method_dev/envs/imitation-py310-h200-headless/bin/python
PYTHONPATH="$PWD" "$PY" -m pytest \
  tests/test_gmp_baseline_compatibility.py \
  tests/test_mikasa_memory_archive_provenance.py \
  tests/test_mikasa_memory_archive_configs.py \
  tests/test_visual_memory_carriers.py \
  tests/test_clean_gru_burn_in_loss_mask.py \
  -v
```

Expected: PASS.

- [ ] **Step 3: Run launcher dry syntax check**

Run:

```bash
bash -n experiments/mikasa_method_archive/launchers/*.sh
```

Expected: PASS.

- [ ] **Step 4: Commit focused fixes if needed**

If Step 1-3 required fixes, commit them:

```bash
git add imitation-learning-policies experiments/mikasa_method_archive
git commit -m "fix: pass mikasa archive smoke tests"
```

Expected: no commit if no fixes were needed.

### Task 9: Write Consolidation Report

**Files:**
- Create: `experiments/mikasa_method_archive/README.md`
- Create: `docs/experiments/2026-06-29-mikasa-memory-method-archive.md`

**Interfaces:**
- Consumes: provenance TSVs, copied result TSVs, launchers, and unified configs.
- Produces: reviewable human summary for future work.

- [ ] **Step 1: Write archive README**

Create `experiments/mikasa_method_archive/README.md` with sections:

```markdown
# Mikasa Memory Method Archive

## Scope

This archive preserves the code paths and reproduction metadata for:

- no-memory, compact mean 0.408
- top1 retrieval, compact mean 0.564/0.565
- selector, compact mean 0.654
- clean GRU, compact mean 0.668
- late cue / direct anchor, historical high-score line above 0.70 and repaired 6/19 seed42 line 0.666
- conditional GRU burn-in follow-up experiment line

## What Is In Git

- Mikasa-only repo-local implementation under `imitation-learning-policies/`
- GMP compatibility guard tests proving original GMP `_target_` strings and import paths remain unchanged
- Small result TSVs under `results/`
- Provenance TSVs under `provenance/`
- Historical launchers under `legacy_launchers/`
- Repo-local launchers under `launchers/`

## What Is Not In Git

Large checkpoints, train logs, rollout videos, eval outputs, datasets, and generated manifests remain under:

`/mnt/3fs1/data/tingwen.du/icra_method_dev`

## Recommended Future Baseline

Use the clean GRU line as the primary code baseline because it has the strongest compact mean among no-memory/top1/selector/GRU, while preserving selector and late-cue/direct-anchor as comparison and idea sources.

## Late-Cue Caveat

The direct-anchor high-score line and repaired 6/19 late-direct line are recorded separately. Treat historical high-score direct-anchor as exploratory/protocol-sensitive until reproduced under the same strict evaluation protocol as the compact table.
```

- [ ] **Step 2: Write experiment report**

Create `docs/experiments/2026-06-29-mikasa-memory-method-archive.md` with:

- Method table.
- Exact source code paths.
- Exact result table paths.
- Which code landed in the unified implementation.
- Which original GMP files and configs were intentionally left untouched for checkpoint/training compatibility.
- Which artifacts are stored only as path references.
- Known caveats: top1 Intercept score mismatch across historical reruns, late-cue direct-anchor robustness/protocol caveat, burn-in not improving enough to replace clean GRU.

- [ ] **Step 3: Commit docs**

Run:

```bash
git add experiments/mikasa_method_archive/README.md docs/experiments/2026-06-29-mikasa-memory-method-archive.md
git commit -m "docs: explain mikasa memory method archive"
```

### Task 10: Final Verification, Push, And PR Prep

**Files:**
- No new files unless verification uncovers missing archive metadata.

**Interfaces:**
- Consumes: all previous commits.
- Produces: pushed branch and PR-ready summary.

- [ ] **Step 1: Verify git diff against base**

Run:

```bash
git status --short --branch
git log --oneline --max-count=8
```

Expected:

- Current branch is `intern_method_dev_gmp/mikasa-memory-method-archive`.
- No unintended edits to `workspace/interns/intern_method_dev_gmp/knowledge.md`.
- Recent commits match Tasks 1-9.

- [ ] **Step 2: Run final focused test suite**

Run:

```bash
cd imitation-learning-policies
PY=/mnt/3fs1/data/tingwen.du/icra_method_dev/envs/imitation-py310-h200-headless/bin/python
PYTHONPATH="$PWD" "$PY" -m pytest \
  tests/test_gmp_baseline_compatibility.py \
  tests/test_mikasa_memory_archive_provenance.py \
  tests/test_mikasa_memory_archive_configs.py \
  tests/test_visual_memory_carriers.py \
  tests/test_clean_gru_burn_in_loss_mask.py \
  -v
```

Expected: PASS.

- [ ] **Step 3: Verify protected GMP files have no diff**

Run from the repository root:

```bash
git diff --exit-code -- \
  imitation-learning-policies/imitation_learning/models/denoising_networks/memory_transformer.py \
  imitation-learning-policies/imitation_learning/policies/history_denoising_policy.py \
  imitation-learning-policies/imitation_learning/configs/workspace/policy/denoising_network/memory_transformer.yaml \
  imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_memory_transformer.yaml \
  imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_gated_transformer.yaml \
  imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_binary_gated_transformer.yaml \
  imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_continuous_gated_transformer.yaml \
  imitation-learning-policies/imitation_learning/configs/workspace/policy/flow_memory_transformer.yaml \
  imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_memory_transformer_large.yaml
```

Expected: no output and exit code 0.

- [ ] **Step 4: Verify shell launchers**

Run:

```bash
bash -n experiments/mikasa_method_archive/launchers/*.sh
bash -n experiments/mikasa_method_archive/legacy_launchers/*.sh
```

Expected: PASS.

- [ ] **Step 5: Verify no large files were added**

Run:

```bash
find experiments/mikasa_method_archive -type f -size +5M -print
```

Expected: no output.

- [ ] **Step 6: Push feature branch**

Run:

```bash
git push -u origin intern_method_dev_gmp/mikasa-memory-method-archive
```

Expected: branch pushed to GitHub. Do not push `main` or `master`.

- [ ] **Step 7: Prepare PR summary**

Use this PR body:

```markdown
## Summary

- Consolidates Mikasa no-memory, top1 retrieval, selector, clean GRU, conditional burn-in, and late-cue/direct-anchor method code paths into Mikasa-only repo-local entrypoints under `imitation-learning-policies/`.
- Preserves original GMP baseline files, `_target_` strings, import paths, and checkpoint/training config compatibility.
- Adds `experiments/mikasa_method_archive/` with reproducibility launchers, historical launcher copies, result TSVs, provenance TSVs, and a 5-task method matrix.
- Documents which large checkpoints/eval outputs remain under `/mnt/3fs1/data/tingwen.du/icra_method_dev` instead of being committed.

## Verification

- `PYTHONPATH="$PWD" $PY -m pytest tests/test_gmp_baseline_compatibility.py tests/test_mikasa_memory_archive_provenance.py tests/test_mikasa_memory_archive_configs.py tests/test_visual_memory_carriers.py tests/test_clean_gru_burn_in_loss_mask.py -v`
- `git diff --exit-code -- imitation-learning-policies/imitation_learning/models/denoising_networks/memory_transformer.py imitation-learning-policies/imitation_learning/policies/history_denoising_policy.py imitation-learning-policies/imitation_learning/configs/workspace/policy/denoising_network/memory_transformer.yaml imitation-learning-policies/imitation_learning/configs/workspace/policy/diffusion_memory_transformer.yaml`
- `bash -n experiments/mikasa_method_archive/launchers/*.sh`
- `bash -n experiments/mikasa_method_archive/legacy_launchers/*.sh`
- `find experiments/mikasa_method_archive -type f -size +5M -print`

## Notes

- BCRNN is intentionally excluded.
- Mikasa configs use `diffusion_mikasa_*` policy entrypoints; original GMP configs continue to use their original `_target_` paths.
- Historical direct-anchor high-score line is preserved as protocol-sensitive lineage, separate from the repaired 6/19 late-direct seed42 line.
- Checkpoints and generated experiment outputs are path-referenced under ICRA base, not stored in git.
```

## Review Gates

Before execution, user should approve these boundaries:

1. Keep a single unified Mikasa implementation under `imitation-learning-policies/`, but add it through Mikasa-only modules/configs rather than mutating original GMP targets.
2. Preserve original GMP checkpoint/training/eval compatibility: old configs, old `_target_` strings, old import paths, and protected files stay unchanged.
3. Preserve historical code copies by path/provenance and copied small launchers/results, not by vendoring full duplicate code trees.
4. Do not commit checkpoints, rollout outputs, datasets, generated eval directories, or train logs.
5. Treat direct-anchor high-score `mean > 0.70` as historical/protocol-sensitive lineage, not as the same strict line as the compact no-memory/top1/selector/GRU table.
6. Exclude BCRNN entirely.

## Self-Review

- Spec coverage: covers all requested method families, Mikasa-only code locations, original GMP compatibility boundaries, reproduction flow, branch/push constraints, and ICRA-base artifact boundary.
- Placeholder scan: no unresolved placeholder markers are present.
- Type consistency: method names, `diffusion_mikasa_*` config names, `MikasaMemoryTransformer`, `MikasaHistoryDenoisingPolicy`, provenance, launchers, tests, and docs are aligned.
