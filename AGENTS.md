# intern_method_dev_gmp Operating Rules

- Primary project code: `/work-agents/intern_method_dev_gmp/gated-memory-policy/`.
- ICRA method-development base: `/mnt/3fs1/data/tingwen.du/icra_method_dev`.
- Put datasets, experiment outputs, manifests, GPU-shared artifacts, and large generated files for this method-development effort under the ICRA base above.
- Do not write to any other `/mnt` path unless the user explicitly adds a new allowed path.
- Existing historical baseline paths under `/mnt/3fs1/data/tingwen.du/gated-memory-policy-*` are read-only references unless the user asks to modify them.

## Mikasa Method-Dev Environments

- Reuse existing environments; do not create new conda/venv environments for this effort unless the user explicitly asks.
- Primary training Python: `/mnt/3fs1/data/tingwen.du/icra_method_dev/envs/imitation-py310-h200-headless/bin/python`.
- For clean GRU burn-in code-copy checks, set `PYTHONPATH` to the active code copy, for example:
  `PYTHONPATH=/mnt/3fs1/data/tingwen.du/icra_method_dev/experiments/memory_method_dev/code/imitation-learning-policies_visual_gru_carrier_clean_burnin_20260620`.
- Historical eval Python commonly used by existing eval wrappers:
  `/mnt/3fs1/data/tingwen.du/gated-memory-policy-envs/task003_robomimic_memory_gate_repro/mikasa-py311/bin/python`.
- Prefer the ICRA-base `imitation-py310-h200-headless` Python for training/import smoke; use the historical `mikasa-py311` Python for eval paths that already depend on it.
