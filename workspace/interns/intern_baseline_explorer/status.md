# intern_baseline_explorer - 状态

<!-- METADATA:STATUS=Working,TASK=task003_robomimic_memory_gate_repro -->

| 字段 | 值 |
|------|-----|
| Name | intern_baseline_explorer |
| Status | Working |
| Current Task | task003_robomimic_memory_gate_repro |
| PR | https://github.com/StevenKKXS/gated-memory-policy/pull/3 |
| Session | 11 |
| 最近进展 | 已梳理 GMP training 的论文逻辑和代码路径：训练数据是 episode-wise zarr/HF dataset，memory/gated policy 使用同一 episode 内多段 action chunk 组成 multi-traj batch；训练入口是 `imitation-learning-policies/shell_scripts/train_sim.sh` / `scripts/train_policy.py`，gate 需要离线生成标签并单独训练后冻结用于 gated policy。 |
