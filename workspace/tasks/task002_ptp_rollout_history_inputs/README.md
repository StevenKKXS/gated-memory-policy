# task002_ptp_rollout_history_inputs

<!-- METADATA:STATUS=Completed,ASSIGNEE=intern_baseline_explorer,BRANCH=intern_baseline_explorer/task002_ptp_rollout_history_inputs,PR=https://github.com/StevenKKXS/gated-memory-policy/pull/2 -->

## 目标

修复 RoboMimic PTP release checkpoint 的 rollout 输入管线，使 mid-history / long-history checkpoint 在单进程与并行 rollout 中拿到与训练配置一致的图像、proprio 和未来 action horizon。

## 方案

- 从 policy release 配置中解析 image/proprio/action indices，并统一设置 agent 的 history frame ids。
- 并行 rollout 根据 history 需求渲染足够的 image frame，避免只保留最后一帧图像。
- RoboMimic agent 支持 `eef_pos`、`eef_quat`、`gripper_qpos` 低维观测过滤。
- 在发往 policy server 前增加 history shape 检查，并检查返回 action horizon 足够。

## 验证

- `python -m compileall` 覆盖修改过的 rollout、agent、task 和 config helper 文件。
- 轻量脚本验证 nohist / midhist PTP / longhist PTP 的解析结果：
  - nohist: `obs_history_len=1`, `action_horizon=16`, `render_image_indices=[-1]`
  - midhist PTP: `obs_history_len=16`, `action_horizon=16`, `render_image_indices=[-8..-1]`
  - longhist PTP: `obs_history_len=121`, `action_horizon=16`, `render_image_indices=[-8..-1]`
- GPU 验证在 `10.100.10.31:24936` 完成，GPU 不联网；代码、checkpoint、HF cache 均通过 NFS 准备，并复制到 GPU 本地 `/tmp/task002_gmp_local` 运行。
- GPU rollout 结果：
  - midhist 单进程 1 episode，seed 10000：完整结束，无 history shape 错误，成功率 `0/1`
  - midhist 并行 2 episodes，seed 10005，env_num 2：完整结束，原 `traj_len(1) != meta.length(16)` 不再出现，成功率 `0/2`
  - longhist 单进程 1 episode，seed 10005：完整结束，无 index 越界或 history shape 错误，成功率 `0/1`
  - midhist 并行 10 episodes，seed 10005，env_num 5：完整结束，成功率 `0/10`
- NFS 结果清单：`/mnt/nfs/tingwen/gated-memory-policy/intern_baseline_explorer/tasks/task002_ptp_rollout_history_inputs/manifests/rollout_validation_pr2.tsv`
- Ceph 小文件归档：`/mnt/cephfs/home/tinwen.du/gated-memory-policy/intern_baseline_explorer/task_archives/task002_ptp_rollout_history_inputs/`
