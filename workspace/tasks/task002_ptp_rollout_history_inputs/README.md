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
