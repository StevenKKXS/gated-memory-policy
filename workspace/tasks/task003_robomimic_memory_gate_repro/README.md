# task003_robomimic_memory_gate_repro

<!-- METADATA:STATUS=InProgress,ASSIGNEE=intern_baseline_explorer,BRANCH=intern_baseline_explorer/task003_robomimic_memory_gate_repro,PR= -->

## 背景

主管要求在 GPU 资源 `10.100.2.39:23494` 上，将当前 Gated Memory Policy 代码下载到 3fs 用户目录 `/mnt/3fs1/data/tingwen.du`，从主分支开维护分支，配置隔离 Python 环境，下载 RoboMimic release checkpoint，并评测 memory gate / PTP 相关 checkpoint 的可复现性。

## 目标

- 在 `/mnt/3fs1/data/tingwen.du/gated-memory-policy` 准备一份基于 `origin/main` 的代码副本。
- 创建维护分支 `intern_baseline_explorer/task003_robomimic_memory_gate_repro`。
- 按 GitHub README 准备 policy serving 与 MuJoCo / RoboMimic rollout 环境。
- 下载 RoboMimic checkpoint，优先覆盖 Tool Hang / Square / Transport 的 memory-history 或 PTP checkpoint。
- 运行至少一个 checkpoint 的 smoke eval；若资源允许，扩展到多 episode 结果。

## 验收标准

- 远端代码路径、分支、环境路径、checkpoint 路径明确记录。
- 环境能完成关键 import 或说明可定位阻塞。
- 至少启动一次 RoboMimic checkpoint eval/rollout；如果失败，给出失败命令、日志位置和下一步定位动作。
- 不直接推送主分支；仓库内记录走 intern 分支和 PR。

## 当前注意事项

- GitHub 仓库默认分支为 `main`，本任务按用户口中的 `master` 代码理解为 `origin/main`。
- GPU 节点的 3fs 挂载实际路径为 `/mnt/3fs1`，不是根目录 `/3fs1`。
- 历史任务 `task001_gmp_robomimic_ptp_release_repro` 和 `task002_ptp_rollout_history_inputs` 已记录 RoboMimic PTP release ckpt 存在复现风险，本任务需要在新 GPU 资源上重新验证。
