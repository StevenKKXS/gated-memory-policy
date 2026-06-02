# task003_robomimic_memory_gate_repro - Task Knowledge

<!-- METADATA:SESSION=1 -->

## 记录规则

- 只记录与本任务复现直接相关的路径、命令、结果和阻塞。
- 大文件 checkpoint、cache、env 不进 git；只记录路径和 manifest。
- 远端运行日志优先保存在 `/mnt/3fs1/data/tingwen.du/gated-memory-policy-runs/task003_robomimic_memory_gate_repro`。

## 知识条目

- 仓库默认分支是 `main`，没有 `master`；用户说的下载 master 代码按 `origin/main` 执行。
- GPU 机器 `10.100.2.39:23494` 的 3fs 路径是 `/mnt/3fs1/data/tingwen.du`。
- README 推荐 checkpoint 下载命令：`hf download yihuai-gao/gated-memory-policy --type model --include "robomimic/**" --local-dir ./data/checkpoints`。
- RoboMimic rollout 需要先 serve policy checkpoint，再由 `mujoco-env` 运行 `shell_scripts/rollout_policy.sh` 或 `rollout_policy_parallel.sh`。
- RoboMimic submodule 版本很关键：README 明确要求 `git submodule update --init --recursive` 后安装 `third_party/robosuite` 和 `third_party/robomimic`。
