# task003_robomimic_memory_gate_repro - Task Knowledge

<!-- METADATA:SESSION=3 -->

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
- GPU 节点不能直接联网；后续 Python 环境应使用 venv + 可用 pip 镜像方案，不先走 conda 在线安装。
- `/mnt/3fs1/data/tingwen.du/gated-memory-policy` 当前需要重新恢复为可用 git checkout，再继续 submodule、venv、checkpoint 和 eval。
- GPU 内部 pip 镜像配置必须在新 GPU 上先执行，再创建 venv / 安装依赖：
  - `python3 -m pip config set --user global.index-url http://10.100.197.13/simple/`
  - `python3 -m pip config set --user global.trusted-host 10.100.197.13`
  - 如系统 site 配置覆盖用户配置，并且当前用户有权限，则同步执行 `python3 -m pip config set --site global.index-url http://10.100.197.13/simple/` 和 `python3 -m pip config set --site global.trusted-host 10.100.197.13`。
  - 清理公网 extra index：`python3 -m pip config unset --user global.extra-index-url || true`，必要时对 `--site` / `--global` 也执行同样命令。
  - 检查命令：`python3 -m pip config list -v`，应看到 `global.index-url='http://10.100.197.13/simple/'` 和 `global.trusted-host='10.100.197.13'`。
- 可直接复制模板 `workspace/tasks/task003_robomimic_memory_gate_repro/gpu_pip.conf` 到新 GPU 的 `~/.config/pip/pip.conf`。
- 如果 `python3 -m pip config debug` 看到 `/etc/xdg/pip/pip.conf`、`/etc/pip.conf` 或 site config 里仍有公网 `extra-index-url`，必须先移除；否则 GPU 离线安装可能会等待公网源超时。
