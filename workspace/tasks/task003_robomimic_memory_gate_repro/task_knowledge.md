# task003_robomimic_memory_gate_repro - Task Knowledge

<!-- METADATA:SESSION=4 -->

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
- `/mnt/3fs1/data/tingwen.du/gated-memory-policy` 已恢复为可用 git checkout，当前分支为 `intern_baseline_explorer/task003_robomimic_memory_gate_repro`，robosuite/robomimic submodule 已初始化。
- GPU 内部 pip 镜像配置必须在新 GPU 上先执行，再创建 venv / 安装依赖：
  - `python3 -m pip config set --user global.index-url http://10.100.197.13/simple/`
  - `python3 -m pip config set --user global.trusted-host 10.100.197.13`
  - 如系统 site 配置覆盖用户配置，并且当前用户有权限，则同步执行 `python3 -m pip config set --site global.index-url http://10.100.197.13/simple/` 和 `python3 -m pip config set --site global.trusted-host 10.100.197.13`。
  - 清理公网 extra index：`python3 -m pip config unset --user global.extra-index-url || true`，必要时对 `--site` / `--global` 也执行同样命令。
  - 检查命令：`python3 -m pip config list -v`，应看到 `global.index-url='http://10.100.197.13/simple/'` 和 `global.trusted-host='10.100.197.13'`。
- 可直接复制模板 `workspace/tasks/task003_robomimic_memory_gate_repro/gpu_pip.conf` 到新 GPU 的 `~/.config/pip/pip.conf`。
- 如果 `python3 -m pip config debug` 看到 `/etc/xdg/pip/pip.conf`、`/etc/pip.conf` 或 site config 里仍有公网 `extra-index-url`，必须先移除；否则 GPU 离线安装可能会等待公网源超时。
- 仿真 ckpt 根目录：`/mnt/3fs1/data/tingwen.du/gated-memory-policy-data/checkpoints`。当前已下载并校验 RoboMimic `*_diffusion_gated.ckpt` / `*_memory_gate.ckpt` 和 Mikasa `*_diffusion_memory.ckpt`，共 15 个文件。
- 直接下载到 3fs 可能触发 Hugging Face consistency check 失败；稳定流程是 CPU 本地盘下载完整文件，再校验大小并复制到 3fs。
- RoboMimic policy server 加载 `robomimic_square_ph_diffusion_gated.ckpt` 时需要 Transformers repo `google/siglip2-base-patch16-256`；GPU 离线会直接失败。已在 CPU 侧下载到 HF cache：`/mnt/3fs1/data/tingwen.du/gated-memory-policy-data/hf-home`，运行 policy server 时设置 `HF_HOME`、`TRANSFORMERS_OFFLINE=1`、`HF_HUB_OFFLINE=1`。
- 隔离 venv 根目录：`/mnt/3fs1/data/tingwen.du/gated-memory-policy-envs/task003_robomimic_memory_gate_repro`。当前分为 `imitation-py312`、`mujoco-py312`、`mikasa-py312`，不要和系统 Python 或其他任务 venv 混用。
- 本轮环境使用 Python 3.12，因为 CPU/GPU 节点仅提供 Python 3.12；README 的 conda Python 3.10/3.11 pin 不能原样用 venv 复现。
- `mujoco-py312` 使用 `ray==2.31.0` 替代 README/env 的 `ray==2.9.0`，原因是 `ray==2.9.0` 不支持 Python 3.12。
- `mikasa-py312` 必须使用 PyPI beta：`mani-skill==3.0.0b15`、`sapien==3.0.0b1`、`pytorch-kinematics==0.7.4`；内部镜像只有 stable 版，stable 版缺 `mani_skill.agents.robots.xmate3`。
- GPU 验证：`10.100.2.39:23494` 上 3 个 venv 均可通过关键 import；`imitation-py312` 和 `mujoco-py312` 使用 `torch==2.8.0+cu128`，`mikasa-py312` 使用 `torch==2.10.0+cu128`，均可见 8 张 H200。
- RoboMimic MuJoCo rollout 的 task 参数使用环境名 `robomimic_square`、`robomimic_tool_hang`、`robomimic_transport`；checkpoint 文件名里的 `_ph` / `_mh` 不作为 `scripts/rollout_policy.py` 的 task 参数。
- 已在 GPU 节点跑通 RoboMimic square 1-episode smoke：ckpt `robomimic/robomimic_square_ph_diffusion_gated.ckpt`，run 目录 `/mnt/3fs1/data/tingwen.du/gated-memory-policy-runs/task003_robomimic_memory_gate_repro/robomimic_square_smoke_20260602_133014`，episode 执行完成，success rate `0.0`，失败视频和 zarr 已保存。
