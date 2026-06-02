# task003_robomimic_memory_gate_repro - History Log

<!-- METADATA:SESSION=6 -->

## Session 1 - 2026-06-02

- 创建任务分支 `intern_baseline_explorer/task003_robomimic_memory_gate_repro`，基线为 `origin/main`。
- 确认 GPU 资源 `10.100.2.39:23494` 可 SSH 登录，主机名 `lg-cmc-b7r201-e03u26-h200-000102`。
- 确认 GPU 节点有 8 张 NVIDIA H200。
- 确认 3fs 挂载为 `/mnt/3fs1`，用户数据目录为 `/mnt/3fs1/data/tingwen.du`。
- 确认远端暂无 conda/mamba，系统 Python 为 3.12.3，需要新建隔离环境。
- 阅读 README：policy 侧环境名 `imitation`，MuJoCo rollout 侧环境名 `mujoco-env`；RoboMimic 任务需安装仓库内 submodule 的 `third_party/robosuite` 和 `third_party/robomimic`。
- 复查历史任务结论：RoboMimic Tool Hang(ph) no-history release ckpt 曾可成功 rollout，但 mid/long history PTP release ckpt 未复现论文 reported success rate；headless 运行需 `MUJOCO_GL=egl` 等覆盖。

## Session 2 - 2026-06-02

- 创建 PR：https://github.com/StevenKKXS/gated-memory-policy/pull/3
- 已将 `origin/main` 的 git bundle 上传到 GPU 节点：`/mnt/3fs1/data/tingwen.du/gated-memory-policy-runs/task003_robomimic_memory_gate_repro/gated-memory-policy-origin-main.bundle`。
- 尝试从 bundle 恢复 `/mnt/3fs1/data/tingwen.du/gated-memory-policy` 时未生成可用 `main` ref，命令停在 `fatal: invalid reference: main`；远端代码副本尚未完成。
- 主管说明使用 venv 的原因是 GPU 侧需要 pip 镜像，GPU 不能直接联网；已暂停 GPU 侧 venv 配置、checkpoint 下载和 eval，等待镜像 / venv 方案后继续。

## Session 3 - 2026-06-02

- 主管提供 GPU 内部 pip 镜像：`http://10.100.197.13/simple/`，trusted host：`10.100.197.13`。
- 已在 GPU 节点 `10.100.2.39:23494` 写入 pip 配置：
  - `/etc/xdg/pip/pip.conf`
  - `/etc/pip.conf`
  - `/usr/pip.conf`
  - `/root/.pip/pip.conf`
  - `/root/.config/pip/pip.conf`
- `python3 -m pip config unset --global global.extra-index-url` 在该 GPU 上触发 pip 内部错误，已改用 Python `configparser` 清理各层 pip 配置中的公网 `extra-index-url` 残留，避免离线 GPU 安装依赖时访问公网源。
- 已复制当前 GPU 配置到任务运行目录：
  - `/mnt/3fs1/data/tingwen.du/gated-memory-policy-runs/task003_robomimic_memory_gate_repro/configs/pip.conf`
  - `/mnt/3fs1/data/tingwen.du/gated-memory-policy-runs/task003_robomimic_memory_gate_repro/configs/system-site-pip.conf`
  - `/mnt/3fs1/data/tingwen.du/gated-memory-policy-runs/task003_robomimic_memory_gate_repro/configs/etc-pip.conf`
- 已在仓库任务目录新增可复用模板：`workspace/tasks/task003_robomimic_memory_gate_repro/gpu_pip.conf`。

## Session 4 - 2026-06-02

- 按主管要求优先在 CPU 节点执行网络操作；CPU 主机可直接访问 `/mnt/3fs1/data/tingwen.du`。
- 修复 3fs 代码目录：将此前失败的空 checkout 移到 `/mnt/3fs1/data/tingwen.du/gated-memory-policy.broken-20260602-123248`，重新 clone `git@github.com:StevenKKXS/gated-memory-policy.git`，切到 `intern_baseline_explorer/task003_robomimic_memory_gate_repro`，并完成 robosuite/robomimic submodule 初始化。
- 从 Hugging Face model repo 选择并下载 gated-memory 仿真 ckpt：RoboMimic `*_diffusion_gated.ckpt` + `*_memory_gate.ckpt`，Mikasa `*_diffusion_memory.ckpt`，共 15 个文件。
- 直接用 HF 写 3fs 曾两次出现 consistency check / incomplete 文件；已改为 CPU 本地盘 `/work-agents/intern_baseline_explorer/outputs/task003_ckpt_download_tmp` 下载完整文件，校验大小后复制到 3fs。
- ckpt 目标目录：`/mnt/3fs1/data/tingwen.du/gated-memory-policy-data/checkpoints`，已按 HF metadata 校验 15/15 文件大小匹配，总量约 11GB。
- 创建 3 个隔离 venv，均位于 `/mnt/3fs1/data/tingwen.du/gated-memory-policy-envs/task003_robomimic_memory_gate_repro/`：
  - `imitation-py312`：policy / checkpoint loading，约 7.8GB。
  - `mujoco-py312`：RoboMimic + MuJoCo rollout env，约 7.6GB。
  - `mikasa-py312`：Mikasa / ManiSkill env，约 8.3GB。
- 由于 CPU/GPU 系统仅有 Python 3.12，本轮使用 Python 3.12 venv；这与 README 中 conda Python 3.10/3.11 不完全一致。
- MuJoCo env 中 `ray==2.9.0` 不支持 Python 3.12，已改用内部镜像最早可用 Py3.12 版本 `ray==2.31.0`。
- Mikasa env 内部镜像没有 README 指定 beta 包，已在 CPU 侧从公网 PyPI 安装 `mani-skill==3.0.0b15`、`sapien==3.0.0b1`、`pytorch-kinematics==0.7.4`；GPU 未参与外网访问。
- 验证结果：
  - CPU 侧 `imitation-py312` 可 import `imitation_learning` 并加载 `robomimic_square_ph_diffusion_gated.ckpt`，ckpt epoch 为 11，配置包含 `diffusion_gated`。
  - CPU 侧 `mujoco-py312` 可 import `mujoco==3.3.5`、`robosuite==1.5.1`、`robomimic==0.4.0`、`ray==2.31.0`。
  - CPU 侧 `mikasa-py312` 可 import `mikasa_robo_suite`，并可找到 `mani_skill.agents.robots.xmate3.xmate3`。
  - GPU 节点 `10.100.2.39:23494` 上 `imitation-py312` 可见 8 张 H200，`torch==2.8.0+cu128` CUDA 可用。
  - GPU 节点上 `mujoco-py312` 可 import RoboMimic/MuJoCo，并成功创建 `MUJOCO_GL=egl` context；退出时出现一个 EGL destructor ignored exception，但 context 创建已成功。
  - GPU 节点上 `mikasa-py312` 可 import ManiSkill beta、SAPIEN beta、xmate3 和 `mikasa_robo_suite`，CUDA 可见。
- 新增复用说明：`workspace/tasks/task003_robomimic_memory_gate_repro/simulation_setup.md`。
- 首次启动 RoboMimic policy server 时发现离线依赖缺失：ckpt 实例化会调用 `SiglipVisionModel.from_pretrained("google/siglip2-base-patch16-256")`，GPU 无法联网导致失败。
- 已在 CPU 侧下载 `google/siglip2-base-patch16-256` 的 7 个 snapshot 文件到本地 cache，再复制到 3fs HF cache：`/mnt/3fs1/data/tingwen.du/gated-memory-policy-data/hf-home`，大小约 1.5GB；`TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1` 下已验证可加载。
- 重新在 GPU 节点启动 policy server，使用 ckpt `/mnt/3fs1/data/tingwen.du/gated-memory-policy-data/checkpoints/robomimic/robomimic_square_ph_diffusion_gated.ckpt`，server 端口 `18923`，成功加载 SigLIP、MemoryGate 和 diffusion policy。
- 已跑通 RoboMimic square 1-episode smoke：MuJoCo task 参数为 `robomimic_square`，run 目录 `/mnt/3fs1/data/tingwen.du/gated-memory-policy-runs/task003_robomimic_memory_gate_repro/robomimic_square_smoke_20260602_133014`。
- Smoke 结果：episode 执行完成，policy inference 约 `0.096s`/chunk，episode reward `0.0`，success rate `0.0`，`episode_data.zarr` 和 4 个 failure mp4 已保存。
- Smoke 结束后已停止临时 policy server，GPU 上未残留 `run_policy_server.py` / `rollout_policy.py` / `mikasa_eval.py` / pip / HF download 进程。

## Session 5 - 2026-06-02

- 回答主管关于 simulation benchmark、环境差异和环境职责的三个问题。
- 从仓库 README 确认 simulation benchmark family 包括 Memimic/MemMimic、RoboMimic、Mikasa-Robo；MemMimic 确认存在，使用 `mujoco-env`。
- 从 `mujoco-env/README.md` 和配置确认 MemMimic/MuJoCo 任务包括 `pick_and_match_color`、`pick_and_match_color_rand_delay`、`pick_and_place_back`、`push_cube`、`fling_cloth`；RoboMimic MuJoCo 任务包括 `robomimic_square`、`robomimic_tool_hang`、`robomimic_transport`。
- 从 `mikasa-robo-env/README.md` 确认当前 README 列出的 Mikasa-Robo eval env/checkpoint 包括 `ShellGameTouch-v0`、`InterceptMedium-v0`、`RememberColor3-v0`、`RememberColor5-v0`、`RememberColor9-v0`；policy configs 里还有更多 Mikasa 任务配置。
- 对比仓库推荐 env：
  - `imitation-learning-policies/env.yaml` 推荐 conda `imitation` + Python 3.10；当前 `imitation-py312` 为 venv + Python 3.12.3，主要 Python 包版本与推荐基本一致，`pip` 为 26.1.2，`torch` 当前为 2.8.0。
  - `mujoco-env/env.yaml` 推荐 conda `mujoco-env` + Python 3.10.15 + `ray-core=2.9.0`；当前 `mujoco-py312` 为 Python 3.12.3，`ray==2.31.0`，原因是 `ray==2.9.0` 不支持 Python 3.12。MuJoCo/torch/robosuite/robomimic 关键版本已验证：`mujoco==3.3.5`、`torch==2.8.0`、submodule `robosuite==1.5.1`、`robomimic==0.4.0`。
  - `mikasa-robo-env/env.yml` 推荐 conda `mikasa` + Python 3.11.15；当前 `mikasa-py312` 为 Python 3.12.3，大多数关键 pip pin 与推荐一致，包括 `torch==2.10.0`、`transformers==5.3.0`、`mani-skill==3.0.0b15`、`sapien==3.0.0b1`。
- 明确环境职责：`imitation-learning-policies` 是训练/推理服务环境，服务 MemMimic、RoboMimic、Mikasa-Robo 以及 real；`mujoco-env` 是 MemMimic + RoboMimic 的 MuJoCo simulator；`mikasa-robo-env` 是 Mikasa-Robo 的 ManiSkill simulator。

## Session 6 - 2026-06-02

- 回答主管关于“能否用 venv 全部对齐 GitHub 推荐环境”的可行性问题，本轮只做分析，不改现有环境。
- 实测 CPU host 和 GPU 节点 `10.100.2.39:23494` 都只有 `/usr/bin/python3.12` / Python 3.12.3；未发现 `python3.10`、`python3.11`、conda、mamba、micromamba、uv、pyenv。
- 结论：现有 venv 不能原地把 Python 3.12 改成推荐的 3.10/3.11；`venv` 绑定创建时使用的解释器。要对齐，必须先提供 Python 3.10/3.11 解释器，再用对应解释器创建新 venv。
- 推荐对齐目标：
  - `imitation`：新增 `imitation-py310`，Python 3.10，按 `imitation-learning-policies/env.yaml` 安装。
  - `mujoco-env`：新增 `mujoco-py310`，Python 3.10.15，按 `mujoco-env/env.yaml` 安装；届时可把当前为适配 Python 3.12 才升级的 `ray==2.31.0` 降回推荐 `ray-core==2.9.0`。
  - `mikasa`：新增 `mikasa-py311`，Python 3.11.15，按 `mikasa-robo-env/env.yml` 安装。
- 可行路径：在 CPU 侧下载或构建 portable CPython 3.10/3.11 到 3fs，再在 GPU 使用这些解释器创建 venv；所有 pip 配置继续使用内部镜像 `http://10.100.197.13/simple/`，公网下载放在 CPU 侧完成。
- 风险：venv 能做到 Python 和大部分 pip 包版本对齐，但不能完全复刻 conda solver / conda channel 里的二进制包选择；因此目标应定义为“功能和关键版本对齐”，不是 bit-identical conda 环境。
- 空间可行：3fs `/mnt/3fs1` 约 883T 可用；当前 3 个 py312 venv 总量约 24GB，保留旧环境并新增对齐 venv 的空间风险低。
