# task003_robomimic_memory_gate_repro - History Log

<!-- METADATA:SESSION=4 -->

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
