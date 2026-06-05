# task003_robomimic_memory_gate_repro - History Log

<!-- METADATA:SESSION=11 -->

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

## Session 7 - 2026-06-02

- 按主管要求新增 Python 版本对齐 venv，保留已有 py312 venv 不动；CPU 侧构建并放置 portable CPython：
  - `/mnt/3fs1/data/tingwen.du/gated-memory-policy-envs/python-interpreters/cpython-3.10.15`
  - `/mnt/3fs1/data/tingwen.du/gated-memory-policy-envs/python-interpreters/cpython-3.11.15`
- 新增 3 个 venv：
  - `imitation-py310`：Python 3.10.15，关键版本 `torch==2.8.0`、`torchvision==0.23.0`、`transformers==4.48.3`、`diffusers==0.33.1`、`accelerate==1.3.0`、`peft==0.14.0`。
  - `mujoco-py310`：Python 3.10.15，关键版本 `mujoco==3.3.5`、`dm_control==1.0.31`、`ray==2.9.0`、editable `robosuite==1.5.1`、editable `robomimic==0.4.0`；`pip check` 通过。
  - `mikasa-py311`：Python 3.11.15，关键版本 `torch==2.10.0`、`torchvision==0.25.0`、`transformers==5.3.0`、`mani_skill==3.0.0b15`、`sapien==3.0.0b1`、`mplib==0.1.1`；`pip check` 通过。
- 三套新 venv 的 effective pip 配置均为内部镜像：`global.index-url=http://10.100.197.13/simple/`、`global.trusted-host=10.100.197.13`。
- CPU/GPU smoke：
  - `imitation-py310` 可离线加载 `robomimic_square_ph_diffusion_gated.ckpt`，GPU CUDA 可见 8 张 H200。
  - `mujoco-py310` 在 GPU 上通过 `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` 创建 MuJoCo EGL context。
  - `mikasa-py311` 在 CPU/GPU 上 import 通过；注意 `pip show mani_skill` 为 `3.0.0b15`，但 `mani_skill.__version__` 报 `3.0.0b14`。
- RoboMimic smoke：手动 venv policy server + MuJoCo env server + orchestrator 跑通 2 episodes，run 目录 `/mnt/3fs1/data/tingwen.du/gated-memory-policy-data/eval_runs/session7/robomimic_smoke_20260602_145418`，`robomimic_square_ph_diffusion_gated` success rate `0.5`。
- RoboMimic 完整复现：`env_num=20` 首次启动 Ray/MuJoCo worker 过慢，改用 `env_num=4` 稳定跑完 5 个 task，每个 100 episodes。结果目录：
  - `/mnt/3fs1/data/tingwen.du/gated-memory-policy-data/eval_runs/session7/robomimic_full_env4_20260602_150528`
  - `robomimic_results.md` / `robomimic_results.json`
- RoboMimic 结果：
  - `robomimic_square_mh` epoch 9，success rate `0.89`。
  - `robomimic_square_ph` epoch 11，success rate `0.96`。
  - `robomimic_tool_hang_ph` epoch 18，success rate `0.80`。
  - `robomimic_transport_mh` epoch 6，success rate `0.78`。
  - `robomimic_transport_ph` epoch 20，success rate `0.13`。
- MiKASA/ManiSkill 离线配置补齐：
  - `HF_HOME=/mnt/3fs1/data/tingwen.du/gated-memory-policy-data/hf-home TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1` 用于 SigLIP 离线加载。
  - `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json` 用于 GPU Vulkan ICD。
  - `MS_ASSET_DIR=/mnt/3fs1/data/tingwen.du/gated-memory-policy-data/maniskill-assets` 用于 ManiSkill 资产。
  - ManiSkill `ycb` 官方 checksum 记录已过期；CPU 侧下载 `mani_skill2_ycb.zip` 后 `unzip -t` 通过，实际 sha256 为 `1551724fd1ac7bad9807ebcf46dd4a788caed5c9499c1225b9bfa080ffbefcb3`，已解压到 `.../maniskill-assets/assets/mani_skill2_ycb`。
  - SAPIEN 首次 `physx.enable_gpu()` 会从 GitHub 拉 `linux-so.zip`；已在 CPU 侧下载 `sapien-physx-105.1-physx-5.3.1.patch0-linux-so.zip` 并在 GPU 解压到 `/root/.sapien/physx/105.1-physx-5.3.1.patch0/libPhysXGpu_64.so`。
- MiKASA smoke：`ShellGameTouch-v0` 2 episodes 跑通，run 目录 `/mnt/3fs1/data/tingwen.du/gated-memory-policy-data/eval_runs/session7/mikasa_smoke_physx_20260602_160257`，`success_once=1.0`，`success_at_end=1.0`。
- MiKASA 完整复现：5 个 task 使用 `num-envs=50`、`num-eval-episodes=100`、`--abs-joint-pos`、`seed=42`，并行分配 GPU0-4，run 目录 `/mnt/3fs1/data/tingwen.du/gated-memory-policy-data/eval_runs/session7/mikasa_full_20260602_160517`，`STATUS=0`。
- MiKASA 结果：
  - `mikasa_shell_game_touch` / `ShellGameTouch-v0` epoch 8，`success_once=0.99`，`success_at_end=0.97`。
  - `mikasa_intercept_medium` / `InterceptMedium-v0` epoch 16，`success_once=0.80`，`success_at_end=0.14`。
  - `mikasa_remember_color_3` / `RememberColor3-v0` epoch 32，`success_once=0.98`，`success_at_end=0.36`。
  - `mikasa_remember_color_5` / `RememberColor5-v0` epoch 50，`success_once=0.72`，`success_at_end=0.27`。
  - `mikasa_remember_color_9` / `RememberColor9-v0` epoch 30，`success_once=0.21`，`success_at_end=0.10`。
- 统一结果表已生成：
  - `/mnt/3fs1/data/tingwen.du/gated-memory-policy-data/eval_runs/session7/gmp_results.md`
  - `/mnt/3fs1/data/tingwen.du/gated-memory-policy-data/eval_runs/session7/gmp_results.json`
- 评测结束后检查 GPU：`run_policy_server.py`、`rollout_policy.py`、`mikasa_eval.py`、Ray worker 均无残留，8 张 GPU 显存回到约 `1 MiB`。

## Session 8 - 2026-06-03

- 继续按主管要求跑 MemMimic/GMP；CPU 侧通过 Hugging Face model repo `yihuai-gao/gated-memory-policy` 确认 MemMimic `diffusion_gated` ckpt 只有 4 个：`fling_cloth`、`pick_and_match_color`、`pick_and_place_back`、`push_cube`。仓库中存在 `pick_and_match_color_rand_delay` 的 dense/sparse memory ckpt，但未发现 rand-delay 的 `diffusion_gated` ckpt。
- CPU 本地盘下载 4 个 MemMimic gated ckpt，再复制到 3fs：`/mnt/3fs1/data/tingwen.du/gated-memory-policy-data/checkpoints/memmimic`。ckpt metadata：`fling_cloth` epoch 14 / global_step 22874，`pick_and_match_color` epoch 5 / global_step 5489，`pick_and_place_back` epoch 13 / global_step 12809，`push_cube` epoch 23 / global_step 36575。
- 直接 `run_policy_server.py --ckpt_path ...` + `rollout_policy.py push_cube` smoke 因 readiness 日志判断和手动清理顺序未形成有效 rollout，失败目录为 `/mnt/3fs1/data/tingwen.du/gated-memory-policy-data/eval_runs/session8/memmimic_smoke_push_cube_20260603_013605`。
- 改用 queue workflow 后 MemMimic smoke 跑通：`push_cube` 2 episodes，`env_num=2`，端口 `39101`，目录 `/mnt/3fs1/data/tingwen.du/gated-memory-policy-data/eval_runs/session8/memmimic_smoke_queue_push_cube_20260603_014231`，success rate `1.0`。
- MemMimic 4-task 全量评测使用 `imitation-py310` + `mujoco-py310`，4 个 policy server 端口 `39120-39123`，GPU0-3，`server.rollout_episode_num=100`，`server.env_num=20`，run 目录 `/mnt/3fs1/data/tingwen.du/gated-memory-policy-data/eval_runs/session8/memmimic_full_gated_env20_20260603_014506`。
- MemMimic 100 episode 结果：
  - `fling_cloth` epoch 14，success rate `0.78`。
  - `pick_and_match_color` epoch 5，success rate `1.0`。
  - `pick_and_place_back` epoch 13，success rate `0.97`。
  - `push_cube` epoch 23，success rate `0.98`。
- 结果表已生成：
  - MemMimic 单独表：`/mnt/3fs1/data/tingwen.du/gated-memory-policy-data/eval_runs/session8/memmimic_gated_results.md` 和 `.json`。
  - Session 8 统一表：`/mnt/3fs1/data/tingwen.du/gated-memory-policy-data/eval_runs/session8/gmp_results.md` 和 `.json`；同内容另存为 `gmp_results_with_memmimic.md/json`。
- 评测结束后检查 GPU：`run_policy_server.py`、`serve_remote_env.py`、`start_multi_gpu_mixed_policy_rollout.py` 均无残留，8 张 GPU 显存回到约 `1 MiB`。

## Session 9 - 2026-06-05

- 回答主管关于本任务使用 GPU 资源的问题：本任务复现使用 GPU 节点 `10.100.2.39`，SSH 端口 `23494`，节点为 8 张 NVIDIA H200。
- MemMimic 全量复现使用该节点 GPU0-3，policy server 端口 `39120-39123`；RoboMimic smoke/全量复现也在该节点运行；Mikasa-Robo 全量复现使用该节点 GPU0-4。
- 上一轮 Session 8 结束时已确认该 GPU 节点无 `run_policy_server.py`、`serve_remote_env.py`、`start_multi_gpu_mixed_policy_rollout.py` 残留，8 张 GPU 显存回到约 `1 MiB`。

## Session 10 - 2026-06-05

- 根据主管要求先给出计划，不启动环境复刻或训练进程。
- 新增目标 GPU：`10.100.4.23`，SSH 端口 `21492`。计划先验证 SSH、3fs 挂载、GPU 型号/数量、pip 内部镜像、CUDA/PyTorch 可见性，再复用 3fs 上已有代码、portable Python、venv、HF cache、checkpoints、ManiSkill assets 和 SAPIEN PhysX GPU 库。
- 训练节点分工计划：`10.100.2.39:23494` 作为主 8 卡节点，因为该节点已完成 RoboMimic、Mikasa-Robo、MemMimic 的 smoke/全量 eval；`10.100.4.23:21492` 作为可中断副节点，运行同任务的短周期 smoke、低 epoch 训练、ablation 或 checkpoint 续跑验证，被停止时不影响主节点结果链路。
- 结合仓库 README 和 `imitation-learning-policies/shell_scripts/train_sim.sh`，GMP simulation training 官方入口为 `shell_scripts/train_sim.sh`，默认配置是 MemMimic `pick_and_place_back` + `diffusion_memory_transformer`；`diffusion_gated_transformer` 需要 `data/checkpoints/<benchmark>/<task>_memory_gate.ckpt`。
- 训练复现分阶段计划：
  - 阶段 0：在 CPU/3fs 准备数据集，优先下载 `memmimic/pick_and_place_back` 和必要 normalizer；若训练 smoke 缺数据，再按官方 dataset repo 补齐目标 task。
  - 阶段 1：在主节点用 `imitation-py310` 单卡跑 `+workspace.trainer.debug=True` smoke，验证 dataloader、SigLIP 离线加载、wandb/offline 日志、checkpoint 保存。
  - 阶段 2：在主节点 8 卡跑 `diffusion_memory_transformer` 的 MemMimic `pick_and_place_back` 主训练，输出目录放到 3fs session10 training runs。
  - 阶段 3：使用已有 `pick_and_place_back_memory_gate.ckpt` 或先训练 memory gate，再跑 `diffusion_gated_transformer` 训练；每个可用 checkpoint 做小规模 eval 对齐 Session 8 的 rollout workflow。
  - 阶段 4：副节点同步跑短周期或可中断实验，例如 `push_cube`、`robomimic_square_ph`、低 epoch gated training 或 memory gate 训练验证；输出目录和 pid/log 单独标注为 interruptible。
- 风险记录：官方脚本依赖 conda 激活，但本任务使用 venv；执行时需要改写为直接调用 `imitation-py310/bin/python` / `imitation-py310/bin/accelerate` 或导出 `CONDA_PREFIX` 兼容脚本。训练数据集约 325GB，全量下载需先按 task 定向下载，避免占用和等待过大。

## Session 11 - 2026-06-05

- 结合论文/project page 和本地代码，回答主管关于 GMP 训练数据、训练代码和自己训练流程的问题。
- 结论：论文层面 GMP 是监督式 imitation/diffusion policy 训练，不是在线 RL；核心是对需要记忆的任务把当前观测/action chunk 与同 episode 历史 chunk 组成训练样本，并用 cross-attention 读历史。gate 不是推荐直接端到端一起训练，而是通过 no-memory 与 memory policy 在 held-out 数据上的逐时刻 action error 差异生成标签，再训练 memory gate，最后冻结 gate 训练/使用 gated policy。
- 代码映射：`shell_scripts/train_sim.sh` 选择 benchmark/task/policy，`shell_scripts/train_policy.sh` 调 `accelerate launch scripts/train_policy.py`，`scripts/train_policy.py` 通过 Hydra compose task/policy/dataset config 并实例化 `BaseWorkspace`，`BaseWorkspace` 构造 train/val dataloader 和 EMA model 后调用 trainer。
- 数据映射：HF datasets 按 `memmimic/**`、`robomimic/**`、`iphumi/**`、`real_world/**` 下载；MuJoCo 数据使用 episode-wise zarr，每个 episode 包含相机、tcp pose、gripper、action pose、action gripper 等字段。代码将 pose+gripper 合成 `robot0_10d` / `action0_10d`，memory/gated 使用 `MujocoMultiTrajDataset` 从同一 episode 采样多段间隔 chunk。
- 自训建议：先定向下载一个任务的数据集，例如 MemMimic `pick_and_place_back`；先训练 `diffusion_memory_transformer` 并 eval；如需要 gated，再准备 no-memory 与 memory ckpt，生成 gate labels，训练 `memory_gate.ckpt`，最后训练/加载 `diffusion_gated_transformer`。
