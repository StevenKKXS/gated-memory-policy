# task003_robomimic_memory_gate_repro - History Log

<!-- METADATA:SESSION=3 -->

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
