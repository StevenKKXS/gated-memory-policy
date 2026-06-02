# task003_robomimic_memory_gate_repro - History Log

<!-- METADATA:SESSION=1 -->

## Session 1 - 2026-06-02

- 创建任务分支 `intern_baseline_explorer/task003_robomimic_memory_gate_repro`，基线为 `origin/main`。
- 确认 GPU 资源 `10.100.2.39:23494` 可 SSH 登录，主机名 `lg-cmc-b7r201-e03u26-h200-000102`。
- 确认 GPU 节点有 8 张 NVIDIA H200。
- 确认 3fs 挂载为 `/mnt/3fs1`，用户数据目录为 `/mnt/3fs1/data/tingwen.du`。
- 确认远端暂无 conda/mamba，系统 Python 为 3.12.3，需要新建隔离环境。
- 阅读 README：policy 侧环境名 `imitation`，MuJoCo rollout 侧环境名 `mujoco-env`；RoboMimic 任务需安装仓库内 submodule 的 `third_party/robosuite` 和 `third_party/robomimic`。
- 复查历史任务结论：RoboMimic Tool Hang(ph) no-history release ckpt 曾可成功 rollout，但 mid/long history PTP release ckpt 未复现论文 reported success rate；headless 运行需 `MUJOCO_GL=egl` 等覆盖。
