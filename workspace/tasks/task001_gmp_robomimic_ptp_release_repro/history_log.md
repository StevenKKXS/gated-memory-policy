# task001_gmp_robomimic_ptp_release_repro - History Log

<!-- METADATA:SESSION=1 -->

## Session 1 - 2026-05-13

- 创建任务，明确复现目标和存储规则。
- 已确认当前代码仓库 origin 为 `git@github.com:StevenKKXS/gated-memory-policy.git`。
- 已确认 NFS 挂载在 `/mnt/nfs`，并创建本任务 NFS/CEPH 目录。
- 初步阅读项目页和 README，确认 RoboMimic 评测涉及 Tool Hang、Square、Transport，`ph`/`mh` 表示数据采集来源类型。
- 已创建 PR：https://github.com/StevenKKXS/gated-memory-policy/pull/1
- 阅读 arXiv TeX source 并记录 RoboMimic claim：Tool Hang(ph) 表中 no-hist DP 82%、mid-hist PTP 83%、long-hist PTP 32%；论文说明每个 checkpoint 用 100 episodes 评测，报告最佳 checkpoint。
- CPU 侧下载并转移所需 release ckpt、SigLIP base model cache、Python wheelhouse、GLVND/EGL runtime deb 解包文件到 NFS；GPU 侧使用 `/tmp/task001_gmp_py312_site` 作为隔离 target-site，没有触碰已有环境。
- 修复 GPU headless rollout 依赖：通过 NFS `libEGL.so.0` / `libOpenGL.so.0` / `libGLdispatch.so.0` 让 MuJoCo EGL context smoke test 通过；关闭 single-process viewer 避免 GLFW/DISPLAY 依赖。
- 验证 `robomimic_tool_hang_ph_midhist_ptp_diffusion.ckpt` 可离线加载，policy config 显示 task 为 `robomimic_tool_hang_ph`，policy 为 `midhist_ptp_diffusion_transformer`，action indices 为 `[-16..15]`，image/proprio indices 为 `[-15..0]`。
- 运行 Tool Hang(ph) release ckpt rollout：
  - mid-hist PTP seed 10000-10009：0/10 成功。
  - mid-hist PTP seed 0-4：0/5 成功。
  - no-history diffusion 对照 seed 10005-10014：9/10 成功。
  - long-hist PTP seed 10005-10009：0/5 成功。
- 复核官方 parallel rollout 入口：当前 release 的 `rollout_policy_parallel.py` 对 RoboMimic 只渲染 `[-1]` 图像，mid-hist PTP checkpoint 要求 16 帧图像，直接报 `traj_len(1) != meta.length(16)`，因此该入口不能原样评测 mid/long-history PTP。
- 汇总结果写入 NFS manifest：`manifests/rollout_results_toolhang_ptp.tsv`。
- 清理 GPU 上本任务启动的 policy servers，确认 4 张 GPU 显存回到空闲。
- 已将 NFS 小文件、日志、manifest、论文材料备份到 Ceph：`/mnt/cephfs/home/tinwen.du/gated-memory-policy/intern_baseline_explorer/task_archives/task001_gmp_robomimic_ptp_release_repro/task001_small_files_20260513_150030.tar.gz`。
