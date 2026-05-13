# task001_gmp_robomimic_ptp_release_repro - Task Knowledge

<!-- METADATA:SESSION=1 -->

## 记录规则

- 只记录与本任务复现直接相关的事实、命令、路径、阻塞和结论。
- 小文件、日志、脚本和 manifest 放到 NFS 任务目录。
- 大文件、checkpoint、缓存和虚拟环境不放入 git；位置写入 manifest。

## 知识条目

- 项目页 RoboMimic Results 段落说明：RoboMimic 评测任务为 Tool Hang、Square、Transport；`ph` 表示 proficient-human，`mh` 表示 multi-human。
- HuggingFace model repo 下 `robomimic/` 包含 `*_midhist_ptp_diffusion.ckpt` 与 `*_longhist_ptp_diffusion.ckpt` 等 release checkpoint。
- HuggingFace dataset repo 下 `robomimic/` 包含对应 normalizer 和 tar.lz4 数据包。
- 论文 RoboMimic 评测说明：T7-9 使用 Tool Hang、Square、Transport；每个 checkpoint 评测 100 episodes，报告训练过程中最佳 checkpoint 的 success rate。
- 论文中 PTP 定义：mid-hist PTP 预测 16 个过去动作 `A_{t-16:t}` 和 16 个未来动作 `A_{t:t+16}`；long-hist PTP 预测 120 个过去动作 `A_{t-120:t}` 和 16 个未来动作。
- Tool Hang(ph) 论文图表 claim：no-hist DP 82%、mid-hist PTP 83%、long-hist PTP 32%。
- GPU 隔离环境路径：`/tmp/task001_gmp_py312_site`；NFS 代码镜像：`/mnt/nfs/tingwen/gated-memory-policy/intern_baseline_explorer/tasks/task001_gmp_robomimic_ptp_release_repro/tmp/code/gated-memory-policy`。
- Headless MuJoCo 需要额外 `LD_LIBRARY_PATH=/mnt/nfs/tingwen/gated-memory-policy/intern_baseline_explorer/tasks/task001_gmp_robomimic_ptp_release_repro/tmp/syslibs_glvnd/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH`，并设置 `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0 MUJOCO_EGL_DEVICE_ID=0`。
- single-process rollout 需要覆盖 `task.use_viewer=false task.show_camera_images=false task.keyboard_address=null`；否则会因无 DISPLAY 触发 GLFW 初始化失败。
- long-hist PTP 的 image indices 是稀疏历史 `[-120, -112, ..., 0]`；在 single-process rollout 中需 `task.render_all_images=true`，否则图像时间轴被压缩后会出现负索引越界。
- 复现实验结论：在同一 Tool Hang(ph) 环境下 no-history release ckpt 能达到 9/10，但 mid-hist PTP 0/15、long-hist PTP 0/5；当前 release PTP ckpt 未复现论文 Tool Hang(ph) PTP claim。
