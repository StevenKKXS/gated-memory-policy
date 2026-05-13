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
