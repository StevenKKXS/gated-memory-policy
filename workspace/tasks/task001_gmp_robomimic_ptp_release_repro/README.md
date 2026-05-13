# task001_gmp_robomimic_ptp_release_repro

<!-- METADATA:STATUS=InProgress,ASSIGNEE=intern_baseline_explorer -->

## 背景

主管要求调研并测试 Gated Memory Policy 论文和 release 资源，重点确认 HuggingFace release 的 RoboMimic PTP checkpoint 是否可以复现论文/项目页中 PTP 相关 claim 的效果。

项目页：https://gated-memory-policy.github.io/

## 目标

- 阅读项目页、论文、GitHub README、HuggingFace model/dataset release，梳理 RoboMimic PTP ckpt、rollout 脚本和环境依赖。
- 解释 RoboMimic ckpt 名称中的 `ph` 和 `mh` 含义。
- 在 CPU 节点完成联网下载、外源库和小文件准备；GPU 节点不联网，仅接收需要转移的数据。
- 不触碰已有 Python/conda 环境；如需环境，创建新的隔离环境。
- 在指定 GPU 资源 `10.100.10.31:24936` 上尝试运行 release checkpoint 的 RoboMimic PTP rollout，记录可复现性、成功率或阻塞原因。

## 验收标准

- 有清晰的 release 资源清单：代码、模型、数据、checkpoint 命名和评测入口。
- 至少完成一个 RoboMimic PTP checkpoint 的加载/rollout smoke test；若不能完成，给出可定位的技术阻塞和复现证据。
- 给出 `ph` / `mh` 的解释和来源。
- 小文件放在 NFS 任务目录，并按规则备份到 Ceph；大文件位置单独记录。
- 不向原始 upstream 推送，不直接推送 `main`/`master`，所有仓库改动通过 `StevenKKXS/gated-memory-policy` 分支和 PR。

## 存储位置

- NFS 小文件目录：`/mnt/nfs/tingwen/gated-memory-policy/intern_baseline_explorer/tasks/task001_gmp_robomimic_ptp_release_repro`
- Ceph 归档目录：`/mnt/cephfs/home/tinwen.du/gated-memory-policy/intern_baseline_explorer/task_archives/task001_gmp_robomimic_ptp_release_repro`
