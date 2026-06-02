# Simulation Setup Notes

<!-- METADATA:SESSION=4 -->

## Paths

- Code: `/mnt/3fs1/data/tingwen.du/gated-memory-policy`
- Checkpoints: `/mnt/3fs1/data/tingwen.du/gated-memory-policy-data/checkpoints`
- HF cache: `/mnt/3fs1/data/tingwen.du/gated-memory-policy-data/hf-home`
- Venv root: `/mnt/3fs1/data/tingwen.du/gated-memory-policy-envs/task003_robomimic_memory_gate_repro`
- Run logs: `/mnt/3fs1/data/tingwen.du/gated-memory-policy-runs/task003_robomimic_memory_gate_repro`

## Venvs

- Policy / ckpt loading: `imitation-py312`
- RoboMimic / MuJoCo: `mujoco-py312`
- Mikasa / ManiSkill: `mikasa-py312`

All venvs have a site `pip.conf` pointing to `http://10.100.197.13/simple/`.

## Checkpoints Downloaded

Downloaded and size-checked from `yihuai-gao/gated-memory-policy`:

- `robomimic/*_diffusion_gated.ckpt`
- `robomimic/*_memory_gate.ckpt`
- `mikasa/*_diffusion_memory.ckpt`

Total: 15 files, about 11GB.

## Environment Notes

- CPU/GPU nodes only had Python 3.12, so the venvs are Python 3.12.
- `mujoco-py312` uses `ray==2.31.0` because `ray==2.9.0` does not support Python 3.12.
- `mikasa-py312` uses PyPI beta packages `mani-skill==3.0.0b15`, `sapien==3.0.0b1`, and `pytorch-kinematics==0.7.4`; the internal mirror stable `mani-skill==3.0.1` lacks `xmate3`.
- RoboMimic policy checkpoints need `google/siglip2-base-patch16-256`; it is cached under the HF cache path above for offline GPU loading.

## Smoke Result

- RoboMimic square smoke run: `/mnt/3fs1/data/tingwen.du/gated-memory-policy-runs/task003_robomimic_memory_gate_repro/robomimic_square_smoke_20260602_133014`
- Checkpoint: `robomimic/robomimic_square_ph_diffusion_gated.ckpt`
- Result: 1 episode completed, reward `0.0`, success rate `0.0`; `episode_data.zarr` and failure mp4 files were saved.

## Smoke Commands

RoboMimic policy server:

```bash
REPO=/mnt/3fs1/data/tingwen.du/gated-memory-policy
ENV_ROOT=/mnt/3fs1/data/tingwen.du/gated-memory-policy-envs/task003_robomimic_memory_gate_repro
CKPT_ROOT=/mnt/3fs1/data/tingwen.du/gated-memory-policy-data/checkpoints
cd $REPO/imitation-learning-policies
export HF_HOME=/mnt/3fs1/data/tingwen.du/gated-memory-policy-data/hf-home
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
PYTHONPATH=$PWD $ENV_ROOT/imitation-py312/bin/python -u scripts/run_policy_server.py \
  --ckpt_path $CKPT_ROOT/robomimic/robomimic_square_ph_diffusion_gated.ckpt \
  --server_endpoint tcp://0.0.0.0:18923
```

RoboMimic MuJoCo env smoke:

```bash
REPO=/mnt/3fs1/data/tingwen.du/gated-memory-policy
ENV_ROOT=/mnt/3fs1/data/tingwen.du/gated-memory-policy-envs/task003_robomimic_memory_gate_repro
cd $REPO/mujoco-env
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0 MUJOCO_EGL_DEVICE_ID=0
PYTHONPATH=$PWD $ENV_ROOT/mujoco-py312/bin/python -u scripts/rollout_policy.py robomimic_square \
  policy_server_port=18923 \
  task.use_viewer=false task.show_camera_images=false task.keyboard_address=null \
  task.data_storage_dir=/mnt/3fs1/data/tingwen.du/gated-memory-policy-runs/task003_robomimic_memory_gate_repro/robomimic_square_smoke/rollout \
  episode_num=1 start_seed=10000
```

Mikasa local smoke:

```bash
REPO=/mnt/3fs1/data/tingwen.du/gated-memory-policy
ENV_ROOT=/mnt/3fs1/data/tingwen.du/gated-memory-policy-envs/task003_robomimic_memory_gate_repro
CKPT_ROOT=/mnt/3fs1/data/tingwen.du/gated-memory-policy-data/checkpoints
cd $REPO/mikasa-robo-env
PYTHONPATH=$PWD:$REPO/imitation-learning-policies $ENV_ROOT/mikasa-py312/bin/python eval/mikasa_eval.py \
  --env-id ShellGameTouch-v0 \
  --checkpoint $CKPT_ROOT/mikasa/mikasa_shell_game_touch_diffusion_memory.ckpt \
  --num-envs 1 --num-eval-episodes 1 --abs-joint-pos \
  --output-dir /mnt/3fs1/data/tingwen.du/gated-memory-policy-runs/task003_robomimic_memory_gate_repro/mikasa_smoke
```
