run_archive_lanes() {
  local mode="${1:-}"
  local train_seed="${TRAIN_SEED:-0}"
  local num_epochs="${NUM_EPOCHS:-1}"
  local lr="${LR:-1e-4}"
  local batch_size="${TRAIN_BATCH_SIZE:-16}"
  local num_workers="${NUM_WORKERS:-4}"
  local run_prefix="${RUN_PREFIX:-$METHOD_LABEL}"
  local results_tsv="$LOG_ROOT/${run_prefix}.results.tsv"
  printf 'lane\tenv_id\tpolicy\trun_dir\tlog_path\n' > "$results_tsv"

  for row in "${LANES[@]}"; do
    IFS=$'\t' read -r lane gpu port task_name env_id base_ckpt traj_num max_history_len action_len traj_interval burn_start burn_num sampling retrieval_topk <<< "$row"
    local run_name="${run_prefix}_${lane}_seed${train_seed}"
    local run_dir="$RUN_ROOT/$run_name"
    local log_path="$LOG_ROOT/$run_name.train.log"
    local action_indices
    case "$action_len" in
      2) action_indices='[0,1]' ;;
      8) action_indices='[0,1,2,3,4,5,6,7]' ;;
      10) action_indices='[0,1,2,3,4,5,6,7,8,9]' ;;
      *) action_indices='[]' ;;
    esac

    local cmd=(
      "$PY" -m accelerate.commands.launch
      --gpu_ids "$gpu"
      --num_processes 1
      --main_process_port "$port"
      scripts/train_policy.py
      +policy_name="$POLICY_NAME"
      +task_name="$task_name"
      +logger_project_name=icra_mikasa_method_archive
      +project_name=icra_method_dev
      +run_name="$run_name"
      +train_server_name=10.100.0.3
      +seed="$train_seed"
      +workspace.logging_cfg.mode=offline
      +workspace.trainer.output_dir="$run_dir"
      +workspace.trainer.num_epochs="$num_epochs"
      +workspace.trainer.rollout_every=0
      +workspace.trainer.val_every=0
      +workspace.trainer.sample_every=0
      +workspace.trainer.optimizer_partial.lr="$lr"
      +workspace.train_dataset.compressed_dir="$ICRA_BASE/datasets/mikasa/zarr"
      +workspace.train_dataset.root_dir="$ICRA_BASE/datasets/mikasa/zarr"
      +workspace.train_dataset.normalizer_dir="$ICRA_BASE/datasets/mikasa/zarr"
      +workspace.train_dataset.name="$env_id"
      +workspace.train_dataset.include_episode_num=-1
      +workspace.train_dataset.starting_percentile_max=1.0
      +workspace.train_dataset.dataloader_cfg.batch_size="$batch_size"
      +workspace.train_dataset.dataloader_cfg.num_workers="$num_workers"
      +workspace.train_dataset.split_dataloader_cfg.batch_size="$batch_size"
      +workspace.train_dataset.split_dataloader_cfg.num_workers="$num_workers"
      ++workspace.train_dataset.traj_num="$traj_num"
      ++workspace.train_dataset.traj_interval_min="$traj_interval"
      ++workspace.train_dataset.traj_interval_max="$traj_interval"
      ++workspace.model.denoising_network_partial.max_history_len="$max_history_len"
      ++workspace.model.action_length="$action_len"
      "++workspace.model.action_indices=$action_indices"
      ++workspace.model.history_action_num_per_chunk="$action_len"
    )

    if [[ "$base_ckpt" != "none" ]]; then
      cmd+=(+base_ckpt_path="$base_ckpt")
    fi
    if [[ "${EPISODE_STARTING_IDX_MAX:-}" != "" ]]; then
      cmd+=(++workspace.train_dataset.episode_starting_idx_max="$EPISODE_STARTING_IDX_MAX")
    fi
    if [[ "${INDEX_POOL_SIZE_PER_EPISODE:-}" != "" ]]; then
      cmd+=(++workspace.train_dataset.index_pool_size_per_episode="$INDEX_POOL_SIZE_PER_EPISODE")
    fi
    if [[ "$burn_start" != "none" && "$burn_num" != "none" ]]; then
      if [[ "$burn_start" != "0" || "$burn_num" != "0" ]]; then
        cmd+=(++workspace.model.burn_in_start_id="$burn_start")
        cmd+=(++workspace.model.burn_in_loss_traj_num="$burn_num")
      fi
    fi
    if [[ "$sampling" != "none" && "$sampling" != "random" ]]; then
      cmd+=(++workspace.model.training_traj_sampling_strategy="$sampling")
    fi
    if [[ "$retrieval_topk" != "none" ]]; then
      cmd+=(++workspace.model.denoising_network_partial.history_retrieval_topk="$retrieval_topk")
    fi
    if [[ "${VISUAL_MEMORY_CARRIER_TYPE:-}" != "" ]]; then
      cmd+=(++workspace.model.denoising_network_partial.visual_memory_carrier_type="$VISUAL_MEMORY_CARRIER_TYPE")
    fi
    if [[ "${INCLUDE_ACTION_HISTORY:-}" != "" ]]; then
      cmd+=(++workspace.model.denoising_network_partial.include_action_history="$INCLUDE_ACTION_HISTORY")
    fi
    if [[ "${LATE_CUE_ANCHOR_ENABLED:-}" != "" ]]; then
      cmd+=(++workspace.model.denoising_network_partial.late_cue_anchor_enabled="$LATE_CUE_ANCHOR_ENABLED")
    fi

    mkdir -p "$run_dir" "$LOG_ROOT"
    printf '%s\t%s\t%s\t%s\t%s\n' "$lane" "$env_id" "$POLICY_NAME" "$run_dir" "$log_path" >> "$results_tsv"
    if [[ "$mode" == "--dry-run" ]]; then
      printf '[dry-run:%s] ' "$lane"
      printf '%q ' "${cmd[@]}"
      printf '\n'
    else
      (cd "$CODE_DIR" && "${cmd[@]}") > "$log_path" 2>&1
    fi
  done
}
