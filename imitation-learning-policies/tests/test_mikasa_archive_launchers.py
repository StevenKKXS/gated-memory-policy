from pathlib import Path
import os
import subprocess


ROOT = Path(__file__).resolve().parents[2]
LAUNCHERS = ROOT / "experiments" / "mikasa_method_archive" / "launchers"


def _dry_run(script_name: str, tmp_path: Path) -> str:
    env = os.environ.copy()
    env["ICRA_BASE"] = str(tmp_path / "icra_method_dev")
    result = subprocess.run(
        ["bash", str(LAUNCHERS / script_name), "--dry-run"],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_no_memory_launcher_does_not_add_mikasa_policy_only_overrides(tmp_path):
    output = _dry_run("run_no_memory_5task.sh", tmp_path)

    assert "+policy_name=diffusion_transformer" in output
    assert "workspace.model.burn_in_start_id" not in output
    assert "workspace.model.burn_in_loss_traj_num" not in output
    assert "workspace.model.training_traj_sampling_strategy" not in output
    assert "workspace.model.denoising_network_partial.history_retrieval_topk" not in output


def test_top1_launcher_sets_topk_and_index_pool_overrides(tmp_path):
    output = _dry_run("run_top1_retrieval_5task.sh", tmp_path)

    assert "+policy_name=diffusion_mikasa_retrieval_memory_transformer" in output
    assert "++workspace.model.denoising_network_partial.history_retrieval_topk=1" in output
    assert "++workspace.train_dataset.index_pool_size_per_episode=4" in output
