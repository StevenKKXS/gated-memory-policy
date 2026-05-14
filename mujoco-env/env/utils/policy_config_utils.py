from dataclasses import dataclass
from typing import Any


_REQUIRED_TIMING_KEYS = (
    "proprio_length",
    "image_length",
    "action_length",
    "action_indices",
    "image_indices",
    "proprio_indices",
)


@dataclass(frozen=True)
class PolicyRolloutTiming:
    proprio_length: int
    image_length: int
    action_length: int
    action_prediction_horizon: int
    action_indices: list[int]
    image_indices: list[int]
    proprio_indices: list[int]
    image_obs_frames_ids: list[int]
    proprio_obs_frames_ids: list[int]
    obs_history_len: int
    source: str


def _as_int_list(values: Any) -> list[int]:
    return [int(value) for value in list(values)]


def _history_len_from_frame_ids(frame_ids: list[int]) -> int:
    if not frame_ids:
        return 0
    return max(abs(min(frame_ids)), len(frame_ids))


def _timing_section(policy_config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    workspace = policy_config["workspace"]
    model_section = workspace.get("model")
    if model_section is not None and all(
        key in model_section for key in _REQUIRED_TIMING_KEYS
    ):
        return model_section, "model"

    train_dataset_section = workspace.get("train_dataset")
    if train_dataset_section is not None and all(
        key in train_dataset_section for key in _REQUIRED_TIMING_KEYS
    ):
        return train_dataset_section, "train_dataset"

    raise KeyError(
        "Policy config must contain timing fields under workspace.model or "
        "workspace.train_dataset"
    )


def extract_policy_rollout_timing(
    policy_config: dict[str, Any],
) -> PolicyRolloutTiming:
    timing_section, source = _timing_section(policy_config)
    proprio_length = int(timing_section["proprio_length"])
    image_length = int(timing_section["image_length"])
    action_length = int(timing_section["action_length"])
    action_indices = _as_int_list(timing_section["action_indices"])
    image_indices = _as_int_list(timing_section["image_indices"])
    proprio_indices = _as_int_list(timing_section["proprio_indices"])

    action_prediction_horizon = action_length
    if action_indices and action_indices[0] < 0:
        action_prediction_horizon = sum(idx >= 0 for idx in action_indices)

    image_obs_frames_ids = [idx - 1 for idx in image_indices]
    proprio_obs_frames_ids = [idx - 1 for idx in proprio_indices]
    obs_history_len = max(
        proprio_length,
        image_length,
        _history_len_from_frame_ids(image_obs_frames_ids),
        _history_len_from_frame_ids(proprio_obs_frames_ids),
    )

    return PolicyRolloutTiming(
        proprio_length=proprio_length,
        image_length=image_length,
        action_length=action_length,
        action_prediction_horizon=action_prediction_horizon,
        action_indices=action_indices,
        image_indices=image_indices,
        proprio_indices=proprio_indices,
        image_obs_frames_ids=image_obs_frames_ids,
        proprio_obs_frames_ids=proprio_obs_frames_ids,
        obs_history_len=obs_history_len,
        source=source,
    )


def needs_full_image_history(timing: PolicyRolloutTiming) -> bool:
    return len(timing.image_obs_frames_ids) > 1 or any(
        idx < -1 for idx in timing.image_obs_frames_ids
    )


def render_indices_for_policy_images(
    timing: PolicyRolloutTiming,
    action_execution_horizon: int,
) -> list[int]:
    if not timing.image_obs_frames_ids:
        return []
    if needs_full_image_history(timing):
        return list(range(-int(action_execution_horizon), 0))
    return [-1]
