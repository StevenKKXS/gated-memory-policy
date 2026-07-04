from __future__ import annotations

import math

import torch


def qframe_candidate_indices(history_len: int, max_candidates: int) -> list[int]:
    if max_candidates <= 0 or history_len <= 0:
        return []
    if history_len <= max_candidates:
        return list(range(history_len))
    if max_candidates == 1:
        return [0]
    return sorted(
        {
            int(math.floor(x))
            for x in torch.linspace(0, history_len - 1, steps=max_candidates)
            .long()
            .tolist()
        }
    )


def fuse_query_scores(
    image_scores: torch.Tensor,
    text_scores: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if image_scores.shape != text_scores.shape:
        raise ValueError(
            "image_scores and text_scores must have the same shape, got "
            f"{tuple(image_scores.shape)} and {tuple(text_scores.shape)}"
        )
    return alpha * image_scores + (1.0 - alpha) * text_scores


def assign_qframe_rows(
    scores: torch.Tensor,
    max_candidates: int,
    high_topk: int,
    low_topk: int,
) -> list[dict[str, object]]:
    if scores.dim() != 1:
        raise ValueError(f"scores must be 1D, got {tuple(scores.shape)}")

    candidate_indices = qframe_candidate_indices(
        history_len=int(scores.shape[0]),
        max_candidates=max_candidates,
    )
    candidate_scores = scores[candidate_indices] if candidate_indices else scores[:0]
    sorted_candidate_positions = (
        torch.argsort(candidate_scores, descending=True).tolist()
        if candidate_indices
        else []
    )
    high_indices = {
        candidate_indices[pos]
        for pos in sorted_candidate_positions[: min(high_topk, len(sorted_candidate_positions))]
    }
    low_start = min(high_topk, len(sorted_candidate_positions))
    low_stop = min(low_start + low_topk, len(sorted_candidate_positions))
    low_indices = {
        candidate_indices[pos]
        for pos in sorted_candidate_positions[low_start:low_stop]
    }

    rows = []
    for hist_idx in range(int(scores.shape[0])):
        if hist_idx not in candidate_indices:
            kind = "history"
        elif hist_idx in high_indices:
            kind = "high"
        elif hist_idx in low_indices:
            kind = "low"
        else:
            kind = "candidate"
        rows.append(
            {
                "history_index": hist_idx,
                "score": float(scores[hist_idx].item()),
                "kind": kind,
            }
        )
    return rows
