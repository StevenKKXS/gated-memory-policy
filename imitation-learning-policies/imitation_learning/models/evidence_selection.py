import torch
import torch.nn.functional as F


def _valid_history_mask(
    history_latents: torch.Tensor,
    history_mask: torch.Tensor | None,
) -> torch.Tensor:
    batch_size, history_len = history_latents.shape[:2]
    if history_mask is None:
        return torch.ones(
            batch_size,
            history_len,
            device=history_latents.device,
            dtype=torch.bool,
        )
    assert history_mask.shape == (batch_size, history_len)
    return history_mask.bool()


def causal_candidate_mask(
    history_latents: torch.Tensor,
    history_mask: torch.Tensor | None,
    max_candidates: int,
) -> torch.Tensor:
    """Subsample already-causal history rows to a bounded Q-Frame candidate set."""
    valid_mask = _valid_history_mask(history_latents, history_mask)
    if max_candidates <= 0:
        return torch.zeros_like(valid_mask)

    candidate_mask = torch.zeros_like(valid_mask)
    for batch_idx in range(valid_mask.shape[0]):
        valid_indices = torch.nonzero(valid_mask[batch_idx], as_tuple=False).flatten()
        if valid_indices.numel() <= max_candidates:
            candidate_mask[batch_idx, valid_indices] = True
            continue

        sample_positions = torch.linspace(
            0,
            valid_indices.numel() - 1,
            steps=max_candidates,
            device=valid_indices.device,
        ).long()
        candidate_mask[batch_idx, valid_indices[sample_positions]] = True
    return candidate_mask


def _topk_mask(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    selected = torch.zeros_like(valid_mask)
    if topk <= 0 or valid_mask.shape[1] == 0:
        return selected

    k = min(topk, valid_mask.shape[1])
    masked_scores = scores.masked_fill(valid_mask.logical_not(), -float("inf"))
    topk_indices = torch.topk(masked_scores, k=k, dim=1).indices
    selected.scatter_(1, topk_indices, True)
    return selected & valid_mask


def select_qframe_causal_evidence_masks(
    query: torch.Tensor,
    history_latents: torch.Tensor,
    history_mask: torch.Tensor | None,
    max_candidates: int,
    high_topk: int,
    low_topk: int,
    history_evidence_features: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rank bounded, already-causal history candidates with a query-history cosine score.

    Returns:
        high_mask: top evidence rows used at full token resolution.
        low_mask: next evidence rows used after per-row token pooling.
        candidate_mask: bounded causal candidate rows considered for ranking.
    """
    assert query.dim() == 2
    assert history_latents.dim() == 4
    assert query.shape[0] == history_latents.shape[0]
    if history_evidence_features is None:
        assert query.shape[1] == history_latents.shape[-1]
    else:
        assert history_evidence_features.dim() == 3
        assert history_evidence_features.shape[:2] == history_latents.shape[:2]
        assert query.shape[1] == history_evidence_features.shape[-1]

    candidate_mask = causal_candidate_mask(
        history_latents,
        history_mask,
        max_candidates=max_candidates,
    )
    if history_latents.shape[1] == 0:
        empty = torch.zeros_like(candidate_mask)
        return empty, empty, candidate_mask

    query_norm = F.normalize(query, dim=-1, eps=1e-6)
    if history_evidence_features is None:
        history_keys = history_latents.mean(dim=2)
    else:
        history_keys = history_evidence_features
    history_keys = F.normalize(history_keys, dim=-1, eps=1e-6)
    scores = torch.einsum("bd,bhd->bh", query_norm, history_keys)

    high_mask = _topk_mask(scores, candidate_mask, high_topk)
    low_candidates = candidate_mask & high_mask.logical_not()
    low_mask = _topk_mask(scores, low_candidates, low_topk)
    return high_mask, low_mask, candidate_mask
