RRF_K = 60


def reciprocal_rank_fusion(ranked_id_lists, k=RRF_K):
    """Fuse several best-first lists of ids into one list of (id, score), best first.

    Each id scores 1 / (k + rank) for every list it appears in (rank starts at 1),
    so ids found by several retrievers rise above ids found by only one.
    """
    scores = {}
    for ranked_ids in ranked_id_lists:
        for rank, item_id in enumerate(ranked_ids, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
