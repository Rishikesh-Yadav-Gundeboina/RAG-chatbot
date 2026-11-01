from backend.rag.fusion import reciprocal_rank_fusion


def test_id_in_both_lists_outranks_single_list_ids():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["d", "c"]])
    assert fused[0][0] == "c"


def test_ordering_within_a_single_list_is_preserved():
    fused = reciprocal_rank_fusion([["a", "b", "c"]])
    assert [item_id for item_id, _ in fused] == ["a", "b", "c"]


def test_scores_use_one_over_k_plus_rank():
    fused = dict(reciprocal_rank_fusion([["a"], ["a"]], k=60))
    assert fused["a"] == 2 / 61


def test_empty_lists():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
