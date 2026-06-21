from __future__ import annotations

from post_train_v2.src.generation.dpo import (
    DPOCandidate,
    compute_category_quotas,
    select_dpo_pairs,
)


def candidate(
    source_id: str,
    category: str,
    route: str,
    rollout_index: int,
) -> DPOCandidate:
    return DPOCandidate(
        source_id=source_id,
        candidate_id=f"{source_id}:{route}:{category}:{rollout_index}",
        generation_route=route,
        rejected=f"<answer>{rollout_index}</answer>",
        rejected_category=category,
        validation={"ok": False, "error": category},
        rollout_index=rollout_index,
    )


def sft_row(source_id: str) -> dict:
    return {
        "id": source_id,
        "prompt": f"prompt-{source_id}",
        "response": f"chosen-{source_id}",
    }


def test_compute_category_quotas_for_target_6000():
    assert compute_category_quotas(6000) == {
        "wrong_value": 4200,
        "number_mismatch": 900,
        "invalid_expression": 600,
        "missing_answer_tag": 180,
        "truncated": 120,
    }


def test_select_dpo_pairs_uses_one_pair_per_question_and_stable_order():
    rows = [sft_row(f"row-{index}") for index in range(6)]
    candidates = [
        candidate(row["id"], "wrong_value", "forced_wrong", 0)
        for row in rows
    ] + [
        candidate("row-0", "number_mismatch", "high_temp", 1),
        candidate("row-1", "unexpected_correct", "high_temp", 1),
    ]

    first = select_dpo_pairs(rows, candidates, target_size=4, seed=11)
    second = select_dpo_pairs(rows, list(reversed(candidates)), target_size=4, seed=11)

    assert first == second
    assert len(first.pairs) == 4
    assert len({pair["provenance"]["source_id"] for pair in first.pairs}) == 4
    assert all(pair["rejected_category"] != "unexpected_correct" for pair in first.pairs)


def test_select_dpo_pairs_balances_routes_when_supply_permits():
    rows = [sft_row(f"row-{index}") for index in range(8)]
    candidates = [
        candidate(f"row-{index}", "wrong_value", route, index)
        for index, route in enumerate(
            [
                "forced_wrong",
                "high_temp",
                "forced_wrong",
                "high_temp",
                "forced_wrong",
                "high_temp",
                "forced_wrong",
                "high_temp",
            ]
        )
    ]

    result = select_dpo_pairs(rows, candidates, target_size=4, seed=1)

    assert result.route_counts == {"forced_wrong": 2, "high_temp": 2}


def test_select_dpo_pairs_fills_shortfall_by_priority():
    rows = [sft_row(f"row-{index}") for index in range(5)]
    candidates = [
        candidate("row-0", "truncated", "high_temp", 0),
        candidate("row-1", "wrong_value", "forced_wrong", 0),
        candidate("row-2", "number_mismatch", "forced_wrong", 0),
        candidate("row-3", "invalid_expression", "high_temp", 0),
        candidate("row-4", "missing_answer_tag", "high_temp", 0),
    ]

    result = select_dpo_pairs(rows, candidates, target_size=3, seed=1)

    assert [pair["rejected_category"] for pair in result.pairs] == [
        "wrong_value",
        "number_mismatch",
        "invalid_expression",
    ]
    assert result.shortfall == 0
