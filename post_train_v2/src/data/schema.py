"""Strict canonical schemas for V2 dataset artifacts."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from math import gcd
from typing import Any

from post_train_v2.src.countdown.bucketing import assign_bucket
from post_train_v2.src.countdown.validation import (
    serialize_fraction,
    validate_countdown_expression,
    validate_countdown_response,
)


NORMALIZED_SOURCE_KEYS = {
    "id",
    "source_index",
    "numbers",
    "target",
    "gold_expr",
    "prompt",
    "bucket",
}
BUCKET_KEYS = {
    "num_count",
    "expr_depth",
    "expr_len",
    "has_division",
    "has_subtraction",
    "score",
    "complexity",
    "bucket_key",
}
VALIDATION_KEYS = {
    "ok",
    "value",
    "used_numbers",
    "expression",
    "error",
}
SFT_RECORD_KEYS = NORMALIZED_SOURCE_KEYS | {
    "response",
    "validation",
    "provenance",
}
DPO_RECORD_KEYS = {
    "prompt",
    "chosen",
    "rejected",
    "rejected_category",
    "generation_route",
    "provenance",
}
VERL_RECORD_KEYS = {
    "data_source",
    "prompt",
    "ability",
    "reward_model",
    "extra_info",
}
CHAT_MESSAGE_KEYS = {"role", "content"}
REWARD_MODEL_KEYS = {"style", "ground_truth"}
GROUND_TRUTH_KEYS = {"numbers", "target"}
COMPLEXITIES = {"easy", "medium", "hard"}
VALIDATION_ERRORS = {
    "missing_answer_tag",
    "invalid_expression",
    "number_mismatch",
    "wrong_value",
}
DPO_REJECTED_CATEGORIES = VALIDATION_ERRORS | {"truncated"}
FRACTION_RE = re.compile(r"-?(?:0|[1-9]\d*)/[1-9]\d*\Z")


def validate_normalized_source(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return an unaliased canonical normalized source record."""

    source = _require_mapping("normalized source", row)
    _require_exact_keys("normalized source", source, NORMALIZED_SOURCE_KEYS)

    row_id = _require_nonempty_string("id", source["id"])
    source_index = _require_nonnegative_int("source_index", source["source_index"])
    numbers = _require_numbers("numbers", source["numbers"])
    target = _require_nonnegative_int("target", source["target"])
    gold_expr = _require_nonempty_string("gold_expr", source["gold_expr"])
    prompt = _require_nonempty_string("prompt", source["prompt"])
    bucket = _validate_bucket(source["bucket"], len(numbers))
    gold_validation = validate_countdown_expression(gold_expr, numbers, target)
    if not gold_validation.ok:
        raise ValueError(f"gold_expr failed validation: {gold_validation.error}")
    canonical_bucket = assign_bucket(numbers, gold_expr)
    if bucket != canonical_bucket:
        differing_fields = sorted(
            key for key in BUCKET_KEYS if bucket[key] != canonical_bucket[key]
        )
        raise ValueError(
            "bucket does not match canonical bucket: "
            + ", ".join(differing_fields)
        )

    return {
        "id": row_id,
        "source_index": source_index,
        "numbers": numbers,
        "target": target,
        "gold_expr": gold_expr,
        "prompt": prompt,
        "bucket": bucket,
    }


def validate_sft_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a canonical SFT or RFT record."""

    record = _require_mapping("SFT record", row)
    _require_exact_keys("SFT record", record, SFT_RECORD_KEYS)
    source = validate_normalized_source(
        {key: record[key] for key in NORMALIZED_SOURCE_KEYS}
    )
    response = _require_nonempty_string("response", record["response"])
    validation = _validate_validation(
        record["validation"],
        response=response,
        numbers=source["numbers"],
        target=source["target"],
    )
    provenance = _normalize_json_mapping("provenance", record["provenance"])
    return {
        **source,
        "response": response,
        "validation": validation,
        "provenance": provenance,
    }


def validate_dpo_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a canonical DPO preference pair."""

    record = _require_mapping("DPO record", row)
    _require_exact_keys("DPO record", record, DPO_RECORD_KEYS)
    normalized = {
        field: _require_nonempty_string(field, record[field])
        for field in (
            "prompt",
            "chosen",
            "rejected",
            "rejected_category",
            "generation_route",
        )
    }
    category = normalized["rejected_category"]
    if category not in DPO_REJECTED_CATEGORIES:
        raise ValueError(
            "rejected_category must be one of "
            + ", ".join(sorted(DPO_REJECTED_CATEGORIES))
        )
    if normalized["chosen"] == normalized["rejected"]:
        raise ValueError("chosen and rejected must differ")
    normalized["provenance"] = _normalize_json_mapping(
        "provenance", record["provenance"]
    )
    return normalized


def validate_verl_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a canonical Arrow-friendly verl record."""

    record = _require_mapping("verl record", row)
    _require_exact_keys("verl record", record, VERL_RECORD_KEYS)
    data_source = _require_nonempty_string("data_source", record["data_source"])
    ability = _require_nonempty_string("ability", record["ability"])
    prompt = _validate_chat_prompt(record["prompt"])
    reward_model = _validate_reward_model(record["reward_model"])
    extra_info = _normalize_arrow_mapping("extra_info", record["extra_info"])
    return {
        "data_source": data_source,
        "prompt": prompt,
        "ability": ability,
        "reward_model": reward_model,
        "extra_info": extra_info,
    }


def validate_unique_ids(
    rows: Sequence[Mapping[str, Any]],
    label: str,
) -> list[dict[str, Any]]:
    """Validate row identities and return deep copies in input order."""

    label = _require_nonempty_string("label", label)
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes, bytearray))
        or isinstance(rows, Mapping)
    ):
        raise ValueError(f"{label} rows must be a sequence")

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} row {index} must be a mapping")
        if "id" not in row:
            raise ValueError(f"{label} row {index} is missing id")
        row_id = row["id"]
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"{label} row {index} id must be a nonempty string")
        if row_id in seen:
            raise ValueError(f"{label} contains duplicate id: {row_id}")
        seen.add(row_id)
        result.append(deepcopy(dict(row)))
    return result


def _validate_bucket(value: Any, number_count: int) -> dict[str, Any]:
    bucket = _require_mapping("bucket", value)
    _require_exact_keys("bucket", bucket, BUCKET_KEYS)
    num_count = _require_nonnegative_int("bucket.num_count", bucket["num_count"])
    expr_depth = _require_nonnegative_int(
        "bucket.expr_depth", bucket["expr_depth"]
    )
    expr_len = _require_nonnegative_int("bucket.expr_len", bucket["expr_len"])
    has_division = _require_bool(
        "bucket.has_division", bucket["has_division"]
    )
    has_subtraction = _require_bool(
        "bucket.has_subtraction", bucket["has_subtraction"]
    )
    score = _require_nonnegative_int("bucket.score", bucket["score"])
    complexity = _require_nonempty_string(
        "bucket.complexity", bucket["complexity"]
    )
    if complexity not in COMPLEXITIES:
        raise ValueError("bucket.complexity must be easy, medium, or hard")
    bucket_key = _require_nonempty_string(
        "bucket.bucket_key", bucket["bucket_key"]
    )
    if num_count != number_count:
        raise ValueError(
            "bucket does not match canonical bucket: num_count"
        )
    expected_key = f"{num_count}_{complexity}"
    if bucket_key != expected_key:
        raise ValueError(
            f"bucket does not match canonical bucket: bucket_key must be {expected_key}"
        )
    return {
        "num_count": num_count,
        "expr_depth": expr_depth,
        "expr_len": expr_len,
        "has_division": has_division,
        "has_subtraction": has_subtraction,
        "score": score,
        "complexity": complexity,
        "bucket_key": bucket_key,
    }


def _validate_validation(
    value: Any,
    *,
    response: str,
    numbers: list[int],
    target: int,
) -> dict[str, Any]:
    validation = _require_mapping("validation", value)
    _require_exact_keys("validation", validation, VALIDATION_KEYS)
    ok = _require_bool("validation.ok", validation["ok"])
    fraction_text = _require_fraction_or_none(
        "validation.value", validation["value"]
    )
    used_numbers = _require_exact_int_list(
        "validation.used_numbers",
        validation["used_numbers"],
        nonnegative=True,
    )
    expression = validation["expression"]
    if expression is not None and not isinstance(expression, str):
        raise ValueError("validation.expression must be a string or null")
    error = validation["error"]
    if error is not None:
        if not isinstance(error, str) or error not in VALIDATION_ERRORS:
            raise ValueError(
                "validation.error must be null or a canonical validation error"
            )

    actual = validate_countdown_response(response, numbers, target)
    expected_fields = {
        "ok": actual.ok,
        "value": serialize_fraction(actual.value),
        "used_numbers": actual.used_numbers,
        "expression": actual.expression,
        "error": actual.error,
    }
    declared_fields = {
        "ok": ok,
        "value": fraction_text,
        "used_numbers": used_numbers,
        "expression": expression,
        "error": error,
    }
    for field, expected in expected_fields.items():
        if declared_fields[field] != expected:
            raise ValueError(
                f"validation.{field} does not match actual response result"
            )

    return {
        "ok": ok,
        "value": fraction_text,
        "used_numbers": used_numbers,
        "expression": expression,
        "error": error,
    }


def _validate_chat_prompt(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("prompt must be a nonempty chat message list")
    messages: list[dict[str, str]] = []
    for index, item in enumerate(value):
        message = _require_mapping(f"prompt[{index}]", item)
        _require_exact_keys(f"prompt[{index}]", message, CHAT_MESSAGE_KEYS)
        messages.append(
            {
                "role": _require_nonempty_string(
                    f"prompt[{index}].role", message["role"]
                ),
                "content": _require_nonempty_string(
                    f"prompt[{index}].content", message["content"]
                ),
            }
        )
    return messages


def _validate_reward_model(value: Any) -> dict[str, Any]:
    reward_model = _require_mapping("reward_model", value)
    _require_exact_keys("reward_model", reward_model, REWARD_MODEL_KEYS)
    style = _require_nonempty_string("reward_model.style", reward_model["style"])
    ground_truth = _require_mapping(
        "reward_model.ground_truth", reward_model["ground_truth"]
    )
    _require_exact_keys(
        "reward_model.ground_truth", ground_truth, GROUND_TRUTH_KEYS
    )
    numbers = _require_arrow_numbers(
        "reward_model.ground_truth.numbers", ground_truth["numbers"]
    )
    target = _require_arrow_nonnegative_int(
        "reward_model.ground_truth.target", ground_truth["target"]
    )
    return {
        "style": style,
        "ground_truth": {"numbers": numbers, "target": target},
    }


def _normalize_json_mapping(name: str, value: Any) -> dict[str, Any]:
    mapping = _require_mapping(name, value)
    normalized = _normalize_json_value(name, mapping)
    assert isinstance(normalized, dict)
    return normalized


def _normalize_arrow_mapping(name: str, value: Any) -> dict[str, Any]:
    mapping = _require_mapping(name, value)
    normalized, _ = _normalize_arrow_value(name, mapping)
    assert isinstance(normalized, dict)
    return normalized


def _normalize_arrow_value(path: str, value: Any) -> tuple[Any, tuple[Any, ...]]:
    if value is None:
        return None, ("null",)
    if type(value) is bool:
        return value, ("bool",)
    if type(value) is int:
        if not -(2**63) <= value < 2**63:
            raise ValueError(f"{path} contains an integer outside Arrow int64")
        return value, ("int",)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value, ("float",)
    if type(value) is str:
        return value, ("string",)
    if isinstance(value, list):
        normalized_items: list[Any] = []
        item_type: tuple[Any, ...] = ("null",)
        for index, item in enumerate(value):
            normalized_item, candidate_type = _normalize_arrow_value(
                f"{path}[{index}]", item
            )
            item_type = _merge_arrow_types(path, item_type, candidate_type)
            normalized_items.append(normalized_item)
        return normalized_items, ("list", item_type)
    if isinstance(value, Mapping):
        normalized_mapping: dict[str, Any] = {}
        fields: list[tuple[str, tuple[Any, ...]]] = []
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{path} contains a non-string mapping key")
        for key in sorted(value):
            normalized_item, item_type = _normalize_arrow_value(
                f"{path}.{key}", value[key]
            )
            normalized_mapping[key] = normalized_item
            fields.append((key, item_type))
        return normalized_mapping, ("mapping", tuple(fields))
    raise ValueError(f"{path} contains a non-Arrow-compatible value")


def _merge_arrow_types(
    path: str,
    left: tuple[Any, ...],
    right: tuple[Any, ...],
) -> tuple[Any, ...]:
    if left == ("null",):
        return right
    if right == ("null",):
        return left
    if left == right:
        return left
    if {left, right} == {("int",), ("float",)}:
        return ("float",)
    if left[0] == right[0] == "list":
        return ("list", _merge_arrow_types(path, left[1], right[1]))
    if left[0] == right[0] == "mapping":
        left_fields = dict(left[1])
        right_fields = dict(right[1])
        field_names = sorted(left_fields.keys() | right_fields.keys())
        return (
            "mapping",
            tuple(
                (
                    key,
                    _merge_arrow_types(
                        f"{path}.{key}",
                        left_fields.get(key, ("null",)),
                        right_fields.get(key, ("null",)),
                    ),
                )
                for key in field_names
            ),
        )
    raise ValueError(f"{path} list elements do not share an Arrow type")


def _normalize_json_value(path: str, value: Any) -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, list):
        return [
            _normalize_json_value(f"{path}[{index}]", item)
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string mapping key")
            normalized[key] = _normalize_json_value(f"{path}.{key}", item)
        return normalized
    raise ValueError(f"{path} contains a non-JSON-compatible value")


def _require_fraction_or_none(name: str, value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or FRACTION_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a reduced n/d string or null")
    numerator_text, denominator_text = value.split("/", 1)
    numerator = int(numerator_text)
    denominator = int(denominator_text)
    if (
        numerator_text != str(numerator)
        or denominator_text != str(denominator)
        or gcd(abs(numerator), denominator) != 1
    ):
        raise ValueError(f"{name} must be a reduced canonical n/d string")
    return value


def _require_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_exact_keys(
    name: str,
    value: Mapping[str, Any],
    expected: set[str],
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError(f"{name} keys mismatch: {', '.join(details)}")


def _require_nonempty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_bool(name: str, value: Any) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _require_nonnegative_int(name: str, value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative exact integer")
    return value


def _require_exact_int_list(
    name: str,
    value: Any,
    *,
    nonnegative: bool,
) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of exact integers")
    result: list[int] = []
    for item in value:
        if type(item) is not int or (nonnegative and item < 0):
            qualifier = "nonnegative " if nonnegative else ""
            raise ValueError(f"{name} must contain {qualifier}exact integers")
        result.append(item)
    return result


def _require_numbers(name: str, value: Any) -> list[int]:
    result = _require_exact_int_list(name, value, nonnegative=True)
    if not result:
        raise ValueError(f"{name} must be a nonempty list")
    return result


def _require_arrow_nonnegative_int(name: str, value: Any) -> int:
    result = _require_nonnegative_int(name, value)
    if result >= 2**63:
        raise ValueError(f"{name} must fit Arrow signed int64")
    return result


def _require_arrow_numbers(name: str, value: Any) -> list[int]:
    result = _require_numbers(name, value)
    for item in result:
        if item >= 2**63:
            raise ValueError(f"{name} must contain Arrow signed int64 values")
    return result
