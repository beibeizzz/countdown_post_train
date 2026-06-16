"""Generation metadata used by DPO and rollout diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class GenerationRecord:
    text: str
    finish_reason: str | None
    token_count: int | None
    stop_reason: str | None
    truncated: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        if self.finish_reason is not None and not isinstance(
            self.finish_reason,
            str,
        ):
            raise ValueError("finish_reason must be a string or None")
        if self.token_count is not None and (
            type(self.token_count) is not int or self.token_count < 0
        ):
            raise ValueError("token_count must be a nonnegative integer or None")
        if self.stop_reason is not None and not isinstance(self.stop_reason, str):
            raise ValueError("stop_reason must be a string or None")
        if type(self.truncated) is not bool:
            raise ValueError("truncated must be a boolean")


def generation_record_from_mapping(value: dict[str, object]) -> GenerationRecord:
    return GenerationRecord(
        text=str(value.get("text", "")),
        finish_reason=_optional_string(value.get("finish_reason")),
        token_count=_optional_int(value.get("token_count")),
        stop_reason=_optional_string(value.get("stop_reason")),
    )


def classify_truncation(
    record: GenerationRecord,
    *,
    max_new_tokens: int,
) -> GenerationRecord:
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    finish_reason = (
        record.finish_reason.casefold() if record.finish_reason is not None else None
    )
    if finish_reason in {"length", "max_tokens"}:
        return replace(record, truncated=True)
    if finish_reason is not None:
        return replace(record, truncated=False)
    return replace(
        record,
        truncated=(
            record.token_count is not None and record.token_count >= max_new_tokens
        ),
    )


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise ValueError("token_count must be an integer or None")
    return value
