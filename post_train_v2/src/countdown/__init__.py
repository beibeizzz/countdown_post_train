"""V2 Countdown domain primitives."""

from post_train_v2.src.countdown.bucketing import assign_bucket
from post_train_v2.src.countdown.prompts import (
    build_chat_messages,
    build_dpo_forced_wrong_prompt,
    build_solution_prompt,
)
from post_train_v2.src.countdown.sampling import (
    ValidationSplits,
    build_validation_splits,
    exclude_ids,
    stratified_sample,
)
from post_train_v2.src.countdown.solver import expression_metadata, solve_countdown
from post_train_v2.src.countdown.validation import (
    ValidationResult,
    extract_answer_text,
    has_complete_answer_tag,
    serialize_fraction,
    validate_countdown_expression,
    validate_countdown_response,
)

__all__ = [
    "ValidationResult",
    "ValidationSplits",
    "assign_bucket",
    "build_chat_messages",
    "build_dpo_forced_wrong_prompt",
    "build_solution_prompt",
    "build_validation_splits",
    "exclude_ids",
    "expression_metadata",
    "extract_answer_text",
    "has_complete_answer_tag",
    "serialize_fraction",
    "solve_countdown",
    "stratified_sample",
    "validate_countdown_expression",
    "validate_countdown_response",
]
