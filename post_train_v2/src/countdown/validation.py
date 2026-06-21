"""Exact validation for Countdown expressions and tagged responses."""

from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction


ANSWER_RE = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    re.IGNORECASE | re.DOTALL,
)
ANSWER_START_RE = re.compile(r"<answer(?=\s|>|$)", re.IGNORECASE)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    value: Fraction | None
    used_numbers: list[int]
    expression: str | None
    error: str | None


def extract_answer_text(text: str) -> str | None:
    starts = list(ANSWER_START_RE.finditer(text))
    if not starts:
        return None

    last_start = starts[-1].start()
    for match in reversed(list(ANSWER_RE.finditer(text))):
        if match.start() == last_start:
            return match.group(1).strip()
    return None


def has_complete_answer_tag(text: str) -> bool:
    return extract_answer_text(text) is not None


def validate_countdown_response(
    text: str,
    numbers: list[int],
    target: int,
) -> ValidationResult:
    expression = extract_answer_text(text)
    if expression is None:
        return ValidationResult(
            ok=False,
            value=None,
            used_numbers=[],
            expression=None,
            error="missing_answer_tag",
        )
    return validate_countdown_expression(expression, numbers, target)


def validate_countdown_expression(
    expr: str,
    numbers: list[int],
    target: int,
) -> ValidationResult:
    expression = expr.strip()
    try:
        tree = ast.parse(expression, mode="eval")
        value, used_numbers = _eval_node(tree.body)
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError, TypeError):
        return ValidationResult(
            ok=False,
            value=None,
            used_numbers=[],
            expression=expression,
            error="invalid_expression",
        )

    if Counter(used_numbers) != Counter(numbers):
        return ValidationResult(
            ok=False,
            value=value,
            used_numbers=used_numbers,
            expression=expression,
            error="number_mismatch",
        )

    if value != Fraction(target):
        return ValidationResult(
            ok=False,
            value=value,
            used_numbers=used_numbers,
            expression=expression,
            error="wrong_value",
        )

    return ValidationResult(
        ok=True,
        value=value,
        used_numbers=used_numbers,
        expression=expression,
        error=None,
    )


def serialize_fraction(value: Fraction | None) -> str | None:
    if value is None:
        return None
    return f"{value.numerator}/{value.denominator}"


def _eval_node(node: ast.AST) -> tuple[Fraction, list[int]]:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise ValueError("only integer constants are allowed")
        return Fraction(node.value), [node.value]

    if not isinstance(node, ast.BinOp):
        raise ValueError("unsupported syntax")

    left_value, left_numbers = _eval_node(node.left)
    right_value, right_numbers = _eval_node(node.right)
    if isinstance(node.op, ast.Add):
        value = left_value + right_value
    elif isinstance(node.op, ast.Sub):
        value = left_value - right_value
    elif isinstance(node.op, ast.Mult):
        value = left_value * right_value
    elif isinstance(node.op, ast.Div):
        value = left_value / right_value
    else:
        raise ValueError("unsupported operator")
    return value, left_numbers + right_numbers
