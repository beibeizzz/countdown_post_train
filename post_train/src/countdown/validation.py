from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction


ANSWER_RE = re.compile(r".*<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


@dataclass
class ValidationResult:
    ok: bool
    value: int | None
    used_numbers: list[int]
    expression: str | None
    error: str | None


def extract_answer_text(text: str) -> str | None:
    match = ANSWER_RE.search(text)
    if match is None:
        return None
    return match.group(1).strip()


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
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError):
        return ValidationResult(
            ok=False,
            value=None,
            used_numbers=[],
            expression=expression,
            error="invalid_expression",
        )

    int_value = int(value) if value.denominator == 1 else None
    if Counter(used_numbers) != Counter(numbers):
        return ValidationResult(
            ok=False,
            value=int_value,
            used_numbers=used_numbers,
            expression=expression,
            error="number_mismatch",
        )

    if value != Fraction(target):
        return ValidationResult(
            ok=False,
            value=int_value,
            used_numbers=used_numbers,
            expression=expression,
            error="wrong_value",
        )

    return ValidationResult(
        ok=True,
        value=int(target),
        used_numbers=used_numbers,
        expression=expression,
        error=None,
    )


def _eval_node(node: ast.AST) -> tuple[Fraction, list[int]]:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise ValueError("only integer constants are allowed")
        return Fraction(node.value), [node.value]

    if isinstance(node, ast.BinOp):
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

    raise ValueError("unsupported syntax")
