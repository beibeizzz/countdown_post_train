"""Exact Fraction-based Countdown solver."""

from __future__ import annotations

import ast
import functools
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class _ExprNode:
    value: Fraction
    expression: str


def solve_countdown(numbers: list[int], target: int) -> str | None:
    _require_non_negative_int(target, "target")
    for index, number in enumerate(numbers):
        _require_non_negative_int(number, f"numbers[{index}]")

    normalized_numbers = tuple(sorted(numbers))
    return _solve_countdown_cached(normalized_numbers, target)


@functools.lru_cache(maxsize=200_000)
def _solve_countdown_cached(
    numbers: tuple[int, ...],
    target: int,
) -> str | None:
    nodes = tuple(_ExprNode(Fraction(number), str(number)) for number in numbers)
    return _search(nodes, Fraction(target), set())


def expression_metadata(expr: str, num_count: int) -> dict[str, int | bool]:
    try:
        tree = ast.parse(expr, mode="eval")
        expr_depth = _depth(tree.body)
    except (SyntaxError, RecursionError):
        expr_depth = 0

    return {
        "num_count": num_count,
        "expr_depth": expr_depth,
        "expr_len": len(expr),
        "has_division": "/" in expr,
        "has_subtraction": "-" in expr,
    }


def _search(
    nodes: tuple[_ExprNode, ...],
    target: Fraction,
    failed_states: set[tuple[Fraction, ...]],
) -> str | None:
    if len(nodes) == 1:
        return nodes[0].expression if nodes[0].value == target else None

    state = tuple(sorted(node.value for node in nodes))
    if state in failed_states:
        return None

    for left_index in range(len(nodes)):
        for right_index in range(left_index + 1, len(nodes)):
            left = nodes[left_index]
            right = nodes[right_index]
            remaining = tuple(
                node
                for index, node in enumerate(nodes)
                if index not in (left_index, right_index)
            )
            for candidate in _combine(left, right):
                result = _search(remaining + (candidate,), target, failed_states)
                if result is not None:
                    return result

    failed_states.add(state)
    return None


def _combine(left: _ExprNode, right: _ExprNode) -> tuple[_ExprNode, ...]:
    candidates = [
        _ExprNode(
            left.value + right.value,
            f"({left.expression}+{right.expression})",
        ),
        _ExprNode(
            left.value - right.value,
            f"({left.expression}-{right.expression})",
        ),
        _ExprNode(
            right.value - left.value,
            f"({right.expression}-{left.expression})",
        ),
        _ExprNode(
            left.value * right.value,
            f"({left.expression}*{right.expression})",
        ),
    ]
    if right.value != 0:
        candidates.append(
            _ExprNode(
                left.value / right.value,
                f"({left.expression}/{right.expression})",
            )
        )
    if left.value != 0:
        candidates.append(
            _ExprNode(
                right.value / left.value,
                f"({right.expression}/{left.expression})",
            )
        )
    return tuple(candidates)


def _depth(node: ast.AST) -> int:
    if isinstance(node, ast.Constant):
        return 1
    if isinstance(node, ast.BinOp):
        return 1 + max(_depth(node.left), _depth(node.right))
    return 0


def _require_non_negative_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
