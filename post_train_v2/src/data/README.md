# V2 Dataset Schemas

`schema.py` is the strict writer-side boundary for canonical V2 dataset
records. Every public validator returns a newly built, deeply unaliased
record and never mutates the supplied mapping.

## Normalized Source

Normalized source records use exactly these fields:

```text
id, source_index, numbers, target, gold_expr, prompt, bucket
```

`id`, `gold_expr`, and `prompt` are nonempty strings. `source_index`,
`target`, and every member of the nonempty `numbers` list are nonnegative
exact integers; booleans and floats are rejected.

`bucket` uses exactly:

```text
num_count, expr_depth, expr_len, has_division, has_subtraction,
score, complexity, bucket_key
```

The integer metadata and score are nonnegative exact integers, the two
feature flags are exact booleans, and complexity is `easy`, `medium`, or
`hard`. `num_count` must equal `len(numbers)` and `bucket_key` must equal
`<num_count>_<complexity>`. The validator also checks that `gold_expr` is a
correct exact solution for `numbers` and `target`, recomputes the bucket
with the V2 `assign_bucket` implementation, and requires every bucket field
to match that canonical result.

## SFT and RFT

SFT/RFT records contain the complete normalized source plus exactly:

```text
response, validation, provenance
```

`response` is the complete nonempty model response. `validation` contains
exactly `ok`, `value`, `used_numbers`, `expression`, and `error`.
`value`, when present, is a reduced canonical fraction string such as
`24/1` or `-3/7`. The allowed error vocabulary is
`missing_answer_tag`, `invalid_expression`, `number_mismatch`, and
`wrong_value`. Success and failure fields must agree with each other and
with the source numbers and target.

`provenance` is a recursively validated JSON mapping. It may contain null,
booleans, exact integers, finite floats, strings, lists, and string-keyed
mappings.

## DPO

DPO records use exactly:

```text
prompt, chosen, rejected, rejected_category, generation_route, provenance
```

All string fields are nonempty, chosen and rejected responses must differ,
and rejected categories are limited to `wrong_value`, `number_mismatch`,
`invalid_expression`, `missing_answer_tag`, and `truncated`.
Problem and candidate IDs belong in `provenance`, not at the DPO top level.

## verl

verl records use exactly:

```text
data_source, prompt, ability, reward_model, extra_info
```

`prompt` is a nonempty list of exact `{role, content}` chat-message
mappings. `reward_model` is exactly `{style, ground_truth}`, and
`ground_truth` is exactly `{numbers, target}` with the same Countdown
integer constraints as normalized source data. `extra_info` is restricted
to recursively Arrow-friendly values. Lists may contain nulls, mergeable
integer/float values, one primitive type, recursively compatible lists, or
mappings with the same keys and recursively compatible field types. Mixed
lists such as `[1, "x"]`, incompatible mapping schemas, bytes, tuples,
Fraction objects, NaN, and infinities are rejected. This Arrow restriction
does not apply to ordinary SFT/RFT/DPO provenance, which remains JSON-only.

## Artifact Boundary

Runtime Countdown validation uses `fractions.Fraction` for exact arithmetic.
Dataset artifacts must never contain a `Fraction` object. Writers serialize
it at the artifact boundary with
`post_train_v2.src.countdown.serialize_fraction`, producing a reduced
`n/d` string before calling the schema validator.

The currently implemented Teacher payload is a legacy producer: it includes
`teacher_expr`, has a smaller validation mapping, and does not yet include
canonical provenance. Compatibility readers or later stage adapters may
consume that payload, but canonical V2 writers must transform it into the
schemas above. This module intentionally does not modify Teacher generation.

Use `validate_unique_ids(rows, label)` before publishing a collection. It
rejects missing, invalid, or duplicate IDs and returns deep copies in the
original input order.
