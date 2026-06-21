# V2 Countdown Domain Core

This package owns framework-neutral Countdown behavior for V2:

- `prompts.py` builds canonical solution, forced-wrong, and chat prompts.
- `validation.py` parses tagged answers and evaluates the restricted expression
  grammar with exact `fractions.Fraction` arithmetic.
- `solver.py` finds exact solutions and emits fully parenthesized `gold_expr`
  values with expression metadata.
- `bucketing.py` assigns stable difficulty buckets from number count and
  expression complexity.
- `sampling.py` provides deterministic balanced sampling, validation-ID
  exclusion, and construction of a random validation set plus a fixed
  evaluation subset.

`Fraction` values are internal domain values. JSON, JSONL, manifests, metrics,
and other artifact boundaries must call `serialize_fraction`; they must not
send `Fraction` objects directly to serializers.
