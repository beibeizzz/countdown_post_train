# V2 Source Data Builder

`build_source.py` reads raw Countdown train Parquet and test JSON inputs,
normalizes them with the V2 exact solver, validates every solved record, and
publishes four JSONL data files plus `manifest.json`.

Run from any current working directory:

```text
python post_train_v2/scripts/data/build_source.py
python post_train_v2/scripts/data/build_source.py --limit 100
```

`source_all.jsonl` and `solvable_train.jsonl` intentionally contain the same
solved train records in source order. `source_all.jsonl` is the compatibility
name used by earlier workflow descriptions; `solvable_train.jsonl` is the
explicit semantic name used by downstream V2 stages. They are independently
published and independently recorded in the manifest with their actual hash,
byte size, row count, and schema.

The raw compatibility layer accepts either `nums` or `numbers`. Number
collections may be lists, tuples, NumPy arrays, or JSON list strings, but
every member must be an exact nonnegative integer. Booleans, floats, and
numeric strings are rejected rather than coerced.

Unsolvable train rows are retained in `unsolved_train.jsonl` with
`reason: "no_solution"`. Every test row must be solvable; an unsolvable test
row fails the build before any new data files or completion manifest are
published.

Before replacing any JSONL file, a rebuild removes and directory-syncs the
old `manifest.json` completion marker. If any JSONL publish fails, completed
files may remain, but the missing manifest makes that partial output
non-consumable. The four JSONL files use atomic replacement, and a new
`manifest.json` is published last as the completion marker.

Configured paths are resolved to absolute paths only for local I/O. The full
four-field manifest configuration snapshot is canonicalized with the
validated seed and logical train, test, and output paths. Absolute fixture
paths are normalized relative to the config directory, so neither the
manifest config hash nor artifact identity contains the checkout root. Raw
parent IDs depend only on input kind and content SHA-256, so production
relative configurations remain stable across checkout locations.

The builder hashes both raw inputs before reading either dataset. After all
rows are normalized and solved, it hashes both inputs again before removing
an old manifest or writing any output. A changed train or test input aborts
the run while preserving the previous manifest and data files. Parent
artifacts use the fixed initial hashes rather than rereading inputs during
manifest construction.
