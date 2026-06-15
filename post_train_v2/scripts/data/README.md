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

Unsolvable train rows are retained in `unsolved_train.jsonl` with
`reason: "no_solution"`. Every test row must be solvable; an unsolvable test
row fails the build before any new data files or completion manifest are
published. The four JSONL files use atomic replacement, and `manifest.json` is
published last as the completion marker.
