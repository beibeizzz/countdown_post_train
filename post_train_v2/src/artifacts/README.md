# Artifacts

This package owns deterministic artifact hashing, atomic JSON and JSONL
publication, and the validated Manifest V2 envelope shared by V2 pipeline
stages.

Writers serialize UTF-8 canonical JSON, create a temporary file in the same
directory as the destination, flush and `fsync` it, and publish with
`os.replace`. Failed publications remove the temporary file so readers see
either the previous complete artifact or the new complete artifact.

Multi-file stages use `exclusive_output_lock()` for an output-directory lock
covering the complete read, build, and publish transaction. Locks use
exclusive creation and owner tokens, so a process removes only the lock it
created. A lock left by a terminated process requires explicit operator
inspection and removal.

`ManifestV2` records files, parent hashes, the full configuration snapshot,
model identity, seed policy, Git and runtime versions, and stage-specific
details under `stage_metadata`. Its stable artifact ID excludes creation time.
