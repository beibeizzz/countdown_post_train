# Generation

`parallel_vllm.py` owns the two spawned vLLM workers, device and cache
isolation, deterministic contiguous sharding, protocol validation, ordered
merge, timeouts, and worker shutdown.

`vllm_client.py` calls `LLM.chat()` with ordered user conversations,
`enable_thinking=False`, and one seeded `SamplingParams` object per request.
`seeding.py` derives stable request seeds from the global seed, stage, sample
ID, and rollout index. `output_lock.py` provides process-aware exclusive
publication with explicit stale-lock recovery.

Workers only initialize a persistent generator and return position-tagged
text. They do not run the Countdown solver and do not read or write output
files. Transactional persistence and resume state belong in
`teacher_state.py`; the coordinator entrypoint owns validation and output
writes.

Local unit tests use injected fake generators, queues, and processes. Real
vLLM initialization and two-GPU behavior are verified only by the remote GPU
smoke gate.

Run the local orchestration suite with:

```text
python -m pytest -q post_train_v2/tests/generation
```
