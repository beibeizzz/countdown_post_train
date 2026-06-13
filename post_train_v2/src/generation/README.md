# Generation

`parallel_vllm.py` owns the two spawned vLLM workers, device and cache
isolation, deterministic contiguous sharding, protocol validation, ordered
merge, timeouts, and worker shutdown.

Workers only initialize a persistent generator and return position-tagged
text. They do not run the Countdown solver and do not read or write output
files. Transactional persistence and resume state belong in
`teacher_state.py`; the coordinator entrypoint owns validation and output
writes.

Local unit tests use injected fake generators, queues, and processes. Real
vLLM initialization and two-GPU behavior are verified only by the remote GPU
smoke gate.
