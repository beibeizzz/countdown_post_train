# Open Questions

The following decisions remain unresolved. Core training implementation should
wait until they are confirmed.

## Blocking Decisions

1. **Pinned dependency runtime verification**

   The intended package baseline is now recorded in `environment.md`.
   The current vLLM 0.20.1 / PyTorch 2.11.0 pair is not the confirmed verl
   0.7.1 rollout baseline. Confirm whether a separate `post_train_v2` virtual
   environment may be created for vLLM 0.17.0 / PyTorch 2.10.0. Remaining
   confirmation then requires the documented import, NCCL, Qwen3/vLLM, TRL
   DPO, PEFT LoRA, and minimal verl GRPO smoke tests.

   Current status: the inspected process reported zero visible GPUs. Repeat
   `nvidia-smi`, `torch.cuda.device_count()`, `nvcc --version`, and the NCCL
   smoke test inside the actual two-GPU job allocation.

2. **Teacher two-GPU topology**

   Should the first implementation use:

   - one vLLM engine with tensor parallel size 2; or
   - two independent one-GPU vLLM engines with deterministic data shards?

   Recommendation: use two independent engines if Qwen3-8B plus the required
   KV cache and batch size fit reliably on one 40 GB GPU. This should be
   confirmed by a short benchmark.

3. **Canonical distributed launcher**

   Should Trainer stages expose:

   - `torchrun` as the canonical interface; or
   - `accelerate launch` as the canonical interface?

   Recommendation: canonicalize on `torchrun` and use Accelerate only through
   Trainer internals unless cluster integration requires Accelerate config
   files.

4. **Effective batch preservation**

   Should two-GPU DDP preserve the current effective global batch exactly, or
   use the additional memory to increase it?

   This affects learning-rate comparability and baseline interpretation.

5. **GRPO KL behavior**

   Is `KL=0` a permanent project requirement or only the initial experiment?

   If permanently zero, V2 should disable the reference model and avoid its
   memory cost. If KL experiments are expected, the schema should reserve a
   reference-model configuration from the start.

6. **GRPO policy update semantics**

   Should the existing intention of two policy updates per rollout be mapped
   directly to two actor epochs in verl, or should the first verl baseline
   use the official conservative default?

7. **Checkpoint compatibility**

   Which existing artifacts must V2 load:

   - full SFT Hugging Face checkpoints;
   - LoRA adapters;
   - DPO checkpoints;
   - current custom GRPO exports;
   - Trainer continuation checkpoints?

   Loading model weights is simpler than resuming optimizer state across the
   old and new implementations.

8. **CLI compatibility**

   Must V2 preserve current command-line options such as `--config`,
   `--max-steps`, `--limit`, and current repository-relative path semantics?

   Recommendation: preserve common smoke/debug flags, but allow cleaner V2
   config schemas.

## Important Non-Blocking Decisions

9. **Gradient checkpointing**

   Should it remain enabled by default for all stages, or be disabled for
   0.6B SFT/LoRA after memory benchmarking?

10. **DPO reference implementation**

    Confirm the target TRL DPO behavior and whether explicit reference-model
    loading is required for the pinned TRL version.

11. **Evaluation during distributed training**

    Is pausing all ranks while rank 0 generates the fixed 50 samples
    acceptable, or should evaluation run as a separate asynchronous process
    from saved snapshots?

    Recommendation for the first version: synchronous rank-0 evaluation for
    correctness and reproducibility.

12. **Best-checkpoint selection**

    Should the project select a best checkpoint automatically? If yes, which
    metric and tie-break rule should be used:

    - validation accuracy;
    - format-correct accuracy;
    - mean reward;
    - shortest average correct response?

13. **Checkpoint retention**

    Confirm save frequency, maximum retained periodic checkpoints, and whether
    optimizer checkpoints may be deleted after final export.

14. **W&B policy**

    Confirm:

    - project and group naming;
    - online/offline mode;
    - whether generation-only jobs also create W&B runs;
    - whether sample tables should be uploaded or remain local files;
    - artifact upload requirements.

15. **Manifest migration**

    Should V2 preserve the current `countdown.post_train.manifest.v1`
    envelope, or introduce a V2 schema with explicit input/output hashes,
    dependency versions, and distributed shard metadata?

    Recommendation: introduce a versioned V2 manifest while keeping a
    compatibility reader for V1.

16. **Generation ordering**

    Teacher acceptance currently follows original source order. With two
    independent workers, should "first 20k accepted" mean:

    - merged original source order; or
    - completion order across workers?

    Recommendation: merge by original source index so distributed execution
    does not change dataset semantics.

17. **Failure and resume granularity**

    How frequently should teacher/RFT/DPO generation shards flush atomic
    progress: every batch, every fixed number of records, or time-based?

18. **verl dataset ground-truth encoding**

    Confirm whether `reward_model.ground_truth` should be stored as a nested
    struct or a JSON string for the pinned verl/PyArrow versions.

19. **GRPO zero-variance groups**

    Should zero-standard-deviation groups remain in the actor batch with zero
    advantage, or be dropped and dynamically replaced?

    The current implementation keeps them with zero advantage. This can waste
    rollout compute when the model becomes consistently right or wrong.

20. **GRPO evaluation/export cadence**

    Confirm whether fixed evaluation remains every 100 actor optimizer steps
    and how often a Hugging Face actor export should be produced.

21. **Output path compatibility**

    Should V2 write only under `post_train_v2/outputs`, or should it optionally
    read/write existing `post_train/data` and `post_train/outputs` paths?

    Recommendation: read legacy inputs through explicit paths but never write
    into the legacy tree by default.

22. **Remote machine topology**

    Confirm:

    - GPU model and NVLink/PCIe connectivity;
    - local disk capacity and speed;
    - whether both GPUs are visible inside one process namespace/container;
    - whether Ray is permitted;
    - expected job scheduler, if any.
