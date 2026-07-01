# OPD Scripts

First-version OPD uses TRL's experimental `GKDTrainer`, not a custom top-k
logit cache. It runs on-policy generalized knowledge distillation:

```text
student prompts -> student on-policy generations -> teacher logits on those sequences -> GKD loss
```

Run:

```bash
uv run python post_train/scripts/opd/train_opd_gkd.py --config post_train/configs/opd_gkd.yaml
```

Smoke run:

```bash
uv run python post_train/scripts/opd/train_opd_gkd.py --config post_train/configs/opd_gkd.yaml --max-steps 2
```

The default config sets `lmbda: 1.0` so every batch uses student-generated
rollouts, `beta: 0.0` for the forward-KL side of TRL's generalized JSD, and
`seq_kd: false` so the teacher scores student rollouts rather than generating
its own sequences.
