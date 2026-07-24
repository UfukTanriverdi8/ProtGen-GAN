# Run & Sweep Documentation Convention

Generic convention for recording individual training runs and sweeps, so results
can be revisited quickly without digging through wandb history. First adopted
for the lambda_kl sweep (`docs/sweeps/lambda-kl-sweep-2026-07.md`), but applies
to every run/sweep going forward.

## Directories

- `docs/runs/` — one file per individual training run.
- `docs/sweeps/` — one file per sweep (a group of runs sharing a common design).

## Filename convention

`<run_name>_<run_id>.md` — the wandb `run_name` (human-readable, set via
`--run_name`) first, the wandb run id (8-character, guarantees uniqueness)
appended after an underscore.

Example: `docs/runs/kl-sweep-p1-kl0.01_a1b2c3d4.md`

## Per-run file (`docs/runs/<run_name>_<run_id>.md`)

```markdown
# <run_name> (<run_id>)

**Date:** YYYY-MM-DD
**Script:** 10p_train.py | fully_masked_train.py
**wandb:** https://wandb.ai/ufuktanriverdi1-hacettepe-university/protgen-gan/runs/<run_id>

## Config

| Param | Value |
|---|---|
| n_critic | ... |
| lr_gen | ... |
| lr_critic | ... |
| lambda_kl | ... |
| temperature | ... |
| n_epochs | ... |
| max_train_seqs | ... |

(any other non-default flags used)

## Final-epoch metrics

| Metric | Value |
|---|---|
| kl_loss | ... |
| gen_grad_norm | ... |
| unique_ratio | ... |
| plddt | ... |
| scAccuracy | ... |
| progres | ... |
| pairwise_tm | ... |

## Per-epoch trend

One line per metric that matters for this run's purpose, e.g.:
- plddt: 0.75 → 0.69 → 0.70 (epochs 1-3)

## Verdict

One paragraph: what this run showed, whether it met its purpose, anything
surprising.
```

## Per-sweep file (`docs/sweeps/<sweep-name>.md`)

```markdown
# <sweep-name>

**Started:** YYYY-MM-DD
**Status:** ...

## Design

Goal/scope, fixed hyperparameters, phases, selection criteria, any prerequisite
changes made for this sweep — written up front, before results land.

---

## Results

## Runs

| Run | lambda_kl (or swept param) | Key metric(s) | Verdict | File |
|---|---|---|---|---|
| kl-sweep-p1-kl0 | 0 | plddt 0.36 (declining) | disqualified | docs/runs/kl-sweep-p1-kl0_xxxx.md |
| ... | ... | ... | ... | ... |

## Winner selection reasoning

Why the chosen candidate(s) were selected over the others, referencing the
design doc's selection criteria.

## Conclusion

What this sweep settled, and what's next.
```
