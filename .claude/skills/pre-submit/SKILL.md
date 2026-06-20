---
name: pre-submit
description: Validate codebase state before submitting a SLURM training job on MN5. Checks unfixed bugs, env vars, wandb config, checkpoint dirs.
disable-model-invocation: true
---

# Pre-Submit Validation

Run all checks below before any SLURM job submission. Report a pass/fail summary table at the end.

## 1. Bug Audit

Grep for each known bug. Report status of each:

### BUG 2 — pLDDT tuple unpacking (CRITICAL)
```bash
grep -n "avg_plddt_score = calculate_plddt_scores" 10p_train.py fully_masked_train.py
```
- FIXED if line shows `avg_plddt_score, _ = calculate_plddt_scores_and_save_pdb(...)`
- BROKEN if no tuple unpack

### BUG 3 — Dead temperature in val_metrics
```bash
grep -n "temperature=fixed_temp\|temperature=random_temp" val_metrics.py
```
- FIXED if `temperature=random_temp`
- BROKEN if `temperature=fixed_temp`

### BUG 4 — Attention mask excludes MASK tokens (blind mode only)
```bash
grep -n "mask_token_id\|pad_token_id" fully_masked_train.py | grep "updated_attention_mask"
```
- FIXED if uses `pad_token_id`
- BROKEN if uses `mask_token_id`

### BUG 5 — NaN guard overwritten
```bash
grep -n "avg_plddt_score" val_metrics.py
```
- FIXED if only one assignment (the NaN-safe one)
- BROKEN if two consecutive assignments exist

### BUG 6 — debug=True default
```bash
grep -n "def generate_fake_batch" fully_masked_train.py
```
- FIXED if `debug=False`
- BROKEN if `debug=True`

## 2. Environment Check

```bash
echo "SOURCE_DIR=$SOURCE_DIR"
echo "WANDB_MODE=$WANDB_MODE"
python3 -c "from config import PROTBERT_PATH, ESMFOLD_PATH, CHECKPOINT_DIR; print(f'PROTBERT: {PROTBERT_PATH}'); print(f'ESMFOLD: {ESMFOLD_PATH}'); print(f'CKPT_DIR: {CHECKPOINT_DIR}')"
```

- Verify SOURCE_DIR is set
- Verify config.py resolves paths without error
- For MN5: verify WANDB_MODE=offline or wandb.init has mode="offline"

## 3. Checkpoint Directory

```bash
ls -la "$SOURCE_DIR/checkpoints/" 2>/dev/null || echo "CHECKPOINT_DIR missing"
```

## 4. Blocking Rules

- **BLOCK submission** if any CRITICAL bug is unfixed (BUG 2)
- **WARN but allow** if non-critical bugs unfixed (BUG 3-6)
- **WARN** if SOURCE_DIR unset (may use MN5 hostname fallback)
- **BLOCK** if config.py import fails

## 5. Summary Table

Output a table:

| Check | Status | Detail |
|-------|--------|--------|
| BUG 2 | ✅/❌ | ... |
| BUG 3 | ✅/❌ | ... |
| ... | ... | ... |
| SOURCE_DIR | ✅/⚠️ | ... |
| config.py | ✅/❌ | ... |
| WANDB_MODE | ✅/⚠️ | ... |

Final verdict: **READY** or **BLOCKED** (with reasons)
