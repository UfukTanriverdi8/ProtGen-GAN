---
name: bug-fix-checklist
description: Check known unfixed bugs in ProtGen before touching training scripts. Auto-apply when editing 10p_train.py, fully_masked_train.py, or val_metrics.py.
user-invocable: false
---

Before proceeding with any edit to training or evaluation scripts, verify status of all known bugs from CLAUDE.md.

## Known Bugs — Check Each Before Editing

### BUG 2 — pLDDT tuple unpacking (CRITICAL)
- **Files**: `val_metrics.py:207`, `10p_train.py:231`, `fully_masked_train.py:162`
- **Check**: grep for `avg_plddt_score = calculate_plddt_scores` — if not followed by tuple unpack, still broken
- **Fix**: `avg_plddt_score, _ = calculate_plddt_scores_and_save_pdb(...)`

### BUG 3 — Dead temperature in val_metrics (Significant)
- **File**: `val_metrics.py:119-132`
- **Check**: grep for `random_temp` — if `generate(..., temperature=fixed_temp)` still present, broken
- **Fix**: replace `fixed_temp` with `random_temp` in the generate call

### BUG 4 — Attention mask excludes MASK tokens in blind mode (Significant)
- **File**: `fully_masked_train.py:224`
- **Check**: grep for `mask_token_id` in attention mask construction — should be `pad_token_id`
- **Fix**: `updated_attention_mask = (final_input_ids != tokenizer.pad_token_id).long()`

### BUG 5 — NaN guard overwritten (Moderate)
- **File**: `val_metrics.py:200-206`
- **Check**: look for two consecutive avg_plddt_score assignments — second one is NaN-unsafe
- **Fix**: delete the second `if len(plddt_scores) > 0` block

### BUG 6 — debug=True floods SLURM logs (QoL)
- **File**: `fully_masked_train.py:199`
- **Check**: grep for `def generate_fake_batch` — if `debug=True`, broken
- **Fix**: change default to `debug=False`

## Architectural Issue (not a one-line fix)
- Generator never receives adversarial gradient — discrete token IDs break backprop
- Recommended fix: soft embedding pass-through (`softmax(logits) @ embedding_matrix`)
- Do NOT claim GAN is training correctly until this is resolved

## Instructions
1. Run the grep checks above on the files being edited
2. Report which bugs are fixed and which remain
3. If editing a training script, flag any unfixed bugs that affect that script
4. Do not proceed silently — surface all unfixed bugs to the user before making changes
