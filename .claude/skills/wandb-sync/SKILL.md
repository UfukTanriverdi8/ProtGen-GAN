---
name: wandb-sync
description: Guide wandb offline run sync from MN5 to wandb cloud via Anzu. Generates SFTP download commands and wandb sync invocation.
disable-model-invocation: true
---

# wandb Sync (MN5 → Anzu → Cloud)

MN5 has no internet. wandb runs log offline there, then must be synced to cloud from Anzu.

## Step 1 — Identify the run directory on MN5

Ask user for the run directory path on MN5. Typical location:
```
/gpfs/projects/etur29/ufuk/gan/wandb/offline-run-YYYYMMDD_HHMMSS-<run_id>/
```

If user doesn't know exact path, suggest:
```bash
# Run on MN5
ls -lt /gpfs/projects/etur29/ufuk/gan/wandb/ | head -10
```

## Step 2 — Generate SFTP download commands

Generate commands to download from MN5 to Anzu:

```bash
# From Anzu — adjust MN5 login node as needed
sftp ufuk@mn5-login.bsc.es <<'EOF'
cd /gpfs/projects/etur29/ufuk/gan/wandb/
get -r <offline-run-directory>
EOF
```

Or using scp:
```bash
scp -r ufuk@mn5-login.bsc.es:/gpfs/projects/etur29/ufuk/gan/wandb/<offline-run-directory> ~/wandb-sync/
```

## Step 3 — Sync to wandb cloud from Anzu

```bash
# From Anzu, in directory containing the downloaded run
wandb sync <offline-run-directory>
```

If auth needed:
```bash
wandb login
# Then retry sync
```

## Step 4 — Verify

```bash
# Check run appeared in wandb dashboard
# Project: protgen-gan (or whatever project name was used)
wandb runs list --project protgen-gan | head -5
```

## Reminders

- For future MN5 runs, ensure training script sets `WANDB_MODE=offline` or passes `mode="offline"` to `wandb.init()`
- Multiple offline runs can be synced at once — just `wandb sync` each directory
- Downloaded run directories can be deleted from Anzu after successful sync
