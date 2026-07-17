# Infrastructure

Hardware specs for the machines this project runs on. Captured here (not just in
per-machine Claude memory) because Claude Code sessions are **not synced across
machines** — a session on the laptop and a session on Anzu each have their own
local memory and no visibility into the other. Anything that needs to be known
regardless of which machine a session starts on belongs in this git-tracked repo,
not in local memory.

See `CLAUDE.md` → Infrastructure table for the one-line role summary of each
machine; this doc has the actual hardware numbers.

## Anzu (BioDataSciLab GPU server)

Shared multi-user server — not dedicated only to this project. Specs as of 2026-07-17:

- **GPUs (7 total, heterogeneous):**
  - 3× NVIDIA RTX 5000 Ada Generation — 32760 MiB (~32GB) each
  - 3× NVIDIA RTX A5000 — 24564 MiB (~24GB) each
  - 1× NVIDIA RTX A6000 — 49140 MiB (~48GB)
  - Driver 570.86.10, CUDA 12.8
- **CPU/RAM:** 64 cores (`nproc`), 251GiB total RAM, 12GiB swap
- **OS:** Ubuntu 20.04.6 LTS, kernel 5.4.0-216-generic
- **`SOURCE_DIR`:** `/media/ubuntu/8TB/ufuk/protgen-gan/models` (see `config.py`)

**Shared server caveat:** Anzu is a shared server, so GPU availability is not guaranteed. If you need to run a large job, check the current GPU usage with `nvidia-smi`. Try to be considerate of other users and avoid hogging resources. Do not use more than one GPU at a time unless I give you the explicit permission to do so.

**VRAM ceiling caveat:** the current GAN architecture sometimes exceeds 48GB VRAM
during training. The A6000 (49GB) is the *only* card on Anzu that fits it, and
it's one of seven cards on a shared box — availability isn't guaranteed. Don't
assume the full 48GB ceiling is available; a safer default is to assume 24-32GB
unless the A6000 is confirmed free (`nvidia-smi`). This is why training on Anzu
runs in smaller/controlled batches, with full-scale runs reserved for MN5's H100s.

If numbers here look stale, re-run:
```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nproc && free -h
```

## MN5 (BSC supercomputer)

No internet access — Claude Code cannot run there at all. If MN5 commands come
up in conversation, the user is running them manually; file transfer only via
scp (upload) / sftp (download). See `CLAUDE.md` → Infrastructure and Git Workflow
sections for the two-remote git setup used to sync code to/from MN5.

## Zenbook (local laptop)

No external GPU — training is impossible here. Used for dev, edits, review, and
small local checks only.
