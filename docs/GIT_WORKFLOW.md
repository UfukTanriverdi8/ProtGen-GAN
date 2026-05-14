# Git Workflow — ProtGen-GAN Multi-Remote Setup

## Overview

This repo uses a two-remote git setup to work around MareNostrum5's (MN5) lack of internet
access. The local machine (or Anzu) acts as a mediator between MN5 and GitHub. Neither MN5
nor the codebase has any special configuration — this is standard git with two named remotes.

```
GitHub (origin)
      ↑↓
Local / Anzu   ←→   MN5 (mn5 remote, no internet)
```

**Anzu** is the Hacettepe BioDataSciLab GPU server, used for mid-scale GPU tasks and as an
internet-connected intermediary. **MN5 (MareNostrum5)** is the BSC supercomputer in Barcelona
used for large-scale training runs, with thousands of H100s but no internet access.

MN5 can talk to local/Anzu over the university network via SSH/SCP/SFTP, but it cannot
reach GitHub directly. All syncing between MN5 and GitHub must be mediated by a machine
with internet access (local or Anzu).

---

## Remote Configuration

From any machine with internet access (local or Anzu), the remotes are configured as:

```bash
git remote -v
# origin    git@github.com:your-username/protgen.git (fetch/push)
# mn5       username@mn5.bsc.es:/path/to/repo/protgen (fetch/push)
```

MN5 itself only has one remote configured — it points back to local or Anzu, not to GitHub.
If you need to verify this while on MN5:

```bash
# On MN5
git remote -v
# local   username@your-local-or-anzu:/path/to/repo/protgen (fetch/push)
```

---

## Core Principle: Local/Anzu as the Post Office

Think of your local machine (or Anzu) as a post office. MN5 drops mail off there, and the
post office forwards it to GitHub. The post office can also send mail to MN5 directly. But
MN5 and GitHub can never write to each other — everything goes through the post office.

This means **all syncing operations happen from local or Anzu**, never from MN5 itself.

---

## Day-to-Day Workflows

### 1. Pushing new code to MN5 (most common)

You develop locally, and you want to get your latest changes onto MN5 for a training run.

```bash
# From local (or Anzu):

# First, sync to GitHub as usual
git push origin main

# Then push the same changes directly to MN5
git push mn5 main
```

Both remotes end up with the same state. Order doesn't matter here.

### 2. Pulling changes made on MN5 back to local

Sometimes you make a quick fix directly on MN5 (a slurm script tweak, a small code patch
during a run). You need to bring those changes back before they get lost or overwritten.

```bash
# On MN5 first — make sure your changes are committed
git add .
git commit -m "fix: tweak slurm memory allocation for mass eval job"

# Then on local (or Anzu):
git fetch mn5                  # download MN5's commits without merging yet
git merge mn5/main             # integrate into your local main branch
git push origin main           # forward to GitHub
```

The `fetch` before `merge` is a good habit — it lets you inspect what changed on MN5
(`git log mn5/main`) before integrating blindly.

### 3. Syncing MN5 after pulling updates from GitHub

You pulled someone else's changes from GitHub (or merged a branch) and now want MN5
to be up to date.

```bash
# From local:
git pull origin main           # get latest from GitHub
git push mn5 main              # forward to MN5
```

### 4. Anzu as the mediator (instead of local)

Anzu has internet access, so it can serve as the mediator just as well as your local machine.
The workflow is identical — just run the same commands from Anzu instead.

---

## Avoiding Merge Conflicts

The most common source of pain is making changes in the same file from two different machines
without syncing in between. The best way to avoid this is to establish a clear division of
responsibility:

**MN5 should only produce outputs, not edit source code.** Training checkpoints, wandb offline
logs, generated sequence files, SLURM stdout/stderr — all of these are outputs that MN5
produces. Source code changes (models.py, training scripts, eval pipeline) should always
originate from local or Anzu, then get pushed to MN5. If you follow this discipline,
MN5 commits become rare and merge conflicts become nearly impossible.

When you do need to make a hotfix directly on MN5 (it happens), **commit it immediately** and
note it somewhere so you remember to fetch it back before your next push from local.

---

## Before Every Push from Local: The Safety Check

If you've been doing work on MN5 in parallel, always run this before pushing from local.
Otherwise you risk overwriting changes that only exist on MN5.

```bash
# From local:
git fetch mn5
git log mn5/main --oneline -10   # inspect what's on MN5
git status                        # check your local state
```

If MN5 is ahead of local, merge first:

```bash
git merge mn5/main
# resolve any conflicts if needed
git push origin main
git push mn5 main
```

---

## wandb Sync Workflow (related, not git)

MN5 has no internet so wandb runs are logged offline. To upload them:

```bash
# On MN5 — find your offline run directory
ls wandb/offline-run-*/

# Download to Anzu via SFTP
sftp username@mn5.bsc.es
> get -r /path/to/wandb/offline-run-TIMESTAMP ./

# On Anzu — sync to wandb cloud
wandb sync ./offline-run-TIMESTAMP
```

This is separate from git but follows the same mediator pattern: MN5 produces the data,
Anzu forwards it to the outside world.

---

## Quick Reference

| Goal | Command (run from local/Anzu) |
|------|-------------------------------|
| Push local changes to GitHub | `git push origin main` |
| Push local changes to MN5 | `git push mn5 main` |
| Fetch MN5 changes | `git fetch mn5` |
| Merge MN5 changes into local | `git merge mn5/main` |
| Sync MN5 with latest GitHub | `git pull origin main && git push mn5 main` |
| Safety check before pushing | `git fetch mn5 && git log mn5/main --oneline -10` |
| Check all remote states | `git fetch --all && git log --oneline --all` |

---

## Notes for Future Agents

If you are an AI agent working in this repo, keep the following in mind:

- **Never assume MN5 and GitHub are in sync.** Always fetch before merging.
- **MN5 commits are rare but possible** -- a quick `git fetch mn5 && git log mn5/main` before
  any rebase or force-push is a safe habit.
- **Do not attempt to push directly from MN5 to GitHub** -- MN5 has no internet access and
  this will fail with a network timeout.
- **The `mn5` remote points to a path over SSH.** If the SSH connection is unavailable (e.g.
  you are running locally without VPN), `git fetch mn5` will fail. This is expected -- just
  work with `origin` only and sync MN5 manually later.
- **Checkpoints, CSVs, and generated sequences are not tracked by git** (see `.gitignore`).
  These are transferred via SCP/SFTP separately, not through git.