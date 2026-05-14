import os
import socket

# Set SOURCE_DIR to your model storage directory before running.
#
#   MN5:  export SOURCE_DIR=/gpfs/projects/etur29/ufuk
#   Anzu: export SOURCE_DIR=/path/to/models
#
# Add this to your SLURM job script or shell config (~/.bashrc).
# If unset, falls back to hostname-based detection for MN5.

def _resolve_main_folder():
    env = os.environ.get("SOURCE_DIR")
    if env:
        return env
    hostname = socket.gethostname()
    if any(hostname.startswith(p) for p in ("gn", "cn", "alogin", "mn5", "bsc")):
        return "/gpfs/projects/etur29/ufuk"
    raise EnvironmentError(
        "SOURCE_DIR is not set and hostname could not be auto-detected.\n"
        "Run: export SOURCE_DIR=/path/to/your/model/storage"
    )

MAIN_FOLDER    = _resolve_main_folder()
PROTBERT_PATH  = os.path.join(MAIN_FOLDER, "dynamic-finetuned-protbert")
PROTBERT_BASE  = os.path.join(MAIN_FOLDER, "protbert-base")
ESMFOLD_PATH   = os.path.join(MAIN_FOLDER, "esmfold")
CHECKPOINT_DIR = os.path.join(MAIN_FOLDER, "gan-checkpoints")
