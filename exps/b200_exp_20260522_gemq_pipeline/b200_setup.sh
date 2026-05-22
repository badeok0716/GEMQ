#!/bin/bash
# Runs ON B200. One-time setup for the GEMQ pipeline:
#   1) clone (or fetch) the badeok0716/GEMQ fork under $B200_ROOT/GEMQ
#   2) check out the SHA passed as $1 (so every later job pins the same code)
#   3) bootstrap a uv-managed Python 3.10 venv with GEMQ dependencies
#   4) ensure the c4 calibration shard is present under data/
#
# Submit from gateway with:
#   B200_ROOT=/NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae
#   EXP=$B200_ROOT/GEMQ/exps/b200_exp_20260522_gemq_pipeline
#   submit_b200.sh --user "$USER" --ngpus 1 --ncpus 8 \
#       --command "bash $EXP/b200_setup.sh <SHA>"
#
# (The submit picks 1 GPU only because setup needs the GPU node's tooling for
#  uv sync; no model load happens.)

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <SHA>  (commit SHA of badeok0716/GEMQ to pin)"; exit 1
fi
SHA="$1"

B200_ROOT=/NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae
export HF_HOME=$B200_ROOT/hf_cache

REPO=$B200_ROOT/GEMQ
EXP_REL=exps/b200_exp_20260522_gemq_pipeline
FORK_URL=https://github.com/badeok0716/GEMQ.git

LOG=$B200_ROOT/setup_gemq_$(date +%Y%m%d_%H%M%S).log
mkdir -p $B200_ROOT
exec > >(tee -a "$LOG") 2>&1

echo "=== node: $(hostname) ==="
nvidia-smi -L || true
echo "=== B200_ROOT: $B200_ROOT ==="
echo "=== target SHA: $SHA ==="

# 1) clone or fetch
if [[ ! -d "$REPO/.git" ]]; then
    echo "[setup] cloning $FORK_URL -> $REPO"
    git clone "$FORK_URL" "$REPO"
else
    echo "[setup] fetching latest from origin in $REPO"
    cd "$REPO"
    git fetch origin
fi

cd "$REPO"
# uv.lock can be (a) tracked-dirty from a previous `uv sync` rewrite, or
# (b) entirely untracked when the prior SHA didn't ship a lockfile. Both
# block `git checkout $SHA` if the new SHA adds/changes uv.lock (EXECUTION.md
# §B200 5a/5b). Both are auto-generated so safe to drop.
git checkout -- uv.lock 2>/dev/null || true
if [ -f uv.lock ] && ! git ls-files --error-unmatch uv.lock >/dev/null 2>&1; then
    rm -f uv.lock
fi
git checkout "$SHA"
echo "[setup] now at $(git rev-parse HEAD)"

# 2) uv venv + sync (Python 3.10, uv-managed; matches MC-MoE)
which uv >/dev/null 2>&1 || { echo "ERROR: uv not on PATH"; exit 1; }
uv python install 3.10
if [[ ! -d .venv ]]; then
    uv venv --python 3.10 --python-preference only-managed
fi
uv sync
echo "[setup] venv ready: $(.venv/bin/python --version)"

# 3) c4 calibration shard (~820 MB uncompressed; required for the layer_re step)
C4_FILE="$REPO/data/c4-train.00000-of-01024.json"
if [[ ! -f "$C4_FILE" ]]; then
    echo "[setup] downloading c4 train shard 0"
    mkdir -p "$REPO/data"
    cd "$REPO/data"
    wget -q https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz
    gunzip -f c4-train.00000-of-01024.json.gz
    ls -lh c4-train.00000-of-01024.json
fi

echo "[setup] DONE — pinned $(git -C "$REPO" rev-parse HEAD)"
