#!/bin/bash
# Runs ON B200. Computes the two model statistics the GEMQ ILP needs:
#   step1: layer output gradients wrt CE loss   (gemq.compute_model_stats --mode layer_grads)
#   step2: weighted layer reconstruction errors (gemq.compute_model_stats --mode layer_re)
#
# Positional args:
#   $1  SHORT_NAME ∈ {mixtral8x7b, deepseekv2lite, qwen15moe}
#   $2  SHA        (commit SHA of badeok0716/GEMQ to check out before running)
#
# Submit from gateway:
#   B200_ROOT=/NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae
#   EXP=$B200_ROOT/GEMQ/exps/b200_exp_20260522_gemq_pipeline
#   submit_b200.sh --user "$USER" --ngpus 4 --ncpus 32 \
#       --command "bash $EXP/b200_compute_stats.sh mixtral8x7b <SHA>"

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <short_name> <SHA>"; exit 1
fi
SHORT="$1"; SHA="$2"

case "$SHORT" in
    mixtral8x7b)
        MODEL="mistralai/Mixtral-8x7B-v0.1"
        FORWARD_BSZ=1
        ;;
    deepseekv2lite)
        MODEL="deepseek-ai/DeepSeek-V2-Lite"
        FORWARD_BSZ=8   # smaller model — larger forward batch fits
        ;;
    qwen15moe)
        MODEL="Qwen/Qwen1.5-MoE-A2.7B"
        FORWARD_BSZ=8
        ;;
    *) echo "ERROR: unknown short name $SHORT"; exit 1 ;;
esac

B200_ROOT=/NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae
export HF_HOME=$B200_ROOT/hf_cache

REPO=$B200_ROOT/GEMQ
EXP=$REPO/exps/b200_exp_20260522_gemq_pipeline
DATASET=c4
NSAMPLES=128
SEQLEN=2048
SEED=0
WBITS="1,2,3"

LOG="$EXP/logs/stats_${SHORT}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$EXP/logs" "$EXP/results"
exec > >(tee -a "$LOG") 2>&1

echo "=== node: $(hostname) ==="
nvidia-smi -L || true
echo "=== model: $MODEL  short=$SHORT  SHA=$SHA ==="
echo "=== start: $(date -Iseconds) ==="
SECONDS=0

cd "$REPO"
git fetch origin
git checkout -- uv.lock || true
git checkout "$SHA"
echo "=== pinned: $(git rev-parse HEAD) ==="
uv sync

GRADS_PATH="$REPO/cache/${MODEL}/LayerGrads_${DATASET}-N${NSAMPLES}-L${SEQLEN}-Seed${SEED}.pt"
LAYER_RE_PATH="$REPO/cache/${MODEL}/LayerRE_${DATASET}-N${NSAMPLES}-L${SEQLEN}-Seed${SEED}_B${WBITS}_faster.pkl"

mkdir -p "$(dirname "$GRADS_PATH")"
mkdir -p "$(dirname "$LAYER_RE_PATH")"

# Need trust_remote_code only for DeepSeek-V2-Lite official impl.
EXTRA=()
if [[ "$SHORT" == "deepseekv2lite" ]]; then
    EXTRA+=(--attn_impl eager)
fi

echo
echo "[step1] layer_grads (full-precision backward; uses device_map='auto')"
if [[ -f "$GRADS_PATH" ]]; then
    echo "  -> already present, skipping: $GRADS_PATH"
else
    uv run python -m gemq.compute_model_stats \
        --mode layer_grads \
        --model "$MODEL" --model_name "$MODEL" \
        --calib_dataset "$DATASET" --seed "$SEED" \
        --nsamples "$NSAMPLES" --seqlen "$SEQLEN" \
        --layer_grads_path "$GRADS_PATH" \
        "${EXTRA[@]}"
    test -f "$GRADS_PATH" || { echo "ERROR: layer_grads not produced"; exit 1; }
fi

echo
echo "[step2] layer_re (weighted reconstruction errors per expert × bit)"
if [[ -f "$LAYER_RE_PATH" ]]; then
    echo "  -> already present, skipping: $LAYER_RE_PATH"
else
    uv run python -m gemq.compute_model_stats \
        --mode layer_re \
        --model "$MODEL" --model_name "$MODEL" \
        --calib_dataset "$DATASET" --seed "$SEED" \
        --nsamples "$NSAMPLES" --seqlen "$SEQLEN" \
        --wbits "$WBITS" \
        --layer_grads_path "$GRADS_PATH" \
        --layer_re_path "$LAYER_RE_PATH" \
        --forward_batch_size "$FORWARD_BSZ" \
        "${EXTRA[@]}"
    test -f "$LAYER_RE_PATH" || { echo "ERROR: layer_re not produced"; exit 1; }
fi

echo
echo "[summary]"
ls -lh "$GRADS_PATH" "$LAYER_RE_PATH"
echo "=== end: $(date -Iseconds) elapsed=${SECONDS}s ==="
