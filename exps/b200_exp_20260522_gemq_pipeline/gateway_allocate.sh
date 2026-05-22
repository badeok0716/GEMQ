#!/bin/bash
# Runs ON gateway. After gateway_pull.sh has staged cache/<MODEL>/LayerRE_*.pkl,
# this script runs:
#   1) the GEMQ global ILP at effective bpe ∈ {1.5, 2.0, 2.5}
#   2) the per-block ILP swept over tb ∈ [tb_min, tb_max] at step 0.125 (effective bits)
#
# Outputs:
#   configs/<MODEL>/GEMQ/C4-Seed0_Eeff{T}_B1,2,3_c2c3.pkl                (3 files)
#   configs/<MODEL>/PerBlockEff/C4-Seed0_tb{lo}-{hi}-{step}_B1,2,3_c2c3.pkl (1 aggregate file)
#
# Positional args:
#   $1  SHORT_NAME ∈ {mixtral8x7b, deepseekv2lite, qwen15moe}
#   $2  tb_min   (default 1.125 — effective bits/expert at 1-bit symmetric+s, gs=128)
#   $3  tb_max   (default 3.250 — effective bits/expert at 3-bit asym+s+z, gs=128)
#   $4  tb_step  (default 0.125)
#
# Notes:
# - "Effective bit" = raw bit + (16+16)/groupsize for asymmetric (k>=2), or
#   16/groupsize for symmetric (k=1, which GEMQ's binary path uses). Default
#   bit_cost mapping at groupsize=128: {1: 1.125, 2: 2.25, 3: 3.25}.
#   See gemq/quantizers/rtn.py:binary and gemq/quantizers/gptq.py:find_params
#   for the symmetric-1-bit + asymmetric-multi-bit convention.
# - Gurobi: a full license is required for DeepSeek-V2-Lite and Qwen1.5-MoE
#   global ILP (variable count > free limit). Per-block sub-problems stay
#   under the free limit. Skip the global step with `SKIP_GLOBAL=1` if needed.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <short_name> [tb_min tb_max [tb_step]]"; exit 1
fi
SHORT="$1"
TB_MIN="${2:-1.125}"
TB_MAX="${3:-3.250}"
TB_STEP="${4:-0.125}"

case "$SHORT" in
    mixtral8x7b)    MODEL="mistralai/Mixtral-8x7B-v0.1" ;;
    deepseekv2lite) MODEL="deepseek-ai/DeepSeek-V2-Lite" ;;
    qwen15moe)      MODEL="Qwen/Qwen1.5-MoE-A2.7B" ;;
    *) echo "ERROR: unknown short name $SHORT"; exit 1 ;;
esac

cd /data_fast/home/deokjae/QUANT_works/GEMQ

LAYER_RE="cache/${MODEL}/LayerRE_c4-N128-L2048-Seed0_B1,2,3_faster.pkl"
test -f "$LAYER_RE" || { echo "ERROR: missing $LAYER_RE — run gateway_pull.sh first"; exit 1; }

EXTRA_CONSTR=c2c3
BIT_CANDS=1,2,3

LOG_DIR=exps/b200_exp_20260522_gemq_pipeline/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/allocate_${SHORT}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== model: $MODEL  layer_re: $LAYER_RE ==="
echo "=== tb sweep: [$TB_MIN, $TB_MAX] step=$TB_STEP (effective bits) ==="

# Step 1 — global ILP at three target effective bpe values
if [[ "${SKIP_GLOBAL:-0}" != "1" ]]; then
    for BPE in 1.5 2.0 2.5; do
        echo
        echo "[global] target effective bpe = $BPE"
        .venv/bin/python -m gemq.allocate_bits \
            --model_name "$MODEL" \
            --layer_re_path "$LAYER_RE" \
            --mode global --budget_kind effective \
            --bit_budget "$BPE" \
            --bit_candidates "$BIT_CANDS" \
            --extra_constr "$EXTRA_CONSTR" \
            --ilp_solver gemq
    done
fi

# Step 2 — per-block ILP sweep (effective bits)
echo
echo "[per_block] sweeping tb=$TB_MIN..$TB_MAX step=$TB_STEP"
.venv/bin/python -m gemq.allocate_bits \
    --model_name "$MODEL" \
    --layer_re_path "$LAYER_RE" \
    --mode per_block --budget_kind effective \
    --tb_min "$TB_MIN" --tb_max "$TB_MAX" --tb_step "$TB_STEP" \
    --bit_candidates "$BIT_CANDS" \
    --extra_constr "$EXTRA_CONSTR" \
    --ilp_solver gemq

echo
echo "=== outputs ==="
find "configs/${MODEL}" -type f | sort
