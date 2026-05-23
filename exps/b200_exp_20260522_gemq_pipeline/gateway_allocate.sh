#!/bin/bash
# Runs ON gateway. After gateway_pull.sh has staged cache/<MODEL>/LayerRE_*.pkl,
# this script runs:
#   1) the GEMQ global ILP at three effective bpe targets
#   2) the per-block ILP swept over tb ∈ [tb_min, tb_max] at step 0.125 (effective bits)
#
# Positional args:
#   $1  SHORT_NAME ∈ {mixtral8x7b, deepseekv2lite, qwen15moe}
#   $2  tb_min   (default depends on QUANT_SCHEME)
#   $3  tb_max   (default depends on QUANT_SCHEME)
#   $4  tb_step  (default 0.125)
#
# Env:
#   QUANT_SCHEME=gemq  (default) — wbits=1,2,3; 1-bit symmetric `binary`
#                                  effective bit_cost = {1:1.125, 2:2.25, 3:3.25}
#                                  default tb sweep [1.125, 3.250]
#   QUANT_SCHEME=mxmoe            — wbits=1,2,3,4; ALL asymmetric per-group scale+zero
#                                  effective bit_cost = {1:1.25, 2:2.25, 3:3.25, 4:4.25}
#                                  default tb sweep [1.250, 4.250]
#                                  REQUIRES the matching `QUANT_SCHEME=mxmoe`
#                                  b200_compute_stats.sh run (LayerRE_*_asym1_*.pkl).
#   SKIP_GLOBAL=1                 — skip the 3 global runs (e.g. Gurobi license issues
#                                  on DeepSeek/Qwen).
#
# Outputs:
#   configs/<MODEL>/GEMQ/C4-Seed0_Eeff{T}_B{cands}_c2c3.pkl                (3 files)
#   configs/<MODEL>/PerBlockEff/C4-Seed0_tb{lo}-{hi}-{step}_B{cands}_c2c3.pkl (1 aggregate)

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: QUANT_SCHEME={gemq|mxmoe} $0 <short_name> [tb_min tb_max [tb_step]]"
    exit 1
fi
SHORT="$1"

case "$SHORT" in
    mixtral8x7b)    MODEL="mistralai/Mixtral-8x7B-v0.1" ;;
    deepseekv2lite) MODEL="deepseek-ai/DeepSeek-V2-Lite" ;;
    qwen15moe)      MODEL="Qwen/Qwen1.5-MoE-A2.7B" ;;
    *) echo "ERROR: unknown short name $SHORT"; exit 1 ;;
esac

cd /data_fast/home/deokjae/QUANT_works/GEMQ

QUANT_SCHEME="${QUANT_SCHEME:-gemq}"
case "$QUANT_SCHEME" in
    gemq)
        BIT_CANDS="1,2,3"
        BIT_COST=""                            # auto-derived: {1:1.125, 2:2.25, 3:3.25}
        STATS_TAG=""
        CALIB_DATASET=c4
        NSAMPLES=128
        SEQLEN=2048
        TB_MIN_DEFAULT=1.125
        TB_MAX_DEFAULT=3.250
        # Global ILP targets (effective bpe). 9 values, 0.125 step in [1.5, 2.5].
        GLOBAL_BPE="1.5 1.625 1.75 1.875 2.0 2.125 2.25 2.375 2.5"
        ;;
    mxmoe)
        BIT_CANDS="1,2,3,4"
        BIT_COST="1:1.25,2:2.25,3:3.25,4:4.25"  # uniform +0.25 overhead, asym throughout
        STATS_TAG="_asym1"
        CALIB_DATASET=wikitext2
        NSAMPLES=128
        SEQLEN=4096
        TB_MIN_DEFAULT=1.250
        TB_MAX_DEFAULT=4.250
        # Global ILP targets (effective bpe). 3 values: 2-bit / 2.5-bit / 3-bit centroids.
        GLOBAL_BPE="2.25 2.75 3.25"
        ;;
    *) echo "ERROR: unknown QUANT_SCHEME=$QUANT_SCHEME"; exit 1 ;;
esac

TB_MIN="${2:-$TB_MIN_DEFAULT}"
TB_MAX="${3:-$TB_MAX_DEFAULT}"
TB_STEP="${4:-0.125}"

LAYER_RE="cache/${MODEL}/LayerRE_${CALIB_DATASET}-N${NSAMPLES}-L${SEQLEN}-Seed0_B${BIT_CANDS}${STATS_TAG}_faster.pkl"
test -f "$LAYER_RE" || { echo "ERROR: missing $LAYER_RE — run gateway_pull.sh first"; exit 1; }

EXTRA_CONSTR=c2c3

LOG_DIR=exps/b200_exp_20260522_gemq_pipeline/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/allocate_${SHORT}_${QUANT_SCHEME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== model: $MODEL  scheme: $QUANT_SCHEME  layer_re: $LAYER_RE ==="
echo "=== bit_cands: $BIT_CANDS  bit_cost: ${BIT_COST:-auto} ==="
echo "=== global bpe targets (effective): $GLOBAL_BPE ==="
echo "=== per-block tb sweep: [$TB_MIN, $TB_MAX] step=$TB_STEP (effective bits) ==="

# Common allocator args
COMMON_ARGS=(
    --model_name "$MODEL"
    --layer_re_path "$LAYER_RE"
    --budget_kind effective
    --bit_candidates "$BIT_CANDS"
    --extra_constr "$EXTRA_CONSTR"
    --ilp_solver gemq
)
if [[ -n "$BIT_COST" ]]; then
    COMMON_ARGS+=(--bit_cost "$BIT_COST")
fi

# Step 1 — global ILP (per-scheme BPE list set above).
# `|| true` so an infeasible bpe (e.g. tight c2c3 at low bpe for small-expert
# models like Mixtral) doesn't abort the rest of the sweep under `set -e`.
if [[ "${SKIP_GLOBAL:-0}" != "1" ]]; then
    for BPE in $GLOBAL_BPE; do
        echo
        echo "[global] target effective bpe = $BPE"
        .venv/bin/python -m gemq.allocate_bits \
            "${COMMON_ARGS[@]}" \
            --mode global --bit_budget "$BPE" || true
    done
fi

# Step 2 — per-block sweep
echo
echo "[per_block] sweeping tb=$TB_MIN..$TB_MAX step=$TB_STEP"
.venv/bin/python -m gemq.allocate_bits \
    "${COMMON_ARGS[@]}" \
    --mode per_block \
    --tb_min "$TB_MIN" --tb_max "$TB_MAX" --tb_step "$TB_STEP"

echo
echo "=== outputs ==="
find "configs/${MODEL}" -type f | sort
