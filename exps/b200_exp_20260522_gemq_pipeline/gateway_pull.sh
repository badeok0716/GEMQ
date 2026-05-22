#!/bin/bash
# Runs ON gateway. sftp the cache/ artefacts (LayerGrads_*.pt + LayerRE_*.pkl)
# and the B200 stats logs back. Layout mirrors upstream GEMQ:
#   cache/<model_name>/LayerGrads_*.pt
#   cache/<model_name>/LayerRE_*.pkl  (one per QUANT_SCHEME — _asym1 tag for mxmoe)

set -euo pipefail

B200_ROOT=/NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae
GW_REPO=/data_fast/home/deokjae/QUANT_works/GEMQ
EXP_REL=exps/b200_exp_20260522_gemq_pipeline

cd "$GW_REPO"
mkdir -p cache "$EXP_REL/logs"
# `hosking` (the sftp wrapper user) must be able to write into the receive dirs.
chmod 777 cache "$EXP_REL" "$EXP_REL/logs" || true

connect_sftp_b200.sh <<EOF
lcd $GW_REPO
get -r GEMQ/cache .
lcd $GW_REPO/$EXP_REL/logs
get -r GEMQ/$EXP_REL/logs/* .
bye
EOF

echo
echo "[pull] cache/ now contains:"
find cache -type f -printf '%p  %s bytes\n' | sort
