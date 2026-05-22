# b200_exp_20260522_gemq_pipeline

GEMQ bit-allocation pipeline for three MoE LLMs, run on B200 (stats) +
gateway (Gurobi ILP). Reference layout mirrors
[`../../../Mixture-Compressor-MoE/exps/b200_exp_20260522_8x22b_correct`](../../../Mixture-Compressor-MoE/exps/b200_exp_20260522_8x22b_correct).

Targets:

| short name        | HF id                              | layers | routed E | shared E |
|-------------------|-------------------------------------|--------|----------|----------|
| `mixtral8x7b`     | `mistralai/Mixtral-8x7B-v0.1`       | 32     | 8        | 0        |
| `deepseekv2lite`  | `deepseek-ai/DeepSeek-V2-Lite`      | 27     | 64       | 2        |
| `qwen15moe`       | `Qwen/Qwen1.5-MoE-A2.7B`            | 24     | 60       | 1        |

Two kinds of solutions:

1. **Global solution** — the GEMQ ILP (paper Eq. 7) solved over all experts
   in all MoE blocks under a single bit budget. Run at target **effective**
   bits-per-expert ∈ {1.5, 2.0, 2.5} with `bit_candidates ∈ {1,2,3}` and the
   paper's `c2c3` per-layer constraint.
2. **Per-block solution** — the same GEMQ objective restricted to **one
   block at a time**, swept over a target effective bits-per-expert
   `tb ∈ [1.125, 3.25]` at 0.125 step (18 points). Each (block `l`, target
   `tb`) produces an `{expert_idx: bit}` assignment.

"Effective bits" is **per-bit**, not a single scalar offset, because
GEMQ's 1-bit quantizer is **symmetric** (`scales = mean(|x|)*2`, zero is
the constant 0.5, no zero stored) while the 2/3/4-bit paths are
**asymmetric** (both scale and zero per group). At `groupsize=128` with
fp16 scale and (where applicable) fp16 zero, the per-weight effective
cost is:

| bit | scheme       | overhead   | effective |
|-----|--------------|------------|-----------|
| 1   | symmetric    | 16/128 = 0.125 | **1.125** |
| 2   | asymmetric   | 32/128 = 0.25  | **2.25**  |
| 3   | asymmetric   | 32/128 = 0.25  | **3.25**  |
| 4   | asymmetric   | 32/128 = 0.25  | 4.25      |

The allocator embeds this `bit_cost` mapping directly in the LP budget
coefficient (`Σ bit_cost[k]·x_ik ≤ tb·N`); the deployed quantizers
(`gemq/quantizers/rtn.py:binary` and
`gemq/quantizers/gptq.py:find_params` when `max_int == 1`) confirm the
symmetric-1-bit convention.

---

## File layout

```
b200_exp_20260522_gemq_pipeline/
├── README.md                # this file
├── b200_setup.sh            # B200: one-time clone+venv+c4-shard install
├── b200_compute_stats.sh    # B200: per-model layer_grads + layer_re
├── gateway_pull.sh          # gateway: sftp cache/* + logs/* back
├── gateway_allocate.sh      # gateway: global + per-block ILP
├── logs/                    # B200 → gateway landing (chmod 777)
└── results/                 # B200 → gateway landing (chmod 777)
```

Code changes on the badeok0716/GEMQ fork (already committed in this PR-less
push):

- `gemq/utils/model_utils.py` — adds `ModelType.QWEN2MOE` registration,
  `get_model_info`, `get_moe_block`, `get_shared_expert_block`,
  `get_sublinear_names`, `get_module_type`, `get_expert_id`,
  `get_all_expert_names`, and a `compute_gate_stats_hook_qwen2moe` so the
  full `compute_model_stats` path supports Qwen1.5-MoE-A2.7B.
- `gemq/allocate_bits.py` — adds `--mode {global, per_block}`,
  `--budget_kind {raw, effective}`, `--groupsize` (default 128),
  `--bit_cost` (optional explicit override of the per-bit effective
  cost, e.g. `"1:1.125,2:2.25,3:3.25"`; otherwise auto-derived from
  `groupsize` using GEMQ's symmetric-1-bit + asymmetric-multi-bit
  convention) and `--tb_min / --tb_max / --tb_step` for the per-block
  sweep.
- `gemq/allocation/ilp_solvers.py` — adds `GEMQSolver.build_block_ilp` /
  `solve_block` for per-block ILPs (silent, returns
  `({l: {j: bit}}, objective_value)`).

---

## Phase 0 — gateway: commit + push to the fork

```bash
cd /data_fast/home/deokjae/QUANT_works/GEMQ
git status
git add gemq/utils/model_utils.py gemq/allocate_bits.py gemq/allocation/ilp_solvers.py \
        exps/b200_exp_20260522_gemq_pipeline/
git commit -m "Qwen1.5-MoE + per-block ILP + B200 pipeline scaffold"
git push origin HEAD:main          # origin = https://github.com/badeok0716/GEMQ.git
SHA=$(git rev-parse HEAD); echo "$SHA"
```

Save `$SHA` — every B200 wrapper checks the fork out at this exact commit.
Note (CLAUDE.md / project rule): **never `gh pr create`**, always direct
`git push` to the fork.

---

## Phase 1 — B200: one-time setup

Bootstraps `$B200_ROOT/GEMQ` (clone, `uv venv` with Python 3.10, sync deps,
download the c4 train shard).

```bash
B200_ROOT=/NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae
EXP=$B200_ROOT/GEMQ/exps/b200_exp_20260522_gemq_pipeline

# 1a. sftp the wrapper to B200 (the repo isn't there yet, so we land it under hf_cache)
connect_sftp_b200.sh <<EOF
put exps/b200_exp_20260522_gemq_pipeline/b200_setup.sh $B200_ROOT/b200_setup.sh
chmod 755 $B200_ROOT/b200_setup.sh
bye
EOF

# 1b. submit
submit_b200.sh --user "$USER" --ngpus 1 --ncpus 8 \
    --command "bash $B200_ROOT/b200_setup.sh $SHA"

# 1c. watch
get_b200_queue.sh
```

After setup completes (`$B200_ROOT/setup_gemq_*.log`), the same wrapper now
lives at `$B200_ROOT/GEMQ/exps/.../b200_setup.sh` for any reruns.

---

## Phase 2 — B200: compute statistics per model

`gemq.compute_model_stats` runs in two passes per model:

1. `--mode layer_grads` — backprop CE loss through the full model with
   `device_map='auto'` (≈ all 4 H200s share the weights).
2. `--mode layer_re`    — layer-by-layer forward with each expert
   quantized at every candidate bit ∈ {1, 2, 3}; produces the ILP
   coefficient table `coef[layer][expert][bit]`.

```bash
B200_ROOT=/NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae
EXP=$B200_ROOT/GEMQ/exps/b200_exp_20260522_gemq_pipeline

for SHORT in mixtral8x7b deepseekv2lite qwen15moe; do
    submit_b200.sh --user "$USER" --ngpus 4 --ncpus 32 \
        --command "bash $EXP/b200_compute_stats.sh $SHORT $SHA"
done

get_b200_queue.sh
```

Outputs land under
`$B200_ROOT/GEMQ/cache/<HF_id>/{LayerGrads_*.pt, LayerRE_*.pkl}` and the
log under `$EXP/logs/stats_<short>_<ts>.log`. The wrappers idempotently
skip whichever artifact already exists.

Walltime budget (rough, B200 4× H200):
- Mixtral-8x7B: layer_grads ~15 min, layer_re ~25 min (32 layers × 8 experts × 3 bits)
- DeepSeek-V2-Lite: layer_grads ~5 min, layer_re ~30 min (26 MoE blocks × 64 experts × 3 bits)
- Qwen1.5-MoE-A2.7B: layer_grads ~5 min, layer_re ~30 min (24 × 60 × 3)

---

## Phase 3 — gateway: pull cache/ back

```bash
cd /data_fast/home/deokjae/QUANT_works/GEMQ
bash exps/b200_exp_20260522_gemq_pipeline/gateway_pull.sh
```

This is an sftp `get -r GEMQ/cache .` plus a logs pull; safe to rerun.

---

## Phase 4 — gateway: run the ILPs

Solving the ILPs is fast (seconds for the global one, < 1 min for the
per-block sweep) but Gurobi-licensed: DeepSeek-V2-Lite and Qwen1.5-MoE
exceed the free-tier variable limit for the global ILP.

```bash
cd /data_fast/home/deokjae/QUANT_works/GEMQ

# Per model — global ILP @ effective bpe ∈ {1.5, 2.0, 2.5} + per-block sweep
for SHORT in mixtral8x7b deepseekv2lite qwen15moe; do
    bash exps/b200_exp_20260522_gemq_pipeline/gateway_allocate.sh $SHORT
done
```

Outputs:

```
configs/<HF_id>/GEMQ/C4-Seed0_Eeff1.500_B1,2,3_c2c3.pkl
configs/<HF_id>/GEMQ/C4-Seed0_Eeff2.000_B1,2,3_c2c3.pkl
configs/<HF_id>/GEMQ/C4-Seed0_Eeff2.500_B1,2,3_c2c3.pkl
configs/<HF_id>/PerBlockEff/C4-Seed0_tb1.125-3.250-0.125_B1,2,3_c2c3.pkl
```

The per-block file is a single dict with this structure (see
`run_gemq_solver_per_block` in `gemq/allocate_bits.py`):

```python
{
  "solutions":  {l: {tb: {expert: bit, ...}}},
  "objectives": {l: {tb: float}},
  "infeasible": {l: {tb: bool}},   # True when block_budget < min cost (low-tb tail
                                   #   for DeepSeek-V2-Lite is shared-pinning-bound)
  "tb_values":  [1.125, 1.25, ..., 3.25],
  "budget_kind": "effective",
  "bit_cost":   {1: 1.125, 2: 2.25, 3: 3.25},  # per-bit effective storage cost
  "bit_candidates": [1, 2, 3],
  "extra_constr": "c2c3",
  "model_name": "...",
}
```

To recover the deployed average effective bits for layer `l` from the saved
solution:

```python
eff_avg_l = sum(payload["bit_cost"][b] for b in payload["solutions"][l][tb].values()) / N_lp
```

where `N_lp = num_routed + min(1, num_shared)` is the number of LP-internal
experts per block.

Set `SKIP_GLOBAL=1` to skip the global step if you only need the per-block
table (or if Gurobi licensing blocks the global ILP for the big models):

```bash
SKIP_GLOBAL=1 bash exps/b200_exp_20260522_gemq_pipeline/gateway_allocate.sh <SHORT>
```

---

## Phase 5 — optional, downstream

Quantization + PPL/lm-eval is **out of scope** for this exp by design:
the deliverable is the bit-allocation `.pkl` files. To consume them, run
`gemq.quantize` with `--mixed --bit_cfg <path.pkl>` per the upstream
`scripts/quantize_*.sh`. Note `gemq.quantize` currently expects a flat
`{layer: {expert: bit}}` pkl — for per-block configs you'll need to
materialize one full-model dict per `(layer, tb)` selection from
`solutions` (or just feed it the `solutions[l][tb]` for the layer you
care about).

---

## Gotchas (recap)

- **B200 `/tmp` forbidden** (umbrella CLAUDE.md §5). Everything lives under
  `$B200_ROOT/GEMQ/...` or the gateway repo path.
- **`submit_b200.sh --command` is not a real shell** — only `bash <abs
  wrapper>` invocations work; chained `&&` / heredocs / nested quoting
  silently drop the job.
- **Before `git checkout <SHA>`** on B200, the wrapper does
  `git checkout -- uv.lock` to discard the uv-sync-dirtied lock
  (EXECUTION.md §B200 5a).
- **chmod 777** the `logs/` + `results/` directories (sftp wrapper runs as
  user `hosking`).
- **Gurobi license** is required on the gateway for DeepSeek-V2-Lite /
  Qwen1.5-MoE-A2.7B global ILP. The per-block sub-problems fit within the
  free tier (≤ 192 binary vars each).
- **Never PR** to the fork — push directly with
  `git push origin HEAD:main` (project rule, see top-level CLAUDE.md
  memory).
