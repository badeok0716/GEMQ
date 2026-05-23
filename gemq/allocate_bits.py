import os
import os.path as osp
import argparse
import json
import pickle
import re

from gemq.utils.model_utils import get_model_info
from gemq.allocation.ilp_solvers import GEMQSolver


def auto_parse_filename(layer_re_path):
    calib_str = ""
    if "math+c4" in layer_re_path:
        calib_str = "MATH+C4"
    elif "wikitext2" in layer_re_path:
        calib_str = "WT2"
    elif "c4" in layer_re_path:
        calib_str = "C4"
    elif "math" in layer_re_path:
        calib_str = "MATH"
    else:
        raise ValueError(f"Cannot parse calibration dataset from layer_re_path: {layer_re_path}")

    # extract seed number from layer_re_path
    match = re.search(r'Seed(\d+)', layer_re_path)
    seed_num = match.group(1) if match else "00"
    calib_str += f"-Seed{seed_num}"

    model_str = ""
    if "Uni" in layer_re_path:
        model_str = "_QT"
    elif "QTFT" in layer_re_path:
        model_str = "_QTFT"

    return calib_str, model_str


# ----- per-bit cost handling ---------------------------------------------------

def default_effective_cost(bit_candidates, groupsize=128, scale_dtype_bits=16, zero_dtype_bits=16):
    """
    Effective per-weight storage cost for the GEMQ / MC-MoE quantization scheme.
    Convention (matches `gemq/quantizers/rtn.py:binary` and
    `gemq/quantizers/gptq.py:find_params` when `max_int == 1`):
      - 1-bit quantizer is **symmetric** — only `scale` per group, no zero.
        ⇒ overhead = scale_bits / groupsize
      - k-bit quantizer for k >= 2 is **asymmetric** — both scale and zero per group.
        ⇒ overhead = (scale_bits + zero_bits) / groupsize
    Effective bit per weight = k + overhead.
    For default fp16 scale+zero and groupsize 128: 1-bit → 1.125, 2-bit → 2.25,
    3-bit → 3.25, 4-bit → 4.25, ...
    """
    cost = {}
    for k in bit_candidates:
        if k <= 0:
            cost[k] = float(k)
            continue
        if k == 1:
            overhead = scale_dtype_bits / groupsize  # symmetric: scale only
        else:
            overhead = (scale_dtype_bits + zero_dtype_bits) / groupsize  # asymmetric
        cost[k] = float(k) + overhead
    return cost


def parse_bit_cost_string(s, bit_candidates):
    """
    Parse "1:1.125,2:2.25,3:3.25" into a {1: 1.125, ...} dict.
    Each value is the FULL per-weight cost (bits + overhead), not just the offset.
    """
    out = {}
    for piece in s.split(","):
        piece = piece.strip()
        if not piece:
            continue
        k_str, v_str = piece.split(":")
        out[int(k_str.strip())] = float(v_str.strip())
    missing = [k for k in bit_candidates if k not in out]
    if missing:
        raise ValueError(f"--bit_cost is missing keys for bit candidates: {missing}")
    return out


def resolve_bit_cost(args, bit_candidates):
    """
    Decide the per-bit cost dictionary that the LP uses for both the budget
    constraint and the (effective-mode) budget computation.
    """
    if args.budget_kind == "raw":
        return {k: float(k) for k in bit_candidates}
    # effective:
    if args.bit_cost:
        return parse_bit_cost_string(args.bit_cost, bit_candidates)
    return default_effective_cost(bit_candidates, groupsize=args.groupsize)


# ----- global / per-block budget computations ----------------------------------

def compute_global_budget(model_info, bpe, max_bit_cost):
    """
    Total budget for the global ILP, in the same units as the bit_cost passed to
    the solver. Mirrors the upstream convention: (num_shared - 1) shared experts
    are *pinned* at max_bit and removed from the LP — they contribute
    `(num_shared - 1) * max_bit_cost` outside the LP.
    """
    bpl = (
        bpe * (model_info.num_routed_experts_per_layer + model_info.num_shared_experts_per_layer)
        - max(0, model_info.num_shared_experts_per_layer - 1) * max_bit_cost
    )
    total_bits = bpl * (model_info.num_layers - model_info.first_k_dense_layers)
    return total_bits, bpl


def compute_per_block_budget(model_info, tb, max_bit_cost):
    """
    Per-block budget at target bits-per-expert = `tb`. Units match `tb`'s units
    (raw vs effective is decided by the caller).
    """
    bpl = (
        tb * (model_info.num_routed_experts_per_layer + model_info.num_shared_experts_per_layer)
        - max(0, model_info.num_shared_experts_per_layer - 1) * max_bit_cost
    )
    return bpl


# ----- solvers -----------------------------------------------------------------

def run_gemq_solver_global(args):
    m = get_model_info(args.model_name)
    bit_cands = list(map(int, args.bit_candidates.split(",")))
    bit_cost = resolve_bit_cost(args, bit_cands)
    max_bit = max(bit_cands)
    max_bit_cost = bit_cost[max_bit]

    total_bits, bpl = compute_global_budget(m, args.bit_budget, max_bit_cost)
    print(f"[global] bpe={args.bit_budget:.4f} ({args.budget_kind}) "
          f"bit_cost={bit_cost} bpl={bpl:.4f} total={total_bits:.2f}")

    solver = GEMQSolver(
        layer_re_path=args.layer_re_path,
        x_space=bit_cands,
        extra_constr=args.extra_constr,
        start_layer_idx=m.first_k_dense_layers,
        bit_cost=bit_cost,
    )
    opt_set = solver.solve_all(total_bits=total_bits)
    if opt_set is None:
        print(f"[global] Skipping save: bpe={args.bit_budget} ({args.budget_kind}) infeasible "
              f"or unsolved. Try a larger budget or drop --extra_constr.")
        return

    save_path = args.save_path
    if not save_path:
        bc_str = ",".join(map(str, bit_cands))
        calib_str, model_str = auto_parse_filename(args.layer_re_path)
        const_str = "" if args.extra_constr == "none" else f"_{args.extra_constr}"
        tag = "E" if args.budget_kind == "raw" else "Eeff"
        save_path = (
            f"configs/{args.model_name}/GEMQ/{calib_str}_{tag}{args.bit_budget:.3f}_B{bc_str}{const_str}{model_str}.pkl"
        )

    os.makedirs(osp.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(opt_set, f)
    print("Global bit config saved to:", save_path)


def run_gemq_solver_per_block(args):
    """
    Per-block (per-layer) ILP swept over a range of target bits-per-expert.
    For each MoE block `l` and each target `tb`, solve the GEMQ objective
    restricted to block `l` with a single-block budget.

    `tb` and the LP cost are in the same units (raw or effective).
    """
    m = get_model_info(args.model_name)
    bit_cands = list(map(int, args.bit_candidates.split(",")))
    bit_cost = resolve_bit_cost(args, bit_cands)
    max_bit = max(bit_cands)
    min_bit = min(bit_cands)
    max_bit_cost = bit_cost[max_bit]
    min_bit_cost = bit_cost[min_bit]

    # tb sweep range
    tb_min = args.tb_min if args.tb_min is not None else min_bit_cost
    tb_max = args.tb_max if args.tb_max is not None else max_bit_cost
    step = args.tb_step
    assert step > 0, "tb_step must be positive"
    assert tb_max >= tb_min, "tb_max must be >= tb_min"

    n_steps = int(round((tb_max - tb_min) / step)) + 1
    tb_values = [round(tb_min + i * step, 6) for i in range(n_steps)]
    print(f"[per_block] tb sweep ({args.budget_kind}): {tb_values}")
    print(f"[per_block] bit_cost: {bit_cost}")

    solver = GEMQSolver(
        layer_re_path=args.layer_re_path,
        x_space=bit_cands,
        extra_constr=args.extra_constr,
        start_layer_idx=m.first_k_dense_layers,
        bit_cost=bit_cost,
    )

    # Aggregate output:
    #   solutions[l][tb]  = {expert_idx: bit, ...}
    #   objectives[l][tb] = float
    #   infeasible[l][tb] = bool
    solutions = {}
    objectives = {}
    infeasible = {}

    # number of LP-internal experts in a block (num_routed + at most 1 shared;
    # additional shared experts are pinned at max_bit outside the LP).
    num_lp_experts = m.num_routed_experts_per_layer + min(1, m.num_shared_experts_per_layer)
    min_cost = num_lp_experts * min_bit_cost
    max_cost = num_lp_experts * max_bit_cost

    for l in range(m.first_k_dense_layers, m.num_layers):
        solutions[l] = {}
        objectives[l] = {}
        infeasible[l] = {}
        for tb in tb_values:
            block_budget = compute_per_block_budget(m, tb, max_bit_cost)

            if block_budget < min_cost - 1e-9:
                print(f"  [block {l} tb={tb}] budget {block_budget:.3f} < min cost {min_cost:.3f} — infeasible, skipping")
                infeasible[l][tb] = True
                continue
            if block_budget > max_cost + 1e-9:
                block_budget = max_cost  # cap (trivially all-experts-at-max)

            res = solver.solve_block(l, block_budget)
            if res is None:
                infeasible[l][tb] = True
                continue
            opt_dict, obj = res
            solutions[l][tb] = opt_dict[l]
            objectives[l][tb] = obj
            infeasible[l][tb] = False

    save_path = args.save_path
    if not save_path:
        bc_str = ",".join(map(str, bit_cands))
        calib_str, model_str = auto_parse_filename(args.layer_re_path)
        const_str = "" if args.extra_constr == "none" else f"_{args.extra_constr}"
        tag = "PerBlock" if args.budget_kind == "raw" else "PerBlockEff"
        save_path = (
            f"configs/{args.model_name}/{tag}/{calib_str}_tb{tb_min:.3f}-{tb_max:.3f}-{step:.3f}_B{bc_str}{const_str}{model_str}.pkl"
        )

    os.makedirs(osp.dirname(save_path), exist_ok=True)
    payload = {
        "solutions":  solutions,          # {layer: {tb: {expert: bit}}}
        "objectives": objectives,         # {layer: {tb: float}}
        "infeasible": infeasible,         # {layer: {tb: bool}}
        "tb_values": tb_values,
        "budget_kind": args.budget_kind,
        "bit_cost":   bit_cost,            # {k: cost-per-weight} matches LP
        "bit_candidates": bit_cands,
        "extra_constr": args.extra_constr,
        "model_name": args.model_name,
    }
    with open(save_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Per-block solutions saved to: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Bit allocation for MoE models.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--layer_re_path", type=str, default="")
    parser.add_argument("--mode", type=str, default="global", choices=["global", "per_block"])

    # how to interpret bit_budget / tb_* and what coefficient the LP uses for each bit
    parser.add_argument("--budget_kind", type=str, default="raw", choices=["raw", "effective"],
                        help="raw: bit_budget/tb in integer bits; LP cost coef = k. "
                             "effective: bit_budget/tb in effective bits including the per-weight "
                             "scale/zero overhead; LP cost coef = k + overhead[k].")
    parser.add_argument("--groupsize", type=int, default=128,
                        help="Group size used by the deployed quantizer (affects effective-bit overhead).")
    parser.add_argument("--bit_cost", type=str, default="",
                        help="Optional explicit per-bit cost override (effective mode only), e.g. "
                             "'1:1.125,2:2.25,3:3.25'. If empty, defaults to "
                             "{k: k + 16/groupsize if k==1 else k + 32/groupsize} — matches GEMQ's "
                             "symmetric-1-bit + asymmetric-multi-bit convention.")

    # global-mode arg
    parser.add_argument("--bit_budget", type=float, default=None,
                        help="Average bits-per-expert (units = budget_kind). Required for global mode.")

    # per-block-mode args
    parser.add_argument("--tb_min", type=float, default=None,
                        help="Per-block tb sweep lower bound (defaults to min bit's cost, e.g. 1.125 "
                             "for effective 1-bit at groupsize=128).")
    parser.add_argument("--tb_max", type=float, default=None,
                        help="Per-block tb sweep upper bound (defaults to max bit's cost).")
    parser.add_argument("--tb_step", type=float, default=0.125,
                        help="Per-block tb sweep step (default 0.125).")

    parser.add_argument("--bit_candidates", type=str, default="1,2,3")
    parser.add_argument("--ilp_solver", type=str, default="gemq", choices=["gemq"])
    parser.add_argument("--extra_constr", type=str, default="none")
    parser.add_argument("--save_path", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=4))

    if args.ilp_solver != "gemq":
        raise ValueError(f"Unknown solver: {args.ilp_solver}")

    if args.mode == "global":
        if args.bit_budget is None:
            raise ValueError("--bit_budget is required for --mode global")
        run_gemq_solver_global(args)
    elif args.mode == "per_block":
        run_gemq_solver_per_block(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
