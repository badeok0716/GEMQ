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


def compute_global_budget(model_info, bpe_raw, max_bit):
    """
    Returns the total raw-bit budget for the global ILP, mirroring the
    upstream convention: shared experts are pinned to max_bit so they
    cost `(num_shared - 1) * max_bit` outside the LP (one shared expert
    stays inside as a free variable).
    """
    bpl = (
        bpe_raw * (model_info.num_routed_experts_per_layer + model_info.num_shared_experts_per_layer)
        - max(0, model_info.num_shared_experts_per_layer - 1) * max_bit
    )
    total_bits = bpl * (model_info.num_layers - model_info.first_k_dense_layers)
    return total_bits, bpl


def compute_per_block_budget(model_info, tb_raw, max_bit):
    """
    Per-block raw-bit budget at target raw bpe = `tb_raw`. Matches the
    global formula evaluated for a single layer.
    """
    bpl = (
        tb_raw * (model_info.num_routed_experts_per_layer + model_info.num_shared_experts_per_layer)
        - max(0, model_info.num_shared_experts_per_layer - 1) * max_bit
    )
    return bpl


def run_gemq_solver_global(args):
    m = get_model_info(args.model_name)
    bit_cands = list(map(int, args.bit_candidates.split(",")))
    max_bit = max(bit_cands)

    # Convert effective bpe to raw bpe if requested.
    if args.budget_kind == "effective":
        bpe_raw = args.bit_budget - args.effective_offset
    else:
        bpe_raw = args.bit_budget

    total_bits, bpl_raw = compute_global_budget(m, bpe_raw, max_bit)
    print(f"[global] bpe(raw)={bpe_raw:.4f} bpl_raw={bpl_raw:.4f} total_raw_bits={total_bits:.2f}")

    solver = GEMQSolver(
        layer_re_path=args.layer_re_path,
        x_space=bit_cands,
        extra_constr=args.extra_constr,
        start_layer_idx=m.first_k_dense_layers,
    )
    opt_set = solver.solve_all(total_bits=total_bits)

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
    For each MoE block `l` and each target `tb`, solve the same GEMQ objective
    restricted to block `l` with raw-bit budget equal to a single-block bpl.
    """
    m = get_model_info(args.model_name)
    bit_cands = list(map(int, args.bit_candidates.split(",")))
    max_bit = max(bit_cands)

    # Build the tb sweep. `min_bit`/`max_bit` defaults to the candidate range
    # in the requested budget_kind units (effective vs raw).
    if args.budget_kind == "effective":
        tb_min = args.tb_min if args.tb_min is not None else (min(bit_cands) + args.effective_offset)
        tb_max = args.tb_max if args.tb_max is not None else (max(bit_cands) + args.effective_offset)
    else:
        tb_min = args.tb_min if args.tb_min is not None else float(min(bit_cands))
        tb_max = args.tb_max if args.tb_max is not None else float(max(bit_cands))
    step = args.tb_step
    assert step > 0, "tb_step must be positive"
    assert tb_max >= tb_min, "tb_max must be >= tb_min"

    # Inclusive sweep with floating-point tolerance.
    n_steps = int(round((tb_max - tb_min) / step)) + 1
    tb_values = [round(tb_min + i * step, 6) for i in range(n_steps)]
    print(f"[per_block] tb sweep ({args.budget_kind}): {tb_values}")

    solver = GEMQSolver(
        layer_re_path=args.layer_re_path,
        x_space=bit_cands,
        extra_constr=args.extra_constr,
        start_layer_idx=m.first_k_dense_layers,
    )

    # Aggregate output:
    #   solutions[l][tb] = {expert_idx: bit, ...}
    #   objectives[l][tb] = float
    #   infeasible[l][tb] = bool
    solutions = {}
    objectives = {}
    infeasible = {}

    for l in range(m.first_k_dense_layers, m.num_layers):
        solutions[l] = {}
        objectives[l] = {}
        infeasible[l] = {}
        for tb in tb_values:
            # convert tb (in budget_kind units) to raw-bit budget for the ILP
            if args.budget_kind == "effective":
                tb_raw = tb - args.effective_offset
            else:
                tb_raw = tb
            block_budget = compute_per_block_budget(m, tb_raw, max_bit)

            # The minimum achievable block raw-bit cost (every routed expert at min_bit;
            # one shared expert at min_bit; the (num_shared - 1) others fixed at max_bit
            # have already been removed from `block_budget`).
            num_lp_experts = m.num_routed_experts_per_layer + min(1, m.num_shared_experts_per_layer)
            min_cost = num_lp_experts * min(bit_cands)
            max_cost = num_lp_experts * max(bit_cands)
            if block_budget < min_cost - 1e-9:
                print(f"  [block {l} tb={tb}] budget {block_budget:.3f} < min cost {min_cost} — infeasible, skipping")
                infeasible[l][tb] = True
                continue
            if block_budget > max_cost + 1e-9:
                # cap to max (all-experts-at-max-bit is the trivial answer)
                block_budget = max_cost

            res = solver.solve_block(l, block_budget)
            if res is None:
                infeasible[l][tb] = True
                continue

            opt_dict, obj = res
            # opt_dict is `{l: {j: k}}`; flatten to {j: k}
            solutions[l][tb] = opt_dict[l]
            objectives[l][tb] = obj
            infeasible[l][tb] = False

    # Persist artifact
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
        "solutions": solutions,           # {layer: {tb: {expert: bit}}}
        "objectives": objectives,         # {layer: {tb: float}}
        "infeasible": infeasible,         # {layer: {tb: bool}}
        "tb_values": tb_values,
        "budget_kind": args.budget_kind,
        "effective_offset": args.effective_offset,
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
    parser.add_argument("--budget_kind", type=str, default="raw", choices=["raw", "effective"],
                        help="Whether bit_budget / tb are raw bits or include the per-weight "
                             "fp16 scale+zero overhead (effective).")
    parser.add_argument("--effective_offset", type=float, default=0.25,
                        help="Effective-bit overhead added per weight (groupsize=128 with fp16 "
                             "scale+zero ⇒ 32/128 = 0.25).")

    # global-mode args
    parser.add_argument("--bit_budget", type=float, default=None,
                        help="Average bits-per-expert (units = budget_kind). Required for global mode.")

    # per-block-mode args
    parser.add_argument("--tb_min", type=float, default=None,
                        help="Per-block tb sweep lower bound (defaults to min(bit_candidates) "
                             "[+ effective_offset if budget_kind=effective]).")
    parser.add_argument("--tb_max", type=float, default=None,
                        help="Per-block tb sweep upper bound.")
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
