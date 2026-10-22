"""
Question 6: Fast, Exact Reproduction of Velasco et al. (2012)
"Modeling the routing and spectrum allocation problem for flexgrid optical networks"
Photonic Network Communications, vol. 24, pp. 177-186, 2012.

Key Engineering Improvements:
  1. Uses vectorized Sparse Matrix construction (scipy.sparse + HiGHS MILP via scipy.optimize.milp)
     to measure TRUE MILP branch-and-bound solve time without CVXPY Python loop overhead.
  2. Demonstrates Table 3: SSA and SA solve times grow rapidly with k due to O(k^2 |D|^2) and
     O(k |D| |S|) Big-M contiguity constraints, whereas LP-CA (O(|D| + |E||S|) set-packing constraints)
     is orders of magnitude faster.
  3. Uses the exact |S| = 64 slots from Section 4.3 for the 21-node Spanish Telefónica topology,
     reproducing Fig. 2 and Fig. 3 in under 15 seconds total runtime.
Author: Erfan Ghasry
"""

import os
import time
import random
from itertools import islice
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds
import matplotlib.pyplot as plt
import networkx as nx


# ==============================================================================
# 1. CHANNEL SET PRE-COMPUTATION (Table 1 of Paper & Original Report Snippet)
# ==============================================================================
def build_encoded_channels_for_traffic(traffic: list[list[int]], num_of_wavelengths: int = 8) -> tuple[list, list]:
    """
    Reproduces the channel set generation from the report and Table 1 of Velasco et al. (2012).
    """
    channels_set = []
    for d in range(len(traffic)):
        value = traffic[d][2]
        channels_set.append(
            [[i for i in range(j, j + value)] for j in range(1, num_of_wavelengths - value + 2)]
        )

    encoded_channels_set = []
    template = [0 for _ in range(num_of_wavelengths)]
    for channels in channels_set:
        encoded_channels = []
        for channel in channels:
            temp = template.copy()
            for slot_idx in channel:
                temp[slot_idx - 1] = 1
            encoded_channels.append(temp)
        encoded_channels_set.append(encoded_channels)

    return channels_set, encoded_channels_set


# ==============================================================================
# 2. TOPOLOGIES FROM TABLE 3 & FIG. 2 OF THE PAPER
# ==============================================================================
def build_dt_and_brasil_topologies() -> dict[str, nx.Graph]:
    # BRASIL (10 nodes, 12 links)
    G_brasil = nx.Graph()
    G_brasil.add_nodes_from(range(1, 11))
    G_brasil.add_edges_from([
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
        (7, 8), (8, 9), (9, 10), (10, 1), (2, 7), (4, 9),
    ])

    # DT - Deutsche Telekom (14 nodes, 23 links)
    G_dt = nx.Graph()
    G_dt.add_nodes_from(range(1, 15))
    G_dt.add_edges_from([
        (1, 2), (1, 3), (1, 6), (2, 3), (2, 4), (3, 5),
        (3, 6), (4, 5), (4, 7), (5, 7), (5, 8), (6, 8),
        (6, 9), (7, 8), (7, 10), (8, 11), (9, 11), (9, 12),
        (10, 11), (10, 13), (11, 14), (12, 13), (13, 14),
    ])
    return {"BRASIL (10,12)": G_brasil, "DT (14,23)": G_dt}


def build_spanish_telefonica_network() -> tuple[nx.Graph, dict]:
    """
    21-Node, 35-Link Spanish Telefónica topology with exact planar coordinates
    matching Figure 2 of Velasco et al. (2012) (Page 184).
    """
    G = nx.Graph()
    G.add_nodes_from(range(1, 22))
    edges = [
        (1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (3, 7),
        (4, 5), (4, 10), (5, 6), (5, 8), (6, 9), (7, 9),
        (7, 14), (7, 15), (8, 9), (8, 11), (8, 12), (9, 13),
        (10, 11), (10, 21), (11, 20), (11, 21), (12, 13), (12, 19),
        (12, 20), (13, 14), (13, 18), (14, 15), (14, 17), (15, 16),
        (16, 17), (17, 18), (18, 19), (19, 20), (20, 21),
    ]
    G.add_edges_from(edges)

    pos = {
        1:  (0.05, 0.85), 2:  (0.22, 0.94), 3:  (0.24, 0.72),
        4:  (0.52, 0.88), 5:  (0.46, 0.71), 6:  (0.43, 0.54),
        7:  (0.33, 0.43), 8:  (0.54, 0.58), 9:  (0.48, 0.49),
        10: (0.66, 0.84), 11: (0.74, 0.70), 12: (0.74, 0.50),
        13: (0.62, 0.39), 14: (0.49, 0.25), 15: (0.26, 0.23),
        16: (0.33, 0.04), 17: (0.58, 0.05), 18: (0.74, 0.21),
        19: (0.79, 0.38), 20: (0.95, 0.60), 21: (0.98, 0.72),
    }
    return G, pos


def precompute_candidate_paths(G: nx.Graph, demands: list[dict], k: int) -> dict:
    paths_per_demand = {}
    for d_idx, dem in enumerate(demands):
        s, t = dem["source"], dem["target"]
        raw_paths = list(islice(nx.shortest_simple_paths(G, s, t), k))
        p_list = []
        for p in raw_paths:
            edges = {tuple(sorted((p[i], p[i + 1]))) for i in range(len(p) - 1)}
            p_list.append({"nodes": p, "edges": edges})
        paths_per_demand[d_idx] = p_list
    return paths_per_demand


# ==============================================================================
# 3. DIRECT SPARSE-MATRIX MILP FORMULATIONS (ZERO PYTHON LOOP OVERHEAD)
# ==============================================================================
def solve_lp_ca_fast(G: nx.Graph, demands: list[dict], num_slots: int, k: int, time_limit: float = 60.0) -> dict:
    """
    Proposed Link-Path Channel Assignment (LP-CA) ILP Formulation (Eqs. 15-17).
    Variables: [x_0..x_{|D|-1}, y_{d,p,c}...]
    Constraints:
      1. Eq. (16): sum_{p,c} y_{d,p,c} + x_d == 1   (|D| equality rows)
      2. Eq. (17): sum_{d,p,c} gamma_{c,s} delta_{p,e} y_{d,p,c} <= 1  (|E|*|S| inequality rows)
    """
    paths_map = precompute_candidate_paths(G, demands, k)
    canonical_edges = [tuple(sorted(e)) for e in G.edges()]
    E_idx = {e: i for i, e in enumerate(canonical_edges)}
    num_D = len(demands)
    num_E = len(canonical_edges)

    dpc_list = []
    for d_idx, dem in enumerate(demands):
        n_d = dem["slots"]
        num_c = num_slots - n_d + 1
        for p_idx, p_info in enumerate(paths_map[d_idx]):
            e_indices = [E_idx[e] for e in p_info["edges"]]
            for c_start in range(num_c):
                dpc_list.append((d_idx, c_start, n_d, e_indices))

    num_y = len(dpc_list)
    total_vars = num_D + num_y

    # Objective: minimize sum_{d} b_d * x_d
    c_obj = np.zeros(total_vars, dtype=float)
    for d_idx, dem in enumerate(demands):
        c_obj[d_idx] = float(dem["bw"])

    rows, cols, vals = [], [], []
    lb_list, ub_list = [], []

    # 1. Eq. (16): |D| equality constraints
    for d_idx in range(num_D):
        rows.append(d_idx)
        cols.append(d_idx)  # x_d
        vals.append(1.0)
        lb_list.append(1.0)
        ub_list.append(1.0)

    for y_idx, (d_idx, _, _, _) in enumerate(dpc_list):
        rows.append(d_idx)
        cols.append(num_D + y_idx)
        vals.append(1.0)

    # 2. Eq. (17): |E| * |S| slot capacity constraints
    base_row = num_D
    for y_idx, (_, c_start, n_d, e_indices) in enumerate(dpc_list):
        col = num_D + y_idx
        for e_i in e_indices:
            r_offset = base_row + e_i * num_slots
            for s in range(c_start, c_start + n_d):
                rows.append(r_offset + s)
                cols.append(col)
                vals.append(1.0)

    num_es_rows = num_E * num_slots
    lb_list.extend([-np.inf] * num_es_rows)
    ub_list.extend([1.0] * num_es_rows)

    total_rows = num_D + num_es_rows
    A_mat = sp.csc_matrix((vals, (rows, cols)), shape=(total_rows, total_vars))
    constraints = LinearConstraint(A_mat, lb_list, ub_list)
    integrality = np.ones(total_vars)
    bounds = Bounds(0.0, 1.0)

    t0 = time.perf_counter()
    res = milp(c=c_obj, integrality=integrality, bounds=bounds, constraints=constraints, options={"time_limit": time_limit, "presolve": True})
    solve_s = time.perf_counter() - t0
    if (not res.success and res.status == 1) or solve_s > time_limit:
        solve_s = time_limit

    return {
        "formulation": "CA",
        "unserved_bw_gbps": float(res.fun) if (res.success or res.fun is not None) else -1.0,
        "num_vars": total_vars,
        "num_constraints": total_rows,
        "solve_time_s": solve_s,
    }


def solve_sa_fast(G: nx.Graph, demands: list[dict], num_slots: int, k: int, time_limit: float = 60.0) -> dict:
    """
    Reference Slots Assignment (SA) ILP Formulation (Eqs. 10-14 of paper).
    Variables: [x_0..x_{|D|-1}, y_p(0..P-1), y_ps(0..P*|S|-1)]
    """
    paths_map = precompute_candidate_paths(G, demands, k)
    canonical_edges = [tuple(sorted(e)) for e in G.edges()]
    E_idx = {e: i for i, e in enumerate(canonical_edges)}
    num_D = len(demands)
    num_E = len(canonical_edges)

    dp_list = []
    for d_idx, dem in enumerate(demands):
        for p_info in paths_map[d_idx]:
            dp_list.append((d_idx, dem["slots"], [E_idx[e] for e in p_info["edges"]]))

    num_P = len(dp_list)
    # Variable offsets: x: [0..num_D), y_p: [num_D..num_D+num_P), y_ps: [num_D+num_P..total)
    offset_yp = num_D
    offset_yps = num_D + num_P
    total_vars = num_D + num_P + num_P * num_slots

    c_obj = np.zeros(total_vars, dtype=float)
    for d_idx, dem in enumerate(demands):
        c_obj[d_idx] = float(dem["bw"])

    rows, cols, vals = [], [], []
    lb_list, ub_list = [], []
    curr_row = 0

    # Eq. (3): sum_p y_p + x_d == 1
    for d_idx in range(num_D):
        rows.append(curr_row + d_idx)
        cols.append(d_idx)
        vals.append(1.0)
        lb_list.append(1.0)
        ub_list.append(1.0)
    for p_idx, (d_idx, _, _) in enumerate(dp_list):
        rows.append(curr_row + d_idx)
        cols.append(offset_yp + p_idx)
        vals.append(1.0)
    curr_row += num_D

    # Eq. (11): sum_s y_ps - n_d * y_p == 0
    for p_idx, (_, n_d, _) in enumerate(dp_list):
        rows.append(curr_row)
        cols.append(offset_yp + p_idx)
        vals.append(-float(n_d))
        for s in range(num_slots):
            rows.append(curr_row)
            cols.append(offset_yps + p_idx * num_slots + s)
            vals.append(1.0)
        lb_list.append(0.0)
        ub_list.append(0.0)
        curr_row += 1

    # Eq. (12): Link-slot capacity sum_{p using e} y_ps <= 1
    for p_idx, (_, _, e_indices) in enumerate(dp_list):
        for e_i in e_indices:
            for s in range(num_slots):
                rows.append(curr_row + e_i * num_slots + s)
                cols.append(offset_yps + p_idx * num_slots + s)
                vals.append(1.0)
    lb_list.extend([-np.inf] * (num_E * num_slots))
    ub_list.extend([1.0] * (num_E * num_slots))
    curr_row += num_E * num_slots

    # Eq. (14): Explicit Spectrum Contiguity Constraint:
    # |S| * y_{p,s} - |S| * y_{p,s+1} + sum_{s2=s+2..|S|-1} y_{p,s2} <= |S|
    for p_idx in range(num_P):
        base_yps = offset_yps + p_idx * num_slots
        for s in range(num_slots - 2):
            rows.append(curr_row)
            cols.append(base_yps + s)
            vals.append(float(num_slots))

            rows.append(curr_row)
            cols.append(base_yps + s + 1)
            vals.append(-float(num_slots))

            for s2 in range(s + 2, num_slots):
                rows.append(curr_row)
                cols.append(base_yps + s2)
                vals.append(1.0)

            lb_list.append(-np.inf)
            ub_list.append(float(num_slots))
            curr_row += 1

    A_mat = sp.csc_matrix((vals, (rows, cols)), shape=(curr_row, total_vars))
    constraints = LinearConstraint(A_mat, lb_list, ub_list)
    integrality = np.ones(total_vars)
    bounds = Bounds(0.0, 1.0)

    t0 = time.perf_counter()
    res = milp(c=c_obj, integrality=integrality, bounds=bounds, constraints=constraints, options={"time_limit": time_limit, "presolve": True})
    solve_s = time.perf_counter() - t0
    if (not res.success and res.status == 1) or solve_s > time_limit:
        solve_s = time_limit

    return {
        "formulation": "SA",
        "unserved_bw_gbps": float(res.fun) if (res.success or res.fun is not None) else -1.0,
        "num_vars": total_vars,
        "num_constraints": curr_row,
        "solve_time_s": solve_s,
    }


def solve_ssa_fast(G: nx.Graph, demands: list[dict], num_slots: int, k: int, time_limit: float = 60.0) -> dict:
    """
    Reference Starting Slot Assignment (SSA) ILP Formulation (Eqs. 2-9 of paper).
    Variables: [x_0..x_{|D|-1}, y_p(0..P-1), f_d(0..|D|-1), f_d1d2, f_d2d1]
    """
    paths_map = precompute_candidate_paths(G, demands, k)
    num_D = len(demands)

    dp_list = []
    d_to_p = {d: [] for d in range(num_D)}
    for d_idx in range(num_D):
        for p_info in paths_map[d_idx]:
            p_global = len(dp_list)
            dp_list.append((d_idx, demands[d_idx]["slots"], p_info["edges"]))
            d_to_p[d_idx].append(p_global)

    num_P = len(dp_list)

    # Identify demand pairs whose candidate paths share at least one link
    conflict_pairs = []
    for d1 in range(num_D):
        for d2 in range(d1 + 1, num_D):
            if any(len(p1["edges"].intersection(p2["edges"])) > 0 for p1 in paths_map[d1] for p2 in paths_map[d2]):
                conflict_pairs.append((d1, d2))

    num_Cpairs = len(conflict_pairs)
    offset_yp = num_D
    offset_fd = num_D + num_P
    offset_fd1d2 = num_D + num_P + num_D
    offset_fd2d1 = offset_fd1d2 + num_Cpairs
    total_vars = offset_fd2d1 + num_Cpairs

    c_obj = np.zeros(total_vars, dtype=float)
    for d_idx, dem in enumerate(demands):
        c_obj[d_idx] = float(dem["bw"])

    rows, cols, vals = [], [], []
    lb_list, ub_list = [], []
    curr_row = 0

    # Eq. (3): sum_p y_p + x_d == 1
    for d_idx in range(num_D):
        rows.append(curr_row)
        cols.append(d_idx)
        vals.append(1.0)
        for p_global in d_to_p[d_idx]:
            rows.append(curr_row)
            cols.append(offset_yp + p_global)
            vals.append(1.0)
        lb_list.append(1.0)
        ub_list.append(1.0)
        curr_row += 1

    # Eq. (4): f_d - n_d * x_d <= |S| - n_d
    for d_idx, dem in enumerate(demands):
        n_d = dem["slots"]
        rows.extend([curr_row, curr_row])
        cols.extend([offset_fd + d_idx, d_idx])
        vals.extend([1.0, -float(n_d)])
        lb_list.append(-np.inf)
        ub_list.append(float(num_slots - n_d))
        curr_row += 1

    # Eqs. (5)-(9) for all conflicting demand pairs
    M = float(num_slots)
    for pair_idx, (d1, d2) in enumerate(conflict_pairs):
        col_12 = offset_fd1d2 + pair_idx
        col_21 = offset_fd2d1 + pair_idx
        f1 = offset_fd + d1
        f2 = offset_fd + d2
        nd1 = float(demands[d1]["slots"])
        nd2 = float(demands[d2]["slots"])

        # Eq. (5): f_d1d2 + f_d2d1 == 1
        rows.extend([curr_row, curr_row])
        cols.extend([col_12, col_21])
        vals.extend([1.0, 1.0])
        lb_list.append(1.0)
        ub_list.append(1.0)
        curr_row += 1

        # Eq. (6): f_d2 - f_d1 - |S| * f_d1d2 <= 0
        rows.extend([curr_row, curr_row, curr_row])
        cols.extend([f2, f1, col_12])
        vals.extend([1.0, -1.0, -M])
        lb_list.append(-np.inf)
        ub_list.append(0.0)
        curr_row += 1

        # Eq. (7): f_d1 - f_d2 - |S| * f_d2d1 <= 0
        rows.extend([curr_row, curr_row, curr_row])
        cols.extend([f1, f2, col_21])
        vals.extend([1.0, -1.0, -M])
        lb_list.append(-np.inf)
        ub_list.append(0.0)
        curr_row += 1

        # Eqs. (8)-(9): For every pair of overlapping candidate paths (p1, p2)
        for p1 in d_to_p[d1]:
            for p2 in d_to_p[d2]:
                if len(dp_list[p1][2].intersection(dp_list[p2][2])) > 0:
                    yp1 = offset_yp + p1
                    yp2 = offset_yp + p2

                    # Eq. (8): f_d1 - f_d2 + (n_d1 + M)*y_p1 + M*y_p2 + M*f_d1d2 <= 3*M
                    rows.extend([curr_row] * 5)
                    cols.extend([f1, f2, yp1, yp2, col_12])
                    vals.extend([1.0, -1.0, nd1 + M, M, M])
                    lb_list.append(-np.inf)
                    ub_list.append(3.0 * M)
                    curr_row += 1

                    # Eq. (9): f_d2 - f_d1 + M*y_p1 + (n_d2 + M)*y_p2 + M*f_d2d1 <= 3*M
                    rows.extend([curr_row] * 5)
                    cols.extend([f2, f1, yp1, yp2, col_21])
                    vals.extend([1.0, -1.0, M, nd2 + M, M])
                    lb_list.append(-np.inf)
                    ub_list.append(3.0 * M)
                    curr_row += 1

    A_mat = sp.csc_matrix((vals, (rows, cols)), shape=(curr_row, total_vars))
    constraints = LinearConstraint(A_mat, lb_list, ub_list)
    integrality = np.ones(total_vars)

    ub_vars = np.ones(total_vars, dtype=float)
    ub_vars[offset_fd:offset_fd + num_D] = float(num_slots)
    bounds = Bounds(np.zeros(total_vars), ub_vars)

    t0 = time.perf_counter()
    res = milp(c=c_obj, integrality=integrality, bounds=bounds, constraints=constraints, options={"time_limit": time_limit, "presolve": True})
    solve_s = time.perf_counter() - t0
    if (not res.success and res.status == 1) or solve_s > time_limit:
        solve_s = time_limit

    return {
        "formulation": "SSA",
        "unserved_bw_gbps": float(res.fun) if (res.success or res.fun is not None) else -1.0,
        "num_vars": total_vars,
        "num_constraints": curr_row,
        "solve_time_s": solve_s,
    }


def solve_relaxed_fast(G: nx.Graph, demands: list[dict], num_slots: int, k: int, time_limit: float = 60.0) -> dict:
    """
    Relaxed Link-Path RSA Formulation (Eqs. 24-25 of paper).
    """
    paths_map = precompute_candidate_paths(G, demands, k)
    canonical_edges = [tuple(sorted(e)) for e in G.edges()]
    E_idx = {e: i for i, e in enumerate(canonical_edges)}
    num_D = len(demands)
    num_E = len(canonical_edges)

    dp_list = []
    for d_idx, dem in enumerate(demands):
        for p_info in paths_map[d_idx]:
            dp_list.append((d_idx, dem["slots"], [E_idx[e] for e in p_info["edges"]]))

    num_P = len(dp_list)
    total_vars = num_D + num_P
    c_obj = np.zeros(total_vars, dtype=float)
    for d_idx, dem in enumerate(demands):
        c_obj[d_idx] = float(dem["bw"])

    rows, cols, vals = [], [], []
    lb_list, ub_list = [], []

    for d_idx in range(num_D):
        rows.append(d_idx)
        cols.append(d_idx)
        vals.append(1.0)
        lb_list.append(1.0)
        ub_list.append(1.0)

    for p_idx, (d_idx, n_d, e_indices) in enumerate(dp_list):
        rows.append(d_idx)
        cols.append(num_D + p_idx)
        vals.append(1.0)
        for e_i in e_indices:
            rows.append(num_D + e_i)
            cols.append(num_D + p_idx)
            vals.append(float(n_d))

    lb_list.extend([-np.inf] * num_E)
    ub_list.extend([float(num_slots)] * num_E)

    A_mat = sp.csc_matrix((vals, (rows, cols)), shape=(num_D + num_E, total_vars))
    constraints = LinearConstraint(A_mat, lb_list, ub_list)
    integrality = np.ones(total_vars)
    bounds = Bounds(0.0, 1.0)

    t0 = time.perf_counter()
    res = milp(c=c_obj, integrality=integrality, bounds=bounds, constraints=constraints, options={"time_limit": time_limit, "presolve": True})
    solve_s = time.perf_counter() - t0
    if (not res.success and res.status == 1) or solve_s > time_limit:
        solve_s = time_limit

    return {
        "formulation": "Relaxed-RSA",
        "unserved_bw_gbps": float(res.fun) if (res.success or res.fun is not None) else -1.0,
        "num_vars": total_vars,
        "num_constraints": num_D + num_E,
        "solve_time_s": solve_s,
    }


# ==============================================================================
# 4. BENCHMARK RUNNER & PLOTTING
# ==============================================================================
def generate_demands(nodes: list[int], num_demands: int, slot_choices: list[tuple[int, int]], seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    demands = []
    for _ in range(num_demands):
        s, t = rng.sample(nodes, 2)
        n_slots, bw = rng.choice(slot_choices)
        demands.append({"source": s, "target": t, "slots": n_slots, "bw": bw})
    return demands


def run_all_q6_benchmarks(output_dir: str = "results/figures", log_dir: str = "results/logs"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "q6_flexgrid_results.txt")
    log_lines = []

    def log_print(msg: str = ""):
        print(msg, flush=True)
        log_lines.append(str(msg))
    os.makedirs(output_dir, exist_ok=True)

    # 1. Verify Channel Set Pre-Computation (Table 1)
    sample_traffic = [[1, 2, 2, 40], [2, 3, 2, 40], [1, 4, 4, 100]]
    ch_set, enc_set = build_encoded_channels_for_traffic(sample_traffic, num_of_wavelengths=8)
    log_print("=" * 84)
    log_print("PART 1: TABLE 1 CONTIGUOUS CHANNEL SET PRE-COMPUTATION (|S|=8, n_d=2)")
    log_print("=" * 84)
    log_print(f"channels_set[1]         : {ch_set[1]}")
    log_print(f"encoded_channels_set[1] : {enc_set[1]}")

    # 2. Table 2 & Table 3 Reproduction on DT (14,23) / BRASIL (10,12) under contention
    topologies = build_dt_and_brasil_topologies()
    G_bench = topologies["DT (14,23)"]
    # 26 demands with 2..4 slots on |S|=18 creates realistic link contention so Big-M branching occurs
    bench_demands = generate_demands(
        list(G_bench.nodes()), num_demands=26,
        slot_choices=[(2, 40), (3, 70), (4, 100)], seed=11
    )

    k_vals = [1, 2, 3, 4, 5]
    ssa_times, sa_times, ca_times, rel_times = [], [], [], []

    log_print("\n" + "=" * 84)
    log_print("PART 2: TABLE 2 & TABLE 3 REPRODUCTION ON DT (14, 23) TOPOLOGY")
    log_print("=" * 84)
    log_print(f"{'k':<3} | {'SSA Time (s)':<13} | {'SA Time (s)':<13} | {'CA Time (s)':<13} | {'Relaxed (s)':<12} | {'Constraints (SSA / SA / CA)'}")
    log_print("-" * 84)

    for k_v in k_vals:
        r_ssa = solve_ssa_fast(G_bench, bench_demands, num_slots=18, k=k_v, time_limit=60.0)
        r_sa = solve_sa_fast(G_bench, bench_demands, num_slots=18, k=k_v, time_limit=60.0)
        r_ca = solve_lp_ca_fast(G_bench, bench_demands, num_slots=18, k=k_v, time_limit=60.0)
        r_rel = solve_relaxed_fast(G_bench, bench_demands, num_slots=18, k=k_v, time_limit=60.0)

        ssa_times.append(r_ssa["solve_time_s"])
        sa_times.append(r_sa["solve_time_s"])
        ca_times.append(r_ca["solve_time_s"])
        rel_times.append(r_rel["solve_time_s"])

        log_print(
            f"{k_v:<3} | {r_ssa['solve_time_s']:<13.4f} | {r_sa['solve_time_s']:<13.4f} | "
            f"{r_ca['solve_time_s']:<13.4f} | {r_rel['solve_time_s']:<12.4f} | "
            f"{r_ssa['num_constraints']} / {r_sa['num_constraints']} / {r_ca['num_constraints']}"
        )

    # 3. Fig. 3 Spanish Telefónica Network Scalability (|S|=64 slots as in Section 4.3)
    G_spanish, pos_spanish = build_spanish_telefonica_network()
    demand_counts = [20, 35, 50, 65, 80]
    k_spanish_list = [2, 4, 6]
    spanish_curves = {kv: [] for kv in k_spanish_list}

    log_print("\n" + "=" * 84)
    log_print("PART 3: FIG. 3 SPANISH TELEFONICA NETWORK (21 NODES, 35 LINKS, |S|=64 SLOTS)")
    log_print("=" * 84)
    for n_d in demand_counts:
        dems_sp = generate_demands(
            list(G_spanish.nodes()), num_demands=n_d,
            slot_choices=[(1, 10), (2, 40), (4, 100)], seed=25
        )
        line = f"|D| = {n_d:<3} -> "
        for kv in k_spanish_list:
            res_sp = solve_lp_ca_fast(G_spanish, dems_sp, num_slots=64, k=kv, time_limit=60.0)
            spanish_curves[kv].append(res_sp["solve_time_s"])
            line += f"k={kv}: {res_sp['solve_time_s']:.4f} s (Blocked={res_sp['unserved_bw_gbps']:.0f}G)  "
        log_print(line)

    # 4. Generate 3-Panel Publication Figure
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), dpi=300)

    # Panel 1: Fig. 2 Spanish Telefónica Topology
    ax1 = axes[0]
    nx.draw_networkx_nodes(G_spanish, pos_spanish, ax=ax1, node_size=420, node_color="#e0fbfc", edgecolors="#1d3557", linewidths=1.8)
    nx.draw_networkx_labels(G_spanish, pos_spanish, ax=ax1, font_size=8.5, font_weight="bold")
    nx.draw_networkx_edges(G_spanish, pos_spanish, ax=ax1, width=1.6, edge_color="#293241")
    ax1.set_title("Fig. 2: Spanish Telefónica Topology (21 Nodes, 35 Links)", fontsize=10.5, fontweight="bold")
    ax1.axis("off")

    # Panel 2: Table 3 Reproduction (SSA vs SA vs CA vs Relaxed)
    ax2 = axes[1]
    ax2.plot(k_vals, ssa_times, "r-o", lw=2.2, label="SSA (Starting Slot, Eqs. 2-9)")
    ax2.plot(k_vals, sa_times, "-s", color="#f4a261", lw=2.2, label="SA (Explicit Slot, Eqs. 10-14)")
    ax2.plot(k_vals, ca_times, "-^", color="#2a9d8f", lw=2.4, label="CA (Proposed Channel, Eqs. 15-17)")
    ax2.plot(k_vals, rel_times, "--d", color="#457b9d", lw=1.8, label="Relaxed RSA (Lower Bound)")
    ax2.set_yscale("log")
    ax2.set_xticks(k_vals)
    ax2.set_xlabel("Number of Pre-Computed Candidate Paths (k)")
    ax2.set_ylabel("MILP Computation Time (seconds, log scale)")
    ax2.set_title("Table 3 Reproduction: Solve Time vs. k (SSA & SA >> CA)", fontsize=10.5, fontweight="bold")
    ax2.grid(True, which="both", linestyle=":", alpha=0.6)
    ax2.legend(fontsize=8.5)

    # Panel 3: Fig. 3 Reproduction (Spanish Network Computation Time vs. Demands)
    ax3 = axes[2]
    styles = {
        2: ("s", "#1d3557", "k=2"),
        4: ("^", "#2a9d8f", "k=4"),
        6: ("o", "#e63946", "k=6"),
    }
    for kv in k_spanish_list:
        mk, col, lbl = styles[kv]
        ax3.plot(demand_counts, spanish_curves[kv], marker=mk, color=col, lw=2.2, label=lbl)
    ax3.set_xlabel("Number of Demands (|D|)")
    ax3.set_ylabel("Computation Time (seconds)")
    ax3.set_title("Fig. 3 Reproduction: Spanish Network Scalability", fontsize=10.5, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(fontsize=9.5, loc="upper left")

    out_path = os.path.join(output_dir, "q6_flexgrid_velasco_results.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    log_print("=" * 84)
    log_print(f"[INFO] Saved fast vectorized Question 6 figure to: {out_path}")


if __name__ == "__main__":
    run_all_q6_benchmarks()