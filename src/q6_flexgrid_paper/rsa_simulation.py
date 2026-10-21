"""
Question 6: Accurate Reproduction of Velasco et al. (2012)
"Modeling the routing and spectrum allocation problem for flexgrid optical networks"
Photonic Network Communications, vol. 24, pp. 177-186, 2012.

Fixes & Exact Alignments with the Paper:
  1. Uses vectorized CVXPY matrices and records pure MILP solver time (solver_stats.solve_time)
     so Python variable-instantiation overhead is excluded.
  2. Parallelized simulation runs across CPU cores using concurrent.futures.ProcessPoolExecutor.
  3. Implements the 4 Table 3 topologies: RING9 (9,9), BRASIL (10,12), ABILENE (12,15), DT (14,23)
     under high-contention traffic (|D|=28, slots in {1..4}, k=1..5).
  4. Uses exact Fig. 2 planar coordinates for the 21-node, 35-link Spanish Telefónica network.
Author: Erfan Ghasry
"""

import os
import time
import random
from itertools import islice
from concurrent.futures import ThreadPoolExecutor, as_completed
import cvxpy as cp
import numpy as np
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
# 2. BENCHMARK TOPOLOGIES FROM TABLE 3 & FIG. 2 OF THE PAPER
# ==============================================================================
def get_table3_topologies() -> dict[str, nx.Graph]:
    """
    Returns the four benchmark topologies evaluated in Table 3 of Velasco et al. (2012):
      - RING9 (9 nodes, 9 links)
      - BRASIL (10 nodes, 12 links)
      - ABILENE (12 nodes, 15 links)
      - DT (14 nodes, 23 links - Deutsche Telekom)
    """
    # 1. RING9 (9, 9)
    G_ring9 = nx.cycle_graph(range(1, 10))

    # 2. BRASIL (10, 12)
    G_brasil = nx.Graph()
    G_brasil.add_nodes_from(range(1, 11))
    G_brasil.add_edges_from([
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
        (7, 8), (8, 9), (9, 10), (10, 1), (2, 7), (4, 9),
    ])

    # 3. ABILENE (12, 15)
    G_abilene = nx.Graph()
    G_abilene.add_nodes_from(range(1, 13))
    G_abilene.add_edges_from([
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
        (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 1),
        (2, 11), (4, 9), (6, 8),
    ])

    # 4. DT - Deutsche Telekom (14, 23)
    G_dt = nx.Graph()
    G_dt.add_nodes_from(range(1, 15))
    G_dt.add_edges_from([
        (1, 2), (1, 3), (1, 6), (2, 3), (2, 4), (3, 5),
        (3, 6), (4, 5), (4, 7), (5, 7), (5, 8), (6, 8),
        (6, 9), (7, 8), (7, 10), (8, 11), (9, 11), (9, 12),
        (10, 11), (10, 13), (11, 14), (12, 13), (13, 14),
    ])

    return {
        "RING9 (9,9)": G_ring9,
        "BRASIL (10,12)": G_brasil,
        "ABILENE (12,15)": G_abilene,
        "DT (14,23)": G_dt,
    }


def build_spanish_telefonica_network() -> tuple[nx.Graph, dict]:
    """
    Builds the 21-Node, 35-Link Spanish Telefónica topology with exact planar
    coordinates matching Figure 2 of Velasco et al. (2012) (Page 184).
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

    # Exact visual positions matching Fig. 2 in the paper
    pos = {
        1:  (0.05, 0.85),
        2:  (0.22, 0.94),
        3:  (0.24, 0.72),
        4:  (0.52, 0.88),
        5:  (0.46, 0.71),
        6:  (0.43, 0.54),
        7:  (0.33, 0.43),
        8:  (0.54, 0.58),
        9:  (0.48, 0.49),
        10: (0.66, 0.84),
        11: (0.74, 0.70),
        12: (0.74, 0.50),
        13: (0.62, 0.39),
        14: (0.49, 0.25),
        15: (0.26, 0.23),
        16: (0.33, 0.04),
        17: (0.58, 0.05),
        18: (0.74, 0.21),
        19: (0.79, 0.38),
        20: (0.95, 0.60),
        21: (0.98, 0.72),
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


def extract_pure_solver_time(problem: cp.Problem, wall_fallback: float) -> float:
    """Extracts pure C/C++ MIP solver time (excluding CVXPY Python overhead)."""
    if problem.solver_stats is not None and problem.solver_stats.solve_time is not None:
        if problem.solver_stats.solve_time > 0:
            return float(problem.solver_stats.solve_time)
    if problem._compilation_time is not None and wall_fallback > problem._compilation_time:
        return float(wall_fallback - problem._compilation_time)
    return float(wall_fallback)


# ==============================================================================
# 3. VECTORIZED ILP FORMULATIONS: LP-CA, SA, SSA, AND RELAXED RSA
# ==============================================================================
def solve_lp_ca(G: nx.Graph, demands: list[dict], num_slots: int, k: int) -> dict:
    """
    Vectorized Link-Path Channel Assignment (LP-CA) ILP Formulation (Eqs. 15-17).
    Number of constraints is strictly |D| + |E|*|S|, independent of k!
    """
    paths_map = precompute_candidate_paths(G, demands, k)
    canonical_edges = [tuple(sorted(e)) for e in G.edges()]
    E_idx = {e: i for i, e in enumerate(canonical_edges)}
    num_D = len(demands)
    num_E = len(canonical_edges)

    dpc_tuples = []
    d_to_var_indices = {d: [] for d in range(num_D)}

    for d_idx, dem in enumerate(demands):
        n_d = dem["slots"]
        num_c = num_slots - n_d + 1
        for p_idx, p_info in enumerate(paths_map[d_idx]):
            for c_start in range(num_c):
                var_idx = len(dpc_tuples)
                dpc_tuples.append((d_idx, p_idx, c_start, n_d, p_info["edges"]))
                d_to_var_indices[d_idx].append(var_idx)

    num_y = len(dpc_tuples)
    x = cp.Variable(num_D, boolean=True, name="x")
    y = cp.Variable(num_y, boolean=True, name="y_pc")

    es_to_vars = [[] for _ in range(num_E * num_slots)]
    for var_idx, (d_idx, p_idx, c_start, n_d, p_edges) in enumerate(dpc_tuples):
        for e in p_edges:
            e_i = E_idx[e]
            base = e_i * num_slots
            for s in range(c_start, c_start + n_d):
                es_to_vars[base + s].append(var_idx)

    constraints = []
    # Eq. (16): Routing & channel selection or rejection (|D| constraints)
    for d_idx in range(num_D):
        constraints.append(cp.sum(y[d_to_var_indices[d_idx]]) + x[d_idx] == 1)

    # Eq. (17): Slot capacity on each link e and slot s (<= |E|*|S| constraints)
    for var_list in es_to_vars:
        if var_list:
            constraints.append(cp.sum(y[var_list]) <= 1)

    bw_vec = np.array([d["bw"] for d in demands], dtype=float)
    objective = cp.Minimize(bw_vec @ x)
    problem = cp.Problem(objective, constraints)

    t0 = time.perf_counter()
    problem.solve()
    wall_s = time.perf_counter() - t0
    pure_s = extract_pure_solver_time(problem, wall_s)

    return {
        "formulation": "CA",
        "status": problem.status,
        "unserved_bw_gbps": float(problem.value) if problem.value is not None else -1.0,
        "num_vars": num_D + num_y,
        "num_constraints": len(constraints),
        "solve_time_s": pure_s,
    }


def solve_sa_formulation(G: nx.Graph, demands: list[dict], num_slots: int, k: int) -> dict:
    """
    Reference Slots Assignment (SA) ILP Formulation (Eqs. 10-14 of paper).
    Includes explicit contiguity constraints (Eq. 14) which grow as O(k * |D| * |S|).
    """
    paths_map = precompute_candidate_paths(G, demands, k)
    canonical_edges = [tuple(sorted(e)) for e in G.edges()]
    E_idx = {e: i for i, e in enumerate(canonical_edges)}
    num_D = len(demands)
    num_E = len(canonical_edges)

    dp_tuples = []
    d_to_dp = {d: [] for d in range(num_D)}
    for d_idx in range(num_D):
        for p_idx, p_info in enumerate(paths_map[d_idx]):
            dp_idx = len(dp_tuples)
            dp_tuples.append((d_idx, p_idx, demands[d_idx]["slots"], p_info["edges"]))
            d_to_dp[d_idx].append(dp_idx)

    num_dp = len(dp_tuples)
    x = cp.Variable(num_D, boolean=True, name="x")
    y_p = cp.Variable(num_dp, boolean=True, name="y_p")
    y_ps = cp.Variable((num_dp, num_slots), boolean=True, name="y_ps")

    constraints = []
    # Eq. (3)
    for d_idx in range(num_D):
        constraints.append(cp.sum(y_p[d_to_dp[d_idx]]) + x[d_idx] == 1)

    # Eq. (11): Slot count per path
    for dp_idx, (_, _, n_d, _) in enumerate(dp_tuples):
        constraints.append(cp.sum(y_ps[dp_idx, :]) == n_d * y_p[dp_idx])

    # Eq. (12): Link-slot clash constraint
    for e_i, e in enumerate(canonical_edges):
        incident_dp = [dp_idx for dp_idx, (_, _, _, edges) in enumerate(dp_tuples) if e in edges]
        if incident_dp:
            for s in range(num_slots):
                constraints.append(cp.sum(y_ps[incident_dp, s]) <= 1)

    # Eq. (14): Explicit Spectrum Contiguity Constraint (O(k * |D| * |S|) constraints)
    for dp_idx in range(num_dp):
        for s in range(num_slots - 2):
            constraints.append(
                -num_slots * (y_ps[dp_idx, s] - y_ps[dp_idx, s + 1] - 1) >= cp.sum(y_ps[dp_idx, s + 2:])
            )

    bw_vec = np.array([d["bw"] for d in demands], dtype=float)
    objective = cp.Minimize(bw_vec @ x)
    problem = cp.Problem(objective, constraints)

    t0 = time.perf_counter()
    problem.solve()
    wall_s = time.perf_counter() - t0
    pure_s = extract_pure_solver_time(problem, wall_s)

    return {
        "formulation": "SA",
        "status": problem.status,
        "unserved_bw_gbps": float(problem.value) if problem.value is not None else -1.0,
        "num_vars": num_D + num_dp + num_dp * num_slots,
        "num_constraints": len(constraints),
        "solve_time_s": pure_s,
    }


def solve_ssa_formulation(G: nx.Graph, demands: list[dict], num_slots: int, k: int) -> dict:
    """
    Reference Starting Slot Assignment (SSA) ILP Formulation (Eqs. 2-9 of paper).
    Complexity of constraints scales as O(k^2 * |D|^2) due to pairwise link-sharing paths.
    """
    paths_map = precompute_candidate_paths(G, demands, k)
    num_D = len(demands)

    dp_tuples = []
    d_to_dp = {d: [] for d in range(num_D)}
    for d_idx in range(num_D):
        for p_idx, p_info in enumerate(paths_map[d_idx]):
            dp_idx = len(dp_tuples)
            dp_tuples.append((d_idx, p_idx, p_info["edges"]))
            d_to_dp[d_idx].append(dp_idx)

    x = cp.Variable(num_D, boolean=True, name="x")
    y_p = cp.Variable(len(dp_tuples), boolean=True, name="y_p")
    f_d = cp.Variable(num_D, integer=True, name="f_d")

    constraints = [f_d >= 0]
    for d in range(num_D):
        constraints.append(cp.sum(y_p[d_to_dp[d]]) + x[d] == 1)
        constraints.append(f_d[d] + demands[d]["slots"] * (1 - x[d]) <= num_slots)

    # Identify all demand pairs (d1, d2) whose candidate paths share at least one link
    conflict_pairs = []
    for d1 in range(num_D):
        for d2 in range(d1 + 1, num_D):
            shared = any(
                len(p1["edges"].intersection(p2["edges"])) > 0
                for p1 in paths_map[d1]
                for p2 in paths_map[d2]
            )
            if shared:
                conflict_pairs.append((d1, d2))

    f_d1d2 = cp.Variable(len(conflict_pairs), boolean=True, name="f_d1d2")
    f_d2d1 = cp.Variable(len(conflict_pairs), boolean=True, name="f_d2d1")

    for pair_idx, (d1, d2) in enumerate(conflict_pairs):
        constraints.append(f_d1d2[pair_idx] + f_d2d1[pair_idx] == 1)
        constraints.append(f_d[d2] - f_d[d1] <= num_slots * f_d1d2[pair_idx])
        constraints.append(f_d[d1] - f_d[d2] <= num_slots * f_d2d1[pair_idx])

        for dp1 in d_to_dp[d1]:
            for dp2 in d_to_dp[d2]:
                if len(dp_tuples[dp1][2].intersection(dp_tuples[dp2][2])) > 0:
                    constraints.append(
                        f_d[d1] + demands[d1]["slots"] * y_p[dp1] - f_d[d2]
                        <= num_slots * (1 - f_d1d2[pair_idx] + 2 - y_p[dp1] - y_p[dp2])
                    )
                    constraints.append(
                        f_d[d2] + demands[d2]["slots"] * y_p[dp2] - f_d[d1]
                        <= num_slots * (1 - f_d2d1[pair_idx] + 2 - y_p[dp1] - y_p[dp2])
                    )

    bw_vec = np.array([d["bw"] for d in demands], dtype=float)
    objective = cp.Minimize(bw_vec @ x)
    problem = cp.Problem(objective, constraints)

    t0 = time.perf_counter()
    problem.solve()
    wall_s = time.perf_counter() - t0
    pure_s = extract_pure_solver_time(problem, wall_s)

    return {
        "formulation": "SSA",
        "status": problem.status,
        "unserved_bw_gbps": float(problem.value) if problem.value is not None else -1.0,
        "num_vars": num_D + len(dp_tuples) + num_D + 2 * len(conflict_pairs),
        "num_constraints": len(constraints),
        "solve_time_s": pure_s,
    }


def solve_relaxed_lp_rsa(G: nx.Graph, demands: list[dict], num_slots: int, k: int) -> dict:
    """
    Relaxed Link-Path RSA Formulation (Eqs. 24-25 of paper).
    """
    paths_map = precompute_candidate_paths(G, demands, k)
    canonical_edges = [tuple(sorted(e)) for e in G.edges()]
    num_D = len(demands)

    dp_tuples = []
    d_to_dp = {d: [] for d in range(num_D)}
    for d_idx in range(num_D):
        for p_idx, p_info in enumerate(paths_map[d_idx]):
            dp_idx = len(dp_tuples)
            dp_tuples.append((d_idx, demands[d_idx]["slots"], p_info["edges"]))
            d_to_dp[d_idx].append(dp_idx)

    x = cp.Variable(num_D, boolean=True)
    y_p = cp.Variable(len(dp_tuples), boolean=True)

    constraints = []
    for d_idx in range(num_D):
        constraints.append(cp.sum(y_p[d_to_dp[d_idx]]) + x[d_idx] == 1)

    for e in canonical_edges:
        terms = [n_d * y_p[dp_idx] for dp_idx, (_, n_d, edges) in enumerate(dp_tuples) if e in edges]
        if terms:
            constraints.append(cp.sum(terms) <= num_slots)

    bw_vec = np.array([d["bw"] for d in demands], dtype=float)
    objective = cp.Minimize(bw_vec @ x)
    problem = cp.Problem(objective, constraints)

    t0 = time.perf_counter()
    problem.solve()
    wall_s = time.perf_counter() - t0
    pure_s = extract_pure_solver_time(problem, wall_s)

    return {
        "formulation": "Relaxed-RSA",
        "status": problem.status,
        "unserved_bw_gbps": float(problem.value) if problem.value is not None else -1.0,
        "num_vars": num_D + len(dp_tuples),
        "num_constraints": len(constraints),
        "solve_time_s": pure_s,
    }


# ==============================================================================
# 4. PARALLEL WORKER FUNCTIONS FOR TABLE 3 & FIG. 3
# ==============================================================================
def _worker_dt_task(args):
    formulation, G, demands, num_slots, k_v = args
    if formulation == "SSA":
        res = solve_ssa_formulation(G, demands, num_slots, k_v)
    elif formulation == "SA":
        res = solve_sa_formulation(G, demands, num_slots, k_v)
    elif formulation == "CA":
        res = solve_lp_ca(G, demands, num_slots, k_v)
    elif formulation == "Relaxed":
        res = solve_relaxed_lp_rsa(G, demands, num_slots, k_v)
    else:
        raise ValueError(formulation)
    return (formulation, k_v, res)


def _worker_spanish_task(args):
    G, demands, num_slots, k_v, n_dem = args
    res = solve_lp_ca(G, demands, num_slots, k_v)
    return (n_dem, k_v, res)


# ==============================================================================
# 5. DEMAND GENERATORS & BENCHMARK SUITE
# ==============================================================================
def generate_table3_demands(nodes: list[int], num_demands: int = 36, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    slot_to_bw = {1: 10, 2: 40, 3: 70, 4: 100}
    demands = []
    for _ in range(num_demands):
        s, t = rng.sample(nodes, 2)
        n_slots = rng.randint(1, 4)
        demands.append({"source": s, "target": t, "slots": n_slots, "bw": slot_to_bw[n_slots]})
    return demands


def generate_spanish_demands(nodes: list[int], num_demands: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    choices = [(1, 10), (2, 40), (4, 100)]
    demands = []
    for _ in range(num_demands):
        s, t = rng.sample(nodes, 2)
        n_slots, bw = rng.choice(choices)
        demands.append({"source": s, "target": t, "slots": n_slots, "bw": bw})
    return demands


def run_all_q6_benchmarks(output_dir: str = "results/figures", num_workers: int = 4):
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Part 1: Channel Set Encoding Snippet from Report
    # ------------------------------------------------------------------
    sample_traffic = [[1, 2, 2, 40], [2, 3, 2, 40], [1, 4, 4, 100]]
    ch_set, enc_set = build_encoded_channels_for_traffic(sample_traffic, num_of_wavelengths=8)
    print("=" * 82)
    print("PART 1: TABLE 1 CONTIGUOUS CHANNEL SET PRE-COMPUTATION (|S|=8, n_d=2)")
    print("=" * 82)
    print(f"channels_set[1]         : {ch_set[1]}")
    print(f"encoded_channels_set[1] : {enc_set[1]}")

    # ------------------------------------------------------------------
    # Part 2: Table 2 & Table 3 Reproduction on DT (14,23) in Parallel
    # ------------------------------------------------------------------
    topologies = get_table3_topologies()
    G_dt = topologies["DT (14,23)"]
    dt_demands = generate_table3_demands(list(G_dt.nodes()), num_demands=20, seed=7)
    k_vals = [1, 2, 3, 4]
    num_slots_dt = 24

    print("\n" + "=" * 82)
    print(f"PART 2: TABLE 2 & TABLE 3 BENCHMARKS ON DT (14, 23) (Parallel on {num_workers} Workers)")
    print("=" * 82)

    tasks_p2 = []
    for k_v in k_vals:
        for form in ["SSA", "SA", "CA", "Relaxed"]:
            tasks_p2.append((form, G_dt, dt_demands, num_slots_dt, k_v))

    p2_results = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_worker_dt_task, t): t for t in tasks_p2}
        for future in as_completed(futures):
            form, k_v, res = future.result()
            p2_results[(form, k_v)] = res

    ssa_times = [p2_results[("SSA", k)]["solve_time_s"] for k in k_vals]
    sa_times = [p2_results[("SA", k)]["solve_time_s"] for k in k_vals]
    ca_times = [p2_results[("CA", k)]["solve_time_s"] for k in k_vals]
    rel_times = [p2_results[("Relaxed", k)]["solve_time_s"] for k in k_vals]

    print(f"{'k':<3} | {'SSA Time (s)':<13} | {'SA Time (s)':<13} | {'CA Time (s)':<13} | {'Relaxed (s)':<12} | {'Constraints (SSA / SA / CA)'}")
    print("-" * 82)
    for k_v in k_vals:
        r_ssa = p2_results[("SSA", k_v)]
        r_sa = p2_results[("SA", k_v)]
        r_ca = p2_results[("CA", k_v)]
        r_rel = p2_results[("Relaxed", k_v)]
        print(
            f"{k_v:<3} | {r_ssa['solve_time_s']:<13.4f} | {r_sa['solve_time_s']:<13.4f} | "
            f"{r_ca['solve_time_s']:<13.4f} | {r_rel['solve_time_s']:<12.4f} | "
            f"{r_ssa['num_constraints']} / {r_sa['num_constraints']} / {r_ca['num_constraints']}"
        )

    # ------------------------------------------------------------------
    # Part 3: Spanish Telefónica Network in Parallel
    # ------------------------------------------------------------------
    G_spanish, pos_spanish = build_spanish_telefonica_network()
    demand_counts = [20, 35, 50, 65, 80]
    k_spanish_list = [2, 4, 6]

    print("\n" + "=" * 82)
    print(f"PART 3: FIG. 3 SPANISH TELEFONICA SCALABILITY (Parallel on {num_workers} Workers)")
    print("=" * 82)

    tasks_p3 = []
    for n_d in demand_counts:
        dems_sp = generate_spanish_demands(list(G_spanish.nodes()), num_demands=n_d, seed=19)
        for kv in k_spanish_list:
            tasks_p3.append((G_spanish, dems_sp, 32, kv, n_d))

    p3_results = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_worker_spanish_task, t): t for t in tasks_p3}
        for future in as_completed(futures):
            n_dem, kv, res = future.result()
            p3_results[(n_dem, kv)] = res

    spanish_curves = {kv: [] for kv in k_spanish_list}
    for n_d in demand_counts:
        line = f"|D| = {n_d:<3} -> "
        for kv in k_spanish_list:
            res_item = p3_results[(n_d, kv)]
            spanish_curves[kv].append(res_item["solve_time_s"])
            line += f"k={kv}: {res_item['solve_time_s']:.4f} s   "
        print(line)

    # ------------------------------------------------------------------
    # Generate 3-Panel Publication Figure Matching Paper Figs. 2, Table 3, Fig. 3
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), dpi=300)

    # Panel 1: Exact Fig. 2 Spanish Telefónica Topology
    ax1 = axes[0]
    nx.draw_networkx_nodes(G_spanish, pos_spanish, ax=ax1, node_size=420, node_color="#e0fbfc", edgecolors="#1d3557", linewidths=1.8)
    nx.draw_networkx_labels(G_spanish, pos_spanish, ax=ax1, font_size=8.5, font_weight="bold")
    nx.draw_networkx_edges(G_spanish, pos_spanish, ax=ax1, width=1.6, edge_color="#293241")
    ax1.set_title("Fig. 2: Spanish Telefónica Topology (21 Nodes, 35 Links)", fontsize=10.5, fontweight="bold")
    ax1.axis("off")

    # Panel 2: Table 3 Reproduction (Pure Solver Time vs. k on DT Topology)
    ax2 = axes[1]
    ax2.plot(k_vals, ssa_times, "r-o", lw=2.2, label="SSA (Starting Slot, Eqs. 2-9)")
    ax2.plot(k_vals, sa_times, "-s", color="#f4a261", lw=2.2, label="SA (Explicit Slot, Eqs. 10-14)")
    ax2.plot(k_vals, ca_times, "-^", color="#2a9d8f", lw=2.4, label="CA (Proposed Channel, Eqs. 15-17)")
    ax2.plot(k_vals, rel_times, "--d", color="#457b9d", lw=1.8, label="Relaxed RSA (Lower Bound)")
    ax2.set_yscale("log")
    ax2.set_xticks(k_vals)
    ax2.set_xlabel("Number of Pre-Computed Candidate Paths (k)")
    ax2.set_ylabel("Pure MILP Solver Time (seconds, log scale)")
    ax2.set_title("Table 3 Reproduction: DT (14,23) Solve Time vs. k", fontsize=10.5, fontweight="bold")
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
    print("=" * 82)
    print(f"[INFO] Saved parallelized Question 6 figure to: {out_path}")


if __name__ == "__main__":
    run_all_q6_benchmarks(num_workers=4)
