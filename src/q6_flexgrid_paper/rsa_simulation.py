"""
Question 6: Final Paper Implementation & Simulation
Paper: L. Velasco, M. Klinkowski, M. Ruiz, J. Comellas,
       "Modeling the routing and spectrum allocation problem for flexgrid optical networks,"
       Photonic Network Communications, vol. 24, pp. 177-186, 2012.

Implements:
  1. Pre-computed Contiguous Channel Generator C(d) (Table 1 of paper & report snippet)
  2. Exact Link-Path Formulations:
     - SSA (Starting Slot Assignment, Eqs. 2-9)
     - SA  (Slots Assignment, Eqs. 10-14)
     - LP-CA (Proposed Link-Path Channel Assignment, Eqs. 15-17)
     - Relaxed Link-Path RSA (Lower-Bound Relaxation, Eqs. 24-25)
  3. Benchmark 1: Solver Runtime vs. Candidate Paths k (SSA vs. SA vs. LP-CA)
  4. Benchmark 2: 21-Node, 35-Link Spanish Telefonica Network (Fig. 2 & Fig. 3 of paper)
                  with 1, 2, and 4-slot demands (10, 40, 100 Gbps) across increasing demand counts & k values.
Author: Erfan Ghasry
"""

import os
import time
import random
from itertools import islice
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# ==============================================================================
# 1. CHANNEL SET PRE-COMPUTATION (Table 1 of Paper & Original Report Code)
# ==============================================================================
def build_encoded_channels_for_traffic(traffic: list[list[int]], num_of_wavelengths: int = 8) -> tuple[list, list]:
    """
    Reproduces the exact channel generation and one-hot slot encoding logic
    from the project report for Table 1 of Velasco et al. (2012).
    Each traffic item is [source, destination, n_d_slots, b_d_gbps].
    """
    channels_set = []
    for d in range(len(traffic)):
        value = traffic[d][2]  # n_d: number of contiguous slots required
        # e.g., if value == 2 and |S| == 8 -> [[1,2], [2,3], ..., [7,8]]
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
# 2. TOPOLOGY DEFINITIONS (Spanish 21-Node Network & Reference Benchmark Graph)
# ==============================================================================
def build_spanish_telefonica_network() -> nx.Graph:
    """
    Builds the 21-Node, 35-Link Spanish Telefonica Optical Network Topology
    shown in Fig. 2 of Velasco et al. (2012).
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
    return G


def precompute_candidate_paths(G: nx.Graph, demands: list[dict], k: int) -> dict:
    """
    Computes up to k shortest simple paths for each demand d in D.
    Returns dict mapping d_idx -> list of path link-sets.
    """
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
# 3. ILP FORMULATIONS: LP-CA (Eqs. 15-17), SA (Eqs. 10-14), SSA (Eqs. 2-9), RELAXED
# ==============================================================================
def solve_lp_ca(G: nx.Graph, demands: list[dict], num_slots: int, k: int) -> dict:
    """
    Proposed Link-Path Channel Assignment (LP-CA) ILP Formulation (Eqs. 15-17).
    Pre-computes contiguous channels gamma_{c,s} to eliminate contiguity constraints.
    """
    t0 = time.perf_counter()
    paths_map = precompute_candidate_paths(G, demands, k)
    canonical_edges = [tuple(sorted(e)) for e in G.edges()]

    # Precompute encoded channels C(d) for each demand size n_d
    traffic_tuples = [[d["source"], d["target"], d["slots"], d["bw"]] for d in demands]
    _, encoded_channels_set = build_encoded_channels_for_traffic(traffic_tuples, num_of_wavelengths=num_slots)

    # Decision Variables:
    # x_d in {0, 1}: 1 if demand d is rejected (blocked)
    x = {d_idx: cp.Variable(boolean=True, name=f"x_{d_idx}") for d_idx in range(len(demands))}

    # y_{d, p, c} in {0, 1}: 1 if demand d uses path p and contiguous channel c
    y = {}
    for d_idx in range(len(demands)):
        for p_idx in range(len(paths_map[d_idx])):
            for c_idx in range(len(encoded_channels_set[d_idx])):
                y[(d_idx, p_idx, c_idx)] = cp.Variable(boolean=True, name=f"y_d{d_idx}_p{p_idx}_c{c_idx}")

    constraints = []

    # Eq. (16): sum_{p in P(d)} sum_{c in C(d)} y_{p,c} + x_d == 1
    for d_idx in range(len(demands)):
        p_c_vars = [
            y[(d_idx, p_idx, c_idx)]
            for p_idx in range(len(paths_map[d_idx]))
            for c_idx in range(len(encoded_channels_set[d_idx]))
        ]
        constraints.append(cp.sum(p_c_vars) + x[d_idx] == 1)

    # Eq. (17): Slot non-overlapping constraint on each link e and slot s
    # sum_{d} sum_{p} sum_{c} gamma_{c,s} * delta_{p,e} * y_{p,c} <= 1
    for e in canonical_edges:
        for s_idx in range(num_slots):
            slot_vars = []
            for d_idx in range(len(demands)):
                for p_idx, p_info in enumerate(paths_map[d_idx]):
                    if e in p_info["edges"]:
                        for c_idx, gamma_vec in enumerate(encoded_channels_set[d_idx]):
                            if gamma_vec[s_idx] == 1:
                                slot_vars.append(y[(d_idx, p_idx, c_idx)])
            if slot_vars:
                constraints.append(cp.sum(slot_vars) <= 1)

    # Eq. (1): Minimize un-served bandwidth Phi = sum_{d in D} x_d * b_d
    objective = cp.Minimize(cp.sum([x[d_idx] * demands[d_idx]["bw"] for d_idx in range(len(demands))]))
    problem = cp.Problem(objective, constraints)
    problem.solve()
    elapsed_s = time.perf_counter() - t0

    return {
        "formulation": "LP-CA",
        "status": problem.status,
        "unserved_bw_gbps": float(problem.value) if problem.value is not None else -1.0,
        "num_vars": len(x) + len(y),
        "num_constraints": len(constraints),
        "solve_time_s": elapsed_s,
    }


def solve_sa_formulation(G: nx.Graph, demands: list[dict], num_slots: int, k: int) -> dict:
    """
    Reference Slots Assignment (SA) ILP Formulation (Eqs. 10-14 of paper).
    Explicitly enforces contiguity via slot-transition constraints (Eq. 14).
    """
    t0 = time.perf_counter()
    paths_map = precompute_candidate_paths(G, demands, k)
    canonical_edges = [tuple(sorted(e)) for e in G.edges()]

    x = {d_idx: cp.Variable(boolean=True) for d_idx in range(len(demands))}
    y_p = {
        (d_idx, p_idx): cp.Variable(boolean=True)
        for d_idx in range(len(demands))
        for p_idx in range(len(paths_map[d_idx]))
    }
    y_ps = {
        (d_idx, p_idx, s_idx): cp.Variable(boolean=True)
        for d_idx in range(len(demands))
        for p_idx in range(len(paths_map[d_idx]))
        for s_idx in range(num_slots)
    }

    constraints = []

    # Eq. (3): Path selection or rejection
    for d_idx in range(len(demands)):
        constraints.append(cp.sum([y_p[(d_idx, p_idx)] for p_idx in range(len(paths_map[d_idx]))]) + x[d_idx] == 1)

    # Eq. (11): Number of assigned slots equals n_d * y_p
    for d_idx, dem in enumerate(demands):
        n_d = dem["slots"]
        for p_idx in range(len(paths_map[d_idx])):
            constraints.append(cp.sum([y_ps[(d_idx, p_idx, s)] for s in range(num_slots)]) == n_d * y_p[(d_idx, p_idx)])

    # Eq. (12): At most 1 demand per link e and slot s
    for e in canonical_edges:
        for s in range(num_slots):
            e_s_vars = [
                y_ps[(d_idx, p_idx, s)]
                for d_idx in range(len(demands))
                for p_idx, p_info in enumerate(paths_map[d_idx])
                if e in p_info["edges"]
            ]
            if e_s_vars:
                constraints.append(cp.sum(e_s_vars) <= 1)

    # Eq. (14): Explicit Spectrum Contiguity Constraint
    for d_idx in range(len(demands)):
        for p_idx in range(len(paths_map[d_idx])):
            for s in range(num_slots - 2):
                rem_slots = cp.sum([y_ps[(d_idx, p_idx, s2)] for s2 in range(s + 2, num_slots)])
                constraints.append(
                    -num_slots * (y_ps[(d_idx, p_idx, s)] - y_ps[(d_idx, p_idx, s + 1)] - 1) >= rem_slots
                )

    objective = cp.Minimize(cp.sum([x[d_idx] * demands[d_idx]["bw"] for d_idx in range(len(demands))]))
    problem = cp.Problem(objective, constraints)
    problem.solve()
    elapsed_s = time.perf_counter() - t0

    return {
        "formulation": "SA",
        "status": problem.status,
        "unserved_bw_gbps": float(problem.value) if problem.value is not None else -1.0,
        "num_vars": len(x) + len(y_p) + len(y_ps),
        "num_constraints": len(constraints),
        "solve_time_s": elapsed_s,
    }


def solve_ssa_formulation(G: nx.Graph, demands: list[dict], num_slots: int, k: int) -> dict:
    """
    Reference Starting Slot Assignment (SSA) ILP Formulation (Eqs. 2-9 of paper).
    Uses starting slot integers f_d and pairwise ordering variables f_{d1, d2}.
    """
    t0 = time.perf_counter()
    paths_map = precompute_candidate_paths(G, demands, k)
    D_range = range(len(demands))

    x = {d: cp.Variable(boolean=True) for d in D_range}
    y_p = {(d, p): cp.Variable(boolean=True) for d in D_range for p in range(len(paths_map[d]))}
    f_d = {d: cp.Variable(integer=True) for d in D_range}
    f_order = {}

    constraints = [f_d[d] >= 0 for d in D_range]

    for d in D_range:
        constraints.append(cp.sum([y_p[(d, p)] for p in range(len(paths_map[d]))]) + x[d] == 1)
        constraints.append(f_d[d] + demands[d]["slots"] * (1 - x[d]) <= num_slots)

    # Pairwise conflict constraints (Eqs. 5-9)
    for d1 in D_range:
        for d2 in D_range:
            if d1 >= d2:
                continue
            # Check if any candidate paths share a link
            shares_link = any(
                len(p1["edges"].intersection(p2["edges"])) > 0
                for p1 in paths_map[d1]
                for p2 in paths_map[d2]
            )
            if shares_link:
                f_order[(d1, d2)] = cp.Variable(boolean=True)
                f_order[(d2, d1)] = cp.Variable(boolean=True)
                constraints.append(f_order[(d1, d2)] + f_order[(d2, d1)] == 1)
                constraints.append(f_d[d2] - f_d[d1] <= num_slots * f_order[(d1, d2)])
                constraints.append(f_d[d1] - f_d[d2] <= num_slots * f_order[(d2, d1)])

                for p1_idx, p1 in enumerate(paths_map[d1]):
                    for p2_idx, p2 in enumerate(paths_map[d2]):
                        if len(p1["edges"].intersection(p2["edges"])) > 0:
                            constraints.append(
                                f_d[d1] + demands[d1]["slots"] * y_p[(d1, p1_idx)] - f_d[d2]
                                <= num_slots * (1 - f_order[(d1, d2)] + 2 - y_p[(d1, p1_idx)] - y_p[(d2, p2_idx)])
                            )
                            constraints.append(
                                f_d[d2] + demands[d2]["slots"] * y_p[(d2, p2_idx)] - f_d[d1]
                                <= num_slots * (1 - f_order[(d2, d1)] + 2 - y_p[(d1, p1_idx)] - y_p[(d2, p2_idx)])
                            )

    objective = cp.Minimize(cp.sum([x[d] * demands[d]["bw"] for d in D_range]))
    problem = cp.Problem(objective, constraints)
    problem.solve()
    elapsed_s = time.perf_counter() - t0

    return {
        "formulation": "SSA",
        "status": problem.status,
        "unserved_bw_gbps": float(problem.value) if problem.value is not None else -1.0,
        "num_vars": len(x) + len(y_p) + len(f_d) + len(f_order),
        "num_constraints": len(constraints),
        "solve_time_s": elapsed_s,
    }


def solve_relaxed_lp_rsa(G: nx.Graph, demands: list[dict], num_slots: int, k: int) -> dict:
    """
    Relaxed Link-Path RSA Formulation (Eqs. 24-25 of paper).
    Relaxes continuity and contiguity to compute a fast lower bound.
    """
    t0 = time.perf_counter()
    paths_map = precompute_candidate_paths(G, demands, k)
    canonical_edges = [tuple(sorted(e)) for e in G.edges()]
    D_range = range(len(demands))

    x = {d: cp.Variable(boolean=True) for d in D_range}
    y_p = {(d, p): cp.Variable(boolean=True) for d in D_range for p in range(len(paths_map[d]))}

    constraints = []
    for d in D_range:
        constraints.append(cp.sum([y_p[(d, p)] for p in range(len(paths_map[d]))]) + x[d] == 1)

    # Eq. (25): Aggregate slot demand on link e <= |S|
    for e in canonical_edges:
        link_load = [
            demands[d]["slots"] * y_p[(d, p_idx)]
            for d in D_range
            for p_idx, p_info in enumerate(paths_map[d])
            if e in p_info["edges"]
        ]
        if link_load:
            constraints.append(cp.sum(link_load) <= num_slots)

    objective = cp.Minimize(cp.sum([x[d] * demands[d]["bw"] for d in D_range]))
    problem = cp.Problem(objective, constraints)
    problem.solve()
    elapsed_s = time.perf_counter() - t0

    return {
        "formulation": "Relaxed-RSA",
        "status": problem.status,
        "unserved_bw_gbps": float(problem.value) if problem.value is not None else -1.0,
        "num_vars": len(x) + len(y_p),
        "num_constraints": len(constraints),
        "solve_time_s": elapsed_s,
    }


# ==============================================================================
# 4. EXPERIMENTS & PLOTTING (Reproducing Report Figures & Paper Benchmarks)
# ==============================================================================
def generate_flexgrid_demands(nodes: list[int], num_demands: int, seed: int = 42) -> list[dict]:
    """
    Generates random demands following Sect. 4.3 of Velasco et al. (2012):
    Bandwidth is uniformly chosen from {10 Gbps (1 slot), 40 Gbps (2 slots), 100 Gbps (4 slots)}.
    """
    rng = random.Random(seed)
    slot_bw_choices = [(1, 10), (2, 40), (4, 100)]
    demands = []
    for _ in range(num_demands):
        s, t = rng.sample(nodes, 2)
        n_slots, bw_gbps = rng.choice(slot_bw_choices)
        demands.append({"source": s, "target": t, "slots": n_slots, "bw": bw_gbps})
    return demands


def run_all_q6_benchmarks(output_dir: str = "results/figures"):
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Part 1: Verify Channel Set Encoding Snippet from Original Report
    # ------------------------------------------------------------------
    sample_traffic = [[1, 2, 2, 40], [2, 3, 2, 40], [1, 4, 4, 100]]
    ch_set, enc_set = build_encoded_channels_for_traffic(sample_traffic, num_of_wavelengths=8)
    print("=" * 78)
    print("QUESTION 6 - PART 1: CONTIGUOUS CHANNEL SET PRE-COMPUTATION (|S|=8, n_d=2)")
    print("=" * 78)
    print(f"channels_set[1] (Contiguous Slot Indices):\n  {ch_set[1]}")
    print("encoded_channels_set[1] (Binary Coincidence Matrix gamma_{c,s}):")
    for row in enc_set[1]:
        print(f"  {row}")

    # ------------------------------------------------------------------
    # Part 2: Compare SSA vs. SA vs. LP-CA vs. Relaxed RSA across k=1..4
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("QUESTION 6 - PART 2: MODEL COMPLEXITY & SOLVE TIME (SSA vs. SA vs. LP-CA)")
    print("=" * 78)
    G_spanish = build_spanish_telefonica_network()
    bench_demands = generate_flexgrid_demands(list(G_spanish.nodes()), num_demands=14, seed=10)
    k_list_comp = [1, 2, 3, 4]
    times_ssa, times_sa, times_ca, times_rel = [], [], [], []

    print(f"{'k':<4} | {'SSA Time (s)':<14} | {'SA Time (s)':<14} | {'LP-CA Time (s)':<15} | {'Relaxed (s)':<12} | {'LP-CA Constraints'}")
    print("-" * 78)
    for k_val in k_list_comp:
        r_ssa = solve_ssa_formulation(G_spanish, bench_demands, num_slots=16, k=k_val)
        r_sa = solve_sa_formulation(G_spanish, bench_demands, num_slots=16, k=k_val)
        r_ca = solve_lp_ca(G_spanish, bench_demands, num_slots=16, k=k_val)
        r_rel = solve_relaxed_lp_rsa(G_spanish, bench_demands, num_slots=16, k=k_val)

        times_ssa.append(r_ssa["solve_time_s"] * 1000)
        times_sa.append(r_sa["solve_time_s"] * 1000)
        times_ca.append(r_ca["solve_time_s"] * 1000)
        times_rel.append(r_rel["solve_time_s"] * 1000)

        print(
            f"{k_val:<4} | {r_ssa['solve_time_s']:<14.4f} | {r_sa['solve_time_s']:<14.4f} | "
            f"{r_ca['solve_time_s']:<15.4f} | {r_rel['solve_time_s']:<12.4f} | {r_ca['num_constraints']}"
        )

    # ------------------------------------------------------------------
    # Part 3: Spanish 21-Node Network Scalability (Solve Time vs. Demands for k=2, 4, 6)
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("QUESTION 6 - PART 3: SPANISH TELEFONICA NETWORK (SOLVE TIME VS. DEMANDS & k)")
    print("=" * 78)
    demand_counts = [15, 25, 35, 45, 55, 65]
    k_curves = [2, 4, 6]
    spanish_times = {k_v: [] for k_v in k_curves}

    for n_dem in demand_counts:
        dems = generate_flexgrid_demands(list(G_spanish.nodes()), num_demands=n_dem, seed=42)
        row_str = f"Demands = {n_dem:<3} -> "
        for k_v in k_curves:
            res_ca = solve_lp_ca(G_spanish, dems, num_slots=32, k=k_v)
            spanish_times[k_v].append(res_ca["solve_time_s"] * 1000)
            row_str += f"k={k_v}: {res_ca['solve_time_s']*1000:.1f} ms (Blocked={res_ca['unserved_bw_gbps']:.0f}G)  "
        print(row_str)

    # ------------------------------------------------------------------
    # Plotting Figure: 3-Panel Summary for Report & GitHub
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=300)

    # Panel 1: 21-Node Spanish Telefonica Topology (Fig. 2 of Paper)
    ax_topo = axes[0]
    pos = nx.spring_layout(G_spanish, seed=19)
    nx.draw_networkx_nodes(G_spanish, pos, ax=ax_topo, node_size=360, node_color="#e0fbfc", edgecolors="#1d3557", linewidths=1.5)
    nx.draw_networkx_labels(G_spanish, pos, ax=ax_topo, font_size=8, font_weight="bold")
    nx.draw_networkx_edges(G_spanish, pos, ax=ax_topo, width=1.4, edge_color="#457b9d")
    ax_topo.set_title("Spanish Telefónica Topology (21 Nodes, 35 Links)", fontsize=10.5, fontweight="bold")
    ax_topo.axis("off")

    # Panel 2: Formulation Comparison (SSA vs SA vs LP-CA vs Relaxed) as function of k
    ax_comp = axes[1]
    ax_comp.plot(k_list_comp, times_ssa, "r-o", lw=2, label="SSA (Starting Slot)")
    ax_comp.plot(k_list_comp, times_sa, "y-s", lw=2, color="#f4a261", label="SA (Explicit Slot)")
    ax_comp.plot(k_list_comp, times_ca, "g-^", lw=2, color="#2a9d8f", label="LP-CA (Proposed Channel)")
    ax_comp.plot(k_list_comp, times_rel, "b--d", lw=1.6, color="#457b9d", label="Relaxed RSA (Lower Bound)")
    ax_comp.set_yscale("log")
    ax_comp.set_xticks(k_list_comp)
    ax_comp.set_xlabel("Number of Candidate Paths (k)")
    ax_comp.set_ylabel("Computation Time (ms, log scale)")
    ax_comp.set_title("Table 3 Reproduction: Solve Time vs. k", fontsize=10.5, fontweight="bold")
    ax_comp.grid(True, which="both", linestyle=":", alpha=0.6)
    ax_comp.legend(fontsize=8.5)

    # Panel 3: Fig. 3 Reproduction (Spanish Network Computation Time vs Number of Demands)
    ax_fig3 = axes[2]
    markers = {2: "s", 4: "^", 6: "o"}
    colors = {2: "#457b9d", 4: "#2a9d8f", 6: "#e63946"}
    for k_v in k_curves:
        ax_fig3.plot(
            demand_counts, spanish_times[k_v],
            marker=markers[k_v], color=colors[k_v], lw=2, label=f"LP-CA (k={k_v})"
        )
    ax_fig3.set_xlabel("Number of Demands (|D|)")
    ax_fig3.set_ylabel("Computation Time (ms)")
    ax_fig3.set_title("Fig. 3 Reproduction: Spanish Network Scalability", fontsize=10.5, fontweight="bold")
    ax_fig3.grid(True, linestyle=":", alpha=0.6)
    ax_fig3.legend(fontsize=9)

    out_path = os.path.join(output_dir, "q6_flexgrid_velasco_results.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("=" * 78)
    print(f"[INFO] Saved Question 6 Paper Reproduction plots to: {out_path}")


if __name__ == "__main__":
    run_all_q6_benchmarks()
