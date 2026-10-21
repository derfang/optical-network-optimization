"""
Question 4: Manycast Routing and Wavelength Assignment (Manycast RWA) ILP
Compares three path generation strategies from the project report:
  1. Single Best Path Only (K=1 shortest distance path, ignoring wavelength congestion) -> W_high = 5
  2. Yen's K-Shortest Weighted Paths (K=2 / K=3 balanced trade-off) -> W_high = 5 or 4
  3. All Possible Simple Paths (Ignoring distance) -> Explodes variable count & solver runtime
Author: Erfan Ghasry
"""

import os
import time
from itertools import islice
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from src.utils.plotting import (
    plot_composite_lightpaths_on_graph,
    plot_wavelength_planes,
)

NODES = [1, 2, 3, 4, 5, 6]

WEIGHTED_EDGES = [
    (1, 2, 23),
    (1, 3, 32),
    (1, 4, 48),
    (2, 3, 30),
    (2, 5, 15),
    (3, 4, 35),
    (3, 6, 200),
    (4, 6, 30),
    (5, 6, 13),
]

NODE_POSITIONS = {
    1: (0.0, 0.5),
    2: (0.8, 1.1),
    3: (1.4, 0.5),
    4: (1.0, -0.2),
    5: (2.1, 1.1),
    6: (2.5, 0.35),
}

MANYCAST_REQUESTS = [
    {"id": 1, "source": 1, "candidates": [2, 5, 6],    "k_dest": 2, "traffic_vol": 1},
    {"id": 2, "source": 3, "candidates": [1, 2, 5, 6], "k_dest": 3, "traffic_vol": 2},
    {"id": 3, "source": 4, "candidates": [2, 3, 5],    "k_dest": 2, "traffic_vol": 1},
    {"id": 4, "source": 5, "candidates": [1, 3, 4],    "k_dest": 2, "traffic_vol": 2},
    {"id": 5, "source": 6, "candidates": [2, 3, 4],    "k_dest": 2, "traffic_vol": 1},
]


def build_q4_graph() -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(NODES)
    for u, v, length in WEIGHTED_EDGES:
        G.add_edge(u, v, weight=length)
    return G


def generate_candidate_paths(G: nx.Graph, source: int, target: int, mode: str = "k_shortest", k: int = 2) -> list[dict]:
    """
    Generates candidate paths between source and target based on the selected strategy:
      - 'best_only': Only the single shortest path (K=1) by physical distance (km).
      - 'k_shortest': Up to K shortest weighted paths avoiding the 200km link unless necessary.
      - 'all_paths': All possible simple paths in the graph (ignores distance).
    """
    if mode == "best_only":
        raw_paths = [nx.shortest_path(G, source, target, weight="weight")]
    elif mode == "k_shortest":
        raw_paths = list(islice(nx.shortest_simple_paths(G, source, target, weight="weight"), k))
    elif mode == "all_paths":
        raw_paths = list(nx.all_simple_paths(G, source, target))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    paths_info = []
    for idx, p in enumerate(raw_paths):
        dist = sum(G[p[i]][p[i + 1]]["weight"] for i in range(len(p) - 1))
        links = {tuple(sorted((p[i], p[i + 1]))) for i in range(len(p) - 1)}
        paths_info.append({
            "k_idx": idx,
            "path": p,
            "distance_km": dist,
            "links": links,
        })
    return paths_info


def solve_manycast_rwa(mode: str = "k_shortest", k: int = 2, max_wavelengths: int = 8) -> dict:
    """
    Solves the Manycast RWA ILP for the specified path generation mode and records
    the total number of decision variables and solver execution time.
    """
    t0 = time.perf_counter()
    G = build_q4_graph()
    W = list(range(1, max_wavelengths + 1))
    canonical_edges = [tuple(sorted((u, v))) for (u, v, _) in WEIGHTED_EDGES]

    candidate_paths = {}
    total_candidate_paths = 0
    for req in MANYCAST_REQUESTS:
        r_id = req["id"]
        s = req["source"]
        for d in req["candidates"]:
            plist = generate_candidate_paths(G, s, d, mode=mode, k=k)
            candidate_paths[(r_id, d)] = plist
            total_candidate_paths += len(plist)

    # Decision Variables
    delta = {
        (req["id"], d): cp.Variable(boolean=True, name=f"delta_r{req['id']}_d{d}")
        for req in MANYCAST_REQUESTS
        for d in req["candidates"]
    }

    c = {}
    for (r_id, d), plist in candidate_paths.items():
        for p_info in plist:
            m_idx = p_info["k_idx"]
            for w in W:
                c[(r_id, d, m_idx, w)] = cp.Variable(
                    boolean=True, name=f"c_r{r_id}_d{d}_m{m_idx}_w{w}"
                )

    U = {w: cp.Variable(boolean=True, name=f"U_w{w}") for w in W}
    W_high = cp.Variable(integer=True, name="W_high")

    num_binary_vars = len(delta) + len(c) + len(U)
    constraints = [W_high >= 0]

    # 1. Select exactly k_r destinations out of D_r
    for req in MANYCAST_REQUESTS:
        r_id = req["id"]
        constraints.append(
            cp.sum([delta[(r_id, d)] for d in req["candidates"]]) == req["k_dest"]
        )

    # 2. Allocate tau_r lightpaths for each selected destination
    for req in MANYCAST_REQUESTS:
        r_id = req["id"]
        tau_r = req["traffic_vol"]
        for d in req["candidates"]:
            plist = candidate_paths[(r_id, d)]
            alloc_vars = [c[(r_id, d, p_info["k_idx"], w)] for p_info in plist for w in W]
            constraints.append(cp.sum(alloc_vars) == tau_r * delta[(r_id, d)])

    # 3. Wavelength Clash Constraint on Physical Links
    for edge in canonical_edges:
        for w in W:
            edge_vars = [
                c[(r_id, d, p_info["k_idx"], w)]
                for (r_id, d), plist in candidate_paths.items()
                for p_info in plist
                if edge in p_info["links"]
            ]
            if edge_vars:
                constraints.append(cp.sum(edge_vars) <= 1)

    # 4. Wavelength Activation & W_high
    for w in W:
        for key, var in c.items():
            if key[3] == w:
                constraints.append(U[w] >= var)
        constraints.append(W_high >= w * U[w])

    # Objective: Minimize weighted wavelength usage + small distance weight (unless 'all_paths' ignores distance)
    wavelength_obj = cp.sum([w * U[w] for w in W])
    if mode != "all_paths":
        dist_obj = cp.sum([
            0.001 * p_info["distance_km"] * c[(r_id, d, p_info["k_idx"], w)]
            for (r_id, d), plist in candidate_paths.items()
            for p_info in plist
            for w in W
        ])
        objective = cp.Minimize(wavelength_obj + dist_obj)
    else:
        objective = cp.Minimize(wavelength_obj)

    problem = cp.Problem(objective, constraints)
    problem.solve()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    selected_destinations = {}
    lightpaths = []
    used_wavelengths = set()

    if problem.status in ["optimal", "optimal_inaccurate"]:
        for req in MANYCAST_REQUESTS:
            r_id = req["id"]
            s = req["source"]
            chosen_d = [
                d for d in req["candidates"]
                if delta[(r_id, d)].value is not None and round(float(delta[(r_id, d)].value)) == 1
            ]
            selected_destinations[r_id] = chosen_d

            for d in chosen_d:
                for p_info in candidate_paths[(r_id, d)]:
                    m_idx = p_info["k_idx"]
                    for w in W:
                        val = c[(r_id, d, m_idx, w)].value
                        if val is not None and round(float(val)) == 1:
                            used_wavelengths.add(w)
                            lightpaths.append({
                                "request_id": r_id,
                                "source": s,
                                "destination": d,
                                "path": p_info["path"],
                                "distance_km": p_info["distance_km"],
                                "wavelength": w,
                            })

    max_w = max(used_wavelengths) if used_wavelengths else 0
    return {
        "mode": mode,
        "k": k,
        "status": problem.status,
        "num_candidate_paths": total_candidate_paths,
        "num_binary_vars": num_binary_vars,
        "max_wavelength": max_w,
        "weighted_index_sum": sum(used_wavelengths),
        "selected_destinations": selected_destinations,
        "lightpaths": lightpaths,
        "total_distance_km": sum(lp["distance_km"] for lp in lightpaths),
        "elapsed_ms": elapsed_ms,
    }


def plot_q4_comparison(results: dict, output_dir: str = "results/figures"):
    """Plots comparative metrics for the 3 path routing candidate modes."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=300)

    modes = list(results.keys())
    labels = [
        "1. Single Best (K=1)\n[Shortest Distance]",
        "2. K-Shortest (K=2)\n[Balanced Design]",
        "3. All Simple Paths\n[Unconstrained]",
    ]

    var_counts = [results[m]["num_binary_vars"] for m in modes]
    runtimes = [results[m]["elapsed_ms"] for m in modes]
    w_highs = [results[m]["max_wavelength"] for m in modes]
    distances = [results[m]["total_distance_km"] for m in modes]

    # Panel 1: Decision Variable Count & Candidate Paths
    ax1 = axes[0]
    bars1 = ax1.bar(labels, var_counts, color=["#457b9d", "#2a9d8f", "#e76f51"], width=0.45, edgecolor="#293241")
    ax1.set_ylabel("Total Binary Decision Variables", fontweight="bold")
    ax1.set_title("Model Complexity (Variable Count)", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    for b in bars1:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 15, f"{int(b.get_height())}", ha="center", fontweight="bold")

    # Panel 2: Max Wavelength Index & Total Distance
    ax2 = axes[1]
    x = np.arange(len(modes))
    width = 0.32
    rects_w = ax2.bar(x - width / 2, w_highs, width, label="Max Wavelength ($W_{high}$)", color="#e63946", edgecolor="#293241")
    ax2.set_ylabel("Max Wavelength Index ($W_{high}$)", fontweight="bold", color="#e63946")
    ax2.set_title("Spectrum & Total Distance Comparison", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylim(0, max(w_highs) + 2)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    for b in rects_w:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"{int(b.get_height())}", ha="center", fontweight="bold")

    ax2_twin = ax2.twinx()
    rects_d = ax2_twin.bar(x + width / 2, distances, width, label="Distance (km)", color="#457b9d", alpha=0.7, edgecolor="#293241")
    ax2_twin.set_ylabel("Total Network Distance (km)", fontweight="bold", color="#457b9d")
    ax2_twin.set_ylim(0, max(distances) * 1.3)
    for b in rects_d:
        ax2_twin.text(b.get_x() + b.get_width() / 2, b.get_height() + 15, f"{int(b.get_height())}km", ha="center", fontsize=8)

    # Panel 3: Solver Runtime
    ax3 = axes[2]
    bars3 = ax3.bar(labels, runtimes, color=["#a8dadc", "#2a9d8f", "#f4a261"], width=0.45, edgecolor="#293241")
    ax3.set_ylabel("Solver Execution Time (ms)", fontweight="bold")
    ax3.set_title("Solver Runtime Overhead", fontsize=11, fontweight="bold")
    ax3.grid(axis="y", linestyle=":", alpha=0.6)
    for b in bars3:
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + (max(runtimes) * 0.02), f"{b.get_height():.1f} ms", ha="center", fontweight="bold")

    out_path = os.path.join(output_dir, "q4_manycast_comparison.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved Q4 comparison plot to: {out_path}")


if __name__ == "__main__":
    modes_to_test = [
        ("best_only", 1, "Single Best Path (K=1)"),
        ("k_shortest", 2, "K-Shortest Paths (K=2)"),
        ("all_paths", None, "All Simple Paths"),
    ]

    results = {}
    print("=" * 80)
    print("QUESTION 4: MANYCAST RWA MULTI-MODE STRATEGY COMPARISON")
    print("=" * 80)

    for mode, k_val, desc in modes_to_test:
        print(f"\n>>> Solving Mode: {desc} ...")
        res = solve_manycast_rwa(mode=mode, k=k_val if k_val else 2, max_wavelengths=8)
        results[mode] = res
        print(f"    Status: {res['status']} | Paths: {res['num_candidate_paths']} | Variables: {res['num_binary_vars']}")
        print(f"    W_high: {res['max_wavelength']} | Total Distance: {res['total_distance_km']} km | Time: {res['elapsed_ms']:.1f} ms")
        print("    Selected Destinations:")
        for r_id, d_list in res["selected_destinations"].items():
            print(f"      Request #{r_id}: {d_list}")

    plot_q4_comparison(results)

    # Save detailed lightpaths for the balanced K-Shortest (K=2) mode
    G = build_q4_graph()
    k_res = results["k_shortest"]
    routes_formatted = [
        {"demand": (lp["source"], lp["destination"]), "wavelength": lp["wavelength"], "path": lp["path"]}
        for lp in k_res["lightpaths"]
    ]

    plot_composite_lightpaths_on_graph(
        G,
        NODE_POSITIONS,
        routes_formatted,
        title=f"Q4 Manycast RWA: Established Lightpaths (K=2, W_high = {k_res['max_wavelength']})",
        save_path="results/figures/q4_manycast_graph_lightpaths.png",
    )
    plot_wavelength_planes(
        G,
        NODE_POSITIONS,
        routes_formatted,
        num_wavelengths=k_res["max_wavelength"],
        title=f"Q4 Manycast RWA: Wavelength Planes (W1 - W{k_res['max_wavelength']})",
        save_path="results/figures/q4_manycast_wavelength_planes.png",
    )
