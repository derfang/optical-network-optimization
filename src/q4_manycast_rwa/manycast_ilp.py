"""
Question 4: Manycast Routing and Wavelength Assignment (Manycast RWA) ILP
Selects optimal destination subsets D_r (size k_r) for each source s_r and allocates
paths and wavelengths for the required traffic volume tau_r.
Author: Erfan Ghasry
"""

import os
from itertools import islice
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from src.utils.plotting import (
    plot_composite_lightpaths_on_graph,
    plot_wavelength_planes,
)

# 1. 6-Node Weighted Bidirectional Topology (Link lengths in km)
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

# 2. Manycast Requests Table from Question 4
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


def get_k_shortest_paths_weighted(G: nx.Graph, source: int, target: int, k: int = 3) -> list[dict]:
    """
    Returns up to K shortest simple paths (Yen's algorithm) ordered by physical distance (km).
    """
    paths_info = []
    try:
        raw_paths = list(islice(nx.shortest_simple_paths(G, source, target, weight="weight"), k))
        for idx, p in enumerate(raw_paths):
            dist = sum(G[p[i]][p[i + 1]]["weight"] for i in range(len(p) - 1))
            links = {tuple(sorted((p[i], p[i + 1]))) for i in range(len(p) - 1)}
            paths_info.append({
                "k_idx": idx,
                "path": p,
                "distance_km": dist,
                "links": links,
            })
    except nx.NetworkXNoPath:
        pass
    return paths_info


def solve_manycast_rwa_ilp(
    max_wavelengths: int = 10,
    k_paths: int = 3,
    undirected_capacity: bool = True,
    distance_tiebreak: bool = True,
) -> dict:
    """
    Solves the Manycast RWA ILP problem.
    Determines:
      1. Selected destination nodes for each request (delta_{r, d})
      2. Assigned path and wavelength(s) for each selected destination (c_{r, d, m, w})
    """
    G = build_q4_graph()
    W = list(range(1, max_wavelengths + 1))
    canonical_edges = [tuple(sorted((u, v))) for (u, v, _) in WEIGHTED_EDGES]
    directed_arcs = [(u, v) for (u, v, _) in WEIGHTED_EDGES] + [(v, u) for (u, v, _) in WEIGHTED_EDGES]

    # Precompute K shortest weighted candidate paths for every (source, candidate_dest)
    candidate_paths = {}
    for req in MANYCAST_REQUESTS:
        r_id = req["id"]
        s = req["source"]
        for d in req["candidates"]:
            candidate_paths[(r_id, d)] = get_k_shortest_paths_weighted(G, s, d, k=k_paths)

    # Decision Variables:
    # delta[(r_id, d)] in {0, 1}: 1 if candidate destination d is chosen for request r_id
    delta = {
        (req["id"], d): cp.Variable(boolean=True, name=f"delta_r{req['id']}_d{d}")
        for req in MANYCAST_REQUESTS
        for d in req["candidates"]
    }

    # c[(r_id, d, m_idx, w)] in {0, 1}: 1 if wavelength w is assigned to path m_idx for (r_id, d)
    c = {}
    for (r_id, d), plist in candidate_paths.items():
        for p_info in plist:
            m_idx = p_info["k_idx"]
            for w in W:
                c[(r_id, d, m_idx, w)] = cp.Variable(
                    boolean=True, name=f"c_r{r_id}_d{d}_m{m_idx}_w{w}"
                )

    # U[w] in {0, 1}: 1 if wavelength w is used anywhere in the network
    U = {w: cp.Variable(boolean=True, name=f"U_w{w}") for w in W}
    W_high = cp.Variable(integer=True, name="W_high")

    constraints = [W_high >= 0]

    # 1. Manycast Destination Count Constraint: sum_{d in D_r} delta_{r,d} == k_r
    for req in MANYCAST_REQUESTS:
        r_id = req["id"]
        constraints.append(
            cp.sum([delta[(r_id, d)] for d in req["candidates"]]) == req["k_dest"]
        )

    # 2. Traffic Volume Satisfaction Constraint:
    # If delta_{r,d} == 1, allocate exactly tau_r wavelengths across candidate paths
    for req in MANYCAST_REQUESTS:
        r_id = req["id"]
        tau_r = req["traffic_vol"]
        for d in req["candidates"]:
            plist = candidate_paths[(r_id, d)]
            alloc_vars = [c[(r_id, d, p_info["k_idx"], w)] for p_info in plist for w in W]
            constraints.append(cp.sum(alloc_vars) == tau_r * delta[(r_id, d)])

    # 3. Wavelength Clash Constraint on Physical Links
    if undirected_capacity:
        for edge in canonical_edges:
            for w in W:
                edge_vars = []
                for (r_id, d), plist in candidate_paths.items():
                    for p_info in plist:
                        if edge in p_info["links"]:
                            edge_vars.append(c[(r_id, d, p_info["k_idx"], w)])
                if edge_vars:
                    constraints.append(cp.sum(edge_vars) <= 1)
    else:
        for arc in directed_arcs:
            for w in W:
                arc_vars = []
                for (r_id, d), plist in candidate_paths.items():
                    for p_info in plist:
                        p = p_info["path"]
                        p_arcs = {(p[i], p[i + 1]) for i in range(len(p) - 1)}
                        if arc in p_arcs:
                            arc_vars.append(c[(r_id, d, p_info["k_idx"], w)])
                if arc_vars:
                    constraints.append(cp.sum(arc_vars) <= 1)

    # 4. Link Wavelength Usage U[w] and W_high
    for w in W:
        for key, var in c.items():
            if key[3] == w:
                constraints.append(U[w] >= var)
        constraints.append(W_high >= w * U[w])

    # Objective: Minimize weighted wavelength index (plus tiny distance tie-breaker for shortest km paths)
    wavelength_cost = cp.sum([w * U[w] for w in W])
    if distance_tiebreak:
        dist_cost = cp.sum([
            0.0001 * p_info["distance_km"] * c[(r_id, d, p_info["k_idx"], w)]
            for (r_id, d), plist in candidate_paths.items()
            for p_info in plist
            for w in W
        ])
        objective = cp.Minimize(wavelength_cost + dist_cost)
    else:
        objective = cp.Minimize(wavelength_cost)

    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Extract Solution
    selected_destinations = {}
    lightpaths = []
    used_wavelengths = set()

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
                            "traffic_vol_req": req["traffic_vol"],
                            "path": p_info["path"],
                            "distance_km": p_info["distance_km"],
                            "wavelength": w,
                        })

    max_w = max(used_wavelengths) if used_wavelengths else 0
    return {
        "status": problem.status,
        "max_wavelength": max_w,
        "weighted_index_sum": sum(used_wavelengths),
        "selected_destinations": selected_destinations,
        "lightpaths": lightpaths,
        "total_distance_km": sum(lp["distance_km"] for lp in lightpaths),
    }


def plot_q4_topology_and_allocations(result: dict, output_dir: str = "results/figures"):
    os.makedirs(output_dir, exist_ok=True)
    G = build_q4_graph()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Left: Weighted Physical Topology
    ax_topo = axes[0]
    nx.draw_networkx_nodes(G, NODE_POSITIONS, ax=ax_topo, node_size=850, node_color="#e0fbfc", edgecolors="#293241", linewidths=2)
    nx.draw_networkx_labels(G, NODE_POSITIONS, ax=ax_topo, font_size=11, font_weight="bold")
    nx.draw_networkx_edges(G, NODE_POSITIONS, ax=ax_topo, width=2.0, edge_color="#3d5a80")
    edge_labels = {(u, v): f"{w} km" for (u, v, w) in WEIGHTED_EDGES}
    nx.draw_networkx_edge_labels(G, NODE_POSITIONS, edge_labels=edge_labels, ax=ax_topo, font_size=8.5, font_weight="bold")
    ax_topo.set_title("Q4: 6-Node Weighted Optical Network Topology", fontsize=12, fontweight="bold")
    ax_topo.axis("off")

    # Right: Link vs. Wavelength Utilization Matrix
    ax_grid = axes[1]
    max_w = result["max_wavelength"]
    edges = [tuple(sorted((u, v))) for (u, v, _) in WEIGHTED_EDGES]
    grid = np.zeros((len(edges), max_w), dtype=int)
    text_grid = [["" for _ in range(max_w)] for _ in range(len(edges))]

    for lp in result["lightpaths"]:
        s, d, w = lp["source"], lp["destination"], lp["wavelength"] - 1
        p = lp["path"]
        for i in range(len(p) - 1):
            e = tuple(sorted((p[i], p[i + 1])))
            e_idx = edges.index(e)
            grid[e_idx, w] = 1
            text_grid[e_idx][w] = f"{s}->{d}"

    cmap = plt.get_cmap("YlGnBu", 2)
    ax_grid.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=1.3)
    for i in range(len(edges)):
        for j in range(max_w):
            txt = text_grid[i][j] if grid[i, j] == 1 else "-"
            color = "white" if grid[i, j] == 1 else "#adb5bd"
            weight = "bold" if grid[i, j] == 1 else "normal"
            ax_grid.text(j, i, txt, ha="center", va="center", color=color, fontsize=8, fontweight=weight)

    ax_grid.set_xticks(range(max_w))
    ax_grid.set_xticklabels([f"W{w}" for w in range(1, max_w + 1)], fontweight="bold")
    ax_grid.set_yticks(range(len(edges)))
    ax_grid.set_yticklabels([f"Link {e}" for e in edges], fontweight="bold")
    ax_grid.set_xlabel("Wavelength Index (w)")
    ax_grid.set_ylabel("Physical Link")
    ax_grid.set_title(f"Q4: Manycast RWA Spectrum Allocation (W_high = {max_w})", fontsize=12, fontweight="bold")

    out_path = os.path.join(output_dir, "q4_manycast_rwa.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved Q4 Manycast RWA figure to: {out_path}")


if __name__ == "__main__":
    res = solve_manycast_rwa_ilp(max_wavelengths=10, k_paths=3, undirected_capacity=True)

    print("=" * 78)
    print("QUESTION 4: MANYCAST RWA ILP SOLUTION")
    print("=" * 78)
    print(f"Solver Status               : {res['status']}")
    print(f"Max Used Wavelength (W_high): {res['max_wavelength']}")
    print(f"Weighted Wavelength Sum     : {res['weighted_index_sum']}")
    print(f"Total Routed Path Distance  : {res['total_distance_km']} km")
    print("-" * 78)
    print("Selected Destination Subsets per Manycast Source:")
    for req in MANYCAST_REQUESTS:
        r_id = req["id"]
        s = req["source"]
        chosen = res["selected_destinations"][r_id]
        print(
            f"  Source {s} (Candidates {req['candidates']}, Need {req['k_dest']}, "
            f"Vol={req['traffic_vol']}) -> Selected Destinations: {chosen}"
        )

    print("-" * 78)
    print("Allocated Lightpaths (Source -> Selected Dest | Wavelength | Path | Length):")
    for lp in res["lightpaths"]:
        path_str = " -> ".join(map(str, lp["path"]))
        print(
            f"  Req #{lp['request_id']} ({lp['source']} -> {lp['destination']}) | "
            f"Wavelength W{lp['wavelength']} | Path: {path_str:<15} | {lp['distance_km']} km"
        )
    print("=" * 78)

    plot_q4_topology_and_allocations(res)

    # Generate graph-level visualizations of lightpath routes and wavelength planes
    G = build_q4_graph()
    routes_formatted = [
        {"demand": (lp["source"], lp["destination"]), "wavelength": lp["wavelength"], "path": lp["path"]}
        for lp in res["lightpaths"]
    ]

    plot_composite_lightpaths_on_graph(
        G,
        NODE_POSITIONS,
        routes_formatted,
        title=f"Q4 Manycast RWA: Established Lightpaths on Physical Topology (W_high = {res['max_wavelength']})",
        save_path="results/figures/q4_manycast_graph_lightpaths.png",
    )
    plot_wavelength_planes(
        G,
        NODE_POSITIONS,
        routes_formatted,
        num_wavelengths=res["max_wavelength"],
        title=f"Q4 Manycast RWA: Wavelength Planes (W1 - W{res['max_wavelength']})",
        save_path="results/figures/q4_manycast_wavelength_planes.png",
    )
    print("[INFO] Saved Q4 Manycast graph-level visualizations to results/figures/")
