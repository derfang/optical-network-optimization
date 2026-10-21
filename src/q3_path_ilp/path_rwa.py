"""
Question 3: Path-Based ILP Formulation with Yen's K-Shortest Link-Disjoint Paths
Objective: Minimize the highest used wavelength index (W_high).
Author: Erfan Ghasry
"""

import os
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt
from src.utils.q2_q3_data import (
    build_nx_graph,
    UNDIRECTED_EDGES,
    NODE_POSITIONS,
    get_undirected_demands,
)
from src.utils.plotting import (
    plot_composite_lightpaths_on_graph,
    plot_wavelength_planes,
)


def get_link_disjoint_paths(G: nx.Graph, source: int, target: int, k: int = 3) -> list[list[int]]:
    """
    Computes up to k link-disjoint paths between source and target using
    successive shortest path removal (link-disjoint Yen's variation).
    """
    paths = []
    G_temp = G.copy()

    for _ in range(k):
        try:
            path = nx.shortest_path(G_temp, source=source, target=target, weight=None)
            paths.append(path)
            # Remove edges of this path to ensure link-disjointness for the next candidate
            edge_list = list(zip(path[:-1], path[1:]))
            G_temp.remove_edges_from(edge_list)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            break

    return paths


def solve_path_based_ilp(max_wavelengths: int = 6) -> dict:
    G = build_nx_graph()
    demands = get_undirected_demands()
    W = range(1, max_wavelengths + 1)

    # 1. Precompute up to K=3 link-disjoint paths for each demand
    candidate_paths = {}  # (s, d): list of paths
    path_indices = []     # flat list of ((s, d), path_idx, path_nodes, link_set)

    for (s, d) in demands:
        paths = get_link_disjoint_paths(G, s, d, k=3)
        candidate_paths[(s, d)] = paths
        for idx, path in enumerate(paths):
            link_set = set(tuple(sorted((path[i], path[i+1]))) for i in range(len(path) - 1))
            path_indices.append({
                "sd": (s, d),
                "k": idx,
                "path": path,
                "links": link_set,
            })

    # 2. Decision Variables
    # c[i]: binary variable for the i-th (demand, path_idx, wavelength) tuple
    c = {}
    for item_idx, item in enumerate(path_indices):
        for w in W:
            c[(item_idx, w)] = cp.Variable(boolean=True, name=f"c_sd{item['sd']}_p{item['k']}_w{w}")

    # Maximum wavelength index variable
    W_high = cp.Variable(integer=True, name="W_high")

    # 3. Objective: Minimize W_high
    objective = cp.Minimize(W_high)

    constraints = [W_high >= 0]

    # Demand satisfaction: exactly 1 lightpath per demand
    for (s, d) in demands:
        relevant_vars = [
            c[(idx, w)]
            for idx, item in enumerate(path_indices)
            if item["sd"] == (s, d)
            for w in W
        ]
        if relevant_vars:
            constraints.append(cp.sum(relevant_vars) == 1)

    # Wavelength clash / Link capacity: each physical link can carry at most 1 lightpath per wavelength
    for edge in UNDIRECTED_EDGES:
        canonical_edge = tuple(sorted(edge))
        for w in W:
            link_vars = []
            for idx, item in enumerate(path_indices):
                if canonical_edge in item["links"]:
                    link_vars.append(c[(idx, w)])
            if link_vars:
                constraints.append(cp.sum(link_vars) <= 1)

    # Bounding W_high
    for idx, item in enumerate(path_indices):
        for w in W:
            constraints.append(W_high >= w * c[(idx, w)])

    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Extract solution
    assigned_routes = []
    if problem.status in ["optimal", "optimal_inaccurate"]:
        for idx, item in enumerate(path_indices):
            for w in W:
                var = c[(idx, w)]
                if var.value is not None and round(float(var.value)) == 1:
                    assigned_routes.append({
                        "demand": item["sd"],
                        "path": item["path"],
                        "wavelength": w,
                        "candidate_idx": item["k"],
                    })

    return {
        "status": problem.status,
        "W_high": int(round(float(W_high.value))) if W_high.value is not None else None,
        "routes": assigned_routes,
        "candidate_paths": candidate_paths,
    }


def plot_q3_allocation(result: dict, output_dir: str = "results/figures"):
    """Visualizes candidate paths and final wavelength assignments."""
    os.makedirs(output_dir, exist_ok=True)
    G = build_nx_graph()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # Left plot: Base topology with physical links
    ax_topo = axes[0]
    nx.draw_networkx_nodes(G, NODE_POSITIONS, ax=ax_topo, node_size=900, node_color="#f8f9fa", edgecolors="#2b2d42", linewidths=2)
    nx.draw_networkx_labels(G, NODE_POSITIONS, ax=ax_topo, font_size=12, font_weight="bold")
    nx.draw_networkx_edges(G, NODE_POSITIONS, ax=ax_topo, width=2.5, edge_color="#8d99ae")
    ax_topo.set_title("5-Node Physical Optical Topology", fontsize=12, fontweight="bold")
    ax_topo.axis("off")

    # Right plot: Wavelength allocation chart per demand
    ax_alloc = axes[1]
    routes = result["routes"]
    demands_str = [f"({r['demand'][0]}, {r['demand'][1]})" for r in routes]
    wavelengths = [r["wavelength"] for r in routes]
    paths_str = [" -> ".join(map(str, r["path"])) for r in routes]

    palette = {1: "#2a9d8f", 2: "#e76f51", 3: "#457b9d", 4: "#f4a261", 5: "#e63946"}
    colors = [palette.get(w, "#6c757d") for w in wavelengths]

    y_pos = range(len(routes))
    bars = ax_alloc.barh(y_pos, wavelengths, color=colors, height=0.55, edgecolor="#293241")
    ax_alloc.set_yticks(y_pos)
    ax_alloc.set_yticklabels([f"Demand {d}\n[{p}]" for d, p in zip(demands_str, paths_str)], fontsize=9)
    ax_alloc.set_xlabel("Assigned Wavelength Index ($w$)")
    ax_alloc.set_title(f"Q3 Optimal Allocation (Min Max Wavelength $W_{{high}} = {result['W_high']}$)", fontsize=12, fontweight="bold")
    ax_alloc.set_xlim(0, max(result["W_high"] + 1, 4))
    ax_alloc.grid(axis="x", linestyle=":", alpha=0.6)

    for bar, w in zip(bars, wavelengths):
        ax_alloc.text(w + 0.08, bar.get_y() + bar.get_height() / 2, f"W{w}", va="center", fontweight="bold", fontsize=10)

    out_path = os.path.join(output_dir, "q3_path_rwa_allocation.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved Q3 allocation plot to: {out_path}")


if __name__ == "__main__":
    result = solve_path_based_ilp(max_wavelengths=6)
    print("=" * 65)
    print("QUESTION 3: PATH-BASED ILP (YEN'S K-SHORTEST LINK-DISJOINT PATHS)")
    print("=" * 65)
    print(f"Solver Status       : {result['status']}")
    print(f"Max Wavelength (W_high): {result['W_high']}")
    print("\nCandidate Paths Discovered & Assigned Routes:")
    for sd, paths in result["candidate_paths"].items():
        print(f"  Demand {sd}: {len(paths)} link-disjoint paths found.")
        for i, p in enumerate(paths):
            print(f"    Path k={i}: {' -> '.join(map(str, p))}")

    print("\nFinal Optimal Route & Wavelength Allocation:")
    for route in result["routes"]:
        path_str = " -> ".join(map(str, route["path"]))
        print(f"  Demand {route['demand']} | Wavelength W{route['wavelength']} | Path: {path_str}")
    print("=" * 65)

    plot_q3_allocation(result)

    # Generate graph-level visualizations of lightpath routes and wavelength planes
    G = build_nx_graph()
    plot_composite_lightpaths_on_graph(
        G,
        NODE_POSITIONS,
        result["routes"],
        title="Q3 Path-Based RWA: All Lightpaths on Physical Topology (W_high = 4)",
        save_path="results/figures/q3_graph_composite_lightpaths.png",
    )
    plot_wavelength_planes(
        G,
        NODE_POSITIONS,
        result["routes"],
        num_wavelengths=result["W_high"],
        title="Q3 Path-Based RWA: Wavelength Planes (W1 - W4)",
        save_path="results/figures/q3_graph_wavelength_planes.png",
    )
    print("[INFO] Saved Q3 graph visualizations to results/figures/")
