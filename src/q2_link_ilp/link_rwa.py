"""
Question 2: Link-Based ILP Formulation for Routing and Wavelength Assignment (RWA)
Objective: Maximize total number of served traffic requests under intermediate hop limit H.
 Compares H = 1 vs. H = 2.
Author: Erfan Ghasry
"""

import os
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from src.utils.q2_q3_data import (
    NODES,
    UNDIRECTED_EDGES,
    DIRECTED_ARCS,
    NODE_POSITIONS,
    get_directed_demands,
    get_undirected_demands,
    build_nx_graph,
)
from src.utils.plotting import (
    plot_composite_lightpaths_on_graph,
    plot_wavelength_planes,
)


def solve_link_based_ilp(
    H: int,
    num_wavelengths: int = 1,
    hop_definition: str = "intermediate_nodes",
    undirected_mode: bool = False,
) -> dict:
    """
    Solves the Link-Based ILP to maximize served requests.

    Parameters
    ----------
    H : int
        Maximum hop parameter (H = 1 or H = 2).
    num_wavelengths : int
        Number of available wavelengths per fiber link (default 1; set 3 for multi-wavelength).
    hop_definition : str
        - 'intermediate_nodes': H intermediate nodes => max_links = H + 1 (Literal Persian text: "hop های میانی")
        - 'total_links': H total links => max_links = H
    undirected_mode : bool
        If False, solves for all 12 directed demands over directed arcs.
        If True, solves for the 6 symmetric demand pairs over undirected edges.
    """
    demands = get_undirected_demands() if undirected_mode else get_directed_demands()
    max_links = H + 1 if hop_definition == "intermediate_nodes" else H
    W = range(num_wavelengths)

    # Decision Variables:
    # y[(s, d, w)] in {0, 1}: 1 if demand (s, d) is served on wavelength w
    # f[(s, d, w, u, v)] in {0, 1}: 1 if demand (s, d) uses directed arc (u, v) on wavelength w
    y = {
        (s, d, w): cp.Variable(boolean=True, name=f"y_{s}_{d}_w{w}")
        for (s, d) in demands
        for w in W
    }
    f = {
        (s, d, w, u, v): cp.Variable(boolean=True, name=f"f_{s}_{d}_w{w}_{u}_{v}")
        for (s, d) in demands
        for w in W
        for (u, v) in DIRECTED_ARCS
    }

    # Objective: Maximize total served requests
    objective = cp.Maximize(cp.sum([y[(s, d, w)] for (s, d) in demands for w in W]))

    constraints = []

    # 1. Each demand can be served on at most one wavelength
    for (s, d) in demands:
        constraints.append(cp.sum([y[(s, d, w)] for w in W]) <= 1)

    # 2. Flow conservation & wavelength continuity at every node
    for (s, d) in demands:
        for w in W:
            for node in NODES:
                out_flow = cp.sum([f[(s, d, w, u, v)] for (u, v) in DIRECTED_ARCS if u == node])
                in_flow = cp.sum([f[(s, d, w, u, v)] for (u, v) in DIRECTED_ARCS if v == node])

                if node == s:
                    constraints.append(out_flow - in_flow == y[(s, d, w)])
                    constraints.append(in_flow == 0)
                elif node == d:
                    constraints.append(in_flow - out_flow == y[(s, d, w)])
                    constraints.append(out_flow == 0)
                else:
                    constraints.append(out_flow - in_flow == 0)
                    constraints.append(out_flow <= y[(s, d, w)])  # Subtour / loop prevention

            # 3. Maximum intermediate hop constraint
            total_links_used = cp.sum([f[(s, d, w, u, v)] for (u, v) in DIRECTED_ARCS])
            constraints.append(total_links_used <= max_links * y[(s, d, w)])

    # 4. Link capacity (No wavelength clash)
    for w in W:
        if undirected_mode:
            # Undirected edge capacity: at most 1 lightpath per wavelength in either direction
            for (u, v) in UNDIRECTED_EDGES:
                constraints.append(
                    cp.sum([f[(s, d, w, u, v)] + f[(s, d, w, v, u)] for (s, d) in demands]) <= 1
                )
        else:
            # Directed arc capacity: at most 1 lightpath per wavelength per directed arc
            for (u, v) in DIRECTED_ARCS:
                constraints.append(
                    cp.sum([f[(s, d, w, u, v)] for (s, d) in demands]) <= 1
                )

    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Reconstruct active paths
    served_details = []
    for (s, d) in demands:
        for w in W:
            if y[(s, d, w)].value is not None and round(float(y[(s, d, w)].value)) == 1:
                active_arcs = [
                    (u, v)
                    for (u, v) in DIRECTED_ARCS
                    if f[(s, d, w, u, v)].value is not None and round(float(f[(s, d, w, u, v)].value)) == 1
                ]
                # Order path from s to d
                path = [s]
                curr = s
                while curr != d and active_arcs:
                    for (u, v) in active_arcs:
                        if u == curr:
                            path.append(v)
                            active_arcs.remove((u, v))
                            curr = v
                            break
                served_details.append({
                    "demand": (s, d),
                    "wavelength": w + 1,
                    "path": path,
                    "num_links": len(path) - 1,
                    "intermediate_hops": len(path) - 2,
                })

    return {
        "H": H,
        "max_links": max_links,
        "num_wavelengths": num_wavelengths,
        "total_demands": len(demands),
        "served_count": int(round(float(problem.value))) if problem.value is not None else 0,
        "served_details": served_details,
        "status": problem.status,
    }


def plot_q2_comparison(results_dict: dict, output_dir: str = "results/figures"):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # Left subplot: Topology visualization
    G = build_nx_graph()
    ax_topo = axes[0]
    nx.draw_networkx_nodes(G, NODE_POSITIONS, ax=ax_topo, node_size=900, node_color="#e0fbfc", edgecolors="#293241", linewidths=2)
    nx.draw_networkx_labels(G, NODE_POSITIONS, ax=ax_topo, font_size=12, font_weight="bold")
    nx.draw_networkx_edges(G, NODE_POSITIONS, ax=ax_topo, width=2.2, edge_color="#3d5a80")
    ax_topo.set_title("Q2 & Q3 Physical Network Topology (5 Nodes, 5 Links)", fontsize=12, fontweight="bold")
    ax_topo.axis("off")

    # Right subplot: Bar chart comparing H=1 vs H=2 across scenarios
    ax_bar = axes[1]
    labels = list(results_dict.keys())
    h1_vals = [results_dict[k]["H1"]["served_count"] for k in labels]
    h2_vals = [results_dict[k]["H2"]["served_count"] for k in labels]
    totals = [results_dict[k]["H1"]["total_demands"] for k in labels]

    x = np.arange(len(labels))
    width = 0.32

    rects1 = ax_bar.bar(x - width / 2, h1_vals, width, label="H = 1", color="#457b9d")
    rects2 = ax_bar.bar(x + width / 2, h2_vals, width, label="H = 2", color="#2a9d8f")

    for i, total in enumerate(totals):
        ax_bar.axhline(y=total, xmin=(i / len(labels)) + 0.05, xmax=((i + 1) / len(labels)) - 0.05,
                       color="#e63946", linestyle="--", linewidth=1.2)

    ax_bar.bar_label(rects1, padding=3, fontweight="bold")
    ax_bar.bar_label(rects2, padding=3, fontweight="bold")
    ax_bar.set_ylabel("Number of Served Requests")
    ax_bar.set_title("Comparison of Served Requests: H = 1 vs. H = 2", fontsize=12, fontweight="bold")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=9)
    ax_bar.set_ylim(0, 14)
    ax_bar.grid(axis="y", linestyle=":", alpha=0.6)
    ax_bar.legend()

    out_path = os.path.join(output_dir, "q2_h1_vs_h2_comparison.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved Q2 comparison plot to: {out_path}")


if __name__ == "__main__":
    scenarios = {
        "Directed (W=1)\n[Interm. Hops]": {"W": 1, "hop_def": "intermediate_nodes", "undir": False},
        "Directed (W=1)\n[Total Links]": {"W": 1, "hop_def": "total_links", "undir": False},
        "Undirected (W=1)\n[Interm. Hops]": {"W": 1, "hop_def": "intermediate_nodes", "undir": True},
        "Undirected (W=3)\n[3 Colors]": {"W": 3, "hop_def": "intermediate_nodes", "undir": True},
    }

    summary_results = {}
    for name, cfg in scenarios.items():
        res_h1 = solve_link_based_ilp(H=1, num_wavelengths=cfg["W"], hop_definition=cfg["hop_def"], undirected_mode=cfg["undir"])
        res_h2 = solve_link_based_ilp(H=2, num_wavelengths=cfg["W"], hop_definition=cfg["hop_def"], undirected_mode=cfg["undir"])
        summary_results[name] = {"H1": res_h1, "H2": res_h2}

        clean_name = name.replace("\n", " ")
        print("=" * 70)
        print(f"SCENARIO: {clean_name}")
        print("=" * 70)
        for res in (res_h1, res_h2):
            print(f"\n--- H = {res['H']} (Max Links = {res['max_links']}) -> Served: {res['served_count']} / {res['total_demands']} ---")
            for item in res["served_details"]:
                path_str = " -> ".join(map(str, item["path"]))
                print(
                    f"  Demand {item['demand']} | Wavelength W{item['wavelength']} | "
                    f"Path: {path_str} ({item['intermediate_hops']} intermediate hops, {item['num_links']} links)"
                )

    plot_q2_comparison(summary_results)

    # Generate graph-level visualizations of lightpath routes for W=3 (H=1 and H=2)
    G = build_nx_graph()
    w3_h1 = summary_results["Undirected (W=3)\n[3 Colors]"]["H1"]
    w3_h2 = summary_results["Undirected (W=3)\n[3 Colors]"]["H2"]

    plot_composite_lightpaths_on_graph(
        G,
        NODE_POSITIONS,
        w3_h1["served_details"],
        title="Q2 Lightpath Routing on Topology: H = 1 (Max 1 Interm. Hop)",
        save_path="results/figures/q2_graph_lightpaths_h1.png",
    )
    plot_composite_lightpaths_on_graph(
        G,
        NODE_POSITIONS,
        w3_h2["served_details"],
        title="Q2 Lightpath Routing on Topology: H = 2 (Max 2 Interm. Hops - Unlocks (2,4)!)",
        save_path="results/figures/q2_graph_lightpaths_h2.png",
    )
    plot_wavelength_planes(
        G,
        NODE_POSITIONS,
        w3_h2["served_details"],
        num_wavelengths=3,
        title="Q2 WDM Layered Wavelength Planes (H = 2, W = 3 Colors)",
        save_path="results/figures/q2_graph_wavelength_planes_h2.png",
    )
    print("[INFO] Saved Q2 graph visualizations to results/figures/")
