"""Standardized visualization utilities for optical network topology and spectrum allocation."""

from typing import Optional, Dict, Any, List, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

WAVELENGTH_COLORS = {
    1: "#2a9d8f",  # Teal / Green (W1)
    2: "#e76f51",  # Coral / Orange (W2)
    3: "#457b9d",  # Steel Blue (W3)
    4: "#e63946",  # Crimson Red (W4)
    5: "#9b5de5",  # Purple (W5)
    6: "#f15bb5",  # Pink (W6)
}


def plot_network_topology(
    graph: nx.Graph,
    pos: Dict[int, Tuple[float, float]],
    title: str = "Optical Network Topology",
    save_path: Optional[str] = None,
) -> None:
    """Plot physical network topology graph with publication styling."""
    plt.figure(figsize=(7, 6), dpi=300)
    nx.draw_networkx_nodes(
        graph, pos, node_size=900, node_color="#e0fbfc", edgecolors="#293241", linewidths=2
    )
    nx.draw_networkx_labels(graph, pos, font_size=12, font_weight="bold", font_color="#293241")
    nx.draw_networkx_edges(graph, pos, width=2.5, edge_color="#3d5a80")
    plt.title(title, fontsize=13, fontweight="bold", pad=15)
    plt.axis("off")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_spectrum_utilization(
    utilization_matrix: Any,
    title: str = "Spectrum Slot Utilization",
    save_path: Optional[str] = None,
) -> None:
    """Plot heatmap / grid of link spectrum slot usage."""
    plt.figure(figsize=(12, 6), dpi=300)
    plt.title(title, fontsize=14, fontweight="bold")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_wavelength_planes(
    graph: nx.Graph,
    pos: Dict[int, Tuple[float, float]],
    routes: List[Dict[str, Any]],
    num_wavelengths: int,
    title: str = "Wavelength Layered Lightpath Allocation",
    save_path: Optional[str] = None,
) -> None:
    """
    Plots an optical network WDM wavelength-plane grid (e.g. 2x2 or 1xN).
    Each panel displays one wavelength plane W_i with all lightpaths assigned to that wavelength.
    """
    cols = min(num_wavelengths, 2 if num_wavelengths <= 4 else 3)
    rows = (num_wavelengths + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.8 * rows), dpi=300)
    if num_wavelengths == 1:
        axes = np.array([axes])
    axes = np.array(axes).reshape(-1)

    for w_idx in range(1, num_wavelengths + 1):
        ax = axes[w_idx - 1]
        w_color = WAVELENGTH_COLORS.get(w_idx, "#333333")

        # Draw background base topology
        nx.draw_networkx_nodes(
            graph, pos, ax=ax, node_size=750, node_color="#f8f9fa", edgecolors="#adb5bd", linewidths=1.5
        )
        nx.draw_networkx_labels(graph, pos, ax=ax, font_size=11, font_weight="bold", font_color="#495057")
        nx.draw_networkx_edges(
            graph, pos, ax=ax, width=1.5, edge_color="#dee2e6", style="dashed"
        )

        # Filter routes using this wavelength
        w_routes = [r for r in routes if r["wavelength"] == w_idx]
        route_labels = []

        for r in w_routes:
            path = r["path"]
            edge_list = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=edge_list,
                ax=ax,
                width=4.5,
                edge_color=w_color,
                alpha=0.85,
            )
            d_str = f"Demand {r['demand']}: " + "→".join(map(str, path))
            route_labels.append(d_str)

        panel_title = f"Wavelength W{w_idx} Plane"
        if route_labels:
            caption = "\n".join(route_labels)
            ax.set_title(f"{panel_title}\n{caption}", fontsize=10, fontweight="bold", color=w_color)
        else:
            ax.set_title(f"{panel_title}\n(Idle)", fontsize=10, fontweight="normal", color="#6c757d")

        ax.axis("off")

    # Hide unused subplots if any
    for idx in range(num_wavelengths, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_composite_lightpaths_on_graph(
    graph: nx.Graph,
    pos: Dict[int, Tuple[float, float]],
    routes: List[Dict[str, Any]],
    title: str = "Optical Lightpaths and Wavelength Assignment",
    save_path: Optional[str] = None,
) -> None:
    """
    Plots a composite single graph where multi-wavelength lightpaths sharing the same
    physical link are rendered with distinct parallel offsets and wavelength badges.
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # Draw underlying fiber links
    nx.draw_networkx_edges(graph, pos, ax=ax, width=8, edge_color="#e9ecef", alpha=0.9)
    nx.draw_networkx_edges(graph, pos, ax=ax, width=1.5, edge_color="#ced4da", style="dotted")

    # Calculate per-edge lightpath usage and offsets
    edge_usages: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for r in routes:
        path = r["path"]
        for i in range(len(path) - 1):
            canon_edge = tuple(sorted((path[i], path[i + 1])))
            edge_usages.setdefault(canon_edge, []).append(r)

    # Draw colored parallel offset tracks along each physical edge
    for (u, v), edge_routes in edge_usages.items():
        p1 = np.array(pos[u])
        p2 = np.array(pos[v])
        edge_vec = p2 - p1
        edge_len = np.linalg.norm(edge_vec)
        if edge_len == 0:
            continue
        perp_vec = np.array([-edge_vec[1], edge_vec[0]]) / edge_len

        num_tracks = len(edge_routes)
        spacing = 0.035
        offsets = (np.arange(num_tracks) - (num_tracks - 1) / 2.0) * spacing

        for offset, r in zip(offsets, edge_routes):
            color = WAVELENGTH_COLORS.get(r["wavelength"], "#333333")
            start = p1 + perp_vec * offset
            end = p2 + perp_vec * offset
            ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=3.2, alpha=0.9, solid_capstyle="round")

    # Draw nodes on top
    nx.draw_networkx_nodes(
        graph, pos, ax=ax, node_size=1000, node_color="#f8f9fa", edgecolors="#2b2d42", linewidths=2.5
    )
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=13, font_weight="bold", font_color="#2b2d42")

    # Edge annotations showing wavelength list
    for (u, v), edge_routes in edge_usages.items():
        mid_x = (pos[u][0] + pos[v][0]) / 2.0
        mid_y = (pos[u][1] + pos[v][1]) / 2.0
        w_list = sorted(set(r["wavelength"] for r in edge_routes))
        w_str = "{" + ",".join(f"W{w}" for w in w_list) + "}"
        ax.text(
            mid_x, mid_y + 0.05, w_str,
            fontsize=9, fontweight="bold", color="#1d3557",
            bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec="#457b9d", lw=1.2, alpha=0.95),
            ha="center", va="center", zorder=7
        )

    # Legend for lightpaths
    legend_handles = []
    legend_labels = []
    # Sort by demand
    sorted_routes = sorted(routes, key=lambda x: (x["wavelength"], x["demand"]))
    for r in sorted_routes:
        w = r["wavelength"]
        color = WAVELENGTH_COLORS.get(w, "#333333")
        path_str = "→".join(map(str, r["path"]))
        line = plt.Line2D([0], [0], color=color, lw=3)
        legend_handles.append(line)
        legend_labels.append(f"W{w}: Demand {r['demand']} [{path_str}]")

    ax.legend(
        legend_handles, legend_labels, loc="lower left", fontsize=8.5, framealpha=0.95,
        title="Established Lightpaths", title_fontsize=10
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
