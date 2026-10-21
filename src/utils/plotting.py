"""Standardized visualization utilities for optical network topology and spectrum allocation."""

from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
import networkx as nx


def plot_network_topology(
    graph: nx.Graph,
    title: str = "Optical Network Topology",
    save_path: Optional[str] = None,
) -> None:
    """Plot network topology graph with publication styling."""
    plt.figure(figsize=(10, 6), dpi=300)
    pos = nx.spring_layout(graph, seed=42)
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="skyblue",
        node_size=700,
        edge_color="gray",
        font_size=10,
        font_weight="bold",
    )
    plt.title(title, fontsize=14, fontweight="bold")
    if save_path:
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
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
