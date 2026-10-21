"""
Question 5: Heuristic Network Planner (Semi-Static Distance-Adaptive RWA)
Topology: 12-Node, 20-Link German Backbone Network (160 Wavelength Channels per Link)
Algorithm: Top-3 Shortest Paths + Distance-Adaptive Modulation + Min-Max Channel First-Fit
Author: Erfan Ghasry
"""

import os
import math
import random
from itertools import islice
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# 1. Creating the 12-Node German Network Topology
def build_german_network() -> nx.Graph:
    network = nx.Graph()
    network.add_nodes_from(range(1, 13))
    edges = [
        (1, 5, 358), (1, 6, 344), (1, 8, 202), (2, 5, 460), (2, 3, 87),
        (2, 7, 48),  (3, 7, 98),  (3, 6, 258), (4, 6, 366), (4, 7, 212),
        (4, 8, 414), (4, 10, 261), (4, 11, 242), (5, 6, 188), (6, 8, 300),
        (8, 10, 321), (9, 10, 212), (9, 12, 170), (10, 11, 220), (11, 12, 202),
    ]
    network.add_weighted_edges_from(edges)
    return network


# 2. Precomputing Top-3 Shortest Paths for Every Node Pair
def compute_top_paths(network: nx.Graph, k: int = 3) -> dict:
    top_paths = {}
    for i in range(1, 13):
        top_paths[i] = {}
        for j in range(1, 13):
            if i == j:
                continue
            ans = islice(nx.shortest_simple_paths(network, i, j, weight="weight"), k)
            top_paths[i][j] = list(ans)
    return top_paths


# 3. Distance-Adaptive Modulation & Bandwidth Helper Functions
def modulation_level(length_km: float) -> str:
    """
    Course Appendix Distance-Adaptive Modulation Reach Table:
      - L <= 600 km        -> 16-QAM (400 Gbps / channel)
      - 600 < L <= 1200 km -> 8-QAM  (300 Gbps / channel)
      - 1200 < L <= 2400   -> QPSK   (200 Gbps / channel)
      - L > 2400 km        -> BPSK   (100 Gbps / channel)
    """
    if length_km <= 600:
        return "16-QAM"
    elif length_km <= 1200:
        return "8-QAM"
    elif length_km <= 2400:
        return "QPSK"
    else:
        return "BPSK"


def requst_bandwidth_finder(value_gbps: float, modulation: str) -> int:
    """
    Returns the number of wavelength channels needed in one direction
    for a demand of `value_gbps` Gbps given the selected modulation format.
    """
    capacity_per_lambda = {
        "16-QAM": 400,
        "8-QAM": 300,
        "QPSK": 200,
        "BPSK": 100,
    }
    rate = capacity_per_lambda.get(modulation, 100)
    return math.ceil(value_gbps / rate)


# 4. Link-Wavelength Usage Matrix Class (160 Channels per Link)
class matrix:
    def __init__(self, network: nx.Graph, num_channels: int = 160):
        self.edges = list(nx.edges(network))
        self.num_channels = num_channels
        self.link_wavelength_usage = [
            [-1 for _ in range(num_channels)] for _ in self.edges
        ]
        self.max_channel_index = 0

    def path_to_links(self, path: list[int]) -> list[tuple[int, int]]:
        return list(nx.utils.pairwise(path))

    def link_to_index(self, link: tuple[int, int]) -> int:
        try:
            index = self.edges.index(link)
        except ValueError:
            index = self.edges.index(link[::-1])
        return index

    def avalible_wavelengths_in_path(self, path: list[int]) -> list[int]:
        avalible_wavelengths = []
        links = self.path_to_links(path)
        for wavelength in range(self.num_channels):
            avalible = True
            for link in links:
                index = self.link_to_index(link)
                if self.link_wavelength_usage[index][wavelength] != -1:
                    avalible = False
                    break
            if avalible:
                avalible_wavelengths.append(wavelength)
        return avalible_wavelengths

    def use_wavelengths_in_path(self, path: list[int], wavelength: int, demand_index: int) -> bool:
        links = self.path_to_links(path)
        for link in links:
            index = self.link_to_index(link)
            self.link_wavelength_usage[index][wavelength] = demand_index
        if self.max_channel_index < wavelength:
            self.max_channel_index = wavelength
        return True


# 5. Heuristic Path & Wavelength Assignment Function
def assign_path_and_wavelength(demand: list[int], m: matrix, network: nx.Graph, top_paths: dict) -> tuple[bool, int]:
    source = demand[0]
    destination = demand[1]
    value = demand[2]

    data = {}
    for path in top_paths[source][destination]:
        length = nx.path_weight(network, path, weight="weight")
        modulation = modulation_level(length)
        # * 2 because bidirectional traffic (s->d and d->s) is allocated on the same path
        needed_wavelengths = requst_bandwidth_finder(value, modulation) * 2

        avalible_wavelengths = m.avalible_wavelengths_in_path(path)
        if len(avalible_wavelengths) >= needed_wavelengths:
            data[str(path)] = (needed_wavelengths, avalible_wavelengths)

    if len(data) == 0:
        return False, value

    # Heuristic metric: prefer the candidate path that minimizes the max channel index metric
    max_index = {}
    for path_str in data.keys():
        max_index[path_str] = min(data[path_str][1]) + data[path_str][0] - 1

    sorted_paths = sorted(max_index.items(), key=lambda x: x[1])
    best_path_str = sorted_paths[0][0]
    path = [int(i) for i in best_path_str[1:-1].split(",")]

    needed_w, avail_w_list = data[best_path_str]
    for i in range(needed_w):
        m.use_wavelengths_in_path(path, avail_w_list[i], value)

    return True, value


def generate_random_traffic(num_demands: int = 1000, seed: int = 42) -> list[list[int]]:
    random.seed(seed)
    traffic = []
    for _ in range(num_demands):
        source = random.randint(1, 12)
        destination = random.randint(1, 12)
        while source == destination:
            destination = random.randint(1, 12)
        value = random.randint(100, 800)  # 100 to 800 Gbps
        traffic.append([source, destination, value])
    return traffic


def run_planner_simulation(num_demands: int = 1000, seed: int = 42) -> dict:
    network = build_german_network()
    top_paths = compute_top_paths(network, k=3)
    traffic = generate_random_traffic(num_demands=num_demands, seed=seed)

    m = matrix(network, num_channels=160)
    blocked = 0
    assigned = 0
    max_index_plot = []
    max_index_plot_1_percent = []
    one_percent_info = {}
    flag = False

    for demand in traffic:
        is_assigned, value = assign_path_and_wavelength(demand, m, network, top_paths)
        if not is_assigned:
            blocked += value
        else:
            assigned += value

        if blocked >= 0.01 * (assigned + blocked) and not flag:
            max_index_plot_1_percent = max_index_plot.copy()
            one_percent_info = {
                "assigned_traffic_gbps": assigned,
                "blocked_traffic_gbps": blocked,
                "num_demands_at_1_percent": len(max_index_plot),
                "f_max_at_1_percent": m.max_channel_index,
            }
            print("1% of the demands are blocked")
            print(f"value of assigned traffic : {assigned}")
            print(f"value of blocked traffic : {blocked}")
            print(f"number of demands : {len(max_index_plot)}")
            flag = True

        max_index_plot.append(m.max_channel_index)

    if not flag:
        print("all demands are assigned")

    return {
        "network": network,
        "matrix": m,
        "max_index_plot": max_index_plot,
        "max_index_plot_1_percent": max_index_plot_1_percent,
        "one_percent_info": one_percent_info,
        "total_assigned_gbps": assigned,
        "total_blocked_gbps": blocked,
        "blocking_ratio": blocked / (assigned + blocked) if (assigned + blocked) > 0 else 0,
    }


def plot_q5_results(sim_res: dict, output_dir: str = "results/figures"):
    os.makedirs(output_dir, exist_ok=True)
    network = sim_res["network"]
    plot_1pct = sim_res["max_index_plot_1_percent"]
    plot_all = sim_res["max_index_plot"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=300)

    # Panel 1: 12-Node German Topology
    ax_topo = axes[0]
    pos = nx.spring_layout(network, seed=7)
    nx.draw_networkx_nodes(network, pos, ax=ax_topo, node_size=520, node_color="#a8dadc", edgecolors="#1d3557", linewidths=1.8)
    nx.draw_networkx_labels(network, pos, ax=ax_topo, font_size=9.5, font_weight="bold")
    nx.draw_networkx_edges(network, pos, ax=ax_topo, width=1.6, edge_color="#457b9d")
    edge_labels = nx.get_edge_attributes(network, "weight")
    nx.draw_networkx_edge_labels(network, pos, edge_labels=edge_labels, ax=ax_topo, font_size=7)
    ax_topo.set_title("Q5: 12-Node German Backbone Topology", fontsize=11, fontweight="bold")
    ax_topo.axis("off")

    # Panel 2: F_max vs Number of Demands up to 1% Blocking (Linear Growth Regime)
    ax_1pct = axes[1]
    ax_1pct.plot(range(1, len(plot_1pct) + 1), plot_1pct, color="#2a9d8f", linewidth=2.2)
    ax_1pct.set_xlabel("Number of Demands")
    ax_1pct.set_ylabel("Max Channel Index Used (F_max)")
    ax_1pct.set_title(f"F_max vs. Demands (Up to 1% Blocking: N={len(plot_1pct)})", fontsize=11, fontweight="bold")
    ax_1pct.grid(True, linestyle=":", alpha=0.6)

    # Panel 3: F_max vs Number of Demands for All Traffic (Saturation Regime)
    ax_all = axes[2]
    ax_all.plot(range(1, len(plot_all) + 1), plot_all, color="#e63946", linewidth=2.0)
    ax_all.axhline(159, color="#1d3557", linestyle="--", linewidth=1.2, label="Capacity Ceiling (160 Channels)")
    if len(plot_1pct) > 0:
        ax_all.axvline(len(plot_1pct), color="#2a9d8f", linestyle="-.", linewidth=1.2, label=f"1% Blocking (N={len(plot_1pct)})")
    ax_all.set_xlabel("Number of Demands")
    ax_all.set_ylabel("Max Channel Index Used (F_max)")
    ax_all.set_title("F_max vs. Demands (Full Traffic Saturation Curve)", fontsize=11, fontweight="bold")
    ax_all.legend(loc="lower right", fontsize=8.5)
    ax_all.grid(True, linestyle=":", alpha=0.6)

    out_path = os.path.join(output_dir, "q5_heuristic_planner.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved Q5 Heuristic Planner plots to: {out_path}")


if __name__ == "__main__":
    res = run_planner_simulation(num_demands=1000, seed=42)
    plot_q5_results(res)
