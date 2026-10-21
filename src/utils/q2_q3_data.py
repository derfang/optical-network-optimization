"""
Shared Topology and Traffic Matrix for Questions 2 and 3.
"""

import numpy as np
import networkx as nx

# 5-Node Topology from Question 2 & 3 Figure
NODES = [1, 2, 3, 4, 5]

# Undirected physical edges: (1,2), (1,3), (1,5), (2,5), (3,4)
UNDIRECTED_EDGES = [
    (1, 2),
    (1, 3),
    (1, 5),
    (2, 5),
    (3, 4),
]

# Directed arcs (bidirectional fiber links)
DIRECTED_ARCS = []
for u, v in UNDIRECTED_EDGES:
    DIRECTED_ARCS.append((u, v))
    DIRECTED_ARCS.append((v, u))

# Node positions matching the assignment diagram
NODE_POSITIONS = {
    1: (0.0, 1.0),
    2: (-0.8, 0.4),
    5: (0.8, 0.4),
    3: (-0.3, -0.5),
    4: (0.4, -0.5),
}

# 5x5 Traffic Matrix A_{5x5}
TRAFFIC_MATRIX = np.array([
    [0, 1, 1, 1, 0],  # Node 1 -> 2, 3, 4
    [1, 0, 0, 1, 0],  # Node 2 -> 1, 4
    [1, 0, 0, 1, 1],  # Node 3 -> 1, 4, 5
    [1, 1, 1, 0, 0],  # Node 4 -> 1, 2, 3
    [0, 0, 1, 0, 0],  # Node 5 -> 3
], dtype=int)


def get_directed_demands() -> list[tuple[int, int]]:
    """Returns all 12 directed (s, d) pairs where A[s, d] == 1."""
    demands = []
    for i in range(5):
        for j in range(5):
            if TRAFFIC_MATRIX[i, j] == 1:
                demands.append((i + 1, j + 1))
    return demands


def get_undirected_demands() -> list[tuple[int, int]]:
    """Returns the 6 unique symmetric demand pairs (s < d) where A[s, d] == 1."""
    demands = []
    for i in range(5):
        for j in range(i + 1, 5):
            if TRAFFIC_MATRIX[i, j] == 1 or TRAFFIC_MATRIX[j, i] == 1:
                demands.append((i + 1, j + 1))
    return demands


def build_nx_graph() -> nx.Graph:
    """Builds the undirected NetworkX graph for Q2/Q3."""
    G = nx.Graph()
    G.add_nodes_from(NODES)
    G.add_edges_from(UNDIRECTED_EDGES)
    return G
