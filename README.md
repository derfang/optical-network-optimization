# Optical Network Optimization: WDM RWA & Flex-Grid RSA Algorithms

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Optimization: CVXPY / PuLP](https://img.shields.io/badge/Optimization-CVXPY%20%7C%20PuLP-green.svg)](https://www.cvxpy.org/)
[![Network: NetworkX](https://img.shields.io/badge/Graph-NetworkX-orange.svg)](https://networkx.org/)

A production-grade algorithmic and mathematical optimization framework for **Wavelength Division Multiplexing (WDM) Routing and Wavelength Assignment (RWA)**, **Manycast RWA**, and **Elastic Optical Network (EON) / Flex-Grid Routing and Spectrum Allocation (RSA)**.

Authored by **[Erfan Ghasry (derfang)](https://github.com/derfang)**.

---

## 📌 Executive Summary & Key Highlights

This repository contains an end-to-end suite of exact Mixed-Integer Linear Programming (MILP/ILP) formulations and large-scale heuristics for modern optical communication backbones:
- **Exact Cut-Capacity Proofs**: Proved that $W_{\text{high}} = 4$ is the theoretical minimum wavelength bound on 5-node cut-sets via flow conservation and wavelength clash constraints.
- **Pareto Path Generation**: Compared Dijkstra $K=1$, Yen's $K$-shortest, and unconstrained loopless path sets, demonstrating that $K=2$ reduces spectrum utilization by $16.7\%$ with negligible ($<6\%$) latency overhead.
- **Carrier-Scale Simulation**: Engineered a distance-adaptive RWA planner for the 12-node German network carrying over **$84.9\text{ Tbps}$** under a $1\%$ blocking threshold across $1,000$ dynamic requests.
- **Paper Reproduction**: Implemented the Link-Path Channel Assignment (LP-CA) model from *Velasco et al. (Photonic Network Communications 2012)*, demonstrating up to **$100\times$ faster solver convergence** over traditional Starting Slot (SSA) and Slot Assignment (SA) formulations.

---

## 🏗️ Repository Architecture

```text
optical-network-optimization/
├── README.md
├── requirements.txt
├── data/
│   ├── topologies/          # Adjacency matrices, link lengths (km), NSFNET/DT graphs
│   └── traffic_matrices/    # Static demand matrices & Manycast candidate tables
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── graph_tools.py   # Topology loaders, Yen's K-shortest link-disjoint paths
│   │   ├── plotting.py      # Publication-quality network & spectrum plots
│   │   └── q2_q3_data.py    # Shared 5-node topology & traffic demand matrices
│   ├── q1_lp_intro/
│   │   └── beverage_lp.py   # Continuous LP vs. Discrete 100mL MILP (CVXPY & BFS)
│   ├── q2_link_ilp/
│   │   └── link_rwa.py      # Link-based ILP maximizing served requests (H=1 vs H=2)
│   ├── q3_path_ilp/
│   │   └── path_rwa.py      # Yen's K=3 link-disjoint + Path-based ILP minimizing max wavelength
│   ├── q4_manycast_rwa/
│   │   └── manycast_ilp.py  # Destination candidate selection + RWA ILP (K=1 vs K=2 vs All Paths)
│   ├── q5_heuristic_planner/
│   │   └── planner.py       # Heuristic network planner (12-Node German Network, 160 channels)
│   └── q6_flexgrid_paper/
│       └── rsa_simulation.py # Reproduction of Velasco et al. (2012) Flex-Grid RSA
├── results/
│   ├── figures/             # 300 DPI publication plots and topology visualizations
│   └── logs/                # Solver convergence logs, routing tables, and throughput records
└── docs/
    └── formulations.md      # Complete mathematical derivations, sets, and LaTeX equations
```

---

## 🔬 Problem Modules & Results

### 1. Q1: Continuous LP vs. Discrete MILP & BFS State-Space
Demonstrates fundamental mathematical programming principles:
- **Continuous LP**: $(0.625\text{ L}, 0.375\text{ L}) \to 0.125\text{ L}$ pure juice.
- **LP Relaxation + Rounding**: Truncating continuous solution to $100\text{ mL}$ packages yields suboptimal $0.108\text{ L}$.
- **Exact Discrete MILP & BFS**: Finds the global discrete optimum at $(5, 4) \to 0.120\text{ L}$ pure juice at $\$10.00$ budget.
- **Artifact**: `results/figures/q1_lp_vs_ilp.png`

### 2. Q2: Link-Based RWA ILP ($H=1$ vs. $H=2$ Hop Constraints)
Maximizes admitted lightpaths under flow conservation and no-converter wavelength continuity:
- **Hop Constraint Unlock**: At $H=1$, demand $(2, 4)$ is blocked because path $2 \to 1 \to 3 \to 4$ has 2 intermediate hops. Increasing $H=2$ routes $(2, 4)$ on $W_1$ and detours $(1, 2)$ via $1 \to 5 \to 2$.
- **Artifacts**: `results/figures/q2_h1_vs_h2_comparison.png`, `results/figures/q2_graph_lightpaths_h2.png`, `results/figures/q2_graph_wavelength_planes_h2.png`

### 3. Q3: Path-Based RWA ILP with Yen's Link-Disjoint Paths
Pre-computes candidate paths using Yen's algorithm ($K=3$) and minimizes the highest used wavelength index ($W_{\text{high}}$):
- **Theoretical Lower Bound**: Cut-edge $(1, 3)$ carries 4 intersecting demands $\implies W_{\text{high}} = 4$ is proven optimal.
- **Artifacts**: `results/figures/q3_path_rwa_allocation.png`, `results/figures/q3_graph_composite_lightpaths.png`, `results/figures/q3_graph_wavelength_planes.png`

### 4. Q4: Manycast RWA ILP (Destination Selection & Path Strategies)
Optimizes destination subsets $D_r \subseteq C_r$ ($|D_r| = k_r$) for 5 manycast sources across a 6-node weighted network:
- **Routing Strategy Trade-Offs**:
  - *Single Best (K=1)*: $W_{\text{high}} = 6$, $573\text{ km}$, $67.6\text{ ms}$.
  - *K-Shortest (K=2)*: **$W_{\text{high}} = 5$**, $606\text{ km}$, $160.9\text{ ms}$ (Balanced Pareto optimum).
  - *All Simple Paths*: $W_{\text{high}} = 3$, $1,195\text{ km}$, $1,088$ variables, $710.7\text{ ms}$ (Explodes complexity).
- **Artifacts**: `results/figures/q4_manycast_comparison.png`, `results/figures/q4_manycast_graph_lightpaths.png`, `results/figures/q4_manycast_wavelength_planes.png`

### 5. Q5: Distance-Adaptive Heuristic Carrier Planner
Simulates a semi-static optical planner on the 12-node German network ($160$ channels/link):
- **Coherent Transponder Reach**: 16-QAM (400G $\le 600\text{ km}$), 8-QAM (300G $\le 1200\text{ km}$), QPSK (200G $\le 2400\text{ km}$), BPSK (100G).
- **1% Blocking Threshold**: Reached at $N = 194$ demands with **$84,929\text{ Gbps}$** ($84.9\text{ Tbps}$) carried traffic.
- **Full Run Carried Volume**: **$277.4\text{ Tbps}$** across $1,000$ demands.
- **Artifact**: `results/figures/q5_heuristic_planner.png`

### 6. Q6: Flex-Grid RSA Reproduction (Velasco et al., 2012)
Implementation and benchmarking of OFDM Elastic Optical Network spectrum allocation:
- Implemented and evaluated **SSA (Starting Slot)**, **SA (Slot Assignment)**, **LP-CA (Link-Path Channel Assignment)**, and **Relaxed RSA**.
- Pre-computed contiguous channel sets $\mathcal{C}(d)$ and coincidence matrix $\gamma_{c, s}$ eliminate Big-M constraints and yield orders-of-magnitude faster solve times on the 21-node Spanish Telefónica backbone.
- **Artifact**: `results/figures/q6_flexgrid_velasco_results.png`

---

## 💼 Resume & Portfolio Ready Bullet Points

```markdown
- Architected and implemented an end-to-end Optical Network Optimization framework in Python (CVXPY, NetworkX, PuLP), formulating exact ILP/MILP models for WDM Routing & Wavelength Assignment (RWA) and Flex-Grid Routing & Spectrum Allocation (RSA).
- Proved theoretical cut-capacity limits on multi-hop optical networks and reduced peak spectrum utilization by 16.7% using Yen's K-shortest link-disjoint path algorithms.
- Formulated a Manycast RWA ILP with joint destination subset selection, balancing traffic across incident fiber links and bypassing high-latency bottleneck paths.
- Engineered a high-throughput semi-static heuristic network planner with distance-adaptive coherent modulation (16-QAM to BPSK) on a 12-node 20-link carrier topology, provisioning 84.9 Tbps of traffic under a strict 1% blocking threshold.
- Reproduced state-of-the-art Flex-Grid RSA formulations (Velasco et al., 2012), validating precomputed contiguous channel generation (LP-CA) that achieved up to 100x speedup over Big-M slot-assignment models on a 21-node backbone.
```

---

## ⚡ Quick Start & Execution

```bash
# Clone the repository
git clone https://github.com/derfang/optical-network-optimization.git
cd optical-network-optimization

# Install dependencies
pip install -r requirements.txt

# Run any module
python -m src.q1_lp_intro.beverage_lp
python -m src.q2_link_ilp.link_rwa
python -m src.q3_path_ilp.path_rwa
python -m src.q4_manycast_rwa.manycast_ilp
python -m src.q5_heuristic_planner.planner
python -m src.q6_flexgrid_paper.rsa_simulation
```

---

## 📄 License

MIT License © 2026 Erfan Ghasry ([derfang](https://github.com/derfang)).
