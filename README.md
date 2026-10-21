# Optical Network Optimization: WDM RWA & Flex-Grid RSA Algorithms

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive framework for **Wavelength Division Multiplexing (WDM) Routing and Wavelength Assignment (RWA)** and **Elastic Optical Network (EON) / Flex-Grid Routing and Spectrum Allocation (RSA)**.

Developed by [derfang](https://github.com/derfang).

---

## 📌 Overview

This repository implements exact mathematical programming formulations (LP, ILP, MILP) and heuristic algorithms for optical network planning, routing, wavelength allocation, and spectrum management:

1. **Q1: Linear & Mixed-Integer Foundations** — Continuous LP vs. Discrete MILP production optimization using CVXPY.
2. **Q2: Link-based RWA ILP** — Flow conservation, wavelength clash/continuity, and hop-bounded maximization of served lightpath requests ($H=1$ vs. $H=2$).
3. **Q3: Path-based RWA ILP** — Yen's $K$-shortest link-disjoint path pre-computation coupled with path-formulated ILP minimizing the maximum assigned wavelength index.
4. **Q4: Manycast RWA ILP** — Joint destination candidate selection and lightpath provisioning under wavelength continuity constraints.
5. **Q5: Heuristic Optical Network Planner** — Dynamic and static heuristic evaluation, blocking probability, and resource utilization analysis.
6. **Q6: Flex-Grid RSA Reproduction** — Implementation and benchmark reproduction of spectrum slot allocation (contiguity & continuity constraints) in OFDM Elastic Optical Networks.

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
│   │   └── plotting.py      # Publication-quality network & spectrum plots
│   ├── q1_lp_intro/
│   │   └── beverage_lp.py   # Continuous LP vs. Discrete 100mL MILP (cvxpy)
│   ├── q2_link_ilp/
│   │   └── link_rwa.py      # Link-based ILP maximizing served requests (H=1 vs H=2)
│   ├── q3_path_ilp/
│   │   └── path_rwa.py      # Yen's K=3 link-disjoint + Path-based ILP minimizing max wavelength
│   ├── q4_manycast_rwa/
│   │   └── manycast_ilp.py  # Destination candidate selection + RWA ILP
│   ├── q5_heuristic_planner/
│   │   └── planner.py       # Heuristic network planner simulation & behavioral analysis
│   └── q6_flexgrid_paper/
│       └── rsa_simulation.py # Reproduction of Flex-Grid RSA / OFDM RMLSA paper
├── results/
│   ├── figures/             # High-res vector/PNG publication figures
│   └── logs/                # Solver output tables, routes, wavelengths, slot logs
└── docs/
    └── formulations.md      # Mathematical LP/ILP formulations in LaTeX
```

---

## ⚡ Quick Start

### 1. Installation

```bash
git clone https://github.com/derfang/optical-network-optimization.git
cd optical-network-optimization
pip install -r requirements.txt
```

### 2. Execution

Run specific modules via CLI:

```bash
# Question 1: Introductory LP/MILP
python -m src.q1_lp_intro.beverage_lp

# Question 2: Link-based RWA ILP
python -m src.q2_link_ilp.link_rwa

# Question 3: Path-based RWA ILP
python -m src.q3_path_ilp.path_rwa

# Question 4: Manycast RWA ILP
python -m src.q4_manycast_rwa.manycast_ilp

# Question 5: Heuristic Planner
python -m src.q5_heuristic_planner.planner

# Question 6: Flex-Grid RSA Simulation
python -m src.q6_flexgrid_paper.rsa_simulation
```

---

## 📐 Mathematical Formulations

See [docs/formulations.md](docs/formulations.md) for full mathematical definitions of sets, decision variables, objective functions, and constraints.

---

## 📊 Results & Benchmarks

All generated network topology visualizations, spectrum slot utilization heatmaps, and performance comparison curves are stored in `results/figures/` with corresponding execution logs in `results/logs/`.
