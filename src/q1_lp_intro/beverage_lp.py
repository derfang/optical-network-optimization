"""
Question 1: Beverage Blending Optimization (Continuous LP vs. Discrete 100mL ILP & BFS)
Course: Optical Networks Optimization
Author: Erfan Ghasry
"""

import os
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt


def solve_continuous_lp() -> dict:
    """
    Part (a): Solves the continuous Linear Programming (LP) problem.
    Variables x_A, x_B represent volume in Liters.
    """
    x_A = cp.Variable(nonneg=True, name="x_A")
    x_B = cp.Variable(nonneg=True, name="x_B")

    objective = cp.Maximize(0.08 * x_A + 0.20 * x_B)
    constraints = [
        4.0 * x_A + 20.0 * x_B <= 10.0,  # Budget <= $10
        x_A + x_B <= 1.0,                # Cup capacity <= 1L
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return {
        "x_A": float(x_A.value),
        "x_B": float(x_B.value),
        "juice": float(problem.value),
        "cost": float(4.0 * x_A.value + 20.0 * x_B.value),
    }


def solve_naive_lp_relaxation_floor() -> dict:
    """
    Part (b) Baseline: Solves continuous LP in 100mL units and floors the result.
    Demonstrates why simple LP relaxation + rounding (0.60L A, 0.30L B -> 0.108L)
    is suboptimal compared to true discrete optimization.
    """
    lp = solve_continuous_lp()
    a_floor = int(np.floor(lp["x_A"] * 10))  # floor(6.25) = 6
    b_floor = int(np.floor(lp["x_B"] * 10))  # floor(3.75) = 3
    juice = (0.08 * a_floor + 0.20 * b_floor) / 10.0
    return {
        "a_packs": a_floor,
        "b_packs": b_floor,
        "x_A": a_floor / 10.0,
        "x_B": b_floor / 10.0,
        "juice": juice,
        "cost": 0.4 * a_floor + 2.0 * b_floor,
    }


def solve_bfs_discrete_search() -> dict:
    """
    Part (b) Method 1: Breadth-First Search (BFS) over the discrete 100mL state space.
    Explores all valid integer package combinations (a, b) to find the global maximum.
    """
    a_max, b_max = 10, 10

    def check_constraint(a: int, b: int) -> bool:
        return (a + b <= 10) and (0.4 * a + 2.0 * b <= 10.0 + 1e-9)

    def pure_juice(a: int, b: int) -> float:
        return 0.08 * a + 0.20 * b  # scaled by 10 (divide by 10 for liters)

    queue = [(0, 0)]
    visited = set()
    best_juice = -1.0
    best_a, best_b = 0, 0

    while queue:
        location = queue.pop(0)
        if location in visited:
            continue
        visited.add(location)
        a, b = location

        # Enforce buying from both drinks (a >= 1, b >= 1) or general maximum
        if a >= 1 and b >= 1 and pure_juice(a, b) > best_juice:
            best_juice = pure_juice(a, b)
            best_a, best_b = a, b

        if a < a_max and check_constraint(a + 1, b):
            queue.append((a + 1, b))
        if b < b_max and check_constraint(a, b + 1):
            queue.append((a, b + 1))

    return {
        "a_packs": best_a,
        "b_packs": best_b,
        "x_A": best_a / 10.0,
        "x_B": best_b / 10.0,
        "juice": best_juice / 10.0,
        "cost": 0.4 * best_a + 2.0 * best_b,
    }


def solve_cvxpy_mip() -> dict:
    """
    Part (b) Method 2: Exact Integer Linear Programming (ILP) using CVXPY's MIP solver.
    """
    a = cp.Variable(integer=True, name="a_packs")
    b = cp.Variable(integer=True, name="b_packs")

    objective = cp.Maximize((0.08 * a + 0.20 * b) / 10.0)
    constraints = [
        0.4 * a + 2.0 * b <= 10.0,
        a + b <= 10,
        a >= 1,
        b >= 1,
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    a_val = int(round(float(a.value)))
    b_val = int(round(float(b.value)))
    return {
        "a_packs": a_val,
        "b_packs": b_val,
        "x_A": a_val / 10.0,
        "x_B": b_val / 10.0,
        "juice": float(problem.value),
        "cost": 0.4 * a_val + 2.0 * b_val,
    }


def plot_results(lp: dict, floor_res: dict, bfs_res: dict, output_dir: str = "results/figures"):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    x_vals = np.linspace(0, 1.05, 400)
    y_cap = 1.0 - x_vals
    y_bud = (10.0 - 4.0 * x_vals) / 20.0
    y_feasible = np.minimum(np.maximum(y_cap, 0), np.maximum(y_bud, 0))

    ax.fill_between(x_vals, 0, y_feasible, where=(x_vals <= 1.0), color="#a8dadc", alpha=0.4, label="LP Feasible Region")
    ax.plot(x_vals, y_cap, "b--", lw=1.6, label=r"Capacity ($x_A + x_B \leq 1\text{ L}$)")
    ax.plot(x_vals, y_bud, "r-.", lw=1.6, label=r"Budget ($4x_A + 20x_B \leq \$10$)")

    pts_x, pts_y = [], []
    for na in range(11):
        for nb in range(11):
            if na + nb <= 10 and 0.4 * na + 2.0 * nb <= 10.0 + 1e-9:
                pts_x.append(na / 10.0)
                pts_y.append(nb / 10.0)
    ax.scatter(pts_x, pts_y, color="#457b9d", s=30, zorder=3, label="Discrete 100mL Lattice")

    # Plot LP Optimum, Naive Floor, and Exact Discrete Optimum
    ax.scatter([lp["x_A"]], [lp["x_B"]], color="#e63946", marker="*", s=220, zorder=5,
               label=f"Continuous LP: ({lp['x_A']:.3f}L, {lp['x_B']:.3f}L) -> {lp['juice']:.3f}L")
    ax.scatter([floor_res["x_A"]], [floor_res["x_B"]], color="#f4a261", marker="s", s=100, zorder=5,
               label=f"Floor-Rounded LP: ({floor_res['x_A']:.2f}L, {floor_res['x_B']:.2f}L) -> {floor_res['juice']:.3f}L")
    ax.scatter([bfs_res["x_A"]], [bfs_res["x_B"]], color="#2a9d8f", marker="D", s=110, zorder=5,
               label=f"BFS / Exact ILP: ({bfs_res['x_A']:.2f}L, {bfs_res['x_B']:.2f}L) -> {bfs_res['juice']:.3f}L")

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 0.60)
    ax.set_xlabel("Drink A Volume (Liters)")
    ax.set_ylabel("Drink B Volume (Liters)")
    ax.set_title("Q1: Continuous LP vs. Floor-Relaxation vs. Exact Discrete Search")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8.5)

    out_file = os.path.join(output_dir, "q1_lp_vs_ilp.png")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


if __name__ == "__main__":
    lp = solve_continuous_lp()
    fl = solve_naive_lp_relaxation_floor()
    bfs = solve_bfs_discrete_search()
    mip = solve_cvxpy_mip()

    print("=== Part (a): Continuous LP ===")
    print(f"Drink A: {lp['x_A']:.3f} L | Drink B: {lp['x_B']:.3f} L | Juice: {lp['juice']:.3f} L")

    print("\n=== Part (b1): LP Relaxation + Floor Rounding ===")
    print(f"Drink A: {fl['x_A']:.2f} L | Drink B: {fl['x_B']:.2f} L | Juice: {fl['juice']:.3f} L")

    print("\n=== Part (b2): BFS Graph Search (100mL Packs) ===")
    print(f"Drink A: {bfs['x_A']:.2f} L | Drink B: {bfs['x_B']:.2f} L | Juice: {bfs['juice']:.3f} L")

    print("\n=== Part (b3): CVXPY Exact MIP Verification ===")
    print(f"Drink A: {mip['x_A']:.2f} L | Drink B: {mip['x_B']:.2f} L | Juice: {mip['juice']:.3f} L")

    plot_results(lp, fl, bfs)
