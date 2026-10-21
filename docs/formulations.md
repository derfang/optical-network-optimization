# Mathematical Formulations: Optical Network Optimization

This document contains the formal mathematical derivations, sets, parameters, decision variables, objectives, and constraints for all linear and mixed-integer linear programming formulations in the project.

---

## Question 1: Beverage Production Optimization (LP vs. MILP)

### Problem Description
A beverage blending optimization problem used as an introductory foundation for mathematical programming, demonstrating continuous Linear Programming (LP), LP relaxation floor-rounding suboptimality, state-space Breadth-First Search (BFS), and exact Mixed-Integer Linear Programming (MILP).

### 1. Part (a): Continuous Linear Program (LP)
Find the optimal blend of Drink $A$ and Drink $B$ in liters to maximize the total volume of pure fruit juice under budget and volume capacity limits.

#### Decision Variables
- $x_A \ge 0$: Volume of Drink $A$ in liters.
- $x_B \ge 0$: Volume of Drink $B$ in liters.

#### Objective Function
Maximize total pure fruit juice ($8\%$ in Drink $A$, $20\%$ in Drink $B$):
$$\max \quad Z_{\text{LP}} = 0.08 x_A + 0.20 x_B$$

#### Constraints
1. **Budget Limitation** (\$4/L for $A$, \$20/L for $B$, total budget \$10):
   $$4.0 x_A + 20.0 x_B \le 10.0$$
2. **Container Capacity** (Maximum container volume of 1.0 L):
   $$x_A + x_B \le 1.0$$
3. **Non-negativity**:
   $$x_A \ge 0, \quad x_B \ge 0$$

---

### 2. Part (b): Discrete Integer Linear Program (100 mL Units)
When drinks are sold exclusively in discrete 100 mL packages, let $a, b \in \mathbb{Z}^+$ denote the number of 100 mL packages of Drink $A$ and Drink $B$ respectively ($x_A = a/10$, $x_B = b/10$).

#### Decision Variables
- $a \in \mathbb{Z}^+$: Number of 100 mL packs of Drink $A$.
- $b \in \mathbb{Z}^+$: Number of 100 mL packs of Drink $B$.

#### Objective Function
$$\max \quad Z_{\text{ILP}} = \frac{0.08 a + 0.20 b}{10}$$

#### Constraints
1. **Discrete Budget Constraint** (\$0.40/pack for $A$, \$2.00/pack for $B$):
   $$0.4 a + 2.0 b \le 10.0$$
2. **Discrete Container Capacity** (At most 10 packs of 100 mL = 1.0 L):
   $$a + b \le 10$$
3. **Product Inclusion**:
   $$a \ge 1, \quad b \ge 1, \quad a, b \in \mathbb{Z}$$

#### Comparison of Solution Approaches
- **Continuous LP Relaxation + Floor Rounding**:
  Solving continuous LP gives $(x_A^*, x_B^*) = (0.625\text{ L}, 0.375\text{ L})$. Floor rounding to 100 mL packages yields $a = \lfloor 6.25 \rfloor = 6$ ($0.60$ L) and $b = \lfloor 3.75 \rfloor = 3$ ($0.30$ L), delivering **$0.108\text{ L}$** pure juice for $\$8.40$. This highlights the suboptimality of naive heuristic rounding.
- **BFS State-Space Search & Exact MILP Solver**:
  Exploring the discrete integer lattice finds the global discrete optimum at $(a^*, b^*) = (5, 4)$, representing $0.50\text{ L}$ Drink $A$ and $0.40\text{ L}$ Drink $B$, achieving **$0.120\text{ L}$** pure juice at exactly $\$10.00$.

---

## Question 2: Link-based ILP for RWA (Max Served Requests)

### Problem Description
Given a physical optical topology graph $G = (V, E)$ and a static traffic demand matrix $A_{|V| \times |V|}$, route and assign wavelengths to lightpath requests without wavelength converters (wavelength continuity constraint) to maximize total served traffic requests, subject to a maximum hop limit $H$.

### Sets and Parameters
- $V$: Set of optical network nodes ($|V| = 5$).
- $E$: Set of physical fiber links: $E = \{(1, 2), (1, 3), (1, 5), (2, 5), (3, 4)\}$.
- $A$: Set of directed fiber arcs: $A = \{(u, v), (v, u) \mid \{u, v\} \in E\}$.
- $W$: Set of available optical wavelengths per link: $W = \{1, 2, \dots, |W|\}$.
- $K$: Set of active connection requests $(s, d)$ where $A[s, d] = 1$.
- $H$: Maximum allowable hop count parameter.
  - *Intermediate Hops Interpretation*: Path contains at most $H$ intermediate nodes $\implies \text{max\_links} = H + 1$.
  - *Total Links Interpretation*: Path traverses at most $H$ physical links $\implies \text{max\_links} = H$.

### Decision Variables
- $y_{s, d, w} \in \{0, 1\}$: Binary variable equal to $1$ if connection request $(s, d) \in K$ is served on wavelength $w \in W$; $0$ otherwise.
- $f_{s, d, w}^{u, v} \in \{0, 1\}$: Binary routing variable equal to $1$ if lightpath $(s, d)$ on wavelength $w$ traverses directed arc $(u, v) \in A$; $0$ otherwise.

### Objective Function
Maximize the total number of successfully established lightpath requests:
$$\max \quad \sum_{(s, d) \in K} \sum_{w \in W} y_{s, d, w}$$

### Constraints

1. **Single Wavelength Allocation per Demand**:
   Each admitted connection request is assigned to at most one optical channel:
   $$\sum_{w \in W} y_{s, d, w} \le 1, \quad \forall (s, d) \in K$$

2. **Flow Conservation & Wavelength Continuity**:
   Ensures end-to-end lightpath continuity on the same wavelength $w$ across intermediate transparent optical nodes (no wavelength conversion):
   $$\sum_{v: (u, v) \in A} f_{s, d, w}^{u, v} - \sum_{v: (v, u) \in A} f_{s, d, w}^{v, u} = \begin{cases} 
   y_{s, d, w}, & u = s \\ 
   -y_{s, d, w}, & u = d \\ 
   0, & \forall u \in V \setminus \{s, d\} 
   \end{cases} \quad \forall (s, d) \in K, \forall w \in W$$
   Additionally, inflow at source $s$ and outflow at destination $d$ are strictly zero:
   $$\sum_{v: (v, s) \in A} f_{s, d, w}^{v, s} = 0, \quad \sum_{v: (d, v) \in A} f_{s, d, w}^{d, v} = 0, \quad \forall (s, d) \in K, \forall w \in W$$

3. **Subtour & Self-Loop Elimination**:
   Precludes cyclic flow across non-terminal nodes:
   $$\sum_{v: (u, v) \in A} f_{s, d, w}^{u, v} \le y_{s, d, w}, \quad \forall u \in V \setminus \{d\}, \forall (s, d) \in K, \forall w \in W$$

4. **Maximum Hop Bound Constraint**:
   Constrains path length according to $H$:
   $$\sum_{(u, v) \in A} f_{s, d, w}^{u, v} \le L_{\max} \cdot y_{s, d, w}, \quad \forall (s, d) \in K, \forall w \in W$$
   where $L_{\max} = H + 1$ (for intermediate node limits) or $L_{\max} = H$ (for total link limits).

5. **Wavelength Clash & Capacity Constraint**:
   Two lightpaths sharing the same fiber cannot be assigned the same wavelength:
   - *Directed Arcs*:
     $$\sum_{(s, d) \in K} f_{s, d, w}^{u, v} \le 1, \quad \forall (u, v) \in A, \forall w \in W$$
   - *Undirected Edges*:
     $$\sum_{(s, d) \in K} \left( f_{s, d, w}^{u, v} + f_{s, d, w}^{v, u} \right) \le 1, \quad \forall \{u, v\} \in E, \forall w \in W$$

---

## Question 3: Path-based ILP for RWA (Min Max Wavelength Index)

*(LaTeX formulations to be populated)*

---

## Question 4: Manycast RWA ILP

*(LaTeX formulations to be populated)*

---

## Question 5: Heuristic Network Planner

*(Algorithm pseudo-code and formulation details to be populated)*

---

## Question 6: Flex-Grid RSA / OFDM RMLSA Formulation

*(LaTeX formulations to be populated)*
