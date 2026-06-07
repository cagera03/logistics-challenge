"""Module 1 — Capacitated transportation problem (PuLP).

Minimize total distribution cost from 5 warehouses to 20 demand points
subject to warehouse capacity and full demand satisfaction.

    min  sum_{i,j} c_ij * x_ij
    s.t. sum_j x_ij <= cap_i      (capacity, per warehouse i)
         sum_i x_ij >= dem_j      (demand, per point j)
         x_ij >= 0

Why PuLP: declarative, readable formulation; CBC solver ships with the
package so the evaluator needs no extra installs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pulp

from src.utils import FIG_DIR, SEED, generate_network, set_seeds


@dataclass
class OptimizationResult:
    """Solved transportation problem: status, cost, flows and utilization.

    `utilization` includes per-warehouse capacity, used units, % utilization
    and the capacity shadow price (dual).
    """

    status: str
    total_cost: float
    flows: pd.DataFrame  # columns: warehouse_id, point_id, units, cost
    utilization: pd.DataFrame = field(default=None)  # incl. capacity shadow price


def solve_transport(
    demand_multiplier: float = 1.0, seed: int = SEED
) -> OptimizationResult:
    """Solve the capacitated transportation problem.

    demand_multiplier scales every demand (1.2 = +20% scenario) so the
    same function serves the base case, sensitivity analysis, and the
    agent's `run_optimization` tool.
    """
    net = generate_network(seed)
    costs, wh, dp = net["costs"], net["warehouses"], net["demand_points"]
    cap = dict(zip(wh["warehouse_id"], wh["capacity"]))
    dem = {
        r["point_id"]: r["demand"] * demand_multiplier for _, r in dp.iterrows()
    }

    prob = pulp.LpProblem("transport_min_cost", pulp.LpMinimize)
    x = {
        (i, j): pulp.LpVariable(f"x_{i}_{j}", lowBound=0)
        for i in cap
        for j in dem
    }
    prob += pulp.lpSum(costs.loc[i, j] * x[i, j] for i, j in x)
    for i in cap:
        prob += pulp.lpSum(x[i, j] for j in dem) <= cap[i], f"cap_{i}"
    for j in dem:
        prob += pulp.lpSum(x[i, j] for i in cap) >= dem[j], f"dem_{j}"

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]

    # Dual values (shadow prices). For a capacity constraint sum_j x_ij <= cap_i
    # the dual is the marginal change in total cost per extra unit of capacity:
    # the most negative one points at the warehouse worth expanding first.
    cap_shadow = {i: prob.constraints[f"cap_{i}"].pi for i in cap}

    rows = [
        {
            "warehouse_id": i,
            "point_id": j,
            "units": x[i, j].value(),
            "cost": x[i, j].value() * costs.loc[i, j],
        }
        for i, j in x
        if x[i, j].value() and x[i, j].value() > 1e-6
    ]
    flows = pd.DataFrame(rows)
    used = flows.groupby("warehouse_id")["units"].sum() if not flows.empty else pd.Series(dtype=float)
    utilization = pd.DataFrame(
        {
            "capacity": pd.Series(cap),
            "used": used.reindex(cap.keys()).fillna(0),
        }
    )
    utilization["utilization_pct"] = (
        100 * utilization["used"] / utilization["capacity"]
    ).round(1)
    # Marginal value of one extra unit of capacity ($ saved). 0 for slack
    # warehouses; negative where the capacity constraint binds.
    utilization["shadow_price"] = pd.Series(cap_shadow).reindex(cap.keys()).round(3)

    total = pulp.value(prob.objective) if status == "Optimal" else float("nan")
    return OptimizationResult(status, round(total, 2), flows, utilization)


def sensitivity_analysis(seed: int = SEED) -> pd.DataFrame:
    """Compare base case vs +20% demand (plus intermediate steps)."""
    rows = []
    for m in [1.0, 1.05, 1.10, 1.15, 1.20]:
        res = solve_transport(demand_multiplier=m, seed=seed)
        rows.append(
            {
                "demand_multiplier": m,
                "status": res.status,
                "total_cost": res.total_cost,
                "max_utilization_pct": res.utilization["utilization_pct"].max(),
            }
        )
    return pd.DataFrame(rows)


def plot_solution(res: OptimizationResult, seed: int = SEED, suffix: str = "base") -> None:
    """Flow map + warehouse utilization chart, saved under figures/."""
    FIG_DIR.mkdir(exist_ok=True)
    net = generate_network(seed)
    wh, dp = net["warehouses"], net["demand_points"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    max_u = res.flows["units"].max()
    for _, f in res.flows.iterrows():
        w = wh[wh["warehouse_id"] == f["warehouse_id"]].iloc[0]
        p = dp[dp["point_id"] == f["point_id"]].iloc[0]
        ax.plot(
            [w["x"], p["x"]],
            [w["y"], p["y"]],
            color="steelblue",
            alpha=0.5,
            lw=0.5 + 3 * f["units"] / max_u,
        )
    ax.scatter(wh["x"], wh["y"], s=wh["capacity"] / 5, c="crimson", zorder=3, label="Warehouses (size=capacity)")
    ax.scatter(dp["x"], dp["y"], s=dp["demand"] / 2, c="darkgreen", zorder=3, label="Demand points (size=demand)")
    for _, w in wh.iterrows():
        ax.annotate(w["warehouse_id"], (w["x"], w["y"]), fontsize=9, fontweight="bold")
    ax.set_title(f"Optimal flow map ({suffix}) — total cost ${res.total_cost:,.0f}")
    ax.legend(fontsize=8)

    ax = axes[1]
    u = res.utilization
    ax.bar(u.index, u["capacity"], color="lightgray", label="Capacity")
    ax.bar(u.index, u["used"], color="steelblue", label="Used")
    for k, (idx, r) in enumerate(u.iterrows()):
        ax.text(k, r["used"] + 20, f"{r['utilization_pct']:.0f}%", ha="center", fontsize=9)
    ax.set_title("Warehouse utilization")
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"optimization_{suffix}.png", dpi=150)
    plt.close(fig)


def run_module1() -> dict:
    """Entry point used by main.py. Returns key results."""
    set_seeds()
    base = solve_transport()
    plot_solution(base, suffix="base")
    stressed = solve_transport(demand_multiplier=1.2)
    plot_solution(stressed, suffix="demand_plus20")
    sens = sensitivity_analysis()

    FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sens["demand_multiplier"], sens["total_cost"], marker="o")
    ax.set_xlabel("Demand multiplier")
    ax.set_ylabel("Total cost ($)")
    ax.set_title("Sensitivity: total cost vs demand growth")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sensitivity_cost.png", dpi=150)
    plt.close(fig)

    return {"base": base, "stressed": stressed, "sensitivity": sens}


if __name__ == "__main__":
    out = run_module1()
    print(f"Base:     {out['base'].status}, cost=${out['base'].total_cost:,.2f}")
    print(f"+20%:     {out['stressed'].status}, cost=${out['stressed'].total_cost:,.2f}")
    print(out["sensitivity"].to_string(index=False))
