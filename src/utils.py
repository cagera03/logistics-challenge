"""Shared utilities: reproducibility, paths, and synthetic data generation.

All randomness in the project flows through the single SEED defined here.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_WAREHOUSES = 5
N_DEMAND_POINTS = 20
N_WEEKS = 52

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"


def set_seeds(seed: int = SEED) -> None:
    """Fix every random source used in the project."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def generate_network(seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Generate the synthetic logistics network (Module 1 inputs).

    Returns dict with:
      - costs: transport cost matrix (warehouse x demand point), $/unit
      - warehouses: capacity [500-2000] and coordinates
      - demand_points: weekly demand [50-300] and coordinates
    """
    rng = np.random.default_rng(seed)

    wh_ids = [f"W{i+1}" for i in range(N_WAREHOUSES)]
    dp_ids = [f"D{j+1}" for j in range(N_DEMAND_POINTS)]

    # Geographic coordinates on a 100x100 grid (used for distance & plots)
    wh_xy = rng.uniform(10, 90, size=(N_WAREHOUSES, 2))
    dp_xy = rng.uniform(0, 100, size=(N_DEMAND_POINTS, 2))

    # Cost = base distance cost + random handling component
    dist = np.linalg.norm(wh_xy[:, None, :] - dp_xy[None, :, :], axis=2)
    costs = np.round(0.08 * dist + rng.uniform(1.0, 4.0, size=dist.shape), 2)

    warehouses = pd.DataFrame(
        {
            "warehouse_id": wh_ids,
            "capacity": rng.integers(500, 2001, size=N_WAREHOUSES),
            "x": wh_xy[:, 0],
            "y": wh_xy[:, 1],
        }
    )
    demand_points = pd.DataFrame(
        {
            "point_id": dp_ids,
            "demand": rng.integers(50, 301, size=N_DEMAND_POINTS),
            "x": dp_xy[:, 0],
            "y": dp_xy[:, 1],
        }
    )
    cost_df = pd.DataFrame(costs, index=wh_ids, columns=dp_ids)
    return {"costs": cost_df, "warehouses": warehouses, "demand_points": demand_points}


def generate_demand_series(seed: int = SEED) -> pd.DataFrame:
    """Generate 52-week demand series per point (Module 2A input).

    demand_t = base * (1 + trend_t) * seasonality_t + noise
    Seasonality: annual sine cycle with point-specific phase.
    """
    rng = np.random.default_rng(seed + 1)
    net = generate_network(seed)
    dp = net["demand_points"]

    weeks = np.arange(N_WEEKS)
    rows = []
    for _, r in dp.iterrows():
        base = r["demand"]
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.10, 0.30)
        trend = rng.uniform(-0.002, 0.004)  # weekly drift
        season = 1 + amp * np.sin(2 * np.pi * weeks / 52 + phase)
        noise = rng.normal(0, 0.07 * base, size=N_WEEKS)
        series = np.maximum(base * (1 + trend * weeks) * season + noise, 5).round(1)
        for w, v in zip(weeks, series):
            rows.append({"point_id": r["point_id"], "week": int(w), "demand": v})
    return pd.DataFrame(rows)


def generate_risk_features(seed: int = SEED) -> pd.DataFrame:
    """Generate stockout-risk dataset per point/week (Module 2B input).

    Features: current_stock, lead_time_days, projected_demand, distance_km.
    Target risk_high = 1 when projected coverage (stock vs demand over lead
    time) falls short — plus label noise so the problem isn't trivially
    separable. Class imbalance arises naturally (~20% positives).
    """
    rng = np.random.default_rng(seed + 2)
    series = generate_demand_series(seed)
    net = generate_network(seed)
    dp = net["demand_points"].set_index("point_id")
    wh = net["warehouses"]

    # Distance from each point to its nearest warehouse
    wh_xy = wh[["x", "y"]].to_numpy()
    rows = []
    for pid, grp in series.groupby("point_id"):
        p_xy = dp.loc[pid, ["x", "y"]].to_numpy(dtype=float)
        dist_km = float(np.min(np.linalg.norm(wh_xy - p_xy, axis=1)) * 4.2)
        for _, r in grp.iterrows():
            proj = r["demand"] * rng.uniform(0.9, 1.15)
            stock = rng.uniform(0.3, 2.5) * proj
            lead = rng.integers(1, 11)
            # Coverage ratio: weeks of stock vs lead time (+ distance penalty)
            coverage = stock / (proj * (1 + lead / 14) * (1 + dist_km / 800))
            risk = int(coverage < 0.55)
            if rng.uniform() < 0.06:  # label noise
                risk = 1 - risk
            rows.append(
                {
                    "point_id": pid,
                    "week": int(r["week"]),
                    "current_stock": round(stock, 1),
                    "lead_time_days": int(lead),
                    "projected_demand": round(proj, 1),
                    "distance_km": round(dist_km, 1),
                    "risk_high": risk,
                }
            )
    return pd.DataFrame(rows)


def save_all_data() -> None:
    """Materialize every synthetic dataset under data/ (reproducible)."""
    DATA_DIR.mkdir(exist_ok=True)
    net = generate_network()
    net["costs"].to_csv(DATA_DIR / "transport_costs.csv")
    net["warehouses"].to_csv(DATA_DIR / "warehouses.csv", index=False)
    net["demand_points"].to_csv(DATA_DIR / "demand_points.csv", index=False)
    generate_demand_series().to_csv(DATA_DIR / "demand_series.csv", index=False)
    generate_risk_features().to_csv(DATA_DIR / "risk_features.csv", index=False)


if __name__ == "__main__":
    set_seeds()
    save_all_data()
    print(f"Synthetic data written to {DATA_DIR}")
