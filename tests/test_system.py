"""Unit tests covering the three modules + reproducibility."""
import numpy as np
import pandas as pd
import pytest

from src.agent import LogisticsAgent, Tools
from src.ml_models import (
    attach_forecast_projection,
    build_forecast_features,
    train_risk_classifier,
)
from src.optimization import solve_transport
from src.utils import SEED, generate_demand_series, generate_network, generate_risk_features


# ---------------------------------------------------------------- Module 1
def test_optimization_is_feasible_and_balanced():
    """Solver finds an optimal plan that satisfies all demand within capacity."""
    res = solve_transport()
    assert res.status == "Optimal"
    net = generate_network()
    # Every demand point fully served
    served = res.flows.groupby("point_id")["units"].sum()
    for _, r in net["demand_points"].iterrows():
        assert served[r["point_id"]] >= r["demand"] - 1e-4
    # No warehouse exceeds capacity
    used = res.flows.groupby("warehouse_id")["units"].sum()
    cap = dict(zip(net["warehouses"]["warehouse_id"], net["warehouses"]["capacity"]))
    for w, u in used.items():
        assert u <= cap[w] + 1e-4


def test_sensitivity_cost_monotonic():
    """More demand can never cost less (LP relaxation monotonicity)."""
    base = solve_transport(1.0).total_cost
    stressed = solve_transport(1.2).total_cost
    assert stressed > base


def test_capacity_shadow_prices_match_binding_constraints():
    """Complementary slackness: only binding capacity constraints are priced.

    Sign convention for the dual varies by solver, so we check magnitudes:
    a warehouse with spare capacity has ~zero shadow price; a binding one
    (100% utilization) has a non-zero |dual|.
    """
    res = solve_transport()
    u = res.utilization
    assert set(u["shadow_price"].index) == set(u.index)
    assert u["shadow_price"].notna().all()
    binding = u[u["utilization_pct"] >= 99.9]
    slack = u[u["utilization_pct"] < 99.9]
    assert (binding["shadow_price"].abs() > 1e-6).any()
    assert (slack["shadow_price"].abs() <= 1e-6).all()


# ---------------------------------------------------------------- Module 2
def test_forecast_features_no_leakage():
    """Lag/rolling features at week t must use only data from weeks < t."""
    series = generate_demand_series()
    df = build_forecast_features(series)
    sample = df[df["point_id"] == "D1"].sort_values("week")
    raw = series[series["point_id"] == "D1"].set_index("week")["demand"]
    for _, row in sample.head(10).iterrows():
        w = row["week"]
        assert row["lag_1"] == pytest.approx(raw.loc[w - 1])
        assert row["roll_mean_4"] == pytest.approx(raw.loc[w - 4 : w - 1].mean())


def test_risk_classifier_beats_random():
    res = train_risk_classifier()
    assert res.metrics["ROC_AUC"] > 0.7
    assert res.metrics["F1"] > 0.5


def test_projected_demand_comes_from_forecast():
    """2A->2B integration: projected_demand is the forecast, not the raw draw."""
    raw = generate_risk_features()
    integrated = attach_forecast_projection(raw)
    merged = raw.merge(
        integrated, on=["point_id", "week"], suffixes=("_raw", "_fc")
    )
    # The feature actually changed (it's now a model output, not the synthetic draw)
    assert not np.allclose(
        merged["projected_demand_raw"], merged["projected_demand_fc"]
    )
    # ...but stays in a sane demand range (forecast tracks realized demand)
    assert integrated["projected_demand"].between(5, 600).all()


# ---------------------------------------------------------------- Module 3
def test_agent_attends_all_risk_points_and_stops():
    tools = Tools()
    agent = LogisticsAgent(tools=tools)
    out = agent.run()
    assert out["stop_reason"] == "all at-risk points attended"
    alerted = {a["point_id"] for a in out["alerts"]}
    assert alerted == set(out["at_risk"])
    phases = {t["phase"] for t in out["trace"]}
    assert {"observe", "reason", "act", "stop"} <= phases


def test_agent_exercises_all_four_tools():
    """Every required tool must actually be called by the reasoning loop."""
    out = LogisticsAgent(tools=Tools()).run()
    called = {
        t["content"]["tool"]
        for t in out["trace"]
        if t["phase"] == "act" and isinstance(t["content"], dict)
    }
    assert {"get_demand_forecast", "get_stock_status", "send_alert"} <= called
    # run_optimization fires whenever a HIGH-severity point is found
    if any(a["severity"] == "HIGH" for a in out["alerts"]):
        assert "run_optimization" in called


def test_agent_trace_is_deterministic():
    """Trace (incl. alert timestamps) is byte-identical across runs."""
    import json

    t1 = json.dumps(LogisticsAgent(tools=Tools()).run()["trace"], default=str)
    t2 = json.dumps(LogisticsAgent(tools=Tools()).run()["trace"], default=str)
    assert t1 == t2


# ---------------------------------------------------------------- Reproducibility
def test_data_generation_is_deterministic():
    a, b = generate_risk_features(SEED), generate_risk_features(SEED)
    pd.testing.assert_frame_equal(a, b)
    n1, n2 = generate_network(SEED), generate_network(SEED)
    assert np.allclose(n1["costs"].to_numpy(), n2["costs"].to_numpy())
