"""Module 3 — Logistics alert & recommendation agent (ReAct-style).

A deterministic, fully reproducible Observe -> Reason -> Act loop. No
external LLM is required: the reasoning policy is rule-based and every
step is logged for traceability. The architecture mirrors LLM
tool-calling (tools registry + reasoning step + action selection), so
swapping the policy for an LLM is a localized change (see README).

Tools exposed to the agent:
  - get_stock_status(warehouse_id)        -> inventory level
  - get_demand_forecast(point_id, weeks)  -> Module 2 forecast model
  - run_optimization(scenario)            -> Module 1 solver
  - send_alert(point_id, message, severity) -> simulated notification

Stop criterion: every demand point flagged as high-risk has been
attended (alert sent + reallocation recommended), or max_steps reached.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.ml_models import (
    ForecastResult,
    RISK_FEATURES,
    attach_forecast_projection,
    forecast_demand,
    train_forecast_models,
    train_risk_classifier,
)
from src.optimization import solve_transport
from src.utils import ROOT, SEED, generate_network, generate_risk_features, set_seeds


# ------------------------------------------------------------------ tools
class Tools:
    """Tool registry. Each method = one tool callable by the agent."""

    def __init__(self, seed: int = SEED):
        rng = np.random.default_rng(seed + 3)
        net = generate_network(seed)
        self._wh = net["warehouses"].set_index("warehouse_id")
        self._dp = net["demand_points"].set_index("point_id")
        # Simulated current stock per warehouse: 25–95% of capacity
        self._stock = {
            w: int(c * rng.uniform(0.25, 0.95))
            for w, c in self._wh["capacity"].items()
        }
        self._forecast_result: ForecastResult = train_forecast_models(seed)
        self._risk_model = train_risk_classifier(seed)
        # Latest-week snapshot of risk features (the agent's "now"). projected_demand
        # comes from the Module 2A forecast, matching how the risk model was trained.
        rf = attach_forecast_projection(generate_risk_features(seed), seed)
        self._snapshot = rf[rf["week"] == rf["week"].max()].set_index("point_id")
        self.alerts_sent: list[dict] = []

    def get_stock_status(self, warehouse_id: str) -> dict:
        """Tool: current inventory level and capacity of one warehouse."""
        return {"warehouse_id": warehouse_id, "stock": self._stock[warehouse_id],
                "capacity": int(self._wh.loc[warehouse_id, "capacity"])}

    def get_demand_forecast(self, point_id: str, weeks: int = 4) -> dict:
        """Tool: multi-step demand forecast for a point (Module 2A model)."""
        preds = forecast_demand(self._forecast_result, point_id, weeks)
        return {"point_id": point_id, "weeks": weeks, "forecast": preds}

    def run_optimization(self, scenario: str = "base") -> dict:
        """Tool: re-solve the network (Module 1) under a demand scenario."""
        mult = {"base": 1.0, "demand_plus20": 1.2}.get(scenario, 1.0)
        res = solve_transport(demand_multiplier=mult)
        return {"scenario": scenario, "status": res.status,
                "total_cost": res.total_cost,
                "max_utilization_pct": float(res.utilization["utilization_pct"].max())}

    def send_alert(self, point_id: str, message: str, severity: str) -> dict:
        """Tool: emit (simulate) a client notification and record it."""
        # Deterministic synthetic timestamp (1 min apart) so the trace artifact
        # is byte-identical across runs — no wall-clock dependency.
        ts = (
            datetime(2026, 1, 1, tzinfo=timezone.utc)
            + timedelta(minutes=len(self.alerts_sent))
        ).isoformat()
        alert = {"point_id": point_id, "message": message, "severity": severity,
                 "ts": ts}
        self.alerts_sent.append(alert)
        return {"delivered": True, **alert}

    # -- internal helper (not an agent tool): nearest warehouse to a point
    def nearest_warehouse(self, point_id: str) -> str:
        """Warehouse id geographically closest to a demand point."""
        p = self._dp.loc[point_id, ["x", "y"]].to_numpy(dtype=float)
        w = self._wh[["x", "y"]].to_numpy(dtype=float)
        return str(self._wh.index[int(np.argmin(np.linalg.norm(w - p, axis=1)))])

    # -- internal helper (not an agent tool): risk scores for observation
    def risk_scores(self) -> pd.Series:
        """Stockout-risk probability per demand point (sorted desc)."""
        proba = self._risk_model.model.predict_proba(
            self._snapshot[RISK_FEATURES]
        )[:, 1]
        return pd.Series(proba, index=self._snapshot.index).sort_values(ascending=False)


# ------------------------------------------------------------------ agent
@dataclass
class LogisticsAgent:
    """Rule-based ReAct agent: scores risk, alerts, and re-optimizes.

    Holds the tool registry, a risk threshold for flagging points, a safety
    cap on steps, and the running trace of every observe/reason/act/stop entry.
    """

    tools: Tools
    risk_threshold: float = 0.5
    max_steps: int = 60
    trace: list[dict] = field(default_factory=list)

    def _log(self, step: int, phase: str, content: dict | str) -> None:
        entry = {"step": step, "phase": phase, "content": content}
        self.trace.append(entry)

    def run(self) -> dict:
        """ReAct loop: Observe -> Reason -> Act, until stop criterion."""
        step = 0

        # OBSERVE (initial): score every demand point
        scores = self.tools.risk_scores()
        at_risk = list(scores[scores >= self.risk_threshold].index)
        self._log(step, "observe", {
            "n_points": len(scores),
            "n_at_risk": len(at_risk),
            "at_risk": {p: round(float(scores[p]), 3) for p in at_risk},
        })
        self._log(step, "reason",
                  f"{len(at_risk)} points exceed risk threshold "
                  f"{self.risk_threshold}. Plan: for each, check demand "
                  f"forecast and nearest warehouse stock, then alert with "
                  f"severity from risk score, demand trend and warehouse stock.")

        pending = list(at_risk)
        reoptimize_needed = False

        while pending and step < self.max_steps:
            step += 1
            pid = pending.pop(0)
            score = float(scores[pid])

            # ACT: gather context via tools
            fc = self.tools.get_demand_forecast(pid, weeks=4)
            self._log(step, "act", {"tool": "get_demand_forecast", "args": {"point_id": pid}, "result": fc})

            # ACT: check the warehouse that would actually serve this point
            wh_id = self.tools.nearest_warehouse(pid)
            wh = self.tools.get_stock_status(wh_id)
            self._log(step, "act", {"tool": "get_stock_status", "args": {"warehouse_id": wh_id}, "result": wh})

            snap = self.tools._snapshot.loc[pid]
            trend_up = fc["forecast"][-1] > fc["forecast"][0]
            # Can the serving warehouse cover this point's next-4-weeks demand?
            need_4w = float(sum(fc["forecast"]))
            wh_can_cover = wh["stock"] >= need_4w
            wh_low = wh["stock"] < 0.35 * wh["capacity"]

            # OBSERVE + REASON on gathered context. The serving warehouse being
            # low or unable to cover demand escalates an otherwise-medium case.
            severity = (
                "HIGH"
                if score >= 0.8 or (score >= 0.6 and (trend_up or wh_low or not wh_can_cover))
                else "MEDIUM"
            )
            reason = (
                f"{pid}: risk={score:.2f}, stock={snap['current_stock']}, "
                f"lead_time={snap['lead_time_days']}d, 4-week forecast="
                f"{fc['forecast']} ({'rising' if trend_up else 'stable/falling'}). "
                f"Nearest warehouse {wh_id} stock={wh['stock']}/{wh['capacity']} "
                f"({'can' if wh_can_cover else 'CANNOT'} cover ~{need_4w:.0f}u 4-week need"
                f"{', running low' if wh_low else ''}). Severity -> {severity}."
            )
            self._log(step, "reason", reason)
            if severity == "HIGH":
                reoptimize_needed = True

            # ACT: send alert. Recommendation is tailored to actual warehouse stock.
            action = (
                f"expedite replenishment from {wh_id}"
                if wh_can_cover
                else f"replenish from {wh_id} and pull from a secondary warehouse "
                f"({wh_id} alone cannot cover the projected need)"
            )
            msg = (
                f"Stockout risk {score:.0%}. Current stock {snap['current_stock']} vs "
                f"projected demand {snap['projected_demand']} (lead time "
                f"{snap['lead_time_days']}d). Forecast next 4 weeks: {fc['forecast']}. "
                f"Nearest warehouse {wh_id}: {wh['stock']}/{wh['capacity']} units. "
                f"Recommendation: {action}."
            )
            res = self.tools.send_alert(pid, msg, severity)
            self._log(step, "act", {"tool": "send_alert", "args": {"point_id": pid, "severity": severity}, "result": {"delivered": res["delivered"]}})

        # Optional final action: re-run optimization under stressed demand
        if reoptimize_needed and step < self.max_steps:
            step += 1
            self._log(step, "reason",
                      "HIGH-severity points detected -> evaluate network under "
                      "+20% demand to validate reallocation feasibility.")
            opt = self.tools.run_optimization("demand_plus20")
            self._log(step, "act", {"tool": "run_optimization", "args": {"scenario": "demand_plus20"}, "result": opt})
            self._log(step, "observe",
                      f"Re-optimization {opt['status']}, cost ${opt['total_cost']:,.0f}, "
                      f"max warehouse utilization {opt['max_utilization_pct']:.0f}%.")

        stopped = "all at-risk points attended" if not pending else "max_steps reached"
        self._log(step, "stop", {"criterion": stopped,
                                 "alerts_sent": len(self.tools.alerts_sent)})
        return {
            "at_risk": at_risk,
            "alerts": self.tools.alerts_sent,
            "trace": self.trace,
            "stop_reason": stopped,
        }

    def save_trace(self, path=None) -> str:
        """Persist the full reasoning trace to JSON (default: data/)."""
        path = path or (ROOT / "data" / "agent_trace.json")
        with open(path, "w") as f:
            json.dump(self.trace, f, indent=2, default=str)
        return str(path)


def run_module3() -> dict:
    """Entry point used by main.py: run the agent and save its trace."""
    set_seeds()
    tools = Tools()
    agent = LogisticsAgent(tools=tools)
    result = agent.run()
    trace_path = agent.save_trace()
    result["trace_path"] = trace_path
    return result


if __name__ == "__main__":
    out = run_module3()
    print(f"At-risk points: {out['at_risk']}")
    print(f"Alerts sent: {len(out['alerts'])} | Stop: {out['stop_reason']}")
    print(f"Trace saved to: {out['trace_path']}\n")
    for t in out["trace"][:8]:
        print(f"[{t['step']:>2}] {t['phase']:<8} {str(t['content'])[:110]}")
