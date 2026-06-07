# Logistics Challenge — Optimization · ML · Agents

Integrated decision system for an industrial logistics network: **5 regional
warehouses → 20 demand points**. Three weekly decisions, solved by three
integrated modules:

| Module | Decision | Approach |
|---|---|---|
| 1 · Optimization | How to allocate flows | Capacitated transportation LP (PuLP/CBC) |
| 2 · ML | How much demand to expect & where risk is | RF/XGBoost forecast (walk-forward) + risk classifier (SHAP) |
| 3 · Agent | When to alert clients proactively | Deterministic ReAct loop calling Modules 1–2 as tools |

## Quickstart

```bash
pip install -r requirements.txt
python main.py          # runs the full pipeline end to end
pytest tests/ -q        # 10 unit tests
```

`main.py` regenerates all synthetic data (seed=42), solves the base and +20%
demand scenarios, trains and validates both ML models, and runs the agent.
Outputs land in `data/` (CSVs + `agent_trace.json`) and `figures/` (PNGs).
Two consecutive runs produce byte-identical output — console **and**
`agent_trace.json` (the trace uses synthetic, not wall-clock, timestamps).
Determinism is guaranteed *per environment* (hence the pinned versions in
`requirements.txt`): across OS / Python versions, floating-point differences
in the tree libraries can nudge risk scores sitting near the 0.5 threshold,
so the at-risk count may vary slightly (e.g. 8–9 points). Reference numbers
below come from the environment that produced the committed figures.

### Troubleshooting (macOS)

Two native dependencies are not pip-installable and only bite on macOS; Linux
wheels bundle both, so CI / a Linux evaluator needs nothing extra.

- **XGBoost needs OpenMP.** If `import xgboost` raises
  `Library not loaded: libomp.dylib`, run `brew install libomp`.
- **PuLP's bundled CBC solver is an x86 binary.** On Apple Silicon install
  Rosetta once: `softwareupdate --install-rosetta --agree-to-license`.

## Project structure

```
logistics-challenge/
├── data/               # synthetic datasets + agent trace (regenerated each run)
├── figures/            # generated plots
├── notebooks/          # one exploratory notebook per module (pre-executed)
├── src/
│   ├── utils.py        # SEED, paths, all synthetic data generation
│   ├── optimization.py # Module 1: LP model, sensitivity, plots
│   ├── ml_models.py    # Module 2: forecast, risk classifier, SHAP
│   └── agent.py        # Module 3: tools registry + ReAct loop
├── tests/              # 10 unit tests (feasibility, duals, leakage, 2A->2B,
│                       #   agent tool-use, determinism)
├── main.py             # single entry point
├── requirements.txt
└── report.pdf          # executive report (Spanish, 3 pages)
```

## Design decisions (summary — full rationale in report.pdf)

- **PuLP over OR-Tools**: declarative formulation reads like the math; the
  CBC solver ships with the package, so the evaluator installs nothing extra.
- **ML stack (scikit-learn + XGBoost + SHAP)**: scikit-learn for its consistent
  API and RandomForest as a robust no-tuning baseline; XGBoost for native class
  imbalance handling (`scale_pos_weight`) and exact TreeSHAP support; SHAP for
  interpretability grounded in Shapley values rather than heuristic importances.
  The two forecast models are picked by empirical comparison (lowest MAPE wins).
- **Sensitivity = scenario sweep + duals**: besides the +20% sweep, the LP's
  capacity **shadow prices** (constraint duals) are reported — by complementary
  slackness only binding warehouses are priced, and the largest |dual| is the
  capacity bottleneck worth expanding first (marginal $ per extra unit).
- **Walk-forward validation** with expanding window: 4 folds × 4 weeks.
  Lag/rolling features are shifted so week *t* only ever sees weeks < *t*
  (covered by a dedicated leakage unit test).
- **2A → 2B integration**: the risk classifier's `projected_demand` feature is
  the Module 2A forecast (trained on weeks < 40, so the projection on the risk
  test weeks is out-of-sample), not a standalone random draw — the same
  projection feeds the agent's live risk scoring.
- **Class imbalance** (~26% positives) handled via `scale_pos_weight`
  rather than oversampling, keeping the temporal test split untouched.
- **Agent without external LLM**: the reasoning policy is rule-based and
  deterministic, so the whole system is reproducible offline with no API
  keys. The architecture mirrors LLM tool-calling — a `Tools` registry with
  typed signatures and a policy isolated in `LogisticsAgent.run()` — so
  swapping in an LLM (LangChain or direct function-calling) is a localized
  change. All four tools are exercised each cycle — for every at-risk point the
  agent forecasts demand, checks the **nearest** warehouse's stock
  (`get_stock_status`) to see if it can cover the projected 4-week need, alerts
  with stock-aware severity, and re-optimizes under +20% demand when any HIGH
  case appears. Every step is logged to `data/agent_trace.json`
  (observe / reason / act / stop) for full traceability.
- **Single seed (42)** in `src/utils.py` governs every random source.

## Key results (seed=42)

All figures below are deterministic (seed=42) — `python main.py` reprints them.

- Optimization: optimal plan at **$14,791** base; **$17,890** at +20% demand
  (still feasible; cost grows ~21% as flow shifts to costlier lanes). The only
  binding warehouse is **W1** (shadow price −0.29): each extra unit of W1
  capacity cuts total cost by ~$0.29, so it is the one to expand first.
- Forecast: RandomForest wins with **MAPE 8.2%** (walk-forward).
- Risk classifier: **ROC-AUC 0.87, F1 0.77**; SHAP shows `current_stock` is the
  dominant driver, followed by `projected_demand` (now the 2A forecast). These
  are lower than an earlier version with a random `projected_demand` (~0.91) —
  removing that circularity costs a few points but makes the metrics honest.
- Agent: detects 9 high-risk points, alerts each with stock-aware severity,
  triggers a stressed re-optimization, and stops when all are attended.
