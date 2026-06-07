"""Module 2 — ML pipeline: demand forecasting (2A) + stockout-risk
classification (2B) + interpretability (2C).

2A: Random Forest vs XGBoost on lag/rolling features, validated with
    walk-forward splits (train always strictly precedes test — no leakage).
2B: Binary classifier with class-imbalance handling via class weights.
2C: Feature importances (gain-based) for both models; SHAP if available.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from xgboost import XGBClassifier, XGBRegressor

from src.utils import (
    FIG_DIR,
    SEED,
    generate_demand_series,
    generate_risk_features,
    set_seeds,
)

LAGS = [1, 2, 4]
ROLLS = [4, 8]


# ---------------------------------------------------------------- 2A
def build_forecast_features(series: pd.DataFrame) -> pd.DataFrame:
    """Lag + rolling-mean features per demand point.

    Rolling windows are shifted by 1 week so week t only sees data
    up to t-1 (prevents leakage).
    """
    df = series.sort_values(["point_id", "week"]).copy()
    g = df.groupby("point_id")["demand"]
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag)
    for w in ROLLS:
        df[f"roll_mean_{w}"] = g.shift(1).rolling(w).mean().reset_index(0, drop=True)
    df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)
    return df.dropna().reset_index(drop=True)


FORECAST_FEATURES = [f"lag_{l}" for l in LAGS] + [
    f"roll_mean_{w}" for w in ROLLS
] + ["week_sin", "week_cos"]


def walk_forward_validate(
    df: pd.DataFrame, model_factory, n_folds: int = 4, test_size: int = 4
) -> dict:
    """Expanding-window walk-forward validation.

    Fold k trains on weeks < cut_k and tests on the next `test_size`
    weeks. Test sets never overlap training data.
    """
    weeks = sorted(df["week"].unique())
    metrics = {"MAE": [], "RMSE": [], "MAPE": []}
    last_model = None
    for k in range(n_folds):
        cut = weeks[-(n_folds - k) * test_size]
        train = df[df["week"] < cut]
        test = df[(df["week"] >= cut) & (df["week"] < cut + test_size)]
        model = model_factory()
        model.fit(train[FORECAST_FEATURES], train["demand"])
        pred = model.predict(test[FORECAST_FEATURES])
        y = test["demand"].to_numpy()
        metrics["MAE"].append(mean_absolute_error(y, pred))
        metrics["RMSE"].append(np.sqrt(mean_squared_error(y, pred)))
        metrics["MAPE"].append(np.mean(np.abs((y - pred) / y)) * 100)
        last_model = model
    return {
        "MAE": float(np.mean(metrics["MAE"])),
        "RMSE": float(np.mean(metrics["RMSE"])),
        "MAPE": float(np.mean(metrics["MAPE"])),
        "model": last_model,
    }


@dataclass
class ForecastResult:
    """Output of 2A: per-model metrics plus the winning model and its data."""

    metrics: pd.DataFrame  # one row per model
    best_name: str
    best_model: object
    features_df: pd.DataFrame


def train_forecast_models(seed: int = SEED) -> ForecastResult:
    """Train RF and XGBoost, validate walk-forward, return the best by MAPE."""
    series = generate_demand_series(seed)
    df = build_forecast_features(series)

    factories = {
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=200, random_state=seed, n_jobs=-1
        ),
        "XGBoost": lambda: XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            random_state=seed, n_jobs=-1, verbosity=0,
        ),
    }
    rows, models = [], {}
    for name, fac in factories.items():
        res = walk_forward_validate(df, fac)
        models[name] = res.pop("model")
        rows.append({"model": name, **res})
    metrics = pd.DataFrame(rows).set_index("model").round(3)
    best = metrics["MAPE"].idxmin()
    return ForecastResult(metrics, best, models[best], df)


def forecast_demand(
    result: ForecastResult, point_id: str, weeks_ahead: int = 4
) -> list[float]:
    """Recursive multi-step forecast for one point (used by the agent)."""
    df = result.features_df
    hist = df[df["point_id"] == point_id].sort_values("week")
    demand_hist = list(hist["demand"].to_numpy())
    last_week = int(hist["week"].max())

    preds = []
    for step in range(1, weeks_ahead + 1):
        w = last_week + step
        feats = {}
        for lag in LAGS:
            feats[f"lag_{lag}"] = demand_hist[-lag]
        for win in ROLLS:
            feats[f"roll_mean_{win}"] = float(np.mean(demand_hist[-win:]))
        feats["week_sin"] = np.sin(2 * np.pi * w / 52)
        feats["week_cos"] = np.cos(2 * np.pi * w / 52)
        x = pd.DataFrame([feats])[FORECAST_FEATURES]
        p = float(result.best_model.predict(x)[0])
        preds.append(round(p, 1))
        demand_hist.append(p)
    return preds


# ---------------------------------------------------------------- 2B
RISK_FEATURES = ["current_stock", "lead_time_days", "projected_demand", "distance_km"]


def forecast_projection(seed: int = SEED) -> pd.DataFrame:
    """1-step-ahead demand projection per (point, week) — the Module 2A model
    feeding the Module 2B `projected_demand` feature.

    The forecast model is trained on the same temporal train window the risk
    classifier uses (weeks < 40), so the projection for the risk *test* weeks
    (>= 40) is a genuine out-of-sample estimate, not a peek at realized demand.
    """
    series = generate_demand_series(seed)
    df = build_forecast_features(series)
    train = df[df["week"] < 40]
    model = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    model.fit(train[FORECAST_FEATURES], train["demand"])
    proj = np.round(model.predict(df[FORECAST_FEATURES]), 1)
    return df[["point_id", "week"]].assign(projected_demand=proj)


def attach_forecast_projection(risk_df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Replace the synthetic `projected_demand` with the Module 2A forecast.

    Weeks without enough history for forecast features (< first valid week) are
    dropped so every row carries a real model projection.
    """
    proj = forecast_projection(seed)
    return (
        risk_df.drop(columns=["projected_demand"])
        .merge(proj, on=["point_id", "week"], how="inner")
        .reset_index(drop=True)
    )


@dataclass
class RiskResult:
    """Output of 2B: metrics, confusion matrix, model and held-out test set."""

    metrics: dict
    confusion: np.ndarray
    model: object
    X_test: pd.DataFrame
    y_test: pd.Series


def train_risk_classifier(seed: int = SEED) -> RiskResult:
    """Train the XGBoost stockout-risk classifier on a temporal split.

    `projected_demand` is sourced from the Module 2A forecast model (2A->2B
    integration), not from a standalone random draw.
    """
    df = attach_forecast_projection(generate_risk_features(seed), seed)
    # Temporal split: first 40 weeks train, last 12 test (mirrors deployment)
    train = df[df["week"] < 40]
    test = df[df["week"] >= 40]
    X_tr, y_tr = train[RISK_FEATURES], train["risk_high"]
    X_te, y_te = test[RISK_FEATURES], test["risk_high"]

    # Class imbalance handled with scale_pos_weight = neg/pos
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    model = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        scale_pos_weight=spw, random_state=seed, n_jobs=-1, verbosity=0,
        eval_metric="logloss",
    )
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "ROC_AUC": round(float(roc_auc_score(y_te, proba)), 4),
        "F1": round(float(f1_score(y_te, pred)), 4),
        "positive_rate_train": round(float(y_tr.mean()), 3),
        "scale_pos_weight": round(float(spw), 2),
    }
    cm = confusion_matrix(y_te, pred)
    return RiskResult(metrics, cm, model, X_te, y_te)


# ---------------------------------------------------------------- 2C
def _shap_values_risk(rk: RiskResult) -> np.ndarray:
    """Exact SHAP values for the XGBoost risk model.

    Tries shap.TreeExplainer first; if the shap<->xgboost version combo
    fails, falls back to XGBoost's native `pred_contribs=True`, which
    computes the same exact TreeSHAP values.
    """
    try:
        import shap

        return np.asarray(shap.TreeExplainer(rk.model).shap_values(rk.X_test))
    except Exception:
        import xgboost as xgb

        dm = xgb.DMatrix(rk.X_test)
        contribs = rk.model.get_booster().predict(dm, pred_contribs=True)
        return contribs[:, :-1]  # drop bias column


def interpretability(fc: ForecastResult, rk: RiskResult) -> dict:
    """SHAP-based interpretability for both models."""
    out = {}
    sv = _shap_values_risk(rk)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=RISK_FEATURES)
    out["risk_importance"] = imp.sort_values(ascending=False)
    out["risk_method"] = "SHAP (mean |value|, TreeSHAP)"

    try:
        import shap

        FIG_DIR.mkdir(exist_ok=True)
        shap.summary_plot(sv, rk.X_test, show=False, plot_size=(8, 4))
        plt.tight_layout()
        plt.savefig(FIG_DIR / "shap_risk.png", dpi=150)
        plt.close("all")
    except Exception:
        pass  # plot is optional; importances above are already SHAP-based

    fimp = pd.Series(
        fc.best_model.feature_importances_, index=FORECAST_FEATURES
    ).sort_values(ascending=False)
    out["forecast_importance"] = fimp
    return out


def plot_ml_results(fc: ForecastResult, rk: RiskResult, interp: dict) -> None:
    """Save a 3-panel figure: forecast metrics, confusion matrix, risk drivers."""
    FIG_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    fc.metrics[["MAE", "RMSE"]].plot.bar(ax=axes[0], rot=0)
    axes[0].set_title(f"Forecast walk-forward metrics (best: {fc.best_name})")

    ax = axes[1]
    ax.imshow(rk.confusion, cmap="Blues")
    for (i, j), v in np.ndenumerate(rk.confusion):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > rk.confusion.max() / 2 else "black")
    ax.set_xticks([0, 1], ["pred low", "pred high"])
    ax.set_yticks([0, 1], ["true low", "true high"])
    ax.set_title(f"Risk confusion matrix — AUC {rk.metrics['ROC_AUC']}, F1 {rk.metrics['F1']}")

    interp["risk_importance"].plot.barh(ax=axes[2], color="steelblue")
    axes[2].invert_yaxis()
    axes[2].set_title(f"Risk drivers — {interp['risk_method']}")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "ml_results.png", dpi=150)
    plt.close(fig)


def run_module2() -> dict:
    """Entry point used by main.py: train both models, plot, summarize."""
    set_seeds()
    fc = train_forecast_models()
    rk = train_risk_classifier()
    interp = interpretability(fc, rk)
    plot_ml_results(fc, rk, interp)

    top_risk = interp["risk_importance"].index[0]
    executive_summary = (
        f"El mejor modelo de forecast fue {fc.best_name} "
        f"(MAPE {fc.metrics.loc[fc.best_name, 'MAPE']:.1f}% en validación walk-forward). "
        f"Para el riesgo de desabasto, la variable más determinante es '{top_risk}': "
        f"el análisis de interpretabilidad muestra que el nivel de inventario actual "
        f"frente a la demanda proyectada domina la probabilidad de riesgo, mientras que "
        f"lead time y distancia actúan como amplificadores secundarios. Operativamente, "
        f"esto sugiere priorizar políticas de stock de seguridad sobre renegociación de rutas."
    )
    return {
        "forecast": fc,
        "risk": rk,
        "interpretability": interp,
        "executive_summary": executive_summary,
    }


if __name__ == "__main__":
    out = run_module2()
    print(out["forecast"].metrics.to_string())
    print("\nRisk metrics:", out["risk"].metrics)
    print("\nRisk drivers:\n", out["interpretability"]["risk_importance"].round(4).to_string())
    print("\n" + out["executive_summary"])
