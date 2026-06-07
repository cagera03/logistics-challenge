"""Single entry point — runs the full integrated pipeline.

Usage:
    pip install -r requirements.txt
    python main.py
"""
from src.agent import run_module3
from src.ml_models import run_module2
from src.optimization import run_module1
from src.utils import save_all_data, set_seeds


def main() -> None:
    """Run the full pipeline: data -> optimization -> ML -> agent."""
    set_seeds()

    print("=" * 70)
    print("STEP 0 · Generating synthetic data (seed=42)")
    save_all_data()
    print("  -> data/ written")

    print("=" * 70)
    print("STEP 1 · Optimization: capacitated transportation problem")
    m1 = run_module1()
    print(f"  Base cost:        ${m1['base'].total_cost:,.2f} ({m1['base'].status})")
    print(f"  +20% demand cost: ${m1['stressed'].total_cost:,.2f} ({m1['stressed'].status})")
    print("  Sensitivity:")
    print(m1["sensitivity"].to_string(index=False))
    sp = m1["base"].utilization["shadow_price"]
    bottleneck = sp.abs().idxmax()
    print(f"  Capacity shadow prices ($/unit): {sp.round(3).to_dict()}")
    print(
        f"  -> expand {bottleneck} first: its capacity binds, so one extra unit "
        f"moves total cost by ~${abs(sp[bottleneck]):.3f} (largest |dual|)"
    )
    print("  -> figures/optimization_*.png, figures/sensitivity_cost.png")

    print("=" * 70)
    print("STEP 2 · Machine Learning: forecast + risk classification")
    m2 = run_module2()
    print(m2["forecast"].metrics.to_string())
    print(f"  Risk classifier: {m2['risk'].metrics}")
    print(f"  Executive summary: {m2['executive_summary']}")
    print("  -> figures/ml_results.png, figures/shap_risk.png")

    print("=" * 70)
    print("STEP 3 · Agent: monitoring, alerts and recommendations")
    m3 = run_module3()
    print(f"  At-risk points: {m3['at_risk']}")
    print(f"  Alerts sent:    {len(m3['alerts'])}")
    print(f"  Stop reason:    {m3['stop_reason']}")
    print(f"  Trace:          {m3['trace_path']}")

    print("=" * 70)
    print("DONE. See figures/ for plots and data/agent_trace.json for the agent log.")


if __name__ == "__main__":
    main()
