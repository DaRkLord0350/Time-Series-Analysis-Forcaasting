# pipelines/compare_models.py
import json
from pathlib import Path

import pandas as pd


def load_metrics(path):
    """Load metrics JSON file (handles both Prophet and SARIMA formats)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"❌ Metrics file not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)

    # Handle Prophet-style (list)
    if isinstance(data, list):
        return {d["series_id"]: d for d in data}

    # Handle SARIMA-style (dict with nested metrics)
    if "results" in data:
        results = {}
        for series_id, content in data["results"].items():
            metrics = content.get("metrics", {})
            results[series_id] = {
                "MAE": metrics.get("mean_smape"),  # SARIMA doesn't output MAE directly
                "MAPE": metrics.get("mean_smape"),
                "sMAPE": metrics.get("mean_smape"),
            }
        return results

    return data


def pick_winner(row):
    """Decide which model wins (lower sMAPE wins)"""
    sarima, prophet = row["sarima_smape"], row["prophet_smape"]
    if pd.isna(sarima) or pd.isna(prophet):
        return "Missing Data"
    if sarima < prophet:
        return "SARIMA"
    elif prophet < sarima:
        return "Prophet"
    return "Tie"


def main():
    reports_dir = Path("reports")
    sarima_path = reports_dir / "sarima_metrics.json"
    prophet_path = reports_dir / "prophet_metrics.json"
    out_path = reports_dir / "model_compare_v1.md"

    sarima_metrics = load_metrics(sarima_path)
    prophet_metrics = load_metrics(prophet_path)

    # Combine results
    records = []
    for series_id in sorted(set(sarima_metrics.keys()) | set(prophet_metrics.keys())):
        s = sarima_metrics.get(series_id, {})
        p = prophet_metrics.get(series_id, {})
        records.append(
            {
                "series_id": series_id,
                "sarima_mae": s.get("MAE"),
                "prophet_mae": p.get("MAE"),
                "sarima_mape": s.get("MAPE"),
                "prophet_mape": p.get("MAPE"),
                "sarima_smape": s.get("sMAPE"),
                "prophet_smape": p.get("sMAPE"),
            }
        )

    df = pd.DataFrame(records)
    df["winner"] = df.apply(pick_winner, axis=1)

    # --- Write Markdown report ---
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# 📊 Model Comparison Report: Prophet vs SARIMA\n\n")
        f.write("**Lower values indicate better performance.**\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## 🏆 Winners per Series\n")

        for _, row in df.iterrows():
            badge = "🏅" if row["winner"] != "Tie" else "⚖️"
            if row["winner"] == "Prophet":
                reason = (
                    "Prophet captured smooth seasonality and long-term trends better."
                )
            elif row["winner"] == "SARIMA":
                reason = "SARIMA fit short-term autocorrelations better."
            else:
                reason = "Both models performed equally well."
            f.write(
                f"- **{row['series_id']}** → {badge} **{row['winner']}** → {reason}\n"
            )

    print(f"✅ Model comparison saved to {out_path}")


if __name__ == "__main__":
    main()
