#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ModuleNotFoundError as exc:
    plt = pd = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


def ensure_deps() -> None:
    if IMPORT_ERROR is not None:
        raise SystemExit(f"Missing Python dependency: {IMPORT_ERROR.name}") from IMPORT_ERROR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect baseline experiment metrics into common tables and plots.")
    parser.add_argument("--results-root", default="results/baselines_phase0")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def load_rows(results_root: Path) -> list[dict[str, object]]:
    rows = []
    for metrics_path in sorted(results_root.glob("*/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        test_float = payload["metrics"].get("test_float", {})
        val_float = payload["metrics"].get("val_float", {})
        test_int8 = payload["metrics"].get("test_int8", {})
        export_paths = payload.get("export_paths", {})
        rows.append(
            {
                "experiment_id": payload["experiment_id"],
                "model_type": payload["model_type"],
                "preprocessing": payload["preprocessing"],
                "feature_set": payload["feature_set"],
                "val_f1": val_float.get("f1"),
                "val_recall": val_float.get("recall"),
                "test_accuracy": test_float.get("accuracy"),
                "test_precision": test_float.get("precision"),
                "test_recall": test_float.get("recall"),
                "test_f1": test_float.get("f1"),
                "test_auc_roc": test_float.get("auc_roc"),
                "test_pr_auc": test_float.get("pr_auc"),
                "int8_test_f1": test_int8.get("f1"),
                "int8_delta_f1": None
                if not test_int8
                else (test_float.get("f1") - test_int8.get("f1")),
                "int8_size_kb": export_paths.get("model_int8_size_kb"),
                "int8_export_error": export_paths.get("int8_export_error"),
                "metrics_path": str(metrics_path),
            }
        )
    return rows


def save_plot(df: pd.DataFrame, output_dir: Path, columns: list[str], filename: str, title: str) -> None:
    available = [col for col in columns if col in df.columns and df[col].notna().any()]
    if not available:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    df.set_index("experiment_id")[available].plot(kind="bar", ax=ax)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=160)
    plt.close(fig)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def main() -> None:
    ensure_deps()
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir) if args.output_dir else results_root / "common"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(results_root)
    if not rows:
        raise SystemExit(f"No metrics.json files found below {results_root}")
    df = pd.DataFrame(rows).sort_values(["test_f1", "test_recall"], ascending=False)
    df.to_csv(output_dir / "phase0_summary.csv", index=False)
    (output_dir / "phase0_summary.json").write_text(df.to_json(orient="records", indent=2, force_ascii=False))
    save_plot(df, output_dir, ["test_f1"], "compare_f1_bar.png", "Phase 0 F1")
    save_plot(df, output_dir, ["test_precision", "test_recall"], "compare_precision_recall.png", "Precision / Recall")
    save_plot(df, output_dir, ["test_auc_roc", "test_pr_auc"], "compare_auc.png", "AUC")
    save_plot(df, output_dir, ["test_f1", "int8_test_f1"], "compare_quantization_delta.png", "Float vs INT8 F1")
    if "int8_size_kb" in df and df["int8_size_kb"].notna().any():
        fig, ax = plt.subplots(figsize=(9, 4))
        df.set_index("experiment_id")["int8_size_kb"].plot(kind="bar", ax=ax)
        ax.set_ylabel("KB")
        ax.set_title("INT8 model size")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "compare_model_size.png", dpi=160)
        plt.close(fig)
    report = ["# Phase 0 Baseline Report", "", dataframe_to_markdown(df)]
    (output_dir / "phase0_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"wrote {output_dir / 'phase0_summary.csv'}")


if __name__ == "__main__":
    main()
