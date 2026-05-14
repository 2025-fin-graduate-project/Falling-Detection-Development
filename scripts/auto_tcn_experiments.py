#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


DATASETS = {
    "raw_lb2": {
        "pp": "raw",
        "lb": "2",
        "files": [
            "dataset/splits_v2/train.csv",
            "dataset/splits_v2/val.csv",
            "dataset/splits_v2/test.csv",
        ],
        "args": [
            "--preprocessing", "raw",
            "--train-csv", "dataset/splits_v2/train.csv",
            "--val-csv", "dataset/splits_v2/val.csv",
            "--test-csv", "dataset/splits_v2/test.csv",
            "--label-column", "label",
        ],
    },
    "filtered_lb2": {
        "pp": "filtered",
        "lb": "2",
        "files": [
            "dataset/splits_v2_filtered/train.csv",
            "dataset/splits_v2_filtered/val.csv",
            "dataset/splits_v2_filtered/test.csv",
        ],
        "args": [
            "--preprocessing", "filtered",
            "--train-csv", "dataset/splits_v2_filtered/train.csv",
            "--val-csv", "dataset/splits_v2_filtered/val.csv",
            "--test-csv", "dataset/splits_v2_filtered/test.csv",
            "--label-column", "label",
        ],
    },
    "raw_lb3": {
        "pp": "raw",
        "lb": "3",
        "files": [
            "dataset/lb3_v2/train.csv",
            "dataset/lb3_v2/val.csv",
            "dataset/lb3_v2/test.csv",
        ],
        "args": [
            "--preprocessing", "raw",
            "--train-csv", "dataset/lb3_v2/train.csv",
            "--val-csv", "dataset/lb3_v2/val.csv",
            "--test-csv", "dataset/lb3_v2/test.csv",
            "--label-column", "label_3class",
            "--positive-labels", "1,2",
            "--num-classes", "3",
        ],
    },
    "filtered_lb3": {
        "pp": "filtered",
        "lb": "3",
        "files": [
            "dataset/splits_v2_filtered/train.csv",
            "dataset/splits_v2_filtered/val.csv",
            "dataset/splits_v2_filtered/test.csv",
        ],
        "args": [
            "--preprocessing", "filtered",
            "--train-csv", "dataset/splits_v2_filtered/train.csv",
            "--val-csv", "dataset/splits_v2_filtered/val.csv",
            "--test-csv", "dataset/splits_v2_filtered/test.csv",
            "--label-column", "label_3class",
            "--positive-labels", "1,2",
            "--num-classes", "3",
        ],
    },
}

BASE_ARGS = [
    "--model-type", "tcn",
    "--tcn-channels", "32,32,64,96",
    "--tcn-dilations", "1,1,1,1",
    "--tcn-kernel-size", "3",
    "--feature-set", "kp12",
    "--data-scope", "all",
    "--dropout-rate", "0.3",
    "--noise-std", "0.02",
    "--train-negative-stride", "2",
    "--early-stop-patience", "15",
    "--epochs", "100",
    "--min-val-precision", "0.90",
    "--quiet",
]

PASS_CRITERIA = {
    "test_video_f1": 0.91,
    "test_video_recall": 0.90,
    "test_video_min_precision": 0.90,
    "test_int8_f1": 0.90,
}

MIN_ACCEPTABLE_PRECISION = 0.89


@dataclass
class Experiment:
    experiment_id: str
    dataset_key: str
    purpose: str
    extra_args: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and adapt TCN baseline experiments.")
    parser.add_argument("--output-root", default="results/tcn_auto")
    parser.add_argument("--mode", choices=["smoke", "full", "auto"], default="auto")
    parser.add_argument("--max-experiments", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--poll-seconds", type=int, default=60, help="Reserved for nohup log watchers; runs are synchronous.")
    parser.add_argument("--skip-gpu-check", action="store_true")
    parser.add_argument("--no-int8", action="store_true", help="Skip TFLite export/eval for quick non-deployment checks.")
    return parser.parse_args()


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def cuda_env(project_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    paths = ["/usr/lib/wsl/lib"]
    extra_path_entries: list[str] = []
    for nvidia_root in (project_root / ".venv" / "lib").glob("python*/site-packages/nvidia"):
        paths.extend(str(path) for path in sorted(nvidia_root.glob("*/lib")) if path.is_dir())
        cuda_nvcc = nvidia_root / "cuda_nvcc"
        if cuda_nvcc.is_dir():
            extra_path_entries.append(str(cuda_nvcc / "bin"))
            existing_xla = env.get("XLA_FLAGS", "")
            env["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={cuda_nvcc} {existing_xla}".strip()
    current = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = ":".join(paths + ([current] if current else []))
    if extra_path_entries:
        env["PATH"] = ":".join(extra_path_entries + [env.get("PATH", "")])
    return env


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def missing_files(project_root: Path, dataset_key: str) -> list[str]:
    return [path for path in DATASETS[dataset_key]["files"] if not (project_root / path).is_file()]


def architecture_queue() -> list[Experiment]:
    return [
        Experiment("TCN-SAFE-v01", "raw_lb2", "safe baseline channels 32,32,64,96"),
        Experiment(
            "TCN-SAFE-v02",
            "raw_lb2",
            "compact channels 24,24,48,64 for INT8 stability",
            ["--tcn-channels", "24,24,48,64"],
        ),
        Experiment(
            "TCN-SAFE-v03",
            "raw_lb2",
            "wide channels 32,64,96,128 for capacity ceiling",
            ["--tcn-channels", "32,64,96,128"],
        ),
        Experiment(
            "TCN-SAFE-v04",
            "raw_lb2",
            "deeper non-dilated stack",
            ["--tcn-channels", "32,32,64,64,96", "--tcn-dilations", "1,1,1,1,1"],
        ),
        Experiment(
            "TCN-SAFE-v05",
            "raw_lb2",
            "wider temporal kernel without dilation",
            ["--tcn-kernel-size", "5"],
        ),
    ]


def metric_value(metrics: dict[str, Any], section: str, field_name: str) -> float | None:
    value = metrics.get("metrics", {}).get(section, {}).get(field_name)
    return None if value is None else float(value)


def per_class_precision(section_metrics: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    cm = section_metrics.get("confusion_matrix")
    if not cm or len(cm) < 2 or len(cm[0]) < 2 or len(cm[1]) < 2:
        return None, None, None
    tn, fp = float(cm[0][0]), float(cm[0][1])
    fn, tp = float(cm[1][0]), float(cm[1][1])
    fall_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    nonfall_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return fall_precision, nonfall_precision, min(fall_precision, nonfall_precision)


def load_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def summarize_metrics(experiment: Experiment, metrics: dict[str, Any] | None, status: str, reason: str = "") -> dict[str, Any]:
    dataset = DATASETS[experiment.dataset_key]
    row = {
        "experiment_id": experiment.experiment_id,
        "status": status,
        "reason": reason,
        "purpose": experiment.purpose,
        "extra_args": json.dumps(experiment.extra_args, ensure_ascii=False),
        "dataset_key": experiment.dataset_key,
        "preprocessing": dataset["pp"],
        "label_schema": dataset["lb"],
        "test_video_f1": "",
        "test_video_fall_precision": "",
        "test_video_nonfall_precision": "",
        "test_video_min_precision": "",
        "test_video_recall": "",
        "val_video_f1": "",
        "test_int8_f1": "",
        "test_int8_fall_precision": "",
        "test_int8_nonfall_precision": "",
        "test_int8_min_precision": "",
        "threshold": "",
        "min_consecutive": "",
        "model_int8_size_kb": "",
        "pass": "",
    }
    if metrics is None:
        return row

    tv = metrics.get("metrics", {}).get("test_video", {})
    vv = metrics.get("metrics", {}).get("val_video", {})
    int8 = metrics.get("metrics", {}).get("test_int8", {})
    tv_fall_p, tv_nonfall_p, tv_min_p = per_class_precision(tv)
    int8_fall_p, int8_nonfall_p, int8_min_p = per_class_precision(int8)
    threshold = metrics.get("threshold_selection", {})
    export_paths = metrics.get("export_paths", {})
    row.update(
        {
            "test_video_f1": tv.get("f1", ""),
            "test_video_fall_precision": tv_fall_p if tv_fall_p is not None else "",
            "test_video_nonfall_precision": tv_nonfall_p if tv_nonfall_p is not None else "",
            "test_video_min_precision": tv_min_p if tv_min_p is not None else "",
            "test_video_recall": tv.get("recall", ""),
            "val_video_f1": vv.get("f1", ""),
            "test_int8_f1": int8.get("f1", ""),
            "test_int8_fall_precision": int8_fall_p if int8_fall_p is not None else "",
            "test_int8_nonfall_precision": int8_nonfall_p if int8_nonfall_p is not None else "",
            "test_int8_min_precision": int8_min_p if int8_min_p is not None else "",
            "threshold": threshold.get("threshold", ""),
            "min_consecutive": threshold.get("min_consecutive", ""),
            "model_int8_size_kb": export_paths.get("model_int8_size_kb", ""),
            "pass": passes_criteria(metrics),
        }
    )
    return row


def passes_criteria(metrics: dict[str, Any]) -> bool:
    checks = {
        "test_video_f1": metric_value(metrics, "test_video", "f1"),
        "test_video_recall": metric_value(metrics, "test_video", "recall"),
        "test_int8_f1": metric_value(metrics, "test_int8", "f1"),
    }
    _, _, video_min_precision = per_class_precision(metrics.get("metrics", {}).get("test_video", {}))
    checks["test_video_min_precision"] = video_min_precision
    return all(value is not None and value >= PASS_CRITERIA[key] for key, value in checks.items())


def choose_best_completed(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    completed = [row for row in rows if row["status"] == "completed" and row["test_video_f1"] != ""]
    if not completed:
        return None
    return max(
        completed,
        key=lambda row: (
            float(row["test_video_min_precision"] or 0.0),
            float(row["test_video_f1"]),
            float(row["test_int8_f1"] or 0.0),
        ),
    )


def _row_extra_args(row: dict[str, Any]) -> list[str]:
    raw = row.get("extra_args") or "[]"
    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def _has_prefix(rows: list[dict[str, Any]], prefix: str) -> bool:
    return any(str(row["experiment_id"]).startswith(prefix) for row in rows)


def variable_candidates(rows: list[dict[str, Any]]) -> list[Experiment]:
    best = choose_best_completed(rows)
    if best is None:
        return []
    best_args = _row_extra_args(best)
    best_id = str(best["experiment_id"])
    return [
        Experiment("TCN-VAR-v01", "filtered_lb2", f"filtered LB-2 using architecture from {best_id}", best_args),
        Experiment("TCN-VAR-v02", "raw_lb3", f"raw LB-3 using architecture from {best_id}", best_args),
        Experiment("TCN-VAR-v03", "filtered_lb3", f"filtered LB-3 using architecture from {best_id}", best_args),
    ]


def adaptive_candidates(rows: list[dict[str, Any]]) -> list[Experiment]:
    if not _has_prefix(rows, "TCN-VAR-"):
        return variable_candidates(rows)
    if _has_prefix(rows, "TCN-TUNE-"):
        return []

    best = choose_best_completed(rows)
    if best is None:
        return []

    dataset_key = str(best["dataset_key"])
    best_id = str(best["experiment_id"])
    best_args = _row_extra_args(best)
    candidates: list[Experiment] = []

    min_precision = float(best["test_video_min_precision"] or 0.0)
    recall = float(best["test_video_recall"] or 0.0)
    int8_f1 = float(best["test_int8_f1"] or 0.0)

    if min_precision < PASS_CRITERIA["test_video_min_precision"]:
        min_consecutive_values = "3,5,7" if min_precision >= MIN_ACCEPTABLE_PRECISION else "5,7,9"
        min_val_precision = "0.92" if min_precision >= MIN_ACCEPTABLE_PRECISION else "0.93"
        candidates.append(
            Experiment(
                "TCN-AUTO-v05",
                dataset_key,
                f"precision hardening based on {best_id}",
                [*best_args, "--min-val-precision", min_val_precision, "--min-consecutive-values", min_consecutive_values],
            )
        )
    elif recall < PASS_CRITERIA["test_video_recall"]:
        candidates.append(
            Experiment(
                "TCN-AUTO-v05",
                dataset_key,
                f"recall/consecutive sweep based on {best_id}",
                [*best_args, "--min-consecutive-values", "1,2,3,5"],
            )
        )
    elif int8_f1 < PASS_CRITERIA["test_int8_f1"]:
        candidates.append(
            Experiment(
                "TCN-AUTO-v05",
                dataset_key,
                f"INT8 calibration expansion based on {best_id}",
                [*best_args, "--representative-samples", "512", "--quant-eval-max-windows", "0"],
            )
        )
        candidates.append(
            Experiment(
                "TCN-AUTO-v06",
                dataset_key,
                f"INT8-friendly smaller TCN based on {best_id}",
                ["--tcn-channels", "24,24,48,64", "--representative-samples", "512"],
            )
        )
    for index, candidate in enumerate(candidates, start=1):
        candidate.experiment_id = f"TCN-TUNE-v{index:02d}"
    return candidates


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment_id",
        "status",
        "reason",
        "purpose",
        "extra_args",
        "dataset_key",
        "preprocessing",
        "label_schema",
        "test_video_f1",
        "test_video_fall_precision",
        "test_video_nonfall_precision",
        "test_video_min_precision",
        "test_video_recall",
        "val_video_f1",
        "test_int8_f1",
        "test_int8_fall_precision",
        "test_int8_nonfall_precision",
        "test_int8_min_precision",
        "threshold",
        "min_consecutive",
        "model_int8_size_kb",
        "pass",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_command(command: list[str], project_root: Path, log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log("running " + " ".join(command))
    with log_path.open("w", encoding="utf-8") as fh:
        process = subprocess.Popen(
            command,
            cwd=project_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            fh.write(line)
        return process.wait()


def run_gpu_check(project_root: Path, output_root: Path, env: dict[str, str]) -> bool:
    code = run_command(["bash", "scripts/check_tensorflow_gpu.sh"], project_root, output_root / "gpu_check.log", env)
    return code == 0


def run_experiment(
    experiment: Experiment,
    project_root: Path,
    output_root: Path,
    mode: str,
    env: dict[str, str],
    no_int8: bool,
) -> int:
    command = [
        "uv",
        "run",
        "python",
        "scripts/train_baseline.py",
        "--experiment-id",
        experiment.experiment_id,
        "--output-root",
        str(output_root),
        *BASE_ARGS,
        *DATASETS[experiment.dataset_key]["args"],
        *experiment.extra_args,
    ]
    if mode == "smoke":
        command.extend(["--smoke", "--max-rows", "200000"])
    if no_int8:
        command.extend(["--no-export-tflite"])
    return run_command(command, project_root, output_root / f"{experiment.experiment_id}.log", env)


def main() -> int:
    args = parse_args()
    project_root = Path.cwd()
    output_root = (project_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    env = cuda_env(project_root)
    state_path = output_root / "state.json"
    summary_path = output_root / "summary.csv"

    rows: list[dict[str, Any]] = []
    state: dict[str, Any] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "output_root": str(output_root),
        "pass_criteria": PASS_CRITERIA,
        "experiments": [],
    }

    if not args.skip_gpu_check:
        state["gpu_check_ok"] = run_gpu_check(project_root, output_root, env)
        if not state["gpu_check_ok"]:
            write_json(state_path, state)
            return 2

    queue = architecture_queue()
    if args.mode == "smoke":
        queue = queue[:1]

    executed = 0
    seen: set[str] = set()
    idx = 0
    while idx < len(queue):
        experiment = queue[idx]
        idx += 1
        if experiment.experiment_id in seen:
            continue
        seen.add(experiment.experiment_id)

        missing = missing_files(project_root, experiment.dataset_key)
        metrics_path = output_root / experiment.experiment_id / "metrics.json"
        metrics = load_metrics(metrics_path)

        if missing:
            row = summarize_metrics(experiment, None, "pending_data", ";".join(missing))
            rows.append(row)
            state["experiments"].append({**asdict(experiment), "status": "pending_data", "missing": missing})
            log(f"pending {experiment.experiment_id}: missing {', '.join(missing)}")
        elif metrics is not None:
            row = summarize_metrics(experiment, metrics, "completed", "metrics already present")
            rows.append(row)
            state["experiments"].append({**asdict(experiment), "status": "completed", "reused": True})
            log(f"skip {experiment.experiment_id}: metrics already present")
        else:
            if args.max_experiments and executed >= args.max_experiments:
                row = summarize_metrics(experiment, None, "queued", "max_experiments reached")
                rows.append(row)
                state["experiments"].append({**asdict(experiment), "status": "queued"})
                log(f"queued {experiment.experiment_id}: max_experiments reached")
            else:
                code = run_experiment(experiment, project_root, output_root, args.mode, env, args.no_int8)
                executed += 1
                metrics = load_metrics(metrics_path)
                status = "completed" if code == 0 and metrics is not None else "failed"
                reason = "" if status == "completed" else f"exit_code={code}"
                row = summarize_metrics(experiment, metrics, status, reason)
                rows.append(row)
                state["experiments"].append({**asdict(experiment), "status": status, "exit_code": code})
                if status == "failed":
                    log(f"failed {experiment.experiment_id}: {reason}")

        write_summary(summary_path, rows)
        write_json(state_path, state)

        if args.mode == "auto" and idx == len(queue):
            additions = [item for item in adaptive_candidates(rows) if item.experiment_id not in seen]
            if additions:
                queue.extend(additions)

    state["finished_at"] = datetime.now().isoformat(timespec="seconds")
    state["executed_count"] = executed
    state["summary_path"] = str(summary_path)
    write_summary(summary_path, rows)
    write_json(state_path, state)
    log(f"summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
