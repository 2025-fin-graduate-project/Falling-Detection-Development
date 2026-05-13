#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from time import perf_counter

import pandas as pd


CHUNK_SIZE = 100_000
TABLE_NAME = "fall_frames"


def log_progress(message: str) -> None:
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load final_dataset.csv into SQLite with metadata.")
    parser.add_argument(
        "--csv-path",
        default="dataset/final_dataset.csv",
        help="Path to source CSV.",
    )
    parser.add_argument(
        "--sqlite-path",
        default="dataset/final_dataset.sqlite",
        help="Path to output SQLite DB.",
    )
    parser.add_argument("--monitor-start-sec", type=float, default=5.0)
    parser.add_argument("--monitor-end-sec", type=float, default=9.0)
    parser.add_argument("--target-fps", type=int, default=15)
    parser.add_argument("--clip-duration-sec", type=float, default=10.0)
    parser.add_argument(
        "--label-policy",
        default="segment_max_on_5_to_9_sec",
        help="Metadata note for segment-level label policy.",
    )
    return parser.parse_args()


def store_metadata(
    conn: sqlite3.Connection,
    csv_path: Path,
    sqlite_path: Path,
    args: argparse.Namespace,
) -> None:
    log_progress("metadata table upsert start")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_metadata (
            meta_key TEXT PRIMARY KEY,
            meta_value TEXT NOT NULL
        );
        """
    )
    metadata_rows = {
        "source_csv_path": str(csv_path.resolve()),
        "sqlite_path": str(sqlite_path.resolve()),
        "base_table": TABLE_NAME,
        "target_fps": str(args.target_fps),
        "clip_duration_sec": str(args.clip_duration_sec),
        "monitor_start_sec": str(args.monitor_start_sec),
        "monitor_end_sec": str(args.monitor_end_sec),
        "target_monitor_steps": "60",
        "label_policy": args.label_policy,
        "feature_engineering_notes": "kp0~kp16 xyz-score style columns + HSSC_y HSSC_x RWHC VHSSC",
        "created_for": "STM32N6 fall detection TFLite flow",
    }
    conn.executemany(
        "INSERT OR REPLACE INTO dataset_metadata(meta_key, meta_value) VALUES (?, ?)",
        list(metadata_rows.items()),
    )
    log_progress(f"metadata table upsert done rows={len(metadata_rows)}")


def store_column_schema(conn: sqlite3.Connection, csv_path: Path) -> None:
    log_progress("column schema load start")
    header_df = pd.read_csv(csv_path, nrows=0)
    schema_df = pd.DataFrame(
        {
            "column_index": list(range(len(header_df.columns))),
            "column_name": list(header_df.columns),
        }
    )
    schema_df.to_sql("dataset_columns", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dataset_columns_name ON dataset_columns(column_name);")
    log_progress(f"column schema load done columns={len(schema_df)}")


def load_csv_to_sqlite(conn: sqlite3.Connection, csv_path: Path) -> int:
    log_progress(f"csv chunk load start chunk_size={CHUNK_SIZE}")
    total_rows = 0
    for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, chunksize=CHUNK_SIZE)):
        chunk.to_sql(TABLE_NAME, conn, if_exists="append", index=False)
        total_rows += len(chunk)
        log_progress(f"chunk={chunk_idx:03d} chunk_rows={len(chunk)} rows_loaded={total_rows}")
    log_progress(f"csv chunk load done total_rows={total_rows}")
    return total_rows


def create_indexes(conn: sqlite3.Connection) -> None:
    sql_list = [
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_video_id ON {TABLE_NAME}(video_id);",
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_video_time ON {TABLE_NAME}(video_id, time_sec);",
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_label ON {TABLE_NAME}(label);",
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_time ON {TABLE_NAME}(time_sec);",
    ]
    for idx, sql in enumerate(sql_list, start=1):
        log_progress(f"index create start step={idx}/{len(sql_list)} sql={sql}")
        conn.execute(sql)
        log_progress(f"index create done step={idx}/{len(sql_list)}")


def create_monitoring_views(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    log_progress("monitoring view rebuild start")
    conn.execute("DROP VIEW IF EXISTS monitoring_frames;")
    conn.execute(
        f"""
        CREATE VIEW monitoring_frames AS
        SELECT *
        FROM {TABLE_NAME}
        WHERE time_sec >= {args.monitor_start_sec} AND time_sec < {args.monitor_end_sec};
        """
    )
    log_progress("monitoring_frames view ready")

    log_progress("monitoring_segments table rebuild start")
    conn.execute("DROP TABLE IF EXISTS monitoring_segments;")
    conn.execute(
        """
        CREATE TABLE monitoring_segments AS
        SELECT
            video_id,
            MIN(time_sec) AS segment_start_sec,
            MAX(time_sec) AS segment_end_sec,
            COUNT(*) AS segment_frames,
            MAX(label) AS segment_label,
            SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS positive_frame_count,
            AVG(CASE WHEN label = 1 THEN 1.0 ELSE 0.0 END) AS positive_frame_ratio
        FROM monitoring_frames
        GROUP BY video_id;
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_monitoring_segments_video_id ON monitoring_segments(video_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_monitoring_segments_label ON monitoring_segments(segment_label);")
    log_progress("monitoring_segments table rebuild done")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    sqlite_path = Path(args.sqlite_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    if sqlite_path.exists():
        sqlite_path.unlink()

    with sqlite3.connect(sqlite_path) as conn:
        start = perf_counter()
        store_metadata(conn, csv_path, sqlite_path, args)
        store_column_schema(conn, csv_path)
        total_rows = load_csv_to_sqlite(conn, csv_path)
        create_indexes(conn)
        create_monitoring_views(conn, args)
        log_progress("final commit start")
        conn.commit()
        log_progress(f"final commit done elapsed_sec={perf_counter() - start:.2f}")

        log_progress("verification query start total_rows")
        total_db_rows = pd.read_sql_query(f"SELECT COUNT(*) AS total_rows FROM {TABLE_NAME};", conn).iloc[0, 0]
        log_progress("verification query done total_rows")
        log_progress("verification query start unique_videos")
        total_videos = pd.read_sql_query(
            f"SELECT COUNT(DISTINCT video_id) AS unique_videos FROM {TABLE_NAME};", conn
        ).iloc[0, 0]
        log_progress("verification query done unique_videos")
        log_progress("verification query start monitoring_rows")
        monitoring_rows = pd.read_sql_query(
            "SELECT COUNT(*) AS monitoring_rows FROM monitoring_frames;", conn
        ).iloc[0, 0]
        log_progress("verification query done monitoring_rows")
        log_progress("verification query start monitoring_segments")
        monitoring_segments = pd.read_sql_query(
            "SELECT COUNT(*) AS clip_count FROM monitoring_segments;", conn
        ).iloc[0, 0]
        log_progress("verification query done monitoring_segments")

    print(f"CSV rows loaded: {total_rows}")
    print(f"DB rows verified: {total_db_rows}")
    print(f"Unique videos: {total_videos}")
    print(f"Monitoring rows (5~9s): {monitoring_rows}")
    print(f"Monitoring segments: {monitoring_segments}")
    print(f"SQLite written to: {sqlite_path}")


if __name__ == "__main__":
    main()
