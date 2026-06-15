#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import subprocess
import time


ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON = pathlib.Path("/vla/users/niejunnan/codebase/openpi-modified/.venv/bin/python")


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def status_counts(status_dir: pathlib.Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in status_dir.glob("*.json"):
        try:
            status = json.loads(path.read_text(encoding="utf-8")).get("status", "?")
        except Exception:
            status = "bad"
        counts[status] = counts.get(status, 0) + 1
    return counts


def episode_rows(shard_dir: pathlib.Path) -> tuple[int, int]:
    files = 0
    rows = 0
    for path in shard_dir.glob("*.csv"):
        files += 1
        try:
            with path.open(encoding="utf-8") as f:
                rows += max(0, sum(1 for _ in f) - 1)
        except Exception:
            pass
    return files, rows


def merge(run: str, partial: bool) -> int:
    suffix = "_partial" if partial else ""
    cmd = [
        str(PYTHON),
        "igc_gensg/scripts/merge_stage2_gensg_results.py",
        "--manifest",
        f"igc_gensg/configs/{run}_manifest.csv",
        "--shard-dir",
        f"igc_gensg/results/{run}_shards",
        "--job-log-dir",
        f"igc_gensg/logs/{run}_jobs",
        "--episode-out",
        f"igc_gensg/results/{run}_episode_results{suffix}.csv",
        "--query-out",
        f"igc_gensg/results/{run}_query_logs{suffix}.jsonl",
        "--report",
        f"igc_gensg/reports/{run}_report{suffix}.md",
    ]
    completed = subprocess.run(cmd, cwd=str(ROOT))
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor a Stage2 full aligned GenSG run and merge reports.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--expected-jobs", type=int, required=True)
    parser.add_argument("--poll-sec", type=int, default=600)
    parser.add_argument("--partial-merge-every", type=int, default=6, help="Merge partial report every N polls; 0 disables.")
    args = parser.parse_args()

    run = args.run
    status_dir = ROOT / f"igc_gensg/logs/{run}_jobs/status"
    shard_dir = ROOT / f"igc_gensg/results/{run}_shards"
    poll_idx = 0
    print(f"[{now()}] monitor start run={run} expected_jobs={args.expected_jobs}", flush=True)
    while True:
        poll_idx += 1
        counts = status_counts(status_dir)
        files, rows = episode_rows(shard_dir)
        total = sum(counts.values())
        print(f"[{now()}] poll={poll_idx} statuses={counts} total_status={total}/{args.expected_jobs} shard_files={files} episode_rows={rows}", flush=True)

        if total >= args.expected_jobs:
            print(f"[{now()}] all status files present; running final merge", flush=True)
            rc = merge(run, partial=False)
            print(f"[{now()}] final merge rc={rc}", flush=True)
            return rc

        if args.partial_merge_every > 0 and poll_idx % args.partial_merge_every == 0:
            print(f"[{now()}] running partial merge", flush=True)
            rc = merge(run, partial=True)
            print(f"[{now()}] partial merge rc={rc}", flush=True)

        time.sleep(args.poll_sec)


if __name__ == "__main__":
    raise SystemExit(main())
