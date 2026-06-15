from __future__ import annotations

import argparse
import csv
import pathlib
import socket
import subprocess
import sys
import time
from collections import defaultdict


ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON = pathlib.Path("/vla/users/niejunnan/codebase/openpi-modified/.venv/bin/python")


def load_rows(path: pathlib.Path, host_filter: str) -> list[dict[str, str]]:
    aliases = {host_filter, socket.gethostname()}
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("hostname")) in aliases:
                rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch Stage2 GenSG jobs assigned to this host.")
    parser.add_argument("--manifest", default="igc_gensg/configs/stage2_gensg_manifest.csv")
    parser.add_argument("--host-ip", required=True)
    parser.add_argument("--base-port", type=int, default=9000)
    parser.add_argument("--sleep-between-jobs", type=float, default=3.0)
    parser.add_argument("--kill-lerobot", action="store_true")
    parser.add_argument("--no-save-video", action="store_true")
    parser.add_argument("--save-video-mode", choices=["all", "none", "selective"], default="selective")
    parser.add_argument("--attention-query-policy", choices=["all", "none", "early"], default="early")
    parser.add_argument("--attention-early-queries", type=int, default=2)
    parser.add_argument("--max-jobs-per-gpu", type=int, default=0)
    parser.add_argument("--artifact-prefix", default="stage2_gensg")
    parser.add_argument("--job-script", default="igc_gensg/scripts/run_stage2_gensg_job.py")
    args = parser.parse_args()

    manifest = ROOT / args.manifest
    rows = load_rows(manifest, args.host_ip)
    by_gpu: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_gpu[str(row["gpu_id"])].append(row)
    for gpu_rows in by_gpu.values():
        gpu_rows.sort(key=lambda r: r["job_id"])

    artifact_prefix = str(args.artifact_prefix).strip() or "stage2_gensg"
    log_dir = ROOT / f"igc_gensg/logs/{artifact_prefix}_launchers"
    log_dir.mkdir(parents=True, exist_ok=True)
    procs: list[subprocess.Popen] = []
    for gpu, gpu_rows in sorted(by_gpu.items(), key=lambda x: int(x[0])):
        if args.max_jobs_per_gpu > 0:
            gpu_rows = gpu_rows[: args.max_jobs_per_gpu]
        cmd = [
            str(PYTHON),
            "-u",
            "-c",
            (
                "import subprocess, time, sys; "
                f"jobs={ [r['job_id'] for r in gpu_rows]!r}; "
                f"py={str(PYTHON)!r}; manifest={args.manifest!r}; base={args.base_port!r}; job_script={args.job_script!r}; artifact={artifact_prefix!r}; "
                f"kill={bool(args.kill_lerobot)!r}; save={not bool(args.no_save_video)!r}; "
                f"mode={args.save_video_mode!r}; aqp={args.attention_query_policy!r}; ae={args.attention_early_queries!r}; sleep_s={args.sleep_between_jobs!r}; "
                "root='/vla/users/niejunnan/knows/openpi'; "
                "rc=0; "
                "\nfor job in jobs:\n"
                "    cmd=[py,job_script,'--manifest',manifest,'--job-id',job,'--base-port',str(base),'--save-video-mode',mode,'--attention-query-policy',aqp,'--attention-early-queries',str(ae),'--artifact-prefix',artifact]\n"
                "    if kill: cmd.append('--kill-lerobot')\n"
                "    if not save: cmd.append('--no-save-video')\n"
                "    print('LAUNCH_JOB', job, ' '.join(cmd), flush=True)\n"
                "    p=subprocess.run(cmd, cwd=root)\n"
                "    print('DONE_JOB', job, p.returncode, flush=True)\n"
                "    rc=max(rc, int(p.returncode != 0))\n"
                "    time.sleep(float(sleep_s))\n"
                "sys.exit(rc)\n"
            ),
        ]
        log_path = log_dir / f"{args.host_ip.replace('.', '_')}_gpu{gpu}.log"
        with log_path.open("w", encoding="utf-8") as f:
            proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
        print(f"started gpu={gpu} jobs={len(gpu_rows)} pid={proc.pid} log={log_path}")
        procs.append(proc)
    if not procs:
        print(f"no jobs for host {args.host_ip} in {manifest}", file=sys.stderr)
        return 1
    failed = 0
    while procs:
        live = []
        for proc in procs:
            if proc.poll() is None:
                live.append(proc)
            elif proc.returncode != 0:
                failed += 1
        procs = live
        if procs:
            time.sleep(10)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
