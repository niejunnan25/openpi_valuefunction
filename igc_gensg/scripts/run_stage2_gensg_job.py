#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
OPENPI_PYTHON = pathlib.Path('/vla/users/niejunnan/codebase/openpi-modified/.venv/bin/python')
LIBERO_PYTHON = pathlib.Path('/vla/users/niejunnan/envs/libero/bin/python')
BASELINE_METHODS = {'pi0_k1', 'k4_first', 'k4_random'}


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec='seconds')


def append_line(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(text.rstrip() + '\n')


def shlex_quote(s: str) -> str:
    if not s:
        return "''"
    safe = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_@%+=:,./-'
    if all(c in safe for c in s):
        return s
    return "'" + s.replace("'", "'\\''") + "'"


def shell_join(cmd: list[str]) -> str:
    return ' '.join(shlex_quote(str(x)) for x in cmd)


def run_capture(cmd: list[str], timeout: float = 30.0) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(ROOT), text=True, stderr=subprocess.STDOUT, timeout=timeout)
    except Exception as exc:
        return f'[capture failed] {type(exc).__name__}: {exc}'


def load_manifest_row(path: pathlib.Path, job_id: str) -> dict[str, str]:
    with path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('job_id') == job_id:
                return row
    raise KeyError(f'job_id {job_id!r} not found in {path}')


def csv_episode_count(path: pathlib.Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open(newline='', encoding='utf-8') as f:
        return sum(1 for _ in csv.DictReader(f))


def completed_episode_ids(path: pathlib.Path, job_id: str) -> set[int]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    out: set[int] = set()
    with path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('job_id') != job_id:
                continue
            try:
                out.add(int(row.get('episode_id', -1)))
            except Exception:
                pass
    return out


def next_resume_start(path: pathlib.Path, job_id: str, episode_start: int, episode_end: int) -> tuple[int, int, bool]:
    done = completed_episode_ids(path, job_id)
    wanted = list(range(episode_start, episode_end))
    missing = [x for x in wanted if x not in done]
    if not missing:
        return episode_end, len(done.intersection(wanted)), True
    first_missing = min(missing)
    # Only support prefix completion. This matches crash behavior and prevents
    # duplicate rows from arbitrary holes.
    prefix_ok = all(x in done for x in range(episode_start, first_missing))
    suffix_clean = all(x not in done for x in range(first_missing, episode_end))
    return first_missing, len(done.intersection(wanted)), bool(prefix_ok and suffix_clean)


def write_status(path: pathlib.Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(tmp, path)


class JobLock:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self._fh = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open('a+', encoding='utf-8')
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'job is already locked by another runner: {self.path}') from exc
        self._fh.write(json.dumps({'pid': os.getpid(), 'hostname': socket.gethostname(), 'timestamp': now()}) + '\n')
        self._fh.flush()
        os.fsync(self._fh.fileno())
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fh is not None:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            finally:
                self._fh.close()
                self._fh = None


def wait_for_port(port: int, proc: subprocess.Popen, timeout: float, log_path: pathlib.Path) -> None:
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = ''
            try:
                tail = ''.join(log_path.read_text(errors='replace').splitlines(True)[-120:])
            except Exception:
                pass
            raise RuntimeError(f'server exited before port ready code={proc.returncode}\n{tail}')
        try:
            with socket.create_connection(('127.0.0.1', int(port)), timeout=2.0):
                return
        except OSError as exc:
            last_err = exc
            time.sleep(2.0)
    raise TimeoutError(f'timeout waiting for port {port}: {last_err}')


def port_is_open(port: int) -> bool:
    try:
        with socket.create_connection(('127.0.0.1', int(port)), timeout=1.0):
            return True
    except OSError:
        return False


def terminate(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 30
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.5)
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def maybe_kill_lerobot(commands_log: pathlib.Path, gpu_log: pathlib.Path, gpu_id: str, enabled: bool) -> None:
    if not enabled:
        return
    # Match only the actual auto-launched lerobot training commands. Do not match
    # this runner's --kill-lerobot flag or launcher script arguments.
    pattern = r"/codebase/lerobot|lerobot_auto|lerobot/scripts/train.py"
    out = run_capture(['bash', '-lc', f"ps -eo pid,ppid,cmd | grep -E {shlex_quote(pattern)} | grep -v grep || true"])
    append_line(gpu_log, f'### lerobot scan {now()} gpu={gpu_id}\n```text\n{out.rstrip() or "-"}\n```')
    pids = []
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if parts and parts[0].isdigit():
            pids.append(parts[0])
    for pid in sorted(set(pids)):
        append_line(commands_log, f'[{now()}] kill lerobot pid={pid} gpu={gpu_id}')
        subprocess.run(['kill', '-9', pid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> int:
    parser = argparse.ArgumentParser(description='Run one Stage2 GenSG-IGC LIBERO manifest job.')
    parser.add_argument('--manifest', default='igc_gensg/configs/stage2_gensg_manifest.csv')
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--base-port', type=int, default=8900)
    parser.add_argument('--server-timeout-sec', type=float, default=900.0)
    parser.add_argument('--run-id-override', default='')
    parser.add_argument('--kill-lerobot', action='store_true')
    parser.add_argument('--no-skip-complete', action='store_true')
    parser.add_argument('--save-video', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--save-video-mode', choices=['all', 'none', 'selective'], default='selective')
    parser.add_argument('--attention-query-policy', choices=['all', 'none', 'early'], default='early')
    parser.add_argument('--attention-early-queries', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--artifact-prefix', default='stage2_gensg')
    args = parser.parse_args()

    manifest = ROOT / args.manifest
    row = load_manifest_row(manifest, args.job_id)
    run_id = args.run_id_override.strip() or row['run_id']
    method = row['method']
    gpu_id = str(row['gpu_id'])
    k = int(row.get('k') or (1 if method == 'pi0_k1' else 4))
    execution_action_tokens = int(row.get('execution_action_tokens') or 5)
    episode_start = int(row['episode_start'])
    episode_end = int(row['episode_end'])
    expected = episode_end - episode_start
    port = int(args.port if args.port is not None else args.base_port + int(gpu_id))
    host = socket.gethostname()

    artifact_prefix = str(args.artifact_prefix).strip() or 'stage2_gensg'
    results_dir = ROOT / f'igc_gensg/results/{artifact_prefix}_shards'
    job_log_dir = ROOT / f'igc_gensg/logs/{artifact_prefix}_jobs'
    status_dir = job_log_dir / 'status'
    video_dir = ROOT / f'igc_gensg/videos/{artifact_prefix}' / row['suite'] / f"task{int(row['task_id']):02d}" / method
    attention_dir = ROOT / f'igc_gensg/figures/{artifact_prefix}_attention' / row['job_id']
    for p in (results_dir, job_log_dir, status_dir, video_dir, attention_dir):
        p.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{row['job_id']}.csv"
    query_path = job_log_dir / f"{row['job_id']}.queries.jsonl"
    server_log = job_log_dir / f"{row['job_id']}.server.log"
    eval_log = job_log_dir / f"{row['job_id']}.eval.log"
    commands_log = ROOT / f'igc_gensg/logs/commands_{artifact_prefix}.txt'
    gpu_log = ROOT / f'igc_gensg/logs/gpu_allocation_{artifact_prefix}.md'
    status_path = status_dir / f"{row['job_id']}.json"
    lock_path = job_log_dir / 'locks' / f"{row['job_id']}.lock"

    try:
        lock_ctx = JobLock(lock_path)
        lock_ctx.__enter__()
    except RuntimeError as exc:
        status = {'job_id': row['job_id'], 'status': 'locked', 'error': str(exc), 'timestamp': now(), 'hostname': host, 'gpu_id': gpu_id}
        write_status(status_path, status)
        print(json.dumps(status, sort_keys=True), file=sys.stderr)
        return 5

    try:
        resume_start, completed_count, can_resume = next_resume_start(csv_path, row['job_id'], episode_start, episode_end)
        if not args.no_skip_complete and resume_start >= episode_end:
            status = {'job_id': row['job_id'], 'status': 'skipped_complete', 'csv_path': str(csv_path), 'episodes': completed_count, 'timestamp': now(), 'hostname': host, 'gpu_id': gpu_id}
            write_status(status_path, status)
            print(json.dumps(status, sort_keys=True))
            return 0
        if not can_resume:
            status = {'job_id': row['job_id'], 'status': 'failed', 'error': 'noncontiguous_partial_csv', 'csv_path': str(csv_path), 'episodes': completed_count, 'expected_episodes': expected, 'timestamp': now(), 'hostname': host, 'gpu_id': gpu_id}
            write_status(status_path, status)
            print(json.dumps(status, sort_keys=True), file=sys.stderr)
            return 4

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu_id
        env['PYTHONUNBUFFERED'] = '1'
        append_line(commands_log, f"[{now()}] start job={row['job_id']} host={host} assigned_host={row.get('hostname')} gpu={gpu_id} method={method} suite={row['suite']} task={row['task_id']} eps={resume_start}:{episode_end} original_eps={episode_start}:{episode_end} completed_prefix={completed_count} port={port}")
        append_line(gpu_log, f"\n## job {row['job_id']} preflight {now()} host={host} assigned_host={row.get('hostname')} gpu={gpu_id}\n```text\n{run_capture(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits']).rstrip()}\n```")
        maybe_kill_lerobot(commands_log, gpu_log, gpu_id, args.kill_lerobot)
        if port_is_open(port):
            status = {
                'job_id': row['job_id'], 'status': 'failed', 'error': f'port {port} already in use before server launch',
                'hostname': host, 'assigned_hostname': row.get('hostname'), 'gpu_id': gpu_id, 'method': method,
                'suite': row['suite'], 'task_id': int(row['task_id']), 'timestamp': now(),
            }
            write_status(status_path, status)
            append_line(commands_log, f"[{now()}] fail job={row['job_id']} reason=port_in_use port={port}")
            print(json.dumps(status, sort_keys=True), file=sys.stderr)
            return 6

        server_cmd = [
            str(OPENPI_PYTHON), '-m', 'igc_gensg.scripts.serve_gensg_igc_policy',
            '--port', str(port),
            '--method', method,
            '--generation-heads', row.get('generation_head_spec') or '14:6',
            '--rescore-heads', row.get('rescore_head_spec') or '17:0',
            '--k', str(k),
            '--seed', str(row['seed_start']),
            '--execution-action-tokens', str(execution_action_tokens),
            '--attention-dir', str(attention_dir.relative_to(ROOT)),
            '--disable-torch-compile',
        ]
        if method in BASELINE_METHODS:
            server_cmd.append('--disable-attention-npz')
        append_line(commands_log, f"[{now()}] server job={row['job_id']}: CUDA_VISIBLE_DEVICES={gpu_id} {shell_join(server_cmd)}")
        server_proc: subprocess.Popen | None = None
        started = time.monotonic()
        try:
            with server_log.open('w', encoding='utf-8') as f_server:
                server_proc = subprocess.Popen(server_cmd, cwd=str(ROOT), env=env, stdout=f_server, stderr=subprocess.STDOUT, start_new_session=True)
                wait_for_port(port, server_proc, args.server_timeout_sec, server_log)
                append_line(commands_log, f"[{now()}] server ready job={row['job_id']} pid={server_proc.pid} port={port}")

                eval_cmd = [
                    str(LIBERO_PYTHON), 'igc_gensg/scripts/eval_stage2_gensg_online.py',
                    '--host', '127.0.0.1',
                    '--port', str(port),
                    '--method', method,
                    '--run-id', run_id,
                    '--job-id', row['job_id'],
                    '--task-suite-name', row['suite'],
                    '--task-ids', str(row['task_id']),
                    '--episode-start', str(resume_start),
                    '--episode-end', str(episode_end),
                    '--num-episodes', str(episode_end - resume_start),
                    '--seed', str(row['seed_start']),
                    '--output-csv', str(csv_path.relative_to(ROOT)),
                    '--query-jsonl', str(query_path.relative_to(ROOT)),
                    '--video-dir', str(video_dir.relative_to(ROOT)),
                    '--save-video-mode', args.save_video_mode,
                    '--video-success-limit', '1',
                    '--video-failure-limit', '2',
                    '--attention-query-policy', args.attention_query_policy,
                    '--attention-early-queries', str(args.attention_early_queries),
                ]
                if args.save_video:
                    eval_cmd.append('--save-video')
                if args.max_steps is not None:
                    eval_cmd.extend(['--max-steps', str(args.max_steps)])
                eval_env = env.copy()
                eval_env['MUJOCO_GL'] = 'egl'
                append_line(commands_log, f"[{now()}] eval job={row['job_id']}: MUJOCO_GL=egl CUDA_VISIBLE_DEVICES={gpu_id} {shell_join(eval_cmd)}")
                with eval_log.open('w', encoding='utf-8') as f_eval:
                    completed = subprocess.run(eval_cmd, cwd=str(ROOT), env=eval_env, stdout=f_eval, stderr=subprocess.STDOUT)
                elapsed = time.monotonic() - started
                count = csv_episode_count(csv_path)
                unique_done = len(completed_episode_ids(csv_path, row['job_id']).intersection(range(episode_start, episode_end)))
                status = {
                    'job_id': row['job_id'], 'status': 'complete' if completed.returncode == 0 and unique_done >= expected else 'failed',
                    'returncode': int(completed.returncode), 'episodes': count, 'unique_episodes': unique_done, 'expected_episodes': expected, 'resume_start': resume_start,
                    'hostname': host, 'assigned_hostname': row.get('hostname'), 'gpu_id': gpu_id, 'method': method,
                    'suite': row['suite'], 'task_id': int(row['task_id']), 'episode_start': episode_start, 'episode_end': episode_end,
                    'csv_path': str(csv_path), 'query_path': str(query_path), 'server_log': str(server_log), 'eval_log': str(eval_log),
                    'walltime_sec': elapsed, 'timestamp': now(),
                }
                write_status(status_path, status)
                append_line(commands_log, f"[{now()}] finish job={row['job_id']} status={status['status']} rc={completed.returncode} episodes={count}/{expected} unique={unique_done}/{expected} wall={elapsed:.1f}s")
                return 0 if status['status'] == 'complete' else 2
        except Exception as exc:
            status = {
                'job_id': row['job_id'], 'status': 'crashed', 'error': repr(exc), 'hostname': host,
                'assigned_hostname': row.get('hostname'), 'gpu_id': gpu_id, 'method': method,
                'suite': row['suite'], 'task_id': int(row['task_id']), 'server_log': str(server_log), 'eval_log': str(eval_log),
                'timestamp': now(),
            }
            write_status(status_path, status)
            append_line(commands_log, f"[{now()}] crash job={row['job_id']} error={repr(exc)}")
            print(json.dumps(status, sort_keys=True), file=sys.stderr)
            return 3
        finally:
            terminate(server_proc)
            append_line(gpu_log, f"\n## job {row['job_id']} post {now()} host={host} gpu={gpu_id}\n```text\n{run_capture(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits']).rstrip()}\n```")
    finally:
        lock_ctx.__exit__(None, None, None)


if __name__ == '__main__':
    raise SystemExit(main())
