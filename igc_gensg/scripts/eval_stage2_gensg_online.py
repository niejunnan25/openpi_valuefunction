
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import logging
import math
import os
import pathlib
import re
import sys
import time
from typing import Any

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
OPENPI_CLIENT_SRC = PROJECT_ROOT / "packages" / "openpi-client" / "src"
LIBERO_CANDIDATES = (
    PROJECT_ROOT / "third_party" / "LIBERO",
    PROJECT_ROOT.parent / "LIBERO",
    pathlib.Path("/vla/users/niejunnan/codebase/LIBERO"),
    pathlib.Path("/vla/users/niejunnan/codebase/openpi/third_party/LIBERO"),
)


def _is_complete_libero_root(root: pathlib.Path) -> bool:
    package = root / "libero" / "libero"
    return all(
        p.exists()
        for p in (
            package / "bddl_files",
            package / "init_files",
            package / "assets" / "scenes" / "libero_tabletop_base_style.xml",
        )
    )


def _select_libero_root() -> pathlib.Path:
    for candidate in LIBERO_CANDIDATES:
        if _is_complete_libero_root(candidate):
            return candidate
    raise FileNotFoundError(f"No complete LIBERO root found in {LIBERO_CANDIDATES}")


def _write_libero_config(libero_root: pathlib.Path) -> pathlib.Path:
    package = libero_root / "libero" / "libero"
    datasets = package.parent / "datasets"
    config_dir = PROJECT_ROOT / ".libero"
    config_dir.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            f"benchmark_root: {package.as_posix()}",
            f"bddl_files: {(package / 'bddl_files').as_posix()}",
            f"init_states: {(package / 'init_files').as_posix()}",
            f"datasets: {datasets.as_posix()}",
            f"assets: {(package / 'assets').as_posix()}",
        ]
    ) + "\n"
    (config_dir / "config.yaml").write_text(text)
    return config_dir


def setup_libero_imports() -> pathlib.Path:
    libero_root = _select_libero_root()
    os.environ["LIBERO_CONFIG_PATH"] = str(_write_libero_config(libero_root))
    for path in (str(libero_root), str(OPENPI_CLIENT_SRC)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return libero_root


_LIBERO_ROOT = setup_libero_imports()
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402


def _quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def _sanitize(text: str) -> str:
    return "".join(c if c.isalnum() or c in {"_", "-"} else "_" for c in text)[:150]


def _json_default(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(type(obj).__name__)


def _append_jsonl(path: pathlib.Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=_json_default, sort_keys=True) + "\n")


def _make_env(task, resolution: int, seed: int):
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=task_bddl_file, camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)
    return env, str(task.language)


def _obs_state(obs: dict) -> np.ndarray:
    return np.concatenate((obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))


def _policy_element(obs: dict, prompt: str, resize_size: int) -> dict:
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize_size, resize_size))
    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": _obs_state(obs),
        "prompt": prompt,
    }


def parse_task_ids(spec: str, n_tasks: int) -> list[int]:
    if spec.strip().lower() == "all":
        return list(range(n_tasks))
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = [int(x) for x in part.split("-", 1)]
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return [x for x in dict.fromkeys(out) if 0 <= x < n_tasks]


def max_steps_for_suite(name: str) -> int:
    return {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }[name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LIBERO online rollouts against a GenSG-IGC server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument("--task-ids", default="4")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--method", required=True)
    parser.add_argument("--run-id", default="pilot")
    parser.add_argument("--output-csv", default="igc_gensg/results/stage2_gensg_episode_results.csv")
    parser.add_argument("--query-jsonl", default="igc_gensg/results/stage2_gensg_query_logs.jsonl")
    parser.add_argument("--video-dir", default="igc_gensg/videos/stage2_gensg")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--save-video-mode", choices=["all", "none", "selective"], default="all")
    parser.add_argument("--video-success-limit", type=int, default=2)
    parser.add_argument("--video-failure-limit", type=int, default=2)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-end", type=int, default=None)
    parser.add_argument("--job-id", default="")
    parser.add_argument("--attention-query-policy", choices=["all", "none", "early"], default="all")
    parser.add_argument("--attention-early-queries", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    output_csv = pathlib.Path(args.output_csv)
    query_jsonl = pathlib.Path(args.query_jsonl)
    video_dir = pathlib.Path(args.video_dir)
    for p in (output_csv.parent, query_jsonl.parent, video_dir):
        p.mkdir(parents=True, exist_ok=True)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    logging.info("Server metadata: %s", metadata)
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task_ids = parse_task_ids(args.task_ids, suite.n_tasks)
    max_steps = int(args.max_steps if args.max_steps is not None else max_steps_for_suite(args.task_suite_name))
    write_header = not output_csv.exists() or output_csv.stat().st_size == 0
    with output_csv.open("a", newline="", encoding="utf-8") as f_csv:
        fields = [
            "run_id", "job_id", "suite", "task_id", "task", "instruction", "method", "episode_id", "seed", "init_state_idx",
            "success", "episode_length", "num_policy_queries", "mean_query_latency_ms", "mean_server_latency_ms",
            "mean_score_variance", "selected_index_counts", "failure_type", "video_path", "total_walltime_sec", "timestamp",
            "server_heads", "server_k", "server_method", "server_host", "server_gpu",
            "server_generation_heads", "server_rescore_heads",
        ]
        video_counts = collections.Counter()
        writer = csv.DictWriter(f_csv, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for task_id in task_ids:
            task = suite.get_task(task_id)
            initial_states = suite.get_task_init_states(task_id)
            env, task_description = _make_env(task, LIBERO_ENV_RESOLUTION, args.seed + task_id)
            try:
                episode_end = int(args.episode_end) if args.episode_end is not None else int(args.episode_start + args.num_episodes)
                episode_iter = range(int(args.episode_start), episode_end)
                for episode_idx in tqdm.tqdm(episode_iter, desc=f"{args.method} {args.task_suite_name}:{task_id}"):
                    episode_wall_start = time.monotonic()
                    episode_seed = int(args.seed + episode_idx)
                    init_state_idx = episode_idx % len(initial_states)
                    obs = env.reset()
                    obs = env.set_init_state(initial_states[init_state_idx])
                    action_plan: collections.deque = collections.deque()
                    replay_images = []
                    done = False
                    t = 0
                    policy_queries = 0
                    query_records = []
                    while t < max_steps + args.num_steps_wait:
                        if t < args.num_steps_wait:
                            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                            t += 1
                            continue
                        element = _policy_element(obs, task_description, args.resize_size)
                        replay_images.append(element["observation/image"])
                        if not action_plan:
                            save_attention = (
                                args.attention_query_policy == "all"
                                or (args.attention_query_policy == "early" and policy_queries < args.attention_early_queries)
                            )
                            context = {
                                "run_id": args.run_id,
                                "job_id": args.job_id,
                                "suite": args.task_suite_name,
                                "task_id": task_id,
                                "task": str(task_description),
                                "episode_id": episode_idx,
                                "method": args.method,
                                "seed": episode_seed,
                                "query_index": policy_queries,
                                "timestep": t,
                                "save_attention": save_attention,
                            }
                            payload = dict(element)
                            payload["__stage2_context"] = context
                            q0 = time.monotonic()
                            result = client.infer(payload)
                            client_ms = (time.monotonic() - q0) * 1000.0
                            action_chunk = np.asarray(result["actions"], dtype=np.float32)
                            if len(action_chunk) < args.replan_steps:
                                raise RuntimeError(f"short action chunk: {len(action_chunk)}")
                            action_plan.extend(action_chunk[: args.replan_steps])
                            rec = dict(result.get("igc_select", {}))
                            rec.update(context)
                            rec["client_latency_ms"] = client_ms
                            rec["server_timing"] = result.get("server_timing", {})
                            rec["policy_timing"] = result.get("policy_timing", {})
                            query_records.append(rec)
                            policy_queries += 1
                        action = action_plan.popleft()
                        obs, _, done, info = env.step(np.asarray(action).tolist())
                        if done:
                            break
                        t += 1
                    suffix = "success" if done else "failure"
                    episode_walltime = time.monotonic() - episode_wall_start
                    video_path = ""
                    should_save_video = bool(args.save_video and replay_images and args.save_video_mode != "none")
                    if args.save_video_mode == "selective":
                        key = "success" if done else "failure"
                        limit = int(args.video_success_limit if done else args.video_failure_limit)
                        should_save_video = bool(should_save_video and video_counts[(args.task_suite_name, task_id, args.method, key)] < limit)
                    if should_save_video:
                        name = f"{args.run_id}_{args.method}_{args.task_suite_name}_task{task_id:02d}_ep{episode_idx:03d}_{suffix}_{_sanitize(task_description)}.mp4"
                        video_path = str(video_dir / name)
                        imageio.mimwrite(video_path, replay_images, fps=10)
                        key = "success" if done else "failure"
                        video_counts[(args.task_suite_name, task_id, args.method, key)] += 1
                    for rec in query_records:
                        rec["success"] = bool(done)
                        rec["failure_type"] = "success" if done else "no_success"
                        rec["episode_length"] = int(t)
                        rec["num_policy_queries"] = int(policy_queries)
                        rec["video_path"] = video_path
                        _append_jsonl(query_jsonl, rec)
                    selected = [int(r.get("selected_index", -1)) for r in query_records if r.get("selected_index") is not None]
                    score_vars = [float(r.get("score_variance", 0.0)) for r in query_records if r.get("score_variance") is not None]
                    server_lat = [
                        float(
                            r.get("server_timing", {}).get(
                                "infer_ms",
                                r.get("policy_timing", {}).get("infer_ms", r.get("runtime_ms", 0.0)),
                            )
                            or 0.0
                        )
                        for r in query_records
                    ]
                    client_lat = [float(r.get("client_latency_ms", 0.0)) for r in query_records]
                    row = {
                        "run_id": args.run_id,
                        "job_id": args.job_id,
                        "suite": args.task_suite_name,
                        "task_id": task_id,
                        "task": str(task_description),
                        "instruction": str(task_description),
                        "method": args.method,
                        "episode_id": episode_idx,
                        "seed": episode_seed,
                        "init_state_idx": init_state_idx,
                        "success": bool(done),
                        "episode_length": int(t),
                        "num_policy_queries": int(policy_queries),
                        "mean_query_latency_ms": float(np.mean(client_lat)) if client_lat else 0.0,
                        "mean_server_latency_ms": float(np.mean(server_lat)) if server_lat else 0.0,
                        "mean_score_variance": float(np.mean(score_vars)) if score_vars else 0.0,
                        "selected_index_counts": json.dumps(dict(collections.Counter(selected)), sort_keys=True),
                        "failure_type": "success" if done else "no_success",
                        "video_path": video_path,
                        "total_walltime_sec": float(episode_walltime),
                        "timestamp": dt.datetime.now().isoformat(),
                        "server_heads": json.dumps(metadata.get("heads")),
                        "server_generation_heads": json.dumps(metadata.get("generation_heads")),
                        "server_rescore_heads": json.dumps(metadata.get("rescore_heads")),
                        "server_k": metadata.get("k"),
                        "server_method": metadata.get("method"),
                        "server_host": metadata.get("server_type"),
                        "server_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    }
                    writer.writerow(row)
                    f_csv.flush()
                    logging.info("episode row: %s", row)
            finally:
                env.close()


if __name__ == "__main__":
    main()
