#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import pathlib
import sys
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from igc_self_grounded.scripts.stage1_oracle_rollout import (
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    _make_env,
    benchmark,
    oracle_masks,
)


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        if not fields:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    if total <= eps:
        return np.full(arr.shape, 1.0 / max(1, arr.size), dtype=np.float64)
    return arr / total


def density_lift(attn: np.ndarray, mask: np.ndarray) -> float:
    p = normalize(attn)
    m = np.clip(np.asarray(mask, dtype=np.float64).reshape(-1), 0.0, 1.0)
    area = float(m.sum())
    if area <= 1e-9:
        return float("nan")
    return float(((p * m).sum() / area) * p.size)


def rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2:
        return float("nan")
    xr = np.argsort(np.argsort(x)).astype(np.float64)
    yr = np.argsort(np.argsort(y)).astype(np.float64)
    xs = float(np.std(xr))
    ys = float(np.std(yr))
    if xs <= 1e-12 or ys <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def attention_array(data: np.lib.npyio.NpzFile) -> np.ndarray | None:
    for key in data.files:
        if key.endswith("_A") and key != "selected_A":
            arr = np.asarray(data[key], dtype=np.float64)
            if arr.ndim == 2:
                return arr
            if arr.ndim == 3:
                return arr.reshape(arr.shape[0], -1)
    if "selected_A" in data.files:
        arr = np.asarray(data["selected_A"], dtype=np.float64)
        if arr.ndim == 1:
            return arr[None, :]
        if arr.ndim == 2:
            return arr.reshape(1, -1)
    return None


def load_attention(path: pathlib.Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    with np.load(path, allow_pickle=True) as data:
        attn = attention_array(data)
        scores = np.asarray(data["scores"], dtype=np.float64).reshape(-1) if "scores" in data.files else None
    return attn, scores


def make_mask_cache(manifest_rows: dict[str, dict[str, str]], resize_size: int, grid_size: int, wait_steps: int):
    suites: dict[str, Any] = {}
    env_cache: dict[tuple[str, int], tuple[Any, list[Any], Any]] = {}
    mask_cache: dict[tuple[str, int, int], dict[str, Any]] = {}

    def get_masks(rec: dict[str, Any]) -> dict[str, Any]:
        suite_name = str(rec["suite"])
        task_id = int(rec["task_id"])
        episode_id = int(rec["episode_id"])
        key = (suite_name, task_id, episode_id)
        if key in mask_cache:
            return mask_cache[key]
        if suite_name not in suites:
            suites[suite_name] = benchmark.get_benchmark_dict()[suite_name]()
        suite = suites[suite_name]
        env_key = (suite_name, task_id)
        if env_key not in env_cache:
            task = suite.get_task(task_id)
            row = manifest_rows.get(str(rec.get("job_id")), {})
            seed_start = int(row.get("seed_start", rec.get("seed", 0)))
            env, _ = _make_env(task, LIBERO_ENV_RESOLUTION, seed_start + task_id, 0)
            env_cache[env_key] = (env, suite.get_task_init_states(task_id), task)
        env, initial_states, _task = env_cache[env_key]
        init_state_idx = episode_id % len(initial_states)
        obs = env.reset()
        obs = env.set_init_state(initial_states[init_state_idx])
        for _ in range(wait_steps):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        info = oracle_masks(obs, env, resize_size=resize_size, grid_size=grid_size, entity_names=[])
        mask_cache[key] = info
        return info

    return get_masks


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["mask_name"]))].append(row)
    out: list[dict[str, Any]] = []
    for (method, mask_name), rs in sorted(grouped.items()):
        selected = np.asarray([float(r["selected_lift"]) for r in rs], dtype=np.float64)
        mean_all = np.asarray([float(r["mean_candidate_lift"]) for r in rs], dtype=np.float64)
        best = np.asarray([float(r["oracle_best_lift"]) for r in rs], dtype=np.float64)
        spearman = np.asarray([float(r["score_oracle_spearman"]) for r in rs], dtype=np.float64)
        finite_s = spearman[np.isfinite(spearman)]
        out.append(
            {
                "method": method,
                "mask_name": mask_name,
                "n": len(rs),
                "selected_lift_mean": float(np.nanmean(selected)),
                "mean_candidate_lift": float(np.nanmean(mean_all)),
                "oracle_best_lift_mean": float(np.nanmean(best)),
                "selected_minus_mean": float(np.nanmean(selected - mean_all)),
                "selected_is_oracle_best_frac": float(np.mean([bool(r["selected_is_oracle_best"]) for r in rs])),
                "score_oracle_spearman_mean": float(np.mean(finite_s)) if finite_s.size else float("nan"),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle query-0 diagnostic for completed Stage2 GenSG rollouts.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--query-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--methods", default="gensg_last3,gensg_last3_no_q,gensg_last3_no_W,gensg_last3_score_mismatch,2.2_pure_last3_imc")
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    query_path = ROOT / args.query_jsonl
    manifest_rows = {row["job_id"]: row for row in read_csv(manifest_path)}
    methods = {x.strip() for x in args.methods.split(",") if x.strip()}
    records = [
        r
        for r in read_jsonl(query_path)
        if int(r.get("query_index", -1)) == 0
        and str(r.get("method")) in methods
        and r.get("attention_map_path")
    ]
    get_masks = make_mask_cache(manifest_rows, args.resize_size, args.grid_size, args.wait_steps)
    metric_rows: list[dict[str, Any]] = []
    for rec in records:
        attn_path = pathlib.Path(str(rec["attention_map_path"]))
        if not attn_path.is_absolute():
            attn_path = ROOT / attn_path
        if not attn_path.exists():
            continue
        attn, scores = load_attention(attn_path)
        if attn is None or scores is None or attn.shape[0] != scores.shape[0]:
            continue
        selected = int(rec.get("selected_index", int(np.argmax(scores))))
        mask_info = get_masks(rec)
        masks = mask_info["masks"]
        mask_names = ["task_union", "manipulated", "destination"]
        for obj in mask_info.get("obj_of_interest", []):
            mask_names.append(f"object::{obj}")
        for mask_name in mask_names:
            mask = masks.get(mask_name)
            if mask is None or float(np.sum(mask)) <= 1e-9:
                continue
            lifts = np.asarray([density_lift(a, mask) for a in attn], dtype=np.float64)
            oracle_best = int(np.nanargmax(lifts))
            metric_rows.append(
                {
                    "run_id": rec.get("run_id"),
                    "job_id": rec.get("job_id"),
                    "suite": rec.get("suite"),
                    "task_id": rec.get("task_id"),
                    "episode_id": rec.get("episode_id"),
                    "method": rec.get("method"),
                    "mask_name": mask_name,
                    "selected_index": selected,
                    "oracle_best_index": oracle_best,
                    "selected_lift": float(lifts[selected]),
                    "mean_candidate_lift": float(np.nanmean(lifts)),
                    "oracle_best_lift": float(lifts[oracle_best]),
                    "selected_minus_mean": float(lifts[selected] - np.nanmean(lifts)),
                    "selected_is_oracle_best": bool(selected == oracle_best),
                    "score_oracle_spearman": rank_corr(scores, lifts),
                    "obj_of_interest": json.dumps(mask_info.get("obj_of_interest", []), sort_keys=True),
                    "attention_path": str(attn_path.relative_to(ROOT)),
                }
            )
    summary_rows = summarize(metric_rows)
    output_csv = ROOT / args.output_csv
    summary_csv = ROOT / args.summary_csv
    report_path = ROOT / args.report
    write_csv(output_csv, metric_rows)
    write_csv(summary_csv, summary_rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage2 GenSG Oracle Query-0 Diagnostic",
        "",
        "- Oracle masks are simulator-only and used only for offline diagnostics.",
        f"- Query records inspected: `{len(records)}`; metric rows: `{len(metric_rows)}`.",
        "- This diagnostic only covers `query_index=0`, because later simulator states were not saved.",
        "",
        "## Summary",
        "",
        "| method | mask | n | selected lift | candidate mean | oracle best | selected-mean | best frac | score/oracle rank corr |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {method} | {mask_name} | {n} | {selected_lift_mean:.4f} | {mean_candidate_lift:.4f} | "
            "{oracle_best_lift_mean:.4f} | {selected_minus_mean:.4f} | {selected_is_oracle_best_frac:.3f} | "
            "{score_oracle_spearman_mean:.4f} |".format(**row)
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"oracle_query0_records={len(records)} metrics={len(metric_rows)} report={report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
