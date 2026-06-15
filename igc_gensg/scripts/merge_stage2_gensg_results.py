#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import pathlib
import re
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
BASELINES = ["pi0_k1", "k4_first", "k4_random"]
REFERENCE = "2.1_rescore_best"
PURE_22 = ["2.2_pure_last1_imc", "2.2_pure_last3_imc"]
GENSG = ["gensg_last1", "gensg_last3", "gensg_last3_no_q", "gensg_last3_no_W"]
STAGE3_CONTROLS = [
    "gensg_early",
    "gensg_middle",
    "gensg_all_steps",
    "gensg_last3_random_token_map",
    "gensg_last3_shuffled_token_map",
    "gensg_last3_random_prefix_head",
    "gensg_last3_random_action_head",
    "gensg_last3_bottom_action_head",
    "gensg_last3_prefix_only",
    "gensg_last3_score_mismatch",
]
SELECTORS_22 = PURE_22 + GENSG + STAGE3_CONTROLS
FOCUS_22 = [
    "gensg_last3",
    "gensg_last1",
    "2.2_pure_last3_imc",
    "2.2_pure_last1_imc",
    "gensg_early",
    "gensg_middle",
    "gensg_all_steps",
    "gensg_last3_random_token_map",
    "gensg_last3_shuffled_token_map",
    "gensg_last3_random_prefix_head",
    "gensg_last3_random_action_head",
    "gensg_last3_bottom_action_head",
    "gensg_last3_prefix_only",
    "gensg_last3_score_mismatch",
]


def as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "yes", "success"}


def read_csv_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
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


def merge_csv_shards(shard_dir: pathlib.Path, allowed_job_ids: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard in sorted(shard_dir.glob("*.csv")):
        rows.extend(read_csv_rows(shard))
    if allowed_job_ids:
        rows = [r for r in rows if r.get("job_id") in allowed_job_ids]
    return rows


def read_jsonl_shards(job_log_dir: pathlib.Path, allowed_job_ids: set[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for shard in sorted(job_log_dir.glob("*.queries.jsonl")):
        if not shard.exists() or shard.stat().st_size == 0:
            continue
        with shard.open(encoding="utf-8") as fin:
            for line in fin:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if allowed_job_ids and rec.get("job_id") not in allowed_job_ids:
                    continue
                records.append(rec)
    return records


def write_jsonl(path: pathlib.Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for rec in records:
            fout.write(json.dumps(rec, sort_keys=True) + "\n")


def _to_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _timestamp_key(value: Any) -> tuple[int, str]:
    text = str(value or "")
    if not text:
        return (1, "")
    return (0, text)


def _job_sort_key(job_id: str) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", str(job_id))
    out: list[Any] = []
    for part in parts:
        if part.isdigit():
            out.append(int(part))
        elif part:
            out.append(part)
    return tuple(out)


def expected_episode_keys(manifest_rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for row in manifest_rows:
        job_id = str(row.get("job_id") or "")
        if not job_id:
            continue
        for ep in range(_to_int(row.get("episode_start"), 0), _to_int(row.get("episode_end"), 0)):
            keys.add((job_id, ep))
    return keys


def dedupe_episode_rows(rows: list[dict[str, Any]], manifest_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = expected_episode_keys(manifest_rows)
    invalid_rows = 0
    unexpected_rows = 0
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        job_id = str(row.get("job_id") or "")
        ep = _to_int(row.get("episode_id"))
        if not job_id or ep < 0:
            invalid_rows += 1
            continue
        key = (job_id, ep)
        if expected and key not in expected:
            unexpected_rows += 1
            continue
        grouped[key].append(row)

    kept: list[dict[str, Any]] = []
    duplicate_groups = 0
    duplicate_rows = 0
    duplicate_success_conflicts = 0
    duplicate_examples = []
    for key, rs in grouped.items():
        if len(rs) > 1:
            duplicate_groups += 1
            duplicate_rows += len(rs) - 1
            success_values = {str(r.get("success")) for r in rs}
            if len(success_values) > 1:
                duplicate_success_conflicts += 1
            if len(duplicate_examples) < 20:
                duplicate_examples.append(
                    {
                        "job_id": key[0],
                        "episode_id": key[1],
                        "n": len(rs),
                        "success_values": sorted(success_values),
                        "timestamps": [str(r.get("timestamp", "")) for r in rs[:5]],
                    }
                )
        # Prefer the earliest valid row. Duplicate rescue jobs should be identical
        # when deterministic; if not, this conservative rule avoids letting a later
        # rescue overwrite the originally observed episode outcome.
        kept.append(sorted(rs, key=lambda r: (_timestamp_key(r.get("timestamp")), str(r)))[0])

    kept.sort(key=lambda r: (_job_sort_key(str(r.get("job_id"))), _to_int(r.get("episode_id"))))
    missing = sorted(expected.difference(grouped.keys()), key=lambda x: (_job_sort_key(x[0]), x[1])) if expected else []
    diagnostics = {
        "raw_episode_rows": len(rows),
        "final_episode_rows": len(kept),
        "expected_episode_rows": len(expected),
        "invalid_episode_rows_dropped": invalid_rows,
        "unexpected_episode_rows_dropped": unexpected_rows,
        "duplicate_episode_groups": duplicate_groups,
        "duplicate_episode_rows_dropped": duplicate_rows,
        "duplicate_success_conflict_groups": duplicate_success_conflicts,
        "missing_episode_rows": len(missing),
        "missing_episode_examples": [{"job_id": job, "episode_id": ep} for job, ep in missing[:50]],
        "duplicate_episode_examples": duplicate_examples,
    }
    return kept, diagnostics


def dedupe_query_records(records: list[dict[str, Any]], kept_episode_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept_episodes = {(str(r.get("job_id") or ""), _to_int(r.get("episode_id"))) for r in kept_episode_rows}
    grouped: dict[tuple[str, int, int, str], list[dict[str, Any]]] = collections.defaultdict(list)
    invalid = 0
    unmatched = 0
    for rec in records:
        job_id = str(rec.get("job_id") or "")
        ep = _to_int(rec.get("episode_id"))
        q = _to_int(rec.get("query_index"))
        method = str(rec.get("method") or "")
        if not job_id or ep < 0 or q < 0:
            invalid += 1
            continue
        if (job_id, ep) not in kept_episodes:
            unmatched += 1
            continue
        grouped[(job_id, ep, q, method)].append(rec)

    kept: list[dict[str, Any]] = []
    duplicate_groups = 0
    duplicate_records = 0
    duplicate_examples = []
    for key, rs in grouped.items():
        if len(rs) > 1:
            duplicate_groups += 1
            duplicate_records += len(rs) - 1
            if len(duplicate_examples) < 20:
                duplicate_examples.append({"job_id": key[0], "episode_id": key[1], "query_index": key[2], "method": key[3], "n": len(rs)})
        kept.append(sorted(rs, key=lambda r: (_timestamp_key(r.get("timestamp")), str(r)))[0])
    kept.sort(key=lambda r: (_job_sort_key(str(r.get("job_id"))), _to_int(r.get("episode_id")), _to_int(r.get("query_index")), str(r.get("method") or "")))
    diagnostics = {
        "raw_query_records": len(records),
        "final_query_records": len(kept),
        "invalid_query_records_dropped": invalid,
        "unmatched_query_records_dropped": unmatched,
        "duplicate_query_groups": duplicate_groups,
        "duplicate_query_records_dropped": duplicate_records,
        "duplicate_query_examples": duplicate_examples,
    }
    return kept, diagnostics


def enrich(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        row["success_bool"] = as_bool(row.get("success"))
        row["suite_task"] = f"{row.get('suite')}:{row.get('task_id')}"
        for key in [
            "episode_length",
            "num_policy_queries",
            "mean_query_latency_ms",
            "mean_server_latency_ms",
            "mean_score_variance",
            "total_walltime_sec",
        ]:
            try:
                row[key + "_num"] = float(row.get(key, 0) or 0)
            except Exception:
                row[key + "_num"] = 0.0
    return rows


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def fmt(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value:.{digits}f}"


def group(rows: list[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    out: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        out[tuple(row.get(key) for key in keys)].append(row)
    return dict(out)


def summary_rows(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    out = []
    for key_vals, rs in sorted(group(rows, *keys).items()):
        vals = [1.0 if r["success_bool"] else 0.0 for r in rs]
        row = {key: value for key, value in zip(keys, key_vals)}
        row.update(
            {
                "n": len(rs),
                "success": mean(vals),
                "success_count": int(sum(vals)),
                "episode_length": mean([r["episode_length_num"] for r in rs]),
                "queries": mean([r["num_policy_queries_num"] for r in rs]),
                "client_latency_ms": mean([r["mean_query_latency_ms_num"] for r in rs]),
                "server_latency_ms": mean([r["mean_server_latency_ms_num"] for r in rs]),
                "score_variance": mean([r["mean_score_variance_num"] for r in rs]),
                "walltime_sec": mean([r["total_walltime_sec_num"] for r in rs]),
            }
        )
        out.append(row)
    return out


def bootstrap_diff(
    rows: list[dict[str, Any]],
    method: str,
    baseline: str,
    suite_filter: str | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    scoped = [r for r in rows if suite_filter is None or r.get("suite") == suite_filter]
    tasks = sorted({r["suite_task"] for r in scoped})
    method_by_task: dict[str, list[float]] = {}
    base_by_task: dict[str, list[float]] = {}
    pairs = []
    for task in tasks:
        mv = [1.0 if r["success_bool"] else 0.0 for r in scoped if r["suite_task"] == task and r.get("method") == method]
        bv = [1.0 if r["success_bool"] else 0.0 for r in scoped if r["suite_task"] == task and r.get("method") == baseline]
        method_by_task[task] = mv
        base_by_task[task] = bv
        if mv and bv:
            pairs.append((mean(mv), mean(bv)))
    if not pairs:
        return float("nan"), float("nan"), float("nan")
    point = mean([m - b for m, b in pairs])
    diffs = []
    for _ in range(n_boot):
        ds = []
        for task in tasks:
            mv = method_by_task[task]
            bv = base_by_task[task]
            if not mv or not bv:
                continue
            ds.append(float(np.mean(rng.choice(mv, size=len(mv), replace=True)) - np.mean(rng.choice(bv, size=len(bv), replace=True))))
        if ds:
            diffs.append(float(np.mean(ds)))
    if not diffs:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(np.asarray(diffs, dtype=np.float64), [0.025, 0.975])
    return point, float(lo), float(hi)


def win_loss_tie(rows: list[dict[str, Any]], method: str, baseline: str, suite_filter: str | None = None) -> tuple[int, int, int]:
    scoped = [r for r in rows if suite_filter is None or r.get("suite") == suite_filter]
    tasks = sorted({r["suite_task"] for r in scoped})
    wins = losses = ties = 0
    for task in tasks:
        mv = [1.0 if r["success_bool"] else 0.0 for r in scoped if r["suite_task"] == task and r.get("method") == method]
        bv = [1.0 if r["success_bool"] else 0.0 for r in scoped if r["suite_task"] == task and r.get("method") == baseline]
        if not mv or not bv:
            continue
        diff = mean(mv) - mean(bv)
        if diff > 0:
            wins += 1
        elif diff < 0:
            losses += 1
        else:
            ties += 1
    return wins, losses, ties


def compare_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = [m for m in SELECTORS_22 if any(r.get("method") == m for r in rows)]
    scopes = ["overall"] + sorted({str(r.get("suite")) for r in rows})
    baselines = ["k4_random", REFERENCE, "2.2_pure_last3_imc", "gensg_last3_no_q", "gensg_last3_no_W"]
    out = []
    for scope in scopes:
        suite_filter = None if scope == "overall" else scope
        for method in methods:
            for baseline in baselines:
                if method == baseline or not any(r.get("method") == baseline for r in rows):
                    continue
                diff, lo, hi = bootstrap_diff(rows, method, baseline, suite_filter=suite_filter)
                wins, losses, ties = win_loss_tie(rows, method, baseline, suite_filter=suite_filter)
                out.append(
                    {
                        "scope": scope,
                        "method": method,
                        "baseline": baseline,
                        "diff": diff,
                        "ci_low": lo,
                        "ci_high": hi,
                        "wins": wins,
                        "losses": losses,
                        "ties": ties,
                    }
                )
    return out


def query_diagnostics(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_method: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    by_selected: dict[tuple[str, str, str], collections.Counter] = collections.defaultdict(collections.Counter)
    by_var: dict[tuple[str, str, str], list[float]] = collections.defaultdict(list)
    for rec in records:
        method = str(rec.get("method"))
        by_method[method].append(rec)
        if rec.get("selected_index") is not None:
            by_selected[(str(rec.get("suite")), str(rec.get("task_id")), method)][int(rec.get("selected_index"))] += 1
            by_selected[("ALL", "ALL", method)][int(rec.get("selected_index"))] += 1
        try:
            var = float(rec.get("score_variance", 0.0) or 0.0)
        except Exception:
            continue
        by_var[(str(rec.get("suite")), str(rec.get("task_id")), method)].append(var)
        by_var[("ALL", "ALL", method)].append(var)
    method_rows = []
    component_keys = ["image_mass", "visual_concentration", "language_mass", "alignment"]

    def selected_component(rec: dict[str, Any], key: str) -> float:
        value = rec.get("selected_score_components", {})
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                value = {}
        if isinstance(value, dict) and key in value:
            try:
                return float(value[key])
            except Exception:
                return float("nan")
        return float("nan")

    for method, rs in sorted(by_method.items()):
        selected = [int(r.get("selected_index")) for r in rs if r.get("selected_index") is not None]
        counts = collections.Counter(selected)
        total = sum(counts.values())
        probs = np.asarray([v / total for v in counts.values()], dtype=np.float64) if total else np.asarray([])
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)) / math.log(max(len(counts), 2))) if total else float("nan")
        row = {
            "method": method,
            "n_queries": len(rs),
            "selected_counts": json.dumps(dict(sorted(counts.items())), sort_keys=True),
            "selected_max_frac": max(counts.values()) / total if total else float("nan"),
            "selected_entropy_norm": max(0.0, min(1.0, entropy)) if math.isfinite(entropy) else entropy,
            "score_variance_mean": mean([float(r.get("score_variance", 0.0) or 0.0) for r in rs]),
            "runtime_ms": mean([float(r.get("runtime_ms", 0.0) or 0.0) for r in rs]),
            "generation_attention_runtime_ms": mean([float(r.get("generation_attention_runtime_ms", 0.0) or 0.0) for r in rs]),
            "scoring_ms": mean([float(r.get("scoring_ms", 0.0) or 0.0) for r in rs]),
            "candidate_pairwise_l2_mean": mean([float(r.get("candidate_action_pairwise_l2_mean", 0.0) or 0.0) for r in rs]),
        }
        for key in component_keys:
            vals = [selected_component(r, key) for r in rs]
            vals = [v for v in vals if math.isfinite(v)]
            row[f"selected_{key}_mean"] = mean(vals)
        method_rows.append(row)
    selected_rows = []
    for (suite, task_id, method), counts in sorted(by_selected.items()):
        total = sum(counts.values())
        for idx, n in sorted(counts.items()):
            selected_rows.append({"suite": suite, "task_id": task_id, "method": method, "selected_index": idx, "n": n, "frac": n / total if total else float("nan"), "total": total})
    var_rows = []
    for (suite, task_id, method), values in sorted(by_var.items()):
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            continue
        var_rows.append(
            {
                "suite": suite,
                "task_id": task_id,
                "method": method,
                "n": int(arr.size),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "p50": float(np.quantile(arr, 0.50)),
                "p95": float(np.quantile(arr, 0.95)),
                "zero_frac": float(np.mean(arr == 0.0)),
            }
        )
    return method_rows, selected_rows, var_rows


def load_statuses(status_dir: pathlib.Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted(status_dir.glob("*.json")):
        try:
            out.append(json.loads(path.read_text()))
        except Exception:
            pass
    return out


def choose_conclusion(rows: list[dict[str, Any]], comparisons: list[dict[str, Any]], manifest_rows: list[dict[str, Any]]) -> tuple[str, str]:
    if not rows:
        return "证据不足", "No merged episode rows."
    expected_eps = sum(int(r["episode_end"]) - int(r["episode_start"]) for r in manifest_rows) if manifest_rows else 0
    complete_frac = len(rows) / expected_eps if expected_eps else 0.0
    expected_by_method: collections.Counter[str] = collections.Counter()
    for row in manifest_rows:
        expected_by_method[str(row.get("method"))] += int(row["episode_end"]) - int(row["episode_start"])
    observed_by_method: collections.Counter[str] = collections.Counter(str(row.get("method")) for row in rows)
    min_required = {}
    for method in ("gensg_last3", "k4_random"):
        expected = expected_by_method.get(method, 0)
        if expected:
            min_required[method] = min(expected, max(30, math.ceil(0.5 * expected)))
        else:
            min_required[method] = 30
    missing = {
        method: {"observed": observed_by_method.get(method, 0), "required": required}
        for method, required in min_required.items()
        if observed_by_method.get(method, 0) < required
    }
    if missing:
        return "证据不足", f"Primary comparison is under-sampled: {json.dumps(missing, sort_keys=True)}."
    if expected_eps and complete_frac < 0.5:
        return "证据不足", f"Run is less than half complete: merged={len(rows)}, expected={expected_eps}, complete_frac={complete_frac:.3f}."
    primary = next((r for r in comparisons if r["scope"] == "overall" and r["baseline"] == "k4_random" and r["method"] == "gensg_last3"), None)
    if primary is None:
        return "证据不足", "No gensg_last3 vs k4_random comparison row."
    pure = next((r for r in comparisons if r["scope"] == "overall" and r["baseline"] == "2.2_pure_last3_imc" and r["method"] == "gensg_last3"), None)
    positive_suites = sum(1 for r in comparisons if r["scope"] != "overall" and r["baseline"] == "k4_random" and r["method"] == "gensg_last3" and r["diff"] > 0)
    pure_ok = pure is None or (primary["diff"] > 0 and pure["diff"] >= 0)
    if primary["diff"] > 0 and primary["ci_low"] > 0 and positive_suites >= 2 and primary["wins"] > primary["losses"] and pure_ok:
        return "可用", f"gensg_last3 beats k4_random by {primary['diff']:.3f}, CI [{primary['ci_low']:.3f}, {primary['ci_high']:.3f}], positive suites={positive_suites}."
    if primary["diff"] > 0 and positive_suites >= 1 and complete_frac >= 0.5:
        return "部分可用", f"gensg_last3 is above k4_random by {primary['diff']:.3f}, but CI/pure/suite-spread criteria are not fully satisfied; CI [{primary['ci_low']:.3f}, {primary['ci_high']:.3f}], positive suites={positive_suites}, complete_frac={complete_frac:.3f}."
    if primary["diff"] <= 0:
        return "不可用", f"gensg_last3 does not beat k4_random: diff={primary['diff']:.3f}, CI [{primary['ci_low']:.3f}, {primary['ci_high']:.3f}]."
    return "证据不足", "Evidence is incomplete or unstable."


def write_report(
    path: pathlib.Path,
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    statuses: list[dict[str, Any]],
    summary_method: list[dict[str, Any]],
    summary_suite: list[dict[str, Any]],
    summary_task: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    method_diag: list[dict[str, Any]],
    dedup_diag: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_eps = sum(int(r["episode_end"]) - int(r["episode_start"]) for r in manifest_rows) if manifest_rows else 0
    completed_jobs = sum(1 for s in statuses if s.get("status") in {"complete", "skipped_complete"})
    failed_jobs = [s for s in statuses if s.get("status") not in {"complete", "skipped_complete"}]
    conclusion, reason = choose_conclusion(rows, comparisons, manifest_rows)
    methods = [m for m in BASELINES + [REFERENCE] + SELECTORS_22 if any(r.get("method") == m for r in rows)]
    title = "# Stage3 GenSG-IGC Controls and Steering Report" if "stage3" in path.name else "# Stage2 GenSG-IGC Online Reranking Report"
    lines = [
        title,
        "",
        f"- Final conclusion: **{conclusion}**",
        f"- Reason: {reason}",
        f"- Expected episodes: `{expected_eps}`; merged episodes: `{len(rows)}`; query records: `{len(records)}`.",
        f"- Raw episode rows: `{dedup_diag.get('raw_episode_rows', 'NA')}`; duplicate episode rows dropped: `{dedup_diag.get('duplicate_episode_rows_dropped', 'NA')}`; duplicate success-conflict groups: `{dedup_diag.get('duplicate_success_conflict_groups', 'NA')}`.",
        f"- Raw query records: `{dedup_diag.get('raw_query_records', 'NA')}`; duplicate query records dropped: `{dedup_diag.get('duplicate_query_records_dropped', 'NA')}`; unmatched query records dropped: `{dedup_diag.get('unmatched_query_records_dropped', 'NA')}`.",
        f"- Completed/skipped jobs: `{completed_jobs}`; failed status files: `{len(failed_jobs)}`.",
        "- Main baseline: `k4_random`.",
        "- Main reference: `2.1_rescore_best`; pure attention reference: `2.2_pure_last3_imc`.",
        "",
        "## Overall By Method",
        "| method | n | success | client ms | server ms | score var |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        row = next((r for r in summary_method if r.get("method") == method), None)
        if row:
            lines.append(f"| {method} | {row['n']} | {fmt(row['success'])} | {fmt(row['client_latency_ms'],1)} | {fmt(row['server_latency_ms'],1)} | {fmt(row['score_variance'],6)} |")
    lines.extend(["", "## GenSG / Pure / Ablation Comparisons", "| scope | method | baseline | diff | 95% CI | wins | losses | ties |", "|---|---|---|---:|---:|---:|---:|---:|"])
    for row in comparisons:
        if row["method"] in FOCUS_22:
            lines.append(f"| {row['scope']} | {row['method']} | {row['baseline']} | {fmt(row['diff'])} | [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] | {row['wins']} | {row['losses']} | {row['ties']} |")
    lines.extend(["", "## Query Diagnostics", "| method | queries | selected max frac | selected entropy | score var | runtime ms | generation attention ms | 2.1 scoring ms | action L2 | image mass | concentration | language mass | alignment |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in method_diag:
        if row["method"] in methods:
            lines.append(
                f"| {row['method']} | {row['n_queries']} | {fmt(row['selected_max_frac'])} | "
                f"{fmt(row['selected_entropy_norm'])} | {fmt(row['score_variance_mean'],6)} | "
                f"{fmt(row['runtime_ms'],1)} | {fmt(row['generation_attention_runtime_ms'],1)} | "
                f"{fmt(row['scoring_ms'],1)} | {fmt(row['candidate_pairwise_l2_mean'],3)} | "
                f"{fmt(row.get('selected_image_mass_mean', float('nan')),4)} | "
                f"{fmt(row.get('selected_visual_concentration_mean', float('nan')),4)} | "
                f"{fmt(row.get('selected_language_mass_mean', float('nan')),4)} | "
                f"{fmt(row.get('selected_alignment_mean', float('nan')),4)} |"
            )
    lines.extend(["", "## By Suite", "| suite | method | n | success |", "|---|---|---:|---:|"])
    suites = sorted({r.get("suite") for r in summary_suite})
    for suite in suites:
        for method in methods:
            row = next((r for r in summary_suite if r.get("suite") == suite and r.get("method") == method), None)
            if row:
                lines.append(f"| {suite} | {method} | {row['n']} | {fmt(row['success'])} |")
    generation_heads = sorted({str(r.get("generation_head_spec", "")) for r in manifest_rows if r.get("generation_head_spec")})
    rescore_heads = sorted({str(r.get("rescore_head_spec", "")) for r in manifest_rows if r.get("rescore_head_spec")})
    lines.extend(["", "## Required Questions", "- GenSG 是否超过 random-K：看 `gensg_last3` vs baseline=`k4_random`。", "- GenSG 是否超过 2.2 pure：看 `gensg_last3` vs baseline=`2.2_pure_last3_imc`。", "- GenSG 是否接近或超过 2.1：看 `gensg_last3` vs baseline=`2.1_rescore_best`。", "- q_j 和 W 是否必要：看 `gensg_last3` vs `gensg_last3_no_q` / `gensg_last3_no_W`。", "- last1 和 last3 哪个更好：比较 `gensg_last1` 与 `gensg_last3`。", f"- top head 是否稳定：本阶段 generation head 来自本次 stage1 rerun：`{','.join(generation_heads)}`；2.1 rescore head：`{','.join(rescore_heads)}`。", "- LIBERO-10 表现：见本次 run 的 `*_summary_by_suite.csv` 和 `*_summary_by_task.csv`。", "- latency 是否可接受：见 Query Diagnostics 和 summary CSV。"])
    if failed_jobs:
        lines.extend(["", "## Failed Status Files", "| job_id | status | returncode | episodes | suite | method | gpu | timestamp |", "|---|---|---:|---:|---|---|---:|---|"])
        for status in failed_jobs[:80]:
            lines.append(f"| {status.get('job_id')} | {status.get('status')} | {status.get('returncode', 'NA')} | {status.get('episodes', 'NA')} | {status.get('suite', 'NA')} | {status.get('method', 'NA')} | {status.get('gpu_id', 'NA')} | {status.get('timestamp', 'NA')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge Stage2 GenSG-IGC shards and write report.")
    parser.add_argument("--manifest", default="igc_gensg/configs/stage2_gensg_manifest.csv")
    parser.add_argument("--shard-dir", default="igc_gensg/results/stage2_gensg_shards")
    parser.add_argument("--job-log-dir", default="igc_gensg/logs/stage2_gensg_jobs")
    parser.add_argument("--episode-out", default="igc_gensg/results/stage2_gensg_episode_results.csv")
    parser.add_argument("--query-out", default="igc_gensg/results/stage2_gensg_query_logs.jsonl")
    parser.add_argument("--report", default="igc_gensg/reports/stage2_gensg_online_report.md")
    parser.add_argument("--no-dedup", action="store_true", help="Write raw concatenated shards without strict duplicate filtering.")
    args = parser.parse_args()
    manifest_rows = read_csv_rows(ROOT / args.manifest)
    allowed_job_ids = {r.get("job_id") for r in manifest_rows if r.get("job_id")}
    rows_raw = merge_csv_shards(ROOT / args.shard_dir, allowed_job_ids)
    if args.no_dedup:
        rows_out = rows_raw
        dedup_diag = {
            "raw_episode_rows": len(rows_raw),
            "final_episode_rows": len(rows_raw),
            "dedup_disabled": True,
        }
    else:
        rows_out, dedup_diag = dedupe_episode_rows(rows_raw, manifest_rows)
    write_csv(ROOT / args.episode_out, rows_out)
    rows = enrich(rows_out)
    records_raw = read_jsonl_shards(ROOT / args.job_log_dir, allowed_job_ids)
    if args.no_dedup:
        records = records_raw
        dedup_diag.update(
            {
                "raw_query_records": len(records_raw),
                "final_query_records": len(records_raw),
            }
        )
    else:
        records, query_dedup_diag = dedupe_query_records(records_raw, rows_out)
        dedup_diag.update(query_dedup_diag)
    write_jsonl(ROOT / args.query_out, records)
    summary_method = summary_rows(rows, ["method"])
    summary_suite = summary_rows(rows, ["suite", "method"])
    summary_task = summary_rows(rows, ["suite", "task_id", "method"])
    comparisons = compare_rows(rows)
    method_diag, selected_dist, score_var_dist = query_diagnostics(records)
    episode_out_path = ROOT / args.episode_out
    stem = episode_out_path.stem
    if stem.endswith("_episode_results_partial"):
        prefix = stem[: -len("_episode_results_partial")]
    elif stem.endswith("_episode_results"):
        prefix = stem[: -len("_episode_results")]
    elif "_episode_results" in stem:
        prefix = stem.split("_episode_results", 1)[0]
    else:
        prefix = "stage2_gensg"
    write_csv(ROOT / f"igc_gensg/results/{prefix}_summary_by_method.csv", summary_method)
    write_csv(ROOT / f"igc_gensg/results/{prefix}_summary_by_suite.csv", summary_suite)
    write_csv(ROOT / f"igc_gensg/results/{prefix}_summary_by_task.csv", summary_task)
    write_csv(ROOT / f"igc_gensg/results/{prefix}_vs_random_and_21.csv", comparisons)
    write_csv(ROOT / f"igc_gensg/results/{prefix}_query_method_diagnostics.csv", method_diag)
    write_csv(ROOT / f"igc_gensg/results/{prefix}_selected_index_distribution.csv", selected_dist)
    write_csv(ROOT / f"igc_gensg/results/{prefix}_score_variance_distribution.csv", score_var_dist)
    dedup_report = ROOT / f"igc_gensg/results/{prefix}_dedup_summary.json"
    dedup_report.write_text(json.dumps(dedup_diag, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    statuses = load_statuses(ROOT / args.job_log_dir / "status")
    if allowed_job_ids:
        statuses = [s for s in statuses if s.get("job_id") in allowed_job_ids]
    write_report(ROOT / args.report, rows, records, manifest_rows, statuses, summary_method, summary_suite, summary_task, comparisons, method_diag, dedup_diag)
    expected_eps = sum(int(r["episode_end"]) - int(r["episode_start"]) for r in manifest_rows) if manifest_rows else 0
    print(f"merged episodes={len(rows)}/{expected_eps} queries={len(records)} report={args.report} dedup={dedup_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
