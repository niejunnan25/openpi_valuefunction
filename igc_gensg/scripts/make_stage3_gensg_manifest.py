from __future__ import annotations

import argparse
import csv
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[2]

HOST_GPUS = {
    "116.198.44.234": [0, 2, 3, 4, 5, 6, 7],
    "116.198.45.239": [0, 1, 2, 3, 4, 5, 6, 7],
    "116.198.46.225": [0, 1, 4, 6],
}

DEFAULT_CONTROL_METHODS = [
    "k4_random",
    "2.1_rescore_best",
    "2.2_pure_last3_imc",
    "gensg_last3",
    "gensg_early",
    "gensg_middle",
    "gensg_all_steps",
    "gensg_last3_no_q",
    "gensg_last3_no_W",
    "gensg_last3_random_token_map",
    "gensg_last3_shuffled_token_map",
    "gensg_last3_random_prefix_head",
    "gensg_last3_random_action_head",
    "gensg_last3_bottom_action_head",
    "gensg_last3_score_mismatch",
    "gensg_last3_prefix_only",
]


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_tasks(spec: str) -> list[tuple[str, int]]:
    tasks = []
    for suite_part in spec.split(";"):
        suite_part = suite_part.strip()
        if not suite_part:
            continue
        suite, ids = suite_part.split(":", 1)
        for task_id in ids.split(","):
            task_id = task_id.strip()
            if task_id:
                tasks.append((suite, int(task_id)))
    return tasks


def select_stage2_positive_tasks(summary_path: pathlib.Path, min_n: int, max_tasks: int) -> list[tuple[str, int, float]]:
    rows = read_csv(summary_path)
    by_task: dict[tuple[str, int], dict[str, dict[str, str]]] = {}
    for row in rows:
        try:
            key = (str(row["suite"]), int(row["task_id"]))
        except Exception:
            continue
        by_task.setdefault(key, {})[str(row.get("method"))] = row
    selected = []
    for (suite, task_id), methods in by_task.items():
        main = methods.get("gensg_last3")
        base = methods.get("k4_random")
        if not main or not base:
            continue
        try:
            n_main = int(float(main.get("n", 0) or 0))
            n_base = int(float(base.get("n", 0) or 0))
            diff = float(main.get("success", 0) or 0) - float(base.get("success", 0) or 0)
        except Exception:
            continue
        if n_main >= min_n and n_base >= min_n and diff > 0:
            selected.append((suite, task_id, diff))
    selected.sort(key=lambda x: x[2], reverse=True)
    return selected[:max_tasks]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create Stage3 GenSG control manifest gated by Stage2 task-level results.")
    parser.add_argument("--out", default="igc_gensg/configs/stage3_gensg_controls_manifest.csv")
    parser.add_argument("--run-id", default="stage3_gensg_controls")
    parser.add_argument("--stage2-task-summary", default="igc_gensg/results/stage2_gensg_summary_by_task.csv")
    parser.add_argument("--episodes-per-task-method", type=int, default=20)
    parser.add_argument("--episodes-per-shard", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=62000)
    parser.add_argument("--min-stage2-n", type=int, default=20)
    parser.add_argument("--max-tasks", type=int, default=8)
    parser.add_argument("--tasks", default="", help="Manual override: 'suite:0,1;suite2:3'. Bypasses Stage2 positive-task gate.")
    parser.add_argument("--methods", default=",".join(DEFAULT_CONTROL_METHODS))
    parser.add_argument("--generation-heads", default="17:0")
    parser.add_argument("--rescore-heads", default="17:0")
    parser.add_argument("--random-action-head", default="0:0")
    parser.add_argument("--bottom-action-head", default="0:0")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--hosts", default="116.198.44.234,116.198.45.239")
    args = parser.parse_args()

    if args.tasks.strip():
        tasks = [(suite, task_id, 0.0) for suite, task_id in parse_tasks(args.tasks)]
        gate_reason = "manual task override"
    else:
        tasks = select_stage2_positive_tasks(ROOT / args.stage2_task_summary, int(args.min_stage2_n), int(args.max_tasks))
        gate_reason = f"selected from {args.stage2_task_summary}"
    if not tasks:
        print(
            "No Stage2-positive tasks found. Refusing to create large Stage3 controls manifest; "
            "rerun after Stage2 completes or pass --tasks for an explicit smoke test."
        )
        return 2

    host_filter = {x.strip() for x in args.hosts.split(",") if x.strip()}
    slots = [(host, gpu) for host, gpus in HOST_GPUS.items() if host in host_filter for gpu in gpus]
    if not slots:
        raise ValueError(f"No GPU slots selected by --hosts={args.hosts!r}")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    rows = []
    slot_i = 0
    for suite, task_id, stage2_diff in tasks:
        for method in methods:
            k = 1 if method == "pi0_k1" else int(args.k)
            gen_heads = args.generation_heads
            if method == "gensg_last3_random_action_head":
                gen_heads = args.random_action_head
            elif method == "gensg_last3_bottom_action_head":
                gen_heads = args.bottom_action_head
            for start in range(0, int(args.episodes_per_task_method), int(args.episodes_per_shard)):
                end = min(start + int(args.episodes_per_shard), int(args.episodes_per_task_method))
                host, gpu = slots[slot_i % len(slots)]
                slot_i += 1
                rows.append(
                    {
                        "job_id": f"{suite}_t{task_id:02d}_{method.replace('.', 'p')}_e{start:03d}_{end:03d}",
                        "run_id": args.run_id,
                        "hostname": host,
                        "gpu_id": gpu,
                        "suite": suite,
                        "task_id": task_id,
                        "method": method,
                        "episode_start": start,
                        "episode_end": end,
                        "seed_start": int(args.seed_start) + task_id * 1000 + start,
                        "k": k,
                        "generation_head_spec": gen_heads,
                        "rescore_head_spec": args.rescore_heads,
                        "stage2_gensg_vs_random_diff": stage2_diff,
                        "selection_gate": gate_reason,
                    }
                )
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} Stage3 jobs for {len(tasks)} tasks to {out}")
    for suite, task_id, diff in tasks:
        print(f"task {suite}:{task_id} stage2_diff={diff:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
