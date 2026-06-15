from __future__ import annotations

import argparse
import csv
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[2]

DEFAULT_TASKS = {
    "libero_spatial": [0, 1, 4, 8],
    "libero_object": [0, 1, 6, 9],
    "libero_goal": [0, 1, 2, 5],
    "libero_10": [0, 1, 2, 5],
}
DEFAULT_METHODS = [
    "pi0_k1",
    "k4_first",
    "k4_random",
    "2.1_rescore_best",
    "2.2_pure_last1_imc",
    "2.2_pure_last3_imc",
    "gensg_last1",
    "gensg_last3",
    "gensg_last3_no_q",
    "gensg_last3_no_W",
]
HOST_GPUS = {
    "116.198.44.234": [0, 2, 3, 4, 5, 6, 7],
    "116.198.45.239": [0, 1, 2, 3, 4, 5, 6, 7],
    "116.198.46.225": [0, 1, 4, 6],
}


def parse_tasks(spec: str) -> dict[str, list[int]]:
    if not spec or spec == "default":
        return DEFAULT_TASKS
    out: dict[str, list[int]] = {}
    for suite_part in spec.split(";"):
        suite_part = suite_part.strip()
        if not suite_part:
            continue
        suite, ids = suite_part.split(":", 1)
        out[suite] = [int(x) for x in ids.split(",") if x.strip()]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Stage2 GenSG-IGC online rollout manifest.")
    parser.add_argument("--out", default="igc_gensg/configs/stage2_gensg_manifest.csv")
    parser.add_argument("--run-id", default="stage2_gensg_pilot")
    parser.add_argument("--episodes-per-task-method", type=int, default=30)
    parser.add_argument("--episodes-per-shard", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=42000)
    parser.add_argument("--tasks", default="default", help="default or 'suite:0,1;suite2:3,4'")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--generation-heads", default="17:0")
    parser.add_argument("--rescore-heads", default="17:0")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--execution-action-tokens", type=int, default=5)
    parser.add_argument("--hosts", default="all", help="Comma-separated host IPs from the built-in HOST_GPUS table, or all.")
    args = parser.parse_args()

    tasks = parse_tasks(args.tasks)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    host_filter = set(HOST_GPUS) if args.hosts == "all" else {x.strip() for x in args.hosts.split(",") if x.strip()}
    slots = [(host, gpu) for host, gpus in HOST_GPUS.items() if host in host_filter for gpu in gpus]
    if not slots:
        raise ValueError(f"No GPU slots selected by --hosts={args.hosts!r}")
    rows = []
    slot_i = 0
    for suite, task_ids in tasks.items():
        for task_id in task_ids:
            for method in methods:
                k = 1 if method == "pi0_k1" else int(args.k)
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
                            "generation_head_spec": args.generation_heads,
                            "rescore_head_spec": args.rescore_heads,
                            "execution_action_tokens": int(args.execution_action_tokens),
                        }
                    )
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} jobs to {out}")


if __name__ == "__main__":
    main()
