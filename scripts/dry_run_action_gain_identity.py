"""Dry-run identity alignment for action-gain critic LeRobot rollouts.

This script intentionally reads parquet metadata directly instead of constructing
full LeRobotDataset objects. It is meant to catch dataset_key/episode/frame
alignment bugs quickly on large LIBERO mixtures.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openpi.training.action_gain_data import resolve_lerobot_dataset_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerobot-root", required=True, help="Root containing local LeRobot datasets")
    parser.add_argument("--dataset-names", nargs="*", default=None, help="Dataset names or paths to check")
    parser.add_argument("--libero-suite", default="all", help="LIBERO suite shortcut")
    parser.add_argument("--horizon", type=int, default=5, help="K in t -> t+K")
    parser.add_argument("--max-episodes-per-dataset", type=int, default=50)
    parser.add_argument("--labels-path", default=None, help="Optional gain label NPZ to validate")
    parser.add_argument("--max-labels", type=int, default=10000)
    return parser.parse_args()


def _dataset_key(path: str | Path) -> str:
    return Path(path).name


def _episode_key(value) -> str:
    value = value.item() if hasattr(value, "item") else value
    return str(int(value)) if isinstance(value, int | np.integer) else str(value)


def _frame_key(value) -> int:
    value = value.item() if hasattr(value, "item") else value
    return int(value)


def _episode_files(dataset_path: Path) -> list[Path]:
    return sorted((dataset_path / "data").glob("chunk-*/*.parquet"))


def _action_dim(action_scalar) -> int:
    action = action_scalar.as_py()
    return len(action) if hasattr(action, "__len__") else -1


def _check_dataset(
    dataset_path: Path,
    *,
    horizon: int,
    max_episodes: int,
    full_keys: set[tuple[str, str, int]],
    bare_counts: Counter[tuple[str, int]],
) -> dict:
    dataset_key = _dataset_key(dataset_path)
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing {info_path}")
    info = json.loads(info_path.read_text())
    files = _episode_files(dataset_path)
    if not files:
        raise FileNotFoundError(f"No parquet files found under {dataset_path / 'data'}")

    duplicate_full_keys = 0
    nonconsecutive_inside_checked = 0
    nonunit_frame_steps = 0
    t_plus_k_pairs = 0
    action_dims: set[int] = set()
    checked_rows = 0

    for file_path in files[:max_episodes]:
        table = pq.read_table(file_path, columns=["episode_index", "frame_index", "action"])
        episodes = table["episode_index"].to_numpy(zero_copy_only=False)
        frames = table["frame_index"].to_numpy(zero_copy_only=False)
        frame_values = [_frame_key(frame) for frame in frames]
        sorted_frames = sorted(frame_values)
        if any(b - a != 1 for a, b in zip(sorted_frames, sorted_frames[1:], strict=False)):
            nonunit_frame_steps += 1

        frame_set = set(sorted_frames)
        max_frame = max(frame_set)
        for row, (episode, frame) in enumerate(zip(episodes, frames, strict=True)):
            episode = _episode_key(episode)
            frame = _frame_key(frame)
            full_key = (dataset_key, episode, frame)
            if full_key in full_keys:
                duplicate_full_keys += 1
            full_keys.add(full_key)
            bare_counts[(episode, frame)] += 1
            checked_rows += 1
            action_dims.add(_action_dim(table["action"][row]))
            if frame + horizon in frame_set:
                t_plus_k_pairs += 1
            elif frame <= max_frame - horizon:
                nonconsecutive_inside_checked += 1

    return {
        "dataset_key": dataset_key,
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "fps": info.get("fps"),
        "parquet_files": len(files),
        "checked_files": min(max_episodes, len(files)),
        "checked_rows": checked_rows,
        "t_plus_k_pairs": t_plus_k_pairs,
        "nonconsecutive_inside_checked": nonconsecutive_inside_checked,
        "nonunit_frame_step_episodes": nonunit_frame_steps,
        "action_dims": sorted(action_dims),
        "duplicate_full_keys": duplicate_full_keys,
    }


def _validate_labels(labels_path: str, dataset_paths: list[Path], *, max_labels: int) -> dict:
    dataset_to_files = {
        _dataset_key(path): {
            int(file_path.stem.removeprefix("episode_")): file_path
            for file_path in _episode_files(path)
        }
        for path in dataset_paths
    }
    checked = 0
    missing = []
    with np.load(labels_path, allow_pickle=True) as npz:
        dataset_key = np.asarray(npz["dataset_key"])
        episode_index = np.asarray(npz["episode_index"])
        frame_index = np.asarray(npz["frame_index"])
        next_frame_index = np.asarray(npz["next_frame_index"]) if "next_frame_index" in npz else None
        total = len(dataset_key)
        for row in range(min(total, max_labels)):
            ds = _dataset_key(str(dataset_key[row]))
            ep_text = _episode_key(episode_index[row])
            if not ep_text.isdigit():
                raise ValueError(f"Non-numeric episode index in label row {row}: {ep_text}")
            ep = int(ep_text)
            frame = _frame_key(frame_index[row])
            file_path = dataset_to_files.get(ds, {}).get(ep)
            if file_path is None:
                missing.append((ds, ep, frame, "episode_file"))
                continue
            table = pq.read_table(file_path, columns=["frame_index"])
            frames = set(int(x) for x in table["frame_index"].to_numpy(zero_copy_only=False).tolist())
            if frame not in frames:
                missing.append((ds, ep, frame, "frame"))
                continue
            if next_frame_index is not None and _frame_key(next_frame_index[row]) not in frames:
                missing.append((ds, ep, _frame_key(next_frame_index[row]), "next_frame"))
                continue
            checked += 1

    return {
        "labels_total": total,
        "labels_checked": checked,
        "labels_missing": missing[:20],
        "labels_missing_count": len(missing),
    }


def main() -> None:
    args = parse_args()
    dataset_paths = resolve_lerobot_dataset_paths(
        lerobot_root=args.lerobot_root,
        dataset_names=args.dataset_names,
        libero_suite=args.libero_suite,
    )
    if dataset_paths is None:
        raise ValueError("No datasets resolved. Pass --dataset-names or --libero-suite.")
    dataset_paths = [Path(path) for path in dataset_paths]

    print("ACTION_GAIN_IDENTITY_DRY_RUN")
    print("datasets", [path.name for path in dataset_paths])
    print("horizon", args.horizon)
    print("max_episodes_per_dataset", args.max_episodes_per_dataset)

    full_keys: set[tuple[str, str, int]] = set()
    bare_counts: Counter[tuple[str, int]] = Counter()
    summaries = [
        _check_dataset(
            path,
            horizon=args.horizon,
            max_episodes=args.max_episodes_per_dataset,
            full_keys=full_keys,
            bare_counts=bare_counts,
        )
        for path in dataset_paths
    ]

    for summary in summaries:
        print(
            "dataset={dataset_key} total_episodes={total_episodes} total_frames={total_frames} "
            "fps={fps} parquet_files={parquet_files} checked_files={checked_files} "
            "checked_rows={checked_rows} t_plus_k_pairs={t_plus_k_pairs} "
            "nonconsecutive_inside_checked={nonconsecutive_inside_checked} "
            "nonunit_frame_step_episodes={nonunit_frame_step_episodes} action_dims={action_dims} "
            "duplicate_full_keys={duplicate_full_keys}".format(**summary)
        )

    bare_collisions = sum(1 for count in bare_counts.values() if count > 1)
    max_bare_collision = max(bare_counts.values()) if bare_counts else 0
    print("bare_episode_frame_collisions", bare_collisions, "max_count", max_bare_collision)
    print("full_keys_checked", len(full_keys))

    failures = []
    for summary in summaries:
        if summary["duplicate_full_keys"]:
            failures.append(f"{summary['dataset_key']}: duplicate full keys")
        if summary["nonconsecutive_inside_checked"]:
            failures.append(f"{summary['dataset_key']}: missing t+K inside checked episodes")
        if summary["nonunit_frame_step_episodes"]:
            failures.append(f"{summary['dataset_key']}: non-unit frame steps")
        if summary["action_dims"] != [7]:
            failures.append(f"{summary['dataset_key']}: unexpected action dims {summary['action_dims']}")
    if max_bare_collision < 2 and len(dataset_paths) > 1:
        failures.append("expected bare episode/frame collisions across multi-dataset mix")

    if args.labels_path is not None:
        label_result = _validate_labels(args.labels_path, dataset_paths, max_labels=args.max_labels)
        print(
            "labels_total={labels_total} labels_checked={labels_checked} "
            "labels_missing_count={labels_missing_count}".format(**label_result)
        )
        if label_result["labels_missing"]:
            print("labels_missing_examples", label_result["labels_missing"])
            failures.append("label keys missing from parquet metadata")

    if failures:
        print("identity_dry_run_failed")
        for failure in failures:
            print("failure", failure)
        raise SystemExit(1)
    print("identity_dry_run_ok")


if __name__ == "__main__":
    main()
