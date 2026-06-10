"""Smoke-test the action-gain critic dataloader on real LeRobot labels."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

from openpi.training import action_gain_data
from openpi.training import config as _config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_name", help="openpi TrainConfig name")
    parser.add_argument("--labels-path", required=True, help="Gain label NPZ")
    parser.add_argument("--lerobot-root", default=None, help="Root containing local LeRobot datasets")
    parser.add_argument("--dataset-names", nargs="*", default=None, help="Dataset names or paths to load")
    parser.add_argument("--libero-suite", default=None, help="LIBERO suite shortcut: spatial|goal|object|libero_10|all")
    parser.add_argument("--assets-base-dir", default=None, help="Override TrainConfig assets_base_dir")
    parser.add_argument("--norm-stats-path", default=None, help="Path to norm_stats.json or its containing directory")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-norm-stats", action="store_true", help="Debug only")
    parser.add_argument(
        "--no-filter-episodes-from-labels",
        action="store_true",
        help="Debug only: load full datasets instead of the episodes referenced by labels.",
    )
    return parser.parse_args()


def _resolve_dataset_paths(args: argparse.Namespace) -> list[str] | None:
    dataset_paths = action_gain_data.resolve_lerobot_dataset_paths(
        lerobot_root=args.lerobot_root,
        dataset_names=args.dataset_names,
        libero_suite=args.libero_suite,
    )
    if dataset_paths is None and args.lerobot_root is not None:
        with np.load(args.labels_path, allow_pickle=True) as npz:
            if "dataset_key" not in npz:
                raise KeyError("labels_path must contain dataset_key when inferring datasets from --lerobot-root")
            dataset_keys = sorted({Path(str(key)).name for key in npz["dataset_key"]})
        dataset_paths = [str(Path(args.lerobot_root) / key) for key in dataset_keys]
    return dataset_paths


def _shape(value) -> tuple[int, ...] | str:
    shape = getattr(value, "shape", None)
    return tuple(shape) if shape is not None else type(value).__name__


def main() -> None:
    args = parse_args()
    config = _config.get_config(args.config_name)
    dataset_paths = _resolve_dataset_paths(args)
    if args.assets_base_dir is not None:
        config = dataclasses.replace(config, assets_base_dir=args.assets_base_dir)

    loader, data_config = action_gain_data.create_action_gain_data_loader(
        config,
        args.labels_path,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
        dataset_paths=dataset_paths,
        skip_norm_stats=args.skip_norm_stats,
        norm_stats_path=args.norm_stats_path,
        filter_episodes_from_labels=not args.no_filter_episodes_from_labels,
    )

    print("ACTION_GAIN_DATALOADER_SMOKE")
    print("dataset_len", len(loader.dataset))
    print("batch_size", args.batch_size)
    print("dataset_paths", dataset_paths)
    print("norm_stats_loaded", data_config.norm_stats is not None)
    for preview in loader.dataset.preview_identities(5):
        print("preview", preview)

    observation, actions, gain_target_probs, metadata = next(iter(loader))
    print("actions_shape", tuple(actions.shape))
    print("gain_target_probs_shape", tuple(gain_target_probs.shape))
    print("gain_target_sum_minmax", float(gain_target_probs.sum(-1).min()), float(gain_target_probs.sum(-1).max()))
    print("metadata", metadata)
    print("observation_state_shape", _shape(observation.state))
    print("observation_images", {key: _shape(value) for key, value in observation.images.items()})
    print("smoke_dataloader_ok")


if __name__ == "__main__":
    main()
