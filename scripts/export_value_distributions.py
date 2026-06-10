"""Export JoyRA distributional state-value predictions for rollout frames.

The output NPZ is the teacher sidecar consumed by build_gain_distribution_labels.py.
JoyRA is imported at runtime so openpi can still be tested without JoyRA installed
in the active Python environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


def _default_joyra_root() -> Path:
    return Path(__file__).resolve().parents[2] / "JoyRA-RL"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joyra-root", default=str(_default_joyra_root()))
    parser.add_argument("--checkpoint-path", required=True, help="JoyRA value checkpoint")
    parser.add_argument("--config-yaml", required=True, help="JoyRA value config YAML")
    parser.add_argument("--output", required=True, help="Output value-distribution NPZ")
    parser.add_argument("--framework-name", default=None, help="Optional JoyRA framework override")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--data-root-dir", required=True, help="JoyRA/LeRobot data root")
    parser.add_argument("--data-mix", required=True, help="JoyRA DATASET_NAMED_MIXTURES key")
    parser.add_argument("--mode", choices=("train", "val"), default="train")
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--skip-invalid-subtask-frames", action="store_true")
    parser.add_argument("--language-prefix", default=None)

    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--big-negative", type=float, default=100.0)
    parser.add_argument("--success-col", default="episode_success")
    parser.add_argument("--returns-cache-dir", default=None)
    parser.add_argument("--value-num-bins", type=int, default=51)
    parser.add_argument("--value-min", type=float, default=-1.0)
    parser.add_argument("--value-max", type=float, default=0.0)
    parser.add_argument("--bin-range-json", default=None)
    parser.add_argument("--normalize-returns", action="store_true")
    parser.add_argument("--normalize-returns-per-task", action="store_true")
    parser.add_argument("--normalize-use-big-negative-in-denom", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--episodes-per-dataset",
        type=int,
        default=None,
        help=(
            "Export complete, consecutive episodes per dataset instead of the shuffled dataloader prefix. "
            "Use this for small K-step label smoke tests."
        ),
    )
    parser.add_argument("--episode-start", type=int, default=0, help="First numeric episode id for subset export")
    parser.add_argument(
        "--allow-dataset-key-fallback",
        action="store_true",
        help="Allow using --data-mix as dataset_key when JoyRA examples do not expose dataset identity.",
    )
    return parser.parse_args()


def _load_joyra_model(args: argparse.Namespace):
    joyra_root = Path(args.joyra_root).resolve()
    sys.path.insert(0, str(joyra_root))
    try:
        from omegaconf import OmegaConf
        from starVLA.training.train_value import build_value_model
    except ImportError as exc:
        raise ImportError(f"Could not import JoyRA value code from {joyra_root}") from exc

    cfg = OmegaConf.load(args.config_yaml)
    model = build_value_model(cfg, framework_name=args.framework_name)
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[export] missing keys: {missing[:20]}{'...' if len(missing) > 20 else ''}")
    if unexpected:
        print(f"[export] unexpected keys: {unexpected[:20]}{'...' if len(unexpected) > 20 else ''}")
    model.to(args.device)
    model.eval()
    return model


def _build_joyra_loader(args: argparse.Namespace):
    joyra_root = Path(args.joyra_root).resolve()
    sys.path.insert(0, str(joyra_root))
    try:
        from starVLA.training.train_value import build_value_dataloader
    except ImportError as exc:
        raise ImportError(f"Could not import JoyRA dataloader from {joyra_root}") from exc

    loader_args = argparse.Namespace(
        data_root_dir=args.data_root_dir,
        data_mix=args.data_mix,
        frame_stride=args.frame_stride,
        skip_invalid_subtask_frames=args.skip_invalid_subtask_frames,
        language_prefix=args.language_prefix,
        gamma=args.gamma,
        big_negative=args.big_negative,
        success_col=args.success_col,
        num_bins=args.value_num_bins,
        bin_range_json=args.bin_range_json,
        bin_min=args.value_min,
        bin_max=args.value_max,
        normalize_returns=args.normalize_returns,
        normalize_returns_per_task=args.normalize_returns_per_task,
        normalize_use_big_negative_in_denom=args.normalize_use_big_negative_in_denom,
        returns_cache_dir=args.returns_cache_dir,
        train_split=args.train_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
    )
    loader, _ = build_value_dataloader(loader_args, distributed=False, mode=args.mode, seed=42)
    return loader


def _scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        if value.size == 1:
            return value.reshape(()).item()
    if hasattr(value, "item"):
        return value.item()
    return value


def _dataset_key(value: Any) -> str:
    value = _scalar(value)
    text = str(value)
    return Path(text).name if "/" in text else text


def _extract_identity(
    example: dict[str, Any],
    default_dataset_key: str | None,
) -> tuple[str, Any, int, int]:
    dataset = None
    for key in ("dataset_key", "dataset_path", "repo_id"):
        if key in example:
            dataset = example[key]
            break
    if dataset is None:
        if default_dataset_key is None:
            raise KeyError(
                "JoyRA examples must expose dataset_key/dataset_path/repo_id. "
                "Pass --allow-dataset-key-fallback only for known single-dataset exports."
            )
        dataset = default_dataset_key

    episode = None
    for key in ("episode_index", "episode_id", "trajectory_id", "traj_id"):
        if key in example:
            episode = _scalar(example[key])
            break
    frame = None
    for key in ("frame_index", "frame_id", "step", "base_index"):
        if key in example:
            frame = int(_scalar(example[key]))
            break
    if episode is None or frame is None:
        raise KeyError(
            "JoyRA examples must expose episode/frame metadata. "
            "Expected episode_index or trajectory_id, and frame_index or step."
        )
    raw_index = -1
    for key in ("raw_index", "dataset_index"):
        if key in example:
            raw_index = int(_scalar(example[key]))
            break
    return _dataset_key(dataset), episode, frame, raw_index


def _predict_and_append(
    *,
    model,
    examples: list[dict[str, Any]],
    args: argparse.Namespace,
    default_dataset_key: str | None,
    output_lists: dict[str, list],
) -> int:
    if not examples:
        return 0
    result = model.predict_value(examples=examples, bin_min=args.value_min, bin_max=args.value_max)
    logits = np.asarray(result["logits"], dtype=np.float32)
    probs = np.asarray(result["probs"], dtype=np.float32)
    values = np.asarray(result.get("values"), dtype=np.float32)

    for example in examples:
        ds, ep, fr, raw = _extract_identity(example, default_dataset_key)
        output_lists["dataset_key"].append(ds)
        output_lists["episode_index"].append(ep)
        output_lists["frame_index"].append(fr)
        output_lists["raw_index"].append(raw)
        output_lists["success"].append(bool(example.get("success", False)))

    output_lists["value_logits"].append(logits)
    output_lists["value_probs"].append(probs)
    output_lists["value_mean"].append(values)
    return len(examples)


def _dataset_key_from_base_dataset(dataset: Any) -> str:
    for attr in ("dataset_path", "repo_id", "dataset_name"):
        if hasattr(dataset, attr):
            return _dataset_key(getattr(dataset, attr))
    return _dataset_key(str(dataset))


def _select_contiguous_actual_indices(wrapper_dataset: Any, episodes_per_dataset: int, episode_start: int) -> list[int]:
    """Return dataset indices ordered by dataset, episode, frame for complete episodes."""
    if not hasattr(wrapper_dataset, "mixture") or not hasattr(wrapper_dataset.mixture, "sampled_steps"):
        raise TypeError("Contiguous episode export requires JoyRA LeRobotMixtureWithValueTarget")

    base_datasets = list(wrapper_dataset.mixture.datasets)
    dataset_keys = [_dataset_key_from_base_dataset(dataset) for dataset in base_datasets]
    selected_episodes: dict[str, set[int]] = {}
    ordered_episode_ids: dict[str, list[int]] = {}
    for dataset_key, dataset in zip(dataset_keys, base_datasets, strict=True):
        episode_ids = sorted(int(ep) for ep in dataset.trajectory_ids if int(ep) >= episode_start)
        chosen = episode_ids[:episodes_per_dataset]
        if len(chosen) < episodes_per_dataset:
            raise ValueError(
                f"Dataset {dataset_key} has only {len(chosen)} episodes >= {episode_start}, "
                f"requested {episodes_per_dataset}"
            )
        selected_episodes[dataset_key] = set(chosen)
        ordered_episode_ids[dataset_key] = chosen

    key_to_actual_index: dict[tuple[str, int, int], int] = {}
    for actual_index, (dataset_idx, trajectory_id, step) in enumerate(wrapper_dataset.mixture.sampled_steps):
        dataset_key = dataset_keys[int(dataset_idx)]
        trajectory_id = int(trajectory_id)
        if trajectory_id not in selected_episodes[dataset_key]:
            continue
        key = (dataset_key, trajectory_id, int(step))
        if key in key_to_actual_index:
            raise ValueError(f"Duplicate JoyRA sampled step key found: {key}")
        key_to_actual_index[key] = actual_index

    ordered_indices = []
    for dataset_key, dataset in zip(dataset_keys, base_datasets, strict=True):
        length_by_episode = {
            int(ep): int(length)
            for ep, length in zip(dataset.trajectory_ids, dataset.trajectory_lengths, strict=True)
        }
        for episode_id in ordered_episode_ids[dataset_key]:
            for step in range(length_by_episode[episode_id]):
                key = (dataset_key, episode_id, step)
                actual_index = key_to_actual_index.get(key)
                if actual_index is not None:
                    ordered_indices.append(actual_index)

    if not ordered_indices:
        raise ValueError("No JoyRA sampled steps matched the requested episode subset")
    return ordered_indices


def _get_actual_item(wrapper_dataset: Any, actual_index: int) -> dict[str, Any]:
    """Fetch an item by actual sampled_steps index, bypassing train/val split offsets."""
    original_mode = getattr(wrapper_dataset, "mode", None)
    try:
        if original_mode != "train":
            wrapper_dataset.mode = "train"
        return wrapper_dataset[actual_index]
    finally:
        if original_mode is not None:
            wrapper_dataset.mode = original_mode


def main() -> None:
    args = parse_args()
    if args.episodes_per_dataset is not None and args.episodes_per_dataset <= 0:
        raise ValueError("--episodes-per-dataset must be positive")
    if args.episodes_per_dataset is not None and args.max_samples is not None:
        raise ValueError("Use either --episodes-per-dataset or --max-samples, not both")

    model = _load_joyra_model(args)
    loader = _build_joyra_loader(args)

    output_lists = {
        "episode_index": [],
        "frame_index": [],
        "dataset_key": [],
        "raw_index": [],
        "value_logits": [],
        "value_probs": [],
        "value_mean": [],
        "success": [],
    }
    exported = 0
    default_dataset_key = args.data_mix if args.allow_dataset_key_fallback else None

    with torch.inference_mode():
        if args.episodes_per_dataset is not None:
            actual_indices = _select_contiguous_actual_indices(
                loader.dataset,
                episodes_per_dataset=args.episodes_per_dataset,
                episode_start=args.episode_start,
            )
            pending = []
            for actual_index in actual_indices:
                pending.append(_get_actual_item(loader.dataset, actual_index))
                if len(pending) >= args.batch_size:
                    exported += _predict_and_append(
                        model=model,
                        examples=pending,
                        args=args,
                        default_dataset_key=default_dataset_key,
                        output_lists=output_lists,
                    )
                    pending = []
            exported += _predict_and_append(
                model=model,
                examples=pending,
                args=args,
                default_dataset_key=default_dataset_key,
                output_lists=output_lists,
            )
        else:
            for batch in loader:
                if args.max_samples is not None and exported >= args.max_samples:
                    break
                if args.max_samples is not None:
                    batch = batch[: max(0, args.max_samples - exported)]
                exported += _predict_and_append(
                    model=model,
                    examples=batch,
                    args=args,
                    default_dataset_key=default_dataset_key,
                    output_lists=output_lists,
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atoms = np.linspace(args.value_min, args.value_max, args.value_num_bins, dtype=np.float32)
    if exported == 0:
        raise ValueError("No value distributions were exported")
    np.savez_compressed(
        output,
        dataset_key=np.asarray(output_lists["dataset_key"]),
        episode_index=np.asarray(output_lists["episode_index"]),
        frame_index=np.asarray(output_lists["frame_index"], dtype=np.int64),
        raw_index=np.asarray(output_lists["raw_index"], dtype=np.int64),
        value_logits=np.concatenate(output_lists["value_logits"], axis=0).astype(np.float32),
        value_probs=np.concatenate(output_lists["value_probs"], axis=0).astype(np.float32),
        value_mean=np.concatenate(output_lists["value_mean"], axis=0).astype(np.float32),
        value_atoms=atoms,
        success=np.asarray(output_lists["success"], dtype=bool),
        data_mix=np.asarray(args.data_mix),
        data_root_dir=np.asarray(args.data_root_dir),
        episodes_per_dataset=np.asarray(
            -1 if args.episodes_per_dataset is None else args.episodes_per_dataset,
            dtype=np.int64,
        ),
        episode_start=np.asarray(args.episode_start, dtype=np.int64),
    )
    print(f"Exported {exported} value distributions to {output}")


if __name__ == "__main__":
    main()
