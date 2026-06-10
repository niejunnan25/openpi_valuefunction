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


def _extract_episode_frame(example: dict[str, Any]) -> tuple[Any, int]:
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
    return episode, frame


def main() -> None:
    args = parse_args()
    model = _load_joyra_model(args)
    loader = _build_joyra_loader(args)

    episode_index = []
    frame_index = []
    value_logits = []
    value_probs = []
    value_mean = []
    success = []
    exported = 0

    with torch.inference_mode():
        for batch in loader:
            if args.max_samples is not None and exported >= args.max_samples:
                break
            if args.max_samples is not None:
                batch = batch[: max(0, args.max_samples - exported)]
            result = model.predict_value(examples=batch, bin_min=args.value_min, bin_max=args.value_max)
            logits = np.asarray(result["logits"], dtype=np.float32)
            probs = np.asarray(result["probs"], dtype=np.float32)
            values = np.asarray(result.get("values"), dtype=np.float32)

            for example in batch:
                ep, fr = _extract_episode_frame(example)
                episode_index.append(ep)
                frame_index.append(fr)
                success.append(bool(example.get("success", False)))

            value_logits.append(logits)
            value_probs.append(probs)
            value_mean.append(values)
            exported += len(batch)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atoms = np.linspace(args.value_min, args.value_max, args.value_num_bins, dtype=np.float32)
    np.savez_compressed(
        output,
        episode_index=np.asarray(episode_index),
        frame_index=np.asarray(frame_index, dtype=np.int64),
        value_logits=np.concatenate(value_logits, axis=0).astype(np.float32),
        value_probs=np.concatenate(value_probs, axis=0).astype(np.float32),
        value_mean=np.concatenate(value_mean, axis=0).astype(np.float32),
        value_atoms=atoms,
        success=np.asarray(success, dtype=bool),
    )
    print(f"Exported {exported} value distributions to {output}")


if __name__ == "__main__":
    main()
