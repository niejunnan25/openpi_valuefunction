"""Build action-gain distribution labels from exported state-value distributions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from openpi.training.action_gain_utils import aggregate_gain_probs
from openpi.training.action_gain_utils import distribution_difference
from openpi.training.action_gain_utils import expected_value
from openpi.training.action_gain_utils import make_atoms


def _pick_array(npz: np.lib.npyio.NpzFile, *names: str) -> np.ndarray:
    for name in names:
        if name in npz:
            return np.asarray(npz[name])
    raise KeyError(f"Missing any of the required arrays: {names}")


def _episode_key(value) -> str:
    value = value.item() if hasattr(value, "item") else value
    return str(int(value)) if isinstance(value, int | np.integer) else str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--value-distributions", required=True, help="Input NPZ from export_value_distributions.py")
    parser.add_argument("--output", required=True, help="Output NPZ with gain_target_probs")
    parser.add_argument("--horizon", type=int, default=5, help="K in DeltaV^K; default 5")
    parser.add_argument("--value-min", type=float, default=-1.0)
    parser.add_argument("--value-max", type=float, default=0.0)
    parser.add_argument("--value-num-bins", type=int, default=51)
    parser.add_argument("--gain-min", type=float, default=-1.0)
    parser.add_argument("--gain-max", type=float, default=1.0)
    parser.add_argument("--gain-num-bins", type=int, default=101)
    parser.add_argument("--eta", type=float, default=0.02, help="Flat/up/down aggregation threshold")
    parser.add_argument("--batch-size", type=int, default=8192, help="CPU label projection batch size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.value_distributions, allow_pickle=True) as npz:
        episode_index = _pick_array(npz, "episode_index", "episode_id")
        frame_index = _pick_array(npz, "frame_index", "frame_id").astype(np.int64)
        value_probs = np.asarray(npz["value_probs"], dtype=np.float32)
        value_atoms_np = (
            np.asarray(npz["value_atoms"], dtype=np.float32)
            if "value_atoms" in npz
            else np.linspace(args.value_min, args.value_max, args.value_num_bins, dtype=np.float32)
        )
        dataset_index = np.asarray(npz["dataset_index"], dtype=np.int64) if "dataset_index" in npz else None

    if value_probs.ndim != 2:
        raise ValueError(f"value_probs must have shape [N, M], got {value_probs.shape}")
    if value_probs.shape[1] != len(value_atoms_np):
        raise ValueError(f"value_probs has {value_probs.shape[1]} bins but value_atoms has {len(value_atoms_np)}")

    key_to_row = {
        (_episode_key(ep), int(fr)): i for i, (ep, fr) in enumerate(zip(episode_index, frame_index, strict=True))
    }
    curr_rows: list[int] = []
    next_rows: list[int] = []
    for i, (ep, fr) in enumerate(zip(episode_index, frame_index, strict=True)):
        next_row = key_to_row.get((_episode_key(ep), int(fr) + args.horizon))
        if next_row is None:
            continue
        curr_rows.append(i)
        next_rows.append(next_row)

    if not curr_rows:
        raise ValueError(f"No t/t+K pairs found for horizon={args.horizon}")

    value_atoms = torch.as_tensor(value_atoms_np, dtype=torch.float32)
    gain_atoms = make_atoms(args.gain_min, args.gain_max, args.gain_num_bins)

    targets = []
    for start in range(0, len(curr_rows), args.batch_size):
        curr_idx = curr_rows[start : start + args.batch_size]
        next_idx = next_rows[start : start + args.batch_size]
        p_curr = torch.as_tensor(value_probs[curr_idx], dtype=torch.float32)
        p_next = torch.as_tensor(value_probs[next_idx], dtype=torch.float32)
        targets.append(distribution_difference(p_curr, p_next, value_atoms, gain_atoms).cpu())

    gain_target_probs = torch.cat(targets, dim=0)
    target_gain_mean = expected_value(gain_target_probs, gain_atoms).cpu().numpy().astype(np.float32)
    aggregates = aggregate_gain_probs(gain_target_probs, gain_atoms, args.eta)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {
        "episode_index": episode_index[curr_rows],
        "frame_index": frame_index[curr_rows],
        "next_frame_index": frame_index[next_rows],
        "gain_target_probs": gain_target_probs.cpu().numpy().astype(np.float32),
        "target_gain_mean": target_gain_mean,
        "label_p_up": aggregates["p_up"].cpu().numpy().astype(np.float32),
        "label_p_flat": aggregates["p_flat"].cpu().numpy().astype(np.float32),
        "label_p_down": aggregates["p_down"].cpu().numpy().astype(np.float32),
        "value_atoms": value_atoms_np.astype(np.float32),
        "gain_atoms": gain_atoms.cpu().numpy().astype(np.float32),
        "horizon": np.asarray(args.horizon, dtype=np.int64),
        "eta": np.asarray(args.eta, dtype=np.float32),
    }
    if dataset_index is not None:
        save_kwargs["dataset_index"] = dataset_index[curr_rows]
    np.savez_compressed(output, **save_kwargs)
    print(f"Saved {len(curr_rows)} gain labels to {output}")


if __name__ == "__main__":
    main()
