import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def _run_label_builder(input_path: Path, output_path: Path, *extra_args: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"src{os.pathsep}{env.get('PYTHONPATH', '')}"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_gain_distribution_labels.py",
            "--value-distributions",
            str(input_path),
            "--output",
            str(output_path),
            "--horizon",
            "5",
            *extra_args,
        ],
        check=True,
        env=env,
    )


def test_build_gain_labels_uses_dataset_key_for_duplicate_episode_frames(tmp_path: Path):
    value_atoms = np.linspace(-1.0, 0.0, 51, dtype=np.float32)
    dataset_key = []
    episode_index = []
    frame_index = []
    probs = []

    for ds, curr_bin, next_bin in [("libero_spatial", 20, 30), ("libero_goal", 30, 20)]:
        for frame in range(6):
            p = np.zeros(51, dtype=np.float32)
            p[curr_bin if frame == 0 else next_bin] = 1.0
            dataset_key.append(ds)
            episode_index.append(0)
            frame_index.append(frame)
            probs.append(p)

    input_path = tmp_path / "values.npz"
    output_path = tmp_path / "labels.npz"
    np.savez_compressed(
        input_path,
        dataset_key=np.asarray(dataset_key),
        episode_index=np.asarray(episode_index, dtype=np.int64),
        frame_index=np.asarray(frame_index, dtype=np.int64),
        value_probs=np.stack(probs, axis=0),
        value_atoms=value_atoms,
    )

    _run_label_builder(input_path, output_path)

    with np.load(output_path, allow_pickle=True) as data:
        assert data["gain_target_probs"].shape == (2, 101)
        key_to_argmax = {
            str(ds): int(target.argmax())
            for ds, target in zip(data["dataset_key"], data["gain_target_probs"], strict=True)
        }
        assert key_to_argmax["libero_spatial"] == 60
        assert key_to_argmax["libero_goal"] == 40


def test_build_gain_labels_skips_nonconsecutive_frames_by_default(tmp_path: Path):
    value_atoms = np.linspace(-1.0, 0.0, 51, dtype=np.float32)
    probs = np.zeros((6, 51), dtype=np.float32)
    probs[:, 20] = 1.0
    input_path = tmp_path / "values.npz"
    output_path = tmp_path / "labels.npz"
    np.savez_compressed(
        input_path,
        dataset_key=np.asarray(["libero_spatial"] * 6),
        episode_index=np.zeros(6, dtype=np.int64),
        frame_index=np.asarray([0, 2, 4, 6, 8, 10], dtype=np.int64),
        value_probs=probs,
        value_atoms=value_atoms,
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = f"src{os.pathsep}{env.get('PYTHONPATH', '')}"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_gain_distribution_labels.py",
            "--value-distributions",
            str(input_path),
            "--output",
            str(output_path),
            "--horizon",
            "5",
        ],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "No t/t+K pairs found" in result.stderr
