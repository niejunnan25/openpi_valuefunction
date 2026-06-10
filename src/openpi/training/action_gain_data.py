"""LeRobot adapter for action-gain critic training."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Sequence
import dataclasses
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

import openpi.models.model as _model
from openpi.shared import normalize as _normalize
import openpi.training.config as _config
import openpi.transforms as _transforms

logger = logging.getLogger(__name__)


LIBERO_DATASET_NAMES = {
    "spatial": ("libero_spatial_pi0_20260530_lerobot",),
    "goal": ("libero_goal_pi0_20260530_lerobot",),
    "object": ("libero_object_pi0_20260530_lerobot",),
    "libero_10": ("libero_10_pi0_20260603_merged_lerobot",),
    "libero10": ("libero_10_pi0_20260603_merged_lerobot",),
    "10": ("libero_10_pi0_20260603_merged_lerobot",),
    "all": (
        "libero_spatial_pi0_20260530_lerobot",
        "libero_goal_pi0_20260530_lerobot",
        "libero_object_pi0_20260530_lerobot",
        "libero_10_pi0_20260603_merged_lerobot",
    ),
    "libero_all": (
        "libero_spatial_pi0_20260530_lerobot",
        "libero_goal_pi0_20260530_lerobot",
        "libero_object_pi0_20260530_lerobot",
        "libero_10_pi0_20260603_merged_lerobot",
    ),
}


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


def _episode_key(value: Any) -> str:
    value = _scalar(value)
    return str(int(value)) if isinstance(value, int | np.integer) else str(value)


def _frame_key(value: Any) -> int:
    return int(_scalar(value))


def _dataset_key(value: Any) -> str:
    value = _scalar(value)
    text = str(value)
    return Path(text).name if "/" in text else text


def _raw_sequence_key(key: str, dataset_meta: Any) -> str:
    # Local LIBERO parquet files use "action", while the OpenPI transform output
    # key is "actions". LeRobot delta_timestamps must reference raw dataset keys.
    features = getattr(dataset_meta, "features", {}) or {}
    if key in features:
        return key
    if key == "actions" and "action" in features:
        return "action"
    return key


def _with_local_lerobot_aliases(sample: dict[str, Any]) -> dict[str, Any]:
    sample = dict(sample)
    aliases = {
        "actions": "action",
        "state": "observation.state",
        "image": "observation.image",
        "wrist_image": "observation.wrist_image",
        "prompt": "task",
    }
    for target_key, source_key in aliases.items():
        if target_key not in sample and source_key in sample:
            sample[target_key] = sample[source_key]
    return sample


def _pick_array(npz: np.lib.npyio.NpzFile, names: Iterable[str]) -> np.ndarray:
    for name in names:
        if name in npz:
            return np.asarray(npz[name])
    raise KeyError(f"Missing any of the required arrays: {tuple(names)}")


@dataclasses.dataclass(frozen=True)
class ActionGainLabelStore:
    """NPZ-backed label lookup keyed by (dataset_key, episode_index, frame_index)."""

    dataset_key: np.ndarray
    episode_index: np.ndarray
    frame_index: np.ndarray
    gain_target_probs: np.ndarray
    gain_atoms: np.ndarray
    next_frame_index: np.ndarray | None = None
    target_gain_mean: np.ndarray | None = None
    raw_index: np.ndarray | None = None

    @classmethod
    def load(cls, path: str) -> "ActionGainLabelStore":
        with np.load(path, allow_pickle=True) as npz:
            dataset_key = _pick_array(npz, ("dataset_key",))
            episode_index = _pick_array(npz, ("episode_index", "episode_id"))
            frame_index = _pick_array(npz, ("frame_index", "frame_id"))
            gain_target_probs = np.asarray(npz["gain_target_probs"], dtype=np.float32)
            gain_atoms = np.asarray(npz["gain_atoms"], dtype=np.float32)
            next_frame_index = np.asarray(npz["next_frame_index"]) if "next_frame_index" in npz else None
            target_gain_mean = (
                np.asarray(npz["target_gain_mean"], dtype=np.float32) if "target_gain_mean" in npz else None
            )
            raw_index = np.asarray(npz["raw_index"], dtype=np.int64) if "raw_index" in npz else None

        if (
            dataset_key.shape[0] != episode_index.shape[0]
            or dataset_key.shape[0] != frame_index.shape[0]
            or dataset_key.shape[0] != gain_target_probs.shape[0]
            or episode_index.shape[0] != frame_index.shape[0]
            or episode_index.shape[0] != gain_target_probs.shape[0]
        ):
            raise ValueError(
                "dataset_key, episode_index, frame_index, and gain_target_probs must have the same first dimension"
            )
        return cls(
            dataset_key=dataset_key,
            episode_index=episode_index,
            frame_index=frame_index,
            gain_target_probs=gain_target_probs,
            gain_atoms=gain_atoms,
            next_frame_index=next_frame_index,
            target_gain_mean=target_gain_mean,
            raw_index=raw_index,
        )

    def __len__(self) -> int:
        return self.gain_target_probs.shape[0]

    def key_at(self, index: int) -> tuple[str, str, int]:
        return (
            _dataset_key(self.dataset_key[index]),
            _episode_key(self.episode_index[index]),
            _frame_key(self.frame_index[index]),
        )

    def as_key_to_row(self) -> dict[tuple[str, str, int], int]:
        key_to_row = {}
        for i in range(len(self)):
            key = self.key_at(i)
            if key in key_to_row:
                raise ValueError(f"Duplicate gain label key found: {key}")
            key_to_row[key] = i
        return key_to_row

    def dataset_keys(self) -> set[str]:
        return {_dataset_key(key) for key in self.dataset_key}

    def numeric_episodes_by_dataset(self) -> dict[str, list[int]]:
        episodes: dict[str, set[int]] = {}
        for i in range(len(self)):
            dataset_key = _dataset_key(self.dataset_key[i])
            episode = _episode_key(self.episode_index[i])
            if not episode.isdigit():
                raise ValueError(
                    "Episode-filtered loading requires numeric episode indices, "
                    f"got {episode!r} for dataset {dataset_key}"
                )
            episodes.setdefault(dataset_key, set()).add(int(episode))
        return {dataset_key: sorted(values) for dataset_key, values in episodes.items()}


def _find_column(source: Any, names: tuple[str, ...]) -> np.ndarray | None:
    sources = [source]
    for attr in ("hf_dataset", "dataset", "_dataset", "data"):
        if hasattr(source, attr):
            sources.append(getattr(source, attr))

    for candidate in sources:
        column_names = getattr(candidate, "column_names", None)
        for name in names:
            if column_names is not None and name not in column_names:
                continue
            try:
                return np.asarray(candidate[name])
            except Exception:  # noqa: PERF203
                continue
    return None


def _extract_episode_frame(sample: dict[str, Any]) -> tuple[str, int]:
    episode = None
    for key in ("episode_index", "episode_id", "trajectory_id", "traj_id"):
        if key in sample:
            episode = sample[key]
            break
    if episode is None:
        raise KeyError("Could not find episode metadata in sample")

    frame = None
    for key in ("frame_index", "frame_id", "step", "base_index"):
        if key in sample:
            frame = sample[key]
            break
    if frame is None:
        raise KeyError("Could not find frame metadata in sample")
    return (_episode_key(episode), _frame_key(frame))


def _extract_identity(sample: dict[str, Any], dataset_key: str) -> tuple[str, str, int]:
    episode, frame = _extract_episode_frame(sample)
    return (dataset_key, episode, frame)


def _build_key_to_raw_index(raw_dataset: Any, dataset_key: str) -> dict[tuple[str, str, int], int]:
    episodes = _find_column(raw_dataset, ("episode_index", "episode_id", "trajectory_id", "traj_id"))
    frames = _find_column(raw_dataset, ("frame_index", "frame_id", "step", "base_index"))
    key_to_index = {}
    if episodes is not None and frames is not None:
        for i, (ep, fr) in enumerate(zip(episodes, frames, strict=True)):
            key = (dataset_key, _episode_key(ep), _frame_key(fr))
            if key in key_to_index:
                raise ValueError(f"Duplicate LeRobot sample key found: {key}")
            key_to_index[key] = i
        return key_to_index

    logger.warning("Could not read LeRobot metadata columns directly; falling back to per-sample metadata scan.")
    for i in range(len(raw_dataset)):
        key = _extract_identity(raw_dataset[i], dataset_key)
        if key in key_to_index:
            raise ValueError(f"Duplicate LeRobot sample key found: {key}")
        key_to_index[key] = i
    return key_to_index


def libero_dataset_names_for_suite(suite: str) -> tuple[str, ...]:
    suite = suite.lower()
    try:
        return LIBERO_DATASET_NAMES[suite]
    except KeyError as exc:
        raise ValueError(f"Unknown LIBERO suite {suite!r}. Expected one of {sorted(LIBERO_DATASET_NAMES)}") from exc


def resolve_lerobot_dataset_paths(
    *,
    lerobot_root: str | None = None,
    dataset_names: Sequence[str] | None = None,
    libero_suite: str | None = None,
) -> list[str] | None:
    if libero_suite is not None:
        if dataset_names:
            raise ValueError("Pass either dataset_names or libero_suite, not both")
        dataset_names = libero_dataset_names_for_suite(libero_suite)
    if not dataset_names:
        return None
    if lerobot_root is None:
        return [str(name) for name in dataset_names]
    return [str(Path(lerobot_root) / name) for name in dataset_names]


class ActionGainLeRobotDataset(torch.utils.data.Dataset):
    """LeRobot dataset joined with precomputed gain-distribution labels."""

    def __init__(
        self,
        data_config: _config.DataConfig,
        model_config: _model.BaseModelConfig,
        labels_path: str,
        *,
        dataset_paths: Sequence[str] | None = None,
        skip_norm_stats: bool = False,
        filter_episodes_from_labels: bool = True,
    ) -> None:
        if dataset_paths is None and data_config.repo_id is None:
            raise ValueError("ActionGainLeRobotDataset requires dataset_paths or a LeRobot repo_id")
        try:
            import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
        except ImportError as exc:
            raise ImportError("lerobot is required for ActionGainLeRobotDataset") from exc

        dataset_paths = tuple(dataset_paths or (data_config.repo_id,))
        self._labels = ActionGainLabelStore.load(labels_path)
        self.gain_atoms = self._labels.gain_atoms
        self._raw_datasets = []
        self._transforms = []
        self._dataset_keys = []
        dataset_key_to_idx = {}
        episodes_by_dataset = self._labels.numeric_episodes_by_dataset() if filter_episodes_from_labels else None

        norm_stats = {}
        if data_config.repo_id != "fake" and not skip_norm_stats:
            if data_config.norm_stats is None:
                raise ValueError(
                    "Normalization stats not found. "
                    "Run scripts/compute_norm_stats.py or pass --skip-norm-stats for debugging."
                )
            norm_stats = data_config.norm_stats

        key_to_source = {}
        should_build_key_map = True
        if episodes_by_dataset is None and self._labels.raw_index is not None and np.all(self._labels.raw_index >= 0):
            should_build_key_map = False
        for dataset_path in dataset_paths:
            dataset_key = _dataset_key(dataset_path)
            if episodes_by_dataset is not None and dataset_key not in episodes_by_dataset:
                logger.info("Skipping dataset %s because no gain labels reference it", dataset_key)
                continue
            if dataset_key in self._dataset_keys:
                raise ValueError(f"Duplicate dataset key from dataset_paths: {dataset_key}")
            dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(dataset_path)
            episodes = episodes_by_dataset.get(dataset_key) if episodes_by_dataset is not None else None
            raw_dataset = lerobot_dataset.LeRobotDataset(
                dataset_path,
                episodes=episodes,
                delta_timestamps={
                    _raw_sequence_key(key, dataset_meta): [
                        t / dataset_meta.fps for t in range(model_config.action_horizon)
                    ]
                    for key in data_config.action_sequence_keys
                },
            )
            transforms: list[_transforms.DataTransformFn] = []
            if data_config.prompt_from_task:
                transforms.append(_transforms.PromptFromLeRobotTask(dataset_meta.tasks))
            transforms.extend(data_config.repack_transforms.inputs)
            transforms.extend(data_config.data_transforms.inputs)
            transforms.append(_transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm))
            transforms.extend(data_config.model_transforms.inputs)

            dataset_idx = len(self._raw_datasets)
            self._raw_datasets.append(raw_dataset)
            self._transforms.append(_transforms.compose(transforms))
            self._dataset_keys.append(dataset_key)
            dataset_key_to_idx[dataset_key] = dataset_idx

            # Prefer key maps for episode-filtered datasets. The LeRobot row indices
            # are re-based after filtering, so label raw_index values are not used.
            if should_build_key_map:
                for key, raw_index in _build_key_to_raw_index(raw_dataset, dataset_key).items():
                    if key in key_to_source:
                        raise ValueError(f"Duplicate LeRobot key across datasets: {key}")
                    key_to_source[key] = (dataset_idx, raw_index)

        missing_dataset_keys = self._labels.dataset_keys() - set(self._dataset_keys)
        if missing_dataset_keys:
            raise ValueError(
                "Gain labels reference dataset_key values that were not loaded: "
                f"{sorted(missing_dataset_keys)}. Loaded: {sorted(self._dataset_keys)}"
            )

        label_key_to_row = self._labels.as_key_to_row()
        if not should_build_key_map:
            self._sample_sources = []
            for key, label_row in label_key_to_row.items():
                dataset_idx = dataset_key_to_idx[key[0]]
                raw_index = int(self._labels.raw_index[label_row])
                if raw_index >= len(self._raw_datasets[dataset_idx]):
                    raise IndexError(
                        f"raw_index {raw_index} for label key {key} exceeds dataset length "
                        f"{len(self._raw_datasets[dataset_idx])}"
                    )
                self._sample_sources.append((dataset_idx, raw_index, label_row))
        else:
            common_keys = [key for key in label_key_to_row if key in key_to_source]
            missing = len(label_key_to_row) - len(common_keys)
            if missing:
                logger.warning("Skipping %d gain labels that are not present in the loaded LeRobot datasets", missing)

            self._sample_sources = [
                (*key_to_source[key], label_key_to_row[key])
                for key in common_keys
            ]

        if not self._sample_sources:
            raise ValueError("No gain labels matched the LeRobot dataset metadata")

    def __len__(self) -> int:
        return len(self._sample_sources)

    def preview_identities(self, count: int = 5) -> list[dict[str, Any]]:
        previews = []
        for dataset_idx, raw_index, label_row in self._sample_sources[:count]:
            previews.append(
                {
                    "dataset_key": self._dataset_keys[dataset_idx],
                    "episode_index": _episode_key(self._labels.episode_index[label_row]),
                    "frame_index": _frame_key(self._labels.frame_index[label_row]),
                    "next_frame_index": (
                        int(self._labels.next_frame_index[label_row])
                        if self._labels.next_frame_index is not None
                        else -1
                    ),
                    "raw_index": int(raw_index),
                    "label_index": int(label_row),
                }
            )
        return previews

    def __getitem__(self, index: int) -> dict[str, Any]:
        dataset_idx, raw_index, label_row = self._sample_sources[index]
        raw_sample = _with_local_lerobot_aliases(self._raw_datasets[dataset_idx][raw_index])
        dataset_key = self._dataset_keys[dataset_idx]
        episode_index, frame_index = _extract_episode_frame(raw_sample)
        expected_key = self._labels.key_at(label_row)
        actual_key = (dataset_key, episode_index, frame_index)
        if actual_key != expected_key:
            raise ValueError(f"Label/sample key mismatch: sample={actual_key}, label={expected_key}")
        data = self._transforms[dataset_idx](raw_sample)
        return {
            "data": data,
            "gain_target_probs": self._labels.gain_target_probs[label_row],
            "dataset_key": dataset_key,
            "episode_index": episode_index,
            "frame_index": frame_index,
            "next_frame_index": (
                int(self._labels.next_frame_index[label_row]) if self._labels.next_frame_index is not None else -1
            ),
            "raw_index": raw_index,
            "label_index": label_row,
        }


def collate_action_gain_batch(
    items: list[dict[str, Any]],
) -> tuple[_model.Observation, torch.Tensor, torch.Tensor, dict]:
    import jax

    data_items = [item["data"] for item in items]
    batch = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *data_items)
    batch = jax.tree.map(torch.as_tensor, batch)
    observation = _model.Observation.from_dict(batch)
    actions = batch["actions"].to(dtype=torch.float32)
    gain_target_probs = torch.as_tensor(
        np.stack([item["gain_target_probs"] for item in items], axis=0),
        dtype=torch.float32,
    )
    metadata = {
        "dataset_key": [item["dataset_key"] for item in items],
        "episode_index": [item["episode_index"] for item in items],
        "frame_index": torch.as_tensor([item["frame_index"] for item in items], dtype=torch.long),
        "next_frame_index": torch.as_tensor([item["next_frame_index"] for item in items], dtype=torch.long),
        "raw_index": torch.as_tensor([item["raw_index"] for item in items], dtype=torch.long),
        "label_index": torch.as_tensor([item["label_index"] for item in items], dtype=torch.long),
    }
    return observation, actions, gain_target_probs, metadata


def create_action_gain_data_loader(
    config: _config.TrainConfig,
    labels_path: str,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    dataset_paths: Sequence[str] | None = None,
    skip_norm_stats: bool = False,
    norm_stats_path: str | None = None,
    filter_episodes_from_labels: bool = True,
) -> tuple[torch.utils.data.DataLoader, _config.DataConfig]:
    data_config = config.data.create(config.assets_dirs, config.model)
    if norm_stats_path is not None:
        norm_stats_dir = Path(norm_stats_path)
        if norm_stats_dir.is_file():
            norm_stats_dir = norm_stats_dir.parent
        data_config = dataclasses.replace(data_config, norm_stats=_normalize.load(norm_stats_dir))
    dataset = ActionGainLeRobotDataset(
        data_config=data_config,
        model_config=config.model,
        labels_path=labels_path,
        dataset_paths=dataset_paths,
        skip_norm_stats=skip_norm_stats,
        filter_episodes_from_labels=filter_episodes_from_labels,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_action_gain_batch,
        drop_last=shuffle,
        generator=generator,
    )
    return loader, data_config
