"""LeRobot adapter for action-gain critic training."""

from __future__ import annotations

from collections.abc import Iterable
import dataclasses
import logging
from typing import Any

import jax
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.transforms as _transforms

logger = logging.getLogger(__name__)


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


def _pick_array(npz: np.lib.npyio.NpzFile, names: Iterable[str]) -> np.ndarray:
    for name in names:
        if name in npz:
            return np.asarray(npz[name])
    raise KeyError(f"Missing any of the required arrays: {tuple(names)}")


@dataclasses.dataclass(frozen=True)
class ActionGainLabelStore:
    """NPZ-backed label lookup keyed by (episode_index, frame_index)."""

    episode_index: np.ndarray
    frame_index: np.ndarray
    gain_target_probs: np.ndarray
    gain_atoms: np.ndarray
    next_frame_index: np.ndarray | None = None
    target_gain_mean: np.ndarray | None = None
    dataset_index: np.ndarray | None = None

    @classmethod
    def load(cls, path: str) -> "ActionGainLabelStore":
        with np.load(path, allow_pickle=True) as npz:
            episode_index = _pick_array(npz, ("episode_index", "episode_id"))
            frame_index = _pick_array(npz, ("frame_index", "frame_id"))
            gain_target_probs = np.asarray(npz["gain_target_probs"], dtype=np.float32)
            gain_atoms = np.asarray(npz["gain_atoms"], dtype=np.float32)
            next_frame_index = np.asarray(npz["next_frame_index"]) if "next_frame_index" in npz else None
            target_gain_mean = (
                np.asarray(npz["target_gain_mean"], dtype=np.float32) if "target_gain_mean" in npz else None
            )
            dataset_index = np.asarray(npz["dataset_index"], dtype=np.int64) if "dataset_index" in npz else None

        if (
            episode_index.shape[0] != frame_index.shape[0]
            or episode_index.shape[0] != gain_target_probs.shape[0]
        ):
            raise ValueError("episode_index, frame_index, and gain_target_probs must have the same first dimension")
        return cls(
            episode_index=episode_index,
            frame_index=frame_index,
            gain_target_probs=gain_target_probs,
            gain_atoms=gain_atoms,
            next_frame_index=next_frame_index,
            target_gain_mean=target_gain_mean,
            dataset_index=dataset_index,
        )

    def __len__(self) -> int:
        return self.gain_target_probs.shape[0]

    def key_at(self, index: int) -> tuple[str, int]:
        return (_episode_key(self.episode_index[index]), _frame_key(self.frame_index[index]))

    def as_key_to_row(self) -> dict[tuple[str, int], int]:
        return {self.key_at(i): i for i in range(len(self))}


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


def _build_key_to_raw_index(raw_dataset: Any) -> dict[tuple[str, int], int]:
    episodes = _find_column(raw_dataset, ("episode_index", "episode_id", "trajectory_id", "traj_id"))
    frames = _find_column(raw_dataset, ("frame_index", "frame_id", "step", "base_index"))
    if episodes is not None and frames is not None:
        return {(_episode_key(ep), _frame_key(fr)): i for i, (ep, fr) in enumerate(zip(episodes, frames, strict=True))}

    logger.warning("Could not read LeRobot metadata columns directly; falling back to per-sample metadata scan.")
    key_to_index = {}
    for i in range(len(raw_dataset)):
        key_to_index[_extract_episode_frame(raw_dataset[i])] = i
    return key_to_index


class ActionGainLeRobotDataset(torch.utils.data.Dataset):
    """LeRobot dataset joined with precomputed gain-distribution labels."""

    def __init__(
        self,
        data_config: _config.DataConfig,
        model_config: _model.BaseModelConfig,
        labels_path: str,
        *,
        skip_norm_stats: bool = False,
    ) -> None:
        if data_config.repo_id is None:
            raise ValueError("ActionGainLeRobotDataset requires a LeRobot repo_id")
        try:
            import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
        except ImportError as exc:
            raise ImportError("lerobot is required for ActionGainLeRobotDataset") from exc

        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
        self._raw_dataset = lerobot_dataset.LeRobotDataset(
            data_config.repo_id,
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(model_config.action_horizon)]
                for key in data_config.action_sequence_keys
            },
        )
        self._labels = ActionGainLabelStore.load(labels_path)
        self.gain_atoms = self._labels.gain_atoms

        norm_stats = {}
        if data_config.repo_id != "fake" and not skip_norm_stats:
            if data_config.norm_stats is None:
                raise ValueError(
                    "Normalization stats not found. "
                    "Run scripts/compute_norm_stats.py or pass --skip-norm-stats for debugging."
                )
            norm_stats = data_config.norm_stats

        transforms: list[_transforms.DataTransformFn] = []
        if data_config.prompt_from_task:
            transforms.append(_transforms.PromptFromLeRobotTask(dataset_meta.tasks))
        transforms.extend(data_config.repack_transforms.inputs)
        transforms.extend(data_config.data_transforms.inputs)
        transforms.append(_transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm))
        transforms.extend(data_config.model_transforms.inputs)
        self._transform = _transforms.compose(transforms)

        if self._labels.dataset_index is not None:
            self._raw_indices = [int(i) for i in self._labels.dataset_index]
            self._label_rows = list(range(len(self._labels)))
        else:
            key_to_raw_index = _build_key_to_raw_index(self._raw_dataset)
            label_key_to_row = self._labels.as_key_to_row()
            common_keys = [key for key in label_key_to_row if key in key_to_raw_index]
            missing = len(label_key_to_row) - len(common_keys)
            if missing:
                logger.warning("Skipping %d gain labels that are not present in the LeRobot dataset", missing)
            self._raw_indices = [key_to_raw_index[key] for key in common_keys]
            self._label_rows = [label_key_to_row[key] for key in common_keys]

        if not self._raw_indices:
            raise ValueError("No gain labels matched the LeRobot dataset metadata")

    def __len__(self) -> int:
        return len(self._raw_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw_index = self._raw_indices[index]
        label_row = self._label_rows[index]
        raw_sample = self._raw_dataset[raw_index]
        episode_index, frame_index = _extract_episode_frame(raw_sample)
        data = self._transform(raw_sample)
        return {
            "data": data,
            "gain_target_probs": self._labels.gain_target_probs[label_row],
            "episode_index": episode_index,
            "frame_index": frame_index,
            "label_index": label_row,
        }


def collate_action_gain_batch(
    items: list[dict[str, Any]],
) -> tuple[_model.Observation, torch.Tensor, torch.Tensor, dict]:
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
        "episode_index": [item["episode_index"] for item in items],
        "frame_index": torch.as_tensor([item["frame_index"] for item in items], dtype=torch.long),
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
    skip_norm_stats: bool = False,
) -> tuple[torch.utils.data.DataLoader, _config.DataConfig]:
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = ActionGainLeRobotDataset(
        data_config=data_config,
        model_config=config.model,
        labels_path=labels_path,
        skip_norm_stats=skip_norm_stats,
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
