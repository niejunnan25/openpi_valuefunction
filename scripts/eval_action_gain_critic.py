"""Offline evaluation for an action-conditioned distributional gain critic."""

from __future__ import annotations

import argparse
import csv
import dataclasses
from pathlib import Path

import jax
import numpy as np
import safetensors.torch
import torch

import openpi.models.pi0_config as pi0_config
import openpi.models_pytorch.pi0_pytorch as pi0_pytorch
from openpi.models_pytorch.action_gain_critic import ActionGainCritic
from openpi.training import action_gain_data
from openpi.training import config as _config
from openpi.training.action_gain_utils import aggregate_gain_probs
from openpi.training.action_gain_utils import expected_value
from openpi.training.action_gain_utils import soft_cross_entropy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_name", help="openpi TrainConfig name")
    parser.add_argument("--labels-path", required=True)
    parser.add_argument("--critic-checkpoint", required=True)
    parser.add_argument("--pi0-checkpoint", default=None, help="PI0 checkpoint dir or model.safetensors path")
    parser.add_argument("--allow-random-pi0", action="store_true", help="Debug only: allow random PI0 features")
    parser.add_argument("--lerobot-root", default=None, help="Root containing local LeRobot datasets")
    parser.add_argument("--dataset-names", nargs="*", default=None, help="Dataset names or paths to load")
    parser.add_argument("--libero-suite", default=None, help="LIBERO suite shortcut: spatial|goal|object|libero_10|all")
    parser.add_argument("--assets-base-dir", default=None, help="Override TrainConfig assets_base_dir")
    parser.add_argument("--norm-stats-path", default=None, help="Path to norm_stats.json or its containing directory")
    parser.add_argument(
        "--no-filter-episodes-from-labels",
        action="store_true",
        help="Debug only: load full datasets instead of the episodes referenced by labels.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--timestep", type=float, default=1e-3)
    parser.add_argument("--eta", type=float, default=0.02)
    parser.add_argument("--rho-up", type=float, default=0.6)
    parser.add_argument("--rho-down", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-norm-stats", action="store_true")
    parser.add_argument("--predictions-csv", default=None)
    return parser.parse_args()


def _model_config_from_train_config(config: _config.TrainConfig) -> pi0_config.Pi0Config:
    if not isinstance(config.model, pi0_config.Pi0Config):
        raise TypeError("Action gain critic evaluation currently requires a Pi0Config model")
    return dataclasses.replace(
        config.model,
        dtype=config.pytorch_training_precision,
        pytorch_compile_mode=None,
    )


def _resolve_model_safetensors(path: str | None, config: _config.TrainConfig) -> Path | None:
    candidate = path or config.pytorch_weight_path
    if candidate is None:
        return None
    candidate_path = Path(candidate)
    if candidate_path.is_dir():
        candidate_path = candidate_path / "model.safetensors"
    if not candidate_path.is_file():
        raise FileNotFoundError(f"PI0 checkpoint not found: {candidate_path}")
    return candidate_path


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


def load_frozen_pi0(
    config: _config.TrainConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> pi0_pytorch.PI0Pytorch:
    model = pi0_pytorch.PI0Pytorch(_model_config_from_train_config(config)).to(device)
    checkpoint = _resolve_model_safetensors(args.pi0_checkpoint, config)
    if checkpoint is not None:
        safetensors.torch.load_model(model, checkpoint, device=str(device))
    elif not args.allow_random_pi0:
        raise ValueError(
            "A PI0 checkpoint is required. Pass --pi0-checkpoint or use --allow-random-pi0 for debug only."
        )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _kl_div(target: torch.Tensor, pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    target = target.clamp_min(eps)
    pred = pred.clamp_min(eps)
    return (target * (target.log() - pred.log())).sum(dim=-1)


def _js_div(target: torch.Tensor, pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mixture = 0.5 * (target + pred)
    return 0.5 * _kl_div(target, mixture, eps=eps) + 0.5 * _kl_div(pred, mixture, eps=eps)


def main() -> None:
    args = parse_args()
    config = _config.get_config(args.config_name)
    if args.assets_base_dir is not None:
        config = dataclasses.replace(config, assets_base_dir=args.assets_base_dir)
    device = torch.device(args.device)
    batch_size = config.batch_size if args.batch_size is None else args.batch_size
    dataset_paths = _resolve_dataset_paths(args)

    loader, _ = action_gain_data.create_action_gain_data_loader(
        config,
        args.labels_path,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=config.seed,
        dataset_paths=dataset_paths,
        skip_norm_stats=args.skip_norm_stats,
        norm_stats_path=args.norm_stats_path,
        filter_episodes_from_labels=not args.no_filter_episodes_from_labels,
    )
    for preview in loader.dataset.preview_identities(5):
        print(
            "label/sample key: "
            f"dataset={preview['dataset_key']} episode={preview['episode_index']} "
            f"frame={preview['frame_index']} next_frame={preview['next_frame_index']} "
            f"raw_index={preview['raw_index']} label_index={preview['label_index']}"
        )
    gain_atoms = torch.as_tensor(loader.dataset.gain_atoms, dtype=torch.float32, device=device)

    pi0 = load_frozen_pi0(config, args, device)
    checkpoint = torch.load(args.critic_checkpoint, map_location=device, weights_only=False)
    critic_config = checkpoint["critic_config"]
    critic = ActionGainCritic(**critic_config).to(device)
    critic.load_state_dict(checkpoint["critic_state_dict"])
    critic.eval()

    rows = []
    metrics = {
        "ce": [],
        "kl": [],
        "js": [],
        "gain_abs_error": [],
        "pred_p_up": [],
        "target_p_up": [],
        "pred_p_down": [],
        "target_p_down": [],
        "accept": [],
        "label_positive": [],
    }

    with torch.no_grad():
        for batch_idx, (observation, actions, gain_target_probs, metadata) in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            observation = jax.tree.map(lambda x: x.to(device), observation)
            actions = actions.to(device=device, dtype=torch.float32)
            gain_target_probs = gain_target_probs.to(device=device, dtype=torch.float32)

            prefix_context = pi0.encode_prefix_context(observation)
            features = pi0.encode_actions_joint(
                observation,
                actions,
                timestep=args.timestep,
                prefix_context=prefix_context,
            )
            logits = critic(features["action_hidden"])
            pred_probs = torch.softmax(logits.float(), dim=-1)

            pred_gain = expected_value(pred_probs, gain_atoms)
            target_gain = expected_value(gain_target_probs, gain_atoms)
            pred_aggr = aggregate_gain_probs(pred_probs, gain_atoms, args.eta)
            target_aggr = aggregate_gain_probs(gain_target_probs, gain_atoms, args.eta)
            accept = (pred_aggr["p_up"] > args.rho_up) & (pred_aggr["p_down"] < args.rho_down)
            label_positive = target_gain > args.eta

            metrics["ce"].append(soft_cross_entropy(logits, gain_target_probs, reduction="none").cpu())
            metrics["kl"].append(_kl_div(gain_target_probs, pred_probs).cpu())
            metrics["js"].append(_js_div(gain_target_probs, pred_probs).cpu())
            metrics["gain_abs_error"].append((pred_gain - target_gain).abs().cpu())
            for key in ("p_up", "p_down"):
                metrics[f"pred_{key}"].append(pred_aggr[key].cpu())
                metrics[f"target_{key}"].append(target_aggr[key].cpu())
            metrics["accept"].append(accept.cpu())
            metrics["label_positive"].append(label_positive.cpu())

            if args.predictions_csv is not None:
                batch_size_actual = actions.shape[0]
                for i in range(batch_size_actual):
                    rows.append(
                        {
                            "dataset_key": metadata["dataset_key"][i],
                            "episode_index": metadata["episode_index"][i],
                            "frame_index": int(metadata["frame_index"][i]),
                            "next_frame_index": int(metadata["next_frame_index"][i]),
                            "raw_index": int(metadata["raw_index"][i]),
                            "target_gain_mean": float(target_gain[i].cpu()),
                            "pred_gain_mean": float(pred_gain[i].cpu()),
                            "target_p_up": float(target_aggr["p_up"][i].cpu()),
                            "pred_p_up": float(pred_aggr["p_up"][i].cpu()),
                            "target_p_down": float(target_aggr["p_down"][i].cpu()),
                            "pred_p_down": float(pred_aggr["p_down"][i].cpu()),
                            "gate_accept": bool(accept[i].cpu()),
                        }
                    )

    reduced = {key: torch.cat(values) for key, values in metrics.items()}
    accepted = reduced["accept"]
    positives = reduced["label_positive"]
    true_accepts = accepted & positives
    precision = true_accepts.sum().float() / accepted.sum().clamp_min(1)
    recall = true_accepts.sum().float() / positives.sum().clamp_min(1)

    print("Action gain critic evaluation")
    print(f"  samples: {reduced['ce'].numel()}")
    print(f"  CE: {reduced['ce'].mean().item():.6f}")
    print(f"  KL(target||pred): {reduced['kl'].mean().item():.6f}")
    print(f"  JS: {reduced['js'].mean().item():.6f}")
    print(f"  expected gain MAE: {reduced['gain_abs_error'].mean().item():.6f}")
    print(
        f"  pred p_up / target p_up: "
        f"{reduced['pred_p_up'].mean().item():.4f} / {reduced['target_p_up'].mean().item():.4f}"
    )
    print(
        f"  pred p_down / target p_down: "
        f"{reduced['pred_p_down'].mean().item():.4f} / {reduced['target_p_down'].mean().item():.4f}"
    )
    print(f"  gate accept rate: {accepted.float().mean().item():.4f}")
    print(f"  gate precision / recall against target_gain>eta: {precision.item():.4f} / {recall.item():.4f}")

    if args.predictions_csv is not None:
        output = Path(args.predictions_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved per-sample predictions to {output}")


if __name__ == "__main__":
    main()
