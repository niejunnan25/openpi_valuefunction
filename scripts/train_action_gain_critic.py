"""Train an action-conditioned distributional gain critic on PI0 action hidden states."""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path
import time

import jax
import numpy as np
import safetensors.torch
import torch
import tqdm

import openpi.models.pi0_config as pi0_config
import openpi.models_pytorch.pi0_pytorch as pi0_pytorch
from openpi.models_pytorch.action_gain_critic import ActionGainCritic
from openpi.training import action_gain_data
from openpi.training import config as _config
from openpi.training.action_gain_utils import aggregate_gain_probs
from openpi.training.action_gain_utils import expected_value
from openpi.training.action_gain_utils import soft_cross_entropy


def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_name", help="openpi TrainConfig name")
    parser.add_argument("--labels-path", required=True, help="Gain label NPZ from build_gain_distribution_labels.py")
    parser.add_argument("--output-dir", required=True, help="Directory for critic checkpoints")
    parser.add_argument("--pi0-checkpoint", default=None, help="PI0 checkpoint dir or model.safetensors path")
    parser.add_argument("--allow-random-pi0", action="store_true", help="Debug only: allow random PI0 features")
    parser.add_argument("--lerobot-root", default=None, help="Root containing local LeRobot datasets")
    parser.add_argument("--dataset-names", nargs="*", default=None, help="Dataset names or paths to load")
    parser.add_argument("--libero-suite", default=None, help="LIBERO suite shortcut: spatial|goal|object|libero_10|all")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-train-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--timestep", type=float, default=1e-3)
    parser.add_argument("--eta", type=float, default=0.02)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-norm-stats", action="store_true")
    parser.add_argument("--resume", default=None, help="Optional critic checkpoint to resume")
    return parser.parse_args()


def _model_config_from_train_config(config: _config.TrainConfig) -> pi0_config.Pi0Config:
    if not isinstance(config.model, pi0_config.Pi0Config):
        raise TypeError("Action gain critic training currently requires a Pi0Config model")
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
        logging.info("Loading frozen PI0 weights from %s", checkpoint)
        safetensors.torch.load_model(model, checkpoint, device=str(device))
    elif not args.allow_random_pi0:
        raise ValueError(
            "A PI0 checkpoint is required. Pass --pi0-checkpoint or use --allow-random-pi0 for debug only."
        )
    else:
        logging.warning("No PI0 checkpoint was provided; training critic on randomly initialized PI0 features.")

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def save_checkpoint(
    output_dir: Path,
    step: int,
    critic: ActionGainCritic,
    optimizer: torch.optim.Optimizer,
    gain_atoms: np.ndarray,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "critic_state_dict": critic.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "critic_config": {
            "input_dim": critic.input_dim,
            "hidden_dim": critic.hidden_dim,
            "num_gain_bins": critic.num_gain_bins,
            "horizon": critic.horizon,
            "num_layers": critic.num_layers,
            "num_heads": critic.num_heads,
            "dropout": args.dropout,
        },
        "gain_atoms": gain_atoms,
        "args": vars(args),
    }
    step_path = output_dir / f"checkpoint_step_{step}.pt"
    latest_path = output_dir / "latest.pt"
    torch.save(payload, step_path)
    torch.save(payload, latest_path)
    logging.info("Saved critic checkpoint to %s", step_path)


def main() -> None:
    init_logging()
    args = parse_args()
    config = _config.get_config(args.config_name)
    device = torch.device(args.device)
    seed = config.seed if args.seed is None else args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    batch_size = config.batch_size if args.batch_size is None else args.batch_size
    num_workers = config.num_workers if args.num_workers is None else args.num_workers
    num_train_steps = config.num_train_steps if args.num_train_steps is None else args.num_train_steps
    dataset_paths = _resolve_dataset_paths(args)

    loader, _ = action_gain_data.create_action_gain_data_loader(
        config,
        args.labels_path,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        seed=seed,
        dataset_paths=dataset_paths,
        skip_norm_stats=args.skip_norm_stats,
    )
    for preview in loader.dataset.preview_identities(5):
        logging.info(
            "label/sample key: dataset=%s episode=%s frame=%s next_frame=%s raw_index=%s label_index=%s",
            preview["dataset_key"],
            preview["episode_index"],
            preview["frame_index"],
            preview["next_frame_index"],
            preview["raw_index"],
            preview["label_index"],
        )
    gain_atoms = torch.as_tensor(loader.dataset.gain_atoms, dtype=torch.float32, device=device)

    pi0 = load_frozen_pi0(config, args, device)
    input_dim = pi0.action_out_proj.in_features
    critic = ActionGainCritic(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_gain_bins=gain_atoms.numel(),
        horizon=args.horizon,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(critic.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    global_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        critic.load_state_dict(checkpoint["critic_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = int(checkpoint.get("step", 0))
        logging.info("Resumed critic from %s at step %d", args.resume, global_step)

    critic.train()
    output_dir = Path(args.output_dir)
    stats: list[dict[str, float]] = []
    start = time.time()
    pbar = tqdm.tqdm(total=num_train_steps, initial=global_step, desc="action-gain")
    while global_step < num_train_steps:
        for observation, actions, gain_target_probs, _metadata in loader:
            if global_step >= num_train_steps:
                break

            observation = jax.tree.map(lambda x: x.to(device), observation)
            actions = actions.to(device=device, dtype=torch.float32)
            gain_target_probs = gain_target_probs.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                prefix_context = pi0.encode_prefix_context(observation)
                features = pi0.encode_actions_joint(
                    observation,
                    actions,
                    timestep=args.timestep,
                    prefix_context=prefix_context,
                )
                action_hidden = features["action_hidden"]

            gain_logits = critic(action_hidden)
            loss = soft_cross_entropy(gain_logits, gain_target_probs)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                pred_probs = torch.softmax(gain_logits.float(), dim=-1)
                pred_gain = expected_value(pred_probs, gain_atoms)
                target_gain = expected_value(gain_target_probs, gain_atoms)
                pred_aggr = aggregate_gain_probs(pred_probs, gain_atoms, args.eta)
                target_aggr = aggregate_gain_probs(gain_target_probs, gain_atoms, args.eta)
                stats.append(
                    {
                        "loss": float(loss.detach().cpu()),
                        "gain_mae": float((pred_gain - target_gain).abs().mean().cpu()),
                        "pred_p_up": float(pred_aggr["p_up"].mean().cpu()),
                        "target_p_up": float(target_aggr["p_up"].mean().cpu()),
                        "grad_norm": (
                            float(grad_norm.detach().cpu())
                            if isinstance(grad_norm, torch.Tensor)
                            else float(grad_norm)
                        ),
                    }
                )

            global_step += 1
            pbar.update(1)
            if global_step % args.log_interval == 0:
                elapsed = max(time.time() - start, 1e-6)
                avg = {key: sum(item[key] for item in stats) / len(stats) for key in stats[0]}
                logging.info(
                    "step=%d loss=%.4f gain_mae=%.4f pred_p_up=%.3f target_p_up=%.3f grad_norm=%.3f %.2fs/step",
                    global_step,
                    avg["loss"],
                    avg["gain_mae"],
                    avg["pred_p_up"],
                    avg["target_p_up"],
                    avg["grad_norm"],
                    elapsed / len(stats),
                )
                stats = []
                start = time.time()
            if global_step % args.save_interval == 0:
                save_checkpoint(output_dir, global_step, critic, optimizer, loader.dataset.gain_atoms, args)

    pbar.close()
    save_checkpoint(output_dir, global_step, critic, optimizer, loader.dataset.gain_atoms, args)


if __name__ == "__main__":
    main()
