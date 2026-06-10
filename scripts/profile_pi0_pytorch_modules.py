"""Profile PI0 PyTorch module timings and tensor shapes.

This script intentionally uses synthetic BS=32 inputs while loading the real
PyTorch checkpoint. It is meant to isolate model compute cost from dataset I/O.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import statistics
import time
from typing import Any

import torch

from openpi.models import model as _model
from openpi.training import config as _config


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _shape(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, dict):
        return {str(k): _shape(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_shape(v) for v in value]
    if value is None:
        return None
    return str(type(value))


def _cache_shape(past_key_values: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": type(past_key_values).__name__}
    layers = []
    try:
        n_layers = len(past_key_values)
    except Exception:
        n_layers = None
    summary["num_layers"] = n_layers

    if n_layers:
        for idx in [0, n_layers - 1] if n_layers > 1 else [0]:
            try:
                key, value = past_key_values[idx][:2]
            except Exception as exc:
                layers.append({"layer": idx, "error": repr(exc)})
                continue
            layers.append(
                {
                    "layer": idx,
                    "key": _shape(key),
                    "value": _shape(value),
                }
            )
    summary["sample_layers"] = layers
    return summary


def _timed(name: str, fn, *, device: torch.device, warmup: int, iters: int) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
        _sync(device)

    times_ms = []
    output = None
    for _ in range(iters):
        _sync(device)
        start = time.perf_counter()
        output = fn()
        _sync(device)
        times_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "name": name,
        "mean_ms": statistics.mean(times_ms),
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "iters": iters,
        "warmup": warmup,
        "output_shape": _shape(output),
    }


def _load_model(policy_config: str, checkpoint: pathlib.Path, device: torch.device):
    train_config = _config.get_config(policy_config)

    # Disable torch.compile for profiling individual Python methods. Otherwise the
    # first sample_actions call includes compile time and hides inner method costs.
    if hasattr(train_config.model, "pytorch_compile_mode"):
        model_config = dataclasses.replace(train_config.model, pytorch_compile_mode=None)
        train_config = dataclasses.replace(train_config, model=model_config)

    weight_path = checkpoint / "model.safetensors"
    if not weight_path.exists():
        raise FileNotFoundError(f"Missing PyTorch checkpoint: {weight_path}")

    model = train_config.model.load_pytorch(train_config, str(weight_path))
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.to(device)
    model.eval()
    return train_config, model


def _make_observation(train_config, *, batch_size: int, device: torch.device) -> _model.Observation:
    model_config = train_config.model
    image_shape = (batch_size, 3, *_model.IMAGE_RESOLUTION)
    images = {
        key: torch.randn(image_shape, dtype=torch.float32, device=device).clamp(-1.0, 1.0) for key in _model.IMAGE_KEYS
    }
    image_masks = {key: torch.ones((batch_size,), dtype=torch.bool, device=device) for key in _model.IMAGE_KEYS}
    state = torch.randn((batch_size, model_config.action_dim), dtype=torch.float32, device=device).clamp(-1.0, 1.0)

    # Token ids are synthetic but valid. Mask all prompt tokens in so prefix length
    # matches the configured max_token_len.
    tokenized_prompt = torch.ones((batch_size, model_config.max_token_len), dtype=torch.long, device=device)
    tokenized_prompt_mask = torch.ones((batch_size, model_config.max_token_len), dtype=torch.bool, device=device)

    return _model.Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )


def _profile_prefix_context(model, observation, *, device: torch.device, warmup: int, iters: int):
    def run_once():
        result: dict[str, Any] = {}
        marks: dict[str, float] = {}

        def mark(label: str):
            _sync(device)
            marks[label] = time.perf_counter()

        mark("start")
        images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(observation, train=False)  # noqa: SLF001
        mark("preprocess_done")
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        mark("embed_prefix_done")
        prefix_att_2d_masks = (
            model.make_att_2d_masks(prefix_pad_masks, prefix_att_masks) if hasattr(model, "make_att_2d_masks") else None
        )
        if prefix_att_2d_masks is None:
            from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        mark("mask_position_done")
        prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)  # noqa: SLF001
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        mark("prepare_4d_done")
        (prefix_out, _), past_key_values = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        mark("prefix_forward_done")

        result["context"] = {
            "state": state,
            "prefix_out": prefix_out,
            "prefix_pad_masks": prefix_pad_masks,
            "past_key_values": past_key_values,
        }
        result["step_ms"] = {
            "preprocess_observation": (marks["preprocess_done"] - marks["start"]) * 1000.0,
            "embed_prefix": (marks["embed_prefix_done"] - marks["preprocess_done"]) * 1000.0,
            "make_mask_and_positions": (marks["mask_position_done"] - marks["embed_prefix_done"]) * 1000.0,
            "prepare_attention_mask_4d": (marks["prepare_4d_done"] - marks["mask_position_done"]) * 1000.0,
            "prefix_forward_use_cache": (marks["prefix_forward_done"] - marks["prepare_4d_done"]) * 1000.0,
            "total": (marks["prefix_forward_done"] - marks["start"]) * 1000.0,
        }
        result["shapes"] = {
            "images": _shape(images),
            "img_masks": _shape(img_masks),
            "lang_tokens": _shape(lang_tokens),
            "lang_masks": _shape(lang_masks),
            "state": _shape(state),
            "prefix_embs": _shape(prefix_embs),
            "prefix_pad_masks": _shape(prefix_pad_masks),
            "prefix_att_masks": _shape(prefix_att_masks),
            "prefix_att_2d_masks": _shape(prefix_att_2d_masks),
            "prefix_att_2d_masks_4d": _shape(prefix_att_2d_masks_4d),
            "prefix_position_ids": _shape(prefix_position_ids),
            "prefix_out": _shape(prefix_out),
            "past_key_values": _cache_shape(past_key_values),
        }
        return result

    for _ in range(warmup):
        run_once()
        _sync(device)

    runs = []
    out = None
    for _ in range(iters):
        out = run_once()
        _sync(device)
        runs.append(out["step_ms"])

    avg = {key: statistics.mean(run[key] for run in runs) for key in runs[0]}
    return {"step_mean_ms": avg, "shapes": out["shapes"], "context": out["context"]}


def _profile_encode_actions_joint_steps(model, actions, prefix_context, *, timestep: float, device: torch.device):
    result: dict[str, Any] = {}
    marks: dict[str, float] = {}

    def mark(label: str):
        _sync(device)
        marks[label] = time.perf_counter()

    state = prefix_context["state"]
    prefix_pad_masks = prefix_context["prefix_pad_masks"]
    past_key_values = prefix_context["past_key_values"]

    mark("start")
    bsize = actions.shape[0]
    time_tensor = torch.full((bsize,), timestep, dtype=torch.float32, device=actions.device)
    mark("time_tensor_done")
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(state, actions, time_tensor)
    mark("embed_suffix_done")

    if model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
        suffix_embs = suffix_embs.to(torch.bfloat16)

    suffix_len = suffix_pad_masks.shape[1]
    batch_size = prefix_pad_masks.shape[0]
    prefix_len = prefix_pad_masks.shape[1]
    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
    full_att_2d_masks_4d = model._prepare_attention_masks_4d(full_att_2d_masks)  # noqa: SLF001
    mark("mask_position_done")

    model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
    outputs_embeds, _ = model.paligemma_with_expert.forward(
        attention_mask=full_att_2d_masks_4d,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=[None, suffix_embs],
        use_cache=False,
        adarms_cond=[None, adarms_cond],
    )
    mark("suffix_forward_done")
    prefix_out = prefix_context.get("prefix_out")
    suffix_out = outputs_embeds[1]
    action_hidden = suffix_out[:, -model.config.action_horizon :].to(torch.float32)
    pred_velocity = model.action_out_proj(action_hidden)
    mark("action_head_done")

    result["step_ms"] = {
        "make_time_tensor": (marks["time_tensor_done"] - marks["start"]) * 1000.0,
        "embed_suffix": (marks["embed_suffix_done"] - marks["time_tensor_done"]) * 1000.0,
        "make_masks_positions": (marks["mask_position_done"] - marks["embed_suffix_done"]) * 1000.0,
        "cached_suffix_forward": (marks["suffix_forward_done"] - marks["mask_position_done"]) * 1000.0,
        "action_out_projection": (marks["action_head_done"] - marks["suffix_forward_done"]) * 1000.0,
        "total": (marks["action_head_done"] - marks["start"]) * 1000.0,
    }
    result["shapes"] = {
        "state": _shape(state),
        "actions": _shape(actions),
        "time": _shape(time_tensor),
        "suffix_embs": _shape(suffix_embs),
        "suffix_pad_masks": _shape(suffix_pad_masks),
        "suffix_att_masks": _shape(suffix_att_masks),
        "adarms_cond": _shape(adarms_cond),
        "prefix_pad_masks": _shape(prefix_pad_masks),
        "prefix_pad_2d_masks": _shape(prefix_pad_2d_masks),
        "suffix_att_2d_masks": _shape(suffix_att_2d_masks),
        "full_att_2d_masks": _shape(full_att_2d_masks),
        "full_att_2d_masks_4d": _shape(full_att_2d_masks_4d),
        "position_ids": _shape(position_ids),
        "past_key_values": _cache_shape(past_key_values),
        "outputs_embeds": _shape(outputs_embeds),
        "prefix_out": _shape(prefix_out),
        "suffix_out": _shape(suffix_out),
        "action_hidden": _shape(action_hidden),
        "pred_velocity": _shape(pred_velocity),
    }
    return result


def _profile_sample_actions_steps(
    model,
    observation,
    noise,
    prefix_context,
    *,
    device: torch.device,
    num_steps: int,
    reuse_prefix: bool,
):
    result: dict[str, Any] = {}
    marks: dict[str, float] = {}

    def mark(label: str):
        _sync(device)
        marks[label] = time.perf_counter()

    bsize = observation.state.shape[0]
    step_records = []

    mark("start")
    if noise is None:
        actions_shape = (bsize, model.config.action_horizon, model.config.action_dim)
        x_t = model.sample_noise(actions_shape, device)
    else:
        x_t = noise
    mark("noise_ready")

    context = prefix_context if reuse_prefix else model.encode_prefix_context(observation)
    mark("prefix_ready")

    state = context["state"]
    prefix_pad_masks = context["prefix_pad_masks"]
    past_key_values = context["past_key_values"]
    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    time_tensor = torch.tensor(1.0, dtype=torch.float32, device=device)
    mark("loop_ready")

    while time_tensor >= -dt / 2:
        step_start = time.perf_counter()
        expanded_time = time_tensor.expand(bsize)
        _sync(device)
        denoise_start = time.perf_counter()
        v_t = model.denoise_step(
            state,
            prefix_pad_masks,
            past_key_values,
            x_t,
            expanded_time,
        )
        _sync(device)
        denoise_end = time.perf_counter()
        x_t = x_t + dt * v_t
        time_tensor += dt
        _sync(device)
        step_end = time.perf_counter()
        step_records.append(
            {
                "denoise_step": (denoise_end - denoise_start) * 1000.0,
                "euler_update_and_loop_overhead": (step_end - denoise_end) * 1000.0,
                "total": (step_end - step_start) * 1000.0,
            }
        )

    mark("done")

    step_mean_ms = {
        "noise_or_input_ready": (marks["noise_ready"] - marks["start"]) * 1000.0,
        "prefix_context": (marks["prefix_ready"] - marks["noise_ready"]) * 1000.0,
        "loop_setup": (marks["loop_ready"] - marks["prefix_ready"]) * 1000.0,
        "denoise_loop_total": (marks["done"] - marks["loop_ready"]) * 1000.0,
        "total": (marks["done"] - marks["start"]) * 1000.0,
    }
    if step_records:
        step_mean_ms["loop_count"] = len(step_records)
        step_mean_ms["per_step_denoise_mean"] = statistics.mean(step["denoise_step"] for step in step_records)
        step_mean_ms["per_step_euler_update_and_loop_overhead_mean"] = statistics.mean(
            step["euler_update_and_loop_overhead"] for step in step_records
        )
        step_mean_ms["per_step_total_mean"] = statistics.mean(step["total"] for step in step_records)

    result["step_ms"] = step_mean_ms
    result["shapes"] = {
        "x_t_final": _shape(x_t),
        "state": _shape(state),
        "prefix_pad_masks": _shape(prefix_pad_masks),
        "past_key_values": _cache_shape(past_key_values),
        "dt": _shape(dt),
        "time_after_loop": _shape(time_tensor),
    }
    result["per_step_ms"] = step_records
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-config", default="pi0_libero")
    parser.add_argument("--policy-checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--timestep", type=float, default=1e-3)
    parser.add_argument("--output-json", type=pathlib.Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    train_config, model = _load_model(args.policy_config, args.policy_checkpoint, device)
    observation = _make_observation(train_config, batch_size=args.batch_size, device=device)
    actions = torch.randn(
        (args.batch_size, train_config.model.action_horizon, train_config.model.action_dim),
        dtype=torch.float32,
        device=device,
    ).clamp(-1.0, 1.0)
    noise = model.sample_noise(actions.shape, device)
    time_tensor = torch.full((args.batch_size,), args.timestep, dtype=torch.float32, device=device)

    prefix_profile = _profile_prefix_context(
        model,
        observation,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    prefix_context = prefix_profile["context"]

    timings = {}
    timings["embed_prefix"] = _timed(
        "embed_prefix",
        lambda: model.embed_prefix(
            *model._preprocess_observation(observation, train=False)[:4]  # noqa: SLF001
        ),
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    timings["embed_suffix"] = _timed(
        "embed_suffix",
        lambda: model.embed_suffix(prefix_context["state"], actions, time_tensor),
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    timings["denoise_step"] = _timed(
        "denoise_step",
        lambda: model.denoise_step(
            prefix_context["state"],
            prefix_context["prefix_pad_masks"],
            prefix_context["past_key_values"],
            noise,
            torch.ones((args.batch_size,), dtype=torch.float32, device=device),
        ),
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    timings["encode_actions_joint_reuse_prefix"] = _timed(
        "encode_actions_joint_reuse_prefix",
        lambda: model.encode_actions_joint(
            observation,
            actions,
            timestep=args.timestep,
            prefix_context=prefix_context,
        ),
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    timings["encode_actions_joint_standalone"] = _timed(
        "encode_actions_joint_standalone",
        lambda: model.encode_actions_joint(observation, actions, timestep=args.timestep),
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    timings["sample_actions_reuse_prefix"] = _timed(
        "sample_actions_reuse_prefix",
        lambda: model.sample_actions(
            device,
            observation,
            noise=noise,
            num_steps=args.num_steps,
            prefix_context=prefix_context,
        ),
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    timings["sample_actions_standalone"] = _timed(
        "sample_actions_standalone",
        lambda: model.sample_actions(device, observation, noise=noise, num_steps=args.num_steps),
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    timings["forward"] = _timed(
        "forward",
        lambda: model.forward(observation, actions, noise=noise, time=time_tensor),
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )

    encode_step_runs = []
    encode_step_out = None
    for _ in range(args.warmup):
        _profile_encode_actions_joint_steps(
            model,
            actions,
            prefix_context,
            timestep=args.timestep,
            device=device,
        )
    for _ in range(args.iters):
        encode_step_out = _profile_encode_actions_joint_steps(
            model,
            actions,
            prefix_context,
            timestep=args.timestep,
            device=device,
        )
        encode_step_runs.append(encode_step_out["step_ms"])
    encode_steps_mean = {key: statistics.mean(run[key] for run in encode_step_runs) for key in encode_step_runs[0]}

    sample_action_profiles = {}
    for label, reuse_prefix in [
        ("reuse_prefix", True),
        ("standalone", False),
    ]:
        sample_step_runs = []
        sample_step_out = None
        for _ in range(args.warmup):
            _profile_sample_actions_steps(
                model,
                observation,
                noise,
                prefix_context,
                device=device,
                num_steps=args.num_steps,
                reuse_prefix=reuse_prefix,
            )
        for _ in range(args.iters):
            sample_step_out = _profile_sample_actions_steps(
                model,
                observation,
                noise,
                prefix_context,
                device=device,
                num_steps=args.num_steps,
                reuse_prefix=reuse_prefix,
            )
            sample_step_runs.append(sample_step_out["step_ms"])
        sample_steps_mean = {key: statistics.mean(run[key] for run in sample_step_runs) for key in sample_step_runs[0]}
        sample_action_profiles[label] = {
            "step_mean_ms": sample_steps_mean,
            "shapes": sample_step_out["shapes"],
            "per_step_ms_last_run": sample_step_out["per_step_ms"],
        }

    result = {
        "config": {
            "policy_config": args.policy_config,
            "policy_checkpoint": str(args.policy_checkpoint),
            "batch_size": args.batch_size,
            "device": str(device),
            "num_steps": args.num_steps,
            "warmup": args.warmup,
            "iters": args.iters,
            "timestep": args.timestep,
            "action_dim": train_config.model.action_dim,
            "action_horizon": train_config.model.action_horizon,
            "max_token_len": train_config.model.max_token_len,
        },
        "module_timings_ms": timings,
        "encode_prefix_context": {
            "step_mean_ms": prefix_profile["step_mean_ms"],
            "shapes": prefix_profile["shapes"],
        },
        "encode_actions_joint_steps": {
            "step_mean_ms": encode_steps_mean,
            "shapes": encode_step_out["shapes"],
        },
        "sample_actions_steps": sample_action_profiles,
        "input_shapes": {
            "observation": _shape(observation),
            "actions": _shape(actions),
            "noise": _shape(noise),
            "time": _shape(time_tensor),
        },
    }

    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
