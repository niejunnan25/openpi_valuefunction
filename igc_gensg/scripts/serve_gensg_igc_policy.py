from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib
import socket

from igc_gensg.policies.gensg_igc_policy import GenSGIGCConfig, GenSGIGCPolicy, parse_heads
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve pi0 LIBERO policy with GenSG-IGC online reranking.")
    parser.add_argument("--port", type=int, default=8900)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--policy-config", default="pi0_libero")
    parser.add_argument("--policy-dir", default="/vla/users/niejunnan/assets/openpi-assets/checkpoints/pi0_libero_pytorch")
    parser.add_argument("--pytorch-device", default="cuda:0")
    parser.add_argument("--method", default="gensg_last3")
    parser.add_argument("--generation-heads", default="17:0")
    parser.add_argument("--rescore-heads", default="17:0")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--scoring-timestep", type=float, default=1e-3)
    parser.add_argument("--camera", default="base_0_rgb")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefix-top-heads", type=int, default=4)
    parser.add_argument("--execution-action-tokens", type=int, default=5)
    parser.add_argument("--attention-dir", default="igc_gensg/figures/stage2_gensg_attention")
    parser.add_argument("--disable-attention-npz", action="store_true")
    parser.add_argument("--disable-torch-compile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_config = _config.get_config(args.policy_config)
    if args.disable_torch_compile and hasattr(train_config.model, "pytorch_compile_mode"):
        train_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(train_config.model, pytorch_compile_mode=None),
        )
    base_policy = _policy_config.create_trained_policy(
        train_config,
        pathlib.Path(args.policy_dir),
        sample_kwargs={"num_steps": args.num_steps},
        pytorch_device=args.pytorch_device,
    )
    if getattr(base_policy._model.config, "pi05", False):
        raise RuntimeError("GenSG stage2 requires pi0, but loaded pi05=True")
    cfg = GenSGIGCConfig(
        method=args.method,
        generation_heads=tuple(parse_heads(args.generation_heads)),
        rescore_heads=tuple(parse_heads(args.rescore_heads)),
        k=int(args.k),
        num_steps=int(args.num_steps),
        scoring_timestep=float(args.scoring_timestep),
        camera=args.camera,
        seed=int(args.seed),
        prefix_top_heads=int(args.prefix_top_heads),
        execution_action_tokens=int(args.execution_action_tokens),
        attention_dir=pathlib.Path(args.attention_dir),
        save_attention_npz=not args.disable_attention_npz,
    )
    policy = GenSGIGCPolicy(base_policy, cfg)
    logging.info(
        "Creating GenSG-IGC server host=%s bind=%s port=%s method=%s generation_heads=%s rescore_heads=%s k=%s",
        socket.gethostname(),
        args.host,
        args.port,
        args.method,
        args.generation_heads,
        args.rescore_heads,
        args.k,
    )
    server = websocket_policy_server.WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=policy.metadata)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    main()
