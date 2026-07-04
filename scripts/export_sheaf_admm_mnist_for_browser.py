#!/usr/bin/env python3
"""Export a Sheaf-ADMM MNIST checkpoint into browser-loadable tensors.

Usage:
    python scripts/export_sheaf_admm_mnist_for_browser.py \
        /path/to/checkpoint.pkl public/data/demos/sheaf-admm-mnist/params/v1

The browser demo reads the generated manifest and raw float32 tensor shards.
The exported names intentionally mirror the TypeScript inference code, not the
Flax parameter tree.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np


CONFIG_DEFAULTS: dict[str, Any] = {
    "imageSize": 28,
    "patchSize": 3,
    "stride": 3,
    "connectivity": 8,
    "numClasses": 10,
    "numAgents": 81,
    "numEdges": 272,
    "dV": 32,
    "dE": 24,
    "hiddenDim": 256,
    "numDirections": 8,
    "loraRank": 8,
    "loraAlpha": 1.0,
    "l1Weight": 0.006337180166370117,
    "rhoInit": 0.12,
    "rhoLearnable": True,
    "cgIters": 5,
    "tikhonovEps": 1.0e-5,
    "zMode": "project",
    "zInit": "h",
    "evalIterations": 20,
    "checkpointKind": "ema_params",
}


def inverse_softplus(x: float) -> float:
    x = max(x, 1e-7)
    return x if x > 20.0 else math.log(math.expm1(x))


def softplus(x: np.ndarray | float) -> np.ndarray | float:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def write_tensor(out_dir: Path, manifest: dict[str, Any], name: str, value: Any) -> None:
    arr = np.asarray(value, dtype=np.float32)
    file_name = f"{name}.f32.bin"
    path = out_dir / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)
    manifest["tensors"][name] = {
        "shape": list(arr.shape),
        "dtype": "float32",
        "file": file_name,
    }


def export_checkpoint(checkpoint: Path, out_dir: Path, use_params: bool = False) -> None:
    with checkpoint.open("rb") as f:
        data = pickle.load(f)

    root = data["params" if use_params else "ema_params"]["params"]
    enc = root["MLPEncoder_0"]
    dec = root["ClassificationDecoder_0"]
    config = dict(CONFIG_DEFAULTS)
    config["checkpointKind"] = "params" if use_params else "ema_params"
    source_config = data.get("config", {})
    model_cfg = source_config.get("model", {})
    train_cfg = source_config.get("training", {})
    task_cfg = source_config.get("task_cfg", {})
    config["sourceConfig"] = source_config

    config["patchSize"] = int(task_cfg.get("patch_size", config["patchSize"]))
    config["stride"] = int(task_cfg.get("stride", config["stride"]))
    config["connectivity"] = int(task_cfg.get("connectivity", config["connectivity"]))
    config["numClasses"] = int(task_cfg.get("num_classes", config["numClasses"]))
    config["dV"] = int(model_cfg.get("d_v", config["dV"]))
    config["dE"] = int(model_cfg.get("d_e", config["dE"]))
    config["hiddenDim"] = int(model_cfg.get("enc_hidden_dim", config["hiddenDim"]))
    config["numDirections"] = int(model_cfg.get("num_directions", config["numDirections"]))
    config["loraRank"] = int(model_cfg.get("lora_rank", config["loraRank"]))
    config["loraAlpha"] = float(model_cfg.get("lora_alpha", config["loraAlpha"]))
    config["l1Weight"] = float(model_cfg.get("l1_weight", config["l1Weight"]))
    config["rhoInit"] = float(model_cfg.get("rho_init", config["rhoInit"]))
    config["cgIters"] = int(model_cfg.get("cg_iters", config["cgIters"]))
    config["tikhonovEps"] = float(model_cfg.get("tikhonov_eps", config["tikhonovEps"]))
    config["zMode"] = str(model_cfg.get("z_mode", config["zMode"]))
    config["zInit"] = str(model_cfg.get("z_init", config["zInit"]))
    config["evalIterations"] = int(train_cfg.get("K_eval", config["evalIterations"]))

    rho_raw = np.asarray(root["rho_raw"], dtype=np.float32)
    rho = float(softplus(rho_raw + inverse_softplus(float(config["rhoInit"]))))
    config["rho"] = rho

    manifest: dict[str, Any] = {
        "format": "sheaf-admm-mnist-v1",
        "config": config,
        "tensors": {},
    }

    write_tensor(out_dir, manifest, "encoder.inputProj.kernel", enc["input_proj"]["kernel"])
    write_tensor(out_dir, manifest, "encoder.inputProj.bias", enc["input_proj"]["bias"])
    write_tensor(out_dir, manifest, "encoder.block0.norm.scale", enc["block_0"]["norm"]["scale"])
    write_tensor(out_dir, manifest, "encoder.block0.dense1.kernel", enc["block_0"]["dense1"]["kernel"])
    write_tensor(out_dir, manifest, "encoder.block0.dense1.bias", enc["block_0"]["dense1"]["bias"])
    write_tensor(out_dir, manifest, "encoder.block0.dense2.kernel", enc["block_0"]["dense2"]["kernel"])
    write_tensor(out_dir, manifest, "encoder.block0.dense2.bias", enc["block_0"]["dense2"]["bias"])
    write_tensor(out_dir, manifest, "encoder.commHead.kernel", enc["comm_head"]["comm_dense"]["kernel"])
    write_tensor(out_dir, manifest, "encoder.commHead.bias", enc["comm_head"]["comm_dense"]["bias"])
    write_tensor(out_dir, manifest, "encoder.commNorm.scale", enc["comm_head"]["comm_norm"]["scale"])
    write_tensor(out_dir, manifest, "encoder.commNorm.bias", enc["comm_head"]["comm_norm"]["bias"])
    write_tensor(out_dir, manifest, "encoder.qDiag.kernel", enc["objective_heads"]["q_diag_dense"]["kernel"])
    write_tensor(out_dir, manifest, "encoder.qDiag.bias", enc["objective_heads"]["q_diag_dense"]["bias"])
    write_tensor(out_dir, manifest, "encoder.q.kernel", enc["objective_heads"]["q_dense"]["kernel"])
    write_tensor(out_dir, manifest, "encoder.q.bias", enc["objective_heads"]["q_dense"]["bias"])
    write_tensor(out_dir, manifest, "encoder.loraPreLn.scale", enc["lora_pre_ln"]["scale"])
    write_tensor(out_dir, manifest, "encoder.loraPreLn.bias", enc["lora_pre_ln"]["bias"])
    write_tensor(out_dir, manifest, "encoder.loraA.kernel", enc["lora_A_dense"]["kernel"])
    write_tensor(out_dir, manifest, "encoder.loraA.bias", enc["lora_A_dense"]["bias"])
    write_tensor(out_dir, manifest, "encoder.loraB.kernel", enc["lora_B_dense"]["kernel"])
    write_tensor(out_dir, manifest, "encoder.loraB.bias", enc["lora_B_dense"]["bias"])
    write_tensor(out_dir, manifest, "restriction.shared", root["rm"]["R_shared"])
    write_tensor(out_dir, manifest, "decoder.cls.kernel", dec["cls_output"]["kernel"])
    write_tensor(out_dir, manifest, "decoder.cls.bias", dec["cls_output"]["bias"])

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--params", action="store_true", help="Export raw params instead of EMA params")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    export_checkpoint(args.checkpoint, args.out_dir, use_params=args.params)
    print(f"Wrote Sheaf-ADMM MNIST browser artifact to {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
