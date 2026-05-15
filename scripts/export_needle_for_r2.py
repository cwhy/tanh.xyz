#!/usr/bin/env python3
"""Convert a Needle pickle checkpoint into browser-loadable tensor files.

Usage:
    python scripts/export_needle_for_r2.py /path/to/needle.pkl /tmp/needle-r2

Upload the output directory to R2 at data/demos/needle/needle/params/v1/.
The canonical public manifest URL is
https://media.tanh.xyz/data/demos/needle/needle/params/v1/manifest.json.
Tensors are written as BF16 because Needle's upstream loader casts checkpoint
params to bfloat16.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def to_bfloat16_bits(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    bits = arr.view(np.uint32)
    lsb = (bits >> 16) & 1
    rounded = bits + np.uint32(0x7FFF) + lsb
    return (rounded >> 16).astype(np.uint16)


def write_tensor(out_dir: Path, manifest: dict[str, Any], name: str, value: Any) -> None:
    arr = to_bfloat16_bits(value)
    file_name = f"{name}.bf16.bin"
    path = out_dir / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)
    manifest["tensors"][name] = {
        "shape": list(arr.shape),
        "dtype": "bfloat16",
        "file": file_name,
    }


def scalar_or_slice(value: Any, index: int | None = None) -> Any:
    arr = np.asarray(value)
    return arr[index] if index is not None else arr


def camel_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "vocabSize": int(config.get("vocab_size", 8192)),
        "dModel": int(config.get("d_model", 512)),
        "numHeads": int(config.get("num_heads", 8)),
        "numKvHeads": int(config.get("num_kv_heads", 4)),
        "numEncoderLayers": int(config.get("num_encoder_layers", 12)),
        "numDecoderLayers": int(config.get("num_decoder_layers", 8)),
        "maxSeqLen": int(config.get("max_seq_len", 1024)),
        "padTokenId": int(config.get("pad_token_id", 0)),
        "eosTokenId": 1,
        "toolCallTokenId": 4,
        "toolsTokenId": 5,
        "ropeTheta": float(config.get("rope_theta", 10000.0)),
    }


def export_checkpoint(checkpoint: Path, out_dir: Path) -> None:
    with checkpoint.open("rb") as f:
        data = pickle.load(f)

    params = data["params"]
    config = camel_config(data.get("config", {}))
    manifest: dict[str, Any] = {
        "config": config,
        "tensors": {},
    }

    write_tensor(out_dir, manifest, "embedding", params["embedding"]["embedding"])
    write_tensor(out_dir, manifest, "encoderFinalNormScale", params["encoder"]["final_norm"]["scale"])
    write_tensor(out_dir, manifest, "decoderFinalNormScale", params["decoder"]["ZCRMSNorm_0"]["scale"])

    enc = params["encoder"]["layers"]["EncoderBlock_0"]
    for i in range(config["numEncoderLayers"]):
        prefix = f"encoderLayers.{i}"
        write_tensor(out_dir, manifest, f"{prefix}.attnNormScale", scalar_or_slice(enc["ZCRMSNorm_0"]["scale"], i))
        write_tensor(out_dir, manifest, f"{prefix}.attnGate", scalar_or_slice(enc["attn_gate"], i))
        attn = enc["self_attn"]
        write_attention(out_dir, manifest, f"{prefix}.selfAttn", attn, i)

    dec = params["decoder"]["layers"]["DecoderBlock_0"]
    for i in range(config["numDecoderLayers"]):
        prefix = f"decoderLayers.{i}"
        write_tensor(out_dir, manifest, f"{prefix}.selfNormScale", scalar_or_slice(dec["ZCRMSNorm_0"]["scale"], i))
        write_tensor(out_dir, manifest, f"{prefix}.selfAttnGate", scalar_or_slice(dec["self_attn_gate"], i))
        write_attention(out_dir, manifest, f"{prefix}.selfAttn", dec["self_attn"], i)
        write_tensor(out_dir, manifest, f"{prefix}.crossNormScale", scalar_or_slice(dec["ZCRMSNorm_1"]["scale"], i))
        write_tensor(out_dir, manifest, f"{prefix}.crossAttnGate", scalar_or_slice(dec["cross_attn_gate"], i))
        write_attention(out_dir, manifest, f"{prefix}.crossAttn", dec["cross_attn"], i)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def write_attention(out_dir: Path, manifest: dict[str, Any], prefix: str, attn: dict[str, Any], index: int) -> None:
    write_tensor(out_dir, manifest, f"{prefix}.qProj", scalar_or_slice(attn["q_proj"]["kernel"], index))
    write_tensor(out_dir, manifest, f"{prefix}.kProj", scalar_or_slice(attn["k_proj"]["kernel"], index))
    write_tensor(out_dir, manifest, f"{prefix}.vProj", scalar_or_slice(attn["v_proj"]["kernel"], index))
    write_tensor(out_dir, manifest, f"{prefix}.outProj", scalar_or_slice(attn["out_proj"]["kernel"], index))
    write_tensor(out_dir, manifest, f"{prefix}.qNormScale", scalar_or_slice(attn["q_norm"]["scale"], index))
    write_tensor(out_dir, manifest, f"{prefix}.kNormScale", scalar_or_slice(attn["k_norm"]["scale"], index))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    export_checkpoint(args.checkpoint, args.out_dir)
    print(f"Wrote Needle browser artifact manifest to {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
