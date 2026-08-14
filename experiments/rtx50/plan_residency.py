#!/usr/bin/env python3
"""Compute the exact two-GPU placement for a Qwen3.8 NInfer artifact."""

from __future__ import annotations

import argparse
import json
import re
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any

GIB = 1 << 30
PREFIX = struct.Struct("<8sQ")
LAYER_PATTERN = re.compile(r"^text/layers/(\d+)/")


def read_directory(path: Path) -> dict[str, Any]:
    with path.open("rb") as artifact:
        raw_prefix = artifact.read(PREFIX.size)
        if len(raw_prefix) != PREFIX.size:
            raise ValueError("artifact is shorter than the NInfer prefix")
        magic, directory_bytes = PREFIX.unpack(raw_prefix)
        if magic != b"NINFER\x00\x02":
            raise ValueError(f"expected NINFER v2 magic, found {magic!r}")
        raw_directory = artifact.read(directory_bytes)
        if len(raw_directory) != directory_bytes:
            raise ValueError("artifact directory is truncated")
    return json.loads(raw_directory)


def gib(value: int) -> float:
    return value / GIB


def plan(directory: dict[str, Any], fast_vram: int, slow_vram: int,
         fast_bandwidth_gbs: float, output_head_us: float,
         boundary_round_us: float) -> dict[str, Any]:
    tensors = [obj for obj in directory["objects"] if obj["kind"] == "tensor"]
    by_name = {obj["name"]: obj for obj in tensors}
    required = {"text/token_embedding", "text/final_norm", "text/output_head"}
    missing = sorted(required - by_name.keys())
    if missing:
        raise ValueError(f"artifact lacks required Qwen3.8 tensors: {', '.join(missing)}")

    layers: dict[int, int] = defaultdict(int)
    for tensor in tensors:
        match = LAYER_PATTERN.match(tensor["name"])
        if match:
            layers[int(match.group(1))] += int(tensor["bytes"])
    if sorted(layers) != list(range(64)):
        raise ValueError("expected exactly text layers 0 through 63")

    fast_names = {name for name in by_name if LAYER_PATTERN.match(name)}
    fast_names.add("text/final_norm")
    fast_names.update(name for name in by_name if name.startswith("mtp/"))
    slow_names = by_name.keys() - fast_names
    fast_weight_bytes = sum(int(by_name[name]["bytes"]) for name in fast_names)
    slow_weight_bytes = sum(int(by_name[name]["bytes"]) for name in slow_names)

    embedding = by_name["text/token_embedding"]
    hidden = int(embedding["shape"][1])
    embedding_row_bytes = int(embedding["bytes"]) // int(embedding["shape"][0])

    base_fast_stream_bytes = sum(layers.values()) + int(by_name["text/final_norm"]["bytes"])
    # This component floor combines the fast device's measured sustained bandwidth with the
    # separately measured exact W8 output head and staged boundary. It is deliberately not
    # reported as expected throughput: attention, state, launch, and sampling costs remain.
    component_seconds = (base_fast_stream_bytes / (fast_bandwidth_gbs * 1e9) +
                         output_head_us * 1e-6 + boundary_round_us * 1e-6)

    categories: dict[str, int] = defaultdict(int)
    for tensor in tensors:
        name = tensor["name"]
        if LAYER_PATTERN.match(name):
            category = "text_layers"
        elif name in required:
            category = "text_endpoints"
        else:
            category = name.split("/", 1)[0]
        categories[category] += int(tensor["bytes"])

    result = {
        "identity": directory["identity"],
        "strategy": "endpoint-offload",
        "fast_device": {
            "contents": "text/layers/0..63, text/final_norm, and MTP tensors",
            "weight_bytes": fast_weight_bytes,
            "weight_gib": gib(fast_weight_bytes),
            "vram_bytes": fast_vram,
            "headroom_bytes": fast_vram - fast_weight_bytes,
            "headroom_gib": gib(fast_vram - fast_weight_bytes),
        },
        "slow_device": {
            "contents": "token embedding, output head, draft head, Vision, and remaining tensors",
            "weight_bytes": slow_weight_bytes,
            "weight_gib": gib(slow_weight_bytes),
            "vram_bytes": slow_vram,
            "headroom_bytes": slow_vram - slow_weight_bytes,
            "headroom_gib": gib(slow_vram - slow_weight_bytes),
        },
        "boundary": {
            "hidden_width": hidden,
            "bf16_bytes_each_direction_per_decode_row": hidden * 2,
            "embedding_row_bytes_read": embedding_row_bytes,
            "crossings_per_base_round": 2,
        },
        "measured_component_floor": {
            "fast_device_gb_s": fast_bandwidth_gbs,
            "fast_stream_bytes_per_base_round": base_fast_stream_bytes,
            "output_head_us": output_head_us,
            "boundary_round_us": boundary_round_us,
            "seconds_per_base_round": component_seconds,
            "base_rounds_per_second": 1.0 / component_seconds,
            "warning": "component lower bound, not an end-to-end throughput prediction",
        },
        "tensor_bytes_by_category": dict(sorted(categories.items())),
        "fits": fast_weight_bytes < fast_vram and slow_weight_bytes < slow_vram,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--fast-vram-bytes", type=int, default=17_094_475_776)
    parser.add_argument("--slow-vram-bytes", type=int, default=17_102_864_384)
    parser.add_argument("--fast-bandwidth-gbs", type=float, default=852.7)
    parser.add_argument("--output-head-us", type=float, default=3283.232)
    parser.add_argument("--boundary-round-us", type=float, default=142.371)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = plan(read_directory(args.artifact), args.fast_vram_bytes, args.slow_vram_bytes,
                  args.fast_bandwidth_gbs, args.output_head_us, args.boundary_round_us)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        identity = result["identity"]
        print(f"identity: {identity['model_id']}/{identity['weights_id']}")
        print("placement: endpoint-offload")
        for key in ("fast_device", "slow_device"):
            device = result[key]
            print(f"{key}: {device['weight_gib']:.3f} GiB weights, "
                  f"{device['headroom_gib']:.3f} GiB physical headroom")
            print(f"  {device['contents']}")
        boundary = result["boundary"]
        print(f"decode boundary: {boundary['crossings_per_base_round']} x "
              f"{boundary['bf16_bytes_each_direction_per_decode_row']} bytes per row")
        bound = result["measured_component_floor"]
        print(f"measured-component base-round ceiling: {bound['base_rounds_per_second']:.1f}/s "
              "(not an end-to-end prediction)")
        print(f"fits physical VRAM: {str(result['fits']).lower()}")
    return 0 if result["fits"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
