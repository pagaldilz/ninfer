#!/usr/bin/env python3
"""Inspect GGUF metadata and stored tensor formats without loading model payloads."""

from __future__ import annotations

import argparse
import json
import os
import struct
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO

VALUE_FORMATS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}
GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
    13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
    18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
    23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0",
}
SELECTED_KEYS = {
    "general.architecture", "general.name", "general.alignment", "general.file_type",
    "general.quantization_version",
}


def read_exact(stream: BinaryIO, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise ValueError("GGUF metadata is truncated")
    return value


def read_string(stream: BinaryIO) -> str:
    size = struct.unpack("<Q", read_exact(stream, 8))[0]
    return read_exact(stream, size).decode("utf-8", errors="replace")


def read_scalar(stream: BinaryIO, value_type: int) -> Any:
    if value_type == 8:
        return read_string(stream)
    fmt = VALUE_FORMATS.get(value_type)
    if fmt is None:
        raise ValueError(f"unsupported GGUF metadata value type {value_type}")
    return struct.unpack(fmt, read_exact(stream, struct.calcsize(fmt)))[0]


def skip_scalar(stream: BinaryIO, value_type: int) -> None:
    if value_type == 8:
        size = struct.unpack("<Q", read_exact(stream, 8))[0]
        stream.seek(size, os.SEEK_CUR)
        return
    fmt = VALUE_FORMATS.get(value_type)
    if fmt is None:
        raise ValueError(f"unsupported GGUF metadata value type {value_type}")
    stream.seek(struct.calcsize(fmt), os.SEEK_CUR)


def read_value(stream: BinaryIO, value_type: int, keep: bool) -> Any:
    if value_type != 9:
        if keep:
            return read_scalar(stream, value_type)
        skip_scalar(stream, value_type)
        return None
    element_type = struct.unpack("<I", read_exact(stream, 4))[0]
    count = struct.unpack("<Q", read_exact(stream, 8))[0]
    if element_type == 9:
        raise ValueError("nested GGUF arrays are not valid")
    if keep and count <= 64:
        return [read_scalar(stream, element_type) for _ in range(count)]
    for _ in range(count):
        skip_scalar(stream, element_type)
    return {"element_type": element_type, "count": count} if keep else None


def align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def inspect(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        if read_exact(stream, 4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version = struct.unpack("<I", read_exact(stream, 4))[0]
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version {version}")
        tensor_count, metadata_count = struct.unpack("<QQ", read_exact(stream, 16))
        metadata: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = read_string(stream)
            value_type = struct.unpack("<I", read_exact(stream, 4))[0]
            keep = (key in SELECTED_KEYS or key.endswith(".block_count") or
                    key.endswith(".embedding_length") or key.endswith(".context_length"))
            value = read_value(stream, value_type, keep)
            if keep:
                metadata[key] = value

        tensors = []
        for _ in range(tensor_count):
            name = read_string(stream)
            dimensions = struct.unpack("<I", read_exact(stream, 4))[0]
            shape = list(struct.unpack(f"<{dimensions}Q", read_exact(stream, 8 * dimensions)))
            tensor_type = struct.unpack("<I", read_exact(stream, 4))[0]
            offset = struct.unpack("<Q", read_exact(stream, 8))[0]
            tensors.append({"name": name, "shape": shape, "type": tensor_type, "offset": offset})

        alignment = int(metadata.get("general.alignment", 32))
        data_start = align(stream.tell(), alignment)

    file_size = path.stat().st_size
    type_bytes: Counter[str] = Counter()
    ordered_tensors = sorted(tensors, key=lambda item: item["offset"])
    for index, tensor in enumerate(ordered_tensors):
        next_offset = (ordered_tensors[index + 1]["offset"] if index + 1 < len(ordered_tensors)
                       else file_size - data_start)
        stored_bytes = next_offset - tensor["offset"]
        type_name = GGML_TYPES.get(tensor["type"], f"GGML_TYPE_{tensor['type']}")
        type_bytes[type_name] += stored_bytes

    return {
        "path": str(path),
        "file_bytes": file_size,
        "version": version,
        "tensor_count": tensor_count,
        "metadata_count": metadata_count,
        "metadata": metadata,
        "tensor_format_bytes": dict(sorted(type_bytes.items())),
        "sample_tensors": tensors[:12],
        "compatibility": {
            "direct_ninfer_binding": False,
            "reason": "GGUF tensor encodings/layouts differ from NInfer row-split-k128 formats",
            "recommended_boundary": "offline GGUF-to-.ninfer conversion, never runtime repacking",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = inspect(args.gguf)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"GGUF v{result['version']}: {result['tensor_count']} tensors, "
              f"{result['file_bytes'] / (1 << 30):.3f} GiB")
        for key, value in result["metadata"].items():
            print(f"{key}: {value}")
        print("stored tensor bytes by GGML type:")
        for tensor_type, size in result["tensor_format_bytes"].items():
            print(f"  {tensor_type}: {size / (1 << 30):.3f} GiB")
        print("direct NInfer binding: false")
        print("recommended boundary: offline GGUF-to-.ninfer conversion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
