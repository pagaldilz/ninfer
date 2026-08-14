"""Build the RTX 5070 Ti Qwen3.6 artifact from exercised GGUF expert weights.

The existing complete ``.ninfer`` artifact remains the authority for frontend, Text non-expert,
MTP, and Vision objects.  Text routed experts 0..38 are replaced from the exact Unsloth GGUF:

* IQ3_S gate/up is decoded and requantized to NInfer Q3G64_F16S;
* IQ4_XS down is decoded and requantized to NInfer Q4G64_F16S;
* the three promoted down banks and the promoted final gate/up bank retain their original NInfer
  precision.

The script deliberately consumes llama.cpp's official ``gguf-py`` reader/decoder rather than
reimplementing I-quant lookup tables.  The source checkout revision is recorded in the report.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import mmap
from pathlib import Path
import struct
import subprocess
import sys
import time
import types
from typing import BinaryIO, Iterator

import numpy as np


MAGIC = b"NINFER\x00\x01"
PREFIX = struct.Struct("<8sQ")
PAYLOAD_ALIGNMENT = 4096
OBJECT_ALIGNMENT = 256
Q3_FORMAT = "Q3G64_F16S"
Q3_LAYOUT = "group-interleaved-v1"
Q4_FORMAT = "Q4G64_F16S"
Q4_LAYOUT = "row-split-k128-v1"
COMPACT_GATE_LAYERS = tuple(range(39))
COMPACT_DOWN_LAYERS = tuple(layer for layer in range(39) if layer not in (34, 38))
COPY_CHUNK = 16 * 1024 * 1024
EXPERT_BATCH = 16


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def load_gguf(llama_cpp_source: Path):
    package_root = llama_cpp_source / "gguf-py" / "gguf"
    if not (package_root / "gguf_reader.py").is_file():
        raise FileNotFoundError(
            f"llama.cpp gguf-py reader not found below {llama_cpp_source}"
        )
    # Avoid gguf/__init__.py and its converter-only optional dependencies.  The reader and quant
    # modules are a self-contained package subset.
    package = types.ModuleType("gguf")
    package.__path__ = [str(package_root)]
    sys.modules["gguf"] = package
    from gguf.gguf_reader import GGUFReader  # type: ignore
    from gguf.quants import dequantize  # type: ignore

    return GGUFReader, dequantize


def read_ninfer_directory(path: Path) -> tuple[dict[str, object], int]:
    with path.open("rb") as handle:
        magic, json_bytes = PREFIX.unpack(handle.read(PREFIX.size))
        if magic != MAGIC or json_bytes <= 0:
            raise ValueError(f"not a NInfer v1 artifact: {path}")
        directory = json.loads(handle.read(json_bytes).decode("utf-8"))
    if not isinstance(directory, dict) or not isinstance(directory.get("objects"), list):
        raise ValueError("invalid NInfer object directory")
    return directory, align_up(PREFIX.size + json_bytes, PAYLOAD_ALIGNMENT)


def q3_bytes(rows: int, columns: int) -> int:
    if columns % 64:
        raise ValueError("Q3G64 requires complete groups")
    return rows * (columns // 64) * 26


def q4_geometry(rows: int, columns: int) -> tuple[int, int, int]:
    if columns % 64:
        raise ValueError("Q4G64 requires complete groups")
    groups = rows * (columns // 64)
    return groups * 32, groups * 2, groups * 34


def compact_directory(source: dict[str, object]) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    cursor = 0
    objects: list[dict[str, object]] = []
    changed: dict[str, dict[str, object]] = {}
    for raw in source["objects"]:  # type: ignore[index]
        item = dict(raw)
        name = str(item["name"])
        parts = name.split("/")
        text_layer = len(parts) >= 5 and parts[0] == "text" and parts[1] == "layers"
        if text_layer and name.endswith("/moe/routed_gate_up"):
            layer = int(parts[2])
            if layer in COMPACT_GATE_LAYERS:
                item["format"] = Q3_FORMAT
                item["layout"] = Q3_LAYOUT
                item["bytes"] = q3_bytes(262144, 2048)
                changed[name] = item
        elif text_layer and name.endswith("/moe/routed_down"):
            layer = int(parts[2])
            if layer in COMPACT_DOWN_LAYERS:
                item["format"] = Q4_FORMAT
                item["layout"] = Q4_LAYOUT
                item["bytes"] = q4_geometry(524288, 512)[2]
                changed[name] = item
        alignment = OBJECT_ALIGNMENT if item["kind"] == "tensor" else 1
        cursor = align_up(cursor, alignment)
        item["offset"] = cursor
        cursor += int(item["bytes"])
        objects.append(item)
    return {"model_id": source["model_id"], "objects": objects}, changed


def quantize_q3_groups(values: np.ndarray) -> np.ndarray:
    groups = np.asarray(values, dtype=np.float32).reshape(-1, 64)
    maximum = np.max(np.abs(groups), axis=1)
    scale = maximum / np.float32(3.0)
    scale = np.where(maximum == 0, np.float32(1.0), scale).astype(np.float16)
    represented_scale = scale.astype(np.float32)
    codes = np.rint(groups / represented_scale[:, None]).clip(-4, 3).astype(np.int8)
    unsigned = (codes.astype(np.int16) & 7).astype(np.uint8)
    bits = ((unsigned[:, :, None] >> np.arange(3, dtype=np.uint8)) & 1).reshape(-1, 192)
    packed = np.packbits(bits, axis=1, bitorder="little")
    result = np.empty((groups.shape[0], 26), dtype=np.uint8)
    result[:, :24] = packed
    result[:, 24:] = scale.view(np.uint8).reshape(-1, 2)
    return result


def quantize_q4_planes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(values, dtype=np.float32)
    groups = rows.reshape(rows.shape[0], -1, 64)
    maximum = np.max(np.abs(groups), axis=2)
    scale = maximum / np.float32(7.0)
    scale = np.where(maximum == 0, np.float32(1.0), scale).astype(np.float16)
    represented_scale = scale.astype(np.float32)
    codes = np.rint(groups / represented_scale[:, :, None]).clip(-8, 7).astype(np.int8)
    unsigned = (codes.astype(np.int16) & 15).astype(np.uint8)
    packed = unsigned[:, :, 0::2] | (unsigned[:, :, 1::2] << np.uint8(4))
    return packed.reshape(-1), scale.view(np.uint8).reshape(-1)


def write_zeros(handle: BinaryIO, count: int) -> None:
    zeros = b"\x00" * min(COPY_CHUNK, count)
    while count:
        size = min(len(zeros), count)
        handle.write(zeros[:size])
        count -= size


def copy_span(handle: BinaryIO, mapping: mmap.mmap, begin: int, count: int) -> None:
    for offset in range(0, count, COPY_CHUNK):
        handle.write(mapping[begin + offset : begin + min(offset + COPY_CHUNK, count)])


def tensor_map(reader) -> dict[str, object]:
    return {tensor.name: tensor for tensor in reader.tensors}


def require_tensor(tensors: dict[str, object], name: str, qtype: str, shape: tuple[int, ...]):
    tensor = tensors.get(name)
    if tensor is None:
        raise ValueError(f"GGUF tensor is missing: {name}")
    actual_shape = tuple(int(dim) for dim in tensor.shape)
    if tensor.tensor_type.name != qtype or actual_shape != shape:
        raise ValueError(
            f"GGUF tensor mismatch for {name}: {tensor.tensor_type.name} {actual_shape}, "
            f"expected {qtype} {shape}"
        )
    return tensor


def write_q3_gate_up(handle: BinaryIO, layer: int, tensors: dict[str, object], dequantize,
                     workers: int) -> None:
    gate = require_tensor(
        tensors, f"blk.{layer}.ffn_gate_exps.weight", "IQ3_S", (2048, 512, 256)
    )
    up = require_tensor(
        tensors, f"blk.{layer}.ffn_up_exps.weight", "IQ3_S", (2048, 512, 256)
    )
    def encode(expert: int) -> tuple[bytes, bytes]:
        gate_rows = dequantize(gate.data[expert], gate.tensor_type)
        up_rows = dequantize(up.data[expert], up.tensor_type)
        return quantize_q3_groups(gate_rows).tobytes(), quantize_q3_groups(up_rows).tobytes()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for begin in range(0, 256, EXPERT_BATCH):
            for gate_bytes, up_bytes in executor.map(encode, range(begin, begin + EXPERT_BATCH)):
                handle.write(gate_bytes)
                handle.write(up_bytes)


def write_q4_down(
    handle: BinaryIO,
    object_begin: int,
    layer: int,
    tensors: dict[str, object],
    dequantize,
    workers: int,
) -> None:
    down = require_tensor(
        tensors, f"blk.{layer}.ffn_down_exps.weight", "IQ4_XS", (512, 2048, 256)
    )
    base_bytes, _, total_bytes = q4_geometry(524288, 512)
    scale_begin = object_begin + base_bytes
    code_cursor = object_begin
    scale_cursor = scale_begin
    def encode(expert: int) -> tuple[bytes, bytes]:
        rows = dequantize(down.data[expert], down.tensor_type)
        codes, scales = quantize_q4_planes(rows)
        return codes.tobytes(), scales.tobytes()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for begin in range(0, 256, EXPERT_BATCH):
            for code_bytes, scale_bytes in executor.map(encode,
                                                         range(begin, begin + EXPERT_BATCH)):
                handle.seek(code_cursor)
                handle.write(code_bytes)
                code_cursor += len(code_bytes)
                handle.seek(scale_cursor)
                handle.write(scale_bytes)
                scale_cursor += len(scale_bytes)
    if code_cursor != scale_begin or scale_cursor != object_begin + total_bytes:
        raise AssertionError("Q4 down plane length mismatch")


def source_revision(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={path.as_posix()}", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(COPY_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def convert(base: Path, gguf: Path, output: Path, llama_cpp_source: Path,
            workers: int) -> dict[str, object]:
    started = time.perf_counter()
    source_directory, source_payload = read_ninfer_directory(base)
    directory, changed = compact_directory(source_directory)
    encoded_directory = json.dumps(
        directory, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    output_payload = align_up(PREFIX.size + len(encoded_directory), PAYLOAD_ALIGNMENT)
    GGUFReader, dequantize = load_gguf(llama_cpp_source)
    gguf_reader = GGUFReader(gguf)
    tensors = tensor_map(gguf_reader)

    source_objects = {str(item["name"]): item for item in source_directory["objects"]}
    with base.open("rb") as source_file, output.open("wb+") as target:
        source_map = mmap.mmap(source_file.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            target.write(PREFIX.pack(MAGIC, len(encoded_directory)))
            target.write(encoded_directory)
            write_zeros(target, output_payload - PREFIX.size - len(encoded_directory))
            for index, item in enumerate(directory["objects"], start=1):
                name = str(item["name"])
                absolute = output_payload + int(item["offset"])
                target.seek(absolute)
                if name in changed:
                    layer = int(name.split("/")[2])
                    if name.endswith("routed_gate_up"):
                        write_q3_gate_up(target, layer, tensors, dequantize, workers)
                    else:
                        write_q4_down(target, absolute, layer, tensors, dequantize, workers)
                else:
                    old = source_objects[name]
                    copy_span(
                        target,
                        source_map,
                        source_payload + int(old["offset"]),
                        int(old["bytes"]),
                    )
                print(f"[{index}/{len(directory['objects'])}] {name}", flush=True)
            last = directory["objects"][-1]
            final_bytes = output_payload + int(last["offset"]) + int(last["bytes"])
            target.truncate(final_bytes)
        finally:
            source_map.close()

    return {
        "artifact_type": "ninfer_gguf_expert_repack_report",
        "schema_version": 1,
        "base_artifact": str(base.resolve()),
        "gguf": str(gguf.resolve()),
        "llama_cpp_source": str(llama_cpp_source.resolve()),
        "llama_cpp_revision": source_revision(llama_cpp_source),
        "output": str(output.resolve()),
        "base_bytes": base.stat().st_size,
        "output_bytes": output.stat().st_size,
        "saved_bytes": base.stat().st_size - output.stat().st_size,
        "compact_gate_layers": list(COMPACT_GATE_LAYERS),
        "compact_down_layers": list(COMPACT_DOWN_LAYERS),
        "changed_objects": len(changed),
        "workers": workers,
        "elapsed_seconds": time.perf_counter() - started,
        "output_sha256": sha256(output),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--llama-cpp-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {args.out}")
    partial = Path(str(args.out) + ".partial")
    if partial.exists():
        raise FileExistsError(f"remove or inspect stale partial artifact first: {partial}")
    report_path = args.report or Path(str(args.out) + ".conversion.json")
    report = convert(args.base, args.gguf, partial, args.llama_cpp_source, args.workers)
    partial.replace(args.out)
    report["output"] = str(args.out.resolve())
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
