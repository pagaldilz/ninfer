#!/usr/bin/env python3
"""Small dependency-free wall-throughput benchmark for OpenAI-compatible local servers."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request


def completion(url: str, payload: dict[str, object], timeout: float) -> tuple[float, dict]:
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    begin = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    return time.perf_counter() - begin, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:1234")
    parser.add_argument("--model", required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--prompt",
        default="Continue an endless sequence of lowercase alphabet letters separated by spaces. "
                "Output only the sequence and never explain.",
    )
    args = parser.parse_args()
    if args.trials < 1 or args.max_tokens < 1:
        parser.error("--trials and --max-tokens must be positive")

    common = {
        "model": args.model,
        "prompt": args.prompt,
        "temperature": 0,
        "stream": False,
    }
    completion(args.base_url, {**common, "max_tokens": args.warmup_tokens}, args.timeout)

    trials = []
    for index in range(args.trials):
        elapsed, response = completion(
            args.base_url, {**common, "max_tokens": args.max_tokens}, args.timeout
        )
        tokens = int(response.get("usage", {}).get("completion_tokens", 0))
        if tokens <= 0:
            raise RuntimeError("server response did not include a positive completion token count")
        trials.append({
            "trial": index + 1,
            "completion_tokens": tokens,
            "elapsed_seconds": elapsed,
            "wall_tokens_per_second": tokens / elapsed,
        })

    rates = [trial["wall_tokens_per_second"] for trial in trials]
    print(json.dumps({
        "model": args.model,
        "base_url": args.base_url,
        "max_tokens": args.max_tokens,
        "trials": trials,
        "median_wall_tokens_per_second": statistics.median(rates),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
