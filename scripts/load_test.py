"""Concurrent-users benchmark: N simultaneous questions through the real orchestrator.

    .venv/bin/python scripts/load_test.py --users 10 --mode mixed --llm local

Two rounds: "cold" (empty HTTP/plan/summary caches) and "warm" (same questions again).
Reports latency percentiles and what the efficiency mechanisms saved: LLM calls actually made
versus the naive 2 per question, HTTP requests served from cache, fetches shared in flight,
fallbacks and errors. Uses the live public APIs and the selected LLM backend.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signalscope.agent import orchestrator  # noqa: E402
from signalscope.agent.llm import build_llm  # noqa: E402
from signalscope.config import Settings  # noqa: E402

QUESTIONS = [
    "Is sunshine in Rome related to solar power generation in Italy?",
    "Is electricity demand in Germany related to temperature in Berlin?",
    "Is wind related to PM10 pollution in Milan?",
    "Does temperature in Milan affect interest in ice cream?",
    "Is the euro–dollar exchange rate related to interest in Bitcoin?",
]


def one_round(users: int, mode: str, settings: Settings, llm) -> dict:
    questions = [QUESTIONS[0]] * users if mode == "same" else [QUESTIONS[i % len(QUESTIONS)] for i in range(users)]
    results: list = [None] * users
    latencies: list[float] = [0.0] * users
    barrier = threading.Barrier(users)

    def user(i: int) -> None:
        barrier.wait()                                   # all users press "ask" at the same moment
        started = time.perf_counter()
        results[i] = orchestrator.run(questions[i], settings, llm)
        latencies[i] = time.perf_counter() - started

    threads = [threading.Thread(target=user, args=(i,)) for i in range(users)]
    wall = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - wall

    totals = [r.trace.totals() for r in results]
    events = [e for r in results for e in r.trace.events]
    ordered = sorted(latencies)
    return {
        "ok": sum(r.ok for r in results),
        "errors": sorted({r.error for r in results if r.error}),
        "p50": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))],
        "max": ordered[-1],
        "wall": wall,
        "llm_calls": sum(t["llm_calls"] for t in totals),
        "naive_llm_calls": 2 * users,
        "tokens_in": sum(t["prompt_tokens"] for t in totals),
        "tokens_out": sum(t["completion_tokens"] for t in totals),
        "fetches": sum(t["api_calls"] for t in totals),
        "cache_or_shared": sum(t["cache_hits"] for t in totals),
        "shared_in_flight": sum(1 for e in events if e.kind == "api" and e.details.get("shared")),
        "fallbacks": sum(1 for e in events if "falling back" in e.message),
        "llm_ms_share": sum(t["llm_ms"] for t in totals) / max(1.0, sum(t["total_ms"] for t in totals)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--users", type=int, default=10)
    parser.add_argument("--mode", choices=["same", "mixed"], default="mixed")
    parser.add_argument("--llm", choices=["auto", "cloud", "local"], default="local")
    args = parser.parse_args()

    # Fresh on-disk HTTP cache so the cold round really is cold.
    settings = replace(Settings.from_env(), cache_path=Path(tempfile.mkdtemp()) / "http")
    llm = build_llm(settings, mode=args.llm)
    print(f"{args.users} concurrent users · questions: {args.mode} · LLM: {[c.label for c in llm.clients]}\n")
    header = f"{'round':5} {'ok':>5} {'p50 s':>6} {'p95 s':>6} {'max s':>6} {'LLM calls':>12} {'HTTP cache/shared':>18} {'in-flight':>9} {'fallbacks':>9} {'LLM share':>9}"
    print(header)
    print("-" * len(header))
    for name in ("cold", "warm"):
        if name == "cold":
            orchestrator.PLAN_CACHE.clear()
            orchestrator.SUMMARY_CACHE.clear()
        r = one_round(args.users, args.mode, settings, llm)
        print(f"{name:5} {r['ok']:>2}/{args.users:<2} {r['p50']:6.1f} {r['p95']:6.1f} {r['max']:6.1f} "
              f"{r['llm_calls']:>4} of {r['naive_llm_calls']:<4} {r['cache_or_shared']:>7} of {r['fetches']:<7}  "
              f"{r['shared_in_flight']:>9} {r['fallbacks']:>9} {r['llm_ms_share']:>8.0%}")
        for error in r["errors"]:
            print(f"      error: {error}")
        print(f"      tokens {r['tokens_in']} in / {r['tokens_out']} out · wall time {r['wall']:.1f} s")


if __name__ == "__main__":
    main()
