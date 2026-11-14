#!/usr/bin/env python3
"""Summarise Arena performance logs (qkv_arena.sh output): tok/s, TPOT / TTFT, and the server's per-pass device timers.

    perf_summary.py LOG [LOG ...]

The server prints `S2ITER ... name=count/total_ms` timers per window of 64 iterations; they are summed over all windows and divided by the number of 256-row passes (the `hs256_*` counters).
hybrid_step / t_layer_* etc. also count the passes of the other engines, so the hs256 pass time and the t_qkv_* timers are the comparable numbers.
"""
import json
import re
import sys

TIMERS = ["hybrid_step", "t_qkv_s", "t_qkv_f", "t_attn_s", "t_attn_f", "t_dattn_f", "t_layer_s", "t_layer_f", "t_tables", "next_tokens"]


def parse(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    out = {"log": path.split("/")[-1]}
    m = re.findall(r"STAGE2_PERFORMANCE_SUMMARY (\{.*\})", text)
    if m:
        s = json.loads(m[-1])
        met = s["metrics"]
        out.update(tok_s=met["generated_tokens_per_second"], tpot_p50=met["estimated_tpot_ms"]["p50"], tpot_p90=met["estimated_tpot_ms"]["p90"],
                   ttft_p90=met["ttft_ms"]["p90"], passed=f'{s["passed"]}/{s["total"]}')
    timers = {}
    for line in text.splitlines():
        if line.startswith("S2ITER "):
            for k, c, t in re.findall(r"(\w+)=(\d+)/([\d.]+)", line):
                n, ms = timers.get(k, (0, 0.0))
                timers[k] = (n + int(c), ms + float(t))
    # 256-row passes = the hs256_* buckets; every timer below is per 256-row pass
    passes = sum(n for k, (n, _) in timers.items() if k.startswith("hs256_"))
    out["passes"] = passes
    if passes:
        out["hs256_ms"] = round(sum(ms for k, (_, ms) in timers.items() if k.startswith("hs256_")) / passes, 2)
        for k in TIMERS:
            if k in timers:
                out[k] = round(timers[k][1] / passes, 2)
    return out


def main():
    rows = [parse(p) for p in sys.argv[1:]]
    keys = ["log", "tok_s", "tpot_p50", "tpot_p90", "ttft_p90", "passed", "passes", "hs256_ms"] + TIMERS
    print("\t".join(keys))
    for r in rows:
        print("\t".join(f"{r[k]:.2f}" if isinstance(r.get(k), float) else str(r.get(k, "")) for k in keys))


if __name__ == "__main__":
    main()
