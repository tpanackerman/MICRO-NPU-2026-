#!/usr/bin/env python3
"""Compare the generated texts of two Arena accuracy logs (qkv_arena.sh output): same request -> same text?

    compare_outputs.py A.log B.log

Every `STAGE2_RESULT {json}` record with an `output` field is keyed by (suite, request_index); the script prints how many requests match exactly and the first differences.
Exit code 0 only when every request present in both logs has the identical text.
"""
import json
import sys


def records(path):
    out = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        i = line.find("STAGE2_RESULT ")
        if i < 0:
            continue
        try:
            rec = json.loads(line[i + len("STAGE2_RESULT "):])
        except json.JSONDecodeError:
            continue
        if "output" in rec:
            key = (rec.get("suite"), rec.get("request_index"))
            out[key] = rec
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    a, b = records(sys.argv[1]), records(sys.argv[2])
    common = sorted(set(a) & set(b), key=str)
    same = [k for k in common if a[k]["output"] == b[k]["output"]]
    print(f"{sys.argv[1]}: {len(a)} records, {sys.argv[2]}: {len(b)} records, common {len(common)}, identical {len(same)}")
    for k in [k for k in common if a[k]["output"] != b[k]["output"]][:5]:
        print(f"--- request {k} differs\nA: {a[k]['output'][:300]!r}\nB: {b[k]['output'][:300]!r}")
    sa = sum(r.get("score", 0) for r in a.values()) / max(len(a), 1)
    sb = sum(r.get("score", 0) for r in b.values()) / max(len(b), 1)
    print(f"mean score A {sa:.4f}  B {sb:.4f}")
    sys.exit(0 if common and len(same) == len(common) else 1)


if __name__ == "__main__":
    main()
