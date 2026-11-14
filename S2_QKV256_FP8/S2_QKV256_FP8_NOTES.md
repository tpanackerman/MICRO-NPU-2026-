# S2_QKV256_FP8: Stage 2 candidate, not graded yet (Arena-based estimate about 235 generated tokens/s at concurrency 18)

**What this folder is.** A snapshot of the next Stage 2 candidate ("v4"). It has the organisers' baseline files (README, SERVING, scripts, Cargo files, the same as in `S2_203_tok`) plus the `src/` that was tested on the Arena. Three Arena helper scripts were added to `scripts/stage2/` (see below); `moa-submitter` uploads `src/` only, so they do not change a submission.

**Important: this is built on `S2_203_tok`, NOT on `S2_211_tok`.** It was developed in parallel with the S2_211 work and contains none of its changes (t256r, the fused sliding layer, the host TPOT transport). Both trees edit `src/axes.rs` and `src/host/runtime/hybrid256.rs`, so combining them is a merge plus a new Arena check, not a copy. Note also that the S2_211 fused sliding layer is used only when a layer is NOT FP8-folded, so the FP8 change below switches the fused path off on the folded layers.

**Status.** Not submitted. The attempt on 2026-10-10 09:31 UTC (personal account, concurrency 18) was refused by the team limit (4 of 4 in the last 24 h; next slot 2026-10-10 19:33 UTC). To submit (WSL, moa-submitter 2.1.0 or newer, which has `--concurrency`): `export PATH=$HOME/.cargo/bin:$PATH; moa-submitter submit --source <this folder> --concurrency 18`.

## What changed since S2_203_tok

6 files in `src/` (everything else is byte-identical to `S2_203_tok/src`):

| File | Change |
|---|---|
| `src/device/qkvc/quad.rs` (new) | native T = 256 QKV kernels for the 256-row hybrid pass |
| `src/ops_qkvc.rs` | entries `sliding_qkv_quad_direct`, `full_qkv_quad` |
| `src/device/qkvc/mod.rs` | `mod quad` |
| `src/axes.rs` | axis `Rw = 2` (128-row waves) |
| `src/host/runtime/hybrid256.rs` | uses the native-256 QKV, K03-256, 4-unit rope tables packed back to back |
| `src/host/fp8.rs` | FP8 FFN fold ON by default |

1. **Native T = 256 QKV** (`S2_QKV256` sliding, `S2_QKVF256` full, default on, `=0` = old launches). One launch per layer for the four 64-row units instead of two `sliding_qkv_pair_direct_j` (sliding) or four `full_qkv_chunk` (full): every Q / K / V weight tile is streamed, decoded and loaded into the TRF once and contracted against the two 128-row waves (the `layer_*_t256p` pattern). The input RMSNorm runs once over the 256 rows (128 slices per cluster). The sliding post stage uses a two-table RoPE (cos, sin; the low half subtracts), exact because the host tables have cos_lo == cos_hi and sin_lo == -sin_hi, which keeps four post blocks inside the 8 KiB VRF. Static cycles: sliding 177,621 vs 2 x 96,936; full 201,647 vs 4 x 72,220. Arena: QKV per pass 12.02 + 3.36 -> 11.48 + 2.67 ms. Exact: 6/6 suites, 36/36 identical outputs, old path vs new path on the same binary.
2. **K03-256** (`S2_K03_256`, default on). When every row read after the last layer is a decode row (no prompt finishes in the pass), the last layer's AO + FFN runs as `layer_full_t16` / `layer_full_t8` on the decode rows only (the K03 of the 128-row engine). Used in about 76 % of the passes; `t_layer_f` 7.87 -> 7.32 ms per pass. Exact: 6/6 suites, 36/36 identical outputs.
3. **FP8 fold ON by default** (`S2_FP8`, unset = fold, `S2_FP8=off` = bf16). The fold code and kernels are the existing ones (`device/f8/ffn.rs`, `ops_f8`); only the default changed. The bf16 reserve is the existing default `S2_FP8_BF16_LAYERS = 0,1,5,11,17,23,29,35,41,46,47` (first two layers, all 8 full-attention layers, layer 46). **LOSSY.**

## Arena results (concurrency 18; the Arena performance test is short, 36 requests, so tok/s varies by about +-4 run to run; the per-pass timers are the stable numbers)

| Build | pass (256 rows) | t_layer_s | tok/s (short test) | accuracy |
|---|---|---|---|---|
| S2_203_tok (3 runs) | 76.2 ms | 36.9 ms | 166.0 | |
| + native QKV (4 runs) | 75.3 ms | 37.2 ms | 166.9 | 6/6, exact |
| + K03 (4 runs) | 75.3 ms (device -2 ms vs 203, same run class) | 37.1 ms | 162.4 | 6/6, exact |
| **+ FP8 = this folder** (3 runs, no env) | **65.5 ms** | **28.6 ms** | **184.0** | **6/6 pass** |

FP8 accuracy detail (no env, the graded default, binary md5 a05b60c081fc3941d31f60cda33cde83): mmlu, gsm8k, ifeval, vqav2 (C = 18 and C = 1), fleurs_en, fleurs_ko all pass; TPOT p90 69-73 ms. Compared with bf16 outputs: identical in mmlu, vqav2, fleurs_en; text differs in gsm8k (33/36), ifeval (36/36), fleurs_ko (5/36), all still passing.

**Risk.** An earlier fold policy failed one PRIVATE vqav2 case (submission 5111c2a6). This reserve policy has passed every public suite but was never graded on the private set. The exact alternative is the same tree with `src/host/fp8.rs` restored from `S2_203_tok` (native QKV + K03 only).

Estimate (not measured on the grader): S2_203_tok scored 203.42 officially; on the Arena its iteration is about 79.9 ms (pass 76.2 + about 3.7 ms outside the pass) against about 69.2 ms for this folder, so roughly 203.42 x 79.9 / 69.2, about 235 tok/s, if the grader scales like the Arena. For reference, the S2_211_tok notes report an Arena pass timer of 68.5 ms against 65.5 ms here. Full tables, job ids and the rejected experiments (reduce-scatter QKV epilogue, `layer_*_t256q`, shared k/v norm chain): `RESULTS_arena_qkv256.md`.

## Tools added to `scripts/stage2/`

- `qkv_arena.sh`: one Arena accuracy / performance job with a given server binary and extra environment variables (`--env "K=V ..."`), log saved to a directory. Arena quota: 2 active jobs per user.
- `compare_outputs.py A.log B.log`: same request, same generated text? (exactness check of a flag, old path vs new path on one binary).
- `perf_summary.py LOG ...`: tok/s, TPOT, TTFT and the per-256-row-pass device timers (`t_qkv_*`, `t_attn_*`, `t_layer_*`, ...) summed over all `S2ITER` windows.

Build note: Stage 2 needs `cargo-furiosa-opt` 0.8.1 from `cargo binstall cargo-furiosa-opt@0.8.1` (a plain `cargo install` of 0.8.1 gives a stub). A cold build of this tree takes about 30 minutes on the WSL machine used here.
