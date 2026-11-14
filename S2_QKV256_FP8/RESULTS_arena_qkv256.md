# Arena results: native T = 256 QKV (S2_QKV_opt) vs S2_203_tok

Date: 2026-10-10. Arena account TruongNVMM. Source: `2026-MICRO-NPU/S2_QKV_opt` (5 files changed vs `S2_203_tok`: `src/device/qkvc/quad.rs` new, `src/ops_qkvc.rs`, `src/host/runtime/hybrid256.rs`, `src/axes.rs`, `src/device/qkvc/mod.rs`).

| Binary | Build | md5 |
|---|---|---|
| baseline (S2_203_tok, unchanged) | `~/stage2/base`, cargo-furiosa-opt 0.8.1 | ee5a648272741c4ad9d2e03338b65a7e |
| new (native-256 QKV) | `~/stage2/qkv` | bd4c1320feed95e5eccc88c8eda9c81b |

## What changed
- Sliding layers: ONE `sliding_qkv_quad_direct` launch per layer (256 rows) instead of two `sliding_qkv_pair_direct_j` (128 rows each). Static 177,621 vs 2 x 96,936 cycles (-8.4 %).
- Full layers: ONE `full_qkv_quad` launch per layer instead of four `full_qkv_chunk` (64 rows each). Static 201,647 vs 4 x 72,220 cycles (-30 %).
- Input norm over the 256 rows on 128 slices per cluster (was 64 / 32).
- Sliding RoPE with 2 VRF tables instead of 4 (cos_lo == cos_hi and sin_lo == -sin_hi in the host tables; low half uses a subtract), needed to fit 4 post blocks in the 8 KiB VRF.
- Flags: `S2_QKV256=0` / `S2_QKVF256=0` restore the old launches (same binary).

## Exactness (same binary, old path vs new path, concurrency 18, 36 requests per suite)

| Suite | old path | new path | identical outputs |
|---|---|---|---|
| mmlu | 36/36 | 36/36 | 36/36 |
| gsm8k | 36/36 | 36/36 | 36/36 |
| ifeval | 36/36 | 36/36 | 36/36 |
| vqav2 | 36/36 | 36/36 | 36/36 |
| fleurs_en | 36/36 | 36/36 | 36/36 |
| fleurs_ko | 36/36 | 36/36 | 36/36 |

## Performance (performance test, mixed 5 inputs, concurrency 18, 36 requests; per 256-row pass)

| Run | tok/s | TPOT p50 | pass (hs256) ms | t_qkv_s | t_qkv_f | t_attn_s | job |
|---|---|---|---|---|---|---|---|
| base r1 | 170.33 | 75.07 | 75.25 | 11.99 | 3.35 | 9.20 | 204000 |
| base r2 | 162.92 | 77.88 | 76.78 | 12.01 | 3.37 | 10.34 | 204496 |
| base r3 | 164.65 | 75.35 | 76.49 | 12.06 | 3.36 | 9.19 | 204494 |
| new r1 | 167.69 | 77.35 | 75.35 | 11.47 | 2.66 | 9.94 | 204404 |
| new r2 | 172.33 | 73.21 | 73.80 | 11.46 | 2.66 | 9.22 | 204595 |
| new r3 | 165.63 | 72.88 | 74.59 | 11.51 | 2.70 | 9.93 | 204598 |
| **mean base** | 165.97 | 76.10 | 76.17 | 12.02 | 3.36 | | |
| **mean new** | 168.55 | 74.48 | 74.58 | 11.48 | 2.67 | | |
| **delta** | +1.6 % | -2.1 % | **-1.59 ms (-2.1 %)** | -0.54 | -0.69 | | |

Notes
- The Arena performance test is short (36 requests, ~13 s window), so its tok/s is not the official number (203.42 at a longer run) and run-to-run noise is about +-4 tok/s / +-1 ms per pass. The QKV device timers are stable (base 12.0 + 3.36 ms, new 11.47 + 2.67 ms in every run).
- Projected official throughput with the -1.6 ms pass: 203.42 x 76.3 / (76.3 - 1.6) ~ 207.8 tok/s (estimate, not measured).
- Arena quota: max 2 active jobs per user.

## v2 experiment: reduce-scatter GEMM epilogue (rejected)

Output of each row tile spread over all 256 slices (`[Gg % 8, Tc / 2]`) instead of 8 padded slices. Static cycles unchanged (sliding 178,011 vs 177,621; full 199,105 vs 201,647). Binary md5 d0212c22e44ed25a42a09229706e925c.

| Run | tok/s | pass ms | t_qkv_s | t_qkv_f | job |
|---|---|---|---|---|---|
| v2 r1 | 171.73 | 74.63 | 11.71 | 2.66 | 204725 |
| v2 r2 | 163.56 | 75.95 | 11.84 | 2.73 | 204748 |
| v2 r3 | 170.93 | 74.53 | 11.68 | 2.66 | 204750 |
| v1 r4 (same batch) | 161.82 | 77.39 | 11.63 | 2.76 | 204726 |

Sliding QKV is ~0.2 ms SLOWER with reduce-scatter (11.74 vs 11.52 ms mean), full unchanged, so v2 was dropped and the source reverted to v1 (md5 of `quad.rs` e4abdd51...). v2 exactness jobs were cancelled (not needed).

## v3: v1 + K03-256 (last layer AO + FFN on the decode rows only) — current best

When every row read after layer 47 is a decode row (no prompt finishes in the pass), the last layer's AO + FFN runs as `layer_full_t16` / `layer_full_t8` on the 16 / 8 decode rows instead of `layer_full_t256p` on 256 rows. Flag `S2_K03_256=0` = old path. Binary md5 cc7167c95e0a01d28f9ff86d25335fb8.

- Used in ~76 % of the 256-row passes (k03_on 54 / k03_off 17 per run).
- `t_layer_f` 7.87 -> 7.32 ms per pass (-0.55 ms), as expected.
- Exactness (same binary, S2_K03_256=0 vs default): mmlu, gsm8k, ifeval, vqav2, fleurs_en, fleurs_ko all 36/36 passed, 36/36 identical outputs (jobs 206006-206151).

| Run | tok/s | pass ms | t_qkv_s | t_qkv_f | t_layer_f | job |
|---|---|---|---|---|---|---|
| v3 r1 | 161.54 | 74.78 | 11.46 | 2.65 | 7.35 | 205960 |
| v3 r2 | 162.73 | 75.35 | 11.72 | 2.67 | 7.37 | 205962 |
| v3 r3 | 161.66 | 75.64 | 11.51 | 2.66 | 7.20 | 205982 |
| v3 r4 | 163.65 | 75.34 | 11.46 | 2.66 | 7.36 | 205985 |

All four v3 runs landed in the "~71 passes" run class, whose pass time is higher than the "~136 passes" class for every binary (also host timers t_tables / next_tokens were high), so tok/s of this short test is not comparable run to run. Device-only time per pass, same run class (~71 passes): base 73.4 ms, v1 71.9 ms, v3 71.5 ms (about -2 ms / -2.7 % vs base).

Rejected in the same round (static cycles only, not built): `layer_*_t256q` = AO weights decoded once for both halves + one h buffer per half: sliding 766,775 vs 771,562 (-0.6 %), full 849,383 vs 831,936 (+2.1 %); the AO + FFN kernel is Main-bound (Main 72 % busy), so reordering the DMA does not shorten it.

## v4: v3 + FP8 fold ON by default (LOSSY) — current candidate

`host/fp8.rs`: the FFN FP8 weight fold (e4m3 x e4m3 contraction, `device/f8/ffn.rs`) is on when `S2_FP8` is unset; bf16 reserve layers unchanged (`0,1,5,11,17,23,29,35,41,46,47` = first two, all 8 full layers, 46). `S2_FP8=off` = bf16. Binary md5 a05b60c081fc3941d31f60cda33cde83. User allowed lossy if the 6 suites pass.

| Run | tok/s | TPOT p50 / p90 | pass ms | t_layer_s | job |
|---|---|---|---|---|---|
| v4 r1 | 179.17 | 64.7 / 73.4 | 65.70 | 28.58 | qkv_v4_perf_c18_r1 |
| v4 r2 | 184.54 | 68.1 / 69.9 | 65.53 | 28.48 | qkv_v4_perf_c18_r2 |
| v4 r3 | 188.35 | 66.4 / 68.9 | 65.15 | 28.79 | qkv_v4_perf_c18_r3 |

Pass 75.3 -> 65.5 ms (-13 %), t_layer_s 37.1 -> 28.6 ms. Accuracy (no env, the graded default): mmlu, gsm8k, ifeval, vqav2 (C=18 and C=1), fleurs_en, fleurs_ko all pass. Earlier env run (S2_FP8=fold on v3): same results, vqav2 C=12 24/24; outputs differ from bf16 in gsm8k (33/36 differ), ifeval (36/36 differ), fleurs_ko (5/36 differ), identical in mmlu, vqav2, fleurs_en.

Risk: an earlier fold policy failed one PRIVATE vqav2 case (submission 5111c2a6); this reserve policy was never graded on the private set.

Projection anchored on S2_203_tok (203.42 official, Arena iteration about 79.9 ms): iteration about 69.2 ms, about 235 tok/s (estimate). (An earlier version of this file anchored on 211.1 assuming it was v1; the 211.1 submission is S2_211_tok, a different tree.)

Submission folder `2026-MICRO-NPU/S2_QKV_submit` = S2_203_tok + the 6 changed sources of v4 (not submitted).

Logs: `qkv_base_perf_c18_r{1,2,3}.log`, `qkv_new_perf_c18_r{1,2,3}.log`, `qkv_new_acc_<suite>_{old,new}_path.log`.
