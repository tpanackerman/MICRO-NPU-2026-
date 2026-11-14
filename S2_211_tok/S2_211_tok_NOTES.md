# S2_211_tok: Stage 2, official score 211.1379 generated tokens/s (submission 5cc440be, 2026-10-09 UTC)

**What this folder is.** A snapshot of the Stage 2 version that is on the leaderboard now (EDABK-Artemis, 211.14 tok/s, concurrency 18). It has the organisers' baseline files (README, SERVING, scripts, Cargo files, all the same as in `S2_203_tok`) plus the `src/` that was submitted (151 files: the 146 uploaded files plus `src/api`, which is the organisers' own and is replaced at grading). Every file in `src/` matches, by sha256, the manifest recorded when the submission was sent. The Stage 1 folders, `S2_163_tok` and `S2_203_tok` are separate.

**Source.** Built on `S2_203_tok` (s2-hyb-t256c-b8, a08e841) in this order: t256r (commit d421bd4), the fused sliding layer at load-time build (f2, commit 73283d7), and then three host-side TPOT changes (working tree `opt203_tpot`, capacity 18). Toolchain furiosa-opt 0.8.1. The local build of this source gives server sha256 `a5fb0383…cc963a`, and all 79 kernel `.bin` are byte-identical to the f2 build, so the third step changes host code only. Submitted with `moa-submitter submit --source <this src set> --concurrency 18` (submission **5cc440be**, 146 files, 3 of 4 used in the 24 h window). To submit this folder again (WSL, only on purpose): `export PATH=$HOME/.cargo/bin:$PATH; moa-submitter submit --source <this folder> --concurrency 18`.

## What changed since S2_203_tok (all exact, FP8 weight fold still OFF by default)

1. **t256r** (`S2_AOPAIR`, default on). o_proj runs all 256 rows of a hybrid pass in one contraction per weight tile. New kernels `layer_{sliding,full}_t256r`. Static cycles: sliding layer 771.6k → 719.2k, full layer 831.9k → 801.7k. Output matches 203 character by character on 6/6 public suites. Arena pass time 76.05 → 73.33 ms.
2. **Fused sliding layer** (`S2_FUSED`, default on, `S2_FUSED=0` turns it off). `ops_fused::sliding_layer_fused_b16` runs QKV, attention, AO and FFN of one sliding layer in a single launch. It takes 21 arguments: weights grouped along Wkv/Wug/Hi3, a `[Sw]` blob for the small weights, and a `[Tb]` table region. Static 1,156k cycles, image 704 KB. On HW it takes 1.264 ms per launch, against 1.426 ms per layer before fusing. The aim is fewer kernel-image reloads: the firmware keeps only the previous launch's image, so five sliding layers in a row now share one image. The fused path is used only when a pass has exactly 16 decode rows and the layer is not FP8-folded. The rest goes through the t256r path. The grouped operands are built once, at weight load. The per-kernel weights are views of the grouped weights (no duplicate HBM copy), and the sliding tables are views of the `[Tb]` region (Tpk256 = 3,112,960). Arena timer per pass: 72.5 → 68.5 ms (3/3). Output matches t256r character by character on 6/6 suites.
3. **Host TPOT transport** (no kernel change):
   - `src/host/runtime/decode.rs` (`S2_DECODE_PACKED`): `DecodeEngine::run` / `run16` send embeddings, RoPE and attention tables in one packed upload instead of 11 / 21 separate H2D copies. Each tensor is an 8 KiB-aligned view into one reused buffer.
   - `decode.rs` + `runtime.rs` (`S2_DECODE_N2`; `S2_N2=0` also turns it off): the hidden state of the last decode round stays on the NPU, and the LM head reads it there. Earlier groups are still copied back before their scratch is reused, and the existing flush handles the paths that need logits on the CPU.
   - `src/host/runtime/hybrid256.rs` (`S2_HEAD_PACK`): when a prompt finishes, the first decode group of 8 rows stays on the device. Example: 15 decode rows + 1 prompt row now need one LM-head launch (8 device + 8 host) instead of two (7 + 9).
   - To turn off all three for an A/B: `S2_DECODE_PACKED=0 S2_DECODE_N2=0 S2_HEAD_PACK=0 ./server`.

## Official result (organisers' evaluation, submitted 2026-10-09 21:41 UTC, evaluated from 22:06 UTC)

Accuracy 32 / 32 cases passed (2 rounds). Performance at concurrency 18, 18 warmup + 270 measured requests, repeat 3:

| | |
|---|---|
| Throughput | **211.14 tokens/s** (3.31 requests/s) |
| TTFT p50 / p90 | 679.2 / 1077.3 ms (gate 4.0 s) |
| TPOT p50 / p90 | 72.32 / 79.59 ms (gate 100 ms) |

| Input type | TTFT p50 / p90 (ms) | TPOT p50 / p90 (ms) |
|---|---:|---:|
| 256 text | 507.6 / 937.2 | 72.51 / 79.98 |
| 1024 text | 567.1 / 1070.1 | 71.98 / 77.25 |
| 2048 text | 860.8 / 1236.1 | 71.77 / 78.50 |
| 512 image | 638.8 / 1144.7 | 73.19 / 80.64 |
| 768 audio | 616.1 / 1055.0 | 72.38 / 79.59 |

**Grader runs in this series:**

| Submission | Version | Score (tok/s) | TTFT p90 (ms) | TPOT p50 / p90 (ms) |
|---|---|---:|---:|---:|
| 49c88d1b | S2_203_tok | 203.42 | 1154 | 76.3 / 79.3 |
| 3ef19639 | fused f1 (operand build in the first pass) | 202.53 | 1774 | 74.4 / 79.8 |
| 5aef8dff | fused f2, `--bin` trial run (not on the board) | 205.91 | 1096 | 74.9 / 78.6 |
| **5cc440be** | **f2 + host TPOT (this folder)** | **211.14** | **1077** | **72.3 / 79.6** |

## Arena evidence for the host TPOT step

The same binary was run as A (all three changes off) and B (all on). Mixed, 36 requests, C=18, 64 tokens per request. Each run was a separate job, so the runs were not guaranteed to land on the same NPU.

| Run | TPOT p90 (ms) | Throughput (tok/s) | Job |
|---|---:|---:|---:|
| A1 | 75.957 | 171.302 | 200954 |
| B1 | 71.570 | 186.503 | 200957 |
| B2 | 73.586 | 180.787 | 200962 |
| A2 | 72.461 | 178.805 | 200963 |
| A3 | 73.691 | 178.134 | 201027 |
| B3 | 74.329 | 182.952 | 201028 |
| **Mean A / B** | **74.036 / 73.162** | **176.080 / 183.414 (+4.16 %)** | |

Profiler, A → B per pass: decode-kernel wall time −18.3 % on average, H2D 5.1–7.1 → 0.7–1.4 ms, hidden D2H 0.9–3.0 → 0 ms, and LM-head launches per iteration 1.08–1.11 → 1.00. With C=1 and a 256-token prompt, TPOT p90 went 30.50 → 26.98 ms with identical output. Checks: public accuracy 12/12 on all six suites (job 200970); multi-group with `S2_DECODE_ONLY_CAP=0` 36/36 (200971); `GEMMA4_ARGMAX=0` host-argmax flush 8/8 (200982).

## Honest open points

1. The Arena gain (+4.2 %) only partly reached the grader: +2.5 % over the f2 trial run. Arena runs 36 requests with no warmup, while the grader runs 270 after warmup.
2. Mixed C=18 output is not token-identical between Arena runs, even A against A (7 of 36 requests differ). The cause is not found yet, so token identity is claimed only for the C=1 check and the per-suite accuracy runs.
3. About 78 % of the iteration time is still in the hybrid kernels (`HybridEngine256::run`). `MAX_RIDERS = 16` caps how many decoders ride each pass, so with 18 sessions some decoders wait a pass. Neither was changed here.
4. A latency preset (`S2_P256_MIN=257 S2_DEC_CAP=16`) gave public TPOT p90 51.9 ms but −19.7 % throughput (jobs 201046 / 201048). It is not the default.
5. The C=20 changes (5 axes + `CAPACITY_256`) are NOT in this version.
