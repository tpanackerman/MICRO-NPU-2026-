# R10 fin7 RESULTS (integrator; written at the top; PROVISIONAL, updated as items close)
STATUS (13:45): source prepared and committed for both stacks, layer-level proofs for Stack B DONE, NO post-KVD binary yet (WSL crash loop, see PROGRESS.md CURRENT STATE); gates G2/G4/rule 3/safety NOT run on the final binaries. Nothing here is a validation of the stacks yet.

## Stacks (source)
- Stack A `fin7` (git stage2/fin7, branch fin7, commit 55e023e) = fin6 (a6c26dc) + G (layer64k_G_vs_fin6.patch md5 f318f2f8) + cpuplace_harden (cpuhost, md5 1a224dc0) + KVD (kvcache/patch/kvd_on_fin6.patch md5 3cb4ab2a). Compliance 0 hard / 0 warnings. All three patches `git apply` cleanly in order (patches in fin7/patch/).
- Stack B `fin7e2` (git worktree stage2/fin7e2, branch fin7e2, commit 4ad25cb) = Stack A with E10 removed: `git apply -R actdma/patches/actdma_e10_on_e2.patch` applies cleanly on fin6 and leaves ffn.rs, ffnc/mod.rs, host/packed.rs, ops_ffnc.rs BYTE-IDENTICAL to v1f2's (cmp), i.e. the FFN-down layout/numerics are v1f2's (= v1d bit-exact E2). G was ported by hand to the E2 `ffn_core` (same mquad / ffu_tile_sp_into / geglu_pair_sp code as in A). Compliance 0 / 0.
## Stack B layer-level proofs (done; Arena bench jobs b1 = 157401 host 6be072276275, b2 = 157434 host 4f396b05414d; bench built from fin7e2 pre-harden/pre-KVD source, the layer kernels are identical to the final tree)
- (a)+(b): out_hash of the production layer kernels (E2 layout + G) on the actdma fixture equals the v1d production hash in all 8 buckets: s128 5a741cf934d3261c, f128 d961cd274b89f179, s64 4b842cf24c989609, f64 1a4a03fa5c97c013, s16 364d3a69d24e5377, f16 316bb8fdf3bd1bae, s8 69f862f6db83627f, f8 847335665fb28cff. So B's layer is bit-identical to v1d / v1f2 and G is bit-identical on the E2 layout.
- (c) K03: rows 120..127 of layer_{sliding,full}_t128 == layer_*_t8 fed those 8 rows, 3 reps each, ALL_IDENTICAL true (sliding hash fd30dfbe4b1cdf66, full 9b697474c6086f65): K03 stays enabled in B.
- Bench cycles (host 6be072, E2 + G): s128 960,734, f128 1,041,318 (v1d production ~988k / ~1,062k on the same class of board: E2 gain, G adds ~0 here).
## Gates (pre-registered in PROGRESS.md) status
G1 compliance: A 0/0, B 0/0 (source). G2: pre-KVD/pre-harden builds BUILD_RC=0 (A 9cc8590a9e6d5a07d51d9fb7fc457bac 523 s, B f84d78f8d1a5e767b3e1a75bc35e0bd2 426 s, both warm) - SUPERSEDED, final-source builds pending. G4, G5, rule 3, safety re-check (4 items: mmlu.008619, mmlu.009244, vqav2.001253, ifeval.000347) and the full unselected-suite confirmation on B: NOT RUN on final binaries.

# FIN6 RESULTS (hostdiet R9, tree stage2/fin6; written at the top, older v1f text below)
STATUS: **FIN6 READY** (03:31). Every pre-registered gate passed; candidate to replace fin5 (s2-hyb-final1) for the single submission. NOT submitted, NOT pushed.

## MEMBERS, MD5, GATES (summary)
- Members: fin5 (v1f branch fin5 b5b3c4e = v1f4 + enc-lazy + rule5 eosmask + Part A + lmorder O2 + kern2 hand-expanded tiles + stage-pad) + kern2 `order_on_v1f4.patch` (ops_rpa.rs, N3 decode scatters + decode attention before the prefill appends) + hostdiet `cpuplace.patch` (new src/host/cpuplace.rs; hooks in host/mod.rs, host/worker.rs, host/runtime/hybrid.rs prebuild spawn line, mm_norm.rs). Git: stage2/fin6 (base 0569c7a = `git archive fin5`), tag `s2-hyb-fin6`; src only in stage2/_submit_fin6.
- Server md5 **1cac1f60442fdace8157b20ef0cd9482** (`~/fin6_server`, run tree `~/fin6_run` -> `~/hd_run_fin6`); fin5 = be6da09e64b09d52ba614b61a0713fc4.
- G1 compliance 0 hard / 0 warnings. G2 upload-set build BUILD_RC=0, **warm AND cold builds md5-identical (1cac1f60 both; cold 476 s)**. G4 12/12 with output text identical to fin5 and v1f3 (below). G5 every one of the 34 perf jobs 96/96, max TTFT p90 3332 ms, max TPOT p90 57.9 ms. Mixed and text-only: tables below.

## BINARY vs BINARY (slow class = cgroup cpu.max 2 CPUs; hosts 750332 8e6a4e b29838 d00c99; fast host 5a7ead excluded from the statistics)
Own interleaved pairs (fin6 `m6_f6*` / `t6_f6*` on `~/hd_run_fin6`, fin5 arm y `m6_y*` / `t6_y*` on `~/hd_run_fin5`, one at a time or two in parallel, alternating order) plus fin5 jobs imported READ-ONLY from stage2/v1f/runs (y6m_y*, y6t_y*) and v1f5e jobs from stage2/rule5/runs (r5m_n*, r5t_n*, md5 f634261d). `scripts/cmp.py` (Welch t interval and a host-stratified bootstrap over job cells, which is optimistic at 1-3 jobs per cell: the Welch interval is the one to read).
```
## mix
fin6: N=6 tok/s 158.43 (sd 1.89) iter 48.08 thr 0.20 cores 1.39 sqp 0.20 TTFT90 max 2062 TPOT90 max 57.0 hosts [('750332', 1), ('8e6a4e', 1), ('b29838', 1), ('d00c99', 3)]
fin5 own: N=9 tok/s 154.98 (sd 3.64) iter 49.12 thr 0.62 cores 1.92 sqp 0.97 TTFT90 max 2078 TPOT90 max 57.9 hosts [('750332', 2), ('b29838', 4), ('d00c99', 3)]
fin5 imported: N=8 tok/s 155.31 (sd 1.27) iter 49.23 thr 0.77 cores 1.95 sqp nan TTFT90 max 2087 TPOT90 max 61.1 hosts [('750332', 2), ('8e6a4e', 2), ('b29838', 1), ('d00c99', 3)]
vs fin5 own pairs: Welch +2.22 % CI [+0.20, +4.25]; host-stratified +1.70 % CI [+0.04, +3.43] cells {'d00c99': (3, 3), '750332': (1, 2), 'b29838': (1, 4)}; iteration -1.04 ms
vs fin5 own+imported: Welch +2.12 % CI [+0.70, +3.55]; host-stratified +1.96 % CI [+1.08, +2.77] cells {'d00c99': (3, 6), '8e6a4e': (1, 2), '750332': (1, 4), 'b29838': (1, 5)}; iteration -1.10 ms
v1f5e (rule5 imported): N=15 tok/s 150.91 (sd 2.89) iter 50.06 thr 0.65 cores 1.91 sqp nan TTFT90 max 2266 TPOT90 max 63.3 hosts [('750332', 4), ('8e6a4e', 4), ('b29838', 2), ('d00c99', 5)]
fin6 vs v1f5e: Welch +4.98 % CI [+3.47, +6.50]; host-stratified +5.05 % CI [+4.02, +6.12] cells {'d00c99': (3, 5), '8e6a4e': (1, 4), '750332': (1, 4), 'b29838': (1, 2)}; iteration -1.98 ms
fin5 (own+imported) vs v1f5e: Welch +2.80 % CI [+1.46, +4.14]; host-stratified +2.88 % CI [+1.73, +4.10]
```
Decision rule (pre-registered 02:31): mixed point >= +0.5 % AND CI lower bound > -0.6 % (less favourable of Welch / stratified), N >= 6 per arm: fin6 N=6 slow, fin5 N=9 own + 8 imported; vs fin5 own+imported Welch **+2.12 % [+0.70, +3.55]**, stratified +1.96 % [+1.08, +2.77]; own pairs only Welch +2.22 % [+0.20, +4.25], stratified +1.70 % [+0.04, +3.43]: PASS.
```
## txt
fin6: N=6 tok/s 133.75 (sd 2.49) iter 47.85 thr 0.21 cores 1.34 sqp 0.24 TTFT90 max 3332 TPOT90 max 50.1 hosts [('750332', 3), ('8e6a4e', 2), ('d00c99', 1)]
fin5 own: N=5 tok/s 132.33 (sd 2.45) iter 48.56 thr 0.66 cores 1.91 sqp 0.97 TTFT90 max 3069 TPOT90 max 53.2 hosts [('750332', 3), ('b29838', 1), ('d00c99', 1)]
fin5 imported: N=5 tok/s 129.60 (sd 3.09) iter 49.07 thr 0.55 cores 1.91 sqp nan TTFT90 max 2952 TPOT90 max 53.4 hosts [('750332', 1), ('8e6a4e', 1), ('b29838', 3)]
vs fin5 own pairs: Welch +1.08 % CI [-1.53, +3.68]; host-stratified +1.23 % CI [-0.31, +2.88] cells {'750332': (3, 3), 'd00c99': (1, 1)}; iteration -0.71 ms
vs fin5 own+imported: Welch +2.13 % CI [-0.18, +4.44]; host-stratified +2.05 % CI [+1.03, +3.11] cells {'750332': (3, 4), 'd00c99': (1, 1), '8e6a4e': (2, 1)}; iteration -0.97 ms
v1f5e (rule5 imported): N=10 tok/s 129.92 (sd 1.62) iter 49.33 thr 0.61 cores 1.91 sqp nan TTFT90 max 3211 TPOT90 max 52.3 hosts [('750332', 2), ('8e6a4e', 4), ('b29838', 2), ('d00c99', 2)]
fin6 vs v1f5e: Welch +2.95 % CI [+0.88, +5.01]; host-stratified +3.11 % CI [+2.57, +3.66] cells {'750332': (3, 2), 'd00c99': (1, 2), '8e6a4e': (2, 4)}; iteration -1.48 ms
fin5 (own+imported) vs v1f5e: Welch +0.80 % CI [-0.96, +2.57]; host-stratified +0.61 % CI [-0.76, +1.97]
```
Text-only (N >= 4 per arm, point >= -0.3 %): fin6 N=6 slow; vs fin5 own+imported Welch **+2.36 % [-0.02, +4.75]**, stratified +2.41 %; own pairs only +1.33 % / +1.41 %: PASS (point estimates positive). vs plain v1f5e (the question of 03:20): fin6 text-only +2.95 % [+0.88, +5.01] (stratified +3.11 %), fin6 mixed +4.98 % [+3.47, +6.50] (stratified +5.05 %); fin5 vs v1f5e text-only -0.01 % / mixed +2.80 %: cpuplace lifts the text-only stream (about +2.4 % over fin5) as well as the mixed one (+2.1 %).
Mechanism witnesses (slow class means): fin6 poller 0.20 core, throttled-period fraction 0.20, container 1.39 cores; fin5 poller 0.97, throttled 0.62, 1.92 cores (text: fin6 0.24 / 0.21 / 1.34 vs fin5 0.97 / 0.63 / 1.91); iteration -1.0 to -1.3 ms. Fast host 5a7ead (cpuplace not engaged, `cpu.max` 11 CPUs, poller 1.0 core): fin6 mixed 168.3 / 168.3 / 168.7 tok/s, text 141.7 / 141.3 vs fin5 text 141.1 / 140.9 / 141.1 (no mixed fin5 job landed there): neutral, no cost.
Job ids (label=id): m6_f61=155084 m6_y1=155097 m6_y2=155103 m6_f62=155114 m6_f63=155119 t6_f61=155129 m6_y3=155135 t6_y1=155140 m6_y4=155142 t6_y2=155147 m6_f64=155153 t6_f62=155156 m6_f65=155164 t6_f63=155169 m6_y5=155173 t6_y3=155178 m6_y6=155183 t6_y4=155188 m6_f66=155191 t6_f64=155195 m6_f67=155198 t6_f65=155201 m6_y7=155202 t6_y5=155206 m6_y8=155209 t6_y6=155213 m6_f68=155214 t6_f66=155216 m6_f69=155217 t6_f67=155223 m6_y9=155225 t6_y7=155230 t6_f68=155235 t6_y8=155242
G4 jobs: see below. Cold build log runs/fin6_cold.out, warm build runs/fin6_build.out, per-job logs runs/*.log, ledger lines in PROGRESS.md.

## G4 (accuracy, six suites, c1 and c12) on fin6 (md5 1cac1f60): PASS 12/12
Jobs (all SUCCEEDED, ledger in PROGRESS.md): c1 mmlu 154972 (ran with 8 requests, 8/8), gsm8k 154987, ifeval 154997, vqav2 155010, fleurs_en 155015, fleurs_ko 155019 (2/2 each); c12 mmlu 155034, gsm8k 155042, ifeval 155050, vqav2 155061, fleurs_en 155067, fleurs_ko 155073 (24/24 each). (g4_f6_c1_gsm8k first job 154973 was an Arena 70 s timeout caused by my runner exporting S2_ROUNDS=8 to accuracy jobs, not a server failure; re-run 154987 passed.) OUTPUT TEXT IDENTITY: every recorded request equals fin5's (v1f a_y1_* / a_y12_*: c1 12/12, c12 144/144) and v1f3's (a_p1_* / a_p12_*: 12/12, 144/144) character for character (`scripts/outcmp.py`).

## Step 2: code reading, one request is never in both the decode part and the prefill part of a hybrid pass (what the order patch relies on)
Read in src/host/worker.rs (the `loop` of `Worker::run`, lines 408-520) and src/host/generate.rs:
1. `decode` = sessions with `generation.is_decode()` (`matches!(phase, Phase::Decode(_))`); `fast_prefill` = sessions with `session.fast && generation.is_prefill()` (`matches!(phase, Phase::Prompt(_))`). `Phase` is ONE enum field of the session, so a session satisfies at most one of the two predicates; `fast_decode` is a subset of `decode` (cap truncation only removes rows).
2. `plan_hybrid(&fast_prefill, fast_decode.len())` returns chunks only of `candidates.first()` and `candidates.get(1)`, two DIFFERENT elements of `fast_prefill`, and `fast_prefill` has one element per session (built from `sessions.iter().enumerate()`), so the two prefill runs belong to two different sessions; every chunk's session is in `fast_prefill`, never in `fast_decode`.
3. The phase is only changed after the pass (`finish_chunk` / `consume_prompt` / the decode update at lines ~430-522 of generate.rs, called from the post-pass loop), never between planning and `hybrid_step`; `sessions` is not mutated inside the pass; a request whose last prompt chunk is in this pass turns decode only in the NEXT iteration (its first decode row appears in the next plan).
4. A session's sliding-cache slot is `blocks.sliding`, popped from the free list `rings` at admission and returned only when the session finishes, so two live sessions never share a slot; the decode scatters / decode attention of the N3 launch touch only the slots of `fast_decode` sessions, the prefill appends / attention only the slots of `chunks[..n_hybrid]` sessions: the two slot sets are disjoint, so reordering the scatters + decode attention ahead of the prefill appends cannot change any real cache row or output. The shared rows are the scratch row (padding rows of the decode bucket and of the unit tail append to the scratch slot, row 20480), which per kern2's patch note no real gather index selects (I verified the slot disjointness by code reading, NOT the kernels' index tables, so the scratch-row statement is kern2's); the HW identity of the whole mechanism is what the G4 / token-identity runs below test.
Verdict: the precondition of the order patch holds by construction of `plan_hybrid` / `Phase`.

## Compliance (G1) and disclosure
`bash tools/compliance_check.sh stage2/fin6 baseline` = 0 hard failures, 0 warnings (the old `ignore_eos` WARN is gone in fin5/fin6: `ignore_eos` is confined to src/bin/gemma4.rs and src/host/generate.rs). cpuplace disclosure: `src/host/cpuplace.rs` declares `unsafe extern "C"` libc calls (`sched_setaffinity`, `sched_getaffinity`, `sched_getcpu`, `setpriority`; no libc crate). It acts only when the container's cgroup `cpu.max` quota is at most 2.5 CPUs (read from /sys/fs/cgroup/cpu.max; unreadable or "max" = nothing happens), skips (and undoes where it matters) every failing call, and never reads any request field, prompt, output or flag; it only places our own threads and the runtime's kernel submission poller (nice 19, same CPU as the launch thread). New env switch `S2_CPUPLACE` (default ON = validated path, 0 = off).

# hyb RESULTS (task HYB: hybrid single pass + native-T=128 attention; tree `stage2/hyb`; tags `s2-hyb-v1` and `s2-hyb-v1b` = internal checkpoints of the best validated state, NOT final submissions)

## s2-hyb-v1b (2026-10-05) = v1 + host-table prebuild (N1) + drift MASK_PEN (stale-slot fix); binary h13 (`~/hyb_h13_server`, md5 e5922753...)
What changed w.r.t. v1 (everything below is default, no env var):
- **N1 prebuild** (`S2_PREBUILD`, default ON, 0 = old path): `worker::predict_next` predicts the next hybrid pass (decode riders +1 position, finished-by-length requests removed, prompts ending in this pass become decode rows, then the same
  decode-cap + `plan_hybrid` on that shadow state) and passes a `PassDesc` (slots, flat page tables, positions: no token ids) to `HybridEngine::run`, which builds ALL attention / rope tables of that pass and packs them (plus the embedding rows)
  into ONE upload buffer on a thread while the device runs the current pass; the next pass uses the result iff its real `PassDesc` is equal (else it builds synchronously: the old cost). Prediction hit rate on Arena: 100 % (240/240, 266/266, 201/201,
  0 misses). Host-serial time per pass (`scripts/hostgap.py`: iteration minus the device launch timers): tables + upload 2.2-3.3 ms -> 0.5-1.0 ms (same-kind nodes), host-serial total ~6.3 -> ~4.7 ms of ~52 ms.
- Why packing alone did not pay in v1 (h7/h8: jobs 148111/148114): the per-element byte conversion of the masks cost 1.0 ms per 262144 elements on the worker (`test_int --upbench`, job 148253); `scalar_bytes` removes it. A call to `to_hbm` costs a fixed ~55 us,
  so 25 small uploads were ~1.5 ms of call overhead; one 2.9 MB upload is 0.4-0.5 ms. Pinned host memory does not exist on the Arena worker (no dma-heap): not usable.
- **drift MASK_PEN** (`drift/port/hyb_penalty_nohybrid.diff` applied, hybrid.rs hand-merged): the attention max no longer depends on stale ring / page rows, so a request's output is independent of the slot it got and of the previous occupant
  (drift measured 7 of 12 prompts diverging before, 0 of 12 after; cost zero). New per-key penalty tables `pen` next to the masks (Tpk grown to 1462272 elements = 2924544 B, worst case 2916416 B).
- Timers added: `h_embed`, `w_plan`, `w_apply`, `h_tab_wait`, counters `h_prebuilt_hit` / `h_prebuilt_miss` in S2ITER.

| item | result |
|---|---|
| accuracy c1, 6 suites | PASS (jobs 148298, 148301, 148308, 148315, 148321, 148325) |
| accuracy c12, 6 suites x 24 requests | PASS (jobs 148327, 148330, 148333, 148335, 148337, 148338) |
| test_int `--hyb 12` (h11, jobs hs_h) | mixed 11/12, alone 10/12 identical to legacy; the two flips are bf16 near-ties (htf_d, job 148278: margins 0.125 -> 0.000) |
| test_int ring wrap / 1351-token prompt (h13, hs_i) | no fault; 4/5 identical (the same near-tie request) |
| perf C = 10 / 12 / 14 (wave wi, 80 / 96 / 98 requests) | 129.4 / 133.4 / 132.1 tok/s, TTFT p90 1477 / 2306 / 3601 ms, TPOT p90 62.7 / 66.0 / 66.3 ms, all PASS (slow nodes; see below) |
| paired h9 vs h13, C = 12 x 96, three runs each (wave w10) | h9: 128.5, 130.1, 145.3 tok/s (mean 134.6); h13: 135.3, 149.8, 134.4 (mean 139.8, +3.9 %); node noise +-8 %, iteration mean 56.3 -> 51.9 ms |
Recommended submit C stays **12** (C = 14 passes but its TTFT p90 3.6 s leaves little margin; C = 10 is under-loaded).

### Submission checklist (run before every candidate)
1. `bash scripts/hyb_srconly_build.sh` (in WSL): baseline `Cargo.toml` + `rust-toolchain.toml` + the BASELINE `src/api` + this `src/`, `cargo furiosa-opt build --release` of EVERY bin of `src/bin` (gemma4, server, test_kernels). This is the compile check of the upload set
   and of the interfaces `src/api` imports (`host::worker::{Worker, CONTEXT}`, `host::generate::{GenerationRequest, GenerationOutput, FinishReason, self}`, `host::image::{self, ImagePatches}`, `host::audio::{self, AudioFrames}`,
   `host::sampling::SamplingConfig`, `host::tokenizer::Tokenizer`, `host::load`). Result for s2-hyb-v1b: BUILD_RC=0 (563 s, all kernels recompiled), server md5 e59227530e2c... = the validated h13 binary bit for bit.
2. `diff -rq ../baseline/src/api src/api` must print nothing; `diff ../baseline/Cargo.toml Cargo.toml` must print nothing; no `cfg(feature` in `src/`.
3. Test-only code lives OUTSIDE `src/bin`: `tools_bins/test_int.rs` (the dev build scripts copy it into the build copy); `src/bin` holds only the baseline's gemma4 / server / test_kernels.
4. No env var is needed: every `S2_*` / `GEMMA4_*` knob has a validated default (prebuild on, dec-cap 8, pack-swap on, capacity 16, chunk 128, budget 128, views on).
5. Accuracy 6/6 on the final binary with default env (c1 and c12) and one C = 12 perf run (`scripts/hyb_pair.sh`, `scripts/acc_f.sh`, `scripts/sweep_f.sh`).


## s2-hyb-v1 (earlier checkpoint, kept for reference)

Base = `int` s2-best-v1 (commit 179f360). Final-candidate binary = **h9** (`~/hyb_h9_server`, md5 853c2671...; default env, no env var needed).

## What the tag contains
- **Hybrid pass** (`src/host/runtime/hybrid.rs`, `Model::hybrid_step`, `worker::plan_hybrid`): ONE T=128 row batch per iteration through all 48 layers; rows `[0, 128-Bd)` = prefill runs (<= 2 units of 64 rows), rows `[128-Bd, 128)` = decode rows
  (Bd = 8 or 16). The weights stream once for prefill and decode together; the separate decode pass is gone while a prefill is pending. Attention = ragged RPA kernels (prefill pair + fused KV append) + the existing fused decode
  attention on the decode rows; every kernel argument is a typed view of a slice of the 128-row tensors (`host/views.rs`), so there are no copy kernels and no second kernel image. Prefill writes the flat KV caches directly (no migration).
- **RPA layout fix** (also fixes a latent bug of int): sliding ring 1280 rows per slot, full pool with TWO scratch pages (zero page read by unallocated logical pages, trash page written by padding rows).
- **SOFTMAX_EPS = 1e-30** in every attention softmax denominator (chunk, decode, RPA kernels; the drift fix).
- **ffndec fused NVFP4 decode** (`up_fz` / `dn_fz`), bit-identical to the old FFN, +4 % at C = 10.
- **Policy knobs** (one binary): `S2_DEC_CAP` (default 8: at most 8 decode rows are put in a hybrid pass, 0 = off), `S2_PACK_SWAP` (default on: operand-swap row packing), `S2_PACK_UPLOADS` (default OFF, see below),
  `S2_CAPACITY` default 16, `S2_CHUNK` 128, `S2_PREFILL_BUDGET` 128.
- D2H slim (only the decode rows / the last rows of the residual come back).

## Defaults and the recommended submit concurrency
- **Recommended C = 12.** C = 10 is under-loaded (norm 116), C = 12 and 14 are flat (norm 125 / 123); C = 12 has the best TTFT margin (pooled p90 2.4 s) and TPOT p90 66 ms; C = 14 also passes (TTFT p90 3.2 s) but the margin to the 4 s gate is thin
  under node noise. Throughput saturates because the stream is prefill-bound (tok/s ~ 64 * (rows per iteration / iteration time) / 922 rows per request), so a larger C only adds queueing.
- Defaults (no env): `S2_DEC_CAP=8`, `S2_PACK_SWAP=1`, `S2_PACK_UPLOADS=0`, capacity 16, chunk 128, budget 128. `S2_CAPACITY=8` LOSES (jobs 148061 / 148065: norm 104.6 / 102.6): do not cap sessions.
- `S2_PACK_UPLOADS=1` (one packed upload per pass, views of one buffer) is token-identical but NOT faster (jobs 148111 / 148114): packed 3.99 ms vs separate 3.70 ms per pass for h_tables + h_h2d. Default stays off.

## Run table (Arena, mixed stream, default env, h9)
`norm` = tok/s x mean iteration ms / 56 (environment-normalised; Arena nodes differ by +-10 % in host speed). Per-type TTFT p50/p90 from `scripts/ttype.py`.
| job | C | requests | tok/s | TTFT p50/p90 ms | TPOT p90 ms | iteration ms | norm | result |
|---|---|---|---|---|---|---|---|---|
| 148194 | 10 | 80 | 139.2 | 903 / 1194 | 58.5 | 46.8 | 116.2 | PASS |
| 148196 | 12 | 96 | 129.5 | 1631 / 2425 | 66.6 | 54.1 | 125.2 | PASS |
| 148199 | 14 | 98 | 127.6 | 2626 / 3183 | 69.5 | 54.2 | 123.4 | PASS |
Earlier h6 pairs at C = 12 / 14 (same code except the upload default): tok/s 143.9 / 143.2 (fast node), TTFT p90 2359 / 2826-2955, TPOT p90 66-67 (jobs 148018, 148031, 148037).
Baseline (int s2-best-v1): C = 10: 81.8 tok/s, TTFT p90 3469, TPOT p90 94.8 (job 147753); C = 12: 84.2 tok/s, TPOT p90 104.3 = FAIL (job 147754). So the hybrid pass is +57 % at C = 10 and moves the usable C from 10 to 12-14.

Per-type TTFT p50/p90 (ms) of the same runs:
| C | 256 | 1024 | 2048 | image | audio |
|---|---|---|---|---|---|
| 10 | 688 / 847 | 903 / 2676 | 1020 / 1289 | 1033 / 4056 | 795 / 3473 |
| 12 | 1481 / 1529 | 1362 / 2425 | 2105 / 2514 | 2085 / 5162 | 1625 / 3288 |
| 14 | 2419 / 2666 | 2216 / 2881 | 2914 / 3275 | 2638 / 6402 | 2682 / 3280 |
The per-type p90 of image / audio / 1024 is the **t=0 burst** of the load generator (ThreadPoolExecutor of C workers: the first C requests are all dispatched at t = 0, the prefill rows of the burst are ~ C x 922, the last request of the burst waits for all of them:
C = 12 -> ~ 5 s). Without the first C requests (`scripts/waves.py wg_c12`) the C = 12 steady state is image p50/p90 2079 / 2138, 2048-text 2105 / 2299, audio 1613 / 1911, 1024 1355 / 1602, 256 1483 / 1529: image is NOT a laggard in steady state
(enc_ms is 17-57 ms per image request, ~2 % of its TTFT). The pooled p90 stays below the 4 s gate because the burst is < 10 % of the requests.

## Accuracy (h9, default env): 6/6 suites PASS at c1 and at c12 (24 requests each at c12)
| suite | c1 job | c12 job |
|---|---|---|
| mmlu | 148151 | 148165 |
| gsm8k | 148152 | 148170 |
| ifeval | 148153 | 148172 |
| vqav2 | 148156 | 148175 |
| fleurs_en | 148161 | 148176 |
| fleurs_ko | 148163 | 148178 |

## Numerics (hybrid vs legacy / token-serial)
- `test_int --hyb 12 --steps 6 --stagger 1 --deccap 8` (jobs 148004 / 148097): mixed 12/12 requests token-identical to the legacy path; alone 11/12 (request 7, token 4: a 0.125-logit near tie). `--hyb 5 ... --longest 62` (ring wrap, 1351-token prompt, job 148098): 5/5 and 5/5.
- Teacher-forced comparison (job 147971): 63 of 65 steps identical, both flips exact bf16 near-ties; hidden rel-RMS H-S 1.0-1.4 % (the legacy chunk path vs token-serial: 1.0-2.4 %).

## Kernel cache trap (closed)
`target/furiosa-opt/kernel/*/<kernel>.hash` is keyed on the kernel source, not on a `const` it uses (SOFTMAX_EPS). A from-scratch rebuild (`scripts/hyb_clean_build.sh`: every kernel .bin/.hash deleted, lib.rs touched) produced 54 of 54 kernel images byte-identical to
the cache the h5-h7 binaries were built from, so h6-h9 contain the EPS = 1e-30 kernels and the fz FFN. `scripts/kcmp.sh` re-checks a build against that clean set.

## Exact upload set
Only `src/` is uploaded and `src/api/` is replaced by the organizers; `src/api` in this tree is byte-identical to the baseline's (`diff -rq baseline/src/api hyb/src/api`: no output), and the scheduler lives in `src/host/worker.rs` (not in `src/api`).
- Upload: everything under `stage2/hyb/src/` (132 tracked files: `axes.rs`, `lib.rs`, `ops*.rs`, `*_host.rs`, `device/**`, `host/**` incl. `host/runtime/hybrid.rs`, `host/views.rs`, `bin/{server,gemma4,test_kernels,test_int}.rs`).
- Not needed: `Cargo.toml` and `rust-toolchain.toml` are identical to the baseline's (no new dependency, no feature, no `[[bin]]`, no build.rs); `scripts/`, `public-test/`, `lab/`, docs and `runs/` are outside `src/`.
- No test-only cfg or feature is required: the only `cfg` uses are `cfg(backend = "npu")` (set by the furiosa-opt toolchain) and `#[cfg(test)]` unit tests. `src/bin/test_int.rs` is an extra binary that also compiles; `cargo furiosa-opt build --release --bin server` does not need it.
- Optional environment variables are all read with defaults (`S2_*`, `GEMMA4_*`); the server needs only `RNGD_MODEL_DIR` from the platform.
- Src-only rebuild check (baseline Cargo.toml + this `src/` in a fresh target dir): `scripts/hyb_srconly_build.sh`, result in PROGRESS.md.

## Tools
`scripts/hyb_pair.sh <wave> "<tag>:perf|acc:<label>:S2_ROUNDS=N[,ENV=..]:<args>"` (one Arena job per spec, 2 ACTIVE max, retried), `scripts/norm.py`, `scripts/ttype.py` (per-type TTFT), `scripts/waves.py` (burst vs steady state), `scripts/jobtable.py`,
`scripts/hyb_clean_build.sh`, `scripts/kcmp.sh`, `scripts/hyb_srconly_build.sh`, `scripts/cli/*.sh` + `scripts/hyb_cli.sh` (test_int jobs).

## Next (see PROGRESS.md)
Host-side per-pass cost (tables 2.4-4 ms + h2d 0.6 + d2h 0.6 + next_tokens 2.7-3.3 of a 52 ms iteration), drift MASK_PEN port, img fused-layer / `#[unroll]` patch.


# R11 t256c (256-row hybrid pass on fin8): see RESULTS_t256c.md (verdicts, C table, identity, accuracy)

# R12 t256c: build tables (Arena, mixed stream, default env, S2_ROUNDS=6; one row per job; class from the job's cpu.max)
## Build 7nf (tag s2-hyb-t256c-b7nf, server md5 2b3c8dbc33b84bf344dfda86794ac0ef; = build 6 + FP8 fold default OFF)
G4 default env: 12/12 PASS (g7_c1_* and g7_c12_* x mmlu gsm8k ifeval vqav2 fleurs_en fleurs_ko; every STAGE2_SUMMARY passed == total; job ids in the PROGRESS.md ledger). Compliance (src minus api = stage2/_submit_t256c_b7nf, full layout _b7nf_full): 0 hard / 1 WARN (S2_FP8).
256-engine at default env: hs256 counters > 0 (341 256-passes vs 21 128-passes in q7_c15_c), t_layer_s per 256-pass 39.0 (fast) / 39.7-40.6 (slow) ms, t_layer_f 8.4-9.3 ms, hs256_b16 pass 76.7-76.9 ms (fast) / 79.6-81.9 ms (slow).
| label | job | class | host | C | tok/s | TTFT p90 ms | TPOT p90 ms | pass ms (hs256_b16) | t_layer_s ms / pass |
|---|---|---|---|---|---|---|---|---|---|
| q7_c15_a | 176650 | fast | d012bc | 15 | 176.7 | 1707 | 70.8 | 76.7 | - |
| q7_c15_b | 176611 | fast | d012bc | 15 | 176.6 | 1738 | 72.6 | 76.8 | - |
| q7_c15_c | 176544 | fast | dead64 | 15 | 177.0 | 1353 | 72.5 | 76.9 | 39.03 |
| q7_c16_a | 176562 | fast | d012bc | 16 | 183.8 | 2356 | 77.8 | 77.1 | - |
| q7_c16_b | 176579 | slow | 1b2706 | 16 | 168.2 | 2708 | 85.0 | 80.6 | 39.71 |
| q7_c18_a | 176692 | slow | 845af9 | 18 | 168.4 | 3382 | 90.6 | 81.9 | 40.64 |
| q7_c18_b | 176599 | slow | 9cb685 | 18 | 174.3 | 2782 | 89.2 | 79.6 | - |
| q7_c18_c | 176530 | slow | 845af9 | 18 | 170.7 | 2580 | 96.0 | 80.5 | - |
Reading: C=15 fast 176.6-177.0 (N = 3, TPOT p90 70.8-72.6); C=18 slow 168.4-174.3 (N = 3, TPOT p90 89.2-96.0: thin margin to the 100 ms gate); the b6 fold-off control (S2_FP8=off env on build 6, same arithmetic) measured C=18 slow 169.4-175.5, C=15 slow 156.5-158.9 / fast 176.0-176.9: 7nf reproduces it. No slow-class C=15 and no fast-class C=18 draw yet.

## Build 8 (tag s2-hyb-t256c-b8, server md5 5f3cb49c646bb2ca098ea7c306e175ca) = 7nf + native bf16 T=256 layer (S2_NATIVE256BF default ON) vs 7nf, binary vs binary, default env
G4 default env 12/12 PASS (g8_*). Hash identity (S2_FP8=off, test_int --hyb 14): BF=0 f438bc9285e61dd8 == BF=1 f438bc9285e61dd8 (jobs 176942/176933); ring-wrap 89c32013b117797e == 89c32013b117797e (176962/176950). Lean static: sliding 771,562 vs 961.6k, full 831,936 vs ~1,037.8k (-19.8 %). Compliance 0 hard / 1 WARN (S2_FP8).
Mixed stream tables (python3 scripts/cmp78.py; runs/cmp78.out, copied below). Job ids: ledger in PROGRESS.md (q8_* / q7_*).
```
| class | C | arm | n | tok/s median (min-max) | TPOT p90 med | pass ms hs256_b16 med | t_layer_s ms/pass | t_layer_f ms/pass |
|---|---|---|---|---|---|---|---|---|
| fast | 15 | 7nf | 4 | 176.6 (176.0-177.0) | 72.0 | 76.81 | 39.03 | 8.41 |
| fast | 15 | b8 | 3 | 183.5 (183.3-184.3) | 69.7 | 73.10 | 36.07 | 7.65 |
| fast | 16 | 7nf | 1 | 183.8 (183.8-183.8) | 77.8 | 77.07 | 39.03 | 8.41 |
| fast | 18 | 7nf | 1 | 188.1 (188.1-188.1) | 82.5 | 76.90 | 39.03 | 8.41 |
| fast | 18 | b8 | 3 | 195.5 (195.1-195.7) | 79.1 | 73.19 | 36.08 | 7.65 |
| slow | 15 | 7nf | 2 | 160.7 (156.9-164.5) | 80.0 | 80.32 | 39.78 | 8.57 |
| slow | 15 | b8 | 3 | 166.5 (165.9-167.2) | 77.7 | 76.36 | 36.89 | 7.91 |
| slow | 16 | 7nf | 1 | 168.2 (168.2-168.2) | 85.0 | 80.58 | 39.71 | 8.75 |
| slow | 16 | b8 | 2 | 171.5 (167.6-175.4) | 86.7 | 76.75 | 37.10 | 7.93 |
| slow | 18 | 7nf | 3 | 170.7 (168.4-174.3) | 90.6 | 80.53 | 40.64 | 8.84 |
| slow | 18 | b8 | 3 | 178.2 (177.9-187.9) | 86.4 | 76.77 | 36.92 | 7.82 |

| class | C | metric | delta (b8 - 7nf) | 95 % half-width | % of 7nf | point/CI-low % |
|---|---|---|---|---|---|---|
| fast | 15 | tok | +7.13 | 0.85 | +4.04 % | +4.04 / +3.55 |
| fast | 15 | pass | -3.69 | 0.11 | -4.81 % | -4.81 / -4.94 |
| fast | 15 | t_layer_s | -2.96 | 0.02 | -7.58 % | -7.58 / -7.63 |
| fast | 15 | t_layer_f | -0.76 | 0.01 | -9.00 % | -9.00 / -9.09 |
| slow | 15 | tok | +5.83 | 8.78 | +3.63 % | +3.63 / -1.84 |
| slow | 15 | pass | -3.68 | 1.35 | -4.59 % | -4.59 / -6.27 |
| slow | 15 | t_layer_s | -2.88 | 0.40 | -7.24 % | -7.24 / -8.24 |
| slow | 15 | t_layer_f | -0.68 | 0.05 | -7.91 % | -7.91 / -8.45 |
| slow | 18 | tok | +10.20 | 8.52 | +5.96 % | +5.96 / +0.98 |
| slow | 18 | pass | -4.15 | 2.57 | -5.14 % | -5.14 / -8.33 |
| slow | 18 | t_layer_s | -3.57 | 0.45 | -8.79 % | -8.79 / -9.89 |
| slow | 18 | t_layer_f | -1.11 | 0.48 | -12.38 % | -12.38 / -17.73 |

pooled fast pass: 7nf n=6 med 76.9 mean 76.9 sd 0.1 | b8 n=6 med 73.1 mean 73.1 sd 0.0 | delta -3.73 +- 0.12 ms (-4.85 %)
pooled fast t_layer_s: 7nf n=6 med 39.0 mean 39.0 sd 0.0 | b8 n=6 med 36.1 mean 36.1 sd 0.0 | delta -2.95 +- 0.01 ms (-7.56 %)
pooled fast t_layer_f: 7nf n=6 med 8.4 mean 8.4 sd 0.0 | b8 n=6 med 7.7 mean 7.7 sd 0.0 | delta -0.76 +- 0.00 ms (-9.00 %)
pooled slow pass: 7nf n=6 med 80.6 mean 80.5 sd 0.8 | b8 n=8 med 76.6 mean 76.6 sd 0.9 | delta -3.92 +- 1.10 ms (-4.86 %)
pooled slow t_layer_s: 7nf n=6 med 40.1 mean 40.2 sd 0.5 | b8 n=8 med 36.9 mean 37.0 sd 0.2 | delta -3.17 +- 0.48 ms (-7.90 %)
pooled slow t_layer_f: 7nf n=6 med 8.7 mean 8.8 sd 0.3 | b8 n=8 med 7.9 mean 7.9 sd 0.1 | delta -0.91 +- 0.28 ms (-10.30 %)
```
Text-only diagnostic (slow class, C=15, 5 rounds): 7nf qt7_c15_a 145.2 tok/s (pass 80.24 ms), qt7_c15_c 147.9 (79.51); b8 qt8_c15_a 149.5 (75.99), qt8_c15_b 155.6 (75.90): +3..7 %, pass -4 ms; TPOT p90 79.8-83.5 vs 84.8-84.9.
VERDICT: confirmed. Per-pass timer delta negative in both classes (fast -3.7 ms = -4.8 %, slow -3.9 +- 1.1 ms = -4.9 %; t_layer_s -3.0 / -3.2 ms, t_layer_f -0.8 / -0.9 ms), tokens/hash identical, mixed tok/s +4.0 % (fast C=15, CI low +3.6), +6.0 % slow C=18 (CI low +1.0), +3.6 % slow C=15 (CI low -1.8, n=2 vs 3, noisy but the deterministic pass timer resolves it). Rule 3 (point >= -0.3 %, CI low > -0.6 %) met where resolvable; the slow C=15 CI is wide (n=2 baseline) but its point and pass timer are positive. Merge candidate: b8 replaces 7nf. Gap to 200 tok/s remains: fold-off C=18 slow ~178-188, fast ~195.
