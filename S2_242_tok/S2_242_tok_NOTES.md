# S2_242_tok - Stage 2 submission 6abd71db (team EDABK-Artemis)

Official result (2026-10-10 19:35 UTC, `moa-submitter submit --concurrency 22`): **242.4824 tokens/s**, accuracy **32/32** (2 rounds), TTFT p90 1.92 s, TPOT p50/p90 64.3 / 67.1 ms. Grader build about 1557 s.

Version: `big` (stage2 tree, branch big2) = b13s3 (fused K/K'/K_f8 launches, fp8 e4m3 fold of the FFN of sliding layers 24-47 only, all other layers bf16) + S2_211_tok host transport (packed H2D, decode-N2, HEAD_PACK) + K03-256 + t256r (AO in one contraction) + qkvre QP4/QHF + attnre E (hg4 gather) + d13e P0 LM-head path. Source in `src/` is byte-identical to the uploaded set (166 files accepted by the grader); the baseline `scripts/` are copied from S2_211_tok.

Defaults are the shipped ones (fold ON, `S2_FP8=fold`, `S2_FP8_BF16_LAYERS` reserve list for layers 0-23 and the full layers). Gates passed before submit: G-ID, G-ACC 12/12, six suites at C=22/24, vqav2.117333 6/6 at C=22 and C=30.
