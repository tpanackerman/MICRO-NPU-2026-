# S2_355_tok_lossy

Xây trên `c64L11` (furiosa-opt-std 0.9.0) cộng hai patch lossy (`validation/fi8.patch`, `validation/i8d.patch`): down-proj và FFN lớp full chạy int8. Chạy ở **concurrency 30**.

sha256 binary đã đo (`validation/server.sha256`): `e30037fe0cafbed62a27fed2ebb15f51bda756b41fe5c1c54021e5c22b4f0ea9`.

## Kết quả chính thức (chạy thử `--bin`, C30)

- Submission **`5ae97a69`**, 16/10/2026, `--bin`, concurrency 30.
- Accuracy **32/32**, PASS.
- Throughput **355,75 tokens/s** (5,58 request/s).
- TTFT p50/p90 **1,32 / 2,25 s**; TPOT p50/p90 **61,7 / 74,4 ms**. Cả hai cổng đạt.
- Log: `validation/bin_evaluation_log_c30.txt`. Chưa nộp có điểm.

## Lưu ý

- Bản lossy: LM head vẫn bf16 (LM head e4m3 đã trượt public). Không bit-identical với `S2_338_tok_c64L10`.
- Ứng viên `lm8p` (LM head int8 packet 256 B, +1,03 % Arena ở C30) chưa chứa trong thư mục này và chưa có kết quả chính thức.

## Build và nộp

```sh
cd S2_355_tok_lossy
cargo furiosa-opt build --release --bin server   # furiosa-opt 0.9.0
moa-submitter submit --source . --concurrency 30
```

Binary không kèm trong repo.
