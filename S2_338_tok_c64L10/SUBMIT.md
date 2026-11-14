# Nộp c64L10 lên bảng (chỉ khi user ra lệnh)

Cách grader nhận một lượt nộp CÓ ĐIỂM (không `--bin`), xác minh từ log các lượt đã chấm (fb9f0e94 = 175 file, 0759d92a = 166 file, 5cc440be = 146 file):

1. `moa-submitter submit --source <thư mục> --concurrency N` tải lên **chỉ các file `src/**`** của thư mục (log "accepted N file(s)" liệt kê toàn bộ là `src/...`). `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, `manifest.json`, binary `server` không được tải; grader dùng layout chuẩn của nó ("checking the submission layout → layout OK (furiosa-opt-std 0.9.0)"), nên Cargo.toml phải khai `furiosa-opt-std = "0.9.0"` (c64L10: đúng).
2. Thư mục không cần là git clone: các lượt trước nộp từ thư mục phẳng (`--source <this folder>`, RESULTS.md ar01_295).
3. Grader build từ nguồn trên "kernel-builder": fb9f0e94 mất 27 phút 16 giây (1636 s) cho 81 kernel; giới hạn ≈ 1831 s. c64L10 có 72 kernel, ước ≈ 1580 s (tỉ lệ user-CPU so final10).
4. Sau build: accuracy 32 case × 2 vòng (phải 32/32), rồi performance ở concurrency N. Lượt nộp tính vào hạn mức 4 lượt / 24 giờ ngay khi vào chấm.

Lệnh (WSL), chỉ chạy khi user đồng ý:

```sh
export PATH=$HOME/.cargo/bin:$PATH
moa-submitter submit --source /mnt/d/AI_ML_DL/Micro/stage2/steps4_20261014/c64L10/source --concurrency 30
moa-submitter status            # theo dõi; `moa-submitter log <id>` xem build/eval
```

Trạng thái c64L10 trước khi nộp: sha binary bde89a46…, 72 kernel; accuracy công khai C30 30/30 ×5; bộ dò VQAv2 12 đổi (mức nhiễu m5 13); Arena C30 322,9 tok/s, pass 43,5 ms; dự kiến chính thức ≈ 336 tok/s (c64L9 cùng tốc độ đo 336,17 ở C30, TTFT p90 2,28 s, TPOT p90 82 ms).
