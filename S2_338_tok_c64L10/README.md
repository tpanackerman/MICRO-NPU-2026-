# S2_338_tok_c64L10

Xây trên `S2_311_tok_final10` (furiosa-opt-std 0.9.0). Chạy ở **concurrency 30**, không cần biến môi trường. Mọi thay đổi so với m5 (bản đã qua accuracy chính thức 48/48 ở C40, 333,58 tok/s nhưng trượt TPOT) đều **không đổi số học**; phần số học của prompt ảnh được trả về đúng như m5.

sha256 của binary đã đo (`validation/server.sha256`): `bde89a46b54cc9e261cba7c82de17a2c671d6cdb708676858e767870767c3806`. 72 kernel; thời gian build ước trên grader ≈ 1580 s (giới hạn ≈ 1831 s).

## Kết quả chính thức (chạy thử `--bin`, C30)

- Submission **`eea40c97`**, team EDABK-Artemis, 16/10/2026, `--bin`, concurrency **30**.
- Accuracy **32/32** (2 vòng), PASS.
- Throughput **338,67 tokens/s** (5,31 request/s), cao hơn 295,4715 của `S2_295_tok_fo09_int8` 14,6 %.
- TTFT p50/p90 **1,30 / 2,31 s**; TPOT p50/p90 **65,4 / 81,5 ms**. Cả hai cổng đạt.
- Log: `validation/bin_evaluation_log_eea40c97_c30.txt`. Chưa nộp có điểm; lệnh nộp trong `SUBMIT.md` (chỉ `src/**` được tải lên).

## Thay đổi so với m5

1. **Nhân bản hằng số và layout activation** (m3 → m5, +~3 % so với s4m48): các tensor nhỏ (norm, scalar, Hadamard, global scale) được chép một bản cho mỗi slice đọc; vài activation đổi layout để DMA đọc mảnh lớn hơn. Giống từng bit.
2. **E2 + fbf** (+1,4 %): lớp sliding đứng trước lớp full gộp QKV của lớp full đó vào cùng launch; FFN lớp full dùng weight bf16 giải mã sẵn ở host (đã kiểm trên thiết bị: giải mã host và thiết bị trùng từng bit, 0 lệch trên 24 ma trận).
3. **Luôn dùng engine 256 hàng** (+0,3 %), bỏ engine 128 hàng và 18 kernel không dùng ở cấu hình mặc định (ngân sách build).
4. **Không cắt prompt ≤ 128 token qua hai pass** (`S2_NOSPLIT`, mặc định 128; 0 = tắt): câu mmlu.006459 từng đổi đáp án ở token đầu khi prompt bị cắt, vì cửa sổ attention tính theo vị trí bắt đầu đoạn. Chi phí throughput ≈ 0 (−0,05 %).
5. **Chọn 2 hay 4 trang cho prefill lớp full theo từng pass như m5** (`S2_FULL_P4=1` ép 4 trang). Hai bản trước (c64L5, c64L9) ép 4 trang và cùng trượt câu vqav2 riêng của grader; trên bộ dò VQAv2 300 câu ảnh, ép 4 trang lệch m5 25 câu, chọn theo pass lệch 12–14, bằng mức nhiễu m5-với-m5 (13).

Tắt mặc định và không dùng trong bản này: 32 rider mỗi pass (`S2_HYBRID32=1`), pacing (`S2_PACE_MS`, mất ~11 % throughput trên grader).

## Kiểm trước khi đóng gói (Arena, binary này)

- Accuracy công khai ở C30, không env: 30/30 × 5 lượt + 1 lượt measure.py, 0 câu sai.
- Bộ dò VQAv2 (300 câu ảnh, C30) so với m5: 12 đáp án đổi, mất 1, thêm 1 — ngang mức nhiễu m5 chạy lại (13 đổi).
- Bench C30, 4 sóng: 322,9 tok/s Arena, TTFT p90 3,71 s, TPOT p90 58,9 ms, hai cổng Arena đạt; pass 256/16 ≈ 43,5 ms.

## Build và nộp

```sh
cd S2_338_tok_c64L10
cargo furiosa-opt build --release --bin server   # furiosa-opt 0.9.0
moa-submitter submit --source . --concurrency 30
```

Binary không kèm trong repo; build lại bằng lệnh trên (72 kernel, ≈ 1580 s trên grader).

Lưu ý khi build lại: cache kernel của furiosa-opt không băm hằng số trong `axes.rs`; khi đổi số slot phải xoá `target/furiosa-opt/kernel/furiosa-opt-gemma4`.
