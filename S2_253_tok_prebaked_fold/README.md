# S2_253_tok_prebaked_fold

Xây trên `S2_248_tok_decode_pair` (đã dựa trên `S2_242_tok` tại commit `ca794cd1c1aa448dd42bcc874c0a03a0bd347050`), thêm ba thay đổi: P2z (đường dữ liệu phía host), alpha = 4/3 cho trọng số fold FFN trước khi cast e4m3, và PB (tile trọng số FFN e4m3 được giải mã trước cho 20 lớp đã fold).

## Kết quả chính thức

- Submission **`b14aa84e`**, team EDABK-Artemis, nộp source ngày 12/10/2026 lúc 07:34 UTC, hoàn tất lúc 08:52 UTC, concurrency **22**; trạng thái `completed`.
- Score chính thức: **253.9418 tokens/s**. Đây là kết quả tốt nhất mới của team, vượt mốc chính thức trước đó 247.9917 tokens/s.
- Accuracy: **32/32** trong hai vòng.
- TTFT p50 / p90: **1468.96 ms / 1793.03 ms** (giới hạn p90 <= 4000 ms).
- TPOT p50 / p90: **61.83 ms / 64.78 ms** (giới hạn p90 <= 100 ms).
- Log chính thức nằm trong `validation/official_evaluation_log.txt`.

## Cấu hình

So với `S2_248_tok_decode_pair`, source này có ba thay đổi:

1. **P2z**: thay đổi đường dữ liệu phía host (vùng `src/host/runtime/d13e.rs`). Thay đổi này giữ nguyên kết quả tính toán.
2. **alpha = 4/3**: nhân trọng số fold FFN với 4/3 trước khi cast sang e4m3, giúp giảm sai số lượng tử hóa. Phần còn lại của đường tính toán dùng bf16 không đổi.
3. **PB**: tile trọng số FFN e4m3 được giải mã trước cho 20 lớp đã fold. Đổi lại khoảng 3.5 GB bộ nhớ thiết bị để giảm công giải mã trọng số lúc chạy.

P2z, alpha và PB đều là exact hoặc đã được kiểm tra là exact như mô tả trên. Hash các kernel NPU khác giữ nguyên so với `S2_248_tok_decode_pair`. Precision, danh sách lớp BF16/FP8, cache và API giữ nguyên.

Kiểm tra nội bộ trước khi nộp: sáu bộ accuracy đều pass; vqav2.117333 đạt 12/12; không có hồi quy firewall so với `S2_242_tok`; hiệu năng trên Arena cao hơn `S2_242_tok` 2.7 % (nhiễu host). Score chính thức ở trên là kết quả thực tế.

## Build và nộp source

```sh
cd S2_253_tok_prebaked_fold
cargo furiosa-opt build --release --bin server
moa-submitter submit --source . --concurrency 22
```

Lệnh nộp source tạo submission mới và dùng hạn mức nộp của tài khoản.
Thư mục chứa source và bằng chứng kiểm tra; không chứa model weights hay runtime.
