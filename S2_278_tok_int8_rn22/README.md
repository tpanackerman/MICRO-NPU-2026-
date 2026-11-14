# S2_278_tok_int8_rn22

Xây trên `S2_253_tok_prebaked_fold` (decode pair + P2z + alpha 4/3 + PB), thêm bốn thay đổi: QKV e4m3 trên trọng số e4m3 gốc của checkpoint, AO 8-bit, FFN up/gate int8 có xoay Hadamard-32 cho các lớp sliding 2..23 (giữ fold e4m3 ở lớp 24..47), và nâng sức chứa phiên `Rn` từ 18 lên 22.

## Kết quả chính thức

- Submission **`186ea79b`**, team EDABK-Artemis, nộp source ngày 12/10/2026 lúc 15:43 UTC, hoàn tất lúc 16:10 UTC, concurrency **22**; trạng thái `completed`.
- Score chính thức: **278.5180 tokens/s**. Đây là kết quả tốt nhất mới của team, vượt mốc chính thức trước đó 253.9418 tokens/s (+9.7 %).
- Accuracy: **32/32** trong hai vòng.
- TTFT p50 / p90: **676.55 ms / 1297.67 ms** (giới hạn p90 <= 4000 ms).
- TPOT p50 / p90: **66.72 ms / 72.02 ms** (giới hạn p90 <= 100 ms).
- Log chính thức nằm trong `validation/official_evaluation_log.txt`.

## Cấu hình

So với `S2_253_tok_prebaked_fold`, source này có bốn thay đổi:

1. **QKV e4m3**: phép nhân QKV chạy e4m3 x e4m3 trực tiếp trên trọng số e4m3 sẵn có của checkpoint; scale activation theo từng token bị triệt tiêu bởi norm của từng head nên không cần hiệu chỉnh.
2. **AO 8-bit**: phép chiếu đầu ra attention chạy 8-bit.
3. **FFN int8 + Hadamard-32 cho lớp 2..23**: trọng số up/gate được lượng tử int8 khi nạp (scale theo hàng x chunk), activation int8 theo token x chunk, cả hai được xoay Hadamard 32 phần tử trong mỗi chunk trước khi lượng tử (không dùng dữ liệu hiệu chỉnh). Lớp 0, 1 và 8 lớp full-attention giữ bf16; lớp sliding 24..47 giữ fold e4m3.
4. **Rn 18 -> 22**: trục sức chứa phiên biên dịch sẵn (`src/axes.rs`) và `CAPACITY_256` trong `src/host/worker.rs` nâng lên 22, để hàng chờ không còn chặn các pass (pass 128 hàng giảm về 0). `DEC_CAP` giữ 8 vì 16 làm lệch accuracy.

Kiểm tra nội bộ trước khi nộp: nhanh hơn bản không có `Rn 22` 2.1 % và nhanh hơn `S2_253_tok_prebaked_fold` khoảng 6.8 % trên Arena (cùng host); sáu bộ accuracy pass; mmlu-240 x2 và vqav2-80 không có item mới bị lệch; vqav2.117333 trong phép thử có rider lệch 3.5 % số bản sao, không khác biệt thống kê so với bản đối chứng. Build ước tính trên grader 1511-1525 s. Score chính thức ở trên là kết quả thực tế.

## Build và nộp source

```sh
cd S2_278_tok_int8_rn22
cargo furiosa-opt build --release --bin server
moa-submitter submit --source . --concurrency 22
```

Lệnh nộp source tạo submission mới và dùng hạn mức nộp của tài khoản.
Thư mục chứa source và bằng chứng kiểm tra; không chứa model weights hay runtime.
