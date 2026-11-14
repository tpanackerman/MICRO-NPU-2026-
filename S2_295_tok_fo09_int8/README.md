# S2_295_tok_fo09_int8

Xây trên `S2_278_tok_int8_rn22`, chuyển sang **furiosa-opt-std 0.9.0** (API gather/scatter dạng builder) và thêm hai thay đổi chính xác (bit-identical): AO8 pre-cast cho kernel K int8 và K03-full.

## Kết quả chính thức

- Submission **`40b4b47a`**, team EDABK-Artemis, nộp source ngày 12/10/2026 lúc 22:41 UTC, hoàn tất lúc 23:10 UTC, concurrency **22**; trạng thái `completed`.
- Score chính thức: **295.4715 tokens/s**. Đây là kết quả tốt nhất mới của team, vượt mốc chính thức trước đó 278.5180 tokens/s (+6.1 %).
- Accuracy: **32/32** trong hai vòng.
- TTFT p50 / p90: **672.74 ms / 1205.07 ms** (giới hạn p90 <= 4000 ms).
- TPOT p50 / p90: **63.09 ms / 67.95 ms** (giới hạn p90 <= 100 ms).
- Log chính thức nằm trong `validation/official_evaluation_log.txt` (grader nhận `furiosa-opt-std 0.9.0` ở bước kiểm tra layout).

## Cấu hình

So với `S2_278_tok_int8_rn22`:

1. **Port sang furiosa-opt-std 0.9.0**: 47 lệnh gather, 31 scatter và 6 gather khối KV của rider được chuyển sang API builder (`.gather().by_byte_offsets()...`, thêm bước `to_spm` cho chỉ số không scale). Không đổi chức năng; kết quả giống bit với bản 0.8.1. Mỗi lần launch kernel ngắn hơn khoảng 40-50 us nhờ runtime 0.9.0.
2. **AO8 pre-cast (F1)**: epilogue của attention ghi trực tiếp e4m3 theo layout liên tục theo slice cho kernel K int8 (static -68k chu kỳ, 1.078 -> 1.012 ms mỗi launch trên 0.8.1). Kernel K fold không đổi.
3. **K03-full (F2)**: tầng cuối chạy AO + FFN trên 16 hàng decode cả trong các pass có prompt kết thúc.
4. `src/bin/gemma4.rs` là stub rỗng (như `test_kernels.rs`) để bản build không biên dịch kernel hai lần.

Kiểm tra nội bộ trước khi nộp: nhanh hơn R8 (`S2_278`) 6.4 % trên Arena cùng cửa sổ (284.4 so với 267.4 tok/s, C=22); hash test_int giống R8; sáu bộ accuracy pass; vqav2.117333 189/200 so với 188/200 của R8; mmlu-240 x2 không có item mới bị lệch. Build ước tính trên grader 1441 s. Score chính thức ở trên là kết quả thực tế.

## Build và nộp source

```sh
cd S2_295_tok_fo09_int8
cargo furiosa-opt build --release --bin server   # furiosa-opt 0.9.0
moa-submitter submit --source . --concurrency 22
```

Lệnh nộp source tạo submission mới và dùng hạn mức nộp của tài khoản.
Thư mục chứa source và bằng chứng kiểm tra; không chứa model weights hay runtime.
