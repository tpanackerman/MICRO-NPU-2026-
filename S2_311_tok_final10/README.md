# S2_311_tok_final10

Xây trên `S2_295_tok_fo09_int8` (final09), giữ nguyên số học (bit-identical với final09 trên mọi fixture kiểm tra) và thêm hai nhóm thay đổi chính xác: sửa layout DMA (layout_q1) và hàng đợi launch sâu phía host (lq).

## Kết quả trên grader

- Submission **`a20bceb2`**, team EDABK-Artemis, nộp ngày 14/10/2026 lúc 07:27 UTC bằng `--bin` (test run: grader bỏ bước build và chạy thẳng binary đã build), concurrency **22**.
- Throughput: **311.79 tokens/s** (4.89 requests/s), cao hơn mốc chính thức 295.4715 tokens/s của `S2_295_tok_fo09_int8` (+5.5 %).
- Accuracy: **32/32** trong hai vòng (PASS).
- TTFT p50 / p90: **605.19 ms / 1157.08 ms** (giới hạn p90 <= 4000 ms).
- TPOT p50 / p90: **59.29 ms / 64.10 ms** (giới hạn p90 <= 100 ms).
- Log nằm trong `validation/bin_evaluation_log.txt`.
- Lưu ý: test run bằng `--bin` không được đưa lên leaderboard. Muốn có score chính thức thì nộp source của thư mục này (lệnh ở dưới).

## Cấu hình

So với `S2_295_tok_fo09_int8`:

1. **layout_q1 (chính xác)**: lượng tử hóa int8 (pre-FF RMSNorm + Hadamard-32 + quantiser) làm một lần trên layout K-split không nhân bản; epilogue AO8 pre-cast cho kernel K fold (F1f); sắp xếp lại DMA nhân bản đầu vào QKV (L1-P2) và epilogue reduce-scatter của FFN fold (L2); một số chỉnh overlap (P1, P8b, P3).
2. **lq (chính xác, chỉ phía host)**: hàng đợi launch sâu (`S2_LQ_DEPTH=8` mặc định), host gửi trước các launch kế tiếp để thiết bị không phải chờ.

Kiểm tra nội bộ trước khi nộp: hash test_int giống final09 trên 3 fixture + preup-off + N2; nhanh hơn final09 4.9 % trên Arena cùng cửa sổ (297.7 so với 283.8 tok/s, C=22, 12 cặp cùng host); vqav2.117333 191/200 so với 188/200; vqav2-80 và mmlu-240 x2 không có item mới bị lệch; sáu bộ accuracy pass. Build ước tính trên grader 1559-1577 s (giới hạn khoảng 1831 s).

Ghi chú về accuracy (14/10): quét 3150 câu vqav2 lấy từ cùng nguồn với grader (`lmms-lab-encoder/VQAv2`, validation) cho thấy mọi thay đổi làm đổi số học, kể cả chỉ đổi thứ tự cộng f32, đều lật 1-2 % số câu "suýt hòa". Vì vậy bản này giữ đúng số học của final09; các bản nhanh hơn có đổi số học (in-place int8 FFN, e4m3 down, int8 trên full layer, V cache e4m3) đều fail case vqav2 riêng của grader (31/32).

## Build và nộp source

```sh
cd S2_311_tok_final10
cargo furiosa-opt build --release --bin server   # furiosa-opt 0.9.0
moa-submitter submit --source . --concurrency 22
```

Lệnh nộp source tạo submission mới và dùng hạn mức nộp của team (4 lần mỗi 24 giờ).
Thư mục chứa source và bằng chứng kiểm tra; không chứa model weights hay runtime.
