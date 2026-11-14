# S2_231_tok_k_native

Native T256 AO/FFN trong fused K BF16, dựa trên `S2_242_tok` tại commit `ca794cd1c1aa448dd42bcc874c0a03a0bd347050`.

## Kết quả submit thử

- Submission `ad45834b`, ngày 11/10/2026, concurrency **22**.
- Score từ hệ thống chấm: **231.3123 tokens/s**.
- Accuracy **32/32** trong hai vòng.
- TTFT p90 **2033.82 ms**, TPOT p90 **72.51 ms**; đạt cả hai giới hạn latency.
- Chạy `moa-submitter submit --bin`: dùng bài đánh giá của hệ thống chấm, **không ghi lên leaderboard**. Chưa nộp source của bản này để lấy thứ hạng chính thức.
- Log đầy đủ và hash nguồn/binary nằm trong `validation/`.

Các lượt Arena trước đó đạt 233,8707 và 233,0473 tok/s. Đối chiếu Arena: −0,46% qua ba cặp; −0,30% trong chẩn đoán cùng tiến trình.
Một lượt submit thử chưa đủ để kết luận bản này tăng rank ổn định so với bản đội đã nộp 242,4824 tok/s.

## Cấu hình

`S2_K_NATIVE_AO` bật **mặc định** trong source này; đặt `S2_K_NATIVE_AO=0` để chạy đường đối chứng.
So với bản đã kiểm tra trên Arena, chỉ đổi mặc định của cờ host từ tắt sang bật. Hash các kernel NPU giữ nguyên.
Precision, danh sách lớp BF16/FP8, cache và API giữ nguyên bản được dùng để kiểm tra.

## Build và nộp source

```sh
cd S2_231_tok_k_native
cargo furiosa-opt build --release --bin server
moa-submitter submit --source . --concurrency 22
```

Lệnh nộp source tạo submission mới và dùng hạn mức nộp của tài khoản.
Thư mục chứa source và bằng chứng kiểm tra; không chứa model weights hay runtime.
