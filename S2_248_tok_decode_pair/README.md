# S2_248_tok_decode_pair

Ghép hai QKV T8, dùng chung lượt đọc/decode weight, dựa trên `S2_242_tok` tại commit `ca794cd1c1aa448dd42bcc874c0a03a0bd347050`.

## Kết quả chính thức

- Submission **`0759d92a`**, nộp source ngày 11/10/2026, concurrency **22**; trạng thái `completed`.
- Score chính thức: **247.9917 tokens/s**.
- Accuracy: **32/32**.
- TTFT p90 **1847.22 ms**, TPOT p90 **65.93 ms**.
- Log chính thức nằm trong `validation/official_evaluation_log.txt`; source vẫn khớp hash của bản submit thử 248,6798.

## Kết quả submit thử

- Submission `b1df92a0`, ngày 11/10/2026, concurrency **22**.
- Score từ hệ thống chấm: **248.6798 tokens/s**.
- Accuracy **32/32** trong hai vòng.
- TTFT p90 **1898.34 ms**, TPOT p90 **65.63 ms**; đạt cả hai giới hạn latency.
- Score 248,6798 ở trên thuộc `moa-submitter submit --bin`, **không ghi lên leaderboard**.
- Source này đã được chấm chính thức với submission **`0759d92a`**; xem kết quả chính thức phía trên.
- Log đầy đủ và hash nguồn/binary nằm trong `validation/`.

Các lượt Arena trước đó đạt 233,8308 tok/s. Đối chiếu Arena: +0,77% qua ba cặp, dấu của lợi ích thay đổi giữa các lượt.
Một lượt submit thử chưa đủ để kết luận bản này tăng rank ổn định so với bản đội đã nộp 242,4824 tok/s.

## Cấu hình

`S2_DECODE_QKV_PAIR` bật **mặc định** trong source này; đặt `S2_DECODE_QKV_PAIR=0` để chạy đường đối chứng.
So với bản đã kiểm tra trên Arena, chỉ đổi mặc định của cờ host từ tắt sang bật. Hash các kernel NPU giữ nguyên.
Precision, danh sách lớp BF16/FP8, cache và API giữ nguyên bản được dùng để kiểm tra.

## Build và nộp source

```sh
cd S2_248_tok_decode_pair
cargo furiosa-opt build --release --bin server
moa-submitter submit --source . --concurrency 22
```

Lệnh nộp source tạo submission mới và dùng hạn mức nộp của tài khoản.
Thư mục chứa source và bằng chứng kiểm tra; không chứa model weights hay runtime.
