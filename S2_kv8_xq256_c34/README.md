# S2_kv8_xq256_c34

Xây trên `S2_295_tok_fo09_int8`. Bản này **đổi số học** (cache K/V của lớp sliding lưu ở e4m3). Phần mảnh xq và số slot không đổi số học. Chạy ở **concurrency 34**.

> Trạng thái: **chưa có điểm chính thức ở C34, và TTFT p90 ở C34 có rủi ro vượt 4 s (xem đính chính bên dưới)**. Lượt chạy thử grader duy nhất (`29248a81`) dùng cùng kernel nhưng 22 slot, ở C22. Mức khoảng 322 tok/s cho C34 là **ước lượng** từ Arena (xem bên dưới).

## Kết quả

### Grader, chạy thử `--bin` (cùng kernel, 22 slot, C22)

- Submission **`29248a81`**, team EDABK-Artemis, 14/10/2026 10:04 UTC, `--bin`, concurrency **22**.
- Throughput **311.63 tokens/s**, cao hơn 295.4715 của `S2_295_tok_fo09_int8` 5.5 %.
- Accuracy **32/32** trong hai vòng (PASS).
- TTFT p50 / p90: **636.69 / 1165.65 ms**. TPOT p50 / p90: **59.42 / 63.99 ms**.
- Log: `validation/bin_evaluation_log_29248a81_c22.txt`.

### Arena, quét C (bản này: lazycache + N slot; mỗi mức 2 lượt, 22 warmup + 4×C request)

| C | tok/s (Arena) | TTFT p90 (Arena) | TPOT p90 (Arena) |
|---|---|---|---|
| 22 | 294.2 | ~3.0 s | ~57 ms |
| 28 | 299.3 | 3.9 s | 63-64 ms |
| 32 | 302.0 | 4.3-4.5 s | 62-64 ms |
| **34** | **304.6** | **4.5 s** | **62-64 ms** |
| 38 | 305.0 | 4.5-5.1 s | 68-69 ms |
| 40 | 305.0 | 5.2-5.4 s | 67-70 ms |

- Throughput bão hòa từ khoảng C34, vì mỗi pass có tối đa 16 rider decode.
- Ước điểm grader ở C34: 311.63 × 304.6 / 294.2 ≈ **322 tokens/s**.
- **Đính chính (15/10): cách ước TTFT trước đây là sai.** Tỉ lệ Arena/grader khoảng 2.5 lần chỉ đúng ở C thấp (Arena gửi C request cùng lúc nên có đợt dồn đầu). Ở C cao, grader giữ đúng C request đang chạy, nên TTFT theo Little's law:
  - TTFT trung bình ≈ C / (request/s) − 63 × TPOT. Kiểm với grader ở C22: 22 / 4.87 − 3.7 ≈ 0.8 s, khớp p50/p90 0.64 / 1.17 s.
  - Ở C34: 34 / 4.75 − 3.7 ≈ **3.5 s trung bình**, nên **TTFT p90 nhiều khả năng vượt 4 s** (Arena đo được p90 4.5–5.9 s).
  - Với throughput này, C an toàn chỉ khoảng 26–28. C34 có rủi ro trượt cổng latency.

## Thay đổi so với `S2_295_tok_fo09_int8`

1. **kv8 (đổi số học)**: cache K/V phẳng của lớp sliding (`[Rf, Ns, Ds]`, `[Rb32, Rw32, Ns, Ds]`) lưu e4m3 thay vì bf16 (`crate::Kv`, `crate::KV_BYTES = 1`).
   - Writer làm tròn f32 → e4m3 một lần ở epilogue (quadp) hoặc qua một pass `device/kv8.rs` trước scatter.
   - Reader giải mã e4m3 → bf16 trong Fetch Adapter (`fetch_table_lookup::<bf16>()`), không thêm pass.
   - Gom K/V của rider và cửa sổ prefill giảm một nửa số byte.
   - Lịch tĩnh kernel K int8: 893.9k → 833.6k ns.
   - Riêng bước này: Arena C22, 4 cặp, +4.27 % so với final09.
2. **xq 256 B (chính xác)**: khâu norm + quantiser của QKV e4m3 giữ 2 token mỗi slice và ghi theo thứ tự (chunk, token).
   - Scratch xq có layout `[Ub, Tc/2, Kc, Tc%2, Ke]`, nên cả ghi lẫn đọc nhân bản đều dùng mảnh 256 B (trước là 128 B, util 0.29).
   - Lịch tĩnh K: 833.6k → 812.3k.
   - Arena C22, 4 cặp so với kv8: +0.66 %, thắng 4/4.
3. **lazycache (chỉ host)**: cache legacy theo request chỉ cấp khi dùng tới, để có HBM cho 34 slot.
4. **34 slot**: các trục đổi đồng bộ với nhau.
   - `Rn = 34`, `Rf = 44800`, `Rb32 = 1400`, `Pn = 170`, `Pr = 88064`, `Pq = 11008`.
   - `CAPACITY_256 = 34`.
   - Ring 1280 và 16 rider mỗi pass giữ nguyên.

## Rủi ro accuracy

- kv8 đổi số học. Trên bộ dò VQAv2 (450 câu) tổng số câu đúng không giảm (255 so với 250-252), nhưng số đáp án đổi cao hơn nhiễu tự nhiên khoảng 4 lần (22 so với 5 câu, bộ 300 câu). Có 4 câu đang đúng bị làm sai (nhiễu tự nhiên là 2).
- Ghi chú của `S2_311_tok_final10` cho biết các bản đổi số học trước đó (kể cả V cache e4m3) đều trượt case vqav2 riêng của grader (31/32). Bản kv8 này qua 32/32 ở C22 (`29248a81`). Ở C34 cách ghép batch khác đi, nên chưa chắc đáp án giữ nguyên.
- Public accuracy trên Arena ở C34: PASS.

## Build và nộp source

```sh
cd S2_kv8_xq256_c34
cargo furiosa-opt build --release --bin server   # furiosa-opt 0.9.0
moa-submitter submit --source . --concurrency 34
```

Binary đã build và đo: sha256 `bdb75c0f31625ae62423ec0627ade4e129ea6c18132c9692984f0a76b49af312`. Mã nguồn trong thư mục này khớp từng file với manifest của binary đó (181 file). Riêng ký tự xuống dòng thì repo chuẩn hóa về LF (`text=auto eol=lf`): vài file của cây build trộn CRLF/LF, nên khác byte nhưng không khác nội dung.

Lưu ý khi build lại: cache kernel của furiosa-opt không băm theo hằng số trong `axes.rs`. Khi đổi số slot phải xóa `target/furiosa-opt/kernel/furiosa-opt-gemma4` trước khi build.
