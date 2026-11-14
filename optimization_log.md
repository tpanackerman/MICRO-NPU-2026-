# OPTIMIZATION LOG — MICRO 2026 RNGD kernel optimization (Gemma-4 12B: `sliding_project_qkv`, `sliding_attention_output`, `decoder_feedforward`)

Nhật ký tối ưu hoá đầy đủ của team (EDABK-408), từ skeleton gốc (2026-09-14) đến phiên bản cuối V17.3 (2026-09-26). Log ghi **mọi trường hợp đã thử**: cái nào có lợi, cái nào có hại, được gì, mất gì, và bài học rút ra. Tổng hợp từ 5 bản ghi chú đọc nguồn (`analysis/oplog/notes_{early,v10v12,v13v15,v16v17,knowledge}.md`). Số liệu được chép từ các file nguồn đó, không tự đặt ra.

> **Cập nhật 2026-10-03 (kết thúc stage 1).**
> - Phiên bản cuối là **V17.8**, nằm ở root của repo này (cây nguồn hash `69558fa038a0`, presubmit PASS, build lại OK). Bài nộp cuối cùng là `2fe86ca1` = **9.3192×**; điểm chấm cao nhất là 9.3257×. Diễn biến từ V17.3 đến V17.8 và mọi thí nghiệm từ 09-26 tối đến 10-01 nằm ở **§9, cuối file**.
> - Workspace đã được dọn dẹp. Mọi cây thí nghiệm `p081-*` và `submit-snap-*` đã bị xoá; chỉ tài liệu và script công cụ được giữ lại. Vì vậy mọi đường dẫn `E:\2026_Macro_NPU\X` trong log này giờ nằm ở **`E:\2026_Macro_NPU\stage1_archive\X`**. Ví dụ, `p081-v10/v10/experiments/fsl_report.md` nằm ở `stage1_archive/p081-v10/v10/experiments/fsl_report.md`.
> - `stage1_archive/EXPERIMENT_INDEX.md` liệt kê từng track kèm câu kết luận. File đó và §8.2 cùng §9.3 là danh sách những gì KHÔNG làm lại.

## Cách đọc

- **Đơn vị.** "cycles" / "ticks" là số đếm Arena ở mức profile INFO, tức mức grader dùng. Chu kỳ khoảng 0.5 ns (TUC 2 GHz). "static" là makespan của compiler (`--dump-schedule`, mô hình 1 GHz). "HW" là số đo trên phần cứng. "trace" là đo ở mức profile trace, chậm hơn INFO khoảng 0.6-1k ticks và có artefact riêng (xem §7).
- **Điểm số.** Score = geomean(baseline/cycles) của 3 kernel. Baseline bảng xếp hạng (từ 09-23) là QKV 250,514 / AO 404,633 / FFN 3,703,473. Baseline task thời 0.6 là 254,071 / 409,283 / 3,703,637. Một lần chấm = 3 run có seed, lấy median. Run 0 luôn cold, nên median ≈ max của 2 run warm = E[max2].
- **Độ nhạy điểm.** Theo V17_PLAN, ở mức ~9.03, bớt 1k cycles cho QKV được ≈ +0.48 %, cho AO ≈ +0.96 %, cho FFN ≈ +0.16 % điểm.
- **Delta.** Nếu không ghi gì thêm, delta là *paired delta* cùng job, cùng block, ở mức INFO, tính arm − control; số âm nghĩa là nhanh hơn. "± x" là SE của hiệu block-paired. "n/m" là số block mà arm thắng.
- **Job.** Số 5-6 chữ số là job Arena. Số 8 hex là id bài nộp chính thức (moa-submitter). Số 12 hex là hash cây nguồn của snapshot: `cd src && find . -name '*.rs' | sort | xargs md5sum | md5sum | cut -c1-12`.
- **Kết luận** (cột bắt buộc trong các bảng):
  - **CÓ LỢI**: đo được tốt hơn, đã giữ hoặc đã có mặt trong một phiên bản.
  - **CÓ HẠI**: đo được tệ hơn, hoặc sai kết quả.
  - **TRUNG TÍNH**: không phân giải được khỏi nhiễu.
  - **KHÔNG BIỂU DIỄN ĐƯỢC**: toolchain hoặc DSL từ chối (compile/lowering fail, ICE, hoặc không có API).
  - **ĐÓNG BẰNG PHÂN TÍCH**: bị loại ở static gate, bằng mô hình hoặc phân tích trace, không tốn Arena.
  - Một số dòng ghi thêm "(tạm)" hoặc "(sau bị thay)" khi kết luận thay đổi theo thời gian.
- **Nguồn.** Thời 0.6 (09-14..09-23), đường dẫn tính từ `E:\2026_Macro_NPU\`:
  - `TSV:n` = `experiments/experiments.tsv` dòng n;
  - `OL§` = `experiments/OPTIMIZATION_LOG.md`;
  - `LED` = `knowledge/previous_experiments.md`;
  - `RES` = `experiments/RESULTS.md`;
  - `MEM:x` = file memory `x`.

  Từ 0.8.1 v7 trở đi, đường dẫn tính từ `E:\2026_Macro_NPU\p081-v10\v10\`: `experiments/…`, `analysis/…`, `state/…`. Các ghi chú đọc nguồn nằm trong `analysis/oplog/notes_*.md`. Ở đó có đủ bảng gốc, kèm file:line.

## Mục lục
- §0 Tóm tắt
- §1 Lịch sử phiên bản
  - 1.1 thời 0.6
  - 1.2 port 0.8.1 + v7–v9
  - 1.3 V10–V12
  - 1.4 V13–V15.2
  - 1.5 V16–V17.3
  - 1.6 bảng xếp hạng
- §2 QKV
  - 2.0 kiến trúc hiện tại
  - các nhóm thí nghiệm A–K: cấu trúc; layout/stream; head; thứ tự list; contraction; RoPE; norm/V-tail; Main↔Sub; số học; đo đạc; skew
- §3 AO
  - 3.0 kiến trúc hiện tại
  - các nhóm A–G: tiling; layout; head; epilogue; list đuôi; sync/DramReuse; đo đạc
- §4 FFN
  - 4.0 kiến trúc hiện tại
  - các nhóm A–J: kiến trúc/decode; tiling; layout; block scale; head; x hop; epilogue; contraction; số học; đo đạc
- §5 Thí nghiệm liên kernel
- §6 Tri thức phần cứng / toolchain
  - 6.0 luật chấm
  - 6.1 bộ nhớ
  - 6.2 DMA
  - 6.3 scheduler static
  - 6.4 thực thi HW
  - 6.5 sync
  - 6.6 TU/VE/switch
  - 6.7 lowering bị từ chối
  - 6.8 số học
  - 6.9 khẳng định đã bị sửa
  - 6.10 trần
- §7 Phương pháp đo và quy trình
- §8 Nên làm tiếp / KHÔNG làm lại, kèm phụ lục chỉ mục job và danh sách bài nộp
- §9 Bổ sung 2026-10-03: V17.5 → V17.8, bài nộp cuối, KHÔNG làm lại (09-26 tối → 10-01)

---

## 0. Tóm tắt

### Điểm xuất phát
Skeleton upstream (commit `850428729c1b`, 09-14) đo được QKV 254,071 / AO 409,283 / FFN 3,703,637 (job 31766), tức **1.00×**. Trong skeleton chỉ một cluster chạy (`Cluster = m![1 # 2]`). QKV và AO dùng ring-256 broadcast. FFN giải mã toàn bộ trọng số FP4 ra bộ nhớ.

### Hiện tại: V17.3, phiên bản cuối
V17.3 = V17.2 + FFN nd11gz.
- Snapshot: `submit-snap-p081v173`, hash `1ef19c8cd9bd`, binary `db59225b`. Đã tích hợp vào `E:\2026_Macro_NPU\furiosa-opt-gemma4-12B-main`.
- Static: 37,866 / 22,869 / 102,031.
- 4 job normal-harness INFO của integrator:

  | job | 103183 | 103184 | 103186 | 103187 | trung bình |
  |---|---|---|---|---|---|
  | score | 9.1577 | 9.2071 | 9.1280 | 9.1993 | **9.1730 ± 0.037** |

  Median trung bình các kernel: QKV 68,791 / AO 35,437 / FFN 199,530. Nguồn: `state/V173_RESULT.md`.
- **5 job Arena chạy trên chính thư mục `furiosa-opt-gemma4-12B-main`** (theo yêu cầu đóng gói cuối; nguồn `experiments/final_progress.md`, log `experiments/final/n1..n5.log`):

  | job | QKV | AO | FFN | score |
  |---|---|---|---|---|
  | 103215 | 69,294 | 35,595 | 199,141 | 9.1430 |
  | 103216 | 68,748 | 35,239 | 199,309 | 9.1953 |
  | 103221 | 69,657 | 35,035 | 199,303 | 9.1729 |
  | 103225 | 69,148 | 36,107 | 199,433 | 9.1015 |
  | 103231 | 69,351 | 36,223 | 199,365 | 9.0839 |
  | **trung bình** | **69,240** | **35,640** | **199,310** | **9.139 ± 0.047** |

  - 15/15 test PASS. Sai số giống hệt bản tham chiếu V17.x trong mọi job.
  - Speedup từng kernel so với baseline: **QKV 3.62×, AO 11.35×, FFN 18.58×**.
  - Gộp 9 job (4 + 5): trung bình ≈ **9.154**.
  - Ước lượng điểm chính thức: **≈ 9.194** (khoảng 9.184–9.200). Cách ước lượng: lấy điểm chính thức của V11.1 cộng các delta lab cùng job.
- **Bài nộp chính thức gần nhất vẫn là V11.1** (bc26edb1 = **8.8195**: 71,874 / 35,402 / 215,068). V12 → V17.3 chưa nộp, vì user chỉ nộp khi có lệnh "nộp".

### Khoảng cách tới hạng 1
Board ngày 09-26, hạng #1 là IA (a0c366cf): 63,704 / 31,966 / 190,026 = **9.8995**. So với trung bình 5 job cuối của ta:

| kernel | ta chậm hơn IA | tỷ lệ | thứ hạng kernel (INTEL, tính theo V17.2) |
|---|---|---|---|
| QKV | +5.5k | +8.7 % | #7 |
| AO | +3.7k | +11.5 % | #3 |
| FFN | +9.3k | +4.9 % | #10 (riêng FFN 199.3k của V17.3 sẽ đứng ~#6 trên board 09-26) |

- Theo INTEL: nếu ghép kernel tốt nhất của các đội khác IA thì được 9.485×.
- Luồng stream của ta đã đạt 96–98 % giới hạn pin HBM. Vì vậy phần thua là **thời gian không-stream**: đầu kernel, chi phí cố định mỗi lệnh DMA, đuôi kernel, và skew giữa hai cluster. Không phải do băng thông.

### 10 thắng lợi lớn nhất (theo thứ tự thời gian)
1. **CDX-000** (09-14): chạy cả 2 cluster, K-split + inter-slice reduce, fuse lookup vào contraction, FFN dùng partial dot theo block-16. Điểm từ 1.44× lên **3.89×** (job 33638).
2. **FFN-020** (09-15): GeGLU cục bộ trong slice sau một lần regroup DM→DM (120 phần tử/slice). FFN 329k, điểm **5.34×** (job 35641).
3. **v11 của team** (09-21):
   - cos/sin tính trên chip có range reduction;
   - FFN FP4→FP8 giải mã một lần, stream với activation FP8 hi/lo;
   - Up/Gate mỗi ma trận một tile 30 hàng.

   FFN giảm khoảng 22 % HW (258k), điểm 6.3–6.8 (job 71345, 71414).
4. **Port sang 0.8.1 + v2..v6** (09-23): riêng runtime đã bớt ~11k cho QKV. v2 bỏ hop HBM cho FFN, v3e thay reduce có padding, v4b dùng AO `Broadcast1{64,4}`, v5 fuse lookup vào contraction FFN (−15.1k). Điểm **7.66** (job 77990; chấm 11 lần, median ~7.71).
5. **v7 FFN Down K-halves** cộng X1s, E3 và H5: FFN 245.3k → 225.7k (−19.6k), chấm **8.0337** (a0af0cea). Sau đó v8/v9 (qkv8 h4 mỗi head một PE, ao8 e32, ffn8 g2/s12/e16) lên **8.4708** (563b0e6e).
6. **V18 AO K-split 16** của teammate: AO nhanh hơn AO V10 khoảng 2.5k. V11.1 nộp được **8.8195**.
7. **Chuỗi FFN kiến trúc (V13 → V17.3), tổng khoảng −16k.** Các bước (delta pooled, SE):
   - fxv: −3,004 ± 282
   - cc1c (Down tách theo L giữa 2 cluster): −1,308 ± 169
   - k11mmce8 (cân bằng engine): −3,015 ± 120
   - k15 (một DMA cho 15 hàng scale): −1,014 ± 64
   - octk: −1,892 ± 177
   - **nd11gz (bỏ pass decode thuần): −5,553 ± 208**

   FFN từ 215k xuống 199.5k.
8. **Chuỗi QKV (V10 → V17.2), tổng khoảng −3.2k:**
   - far2 (thứ tự packet HBM): −1,275 ± 85
   - x15k2 (layout K15 phủ HBM): −485 ± 75
   - g6z (rms_w nạp một lần mỗi PE + fan-out ring-64 + zip RoPE): −660 ± 70
   - qf7 (thứ tự trọng số Q trước): −524 ± 84
   - q19n2 (V norm 2 pass): −264 ± 43
9. **AO aoYG memset-pin ở đầu kernel:** −756 ± 113 (lab 91244, 91252). Đây là lần đầu dời được sync do compiler đặt, bằng một memset.
10. **Bộ quy trình đo:** lab cùng job, xoay vòng block, ở INFO; tiêu chí KEEP (≥ 2.5 SE, E[max2] < 0, 2 lab); các gate static/PE/switch/emulator. Chính bộ này giúp tránh giữ nhầm cái giả (static tốt nhưng HW tệ) và không bỏ sót cái thật (x15k2: static +2,079 nhưng INFO −485).

### 10 bài học lớn nhất
1. **Static makespan không dự đoán được HW** khi thay đổi thứ tự list, overlap hoặc sync. Ví dụ:
   - x15k2: static +2,079, HW −485.
   - nd11gz: static −1,291, HW −5.6k.
   - AO t90: static −723, HW +731.
   - Thời 0.6 có ≥ 5 thiết kế "static tốt hơn" mà thua trên HW.

   Chỉ quyết định bằng lab cùng job ở INFO.
2. **Trace khác INFO**, có khi ngược dấu. TU-clock "slow contraction" và dead zone cuối kernel AO chỉ là artefact của trace. Riêng trace đã tốn thêm 0.6–1k.
3. **Thứ tự viết trong source không điều khiển scheduler.** Đã có hơn 10 lần kiểm chứng cho kết quả byte-identical. Chỉ phụ thuộc dữ liệu mới dời được: exact-zero pin, memset pin, operand swap, identity-copy chain.
4. **Mô hình máy.**
   - Mỗi cluster có một list TUC in-order, bị chặn head-of-line.
   - DMA chạy FIFO.
   - Mọi DMA biểu diễn được đều dùng engine 0.
   - Một DMA nhỏ tốn ~0.8–1.2k ở bất cứ đâu nó nằm. Chỉ bỏ được lane idle thật, không "giấu" được DMA nhỏ.
5. **Stream đã gần trần pin** (325.5/332.8 B/tick/cluster). Thứ tăng được là thời gian không-stream và cân bằng Main/Sub/lane. Thứ không tăng được là bandwidth.
6. **Xếp chồng:** nhiều thay đổi đứng riêng thì NULL, nhưng gộp lại thì thắng, vì mỗi cái giải phóng một tài nguyên khác nhau trên cùng một đường găng. Ví dụ: fxv gồm x4 +96, varA −270, còn cả stack −3.0k; k11mmce8 cũng vậy. Phải thử cả stack, và thử bỏ từng phần trong stack (leave-one-out).
7. **Số học.**
   - Scale không triệt tiêu qua RMSNorm khi có EPS (sai 70 %, mất 2 job).
   - VE chỉ làm tròn bf16 một lần ở cuối mỗi pass.
   - Mọi KEEP gần đây đều bit-identical nhờ viết lại chính xác: x*1.0, x+0.0, bits AND 0.
8. **Treo thiết bị.** SwitchConfig không được kiểm khi compile: config sai vẫn compile nhưng treo device. Main `to_vrf` lấy thẳng từ `collect` cũng treo. Phải pre-flight bằng host `config_switch` validator hoặc CPU emulator.
9. **Vận hành.**
   - Agent khác sửa cây nguồn trong lúc build.
   - WSL làm đầy ổ C: và crash.
   - Workflow agent chết khi user interrupt hoặc khi đổi tài khoản.
   - File CRLF.
   - Chữ ký `#[device]` bị so khớp nguyên văn với skeleton, và `axes.rs` không được upload.

   Mỗi chuyện đều đã từng làm mất job hoặc mất bài nộp. Quy tắc xử lý nằm ở §7.
10. **Luật chấm:** xếp hạng cuối **chạy lại bài nộp CUỐI CÙNG** trước 2026-10-01 11:59 UTC, không lấy bài tốt nhất đang hiện trên board. Đừng bao giờ để một cây thử nghiệm là bài nộp cuối.

---


## 1. Lịch sử phiên bản

Bảng dưới liệt kê mọi phiên bản và mốc, theo thời gian (giờ địa phương UTC+7).

- **Cột cycles:** "static" là QKV / AO / FFN của compiler. "HW" là cycles đo trên Arena, kèm số job.
- **Cột điểm:**
  - Trước 09-23: geomean theo baseline task.
  - Từ 09-23: theo baseline board. Riêng V13 → V17.3, điểm từng job được tính bằng `8.8195 × cbrt(Π(V11.1_k / median_k))`.
- **Cột "Nộp?":** ghi id bài nộp chính thức (8 hex) và điểm grader chấm, nếu có.
- **Nguồn:** các ghi chú `notes_early.md` §1, `notes_v10v12.md` §1, `notes_v13v15.md` §1, `notes_v16v17.md` §1, cùng các file `state/V*_RESULT.md`.

### 1.1 Thời furiosa-opt 0.6.0 (09-14 .. 09-23 sáng)

| phiên bản / mốc | ngày | thay đổi | static QKV / AO / FFN | HW QKV / AO / FFN (job) | điểm | snapshot / nguồn | Nộp? |
|---|---|---|---|---|---:|---|---|
| BASE-000 skeleton (upstream 850428729c1b) | 09-14 17:32 | – | 116,583 / – / 1,693,200 | 254,071 / 409,283 / 3,703,637 (31766); chạy lại 251,007 / 421,511 / 3,703,133 (33853) | 1.00 | TSV:2,7; `furiosa-opt-gemma4-12B-original/` | – |
| FFN-001 (P1-A decode theo pass) | 09-14 17:34 | FFN Up/Gate decode theo từng pass; Down decode dạng packed-to-columns | – | 247,541 / 408,115 / 1,266,679 (31778) | 1.44 | TSV:3 | – |
| **CDX-000** (Codex viết lại) | 09-14 23:37 | 2 cluster, K-split + inter-slice reduce, fuse lookup vào contraction (QKV/AO), FFN partial dot block-16, bỏ ring broadcast | 72,838 / 35,876 / 218,563 | 169,765 / 78,499 / 490,316 (33638) | **3.89** | TSV:4; experiments/codex-dual-cluster | – |
| REV-B | 09-15 00:16 | FFN-003d (5 tile mới) + QKV-002b + AO-003 (3 tile) | 64,245 / 35,450 / 197,096 | 148,428 / 77,357 / 470,944 (34181) | 4.14 | TSV:36 | – |
| REV-I / K / L / R (FFN-011b / 012 / 012b / 013) | 09-15 | FFN tiling, hàng liên tục, ít bảng lookup hơn | FFN 173,035 → 162,524 → 153,680 → 152,264 | FFN 436,262 (34417) → 390,859 (35294) → 393,122 (35343) → 364,036 (35464) | – | TSV:21,23,24,30 | – |
| REV-P / S / X (AO-004 / 005 / 006) | 09-15 | fuse epilogue / tile 16+14 / FP8 hi/lo | AO 29,908 → 28,735 → 25,827 | 64,668 (35441) → 59,386 (35471) → 57,634 (35543) | – | TSV:28,32,41 | – |
| **REV-AF** = QKV-003c + AO-006 + FFN-020 | 09-15 | GeGLU cục bộ trong slice (120 phần tử/slice) | 62,776 / 25,827 / 144,061 | 153,467 / 49,948 / 329,214 (35641); 152,301 / 58,740 / 335,602 (35647) | **5.34** / ~5.04 | TSV:50,52; experiments/best_sources/ | – |
| REV-AK (bản an toàn để nộp: `Hl` → `Dummy2`) | 09-15 | revert axes.rs vì submitter bỏ qua file này | như trên | 147,597 / 55,202 / 329,784 (37513); 149,385 / 54,102 / 332,200 (37852) | 5.31 / 5.24 | TSV:55,56 | lần nộp đầu **cd8cad57 FAIL** (unresolved import `crate::axes::Hl`); board team 5.1324 (ca54f80c), 5.6453 (ccaf0508) |
| REV-AM (QKV version 7: qkv_r5.rs) | 09-15 | QKV K-sharded | 60,826 / 25,827 / 144,061 | 137,926 / 57,856 / 337,818 (38125) | – | TSV:62 | – |
| V2C = REV-AN | 09-15 | V nằm trên 2 cluster cho tới lúc ghi cache | 58,306 / 25,827 / 144,061 | 135,014 / 72,320 / 335,614 (38188) | 4.90 | TSV:63 | – |
| **M3b** | 09-15 23:08 | Q, K cục bộ trong cluster; cos/sin đóng gói, 1 hop HBM | 55,662 / 25,827 / 144,061 | 126,531 / 61,350 / 334,714 (39037) | 5.29 | TSV:70; experiments/best_sources_M3b/ | – |
| v6 (0.6) = M3b + AO-007/008 + FFN-025/026/030/031 | 09-16 | – | 55,662 / 24,902 / 140,390 | 125,007 / 54,654 / 331,094 (39615); 5 lần chạy lại: 5.074–5.542 | 5.542 | TSV:85; best_sources_v6 | – |
| V8-MERGE → v9 (0.6) | 09-16 | QKV version 8 (full-K rows, ring-8, Q một pass, scale sau gather) + cos/sin đóng gói | 51,098 → 50,660 / 24,902 / 140,390 | 117,373 / 50,904 / 331,104 (41109) = 5.796; 41112, 41114, 41120, 41121 | 5.40–5.80 | TSV:86; best_sources_v9/ | – |
| guide-fork (v9 + AO 5→3 pass + FFN 5→4 pass) | 09-16 | fuse các pass epilogue | ~50,660 / ~24,160 / ~139,989 | 119,241 / 52,005 / 334,109 (41538) | 5.71 | guide-fork/RESULT.md | – |
| **v11 (thư mục của team) chọn làm main** | 09-21 | cos/sin tính trên chip; FFN FP8 hi/lo; Up/Gate một tile 30 hàng | 39,002 / 23,428 / 118,593 | 99,819 / 59,330 / 258,329 (71345) = 6.300; 89,141 / 53,470 / 257,629 (71414) = 6.779 | 6.3–6.8 | experiments/v11_reference/ | – |
| v11 + trì hoãn RoPE | 09-22 | dependency `+0*sum(k)` đẩy ~16 pass RoPE sớm ra sau trọng số K | ~v11 | 100,909 / 54,936 / 260,347 (71718); 85,681 / 51,142 / 257,859 (72065) | 6.44–6.99 | dma-docs/FINDINGS.md | đã nộp: aeea0c97 6.3057, a4e96a99 6.8860, d0bff8ea 6.6209, d38ebfe4 6.5545, f728dd57 6.5950, 7332b8b0 6.6794 |
| **r21n** (main thời 0.6) | 09-23 | Q11/K11 strided theo nhóm head, ring-8 Broadcast1{8,1}, reduce/sqrt trên Sub, operand swap, scale gộp vào head RMSNorm, sin 2 pass | – / 23,428 / 118,593 | 86,402 / 49,614 / 253,629 (77046) = 7.08 | 7.08 | best_sources_r21n; submit-snap-r21n | đã nộp: 9c7a97cd 6.7434, e79ed4a4 6.4623, **92fdd139 7.4345**, cfc3e216 6.5259; bị từ chối: 6e48babf, 22b3e0a0 (đổi tên tham số) |

### 1.2 Port sang furiosa-opt 0.8.1 và các vòng v7 → v9 (09-23)

| phiên bản | ngày | thay đổi | static | HW (job) | điểm | snapshot | Nộp? |
|---|---|---|---|---|---:|---|---|
| P081-v1 | 09-23 | port máy móc (ctx→device); AO tile 10+10+10, FFN Down 3×10 | 39,969 / 26,135 / 144,689 | chấm: 77,796 / 51,587 / 309,773 | 6.71 | submit-snap-p081 | c1928278 6.7089; 46c20165 6.7381 |
| P081-v2 | 09-23 | FFN Down 1×30; gather y của AO+FFN bằng DM→DM đổi Cluster (ClusterSync) | 39,969 / 26,189 / 126,140 | 78,736 / 49,864 / 268,060 (77526) | ~7.09 | submit-snap-p081v2 | – |
| P081-v3e | 09-23 | mean-square của epilogue: partial trong slice + Broadcast1{8,1} + reduce trong slice | 39,969 / 24,185 / 124,136 | 80,019 / 46,307 / 263,596 (77588) | ~7.27 | RES | – |
| P081-v4a | 09-23 | AO 26+4, FFN Down 24+6, dùng `unsafe reshape` sang trục riêng | 39,969 / 23,334 / 120,903 | 79,973 / 45,373 / 260,993 (77915) | – | analysis-work/toolchain081 | – |
| P081-v4b | 09-23 | AO seed x 32 B/slice + Broadcast1{64,4} | AO 22,705 | 80,224 / 42,593 / 261,230 (77919) | – | analysis-work/cp-ao | – |
| P081-v5 | 09-23 | FFN Up/Gate: lookup FP4→FP8 fuse vào contraction | FFN 117,994 | 81,393 / 45,013 / 245,848 (77953) | – | analysis-work/api081 | – |
| **P081-v6** = v4b + v5 | 09-23 15:40 | – | 39,969 / 22,705 / 117,994 | 79,982 / 42,524 / 245,345 (77990) | ~7.66 | submit-snap-p081v6 | nộp 11 lần: 7.7247 (8225589c), 7.6918 (ad44399b), 7.7250, 7.7041, 7.7198, 7.7124, 7.6453, 7.6726, 7.7238, 7.7096 (e1748f19); median ~7.71 |
| **P081-v7** | 09-23 17:5x | FFN Down K-halves + x ring→TRF trên Sub + tile cuối fuse lookup + head fix + đường scale; AO một gather + epilogue 16 slice | 39,969 / 22,714 / 112,660 | 79,578 / 41,509 / 224,529 (79254) = 7.970; 78,192 / 41,667 / 225,425 (79255) = 7.996 | 7.97–8.00 | submit-snap-p081v7 | **a0af0cea 8.0337** |
| teammate ver15_8.1527x (trên ver14 = v7) | 09-23 19:1x | AO RMSNorm 4 → 3 pass: inter-slice reduce trên view regroup KHÔNG padding | AO 22,714 → 22,207 | chấm: 78,108 / 39,545 / 224,291 | 8.15 | experiments/ver15/ | 5f9689dd 8.1527 (của teammate) |
| **P081-v8** = v7 + qkv8 + ao8 + ffn8 | 09-23 19:3x | QKV h4/x8g/gather fuse/1 TRF; AO 24+6 + chain copy + epilogue 32×120; FFN Sequential 64 B + 1 DMA scale 8 hàng + epilogue 16×240 | 37,080 / 22,379 / 111,053 | 74,550 / 39,875 / 216,369 (79793) = 8.357; 74,841 / 40,879 / 216,992 (79795) = 8.269 | 8.27–8.36 | submit-snap-p081v8 | 16b602fe **8.4051**, b1e5c347 8.3711, 8.3601 |
| **P081-v9** = v8 + ffn9 e1 + qkv9 a1qvk3 | 09-23/24 | FFN reduce inter-slice không padding trong epilogue; QKV packet 64 B + pin K-normalize | 36,497 / 22,379 / 110,889 | 73,132 / 39,690 / 216,825 (80799) = 8.4179 | ~8.42 | submit-snap-p081v9 (94d8512bc99b) | **563b0e6e 8.4708** (73,993 / 38,425 / 217,230), 9838fdd4 8.4210, c1de64c3 8.4758 |

### 1.3 V10 → V12 (09-24 .. 09-25 sáng)

| phiên bản | thời điểm | thay đổi | static | Arena (job) | điểm | snapshot | Nộp? |
|---|---|---|---|---|---:|---|---|
| Baseline V10 = bản sao v9 | 09-23 23:57 | cây p081-v10 | 36,497 / 22,379 / 110,889 | 73,132 / 39,690 / 216,825 (80799) | 8.4179 | state/BASELINE.md | – |
| Pha 1: phân tích critical path | 09-24 00:00–01:13 | 3 analyst + 3 reviewer (workflow wf_8d4493ae-73d) | – | – | kỳ vọng +1–2 % | analysis/CRITICAL_PATH_*.md | – |
| **V10** = v9 + AO E201 INTER + AO E205 P2 + QKV E301 far2 | 09-24 06:51–08:06 | – | 36,497 / 22,446 / 110,889 | 72,520 / 39,421 / 216,938 (83170) = 8.4592; 71,640 / 38,595 / 216,801 (83174) = 8.5556 | ~8.46–8.56 | submit-snap-p081v10 (a71773e67e3f) | không |
| Pha 3a: sinh ý tưởng | 09-24 08:13–09:15 | readers + judge | – | – | EV +0.3 % | analysis/IDEAS_*.md | – |
| team **Version_18** (teammate) | 09-24 ~12:40 | v9 + AO projection.rs mới (K-split 16, piece 256 B, contraction 64 phần tử, x 32×512 B + Broadcast1{16,16}, gộp sqrt, tile 96+24, chain 16 copy) | – | chính thức 73,863 / 35,832 / 217,316 | **8.6744** | repo team Version_18_8.67x | (của teammate) |
| **V11** = V18 + QKV far2 | 09-24 12:46 | – | 36,497 / 22,790 / 110,889 | 71,225 / 36,912 / 215,766 (85106) = 8.7144; 70,932 / 37,572 / 218,303 (85109) = 8.6413 | ~8.68 | submit-snap-p081v11 (f0d162084c64) | không |
| **V11.1** = V11 + FFN XG2+SQF+C2 | 09-24 13:45 | mlp.rs = ffn11/src/mlp_xg2sqf.rs (md5 50e459d0) | FFN 109,660 | trace: 85340 8.6326, 85350 8.6384; INFO: 71,046 / 36,414 / 215,718 (85677) = 8.7620; 72,162 / 36,165 / 214,593 (85678) = 8.7518 | ~8.76 | submit-snap-p081v111 (082c21b10cdc) | **bc26edb1 = 8.8195** (71,874 / 35,402 / 215,068), 09-24 14:45 — **bài nộp chính thức hiện tại** |
| Board #1 = 9.728 | 09-24 ~20:00 | 65,498 / 31,867 / 195,361 | – | – | – | memory board-top1 | – |
| **V12** = V11.1 + AO aoYG (memset-pin đầu kernel) | 09-25 ~08:00 | projection.rs md5 af56ea80 | 36,497 / 22,869 / 109,660 | 91191 8.8285, 91198 8.8644, 91220 8.8012, 91221 8.7572; AO lab −756 ± 113 | ước ~8.883 | submit-snap-p081v12 (3674f3c5ef5c) | không |

### 1.4 V13 → V15.2 (09-25 .. 09-26 sáng)

| phiên bản | thời điểm | thay đổi (kernel) | static | NH (job: QKV / AO / FFN = điểm) | delta lab cùng job | ước tính | snapshot | Nộp? |
|---|---|---|---|---|---|---:|---|---|
| **V13** | 09-25 ~10:05 | + FFN "fxv" = x4 head + varA + F-4a (chỉ mlp.rs 69653d0c) | 36,497 / 22,869 / 108,286 | 91771: 71,331 / 35,941 / 212,839 = 8.8280; 91776: 71,543 / 36,130 / 211,574 = 8.8213 | FFN −3,004 ± 282 (91676, 91707; 10/10 block) | ~8.925 | submit-snap-p081v13 (89d2db39b3ee) | không |
| **V13.1** | 09-25 ~13:22 | + QKV "x15k2" = layout K15 + ksq2 (6 file, thêm qkv_k15b.rs) | **38,576** (static +2,079: tệ hơn) / 22,869 / 108,286 | 92963: 70,488 / 35,830 / 211,868 = 8.8857; 92968: 70,672 / 35,290 / 212,296 = 8.9170 | QKV −485 ± 75 IV (91814, 91889, 92953) | ~8.944 | submit-snap-p081v131 (1408f08d0c5c) | không |
| V14 | 09-25 ~10:00–15:40 | vòng planner/critic + qkv14/ffn14/ao14 — **không có KEEP**, V14 = V13.1 | – | chỉ 2 job Arena (93569, 93611) | – | – | – | – |
| **V15** | 09-25 ~18:45 | + FFN cc1c (CC-1: Down tách theo L giữa 2 cluster + nạp TRF bằng CustomBroadcast{ring 1}), mlp.rs b567fa93 | 38,576 / 22,869 / 105,185 | 94546: 70,463 / 35,390 / 209,489 = 8.9571; 94552: 70,938 / 35,632 / 211,034 = 8.8950 | FFN −1,308 ± 169 (94268, 94333; 14/14) | ~8.963 | submit-snap-p081v15 (fa636c2a4748) | không |
| **V15.1** | 09-25 ~21:55 | + FFN k11mmce8 (Down 11+4, mtrf, mce, 2 DMA scale 8 hàng chồng nhau; 22 DMA/cluster), mlp.rs e5d25ada | 38,576 / 22,869 / 104,765 | 96127: 70,828 / 35,260 / 206,249 = 8.9993; 96128: 71,266 / 35,218 / 207,366 = 8.9682 | FFN −3,015 ± 120 (95840, 95904; 11/11) | ~9.006 | submit-snap-p081v151 (08ab7ef63d82) | không |
| **V15.2** | 09-26 ~06:45 | + FFN k15 (cả 15 hàng scale Down trong MỘT DMA; 21 DMA/cluster), mlp.rs 4a0a502a | 38,576 / 22,869 / 103,924 | 96311: 70,959 / 35,028 / 205,356 = 9.0266; 96314: 70,927 / 36,573 / 206,372 = 8.8844 (AO trôi ở run 2) | FFN −1,014 ± 64 (96040, 96112, 96138; 19/20) | ~9.021 | submit-snap-p081v152 (26426858a64c) | không |

### 1.5 V16 → V17.3 + vòng V18 (09-26)

| phiên bản | thời điểm | thay đổi | static | bằng chứng lab (cùng job) | NH (4 job) | ước tính | snapshot | Nộp? |
|---|---|---|---|---|---|---:|---|---|
| **V16** = V15.2 + QKV g6z | 09-26 10:03–10:15 | input RMS weight nạp 1 lần mỗi PE (30,720 B/cluster thay vì 122,880) + fan-out ring-64 bằng Sub Broadcast1{4,16}; exact-zero pin rms; K scale trước K; zip RoPE (bzr); bit-identical | **37,021** / 22,869 / 103,924 | g6z − x15k2: 100276 −853, 100292 −814, 100316 −573, 100360 −735; **pooled −660 SE 70** (27/28 block) | 100413 9.0106, 100418 9.0568, 100427 9.0315, 100436 9.0349 = **9.0335 ± 0.019** (QKV 69,778 / AO 35,429 / FFN 206,010) | ~9.049 | submit-snap-p081v16 (b1a9167504c9) | không |
| **V17.0** = V16 + FFN octk | 09-26 11:49–12:05 | seed block-scale Up/Gate theo octet 240 hàng (piece 256 B, 57.6 KB/leader) + 2 InterTranspose trên Main; bit-identical; chỉ mlp.rs 9c3e6e7c | 37,021 / 22,869 / **103,322** | octk − k15: 100838 −1,814 ± 218, 100847 −2,044 ± 304; **pooled −1,892 ± 177** | 100887 9.0138, 100890 9.0194, 100891 9.0245, 100892 8.9950 = **9.0132** (thấp hơn V16 chỉ vì QKV/AO trôi giữa phiên, +307/+294) | ~9.076 | submit-snap-p081v17 (8c8fcd5c6bcf) | không |
| **V17.1** = V17.0 + QKV qf7 | 09-26 12:41–12:55 | thứ tự trọng số Q trước: `rope_off ¦ rms_w W x ¦ Q W q_scale ¦ k_scale ¦ K W k_rms ¦ q_rms ¦ V`; RoPE rows chạy dưới stream Q; bit-identical | **37,623** (static +602 nhưng HW −524) / 22,869 / 103,322 | qf7 − g6z: 100939 −636, 100958 −413, 101005 +459 (control đảo); **pooled −524 SE 84** | 101188 9.0953, 101193 9.0108, 101200 8.9924, 101205 9.0513 = **9.0375 ± 0.046** | ~9.099 | submit-snap-p081v171 (331d2866a260) | không |
| **V17.2** = V17.1 + QKV q19n2 | 09-26 15:09–15:32 | V normalize per-head 2 pass kiểu "SQF" (bỏ pass sqrt riêng); bit-identical | **37,866** / 22,869 / 103,322 | q19n2 − qf7: 101974 −16, 102090 −336, 102103 −392, 102114 −185, 102121 −414 (bật screen); **pool −264 SE 43** | 102254 9.1098, 102266 9.0016, 102281 9.1149, 102287 9.0731 = **9.0749 ± 0.052** (69,128 / 35,495 / 204,747) | ~9.110 | submit-snap-p081v172 (02e5b6700da0) | không |
| Vòng V18 "backward from target" | 09-26 15:45–18:35 | BACKWARD_{AO,QKV,FFN}, INTEL, fcf18, qkv20, ffn20, ao20 | – | chỉ có ffn20 cho KEEP | – | – | analysis/v18/ | – |
| **V17.3** = V17.2 + FFN nd11gz (**PHIÊN BẢN CUỐI**) | 09-26 17:09–18:15 | FFN không còn pass decode thuần, nên compiler đưa mọi bảng/seed lên trước trọng số của nó; Down piece A = fused lookup ghi f32 block sums + 3 pass VE scale; scale Gate dùng oct seed; seed Up đặt trước chuỗi RMSNorm, gate bằng exact +0; chỉ mlp.rs 54d1252b | 37,866 / 22,869 / **102,031** | nd11gz − octk: 102775 −4,883 ± 348, 102814 −5,929 ± 260; **IV pool −5,553 SE 208 (z −26.7)**; nd11g − octk pool −3,520 ± 124 (4 lab) | 103183 9.1577, 103184 9.2071, 103186 9.1280, 103187 9.1993 = **9.1730 ± 0.037** (68,791 / 35,437 / 199,530) | **~9.194** | submit-snap-p081v173 (**1ef19c8cd9bd**), binary db59225b | không |
| **Đóng gói cuối:** gộp vào `furiosa-opt-gemma4-12B-main`, 5 job NH | 09-26 18:15–18:19 | src/ của `furiosa-opt-gemma4-12B-main` == p081-v173 (diff -rq đồng nhất) | = V17.3 | – | 103215 9.1430, 103216 9.1953, 103221 9.1729, 103225 9.1015, 103231 9.0839 = **9.139 ± 0.047** (69,240 / 35,640 / 199,310); gộp 9 job ≈ 9.154 | ~9.194 | binary db59225b | không |
| ao20 xong, không có KEEP | 09-26 18:35 | dead zone cuối kernel AO chỉ có ở trace | – | 102800, 103229 | – | – | AO giữ nguyên aoYG | – |

### 1.6 Bối cảnh bảng xếp hạng (để đối chiếu)

| thời điểm | đội | QKV / AO / FFN | điểm | nguồn |
|---|---|---|---:|---|
| 09-15 | vinxst (harness cũ, AO x = ±1) | 86,355 / 34,421 / 243,884 | 8.03 | LED |
| 09-23 03:08 UTC | #1 IA (0.8.1) | 74,168 / 33,224 / 201,834 | 9.105 | competition_rules |
| 09-24 12:59 UTC | #1 IA 953875dc | 65,498 / 31,867 / 195,361 | 9.728 | INTEL (một lần nộp: QKV −7,868, AO −1,153, FFN −6,587) |
| 09-25 08:50 UTC | #1 IA / #2 Goat Chovy / #3 PN / #10 **ta (V11.1)** | 63,816/32,059/195,725; 68,016/33,573/207,570; 67,778/36,289/197,974; 71,874/35,402/215,068 | 9.787 / 9.252 / 9.169 / 8.820 | snapshot API leaderboard_20260925_0850utc.json (analysis/v15/sweep_contest_code.md §2.1) |
| 09-26 08:34 UTC | #1 IA a0c366cf / #2 Goat / #3 EoM / #4 PN / #5 Fluidstack / #6 EDABK Spidey / #9 #398 (harness cũ) / #10 ta | 63,704/31,966/190,026; 67,616/33,617/202,271; 66,611/36,234/196,435; 67,778/36,289/197,974; 68,655/35,688/198,921; 68,963/35,920/197,954; 78,870/30,228/226,475; 71,874/35,402/215,068 | 9.8995 / 9.3466 / 9.2514 / 9.1695 / 9.1666 / 9.1481 / 8.8590 / 8.8195 | INTEL §board |

Ghi chú: EDABK Spidey và The_Odyssey_of_EDABK là các đội khác cùng lab (không liên lạc). #398 được chấm theo harness trước khi siết luật (AO x = ±1). Mọi số của bảng chính thức tính theo baseline 250,514 / 404,633 / 3,703,473 đều tái lập được tới 4 chữ số thập phân.

---


## 2. Kernel QKV (`sliding_project_qkv`)

### 2.0 Kiến trúc hiện tại (V17.2 = QKV của V17.3; static 37,866; NH 68,791–69,240)

**Dataflow.**
1. Chuẩn hoá đầu vào: `h = bf16(RMSNorm_3840(x; w_in))`.
2. Chiếu Q/K/V với trọng số FP8 kèm per-row scale. Mỗi cluster có 8 head q + 4 head k + 4 head v: Q 7.86 MB, K 3.93 MB, V 3.93 MB, tổng 15.73 MB/cluster.
3. Per-head RMSNorm (có weight cho q và k; không có weight cho v).
4. RoPE rotate-half cho q và k.
5. Ghi `q_out` và ghi cache k/v bằng `dma_scatter` động theo slot = pos mod 1024.

Lịch sử các bước dẫn tới dạng này nằm trong các bảng 2.1–2.10. Dưới đây là cấu trúc hiện tại.

**Layout trọng số: K15 (x15k2, V13.1).**
- 3,840 B của mỗi hàng trọng số được chia thành 15 chunk 256 B, đặt lên 15 trong 16 slice của một nhóm 16 slice; mỗi cluster có 16 nhóm. Slot thứ 16 là pad.
- Contraction dùng chunk mask và reduce giữa các slice trong nhóm bằng inter-slice reducer, nên mỗi giá trị ra đã đầy đủ trong một slice.
- Luồng stream đạt 304–311 B/tick/cluster, gần pin share (FREECHECKS CJ-3).
- Layout này thay far2 (V10). far2 chỉ đổi thứ tự packet HBM: mỗi bước cột đọc 8 run × 8 hàng.

**Activation.**
- x được chia thành FP8 hi/lo.
- Packet 64 B, và `Gs` (hi/lo) làm time-broadcast trong stream adapter (trick A1) cho Q và V.
- K dùng packet 32 B: A1 trên K luôn có hại.

**Đầu kernel (g6z, V16).**
- rms_w được nạp một lần mỗi PE: 30,720 B/cluster, piece 128 B.
- Sau đó fan-out bằng ring-64 `Broadcast1{slice1:4, slice0:16}` trên Sub.
- x được nhân bản 16× (123 KB/cluster).
- Mean-square của input RMSNorm chạy in-place trên layout contraction (E5, 0.6).

**Thứ tự trọng số: Q trước (qf7, V17.1).**
List DMA static là `rope_off ¦ rms_w W x ¦ Q W q_scale ¦ k_scale ¦ K W k_rms ¦ q_rms ¦ V` (chỉ có 3 wait trước V).
- Các RoPE row (POS, góc, cos/sin), cùng chuỗi RoPE ramp, được gate bằng chunk-mask seed nên chạy dưới stream Q.
- Contraction Q, q gather và q-norm pha 1 chạy dưới stream K.
- K contraction được pin bằng các exact-zero lấy từ q rms và K-scale. Pin giữ đúng thứ tự mà không đổi bit nào.

**Head layout và RoPE.**
- Mỗi PE một head (h4): head ở slice 0/64/128/192, gather bằng `Broadcast1{slice1:64, slice0:1}`.
- cos/sin tính trên chip từ `rope_offset` (v11): range reduction `fxp_to_fp(24)(turns)·(−2π·128) + angle`, còn ramp được tổng hợp bằng cách nhân đôi.
- RoPE dạng zip (bzr, V16): mỗi tensor 2 pass Main (begin_interleaved + unzip + zip), thay cho 3 pass Main + 1 pass Sub.
- Channel scale được gộp vào head RMSNorm (r21, 0.6). Weight scale nhân SAU gather (V8).

**Đuôi kernel (từ lúc V land).**
V contraction (~2.51k) → V gather switch + cast bf16 (~1.17k) → normalize per-head 2 pass kiểu SQF (q19n2, V17.2) → v scatter (index fetch + scatter động, ~1.66k).

Metric là cửa sổ của c0 = cửa sổ riêng của c1 + ~2.0–2.3k (skew + barrier cuối kernel).

**Phân rã lane trên c1 (trace 100979, V17.2, BACKWARD_QKV):**

| đoạn | ticks |
|---|---|
| head | 4.40 |
| Q | 24.02 |
| Q→K | 5.51 (INFO ~2.3–3.5) |
| K | 12.44 |
| K→V | 2.54 |
| V | 12.98 |
| tail | 7.37 (INFO ~6.8–6.9) |

Pure stream là 48.33k. Một lane "hoàn hảo" (không wait) sẽ cho metric 64.6–66.2k, tâm 65.4k. Leader (63.7k) còn thấp hơn mức đó 0.9–2.5k.

### Bảng thí nghiệm QKV

Cột: **tên/id | thay đổi (cơ chế) | kết quả | kết luận | được / mất | lý do, bài học | nguồn**.
- Mã Q01–Q66 và Q30a–f là QKV thời 0.6 (LED §3, TSV).
- QA1–QA21 là 0.8.1 vòng v7–v9 (`experiments/p081/qkv8_notes.md`, `qkv9_notes.md`, `ao_qkv_notes.md`).
- Tên còn lại là từ V10 trở đi (`experiments/*_progress.md`, `V10-E*.md`, `analysis/*`).

#### 2.1 Nhóm A — Kiến trúc tổng thể, chia cluster, hậu xử lý và gather

| tên/id | thay đổi (cơ chế) | kết quả | kết luận | được / mất | lý do, bài học | nguồn |
|---|---|---|---|---|---|---|
| Q01 CDX-000 | Codex viết lại: 2 cluster, K-split + `vector_inter_slice_reduce`, lookup FP8→BF16 fuse vào contraction, bỏ ring broadcast | static 72,838; HW 169,765 (33638); HW/static 2.33 | CÓ LỢI | được: 254k → 170k | fuse lookup→contraction (R2) hợp lệ và có lợi | TSV:4, LED |
| Q02 QKV-002a | norm per-head chỉ trên Main, nạp x một lần, `to_hbm` thẳng từ layout pad 64 slice | static 75,737 (+4 %) | CÓ HẠI (một phần) | được: bỏ norm + x trùng tiết kiệm 12k DMA; mất: store từ layout pad tốn 4.6k/6.9k/6.9k | không bao giờ store HBM từ layout pad nhiều slice; gather về 1 slice/cluster trước (952 + 461) | TSV:11 |
| Q03 QKV-002b | 002a nhưng bỏ store từ layout pad | 64,245; 143,826 (34133), 148,428 (34181); pooled 148,753 (n = 12) | CÓ LỢI | nhanh hơn 003c ~3.65k trên HW (M04) | – | TSV:12, LED |
| Q04 QKV-003 (REV-M) | hàng liên tục trên 2 cluster (Q 8 hàng/slice, K/V 4), hậu xử lý + store trên 2 cluster | 55,362; 164,446 (35403) FAIL q 58 % / k 58 % / v 52 % | CÓ HẠI (sai) | – | trục cluster `Qs/2048` đặt trên tensor `[Ps,H]` làm dữ liệu bị nhân bản; cluster 1 không được ghi; `dma_gather_scaled` chỉ lấp cluster 0 | TSV:25 |
| Q05 QKV-003b (REV-N) | cùng layout, hậu xử lý 1 slice qua hop HBM, trọng số đi trước | 61,933; 144,846 (35421) FAIL k non-finite @1024, v 54 % | CÓ HẠI (sai) | – | `m![Qs/2048]` trên `[Ps,H]` nhân bản hàng 0..1024 | TSV:26 |
| Q06 QKV-003c (REV-O) | sửa bằng `KvCluster = m![Ps/1024]`; Q thành 2 tile × 4 hàng | 62,776; 156,003 (35431); pooled 152,406 (n = 16) | CÓ LỢI (nhìn lại thì là sai lầm) | static −1,469 so với 002b; nhưng HW chậm hơn 002b ~3.65k (2.8σ) | đuôi hậu xử lý trên Main 24k, DMA op nhỏ 25k; static đánh lừa | TSV:27, LED M04 |
| Q07 QKV-004 (REV-Q) | input norm theo layout cặp, hậu xử lý head-local trên 2 cluster fuse RoPE, cos/sin qua hop HBM, `to_hbm_view` + `dma_scatter` trên 2 cluster | 68,489; 178,113 (35448) PASS | CÓ HẠI | mất: 18k DMA idle | scheduler xếp mọi DMA của cây q (kể cả gather/hop chờ Main) trước trọng số K/V; list suy ra từ đồ thị (DfsPostOrder/SethiUllman). Phụ: `to_hbm_view`/`dma_scatter` trên 2 cluster là đúng | TSV:29 |
| Q11 QKV-009 | buffer `Qf` 16 slot dùng chung + FP8 hi/lo | CF | KHÔNG BIỂU DIỄN ĐƯỢC | – | gather qua tile-view hạ xuống IndexAccess (DMA gián tiếp), verifier từ chối | TSV:45 |
| Q14 QKV-012 (REV-AH) | mỗi slice một head | 55,959; 147,516 (35700) FAIL 58 % | CÓ HẠI (lỗi) | – | trục cluster hậu xử lý `Qs/2048` lệch với trục store (Ns,Gs,Ds): cluster 1 không được ghi; lỗi byte-identical với 35403 | TSV:53 |
| Q15 QKV-012b (REV-AI) | như 012 nhưng đổi tên trục cluster thành `Ns/4` | static 56,034 (−6.7k), Main −40 %; HW 170,476 (35707) PASS | CÓ HẠI | static −6.8k; HW +18k (170k so với 152k) | HW/static 3.04 so với 2.43: 50 Sub + 54 PeCore so với 43 + 48; mỗi lần nạp VRF per-head là một sync Main/Sub mà mô hình static không tính | TSV:54 |
| Q17 QKV-013 | hậu xử lý head-local trên 2 cluster (post của 004 + front end của 003c) | 66,777 (+4.0k) | CÓ HẠI | Main 32.3k → 23.6k, DMA 52.9k → 49.8k; nhưng idle 9.9k → 17.0k, nCore 48 → 86 | các DMA của chuỗi post làm nghẽn hàng đợi trên Main | TSV:57 |
| Q21 REV-AM (QKV version 7) | K-sharded `m![Qs/32%64, H/960]`, inter-slice reduce ngay trong contraction, channel scale trên mapping contraction (bỏ 3 Broadcast1 = 3,096 Main) | 60,826; 137,926 (38125) | CÓ LỢI | activation 4,407 → 2,265; Main 32,264 → 28,409; nCore 48 → 36; weight util 0.82 → 0.70–0.73 (+4,330 DMA); HW −7.4 % trong khi static chỉ −3.1 % | HW tính tiền theo số sync | TSV:62, OL§4.2 |
| Q22 REV-AN = V2C | V ở lại 2 cluster tới lúc ghi cache (bỏ to_hbm + reload, norm 1024 thay vì 2048) | 58,306; 135,014 (38188) | CÓ LỢI | bỏ store 434 + reload 499 + độ trễ HBM; nCore 36 → 34 | thắng bản v7 nguyên gốc 153,529 (38261, chạy cách vài phút) | TSV:63, OL§4.3 |
| Q27 **QKV-M3b** | Q, K ở lại cluster tính (cluster c sở hữu head 4c..4c+3; `Qs/2048 == Ns/4`); cos+sin đóng gói vào một tensor 1 KiB `m![Dummy2, Ds]`, 1 lần `to_hbm`, nạp lại với trục cluster `m![Ns/4]` không có trong tensor HBM, nên được nhân bản | 55,662 (−2,644), DMA op 30 → 26, main_busy −3,475, idle −1,100; HW 126,531 (39037) −6.3 % | CÓ LỢI | mô hình dự đoán 129.3–132.7k, thực tế tốt hơn 2.2–4.9 % | cách nhân bản hằng số sang cluster 1 đúng đầu tiên; mô hình bảo thủ với việc bỏ round trip; RoPE giữ 1 head/slice vì trần VRF (q_sin 2,048 B/slice) | qkv-M3/{PLAN,prediction,RESULT}.md |
| Q38 QKV-E8 | E5 + E6 + V gather mỗi slice một head `m![1 # 64, Ps/256 % 4]` | 54,490 (−1,172 so với M3b); HW 129,960 (39534) = +3.4 %; 125,620 (41107) = −2.4 % so với 41088 | TRUNG TÍNH (không phân giải được; đã revert) | – | 2 cặp đo cho dấu ngược nhau; bản thân M3b dao động 4.0 % → luật L7: một cặp không phân giải được < 4 % trên QKV | OL§6.3, §7.1 |
| Q39 QKV-E9 | Q và K cũng mỗi slice một head (bỏ 2 InterTranspose, 2 pass bỏ pad, 2 Broadcast1) | 55,049 (+559 so với E8), dù Main −3,740 và bớt 6 pass | CÓ HẠI | – | Q RoPE xong sớm, scheduler kéo q_out + v_weight_scale lên trước trọng số V; V land trễ 668, đuôi mất 521 overlap | TSV:94 |
| Q51 thăm dò head-gather | gather head từ layout Q11 | `TransposedBroadcast1{4,64}` 1,031 nhưng thứ tự kênh 8s+e; thứ tự tự nhiên qua time + transpose 2,823 (+1.8k/ma trận) | ĐÓNG BẰNG PHÂN TÍCH | – | collect chỉ nhận packet 32 B; packet 2 B bị `commit_trim` từ chối | qkv-r12 |
| QA5 **qkv8 h4** (lab1 79428) | head slice `KvHeadsAcrossSlices = [Ns % 4, 1 # 64]` (mỗi PE một head, slice 0/64/128/192); 3 gather bằng `Broadcast1{slice1:64, slice0:1}` (ring 64, đã kiểm trên host) thay cho head gather TB1{4,64} ring-256 | 39,969 → 38,849; median warm 80,546 → 77,995 (−2.55k, −3.2 %), 8/8; V-contraction stall 7/16 → 0/16 | CÓ LỢI | gather 2,427 → 785 HW (×3); đuôi 11.6k → 8.3k; mất: load nhỏ giờ trên 4 engine và không rẻ hơn (k_rms 1.18 → 1.42–1.49k); stream trọng số chậm ~1.1k | head-per-slice không tệ (sửa kết luận X08) | p081/qkv8_notes.md |
| QA7 qkv8 hv / hvq / h4svtq (lab3 7951x) | hv: K, Q giữ trên PE 0, chỉ trải V; q: Q normalize với q làm stream, norm weight làm Sub VRF | 39,294 / 39,462 / 38,707; base 79,525, h4svt 77,046, hv 78,464, hvq 79,331, h4svtq 78,775 | CÓ HẠI (bỏ hết) | – | hv đưa gather Q ring-256 trở lại đường găng; Q un-swap dời q_rms DMA/to_vrf trong list làm lane +2k | qkv8_notes |
| QA8 qkv8 **g** / gn (lab4 79559) | g: V gather thẳng theo thứ tự kênh tự nhiên (1 kênh/packet, 4 hàng theo time, all-gather ring-64 nối 64 slice vào trong cùng, Transpose gói 4 bước) = 1 pass thay cho gather + transpose; n: pass sqrt head trên Main ghi rms VRF | 38,273 / 38,838; warm base 79,430, h4svt 77,784, h4svtg 77,066, h4svtgn 76,578 | g CÓ LỢI (đuôi −0.5..−0.7k); n TRUNG TÍNH (bỏ) | – | T tốt hơn của n là do launch skew nhỏ hơn (1.9k so với 2.7k): nhiễu | qkv8_notes |
| Q-gather packing / gather K một pass (V10 C3) | gói 8 B cho gather Q (1.69k HW) và K | không build | ĐÓNG BẰNG PHÂN TÍCH | – | cả hai nằm trong slack của Main trước khi V land (1.1–2.0k); chỉ có ý nghĩa nếu list dời (kiểu thất bại đã biết) | analysis/IDEAS_COMPUTE.md C3 |
| qs1 (qkv18 T2 split-Q) | K, Q_a, Q_b, V với store q theo nửa qua `HbmTensorViewMut::tile` | COMPILE FAIL "mir: expected a HbmTensorViewMut" | KHÔNG BIỂU DIỄN ĐƯỢC | – | tile trên HBM view mutable bị MIR từ chối | experiments/qkv18_report.md |
| qsb2 (qkv18 T2) | tách Q, copy từng nửa vào một tensor q gộp, 1 store | static 40,085, list `K W rope_off ¦ Q_a W q_scale ¦ q_rms W Q_b …` (Q_b nằm sau một wait) | ĐÓNG BẰNG PHÂN TÍCH (T2 KILLED) | – | nửa sau của Q bị chặn sau wait | qkv18_report |
| BACKWARD_QKV S6 / S7 / S8 / S9 / S10 / S11 / S13 / S15 | S6 tách K giữa 2 cluster; S7 head bất đối xứng giữa 2 cluster; S8 bớt/thêm lệnh trọng số; S9 RoPE bằng bảng cos/sin HBM qua `dma_gather_scaled(rope_offset)`; S10 thay kv scatter; S11 để Q hoặc K là trọng số cuối; S13 "straddler" overfill static; S15 stream > 651 B/tick | S6 +2..+3k; S7 độ mịn 1 head = 3.0k; S8 +1.3k mỗi lệnh thêm; S9 +2..+3k lane (đường bảng 6.6k lane + sync); S10 `dma_scatter_unscaled` là todo!(); S11 đuôi 7.8–9.2k; S13 cửa sổ static 66 % busy = ~92 % trên HW; S15 chưa đo layout nào > ~655 | ĐÓNG BẰNG PHÂN TÍCH | – | không danh sách lệnh nào của ta đạt 63.7k với cái đuôi hiện tại | analysis/v18/BACKWARD_QKV.md |

#### 2.2 Nhóm B — Layout trọng số trên HBM và tốc độ stream

| tên/id | thay đổi (cơ chế) | kết quả | kết luận | được / mất | lý do, bài học | nguồn |
|---|---|---|---|---|---|---|
| Q24 QKV-FULLK | layout full-K của ta + back end v7 | 59,014 | CÓ HẠI | DMA −2,146 nhưng idle +2,854 | DMA ngắn lại thì lộ đuôi Main | TSV:65 |
| Q26 QKV-030 S1 / S1b / S1c / S1d (M1) | sở hữu cả hàng trọng số (S1: tách H 2 đường; S1b: cả hàng + Switch ring-2 mỗi projection; S1c: 1 Switch dùng chung + scale sau gather; S1d: scale trở lại mapping contraction) | 59,708 / 58,621 / 59,965 / 57,789 (V2C 58,306); dự đoán −0.4 % | TRUNG TÍNH (revert) | trọng số 32,450 → 27,572 (util 0.713 → 0.839) đúng như dự đoán, nhưng trả lại: +1,146 staging x, +719 config DMA của Switch, +780 gather, +1,045 idle | các luật L1–L5 (sau này L1/L2 bị bác) | qkv-030/RESULT.md |
| Q28 qkvM3c | M1 đặt trên M3 | 55,028 (−634, +1 op) | TRUNG TÍNH (PARK) | gather + staging + config DMA trả lại 4,244 trên 4,878 | phạt nằm ở hình học gather cả hàng; V8 sau này làm nó có lợi (Q43) | qkv-M3/RESULT §5 |
| Q34 QKV-E4 | tách H 8 đường (run 480 B) | 61,271 (+5,609) | CÓ HẠI | rate trọng số 1,021 → 836 B/cycle ở cùng hàng 3,840 B (+6,844) | độ dài run có ảnh hưởng dưới ngưỡng 480..960 B; bác luật L1; H/960 đúng là tối ưu | TSV:89 |
| Q35 QKV-M1b | trọng số cả hàng + RMSNorm in-place | 67,709 (+12,047) | CÓ HẠI | – | activation cả hàng qua fan-out HBM thuần tốn 18,400 (220 B/cycle); norm in-place bị VRF chặn (input_rms_weight 3,840 f32 = 15,360 B > 8,192) | TSV:90 |
| Q43 **V8-MERGE** | lấy nguyên QKV của version 8: hàng full-K (Q `m![Qs/8 % 256]` = 30 KiB liên tục/slice, K/V `m![Ps/4 % 256]`); `CustomBroadcast{ring_size:8}` qua `H/480` cho x (3,411 → 1,406); Q là MỘT pass 8 hàng (`transpose::<m![Qs%8/4], m![Qs%8%4 # 16]>`, in_rows ≤ 4; bớt 1 bảng 838); nhân `weight_scale` SAU gather (512 descriptor 8–16 B / 792–838 cyc → 8 descriptor / 541–564) | 51,098 (−4,564); 117,373 (41109), 129,685 (41112), 131,575 (41114) | CÓ LỢI | util trọng số 0.713 → 0.839, 1,218 B/cycle = trần HW; bỏ inter-slice reduce | QKV-030 đóng M1 vì nhìn nhầm biến (thật ra là đường đi của activation) | v8-merge/PLAN.md |
| Q49 qkv-lab đợt 1 (73542) | probe chỉ-DMA Q0..Q5, K0/K1 trên q/k_weight thật | Q0 (v11) 31,603 = 498 B/c; **Q5 hàng trải theo slice (slice s giữ hàng s, s+256, …; piece 3,840 B aligned) 26,261 = 599 B/c (−5,552 ± 96, 3/3)**; Q1 K-split ×2 +4,452; Q3 ×4 +3,410; Q4 2 lệnh +1,341; K1 tệ hơn K0 | phát hiện | – | nhanh khi các slice đồng thời đọc các hàng HBM KỀ NHAU với piece aligned 256 B; layout v11 T = 1,668 + B/525 trên HW | qkv-lab/RESULT.md |
| Q50 qkv-lab đợt 2/3 (73583, 73595) | Q6/Q7 (2–4 hàng liên tiếp strided), Q11 strided theo nhóm head, K5/K6/K11 | Q11 26,144 = 602 B/c (−5,094 ± 230); K11 12,436 = 632 (−2.3k); Q6/K6 tệ hơn; Q7 −341 ± 614 | phát hiện | – | chỉ strided hoàn toàn mới nhanh; strided trong nhóm 64 slice của một head nhanh ngang strided toàn cluster | qkv-lab/RESULT.md |
| Q52 r11 (73671; 74981) | Q11/K11 trong kernel + 3 pass repack | work 89.0k so với v11 86.4–87.2k (+3 %); total mean 93,646 so với 95,026 | TRUNG TÍNH | tiết kiệm 16k DMA nhưng bị ăn hết | QKV bị TU cùng giới hạn (union TU của v11 là 82k/100k) | qkv-r12/RESULT |
| Q65 giữ layout strided xuyên suốt | giữ layout contraction strided qua norm/RoPE/output (bỏ gather + repack) | `probe_v_load` CF "visa: while lowering Dma"; `probe_v_scatter` static 384,846 (so với 913) | KHÔNG BIỂU DIỄN ĐƯỢC | – | bảng per-channel cần piece HBM 2 byte cách nhau 128 B; gather-thứ-tự-tự-nhiên fuse 2,823 so với 1,808 static | MEM:qkv-r12 |
| QA18 qkv9 L3 probe layout (80323; số liệu sai theo thiết kế) | remap DMA trọng số: p1 cluster = chẵn lẻ head; p2 cluster = chẵn lẻ hàng; p3 hàng xen kẽ trên 4 PE; p4 cả hai | 37,080 / 37,337 ×3; T base 73,712, p1 74,674 (+0.96k), p2 80,295 (+6.6k), p3 77,058 (+3.3k), p4 82,630 (+8.9k) | CÓ HẠI (đóng hướng layout) | – | layout v8 tốt nhất; các stream trên hàng HBM gần nhau giao thoa (V10 giải mã lại: do engine/cluster dùng chung dải địa chỉ, không phải do hop rộng) | qkv9_notes; V10-E301.md |
| H1 H-ORDER (phân tích V10) | thứ tự packet HBM trong mỗi engine của stream Q/K/V | static và hwsim không đổi (36,497; 69,789); trần ~5.5k (tốc độ kiểu FFN) / ~2.8k (kiểu AO) | → E301 | – | ẩn số duy nhất là tốc độ HBM | CRITICAL_PATH_QKV.md s4e |
| E301 **far2** | bit DMN = bit hàng 8 (Q) / 7 (K,V); slice-trong-DMN = bit hàng {6,7}+{0..2} (Q) / {5,6}+{0..2}; mỗi bước cột đọc 8 run × 8 hàng trải trên các hàng của engine (stride trong 30,720 B, giống FFN Up); commit của gather khôi phục thứ tự kênh (không thêm pass); bit-identical | lab1 83007 probe 71,534 so với 72,872 (−1.34k, 5/5); lab2 83021 −1,275 (block-paired SE 85, 8/8), E[max2] −1,224; c1 K/Q/V 13.48/26.65/14.04k → 13.08/26.05/13.45k; rate 500 → 515 B/c (+3.0 %), stream đôi 581 → 594 | **CÓ LỢI** (V10, mang sang V11 → V13) | được −1.2k stream; mất: không | độ dài run 8 hàng mới là biến, không phải khoảng hop | V10-E301.md, qkv10_notes.md; qkv_h4.rs:1424-1550 |
| far (probe) | DMN = bit hàng 8/7, run 32 hàng, cùng hop như far2 | 83007: 72,779 so với 72,872 (null), stream đôi 581 | TRUNG TÍNH | – | cùng hop mà không có run 8 hàng → hop không phải biến | V10-E301.md |
| farq | layout far chỉ cho Q, gather thật | 83007: 72,867 (null) | TRUNG TÍNH | – | như far | V10-E301.md |
| wonly (arm chỉ-DMA) | cùng 3 DMA trọng số, không có gì khác | 83007: 57,850; stream đôi 570 B/c < trong kernel 582 | phát hiện | – | context TU của kernel không làm chậm stream; biến là thứ tự HBM | V10-E301.md |
| L3 p1–p4 giải mã lại | giải mã các remap của qkv9 bằng descgeo.py | p1 +2 %, p2 +11–17 %, p3 +4–12 %, p4 +15–19 % | ĐÓNG BẰNG PHÂN TÍCH | – | thua vì engine/cluster dùng chung dải địa chỉ, không vì hop rộng | V10-E301.md |
| E303 f3 | run 4 hàng | 83098: +1,738 so với far2 (SE 286, 6/6 tệ hơn), stream đôi 572 | CÓ HẠI | – | run ngắn hơn thì tệ hơn | V10-E303.md |
| E303 f4 | hàng của mỗi slice liên tục (15 KB, Q 30 KB/slice) | 83098: +8,032 (SE 471), stream đôi 518; static 36,754 | CÓ HẠI | – | run dài liên tục trên từng slice tệ hơn nhiều (như L3 p3) | V10-E303.md |
| E303 f2n | tập hàng của far2 với hop DMN gần (123/246 KB) | 83098: +231 (SE 159, 3/6) | TRUNG TÍNH (= far2) | – | khoảng hop DMN không quan trọng | V10-E303.md |
| run 16 hàng (V10, judge) | hàng xóm chưa thử của far2 | kỳ vọng ≤ 0.3k | ĐÓNG BẰNG PHÂN TÍCH (DROP) | – | trong 4/8/32/64 thì 8 tốt nhất | IDEAS_RANKED.md |
| V12-E702 r16 (qkvA13) | run trọng số 16 hàng thay 8 | 91595: +1,019 | CÓ HẠI | – | 8 hàng tốt nhất (4/16/32/64 đều tệ hơn) | BOTTLENECK_LEDGER V13 |
| qkvOKV (ý tưởng X3 từ note công khai "địa chỉ DM trọng số lẻ ×256 B → contraction 4.9k thay 2.1k") | factor pad ngoài cùng `Dummy2 = 1 # 2` dời offset DM của K/V từ lẻ-256 (3,328 / 11,520) sang 32,768 / 65,536 | static và list giống hệt V11.1; align audit WEIGHT odd 2 → 0; Arena info 91165/91167: QKV warm mean 71,169 so với 71,214 | TRUNG TÍNH (≤ ~0.5k) | – | luật offset lẻ không tái hiện với shape K/V của ta, nên H-LUT bị hạ ưu tiên | ledger dòng 58; experiments/v12okv/ |
| V12-E703 parity K/V (qkvA13) | chẵn lẻ / căn chỉnh K/V | 91595: NULL (contraction 3,117 so với 3,115) | TRUNG TÍNH | – | căn chỉnh đã đóng | ledger V13 |
| qkvRate (job 90846; 5 arm × 8 block × 4) | đo B/tick mỗi cluster của các layout: far2, BW, **K15** (K tách 15 chunk H 256 B, slot slice thứ 16 pad), XQ (byte Q theo shape AO K16), XO (byte O theo shape Q far2), Up U0, AO K16 | far2 K 287/281, Q 298/299, V 287/289 (55,765 trace / 55,918 info); BW = far2; **K15 K 296/301, Q 306/310, V 295/295 (53,728 / 53,694 info) = −2.04k (−3.7 %, ~17 SE)**; XQ nhanh, XO chậm; Up 312/312; AO K16 309/310 | CÓ LỢI (→ x15/x15k2 ở V13) | – | cách truy cập (access pattern) quyết định tốc độ, không phải tensor (hoán đổi XQ/XO); thiếu hụt của QKV là do hàng 3,840 B chia theo luỹ thừa 2 | qkvRate_notes.md |
| bảng rate qkvRate (vết V11.1 + V11) | B/tick mỗi cluster kể cả chi phí cố định | K 286/296, Q 298/300, V 284/286, AO O.a 306/309, O.b 271/290, FFN Up 309/311, Gate 309/311, W12 309/317, W3 280/291, UpScale 189/202 (packet 960 B), GateScale 178/171 (800 B); DMA-only lockstep ~572 chip | đo đạc | – | W12 có đúng cấu trúc far2 nhưng stride không luỹ thừa 2 | qkvRate_notes.md |
| x15 = K15 riêng (qkvB13) | mỗi hàng trọng số 15 chunk 256 B, 1 chunk/slice, slot 16 pad, chunk mask, contraction có mask + reduce-scatter | 91675 info +1,440 (SE 139); 91814 +1,654 (SE 199) 0/8; trace 91691: DMA K/Q/V −1.3k (stream có lợi thật) nhưng contraction Q có mask 5,230 so với 2,990, V 4,079 so với 2,000 | CÓ HẠI | được −1.3k stream; mất +2.9k contraction | contraction có mask 16 slot ăn hết lợi stream (T11: contraction có mask chậm 2×) | ledger V13; v131V_mech.md §4 |
| x15 chi tiết trace (lab2 91691, c1) | – | DMA K 13,274 → 12,873, Q 26,124 → 25,334, V 13,629 → 13,112; Q land −2,419; pass Q 3,114 → 5,191; V issue − Q-contraction end 1,696; DMA idle từ Q land tới V start 3,155 → 7,637; V land +1,345; kết thúc +4.4k trace (x15 +4,965 trace so với +1,440/+1,654 info) | phát hiện | – | các arm K15 có thêm ~3k chi phí chỉ-trace | v131V_mech.md §4 |
| x15r (qkvB13) | biến thể x15, K đi trước do vô tình (memset K15 mask) | +2.6k; trace 91738: Q contraction bắt đầu 652 sau khi Q land (RoPE ramp nằm trong cửa sổ K→Q); k_scale dời ra sau Q contraction | CÓ HẠI | – | cùng chi phí contraction + va chạm với ramp | ledger V13 |
| lab4 91814 (info 8×3, 16 warm/arm) | x15 / x15k2 / x15v so với a_i | x15 +1,654 (199) 0/8; x15k2 −362 (285) 6/8; x15v +199 (265) 2/8 | x15 và x15v CÓ HẠI; x15k2 → ứng viên | – | head x15 là K trước (K@1003 … Q@10371); x15k2 là x trước (rms_w@1003, x@1913, K@2823 …) → khoảng ~2.0k HW là giá của một head K-trước KHÔNG kiểm soát | analysis/v14/qkv_review §1 |
| **x15k2 = K15 + ksq2 + K-scale exact-zero pin (→ V13.1)** | layout K15 + thứ tự ksq2; file mới qkv_k15b.rs + `stage_rope_rows_r17o` + `normalize_key_p1/p2` | lab so với V12: 91814 −362 (285) 6/8; 91889 −429 (121) 9/10; 92953 −541 (102) 14/15; **IV pooled −485 (SE 75) z −6.4**; common-variance −464 (90); 92075 −494 (71); max2 −581; TRACE +1.1k (lỗ!); static 38,576 so với 36,497 | **CÓ LỢI** (V13.1) | được: K15 stream nhanh hơn, Q land −1.28k; mất: pass Q dài hơn, có mode stall PE-issue; cold run 0 +1.2–5.8k (không ảnh hưởng median) | chỉ lợi ở INFO (mức grader); ví dụ rõ nhất cho "static và trace đều nói lỗ, INFO nói lãi" | V131_RESULT; ledger V13 |
| QKV-MIX (K15 cho vài trọng số, far2 cho số còn lại) | chọn layout theo từng trọng số | quy công trọng số (92075 trace): K −643 DMA, contraction nằm ngoài đường găng; Q −528 DMA so với +0.4–0.7k chi phí pass; V −995 DMA so với +794 pass; mix cần layout x thứ 2 (thêm DMA head idle ~1.5–1.9k) | ĐÓNG BẰNG PHÂN TÍCH (EV ~0) | – | chỉ phần lợi của K là miễn phí | critic §5; V14_PLAN §4 |
| covM + FREECHECKS CJ-3 (QKV K15) | độ phủ engine/slice | trọng số K15 phủ 60/64 slice mỗi engine (240/cluster), aligned 256 B; đo 304–311 B/tick/cluster, bằng hoặc dưới request law | phát hiện (0 tick) | – | hiệu ứng "240 trên 256 slice chậm 10 %" không thấy ở đây | FREECHECKS §3 |
| BACKWARD S15 / K5 (V13-V15) | sàn stream QKV | 31.46 MB ở 651 B/tick = 48.3k chip; mỗi cluster ≥ 50.4k kể cả 3 lệnh × ≥ 0.7k cố định; ta 52.5k → ≥ ~4k khoảng cách tới #1 là thời gian không-stream | ĐÓNG BẰNG PHÂN TÍCH | – | stream không phải đòn bẩy | analysis/v14/qkv_gap §0 |



#### 2.3 Nhóm C — Đầu kernel: nạp và phát tán x / rms_w, memset-pin ở đầu kernel

| tên/id | thay đổi (cơ chế) | kết quả | kết luận | được / mất | lý do, bài học | nguồn |
|---|---|---|---|---|---|---|
| Q29 M2 | nhân bản x bằng DM→DM đi thẳng vào layout projection | CF "visa: while lowering Dma / projections have different number of dimensions" | KHÔNG BIỂU DIỄN ĐƯỢC | – | DM→DM không thể vừa nhân bản qua 64 nhóm hàng vừa tách H; không đổi được Cluster (0.6); muốn nhân bản phải đi từ HBM | OL§5.5 |
| Q33 QKV-E2 | nạp x từ HBM thẳng vào 8-slice ReducingSlices | 55,215 (−447, −1 op) | CÓ LỢI (sau bị E5 thay) | – | `H/480` là trục thật của `m![H]` nên load được chia; `m![1#2]` vẫn nhân bản | TSV:88 |
| Q36 QKV-E5 | input RMSNorm in-place trên layout contraction; x + rms weight nạp nhân bản; tổng giữa các shard qua trục reduce dummy SỐNG `Dummy256 % 4` | 55,129 (−533, −3 op, idle −822) | CÓ LỢI (nằm trong E8, chưa đo riêng) | – | bỏ round trip HBM của x; inter-slice reduce với dummy sống = broadcast miễn phí | TSV:91 |
| Q53 **r12** (74981, 6 lần xoay vòng) | r11 + x nạp nhân bản 32× vào `m![Qs/64 % 32, H/480]`, RMS trên cả 256 slice, `CustomBroadcast{ring_size:8}` thay pass ring-256 17,128 cycle (qkv_r5.rs:102, lệnh TU dài nhất) | static 41,979 (tệ hơn v11 39,002; broadcast 8,455 → 504); work 84,263 so với 86,420; total 90,631 so với 95,026 (−4.6 %) | CÓ LỢI (lab) | – | static tệ hơn nhưng HW tốt hơn; load nhân bản 240 KiB tốn 2.7k HW mỗi cái; DMA FIFO: K issue lúc 9.5k, Q +4.2k sau K, V +3.5k sau Q → lane idle 17k | qkv-r12; qkv-e1-run/src/device/sliding/qkv_r12.rs |
| Q58 r18 (76912) | r17c + `Broadcast1{slice1:8, slice0:1}` (không config DMA, đã kiểm trên emulator) + normalize cuối của K gate theo gather Q | 40,914; work 83.3k; total 90,983; lane DMA xong lúc 66.6k (r12 71.4k); K issue 7.5k, Q +0.6k, V +0.6k | CÓ LỢI (lab) | – | DMA đã ổn, giờ hàng đợi Main in-order là nút thắt (~22k pass so với stream V 16.4k; V contraction chờ ~5.7k) | qkv-e1-run/qkv_r17.rs + qkv_r18_norm.rs |
| QA1 ao-qkv h2 (E2) | DMA x và input_rms_weight vào 2 tile của cùng một DM tensor để không có TU xen giữa các DMA đầu (mean-square dời sang Main) | 41,243; +3k | CÓ HẠI | – | load nhân bản vào tile chậm hơn nhiều trên HW (x 4.4k, rms_w 5.7k so với 2.7k), vẫn tuần tự trên engine DMA; stream tile-view + reduce trên Sub → lir ICE | p081/ao_qkv_notes.md |
| QA2 ao-qkv h1 | mean-square dùng VRF weight (x²·w/w) để ép DMA x đi trước rms_w | 39,874 (không chạy) | ĐÓNG BẰNG PHÂN TÍCH | – | scheduler vẫn đặt rms_w trước | ao_qkv_notes |
| QA10 qkv8 E5 probe16 (79618) | x và rms_w nhân bản 16× thay 32× (bỏ RMS, chỉ đo thời gian) | 37,965; x 2,754 → 1,957, rms_w 2,742 → 1,967; T 76,688 → 75,304 | phát hiện | – | chi phí load nhân bản ≈ 1.2k + 6.4 cycle/KiB | qkv8_notes |
| QA11 qkv8 E6 **x16** (79637) | x và rms_w trên `[Qs/128 % 16, 1 # 2, H/480]` (pad cặp nằm ngoài trục shard; `[.., H/480, 1 # 2]` lỗi "VRU reduce axes must be innermost"), rồi `Broadcast01{slice1:2, slice0:8, time0:30}` (đã kiểm trên host) phát 8 shard cho mọi slice | 37,673; T h4svtg 76,608 → x16 75,537 (−1.07k); x16n 76,572 | CÓ LỢI | – | Broadcast01 tốn 1.32k HW trên Main lúc Main rảnh | qkv8_notes |
| QA12 qkv8 E7 x16w / **x8w** (79660) | w: nạp rms_w TRƯỚC + mean-square phụ thuộc nó qua exact zero (`sum(0*w)`), nên K được list ngay sau x; x8w: chỉ nhân bản 8× (`[Qs/256 % 8, 1 # 4, H/480]`, `Broadcast01{4,8,30}`, trục QkvQuad) | 37,519 / 37,080; T x16 75,781, x16w 75,990, x8w 75,288 | x8w CÓ LỢI; x16w TRUNG TÍNH | – | V land sớm hơn 1–2k nhưng scheduler dời k_rms ra sau Q; stall clock-control của V quay lại ở ~9/20 launch (stall khi TU idle < ~0.95k trước contraction; sau này hoá ra là artefact trace) | qkv8_notes |
| QA13 qkv8 E8 x8g / x4g (79688) | g áp cho Q (Q gather theo thứ tự kênh, `finish_query_f`, Broadcast1{64,1}, 8 hàng theo time); x4g: nhân bản x/rms_w 4× (`Broadcast01{8,8,30}`, trục QkvOct) | 37,080 / 36,848; T x8w 74,921, x8g 74,536, x4g 74,739 | cả hai là ứng viên | – | x4g: Broadcast01 ring-64 tốn 4.4k trên Main; nạp TRF chặn pass `pos` (pass Main không overlap với một StoTrf đang chạy trên Sub) | qkv8_notes |
| QA14 qkv8 E9 (79705, 16 warm mỗi arm) | x8g so với x4g | T x8g 73,712, x4g 74,911; cả hai 0/32 V stall | **x8g CÓ LỢI** (vào v8); x4g CÓ HẠI | – | – | qkv8_notes |
| H3 shard head aligned (V10) | 10 × 768 B (`H/384 # 32`) thay 8 × 960 B không aligned | trần ≤ 0.4k | ĐÓNG BẰNG PHÂN TÍCH | – | phần per-KiB của E5 chỉ ~0.38k mỗi load | CRITICAL_PATH_QKV.md H3 |
| steering đầu QKV (V10) | x lên ở lúc dispatch zero-pass của rms_w (1.75k sau launch) | – | ĐÓNG BẰNG PHÂN TÍCH | – | không có cơ chế mới (zero pin cần một op TU nằm giữa 2 load); cùng không gian với qkv8 E7 | IDEAS_SCHED.md s6 |
| qM1 memset-pin đầu QKV (V12-E602) | Sub memset 11.5 KB/slice + zero pass trên Main ghi tile 0 | static: memset, x, rms_w, K lên đầu list (+1,197); PE: preload fence X23 giữ rms_w/K tới khi memset xong | ĐÓNG BẰNG PHÂN TÍCH | – | chỉ dương nếu memset trên HW < ~1.5k (khó tin với 1.7k static); xem lại nếu microbench memset cho < ~1.2k với 11.5 KB | V12-E602.md, MEMSET_PIN_AUDIT.md s3 |
| qM2 | biến thể 6.4 KB/slice | x trước to_vrf(w) nhưng K sau to_vrf(zero) (idx 121 < 122) → K bị gate ~2.8k | ĐÓNG BẰNG PHÂN TÍCH | – | – | V12-E602.md |
| các site memset khác của QKV (audit #5, #7, #10, #12, #13; site của teammate) | – | memset không input luôn đặt ở đầu kernel; to_vrf / gate TRF không có output DM; v_scale không găng (slack 752); config Broadcast có tên không có config DMA | KHÔNG BIỂU DIỄN ĐƯỢC / 0 | – | – | MEMSET_PIN_AUDIT.md s1, s3 |
| mp (qkvB13) | memset-pin đầu QKV để dời x lên sớm | 91675: +2,027 | CÓ HẠI | – | – | ledger V13 |
| bhd8 (qkv15) | memset giữ để trì hoãn rms zero pass | static 37,972; k_scale vào head; trên HW, Sub to_vrf của rms_w chờ memset ~3k | ĐÓNG BẰNG PHÂN TÍCH | – | memset trên HW chiếm Sub ~4× thời gian static | ARCH_QKV_B §6.1 |
| K8 (qkv_gap §5) | x/rms_w không nhân bản + fan-out trên chip | phân tích: fan-out qua 8 nhóm shard cần ring-256 (~62k Main) | ĐÓNG BẰNG PHÂN TÍCH (sau bị g6z bác bỏ) | – | g6z dùng một bản mỗi PE + ring-64, rẻ hơn | qkv_gap §5; GAP_QKV |
| kqa (qkv17) | K-scale pin thêm +0 từ POS | static 38,326 (kéo rope_off + k_scale vào head) | ĐÓNG BẰNG PHÂN TÍCH | – | head sai | qkv17_progress.md |
| hxu / hxm / hxukq (qkv17) | head: rms_w pin + VRF weight gate bằng chuỗi giữ 3 pass trên Main, gốc từ DM chưa khởi tạo / memset Sub nhỏ | static 38,582 / 37,598 (memset list trước x) | ĐÓNG BẰNG PHÂN TÍCH | – | – | qkv17_progress |
| hk0 / hk1 (qkv17) | ép k_scale vào head (zk = AND0(k_scale)·rms_vrf) | 40,116 (K trước) / 37,540 (x trước, ước +1.35k) | ĐÓNG BẰNG PHÂN TÍCH | – | – | qkv17_progress |
| **fw** (qkv17) | input RMS weight nạp MỘT lần mỗi PE, trải trên 4 nhóm hàng (30,720 B, piece 128 B, 60/engine ~1.3–1.4k, so với nhân bản 120 unit/engine 1.65–1.9k) + fan-out ring-64 trên Main `Broadcast1{slice1 4, slice0 16}` | static 37,100; lab1 100190 −433 (142) 6/6; lab3 100226 −520 (134); lab4 100276 −276 (222); pooled −446 (SE 89); trace 100204 −937 | CÓ LỢI (KEEP-READY, sau bị g6z thay) | được: head load rẻ hơn (1.37k so với 1.65k) | nhờ head ngắn hơn, scheduler đặt k_scale trước K | qkv17_progress |
| fxw / fxwm / fx (qkv17) | fw + x cũng trải + fan-out x trên Sub / Main | fxw −287 (114) / −432 (129), trace −1,709; fxwm static 37,112; fx 37,186 (rope_off vào head có wait) | fxw CÓ LỢI (bị thay); fxwm, fx ĐÓNG BẰNG PHÂN TÍCH | – | – | qkv17_progress |
| g1 / g2 / g4 / g5 / g3 (qkv17) | các tổ hợp fan-out Main/Sub | static 37,379 / 37,737 / 37,519 / 37,688; g3 (main/main) lab1 +561 (172) | CÓ HẠI / ĐÓNG BẰNG PHÂN TÍCH | – | – | qkv17_progress |
| g6 (qkv17) | rms trải + fan-out trên Sub, pin trên weight đã trải, x 16× | 37,021; lab3 −532 (160) 5/6; lab4 −356 (172) | CÓ LỢI (bị g6z thay) | – | – | qkv17_progress |
| **g6z (→ V16)** | g6 + zip RoPE (bzr) | 37,021 (list DMA giống g6); lab4 100276 −853 (224) 6/6; lab5 100292 −814 (223) 7/8; lab6 100316 −573 (92) 6/6; lab8 100360 −735 (148) 8/8; **pooled −660 (SE 70)**; trace 100323 −1,555; NH 100373 70,138 | **CÓ LỢI** (V16) | được −0.66k; mất: head thật ra chậm hơn (K land +1.3k: x DMA 2.2k so với 1.67k do switch fan-out Sub chạy bên cạnh) | nhờ static head ngắn hơn, k_scale lên trước K → Kc bắt đầu đúng lúc K land, chuỗi cửa sổ Q xong sớm, V hiệu dụng −0.8k; khi V land sớm thì bzr mới cộng dồn được | qkv17_progress; V16_RESULT |
| g6zn (qkv17) | g6 + bzrn (bỏ pin K-norm p2) | 37,105; lab5 −896 (197) 7/8; lab6 −485 (230) 4/6; trace lab7 +1,049 (Qc chậm 5.3k) | TRUNG TÍNH (chưa xác nhận) | – | nhiễu | qkv17_progress |
| g1z / g1zn / g2z / g2zn (qkv17) | x cũng trải + fan-out, thân zip | 36,852 / 37,111 / 37,038 / 36,697; g1zn lab6 −261 (187) | CÓ HẠI / ĐÓNG BẰNG PHÂN TÍCH | – | – | qkv17_progress |
| fwz / fwzn / fxwz (qkv17) | fw + các biến thể bzr | static 37,529 / 37,337 / 37,642 | ĐÓNG BẰNG PHÂN TÍCH | – | – | qkv17_progress |
| g7z / g8z / g9z / g11z (qkv17) | fan-out gate muộn bằng +0.0 lấy từ mean square; các bản copy nhẹ trên Sub của pin zero | 37,145 / 38,092 / 38,330 / 37,245 (rope_off vào head) | ĐÓNG BẰNG PHÂN TÍCH | – | mean square hoà (tie) với begin của k_scale | qkv17_progress |
| g10z (qkv17) | fan-out trên Main + 5 bản copy nhẹ trên Sub của pin zero | 37,444; lab8 100360 +842 (136) so với x15k2 (+1,577 so với g6z) | CÓ HẠI | – | mỗi pass Sub list trước một DMA đều gate DMA đó (FS trước khi nạp program) | qkv17_progress |
| qf7xs / qf7xm (qkv18) | head qf7 + x trải một bản mỗi PE, fan-out Sub / Main | xs static 38,761 (cửa sổ Q quá tải); xm 36,832, 3 wait PASS; lab4 101005 −102 (131); lab5 101114 +48 (104) → pooled ~−10 | TRUNG TÍNH (KILLED) | – | – | qkv18_report |
| qf7wm / qf7wmxs / qf7wmxm (qkv18) | fan-out rms-weight trên Main (x nhân bản) | wm 37,427 PASS, lab5 +171 (140) 2/8; wmxs 38,328 / wmxm 37,035 (có wait trước Q/K) | CÓ HẠI / ĐÓNG BẰNG PHÂN TÍCH | – | – | qkv18_report |

#### 2.4 Nhóm D — Thứ tự trọng số và điều khiển list DMA (pin, steering, small loads)

Nguyên tắc chung rút ra (xem thêm §6):
- Thứ tự trong source không có tác dụng.
- List static đặt DMA theo thứ tự các consumer của nó.
- Khi hoà (tie), pass TU thắng.
- Load nhỏ được đặt ALAP ngay trước consumer; nếu đụng một load lớn thì bị đẩy lên sớm hơn.
- Công cụ duy nhất là thêm phụ thuộc dữ liệu: exact-zero pin, memset pin, operand swap, identity-copy chain.
- Trên HW có một list in-order duy nhất, bị chặn head-of-line. Mỗi DMA nhỏ tốn ~0.8–1.2k ở bất cứ chỗ nào nó nằm.

| tên/id | thay đổi (cơ chế) | kết quả | kết luận | được / mất | lý do, bài học | nguồn |
|---|---|---|---|---|---|---|
| Q08 QKV-005/006 | lấp trước buffer `to_dm_view`; buffer gather q+k dùng chung | thứ tự byte-identical / CF (fetch `Qf/512 % 2 # 8` lệch) | TRUNG TÍNH / KHÔNG BIỂU DIỄN ĐƯỢC | – | thứ tự chương trình, thứ tự store, `to_dm_view` đều không dời list DMA | OL§3 |
| Q12 QKV-010/b/c | tiêm phụ thuộc zero-VRF lấy từ *trọng số* | 68,897 / 68,897 / 73,467 | CÓ HẠI | – | 010c có x,Q,K,V liền nhau nhưng post 1 slice ~30k lại land sau 42k; zero lấy từ trọng số gate Main cho tới khi mọi trọng số land | TSV:46-48 |
| Q13 QKV-011 | đảo head-local + zero lấy từ *kết quả* K,V đưa vào pass channel-scale của q | 67,833 | CÓ HẠI (nhưng mẹo tái dùng được) | list DMA x, rms, Q0, Q1, K, V liền nhau (trọng số xong lúc 40.7k) | ~20 pass nối tiếp nhỏ → đuôi 27k; zero từ kết quả thì sắp xếp lại mà không gate: mẹo tái dùng | TSV:49 |
| Q18 QKV-014 | 013 + zero phụ thuộc (q ← K,V) | 66,262 | CÓ HẠI (cơ chế giữ lại) | cả 4 load trọng số liền nhau, xong 38,824, KHÔNG có idle | 3 chuỗi post trong đuôi 27k; chỉ thắng nếu post không có DMA | TSV:58 |
| Q19 QKV-015 | tách phụ thuộc q←K, k←V | 69,435 | CÓ HẠI | mất 5.1k idle tự gây | hop HBM cos/sin giữa hàng đợi: ~1,000 độ trễ ghi→đọc HBM mỗi chặng | TSV:59 |
| Q25 QKV-DEP | zero q → K channel-scale | 58,388 (+82) | TRUNG TÍNH | – | hàng đợi không đổi; vị trí đặt zero mới quyết định | TSV:66 |
| Q47 **trì hoãn RoPE** (09-22) | dependency VRF `+0.0*sum(k)` để ~16 pass RoPE nhỏ sớm được list sau trọng số K | 71703 work −2.9 % (4/4), K issue 12.7k → 8.7k; 72017 ×10: work −2.73 % (9/10), total −1.92 % (6/10) | **CÓ LỢI** (main 09-22) | ~+0.6–0.9 % geomean; output bit-identical | HW issue theo thứ tự list static và bị chặn head-of-line; pass nhỏ ở đầu list làm trễ trọng số (hazard Q47) | dma-docs/FINDINGS §1, apply_qkv_rope_defer.py |
| Q54 r13 / r13b | pass copy "chạm" (Main hoặc Sub) để kéo load nhỏ lên sớm | vị trí không đổi | CÓ HẠI | – | pass độc lập được đặt cạnh consumer, không ASAP | MEM:furiosa-static-scheduler-dma-placement |
| Q55 r15 (75191) | operand swap + `rope_offset` nhân bản ra 256 slice | work 96.1k (+11 %); total 106,091 so với r12 95,931 | CÓ HẠI | operand swap đứng riêng thì có lợi: Q issue 0.55k sau K (trước là 4.2k) | tensor 4 byte nhân bản ra 256 slice = 7.4k HW (bị giới hạn bởi số descriptor) | same |
| Q56 r16 (76874) | RoPE ramp không phụ thuộc k, chạy sớm trên Main | work 86.2k; total 91,221 so với r12 88,549 | CÓ HẠI | – | ramp nằm giữa K và Q, chặn issue Q | same |
| Q63 r22 (77009, 77036) | r21 + chuỗi RoPE gate theo rms(K) | work 83.0k/83.6k; total 92,108 / 87,085 | CÓ HẠI | – | – | qkv-r12 |
| QA17 qkv9 steering list static (qd, g1, g2, g12, bk, a1vk) | epilogue contraction nhân chính xác 1/16 lấy từ load nhỏ kế tiếp; K mean-square + exact 0 lấy từ Q contraction (k-pin) | 37,215–38,192 (không chạy; hwsim +0.3..+3.1k) | ĐÓNG BẰNG PHÂN TÍCH | – | scheduler kéo mọi load nhỏ mà contraction cần lên TRƯỚC trọng số đó; pass consumer của nó lại gate issue trọng số (K lúc 4.5k thay vì 2.8k); index lệnh = thứ tự đặt; TU thắng khi hoà; không có scheduling hint | qkv9_notes |
| QA19 qkv9 L4 (80381) | e4 = VRF k_scale bằng pass Main; e4v = + A1 cho V; e4qv = + A1 cho Q,V; a1qv | T base 73,652, e4 74,085, e4v 73,474, e4qv 73,782, a1qv 73,460 | TRUNG TÍNH / CÓ HẠI | – | hwsim dự −0.35/−0.4/−1.1k; phụ phí lane của từng cluster bị cluster kia hấp thụ (mô hình băng thông chung) | qkv9_notes |
| QA20 qkv9 L5 (80412) | pin k_rms ra sau Q: pass CUỐI của K normalize cộng exact-0 VRF từ Q contraction (`zero_from_q_contraction`): bk2 / a1vk2 / a1qvk2 / a1v | T base 74,188, bk2 76,548 (+2.4k; V contraction stall 5.3k), a1vk2 75,109, **a1qvk2 73,168 (−1.02k, 10/10)**, a1v 74,345 | a1qvk2 ứng viên; các biến thể khác CÓ HẠI | – | có A1 cho Q thì Q contraction 2.9k (trước 5.06k), Main xong chuỗi Q→V ~3.9k trước khi V land | qkv9_notes |
| QA21 qkv9 L6 **a1qvk3** (80441, 14 warm) | pin vào exact 0 lấy từ q ĐÃ GATHER (`zero_from_q_heads`), nên V được list sau Q gather | T base 73,873, a1qvk2 73,286 (−0.59k), **a1qvk3 72,960 (−0.91k)** | **CÓ LỢI** (vào v9) | V land sớm hơn a1qvk2 0.4–1.0k; đuôi 6.85–7.0k; sai số giống v8 | – | qkv9_notes |
| H5 load nhỏ trên 1 engine / FIFO từng engine (V10) | ít engine PE hơn cho mỗi load nhỏ | 0 lần hoàn thành sai thứ tự trong 64,350 cặp DMA (82 trace); lợi 1 engine ở qkv8 E3 ≤ 0.26k | ĐÓNG BẰNG PHÂN TÍCH | – | 0.8.1 std không có API chọn engine; hoàn thành theo thứ tự issue | CRITICAL_PATH_QKV.md s4a |
| H6 gộp / dời load nhỏ (V10) | gộp thành ít lệnh hơn, hoặc dời lên đầu/xuống cuối | mô hình: ở đầu/cuối trả 1:1 so với ~0.3 hiện nay; mỗi DMA chỉ một tensor nguồn | ĐÓNG BẰNG PHÂN TÍCH | – | tensor HBM tách biệt, layout cố định | CRITICAL_PATH_QKV.md s4b |
| H7 pha load nhỏ bất đối xứng theo cluster (V10) | bản của c0 trong pha K→Q, của c1 trong Q→V (pad cluster `1 # 2`) | trần 0.08k / 1.46k | ĐÓNG BẰNG PHÂN TÍCH | – | cần DM tensor có 2 writer; to_dm_view vào tile đã FAIL trên HW (ao9) | CRITICAL_PATH_QKV.md H7 |
| qkvMB (microbench từ trace) | DMA nhỏ xếp sau một trọng số đang chạy có giấu được độ trễ cố định không? | rope_offset vào lane rảnh 1.27–1.36k, xếp sau K thì land 0.81–0.93k sau K; nhỏ sau nhỏ +0.75–1.16k; bảng FFN 4 KB lane rảnh 2.0–2.3k, xếp sau 1.6–1.9k; AO ws rảnh 1.6k, xếp 1.4k; no-op 0.16–0.2k rảnh / 0 khi xếp | LUẬT (đo đạc) | – | xếp hàng chỉ giấu được 0.2–0.6k (12–35 %); 0.8–1.7k vẫn tuần tự sau lệnh trước → steering load nhỏ của QKV trị giá tối đa ~1–2k | qkvMB_notes.md |
| qkvD2 V1 / V2 / V3 | dời nguyên văn các load nhỏ qua các pha trọng số | chuỗi x giới hạn bởi TRF 12.2k HW; slack Main sau Q median 1.5k / min 0.74k; consumer của V3 đều trước V | ĐÓNG BẰNG PHÂN TÍCH | – | đòn bẩy thật: DMA nhỏ list TRƯỚC contraction của trọng số liền trước (TU thắng khi hoà); qsim: R trước K ctr −0.70k; KR trước Q ctr −0.33k | qkvD2_notes.md |
| qkvD2 v2qa | chuỗi zero trên Sub sau to_trf → AddF zero ở epilogue K contraction | static 39,011 (+2,514) | ĐÓNG BẰNG PHÂN TÍCH | – | chuỗi pin chỉ ASAP khi nó găng, nếu không thì ALAP cạnh consumer | qkvD2_notes |
| qkvD2 re1 | RoPE ramp gieo từ x đã tách hi/lo | static 37,713; scheduler đổi trọng số thành Q,K,V (v_scale vào head); qsim +2.7k | ĐÓNG BẰNG PHÂN TÍCH (sau Q-first được hồi sinh thành qf7) | – | – | qkvD2_notes |
| qkvD2 v2qb | chuỗi zero trên Main sau switch390 | R đặt trước K ctr nhưng commit pos nhảy lên trước; k_scale + K ms ra sau Q; qsim +1.2k | ĐÓNG BẰNG PHÂN TÍCH | – | slack Main tĩnh trước Q ~1.4k, trên HW ~6k | qkvD2_notes |
| qkvD2 po3 | to_dm rope_offset trước project trong thứ tự chương trình | lịch giống hệt base | TRUNG TÍNH | – | thứ tự chương trình không có tác dụng | qkvD2_notes |
| qkvD2 v2l | rope_off + k_scale ra sau Q (pos và VRF K-scale pin vào zero của q-heads) | static 37,685; hết K→Q nhưng 5 load nhỏ sau Q, V list sau các pass bị pin; qsim +2.0..+2.7k | ĐÓNG BẰNG PHÂN TÍCH | – | lặp lại bài học qkv9 | qkvD2_notes |
| qkvD2 v3l | k_rms/q_scale/q_rms ra sau V | scheduler đổi trọng số thành K,V,Q; static 40,799; qsim +6.0..+6.4k | ĐÓNG BẰNG PHÂN TÍCH | – | – | qkvD2_notes |
| qkvD2 ksq | VRF K-scale = pass Main scale·one_vrf (chính xác) | static 36,448; R lúc 9,499 trước K ctr, Q ngay sau R, nhưng K norm ra sau Q; qsim −64..+172 | TRUNG TÍNH (static/model) | – | – | qkvD2_notes |
| qkvD2 **ksq2** | ksq + Q contraction pin vào exact zero lấy từ K rms (reshape head→QRows + BitAnd), K normalize tách p1/p2 | static 36,418; qsim −0.32k | mang sang V13 (ghép vào x15k2) | – | ý chính: slack Main tĩnh trước Q ~1.4k so với ~6k trên HW, nên việc Main bị pin trước Q có thể làm trễ Q ctr mà không tốn gì trên HW | qkvD2_notes; ledger |
| x15v / x15k2v (qkvB13) | cấp phát trước buffer V | x15k2v +1.1k HW; x15v +199 (265) | CÓ HẠI | – | – | ledger V13; V14_PLAN §4 |
| k2 = ksq2 trên far2 (qkvB13) | pin exact-zero K-scale dời K gather / rope staging / K-norm p1 vào cửa sổ Q-DMA, trên layout far2 của V12 | 91675: +1,007 (qsim dự −0.32k) | CÓ HẠI | – | qsim đoán sai về steering list | ledger V13 |
| QKV-H CR-1 h1 (qkv14, V14-E701) | head K-trước + RoPE ramp dời lên head, gieo từ memset (qsim A2 −1,513) | static h1/h1n/h1r không bao giờ list K trước; ramp gieo gate theo K (h1k) không lower được (visa SRAM alloc) | ĐÓNG BẰNG PHÂN TÍCH (static kill, 0 Arena) | – | chính chuỗi dài treo sau K contraction mới làm K lên trước; dời ramp đi là mất nó | ledger V14 |
| QKV-H h0 / h0m / h0p / hq / h0x / h0y (V14-E702) | giữ x-trước; xếp k_scale sau rope_off và Q sau k_scale bằng pin | static: mọi pin dời POS/kz khỏi lúc DMA của nó land đều khiến scheduler kéo k_scale (và rope_off) vào head và đặt K ALAP sau các pass Sub của chuỗi x (ước K issue +2.6k HW) | ĐÓNG BẰNG PHÂN TÍCH | – | pin làm xáo trộn head | ledger V14 |
| QKV-H h2 / h2x HEAD-2 (V14-E703) | x trước, K thứ hai, rms_w sau K (qsim B −1,113) | static: rms_w vẫn đầu tiên, k_scale vào head, kz/ms gate x/K | ĐÓNG BẰNG PHÂN TÍCH | – | – | ledger V14 |
| qkv14 chi tiết 15 biến thể static | h1, h0, h0m, h1n, h1r, x15kp, hq, h0p, h1k/h1kn, h2, h2x, h0x, h0y | makespan h1 38,747 / h0 37,728 / h0m 37,798 / x15k2 38,576 (ref) / h1n 39,028 / h1r 37,911 / hq 37,811 / h0p 37,734 / h2x 37,289 / h2 39,320 / h0x 38,933 / h0y 38,583; h1k/h1kn không compile (visa V1-Modulo "Insufficient resources … Sram AllocRequest size 0") | ĐÓNG BẰNG PHÂN TÍCH | – | khi ramp không nằm trong chuỗi K-post thì scheduler static không bao giờ list K trước | experiments/qkv14_progress.md |
| HEAD-1 P1 review (qkv_review) | K trước + ramp ở head | SURVIVE có điều kiện, P ~0.25; "Q issue sớm 1.9k" của x15/x15r = stream K nhanh hơn 0.37k + ~0.93k do k_scale vắng mặt khỏi K→Q + chỉ ~0.6k do thứ tự head; hazard Q47; vách chi phí cold của ramp ở head | sau KILLED ở static (qkv14) | – | công cụ steering (memset pin) không với tới các pass có static ≥ 9.5k | analysis/v14/qkv_review §2 |
| P3 queued-head discipline | list các DMA nhỏ trước Q ra trước các pass TU của chuỗi x | reviewer: gộp vào P1; memset-pin không với tới (cần trễ static tới ~10–11k; một memset đủ dài chiếm Sub ~20k HW) | ĐÓNG BẰNG PHÂN TÍCH | – | không biểu diễn được như đã viết | qkv_review §2 |
| P4 HEAD-2 | x trước, K thứ hai, rms_w sau K | qsim B −1,113; có rms_w xếp hàng −952; Bn +8.5..+8.9k | ĐÓNG BẰNG PHÂN TÍCH (bị trội) | – | – | qkv_review §2 |
| CR-1 HEAD-1 đặt lại trên V13.1 | trên x15k2: c1 rms_w ở head idle 34 → 1,702, x idle 1,710 → 3,587 … | critic ước h1 0.4..1.4k P ~0.15; h0 0.3..0.8k P ~0.12 | → qkv14: đều KILLED | – | – | critic §6 |
| qsim what-if (list V12; không chạy) | A2n K-trước không dời ramp; R dời ramp; A2d; A3; A4; QF (Q trước); load nhỏ free; head free; scatter free; trọng số +3 %/+6 %; đuôi V bớt 1 pass Main; V contraction luôn nhanh | A2n +7,912..+8,458; R +8.3k; A2d −1,513; A3 +913; A4 −996; QF +9.6k; 5 load giữa free −2,221/−2,007; head x+rms_w free −1,617; v scatter free −1,474; +3 % −1,574; +6 % −3,058/−2,957; đuôi V −770; V ctr luôn nhanh −376 | ĐÓNG BẰNG PHÂN TÍCH (model) | – | CẢNH BÁO: qsim đoán sai các hiệu ứng list 3 lần 0.8–3k; "QF +9.6k" là sai (qf7 sau này −524) | qkv_gap §2.4, §3 |
| qkv15 s1a / s1b / s1c (QKV-A lane không idle) | mọi load nhỏ sau Q (k_rms, q_scale, q_rms) nạp trước Q, K norm p2 + K rope trong cửa sổ Q, V list lúc Q contraction | static s1a 37,360, s1b 42,793 (V TRƯỚC: loạn), s1c 38,771; lab1 94235 s1a +1,260 (171) E[max2] +1,082; trace 94261: 4 load nhỏ idle ở K→Q | CÓ HẠI | – | dời load nhỏ lên head cộng thêm thời gian "none" (không cluster nào stream) 1:1 | qkv15_progress |
| qkv15 s2a–d | biến thể pin k_scale kz + qz | static 38,267 / 38,228 / 38,355 / 37,781: k_scale bị kéo vào head, K sau Sub ms/Sqrt của chuỗi x (+2k HW); zero15_512 trên type 256 slice sinh thêm config DMA 719 static | ĐÓNG BẰNG PHÂN TÍCH | – | head hỏng | qkv15_progress |
| qkv15 h5a / h5b / h5c | cả 5 load nhỏ qua một chuỗi zero layout head vào K contraction; POS gate theo K gather; K epilogue trong cửa sổ Q | static 38,924 / 39,297 / 39,421; lab1 94235 h5a +2,300 (177) 0/8, h5b +1,637 (204); trace h5b K hiệu dụng 11.5k (so với 3.6k) | CÓ HẠI | – | – | qkv15_progress |
| qkv15 qv2 / qv3 | V (và load nhỏ của q) xếp ngay sau Q qua exact zero từ tile trọng số V, pin vào kết quả k rope | static qv2 40,229, qv3 38,503 | ĐÓNG BẰNG PHÂN TÍCH | – | – | qkv15_progress |
| bqk (qkv15) | q_rms dời lên trước Q; V list trước switch gather q | static 37,567, head FAIL; lab3 95861 +2,529 (160) 0/8 | CÓ HẠI | – | – | ARCH_QKV_B §6.1-6.2 |
| bqk2 (qkv15) | bqk giữ head x15k2; post-Q = 3 DMA ngay sau EXEC(Qc) | static 37,974; lab3 +339 (184) 2/8; trace 95894 Main slack trước V −0.15k | CÓ HẠI | – | cửa sổ V bị Main giới hạn | ARCH_QKV_B §6.2 |
| qs2 / qs2e | – | static 37,938 / 37,859; k_scale vào head | ĐÓNG BẰNG PHÂN TÍCH | – | họ s2 | ARCH_QKV_B §6.1 |
| bqk3 (bỏ K-scale pin) | – | static 39,714; head K-trước | ĐÓNG BẰNG PHÂN TÍCH | – | – | ARCH_QKV_B §6.1 |
| bqkk / bqkk2 | toàn bộ front-end phía Q (+ K epilogue trong cửa sổ Q) | static 42,670 (V trước ở head) / 42,392 | ĐÓNG BẰNG PHÂN TÍCH | – | thiếu pin q-gather của x15k2 thì chuỗi V lên trước | ARCH_QKV_B §6.1 |
| bqkk3a / 3b / 4a / 4c (Route 1) | + q-scale zero trong pin V; + K-scale pin nuôi bởi hàng 0 trọng số K | static 38,255 / 39,117 / 42,734 / 39,008 | ĐÓNG BẰNG PHÂN TÍCH (Route 1 đóng, 6 dạng static) | – | – | ARCH_QKV_B §6.2c |
| engine-0 confinement cho load nhỏ QKV | – | không build | ĐÓNG BẰNG PHÂN TÍCH | – | mọi load nhỏ QKV nuôi slice head 0/64/128/192 (4 engine); giới hạn vào 1 engine cần relayout switch trên Main (0.5–1.2k); lời ≤ ~0.35k mỗi load | ARCH_QKV_B §5.4 |
| x15k2 chi tiết HW post-Q | – | k_rms issue ~2.4–2.65k sau khi Q contraction 128 hàng bắt đầu trên CẢ HAI cluster → TUC hold; Qc và k_rms đều bắt đầu ở static 25,539 = kết thúc của reader page-0 cuối cùng (Sub rope to_vrf) → hazard WAR DM page-0 gate các DMA sau Q | phát hiện | – | – | qkv15_progress 18:45, 20:47 |
| qsw (qkv16) | swap operand q_rms/q_scale | static 37,725: list post-Q không đổi, head hỏng (k_scale trước K) | ĐÓNG BẰNG PHÂN TÍCH | – | dù biểu diễn được cũng chỉ ~0.05k ở INFO | qkv16 |
| ps0000..ps1111 / pz* (qkv16) | 16 hoán vị thứ tự chương trình trong cửa sổ V, mỗi họ | ps0000/0001/0010 lịch GIỐNG HỆT | TRUNG TÍNH | – | beam search xác định bởi DFG; thứ tự source bị bỏ qua | qkv16 |
| qf1 / qf1v / qf2 / qf2v (qkv18) | Q-first đối xứng: Qc pin vào q-scale zero (zero15_512), rope rows gate theo q đã gather, q-norm p1, Kc pin vào q rms | static 40,136 / 41,177 / 41,792 / 42,633 | ĐÓNG BẰNG PHÂN TÍCH | – | pin 256 slice sinh một hằng số compiler 8 KB + wait ở head; ramp (~20 pass Main) nằm trong cửa sổ K ngắn; đuôi 5.6k so với 4.2k | qkv18_report |
| msp / msp3 / msp4 / msp5 (qkv18) | các probe slack Main trong cửa sổ V | đều làm đổi list của V16 | TRUNG TÍNH (probe không sạch) | – | scheduler static mang tính toàn cục | qkv18_report |
| qf3 / qf3v / qf4 / qf4v / qf5 / qf5v (qkv18) | ramp gieo từ chunk mask, các pin Kc, tách k normalize | static 37,252 / 37,587 / 36,942 / 39,759 / 37,573 / 37,526 — 4–5 wait hoặc load nhỏ ở head | ĐÓNG BẰNG PHÂN TÍCH | – | Sub to_vrf k-scale chạy eager; pin POS kéo load nhỏ vào head | qkv18_report |
| qf6 (qkv18) | qf5 + Kc pin thêm zero từ k_scale | static 37,565, 3 wait PASS; lab1 100939 **+3,422 (130)** 0/6 | CÓ HẠI | – | RoPE rows nằm trong cửa sổ K → cửa sổ K bị Main giới hạn | qkv18_report |
| **qf7 (→ V17.1)** | qf6 + RoPE rows (POS, góc, cos/sin) gate theo chunk-mask seed → chạy dưới stream Q, rope_offset lên đầu head | static 37,623; list `rope_off ¦ rms_w W x ¦ Q W q_scale ¦ k_scale ¦ K W k_rms ¦ q_rms ¦ V` (3 wait PASS); lab1 100939 −636 (126) 6/6; lab2 100958 −413 (151) 8/8; lab4 101005 (qf7 làm control) g6z +459 (171); **pooled −524 (SE 84)**; cold k0 +2.9k ở mọi lab; trace 100979 +1.45k (artefact); NH 101221 69,459 | **CÓ LỢI** (V17.1) | được −0.52k; mất: run 0 cold +2.9k (median không ảnh hưởng); static +602 | Q epilogue dưới stream K, K epilogue dưới V; Q-first từng bị qsim ("+9.6k") và static V14 đóng sai | qkv18_report; V171_RESULT |
| qf7b (qkv18) | qf7 + q-norm p2 + q RoPE trước Kc | static 37,402 (list DMA = qf7); lab1 −296 (96) 5/6; lab2 +9 (198) 3/8 | TRUNG TÍNH | – | – | qkv18_report |
| qf9 / qf9k / qf9b / qf10 / qf10k / qf11 (V-early) | đưa q_rms ra khỏi nhóm post-K để V list trước K gather | static 38,347 / 38,915 / 39,277 / 36,942 / 37,187 / 36,942 — đều fail | ĐÓNG BẰNG PHÂN TÍCH | – | mọi phụ thuộc thêm đều rơi vào đường găng Kc hoặc làm hỏng head | qkv18_report |
| PS-1..PS-6 (bộ pin của báo cáo cloud QKV_GAP) | bỏ Q pin, bỏ gate ramp, pin packet-VRF, Main to_vrf cho load nhỏ | dự ~65.4k | TRUNG TÍNH / không áp dụng | – | qf7 đã có thứ tự đó; PS-5 đã nằm trong qf7; PS-4/PS-6 không áp dụng | analysis/v15/QKV_GAP_external.md |
| q19pq / q19pqk (qkv19) | buffer DM q_scale (/ k_scale) pad sang page 2 | list không đổi (q_scale 3 cycle sau Qc) | TRUNG TÍNH (KILLED) | – | WAR page-0 không phải cái đặt q_scale sau Qc | qkv19_report |
| q19pr / pf / prp / prpn / z / prz / prr / pfr / hprp (qkv19) | q epilogue trước Kc / pin Qc vào rows / biến thể head mask-first | static 38,276–40,107 | ĐÓNG BẰNG PHÂN TÍCH | – | beam search sắp lại head (pass chunk-mask 4,659 → 6,505); hằng số 8 KB | qkv19_report |
| q19h (qkv19) | head mask-first (chunk mask ngay sau x, zero4 & mask đưa vào pass ms của x) + q19n2 | static 36,484 (−1,382); lab10 102302 (bật screen) **+1,368 (121) 0/10** | CÓ HẠI (KILLED) | – | Kc bị gate bởi k_scale muộn + pin passes làm tràn cửa sổ V | qkv19_report |
| BACKWARD_QKV static: m0, a4, a5, r1a, r1b, r2, r2m0, qz, qza2 | gate load nhỏ vào head; sắp thứ tự q epilogue; pin Qc | a2 list `k_rms ¦ rope_off ¦ rms_w W x ¦ …` static 39,189; a4 head hỏng 36,484; a5 37,078 x lên trước; r1a/r1b visa "Cannot find evict target for Vrf" (ngân sách VRF); r2m0 39,393; **r3m0** 37,553 (−313); qz/qza2 37,413/37,612 (hằng 8 KB) | phát hiện (static) | – | consumer đầu tiên gate theo x sẽ đưa một load nhỏ vào head mà không có wait; gate từng load nhỏ riêng lẻ; q_scale luôn bắt đầu static cùng lúc với Qc | analysis/v18/BACKWARD_QKV.md |
| r3m0 (BACKWARD → qkv20) | q-norm p2 xếp trước k-p1 bằng exact zero, không pin Kc (dự −0.8k) | static 37,553; lab1 102755 **+3,172 (120) 0/8**; trace 102780 +4,497 | CÓ HẠI (KILLED) | – | cửa sổ K quá tải Main (68 % static → 118 % trên HW); đổi lane là trò chơi tổng bằng 0 (q_rms trước K làm Q→K +1.24k) | experiments/qkv20_report.md |
| a2 (BACKWARD → qkv20) | k_rms gate vào head qua POS gate theo x | static 39,189; lab1 **+3,421 (164) 0/8** | CÓ HẠI (KILLED) | – | cửa sổ Q bị nhồi quá (static +1,323 × HW 2.2–2.5 ≈ +2.9..3.3k ≈ INFO +3.4k: lần này static đoán ĐÚNG) | qkv20_report |
| r1m0 (qkv20) | tách đôi m0 (nạp q/k_rms DM sớm, p2 `_dm`) | static 37,783 = control; lab2 +81 (190) 3/6 | TRUNG TÍNH | – | m0 không phải nguyên nhân | qkv20_report |
| r3a2 / r2m0 (qkv20) | thân r3 + head a2 / sắp toàn bộ q epilogue | static 36,706 / 39,393 | ĐÓNG BẰNG PHÂN TÍCH | – | tính lane ~0; cửa sổ K quá tải theo thiết kế | qkv20_report |
| S2 head burst (R-A, BACKWARD_QKV) | gate mọi load nhỏ vào head, trọng số liền nhau | cần burst 0.3k/load | ĐÓNG BẰNG PHÂN TÍCH | – | fcf18 J4: load nhỏ trong burst tốn 0.82–0.91k (giả thuyết H-A sai) | fcf18_report |
| **luật tổng kết (qkv20)** | – | trên hàng đợi DMA nối tiếp, một load nhỏ tốn ~1.0–1.2k ở mọi vị trí (head, Q→K, K→V) | LUẬT | – | dời load nhỏ giữa các ranh giới không có lợi; chỉ lane idle thật mới bỏ được | qkv20_report |



#### 2.5 Nhóm E — Contraction, packet, tile trọng số, TU-clock

| tên/id | thay đổi (cơ chế) | kết quả | kết luận | được / mất | lý do, bài học | nguồn |
|---|---|---|---|---|---|---|
| Q10 QKV-008 (REV-Y) | contraction Q/K/V FP8×FP8 với x đã normalize tách hi/lo (k = 128), không lookup BF16 | 64,741 (+1,965); HW 158,665 (35549); sai số bit-identical | TRUNG TÍNH (revert) | −6 DMA bảng / −4 DMA hằng (−3,352), nhưng +6 pass Main, +26 PeCore; Main không phải nút thắt của QKV | chứng minh hi/lo chính xác trên device dù lo ≠ 0 | TSV:42 |
| Q23 QKV-QSPLIT | trọng số Q thành 2 tile 16 hàng (Main idle 14,688 ở head) | 59,672 | CÓ HẠI | DMA +1,388, nCore 34 → 46 | thêm 1 pass = thêm 1 DMA bảng (838) | TSV:64 |
| Q30e ứng viên A | dùng lại bảng decode | – | KHÔNG BIỂU DIỄN ĐƯỢC | – | `fetch_table_lookup` không nhận đối số bảng; bảng không nằm trong `input_tensors`; 3 bản ở 3 địa chỉ | qkv-next/LOOKUP_REUSE |
| Q30f ứng viên B = M4 (fuse contraction K+V) | một contraction cho cả K và V | dự +1,300 static (~+2,300 HW); đuôi 7,040 → ~12,300 | ĐÓNG BẰNG PHÂN TÍCH | – | phá cửa sổ 9,487 cycle nơi K post + Q RoPE được che khi trọng số V đang stream | OL§5.7.2 |
| Q32 QKV-E1 | packet fetch contraction 16 B → 960 B | 55,662 byte-identical | TRUNG TÍNH | – | collect engine chuẩn hoá packet về 1 flit; độ rộng fetch không có tác dụng với pass lookup (2 lần kiểm chứng) | TSV:87 |
| QA6 qkv8 E2 **t** (lab2 79483) | một lần nạp TRF cho cả 3 contraction (trọng số K/V dùng `unsafe reshape` sang trục của Q) | h4t 39,051; warm h4 77,821 → h4t 77,446; h4svt 77,027 | **CÓ LỢI** (−0.4k, vào v8) | output bit-identical | – | qkv8_notes |
| QA16 qkv9 L1/L2 A1 (79963, ~79990) | packet stream 64 B + `Gs` (hi/lo) làm time broadcast trong stream adapter → mọi contraction nhanh gấp đôi (mỗi flit trọng số chỉ fetch 1 lần) — cho cả 3 / theo từng contraction (a1, a1v, a1kv, a1qv) | 37,080 → 36,697; L1 base 74,345 so với a1 75,106 (+0.76k); L2 base 73,889, a1 75,762, a1v 74,123, a1kv 74,910, a1qv 73,530 | CÓ HẠI (bỏ riêng lẻ; về sau A1 cho Q,V nằm trong a1qvk3) | contraction HW Q 5,059 → 2,324, V 3,114 → 2,151 | list static đưa k_rms vào cửa sổ K→Q; q_scale/q_rms chạy chen giữa Q contraction dày đặc; A1 trên K luôn có hại (+1–1.9k) | qkv9_notes |
| C5 contraction V hi/lo trên các lane TRF (lg) | Lane = Gs làm hi + lo trong 1 bước MAC | Interleaved fold lỗi lowering ("reduce axis should not have padding"); Sequential compile được: contraction 765 → 746 nhưng nạp TRF theo layout lane tốn một pass Sub 1,223 cycle ở head: makespan 36,175 → 36,887 (+712) | ĐÓNG BẰNG PHÂN TÍCH (static) | – | bước MAC không phải nút thắt (fetch 64 B mới là); các hàm `*_lg` chết trong qkv_h4.rs:1237-1325 | IDEAS_COMPUTE.md C5 |
| qkvD2 t16 | nạp TRF với fetch 16 B | không compile (collect phải 32 B) | KHÔNG BIỂU DIỄN ĐƯỢC | – | – | qkvD2_notes |
| x15k2 stall PE-issue ở pass Q (critic) | sửa mode chậm của pass Q (4.9–5.4k so với 3.5–3.8k) | mode nhanh không dời điểm kết thúc trace (c1 nhanh 69,846 so với chậm 69,921); stall có lẽ phóng đại bởi trace | ĐÓNG BẰNG PHÂN TÍCH | – | không đáng đuổi theo | critic §3.2, §5 |
| q19ks (qkv19) | K contraction thành 2 pass 32 hàng trên 2 tile hàng của một K DM tensor (partial f32, bf16 ở gather, bit-exact) để V list trước k gather | static 36,942 (−681); lab3 102056 **+1,051 (91) 0/12**; trace 102062: nửa contraction 1.86k (bằng 74 % bản đầy đủ 2.51k), Kc_b bắt đầu 0.57k sau Kc_a | CÓ HẠI | – | mỗi contraction có ~1.2k chi phí cố định; khe LD program +0.57k | qkv19_report |
| TU-clock probe J1–J5 (99849, 99881, 99892, 99898, 99902) + lab QKV 99864/99873/99884/99885 | "slow contraction" có phụ thuộc Main idle trước đó không? | slow mode tái hiện ở trace (t_m40: C 4.39–4.42k, +1.74k, hold 2.69–2.73k); vi sai J4: **trace +2.58k, INFO −0.29k** | ĐÓNG BẰNG PHÂN TÍCH (artefact trace) | – | "cho Q ≥ 1k Main idle" không có giá trị ở INFO; mọi kết luận "cửa sổ V bị Main giới hạn" từ trace phần lớn là artefact | analysis/v15/TUCLOCK.md; memory tu-clock-slow-contraction-trace-artifact |
| "cho Q ≥ 1k Main idle" (ARCH_QKV_B §7, qkv16 tuc16.py) | gieo RoPE ramp ở chỗ khác để Q contraction có TU idle | trace: Q nhanh 3,539 khi TU idle ≥ 1.0–1.5k, chậm 5.0–5.3k khi < 0.3k; ở INFO lợi ~0 | ĐÓNG BẰNG PHÂN TÍCH | – | dựa trên artefact trace | qkv16; TUCLOCK |
| thời lượng contraction danh định (TUCLOCK) | – | V nhỏ (64×64) 1.21–1.30k; V/K (64×256) 2.50–2.66k; Q (128×256) 3.50–3.54k; một pass ngay trước C làm C nhanh hơn 0.12k; Sub chạy song song +0.4k (trace) | phát hiện | – | contraction HW ≈ 0.51k + 2.21 × static (26 span) | TUCLOCK; FREECHECKS §5 |

#### 2.6 Nhóm F — RoPE (bảng cos/sin, tính trên chip, ramp, zip)

| tên/id | thay đổi (cơ chế) | kết quả | kết luận | được / mất | lý do, bài học | nguồn |
|---|---|---|---|---|---|---|
| Q09 QKV-007 (REV-T) | cos/sin `dma_gather_scaled` thẳng vào layout 2 cluster | 63,199; 163,529 (35476) FAIL q,k 56 % | CÓ HẠI (sai) | – | gather không nhân bản sang cluster 1; phải đi qua `to_hbm` + `to_dm` | TSV:33 |
| Q20 QKV-020 | layout Q theo cặp kênh (slice giữ d..d+4 và d+128..d+132) để đôi RoPE nằm cục bộ trong slice | 63,122 (+346) | TRUNG TÍNH (khối xây dựng, chưa dùng) | – | util không đổi 0.82; gather DM→DM đặt tên cùng ký hiệu `Qs` sẽ tự bỏ hoán vị | TSV:60 |
| Q44 v9 | port cos/sin đóng gói của M3b (4 op / 2,780) vào rope.rs của V8 (6 op / 3,628) | 50,660 (−438), 29 → 27 op; 119,931 (41120), 122,485 (41121) | CÓ LỢI | – | – | best_sources_v9/README |
| Q46 **v11 QKV** (chọn làm main 09-21) | cos/sin tính trên chip với range reduction `fxp_to_fp(24)(turns)·(−2π·128) + angle`; 15 op DMA | 39,002 (−11.7k); 99,819 (71345), 89,141 (71414) | **CÓ LỢI** | – | bác bỏ "cos/sin trên device là bất khả" (L11): ramp dựng bằng cách nhân đôi; tính góc ngây thơ ra non-finite | v11_reference/RESULTS_v11.md |
| P2 dời RoPE ramp riêng (qkv_review) | ramp gieo từ memset ở head cho lịch K-trước | reviewer: gộp vào P1; riêng lẻ ~0 (x15r bị loại); qsim R +8.3k | ĐÓNG BẰNG PHÂN TÍCH | – | Q47 (giữ từ 09-22) đã đẩy các pass này KHỎI đầu list vì chúng gate issue trọng số | qkv_review §2 |
| bfnr / bfnr2 (qkv15) | fuse q (và k) normalize + RoPE với tile hệ số dựng trên Sub | lir ICE (pass Sub stream một DM tile kèm op vector) | KHÔNG BIỂU DIỄN ĐƯỢC | – | – | ARCH_QKV_B §6.2c |
| bfnr3 (qkv15) | fuse q normalize + RoPE (C, B' trên Sub, đủ độ dài) | static 38,505; lab5 96288 +1,415 (202) 0/10; trace: pass Sub thêm làm chậm Main (q gather switch 2.67k so với 1.69k) | CÓ HẠI | – | – | ARCH_QKV_B §6.2c |
| bfnr4 (qkv15) | fuse q + k | static 40,003 | ĐÓNG BẰNG PHÂN TÍCH | – | bfnr3 đã thua | ARCH_QKV_B §6.2c |
| **bzr** (zip RoPE, qkv15) | RoPE q và k thành pass zip 2 nhóm (begin_interleaved + vector_intra_slice_unzip + vector_fp_zip): mỗi tensor 2 pass Main thay cho 3 Main + 1 Sub; bit-identical trên HW | static 37,983; lab6 99708 −304 (194); lab7 99723 −99 (129); lab8 99742 −319 (141); lab9 99745 +51 (76); IV pooled sau 3 lab −220 ± 85 (z −2.6) | TRUNG TÍNH riêng lẻ (NO KEEP); **CÓ LỢI khi ghép vào g6z (V16)** | được 0.4–0.7k trên c1 bị Main giới hạn; ~0 trên c0 bị lane giới hạn (c0 quyết định metric) | chỉ cộng dồn được khi V land sớm (g6z) | ARCH_QKV_B §0, §6.2f; qkv15_progress |
| bzr lần compile đầu | ghép cặp fp_binary với một VRF mỗi nhóm | mir "this node needs more slots than a VE pass has" → tách 2 op cặp (nhóm 0 × cos; nhóm 1 × sin) | SỬA | – | pass VE có giới hạn slot | qkv15_progress 06:44 |
| bzrn | bzr bỏ pin K-norm-p2 zero_from_q_heads | chưa đo trong vòng này; về sau g6zn −896 / −485, trace +1,049 | TRUNG TÍNH (chưa xác nhận) | – | – | qkv15_progress 07:00; qkv17 |
| brmp (qkv15) | RoPE ramp gieo từ chunk mask (dời ~7k Main từ cửa sổ Q sang K) | static 38,636; lab6 +1,057 (282) | CÓ HẠI | – | k_scale dời vào head | ARCH_QKV_B §6.2c |
| brmpz | brmp + zip RoPE | lab6 +1,889 | CÓ HẠI | – | – | ARCH_QKV_B §6.2c |
| rip / ripz / ripqsw / ripqswz (qkv16) | RoPE ramp in-place (bớt 8 pass) qua alias đọc raw-ptr | COMPILE FAIL "mir: unsupported &raw const" | KHÔNG BIỂU DIỄN ĐƯỢC | – | borrowck cấm vừa đọc vừa ghi tile của cùng một tensor trong một pass | qkv16 |
| hz (qkv16) | RoPE row nửa độ rộng (stage 128 kênh, row C/S, pass pack × POS, một pass sin; −6 Main −2 Sub trong cửa sổ Q) + zip RoPE C/S (nửa thấp zip SubF, bit-identical) | static 37,810 (rope_off vào head); lab4 99885 +1,918 (218) 0/8 | CÓ HẠI | – | – | qkv16 |
| hzqsw / hzw / hzgw (qkv16) | hz + swap / kz lấy từ zero hàng 0 của trọng số K | static 38,010 / 38,233 / 39,305 (head K-trước) | ĐÓNG BẰNG PHÂN TÍCH | – | – | qkv16 |
| hzg (qkv16) | hz với POS gate theo k đã gather | static 37,646 (K dispatch sau mean-square của chuỗi x: ước +2.3k); lab1 99864 +848 (192) 1/8; trace 99873 −234; lab3 99884 +1,000 (105) 0/6 | CÓ HẠI | – | head bị lật (+) lớn hơn lợi trong cửa sổ Q; ở INFO Q vốn đã nhanh | qkv16 |
| hs / hsn (qkv16) | row đủ độ rộng [C, +S], một pass sin, góc gộp vào pass pack | static 37,390 / 37,466; lab1 99864 hs +851 (136) 0/8 | CÓ HẠI | – | cùng kiểu lật head | qkv16 |
| bzr đứng riêng (qkv16) | – | trace 99873 −86 (177); lab4 99885 INFO +2 (180), c1 −169 | TRUNG TÍNH (lab null thứ 5) | – | sau đó ghép vào g6z | qkv16 |
| BÀI HỌC độ mong manh (qkv16) | – | x15k2 giữ K trước chỉ vì chuỗi cửa sổ Q của nó (static) găng trên Main (Qc static 25,645 > Q land 24,476, dư 1.17k); cắt > ~1.2k static là beam search thôi quan tâm và dời k_scale (+ rope_off không gate) lên head | LUẬT | – | – | qkv16 |

#### 2.7 Nhóm G — Chuẩn hoá per-head, channel scale, đuôi V

| tên/id | thay đổi (cơ chế) | kết quả | kết luận | được / mất | lý do, bài học | nguồn |
|---|---|---|---|---|---|---|
| Q30a ứng viên F | gộp các setup Sub của q/k rms-weight | ~0 static (−1 Sub), dự ~124,800 (1.4 %) | ĐÓNG BẰNG PHÂN TÍCH (hạng C, chưa build) | – | dưới CV 2.1 % | qkv-next/CANDIDATES_V2 |
| Q30b ứng viên G | gộp V channel-scale vào pass mean-square | −285, dự ~125,960 (0.45 %) | → E6 | – | – | same |
| Q30c ứng viên D | normalize V trên 4 slice thay vì 1 | −373, dự ~125,790 (0.6 %) | ĐÓNG BẰNG PHÂN TÍCH (hạng C) | – | – | same |
| Q30d ứng viên C | tách trọng số V thành 2 tile | −494, dự ~126,010 (0.4 %) | → E7 (CF) | – | – | same |
| Q37 QKV-E6 | gộp per-channel scale của V vào các pass mean-square + normalize | −285 static; ở 41107 v max abs d 0.03125 → 0.01562 | CÓ LỢI (port, hiện thực trong r21) | bỏ một lần làm tròn bf16 (chính xác hơn) | trọng số V land cuối, pass lộ hoàn toàn; sau inter-slice reduce chỉ có `vector_clip` (không có phép nhân) nên không fuse được vào contraction | TSV:92, qkv-tt/RESULT |
| Q40 QKV-E7 | trọng số V thành tile 12/4 để che contraction tile 1 | CF (3 cách viết) | KHÔNG BIỂU DIỄN ĐƯỢC | – | Transpose: `in_rows` ≤ 4 mỗi packet, không phân rã được trục bị giới hạn (`Ps % 16 = 12`); chia 8/8 chỉ −494 static nhưng +1 DMA +1 Sub | TSV:95 |
| Q45 gộp V-scale trên V8 | port Q37 (−345 static ≈ −785 HW) | chưa đo | (hiện thực trong r21) | – | đuôi V của V8: trọng số V xong 44,126 → contraction 2,663 → gather 1,212 → scale 345 → ms 345 → rms 282 → Sub 267 → normalize 345 → scatter 913 → 600 | OL§7.3 |
| Q61 r21 (76998) | r18 + channel scale gộp vào head RMSNorm (−2 pass) + sin từ half-tile (2 pass thay 3) | work 83.2/83.5k; total 92,963 | CÓ LỢI (bị r21n thay) | – | hiện thực Q37 | qkv-r12/RESULT |
| Q62 **r21n** (76998, 77036) | r21 không trì hoãn normalize của K sau gather Q | work 81.1k (−6.6 % so với v11) / 82.2k; total 93,707 (tệ nhất trong 3) / 86,286 (tốt nhất trong 3); harness chấm 86,402 (77046); q mean abs d 0.001030 (v11 0.000996) | **CÓ LỢI** (main 09-23) | – | chọn theo work trước sync; lần chấm chính thức 89,258 / 94,416 / 78,736 / 89,296 | experiments/best_sources_r21n/ |
| QA3 ao-qkv sv (E3) | sqrt → VRF từ Main trong norm q, k, v | 40,498; +1.9k | CÓ HẠI | – | trong cửa sổ Q→V, Sub bận (x_vrf, sin VRF); pass Main→VRF phải chờ Sub (StoVrf 1.2–1.75k) | ao_qkv_notes |
| QA4 ao-qkv svv (E4+E5) | sqrt → VRF chỉ trong chuỗi V (đuôi, Sub rảnh) | 40,444; −0.3k (n = 4), +0.6k (n = 8) | TRUNG TÍNH (bỏ) | – | đuôi −0.4..−0.8k nhưng cả list static đổi; mức nhiễu | ao_qkv_notes |
| QA6 qkv8 E2 **sv** (lab2 79483) | pass sqrt của V ghi rms VRF từ Main (−1 pass ở đuôi) | h4sv 38,581; warm h4 77,821 → h4sv 77,273 (−0.55k) | **CÓ LỢI** (v8) | output bit-identical | ở đuôi thì Sub rảnh | qkv8_notes |
| H2 V-tail hai mode → E302 steering v_scale (V10) | đưa v_scale của c1 land ≤ ~1.1k sau khi V contraction bắt đầu (bộ dự báo: nhanh 6/6 nếu ≤ 1.21k; chậm 9/10 ở 1.22–1.25k; mode chậm tốn ~0.66k) | chỉ thiết kế: v_scale phải đi trước trọng số V (0.9–1.5k lane) hoặc wait của nó đi trước push; operand swap +0.15k | ĐÓNG BẰNG PHÂN TÍCH | – | không biến thể nào qua được gate lợi ròng 0.4k | V10-E302.md |
| H4 V thành 2 tile 2 hàng (probe `vs`) | đơn vị ghi commit_trim 8 B, commit_view vào 2 nửa | static 37,418 (+921); hwsim 70,220 so với 69,789 (+0.43k) | ĐÓNG BẰNG PHÂN TÍCH | – | lane DMA static là nối tiếp; tile b list sau contraction của tile a + 13 pass chuỗi Q/K | CRITICAL_PATH_QKV.md s4c |
| H8 fuse ms + sqrt | sqrt trong pass mean-square | typestate: sau vector_intra_slice_reduce chỉ còn fp_div / clip | KHÔNG BIỂU DIỄN ĐƯỢC | – | = A20 | CRITICAL_PATH_QKV.md s4c |
| C3 + C1q (pha 3a) | GP: V gather 4 kênh mỗi packet 8 B qua ring; SQF: gộp sqrt vào V normalize (Mode021 FmaF (&ms_vrf, 0.0) + Sqrt trên FpFpu + `fp_div_with_mode(Mode10, Stash)`) | static 36,497 → 36,175 (−322); emulator V-tail lab 0/2,048 khác biệt | → E311/E312 | – | – | IDEAS_COMPUTE.md C1/C3 |
| E311 SQF (arm B) | như C1q | lab 85174 (8 block × 3, 16 warm): +61 (SE 258, 2/8 < 0), E[max2] +154; c1: mất sqrt, normalize cố định 755, nhưng ms→VRF chậm 13/16 (1.40–1.95k) | CÓ HẠI (ở vòng V10) | – | mode chậm dời vào pass đứng đầu block hàng đợi v-scatter động (về sau, trên thân qf7, chính dạng này thành q19n2 KEEP) | V10-E311.md |
| E312 SQF + GP (arm C) | + gather V packet 8 B | 85174: +68 (SE 237, 4/8), E[max2] +43; pass GP 0.83–0.87k (so với 1.18–1.30k) nhưng gather kết thúc = Sub StoVrf của v_scale kết thúc + 3 ở 8/16; ms chậm 16/16 | CÓ HẠI | – | đuôi bị giới hạn bởi block CPU, không phải TU | V10-E312.md |
| C4 đuôi V thành 2 pass gather | gather + ms → VRF; gather + normalize → DM | compile FAIL "Fetch: the packet's innermost axis must be contiguous in DM, but Ps / 8 % 4 has memory stride 8" | KHÔNG BIỂU DIỄN ĐƯỢC | – | to_vrf chỉ sau Collect / VectorFinal | IDEAS_COMPUTE.md C4 |
| cơ chế tailblk qkv11 | so sánh block hàng đợi PE (`qkv11/tailblk.py`) | FACT, cả 96 cluster-run warm: mode chậm chỉ ở pass đi vào block chứa v scatter động | phát hiện | – | tương tự guard MOV-6 bên FFN | V10-E311.md |
| E351 đòn bẩy block v-scatter + lab trace/info | dịch ngược tuc.o của PE runtime; tìm handle hợp lệ để publish block v-scatter sớm hơn | lab 85630: B_i − A_i −111 (242), E[max2] −233 (287); A_i − A_t −713 (184) 7/8; B_i − B_t −601 (151) 8/8 | CÓ HẠI / ĐÓNG | – | không có handle trong source: thứ tự store bị bỏ qua (biến thể d cho PE byte-identical), lowering `dma_scatter` cố định (index-fetch DMA + scatter gián tiếp), `dma_scatter_unscaled` là todo!() | V10-E351.md, qkv12_notes.md |
| rút ngắn Q-post (kiểm lại trên x15k2) | – | V land → kết thúc 7.34k ở cả hai arm | ĐÓNG BẰNG PHÂN TÍCH | – | không găng | critic §5 |
| qe (qkv15) | VRF q-scale bằng pass Main (`normalize_query_scaled_e`): k_rms / q_scale / q_rms liền nhau sau EXEC(Qc) | static 38,391, PE gate PASS; lab3 95861 +1,184 (261) 0/8, E[max2] +1,409; dự −0.9..−2.0k | CÓ HẠI | – | chuỗi post-Q bị Main giới hạn; thêm một pass Main (1.2–1.9k) = +1.2k | ARCH_QKV_B §6.2 |
| bvs2 (qkv15) | V thành 2 nửa 32 hàng (partial f32) | lab3 +1,924 (178) 0/8 | CÓ HẠI | – | 2 contraction 32 hàng + gather f32 = thêm việc Main | ARCH_QKV_B §6.2 |
| bvs2 lần compile đầu | Transpose f32 | visa "f32 Transpose merges at most 2 rows" → sửa gather thành packet f32 2 hàng | SỬA | – | – | qkv15_progress 21:02 |
| bfe2 (qkv15) | bqk2 + bvs2 | lab3 +1,908 (140) 0/8 | CÓ HẠI | – | – | ARCH_QKV_B §6.2 |
| bvs (nửa bf16) | nửa V dạng bf16 | COMPILE FAIL "commit_trim output packet must be one of [8, 16, 24, 32] bytes, got 4" | KHÔNG BIỂU DIỄN ĐƯỢC | – | nửa 32 hàng = 2 bf16/slice < 8 B tối thiểu | ARCH_QKV_B §6.1 |
| bqvr / bqvs | qe + q gather commit vào buffer DMA q_rms / q_scale sau khi nạp VRF của nó (WAR pin) | COMPILE FAIL "owner mismatch" | KHÔNG BIỂU DIỄN ĐƯỢC | – | switch trên Main không commit vào buffer DMA-written trên slice head có pad | ARCH_QKV_B §6.1 |
| QKV-B bhv / bhv2 / bhv2z (từ review bên ngoài) | head V {0,1}/{2,3} thành 2 tile HBM vào tensor riêng, contraction f32 riêng, gather từng nửa Broadcast1{128,1}, normalize từng nửa vào 1 tensor V gộp, 1 scatter | static bhv 39,318; lab8 99742: bhv2 +3,952 (163), bhv2z +4,045 (246) | CÓ HẠI (QKV-B đóng với 0.8.1) | – | trọng số kết thúc đúng lúc contraction của nó bắt đầu; pass cửa sổ V nằm trong cửa sổ static của V_a; tách head ở offset slice không biểu diễn được | ARCH_QKV_B §6.2d |
| QKV-A một pass SS→rsqrt | gộp sqrt vào normalize | thứ tự stage VE: Tag → Logic → Fxp → FxpToFp → Narrow → Fp (unary gồm Sqrt) → IntraSliceReduce → FpDiv → Widen → FpToFxp → Clip; không có unary nào sau reduce | KHÔNG BIỂU DIỄN ĐƯỢC | – | cùng lắm chỉ gộp được pass sqrt→VRF riêng (công thức N4-i) | ARCH_QKV_B §6.2e |
| qf8 / vr19 / vr20 / vr21 / qv1 / vr18 (qkv18 C3 tách V) | toàn bộ q epilogue trước Kc; V theo nửa head hoặc nửa hàng | static fail (K trễ 2.1k; V_b nằm sau chuỗi cửa sổ V_a; cửa sổ K 87 %); vr20/21 tail 3.69k nhưng rope rows rời cửa sổ Q | ĐÓNG BẰNG PHÂN TÍCH | – | luật tie + TU HW ×2.2 so với DMA ×1.85 → V_b issue muộn | qkv18_report |
| **q19n2 (→ V17.2)** | qf7 + V normalize per-head 2 pass "SQF" (ms+EPS → VRF; sqrt trong stream qua FmaF Mode021; FpDiv Mode10 trên stash; bỏ pass sqrt riêng sau Vc) — = T2-1 của cloud = V10-E311 | static 37,866 (+243; đuôi sau V 3,751 so với 4,209); lab1 101974 −16 (107), c1 −839; lab5 102090 −336 (107); lab6 102103 −392 (107); lab7 102114 −185 (78); lab8 102121 bật screen −414 (94) (tắt +409 ± 827, có outlier host-sync 89,443); **pool −264 (SE 43)**; INFO-only −250 (47); NH 102138 68,918 | **CÓ LỢI** (V17.2) | được −0.26k; phần lợi đuôi của c1 tới được c0 khoảng ~40 % | cùng một ý (E311) mà ở V10 thua, sang V17 lại thắng: kết quả phụ thuộc thân kernel; lab 5–8 cho thấy lợi đuôi c1 tới c0 ở 30–80 % | qkv19_report; V172_RESULT |
| q19fq / q19fk / q19fqk (qkv19) | gộp sqrt kiểu T2-1 vào norm q / k / cả hai | static 37,656 / 38,259 / 36,401 (−1,465, lợi chỉ ở static); fqk lab9 102267 +155 (107) | TRUNG TÍNH (KILLED) | – | giảm tải Main trước V không có giá trị cho metric | qkv19_report |
| T2-2 V norm không gather (qkv19 / qkv20 t22p) | ms trên layout contraction, scatter thẳng từ layout đó | t22p: visa "strides([2, 134217728, …]) is not aligned by 8" | KHÔNG BIỂU DIỄN ĐƯỢC | – | piece 2 byte bị từ chối; thiết kế bỏ sót v_scale; không thứ tự reduce nào trên 64 slice × 4 phần tử tái tạo được tổng f32 của layout head → không bao giờ bit-identical | qkv19/qkv20_report |

#### 2.8 Nhóm H — Chuyển việc giữa Main và Sub

| tên/id | thay đổi | kết quả | kết luận | được / mất | lý do, bài học | nguồn |
|---|---|---|---|---|---|---|
| Q48 x-broadcast trên Sub | dời pass broadcast sang Sub | 39,995; −1.5k so với −2.7k khi giữ | CÓ HẠI | – | kernel bị TU giới hạn; Sub không phải năng lực rảnh | dma-docs/FINDINGS §4 |
| Q57 r17c (76912) | operand swap + reduce/sqrt của input RMSNorm trên Sub | work 82.8k (−4 %); total 92,250 so với 92,711 | CÓ LỢI (lab) | – | inter-slice reduce trên Sub compile và chạy được | qkv-r12 |
| Q59 r19 (76955) | ms/sqrt (q, k, v) + q_sin/k_sin dời sang Sub | work 87.9/88.5k (+2 %); total 100,222 so với 95,900 | CÓ HẠI | – | Sub cũng in-order; bóng bàn Main↔Sub quanh to_vrf; V issue muộn 5.5k; `zero_q` trên Sub ICE; `vector_clip` trên Sub ICE → cộng EPS từng phần tử trước reduce | same |
| Q60 r20s (76963) | r19 + 3 pass gather (switch) trên Sub | work 87.9k; total 92,856 so với 88,548 | CÓ HẠI | – | như r19; pass switch compile được trên Sub | same |
| Q64 r23 (77036) | r21n + K gather trên Sub | work 82.3k; total 87,604 | CÓ HẠI | – | lại không lợi khi đẩy việc sang Sub | qkv-r12 |

#### 2.9 Nhóm I — Số học (độ chính xác của activation)

| tên/id | thay đổi | kết quả | kết luận | được / mất | lý do, bài học | nguồn |
|---|---|---|---|---|---|---|
| Q31 mô phỏng FP8 một thành phần | mô phỏng trên host, activation FP8 một thành phần | pre-scale toàn cục 1..256: max 0.14062, 92.87 % FAIL; amax theo nhóm: 0.03125, 100 % "pass" | ĐÓNG BẰNG PHÂN TÍCH | – | "pass" là artefact của fixture: mọi phần tử dồn về code e4m3 448.0 (abs h trong [0.9648, 0.9727], trải 0.81 %); trải ±2 % → 95.07 %, ±5 % 95.51 %, ±10 % 93.46 % | qkv-next/fp8_single_sim.md, tools/qkv_fp8_single_sim.py |
| V12-E701 Q-6 lane f8 đơn (s1, s1k) (qkvA13) | lane f8 đơn cho đường Q; rút chuỗi x 4.4k | 91595 info: s1 +718, s1k +1,282 | CÓ HẠI | – | chuỗi đầu vốn đã được che dưới stream K (K land mới là cái gate); dựa vào fixture là đặc cách fixture | ledger V13 |
| S12 bỏ scalar input RMSNorm (BACKWARD_QKV) | scalar triệt tiêu qua per-head norm (sai số ~5e-7 tương đối) | ~0 cho metric | ĐÓNG BẰNG PHÂN TÍCH (không làm) | – | vi phạm quy tắc team, không lợi | BACKWARD_QKV |

#### 2.10 Nhóm J — Đo đạc, sự cố, phân tích (không phải thay đổi kernel)

| tên/id | nội dung | kết quả | kết luận | nguồn |
|---|---|---|---|---|
| Q16 REV-AJ/AK/AL | xác nhận revert; cây an toàn để nộp | 62,776; 153,537 (35710), 147,597 (37513), 149,385 (37852) | tham chiếu: dải của 003c 146–158k | TSV:55,56 |
| Q30 audit qkv-next | kiểm hằng số, dùng lại bảng, chờ Sub, ứng viên | 11 hằng = 8 tham số HBM riêng (5,259) + 3 bảng decode (2,514); số DMA bảng mỗi pass lookup Main = 1.00 trong 77/77 lịch; 1,441 "chờ Sub" = 280 tile-view + 1,161 chuỗi V post; đuôi V 48,622 → 55,662 = 7,040 (12.6 %) | phân tích | qkv-next/*.md |
| Q41 sự cố job 41088 | định chạy E8 nhưng build ra M3b (agent đồng thời revert 5 file QKV lúc 01:26) | 126,589 (41088) | SỰ CỐ → quy tắc: build từ bản sao cô lập + in md5; chữ ký sai số nhận diện được build | qkv-tt/RESULT |
| Q42 lặp lại M3b/v6 | cùng source | 131,633 (39569), 137,593 (39600), 125,007 (39615), 123,885 (39644), 132,591 (39654), 126,597 (39661), 137,529 (39667) | dải QKV 11 %, CV 4.5 % | LED |
| Q66 kiểm main r21n (77120) | không thay đổi | 105,776 / 54,344 / 254,775 (6.41) | một lần chạy cold có thể tốn ~+20k cho QKV (sync cuối, thời 0.6) | LED |
| QA9 qkv8 ver1 (79589) | h4svtg trong harness thường | 38,273; QKV 77,259 [83,021 / 77,259 / 76,520] | cửa sổ c0 gồm 1.9–2.0k chờ barrier (launch skew) | qkv8_notes |
| QA15 qkv8 kiểm cuối | x8g trong harness thường | 37,080; 79731: 74,637; 79743: 75,107 | QKV −5.1 % so với trung bình v7; c0 chờ 3.4–4.1k ở barrier cuối | qkv8_notes |
| x15k2 cold run 0 | phạt đổi chương trình | cold run cao hơn V13 1.2–5.8k; trong lab xen kẽ arm, delta cold paired chỉ +317 | vô hại (median-of-3 bỏ qua run 0) | V131_RESULT |
| v131V lăng kính THỐNG KÊ | parser riêng vstats.py; metric = `Collector::window_cycles` của grader (khớp LABCYC, 0 sai lệch) | 91814 Welch −362 (234), paired −362 (285); 91889 −429 (174)/(121); pooled IV −419 (111), z −3.77; bootstrap E[max2] −410 (158) CI [−704, −89]; sign test 15/18 p = 0.0038 | XÁC NHẬN (lưu ý winner's curse ở 91814; 91889 là bản lặp đăng ký trước) | experiments/v131V_stats.md |
| v131V lăng kính HỢP LỆ + SỐ HỌC | sigcheck 10 chữ ký; file device mới được chấp nhận; LF-only; chunk_mask = bits(f32(x)) & 0x7fff0000, …; partial ở slot pad ép +0 bitwise trước mọi op FP (NaN không lọt); pin exact-zero là bitwise; `one = sum(0*k + 1/256) = 1.0` | sai số QKV bit-identical với V12 | XÁC NHẬN | experiments/v131V_legal.md |
| v131V lăng kính CƠ CHẾ (+ lab trace riêng 92075) | phân rã loại bỏ skew: c1 luôn găng | dMAX −399 (125), dMIN = Task_c1 −562 (173); theo tercile skew −553 / −400 / −406; trace c1: DMA K/Q/V −643 / −528 / −995; Q land −1,281; Main +3.2k trong cửa sổ Q-DMA; pass Q hai mode 3.5–3.8k / 4.9–5.4k | XÁC NHẬN (một phần: V issue ở lúc Q-contraction kết thúc — sửa lại câu chuyện ban đầu) | experiments/v131V_mech.md |
| phạt cold theo arm | cold − warm | a 3.4k / 3.6k; x15k2 5.6k / 6.2k; k2 +1.5k; x15v +2.8k | chi phí cold đi theo thứ tự ksq2 / việc Main trước Q, không theo layout K15 | v131V_mech §3 |
| chip view x15k2 (trace 94261) | xô stream giữa 2 cluster | cả hai stream 48.0k / một 6.6k / không ai 18.2k = head 3.5 + K→Q 1.3 + Q→V 5.0 + tail 8.3 | phát hiện: block load nhỏ dài hơn skew (~0.9k) là thời gian "none" của chip; dời lên head cộng none 1:1 | qkv15_progress 17:37 |
| list DMA PE của x15k2 (ARCH_QKV_B) | 16 lệnh DMA của PE (trace c0 94261) | rms_w 1.65k WAIT / x 1.70k / K 12.63k / rope_off 0.90k WAIT / k_scale 1.17k WAIT / Q 25.74k / k_rms 2.40k gap + 1.31k / q_scale 0.77k WAIT / q_rms 1.26k / V 12.92k / q store 1.00k / v_scale 1.05k WAIT / k scatter 1.66k / v scatter 1.66k | phát hiện: mỗi WAIT là do consumer hoà với begin static của DMA kế và TU thắng | ARCH_QKV_B §1-2 |
| K6 lane V12 (qkv_gap) | median trace c1, n = 46 | head 3,582 / K 13,065 (301 B/t) / K→Q 2,635 / Q 25,973 (303) / Q→V 3,207 / V 13,484 (292) / tail 7,249 / skew 2,159 → T 71,561 trace; info 71,132; chip: cả hai 46,895, một 12,155, không ai 12,420 | phân tích | qkv_gap §2.1 |
| GAP_QKV (V16, trace 100323) | ngân sách từng lệnh | H1 rms_w 1.37; H2 x 2.32 (1.64 khi không có switch bên cạnh); H3 k_scale 1.10; K 12.55 (sàn 12.08); K→Q 1.65; Q 25.13 (sàn 24.16); post-Q 3.26 + 0.18; V 13.09; tail 7.33 (V ctr 2.51, gather 1.17, norm 1.98, scatter 1.66); c1 riêng 68.34; c0 chờ cuối 2.21 INFO; chip: cả hai 49.0k, một 4.3k, không ai 17.7k | phân tích | analysis/v17/GAP_QKV.md |

#### 2.11 Nhóm K — Skew, bất đối xứng cluster, engine rời nhau

| tên/id | nội dung | kết quả | kết luận | nguồn |
|---|---|---|---|---|
| H-SKEW / cân bằng cluster (qkv9, V10) | cân bằng lại công việc giữa 2 cluster | độ dốc T theo skew −0.21 (ngân sách chung) | ĐÓNG BẰNG PHÂN TÍCH: hai cluster chia chung băng thông stream (~545–570 B/c khi cả hai cùng stream; một cluster stream một mình nhanh ~1.8×) | qkv9_notes; CRITICAL_PATH_QKV |
| K4 giá của skew (qkv_gap §6) | skew QKV (c0 chờ ở barrier cuối) | +0.72 tick/tick ở info (n = 28, corr 0.54), +0.42 ở trace; skew là nhiễu mức job (trung bình a_i cùng code 2.2k / 3.1k / 2.4k); không biến thể head nào đổi được nó | phát hiện | qkv_gap §6 |
| P5 probe DMA trên engine rời nhau (U2) | thử lệnh DMA trên các engine PE khác nhau có overlap không | reviewer KILL như một đòn bẩy V14 (mọi load nhỏ QKV nhắm slice head 0/64/128/192 = cả 4 PE) | ĐÓNG BẰNG PHÂN TÍCH (sau đó probe J1 94294 cho thấy per-engine queue CÓ trên HW nhưng không biểu diễn được, §5) | qkv_gap P5; PROBES |
| c1-light asymmetry L1 (fcf18) | dời một head từ c1 sang c0 | E = 1.97 MB rơi vào trường hợp +0.58k; nửa head làm hỏng per-head norm | ĐÓNG BẰNG PHÂN TÍCH (độ mịn quá thô) | fcf18_report |
| QKV ở INFO | cửa sổ c0 so với cửa sổ c1 | cửa sổ c0 = cửa sổ riêng c1 + 2.0–2.3k (sync cuối); dT NH c0 − c1 median 2,238 (0/50 lần c1 > c0) | phát hiện: lợi TU sau V trên c1 chỉ tới metric ~40 %; thay đổi lane làm V land sớm hơn thì tới c0 đầy đủ | qkv16; qkv19; fcf18 STEP 0 |

---


## 3. Kernel AO (`sliding_attention_output`)

### 3.0 Kiến trúc hiện tại (aoYG trên nền V18 K16; static 22,869; NH ~35.4–35.6k; không đổi từ V12)

**Dataflow.** Tính `y = O·x` với trọng số O dạng FP8 (3,840 hàng × 4,096), sau đó nhân channel scale, rồi RMSNorm với norm_weight, rồi cộng residual. Kết quả ghi đè lên residual (ghi in-place vào tensor HBM residual). Việc ghi in-place này khiến compiler chèn sync DramReuse.

**Contraction và trọng số.**
- K-split 16 của teammate (V18): mỗi slice giữ 120 hàng, mỗi hàng chia thành các piece 256 B. Contraction 64 phần tử, mỗi packet trọng số chỉ đọc một lần. Kết quả partial được cộng dồn bằng inter-slice reducer ngay trong contraction b.
- x được tách thành FP8 hi/lo.
- Seed x có kích thước 32 × 512 B, phát tán bằng `Broadcast1{16,16}`.
- Trọng số chia thành 2 tile, 96 + 24 hàng/nhóm. Sau contraction a có một chuỗi 16 identity-copy (steering chain). Chuỗi này giữ tile b được list sau contraction a và đứng liền sau tile a trong FIFO.

**Memset pin ở đầu kernel (aoYG, V12).**
- (a) Output của switch (`gathered`, 8 KB/slice, có pad `1 # 16`) được tạo bằng DmTensor::new, rồi Sub memset và commit_view. Đây là WAW pin q_wfirst do teammate đề xuất, làm trọng số được list trước switch.
- (b) Một Sub memset lên y. Compiler đặt POST readiness của sync cross-cluster ngay sau lần ghi cục bộ cuối vào y, nên memset này dời POST ra sau (static 1164 → 2619).
- Trên HW, trọng số được issue sớm ~0.9k thay vì ~1.88k.

**Đuôi kernel.** List PE sau tile b của c0:
`[contr b] DMA:ws W(ws) ¦ WAIT1 ¦ DMA:gather S(ws_vrf) M DMA:nw ¦ W(gather) ¦ POST2 ¦ W(nw) DMA:res M(inv_sw) ¦ WAIT2 ¦ StoVrf(y) M(ms) [Sub config res_vrf] ¦ W(res) ¦ POST3 ¦ final ¦ WAIT3 ¦ store ¦ POST4 ¦ WAIT4`
- Epilogue chạy trên 32 slice × 120 kênh của c0 (e32).
- Gather y từ c1 sang c0 là một hop DM→DM cross-cluster (~1.25k).
- Có 4 sync. Một trong số đó là DramReuse, do residual được ghi in-place.

**Ngân sách c0 ở INFO (ao20).** T0 + lane trước WAIT2 (ws 1.56 → gather 1.17 → nw 1.15 → inv_sw 0.5 = T0 + 4.35) + phần sau WAIT2 2.86 (ms 1.04, Sub stretch 0.10, final 0.73, DR 0.11, store 0.87). Lãng phí chỉ ~0.2k.
- Ở INFO, AO bị c0 giới hạn khi s < ~2.3k (20/24 launch). Cắt được bao nhiêu trên đường của c0 thì metric giảm bấy nhiêu (khoảng 1:1).
- Sàn cấu trúc (BACKWARD_AO): sau byte cuối của cluster về muộn cần 6.4k (C_last 1.47 + hop 1.25 + sync ~1.05 + reduce 1.04 + normalise 0.73 + store 0.87 + end ≥ 0.05). Vậy aoYG không lãng phí gì vẫn chỉ đạt ~34.2–34.4k chính thức.
- Leader đạt 31,966, thấp hơn mức đó 2.3–2.5k. Không cấu trúc nào biểu diễn được trong 0.8.1 đạt tới con số này.

### Bảng thí nghiệm AO
- Mã A01–A28 là thời 0.6 (LED §4).
- Mã AA1–AA22 là 0.8.1 v7–v9 (`experiments/p081/ao_qkv_notes.md`, `ao8_notes.md`, `ao9_notes.md`).
- Từ V10 trở đi, mã lấy theo tên arm trong `V10-E*.md`, `experiments/ao1x_progress.md` và `analysis/*`.

#### 3.1 Nhóm A — Kiến trúc tổng thể và cách chia tile trọng số O

| tên/id | thay đổi (cơ chế) | kết quả | kết luận | được / mất | lý do, bài học | nguồn |
|---|---|---|---|---|---|---|
| A01 CDX-000 | 2 cluster, K-split + inter-slice reduce, fuse lookup vào contraction | 35,876; 78,499 (33638) | CÓ LỢI | 409k → 78k | HW/static 2.19 | TSV:4 |
| A03 AO-003 (REV-B) | trọng số O chia 3 tile | 35,450; 77,357 (34181), pooled 79,063 (n = 14) | CÓ LỢI | overlap load với contraction | – | TSV:36 |
| A05 AO-005 (REV-S) | tile 16 + 14 hàng (contraction 1 overlap với load 2) | 28,735; 59,386 (35471), pooled 64,380 (n = 5) | CÓ LỢI | – | – | TSV:32 |
| A07 AO-8+22 | tile theo version 7 | 26,339 (+512) | CÓ HẠI | – | bản thân v7 chỉ 55,532, so với 54,102 của ta | TSV:68 |
| A08 O weight trên 1 cluster | toàn bộ trọng số O trên một cluster để bỏ hop HBM của y | 36,562 (+11,660), trọng số 14,061 → 27,281 | ĐÓNG BẰNG PHÂN TÍCH | – | rate tỷ lệ với số slice sống (~2.40 B/cycle/slice); tác giả v11 đo cùng ý tưởng +23.6k HW | DMA_COST_LAW, DIRECTION H1 |
| A09 **AO-007** | trọng số O 16+14 → 26+4 hàng | 25,184 (−643); probe 24+6 25,315, 28+2 25,927; HW 63,233 (39600, máy chậm 8.7 %) | CÓ LỢI | DMA byte-identical (chia 30 hàng kiểu nào cũng ra cùng 2 op + byte) | dời 22 hàng contraction vào dưới load của tile 2 | TSV:81 |
| A12 K_CHUNK 2048 | chunk K rộng hơn | không map được | KHÔNG BIỂU DIỄN ĐƯỢC | – | 15 f32 = 60 B, không phải bội của 8 B | guide-fork |
| A18 tile trên v11 | 28+2 / 24+6 / 22+8 | +740 / +125 / +256 | CÓ HẠI | – | 26+4 lại là tối ưu | ao-e1 §3 |
| P1/P5 port (0.8.1 v1/v4a) | v1: 10+10+10 (0.8.1 từ chối số hàng tile không chia hết 30); v4a: 26+4 qua `unsafe reshape` sang trục riêng | v1 AO 26,135 static; v4a AO 23,334, HW 45,373 (77915) | v4a CÓ LỢI | −0.9k so với v3e | descriptor byte-identical với tile của 0.6 | analysis-work/toolchain081/ffn24 |
| AA2 3 tile ưu tiên trọng số (8+18+4, 4+22+4); đổi thứ tự source | – | 22,907 / 22,855; giống hệt | CÓ HẠI | – | scheduler vẫn đặt seed DMA lên đầu; index lệnh do scheduler quyết định, không theo source | ao_qkv_notes |
| AA7 ao8 t24 (lab1 79473) | tile 24+6 | 22,398; warm 41,180 so với v7 41,081 | TRUNG TÍNH đứng riêng | ws land trong cửa sổ c6, đuôi −0.9k, c6 kết thúc trễ 0.4k | (có lợi khi đi kèm chain, AA11) | ao8_notes |
| AA10 ao8 t22 | tile 22+8 | 22,526; 41,827 (+0.75k) | CÓ HẠI | – | w8 chậm (8.6–9k) | ao8_notes |
| AA21 ao9 26+4 cùng TB+INTER+T1 (lab7 80462) | ti_t1_t26_c26 / i_t1_t26_c26 | 22,091 / 22,477; 38,812 / 39,651 so với inter 38,989 | TRUNG TÍNH (không giữ) | – | luật: một DMA list sau pass Sub VRF chỉ issue khi pass đó HOÀN TẤT (StoTrf thì khác: issue lúc dispatch); cách sửa là đổi thứ tự, không phải đổi page | ao9_notes |
| **V18 AO (teammate), chọn ở V11** | K-split 16 (120 hàng × piece 256 B/slice, probe 618 so với 581 B/c), contraction 64 phần tử đọc mỗi packet trọng số một lần, x 32 × 512 B + Broadcast1{16,16}, sqrt gộp vào pass final, tile 96+24, chain 16 copy | nhanh hơn AO V10 (INTER+P2, ~38.2k lab) khoảng ~2.5k; chính thức 35,832 | **CÓ LỢI** | – | – | V11_RESULT; memory team-v18-and-v11 |
| một lệnh trọng số (V10) | 24+6 → 1 lệnh | DMA −1.37k nhưng +3.6k (c30 sau T0) | ĐÓNG BẰNG PHÂN TÍCH | – | contraction nằm sau T0 | CRITICAL_PATH_AO.md s0 |
| một lệnh trọng số (V14) | 1 lệnh thay 2 | +2.1k contraction sau T0 | ĐÓNG BẰNG PHÂN TÍCH | – | 120 hàng contraction sau byte cuối | ao_gap §2; V14_PLAN §4 |
| t100 (ao13, lab 91477) | tile 100+20 | +762 (SE 141), 0/12 | CÓ HẠI | – | tile thứ 2/3 tốn hơn số hàng nó tiết kiệm; bất kỳ cách chia nào cũng xáo lại list đuôi | V12-E783.md |
| t90 (ao13) | 90+30 | +731 (SE 135), 0/12; static −723 | CÓ HẠI | – | static chỉ SAI chiều ở đuôi AO | V12-E783.md |
| t3 (ao13) | 96+16+8 | +2,215 (SE 149), 0/12; static +511 | CÓ HẠI | – | – | V12-E783.md |
| tb16 (ao15 lab3 95579) | chỉ tile 104/16 | +942 (paired 122), 0/12 | CÓ HẠI | – | tile cuối nhỏ hơn dời ws ra sau gather (tuần tự hoá lane đuôi của c0) | ARCH_AO_B §5-6 |
| quét tb* static (s7) | tile 104/16 và 100/20 × chain 12/14/18/20 copy | c12–c18 đều đặt ws sau gather (mode tb16); c20 = mode ao13 c20 | ĐÓNG BẰNG PHÂN TÍCH | – | không biến thể tile b nhỏ nào giữ được list đuôi của aoYG | ao15_progress 20:58 |
| quét S5 (ao16): ics × tile {98/22, 100/20, 102/18, 104/16} × copy {14..20} + tb22, tb18 | tile cuối nhỏ hơn | chỉ có tile b = 22 hàng (98/22) giữ được list của aoYG | ĐÓNG BẰNG PHÂN TÍCH | – | – | ao16_progress |
| tb22_c17 (ao16) | aoYG, tile 98/22, 17 steering copy | lab5 100168 −38 (SE 144, pSE 168) 5/12 | TRUNG TÍNH | – | bớt 2 hàng chỉ ~44 cycle trên contraction b của c1, bị pha loãng | ao16_progress |
| fe3 (ao15 lab3 95579) | tile a tách 44+52 hàng (a2 dispatch ~2.9k, sớm 19k trước khi a1 land); 3 contraction | +1,257 (paired 89) 0/12 | CÓ HẠI | – | một lệnh trọng số thêm dù dispatch sớm cũng KHÔNG miễn phí | ARCH_AO_B §6 |
| B 106+14, 19 copy (t106c19) = V10-E223 | c_last ngắn hơn (c14 ~1.24k so với c24 1.39–1.49k) | chỉ trace: +654 (SE 119, 0/6) | CÓ HẠI | – | đổi tile làm xáo list đuôi (gather ở lúc c14 hoàn tất, wsVRF 1.07k sau POST2 trên c1) | V10-E223.md |
| AO-S1 (ao_gap đề xuất + review) | tile cuối nhỏ hơn (104+16 hoặc 106+14) + chain dài hơn (contrB 851 + 22.2/hàng) | đề xuất 100–250; review: AB − A trace +654, info +204 | ĐÓNG BẰNG PHÂN TÍCH (KILL) | – | các biến thể chia tile thua 6/6 trên HW | ao_gap §4; ao_review §2 |
| sd1 / sd2 / sd1x / sd2x (ao19, lab 101926, chỉ-DMA) | 1 lệnh K16 so với 2 lệnh hiện tại | sd1 25,757 / 25,711 (306 B/tick/cl); sd2 tile a 20,651/20,557 (305), tile b 6,338/6,483 (**248 B/t**, +1.1–1.35k), span ~+1.3k; sd1x +0.85k (store e0 chạy đồng thời làm tuần tự hoá) | phát hiện | – | lệnh thứ 2 tốn +1.3k; 1 lệnh thì 120 hàng contraction sau byte cuối (~+2.1k); một DMA nhỏ chạy đồng thời trên engine 0 tốn cho stream của c0 đúng bằng thời lượng của nó | ao19_report |
| K-split P1 thay thế (ao18) | tile a = Qs%256 [0,192), tile b [192,256), 2 TRF, VE cộng | SIM ≤ −0.1..−0.27k nếu free, > 0 nếu bị giữ | ĐÓNG BẰNG PHÂN TÍCH | – | – | ao18_report |
| aoS2 cận dưới (static, V11) | mô hình tường c1 | Arena-warm LB ~35.0k; chính thức ~34.3–34.9k | ĐÓNG (sau bị bác) | – | board #1 AO 31,867 bác bỏ "sàn" này: sàn chỉ là artefact của mô hình | AO_STRUCTURAL.md s2 |
| thiết kế S0–S16 (BACKWARD_AO) | aoYG không lãng phí, hop ngắn hơn, timing sync, T-split, P4, chia bất đối xứng mịn, AO-B1, 1 lệnh trọng số, lane x_hi/x_lo (S11 −0.13), K-split giữa 2 cluster (S12 tệ hơn), tile cuối nhân bản (S13 ≥ +0.5), hop qua HBM (S14 +0.7 static), Σ partial trên c1 (S15 ~0), Gram (S16 +0.1..+1.0) | tốt nhất đạt được ~34.2–34.4k chính thức | ĐÓNG BẰNG PHÂN TÍCH | – | sàn 6.4k sau byte cuối của cluster về muộn | analysis/v18/BACKWARD_AO.md |

#### 3.2 Nhóm B — Layout trọng số O và tốc độ stream

| tên/id | thay đổi | kết quả | kết luận | được / mất | lý do, bài học | nguồn |
|---|---|---|---|---|---|---|
| A24 probe layout O-weight (72167, chỉ-DMA) | P0..P5, P4b/c | P1 1 lệnh −1,440 ± 291; P2 K 8×512 −2,025 ± 66; P3 −1,454; P5 thứ tự slice hoán đổi −1,957 ± 112; P4b cả hàng trên 128 slice +16,901; P4c 64 slice +52,440; T = 1,373 + B/614 | ĐÓNG (plateau, thời 0.6) | – | AO của leader cần ~903 B/cycle (cao hơn mọi số đo 47 %); rate tăng theo số slice sống; trọng số v11 bắt đầu ở 4,455 trên HW (head-of-line sau config DMA x-broadcast + pass) | ao-lab/RESULT.md |
| P3 "K 2×2048" | piece dài hơn | 580 B/c = P1 (72167) | ĐÓNG BẰNG PHÂN TÍCH | – | piece dài hơn trên mỗi slice không giúp | CRITICAL_PATH_AO.md s0 |
| E203 probe P2 chỉ-DMA (82835) | P1 = v9 (K 4×1024, 64 nhóm hàng, 24+6 trên 30) so với P2 (K 8×512, 32 nhóm hàng, 48+12 trên 60, piece 512 B aligned) | DMA wall 28,835 so với 28,263 = −572 (SE 100, 5.7σ); tile A 21,901 → 21,520 | qua kill gate → E205 | – | kết quả 1 lệnh thời 0.6 chuyển sang được dạng 2 lệnh của 0.8.1 | V10-E203.md, microbenchmarks/AO_RESULTS.md |
| E205 **P2 trong kernel (p2c28)** | trên INTER: OutputRowsColumns [H/60%32, Qs/512], seed x [Qs/16%32, Qs/512] + fan-out x Broadcast1{32,8} (kiểm trên host), tile 48+12 trên 60, K reduce inter-slice 8 đường, chain 28 copy | static 22,446 (+67); lab3 82931 −699 (167), E[max2] −837 (179) 8/9; verifier 82986 −614 / −669 (136); pooled bỏ outlier −560 / −670 | CÓ LỢI (V10; sau bị V18 AO thay) | T0 −0.5..−0.7k trên cả 2 cluster | lợi stream dời T0 trên cả hai cluster, không bị coupling barrier hấp thụ | V10-E205.md, ao10_notes.md |
| covM probe layout AO (job 90990) | K16 / K8 / K16s / K4s chỉ-DMA | B/t/cl: K16 309.9, K8 297.6, K16s 304.7, K4s 286.7 (chip 610.9 / 590.8 / 598.7 / 563.7) | phát hiện: K16 (layout V18 đang dùng) nhanh nhất | – | mô hình coverage đoán K8 > K16s: sai | covM_notes.md |
| probe layout stream AO khi chưa có layout ứng viên (V14) | – | mọi layout lớp nhanh đều bão hoà ở 305–312 B/tick/cl chỉ-DMA | ĐÓNG BẰNG PHÂN TÍCH | – | – | V14_PLAN §4 |
| K11 stream AO (ao_gap) | – | stream 2 lệnh dài hơn 1 lệnh chỉ-DMA 1.0–1.3k (khởi động lệnh 2 ~0.6–0.85k + giao thoa ~0.3k) | phát hiện | – | – | ao_gap §2.2 |
| H4 stream (AO_R3) | – | stream 2 lệnh trong kernel dài hơn 1.0–1.3k; phần dư mỗi lệnh ~0.97/1.08k; dispatch sớm chỉ giấu 15–38 %; một lệnh trọng số trong kernel tổng ~+1.0k | phát hiện | – | – | AO_R3_external.md |

#### 3.3 Nhóm C — Đầu kernel: fan-out x, steering đầu, memset pin

| tên/id | thay đổi (cơ chế) | kết quả | kết luận | được / mất | lý do, bài học | nguồn |
|---|---|---|---|---|---|---|
| A22 fan-out x bằng Broadcast1{16,4} | bỏ config DMA 719 cycle | 22,709 (−719); 71578: QKV PASS rồi AO TREO device (HAL timeout) | CÓ HẠI (treo) | – | Broadcast1 là all-gather dọc slice1, không phải replicate; compile không kiểm switch → phải pre-flight trên CPU emulator | ao-e1 §5 |
| A27 xrep (75131, ABBA ×8) | thay `CustomBroadcast` ring-64 (pass Main 9.1k + config DMA 719 giữ issue trọng số O tới ~4.4k) bằng `to_dm` nhân bản vào `OutputBroadcastRows` (2 KiB/slice, 512 KiB/cluster) | work 37,805 so với 38,092; median total 54,450 so với 53,783 | TRUNG TÍNH (đóng) | – | load HBM→DM nhân bản ~130 B/cycle (512 KiB ~4.0k); trọng số bắt đầu 4.0k thay 4.4k; ở 0.6, đầu AO ~4k với mọi cách fan-out | ao-e1/xrep_ab1.log |
| P6 **v4b** (0.8.1) | seed x 32 B/slice + all-gather `Broadcast1{64,4}` (cp-ao xag256, kiểm trên CPU) thay ring-64 + config DMA | AO 22,705; 42,593 (77919) | **CÓ LỢI** | AO −2.8k so với v4a: trọng số issue lúc 2.1k (trước 3.9k); seed DMA 2.0k HW | bác X36 "đầu AO ~4k với mọi cơ chế": seed all-gather rẻ hơn replicate hoặc ring-64 | analysis-work/cp-ao/p081 |
| AA1 ao-qkv zb / z-chain (lab1 79019) | switch Broadcast1 giả (WAW lên xin) để DMA w26 list trước switch thật; 2 cái giả nối nhau | ~22,705 / 22,516; +0.8k | CÓ HẠI | – | lệnh TU ở đầu list làm trễ issue seed DMA; w26 chỉ issue khi pass giả HOÀN TẤT | ao_qkv_notes |
| AA17 ao9 hs / hs2 / hs6 (steering đầu) | pass Sub nhỏ → VRF + pass rác trên Main WAW vào output switch để trọng số list trước switch; hs2 trọng số qua `to_dm_view`; hs6 một switch giả trên Main trước seed | 22,658; hs2 40,345 (+0.8k); hs6 40,428 (+1.04k) | CÓ HẠI (nhánh head đóng — về sau aoYG mở lại) | – | hàng đợi dừng tại lệnh TU của một context BẬN cho tới khi pass hiện tại xong; sàn issue trọng số = seed land + ~0.35k | ao9_notes |
| steering đầu (V10 tổng kết) | issue trọng số sớm hơn | sàn = seed land + 0.35k (v9); hs2/hs6 +0.8 / +1.04k | ĐÓNG (thời v9; mở lại bằng aoYG trên V18) | – | mọi pass TU nằm trước trọng số đều giữ hàng đợi | CRITICAL_PATH_AO.md s4 |
| cổng head H (phân tích V11) | trọng số chờ POST ExplicitSync(21) + ~1.37k | switch dispatch − POST1 = 1,304–1,384 ở 16/16 cluster-launch warm; cold +3.4–3.6k | ban đầu kết luận "không biểu diễn được" → **SAI**, aoYG đã giải được | – | POST gắn với một lần ghi cục bộ dời được | AO_STRUCTURAL*.md s3/s6 |
| aoY probe | chỉ Sub memset vào đích gather y | POST 1164 → 1329 (= lúc memset xong) | phát hiện cơ chế | – | compiler đặt POST phía nhận ngay sau lần ghi cục bộ cuối vào y | pinA_notes.md, MEMSET_PIN_AUDIT.md s2 |
| **aoYG (V12-E601)** | (a) output switch `gathered` = DmTensor::new + Sub memset + commit_view (WAW pin q_wfirst của teammate); (b) Sub memset y (dời POST tới 2619); PE `GEN(memset_g) X22 LD(main_tu_1) DMA(x) ¦ DMA(weight) X23 WDMA(x) EXEC(switch) GEN(memset_y) LD X23 ¦ POST` | static 22,869 (+79); trace 91219: trọng số issue ~0.9k thay ~1.88k; lab 91244 −576 (Welch 202, paired 292, 7/8); lab 91252 −788 (paired 123, 12/12), E[max2] −854; **pooled −756 ± 113 (6.7 SE)**; 0 lần freeze / 80 launch; max abs d giống hệt | **CÓ LỢI → V12** | được −0.76k (≈ +0.7 % điểm); mất: +79 static; seed x issue lúc 0.28k thay 0.098k | memset-pin cũng dời được sync Core do compiler đặt | V12-E601.md, V12_RESULT.md, experiments/pinA/src/aoYG_projection.{rs,diff} |
| aoYp | chỉ memset y đã pad 4.8 KB/slice, không pin switch | POST 1948 sau trọng số 1694 nhưng trọng số vẫn sau switch; thứ tự Sub ở đuôi đổi (static 22,320) | ĐÓNG BẰNG PHÂN TÍCH (dự +0.6–1k) | – | chờ DMA residual sẽ chặn y_vrf và pass ms | V12-E602.md |
| AO-H4 head seed-trước (ao_gap + review) | seed x là lệnh TUC đầu tiên | trần ~0.18k trace (khác biệt = 4 lệnh TUC trong block 0 + một marker chỉ-trace) | ĐÓNG BẰNG PHÂN TÍCH (KILL) | – | dưới sàn lab; memset không input luôn ở đầu kernel | ao_gap §4; ao_review §2 |
| K14 head aoYG (phân tích) | – | 14 op PeCore CoreImp (tile offset cho `.tile()`) + GEN(memset_g) + X22 + LD(main_tu_1) nằm trước DMA(x); seed chậm hơn 0.18–0.24k và phần này lộ ra | phát hiện | – | – | PE dump pinA/s/pe_aoYG/code_0.c |
| X-PUSH (c0 nạp x cho cả 2 cluster) | – | trên giấy | ĐÓNG BẰNG PHÂN TÍCH | – | FIFO + readiness sync: push land sau tile a, hoặc c0 tốn ≥ 1.2k + một sync ở head | ao_gap §4 |
| load operand trước T0 / chia trọng số bất đối xứng / gather về c1 (V10) | – | 3 load ở head tốn ~3.3k so với cắt đuôi 1.3k; một cluster một mình chỉ ~376 B/c; gather về c1 tệ hơn ~0.6k ở skew r2 | ĐÓNG BẰNG PHÂN TÍCH (mô hình) | – | – | CRITICAL_PATH_AO.md s4 |
| operand ở head (ws/nw/res trong cửa sổ skew) (ao_gap) | – | mô hình fluid: c0 mất cửa sổ stream một mình, c1 +0.2k; ròng −0.2..−0.3k | ĐÓNG BẰNG PHÂN TÍCH | – | không hơn R1 | ao_gap §4 |

#### 3.4 Nhóm D — Tính toán epilogue (RMSNorm, scale, số pass)

| tên/id | thay đổi (cơ chế) | kết quả | kết luận | được / mất | lý do, bài học | nguồn |
|---|---|---|---|---|---|---|
| A04 AO-004 (REV-P) | fuse epilogue: channel scale → bf16, post-norm, residual trên 8 slice (10 DMA / 7 op Main); bỏ hop HBM về 1 slice + 5 op | 29,908; 64,668 (35441), max abs d 0.0 | CÓ LỢI | – | làm tròn một lần `bf16(f32(Wx)·s)` khớp tham chiếu tuyệt đối: hãy khớp đúng các điểm làm tròn của reference | TSV:28 |
| A06 AO-006 (REV-X) | contraction O FP8×FP8 với x tách hi/lo (`hi = f8(kx)`, `lo = f8(kx − hi)`, k = 128), 1/k gộp vào pass channel-scale | 25,827 (−2,908); 57,634 (35543), pooled 57,783 (n = 10), min 49,948 | CÓ LỢI | – | AO reduce K hoàn toàn (16 output/slice) nên contraction không lookup chạy nhanh; CHÚ Ý: fixture thời đó x = ±1 → lo = 0, đường lo chưa từng được thử | TSV:41, OL§3c |
| A10 AO-008 | gộp sqrt rms vào reduce giữa các slice (`Clip(Add,EPS)` → `AddF(EPS)`) | 24,902 (−282); 54,654 (39615) | CÓ LỢI | – | `Clip` không chuyển sang được `Narrow`; epilogue Main 2,440 cycle 100 % DMA idle | TSV:82 |
| A11 RMSNorm phân tán | chỉ trao đổi thống kê giữa các cluster | không build | ĐÓNG BẰNG PHÂN TÍCH | – | vẫn 3 op (gather + store scalar + load scalar), độ trễ dời vào chuỗi Main: chắc chắn tệ hơn | guide-fork |
| A14 **epilogue AO 5 → 3 pass** (ao-dist E1) | bỏ 1/k của FP8 split (triệt tiêu qua RMSNorm vì mean-square >> EPS); giải phóng `Mul1`, nên channel scale áp trong cả consumer mean-square lẫn normalise (normalise nhận luôn residual add) | −742; 56,997 (41475) PASS max abs d 0.01562 (0.71 %, tol 5 %); trong ứng viên 41538: 52,005 | CÓ LỢI | – | pass epilogue bị giới hạn bởi chi phí cố định: reduce 1 phần tử 347 so với pass 480 phần tử 401; nên bỏ pass, đừng thu hẹp pass | guide-fork/RESULT.md |
| A15 v11 AO (chọn 09-21) | fork công khai: commit hi/lo trực tiếp, không có pass gộp 1/128 | 23,428; 59,330 (71345), 53,470 (71414) | CÓ LỢI (main) | – | RESULTS_v11 ghi "untiled" nhưng source là 26+4 (projection.rs:379-380): tin source | RESULTS_v11 |
| A20 sqrt vào mean-square | −1 pass | không biểu diễn được | KHÔNG BIỂU DIỄN ĐƯỢC | – | typestate: không có stage Fp sau reduce | ao-e1 §3 |
| P4 port v3e (0.8.1) | 0.8.1 tính inter-slice reduce trên 8 slice sống pad thành `1 # 32` là ~2.3k static / ~4.5k HW (0.6: 462); sửa: partial trong từng slice + all-gather `Broadcast1{8,1}` + reduce trong slice | AO −3.6k so với v2 (AO 24,185 static); sai số bit-identical; probe: không inter-reduce 401, output dummy 2,694, pass InterFirst riêng 2,562, switch gather 289 | CÓ LỢI | – | – | RES |
| AA3 **g1** | một gather DM→DM từ 64 nhóm hàng của cả 2 cluster thẳng vào slice epilogue của cluster 0 (thay gather về 1 slice + dời cluster) | −1.15k HW (lab3) | CÓ LỢI (v7) | – | – | ao_qkv_notes |
| AA4 **g1v** | pass sqrt ghi rms VRF từ Main (`.vector_final().to_vrf(&mut device.sub)`), không commit DM + Sub StoVrf | 22,358; lab2 79106 42,835 → 41,084; lab3 g1v 41,138 so với g1 41,686 (−0.55k) | CÓ LỢI (v7) | – | Sub rảnh trong đuôi AO | ao_qkv_notes |
| AA5 **g1v16** | epilogue trên 16 slice × 240 kênh (`AoEpilogueSlices = m![1 # 16, H/240]`, config Broadcast1{16,1} mới chạy tốt) | 22,714; −0.3..−0.5k (lab4 79167: 40,740 so với 41,231) | CÓ LỢI (v7) | partial-ms 1,094 → 835, normalize 995 → 695 | – | ao_qkv_notes |
| AA13 ao8 **e32** = cand2 | epilogue 32 slice × 120 (`Broadcast1{32,1}`, `config_switch` kiểm trên host; `m![1 # 8, H/120]`) | 22,379; lab4 79712 median warm v7 41,312 / cand1 40,033 / cand2 39,022 (−2.29k, −5.5 %) | CÓ LỢI (v8) | pms 750 → 564, final 681 → 537 | 64 slice không làm được (60 f32 không đủ một packet 32 B) | ao8_notes |
| ver15 của teammate (InterFirst) | inter-slice reduce trên view regroup KHÔNG pad `m![Dummy256/16, H/240]` (reduce có pad `1 # 16` đắt ~5.8×) | AO 22,714 → 22,207; cùng job 12 cặp: inter −1,346 (12/12), `first` −650 (11/12) | CÓ LỢI (của teammate, 8.1527) | – | – | experiments/ver15/ver15_8.1527x_duc/RESULTS.md |
| AA15 ao9 **INTER** (ý tưởng ver15) | mean square + tổng 32 slice trong một pass: inter-slice reduce trên view regroup KHÔNG pad `m![Dummy256 / 32, H/120]` | 22,379 (static trung tính); lab1 79939 39,523 → 39,020; lab2 80296 → 39,227; lab3 +0.04k; lab4 −0.37k (TB −0.29k qua 4 lab) | CÓ LỢI (lợi duy nhất lặp lại được ở ao9) | lợi nằm ở CLUSTER 1 (không còn switch all-gather ghép cặp trên cluster pad) | c0 đuôi riêng dài hơn | ao9_notes |
| E201 INTER (hợp nhất ao9, V10) | pass mean-square kết thúc bằng `vector_inter_slice_reduce` 8 đường; bỏ pass reduce all-gather riêng | lab1 82835 −341 (boot SE 128) 6/6; lab2 82904 −699 (176) 6/6; **pooled −520 / −541 (SE ~0.12k), 12/12**; 36/36 PASS | CÓ LỢI (V10; sau bị V18 AO thay) | chuỗi sau sync ngắn hơn trên cả hai cluster | vượt trần ~0.3k mà reviewer đặt ra | V10-E201.md, ao10_notes.md |
| AA16 ao9 tb / tbi | contraction với `Dummy2` làm hệ số time-broadcast (mỗi flit trọng số fetch 1 lần, packet 64 B) | 22,866 / 22,803; 39,470 / 39,658 | TRUNG TÍNH | c24 2.8k (v8 4.3k), c6 1.03k (v8 1.8k) nhưng ws/nw/res bị tuần tự sau Sub VRF của chúng (page-0 WAR) | residual muộn, DramReuse gate store | ao9_notes |
| TB contraction (V10 tổng kết) | c6 1.82k → 1.03k | 5 lab: −0.23 / +0.13 / +0.03 / −0.18 / +0.37k | TRUNG TÍNH (đóng) | – | load ws giữ lane suốt cửa sổ c6 | CRITICAL_PATH_AO.md s4 |
| C1a AO SQF (pha 3a) | gộp sqrt vào pass final | static pass final −156, makespan 22,446 (0; gate bởi DramReuse) | ĐÓNG BẰNG PHÂN TÍCH (DROP) | – | pass ms sẽ thành Main→VRF khi Sub StoVrf res VRF của c0 đang chạy 1.07–2.15k: r1 +0.66k | IDEAS_COMPUTE.md C1 |
| AO-WSFOLD | gộp ws vào output contraction (MulF(&ws_vrf) sau inter-slice reduce qua InterFirst) để slot trước gather nhường cho nw | số học tie: chuỗi operand và chuỗi gather+sync lệch nhau ≤ 10 cycle; ws phải land trong stream bão hoà (+0.4–0.9k T0) | ĐÓNG BẰNG PHÂN TÍCH (ròng ≤ 0) | – | – | IDEAS_SCHED.md s4 |
| AO-E64 (ao_gap đề xuất + review) | epilogue 64 slice (60 kênh + 4 pad 0) | ước 50–300; review: bước e32 của ao8 cho ~1.55/~1.2 tick/phần tử → E64 ~−0.15k trước chi phí (reduce 64 slice ~2.3k static có pad, gather 32×240 B vào slot 256 B, memset pad, store 64×120 B) | ĐÓNG BẰNG PHÂN TÍCH (KILL) | – | ròng ≤ ~0.1k, dưới sàn lab | ao_gap §4; ao_review §2 |
| isr (ao15c) | pass ms không có inter-slice reduce; tổng 32 slice InterFirst trong final | static ms 588 → 311, final 402 giữ nguyên, makespan 22,869 (final bị gate bởi Sub y_vrf/res_vrf) | ĐÓNG BẰNG PHÂN TÍCH (~0) | – | – | ao15c_progress 07:02 |
| yst (ao15c) | final stream y; ms→VRF giảm một nửa + eps/64; pass sw thay inv_sw; bỏ y_vrf | static 22,067, chuỗi sw trước WAIT2 | ĐÓNG BẰNG PHÂN TÍCH (không lab) | – | – | ao15c_progress |
| yst2 (ao15c lab1 99694) | ws_vrf·nw_vrf trong final, không có sw/inv_sw trên Main | static 22,099; +1,092 (pSE 111) 0/8; trace: final stream y = 0.57k (aoYG 0.73k) TỐT, nhưng WAIT2 qua ở T0 + 5.97 (aoYG 4.53) | CÓ HẠI | – | Sub to_vrf tốn 0.67–1.07k khi program của nó nạp cùng queue block (res_vrf 0.37k của aoYG có program nạp trước) | ao15c_progress |
| yst4 (ao15c) | yst + inv_sw của aoYG, đúng thứ tự op của final | static 23,134: res_vrf sau ms (Sub bận ghi VRF ms) → tuần tự | ĐÓNG BẰNG PHÂN TÍCH | – | – | ao15c_progress |
| rpy (ao15c lab2 99736) | head rpa + epilogue yst4 | static 22,382; +1,183 (pSE 131) 0/8 | CÓ HẠI | – | final stream y + ms→VRF tốn ~+0.66k thêm trên rpa | ao15c_progress |
| D3 epilogue Gram-contraction (cloud R1 / triage) | bình phương + tổng trong contraction engine, all-reduce broadcast giữa các slice, một pass Σy² → rsqrt → normalise → +res | phân tích +0.1..+1.0k | ĐÓNG BẰNG PHÂN TÍCH (KILL) | – | không có MAC f32 (cast.rs:365-405); TRF phải cùng Chip/Cluster/Slice với stream; contraction ≥ 1.4k so với ms 1.04k | AO_REDESIGN_triage.md |
| ic (ao16) | inv_sw → commit DM trên Main + Sub StoVrf (chính xác) | lab1 100086 +15 (pSE 78) 3/8 | TRUNG TÍNH | – | giữ fence F trước nw; thêm Sub StoVrf fence y_vrf | ao16_progress |
| ics (ao16) | inv_sw là pass VE trên Sub | lab1 100086 −162 (142); lab2 100103 −101 (100); lab3 100136 trace −104 (113); lab4 100151 (đổi thứ tự) +72 (95); pooled INFO **−46 ± 70** | TRUNG TÍNH | – | cắt chỉ chuỗi c0 thì bị pha loãng | ao16_progress |
| hi/lo contraction trên lane MAC (ao_gap) | – | trên giấy | ĐÓNG BẰNG PHÂN TÍCH | – | contraction bị giới hạn fetch 32 B/cycle; giảm một nửa bước time không được gì | ao_gap §4 |
| activation 1-f8 cho AO (ao_gap) | – | trên giấy | ĐÓNG BẰNG PHÂN TÍCH | – | sai số e4m3 ~3.6 % rms → ~0.14 max sau norm, vượt atol 0.05 + 1 % | ao_gap §4 |
| partial RMS trước gather (ao_gap) | – | trên giấy | ĐÓNG BẰNG PHÂN TÍCH | – | cần ws trên c1: +0.8–1.2k tuần tự trên lane găng, hoặc một exchange + sync thứ 2 | ao_gap §4 |
| NEW-YVRF (triage) | "fast mode" của y StoVrf | y StoVrf 0.457k khi không có Sub program config nào theo sau trong queue block so với 0.748–0.753k khi config triple của res_vrf nạp phía sau | cơ chế là FACT; build: ĐÓNG BẰNG PHÂN TÍCH | trị giá −0.15..−0.30k đối xứng | không có handle: res_vrf cần VRF f32 mà residual là bf16; StoVrf không cast được; config Sub nhúng địa chỉ | AO_REDESIGN_triage §2.3 |
| NEW-FINALIL (triage) | y là DM stream thứ 2 của final (begin_interleaved) | – | KHÔNG BIỂU DIỄN ĐƯỢC | – | begin_interleaved cần D/Chip/Cluster/Slice/Element giống hệt (context.rs:97-101); hướng y-primary = họ yst/yst2/rpy +1.09..+1.18k | triage |



#### 3.5 Nhóm E — Thứ tự list ở đuôi, nạp operand (ws/nw/res), steering chain

Kết luận chung của nhóm này: list đuôi của aoYG là một **tối ưu cục bộ**. Mọi thay đổi đặt `wdma(res) POST3` lên trước WAIT2, hoặc đẩy Sub ws_vrf ra sau gather, đều tốn +0.7..+2.4k (ao13, H11 trong §6).

| tên/id | thay đổi (cơ chế) | kết quả | kết luận | được / mất | lý do, bài học | nguồn |
|---|---|---|---|---|---|---|
| A02 AO-002 | kéo load residual + channel-scale lên cạnh trọng số O (thứ tự chương trình) | 35,876 giống hệt | TRUNG TÍNH | – | scheduler đã tự đặt chúng; thứ tự chương trình không có tác dụng (lần xác nhận thứ 1) | TSV:6 |
| A17 AO-E1a | xếp load `norm_weight`/`residual`/`weight_scale` vào các khe DMA của v11 (79/508/1,412/9) | 23,428 timestamp-identical; trần thật −88 (không phải −2,008) | ĐÓNG BẰNG PHÂN TÍCH | – | khe 1,412 = đuôi Main 4 pass, không còn DMA; khe 508 = phần dư của độ trễ cố định `y.to_hbm` → `y.to_dm` 1,600 cycle, đã lấp 1,092 bằng load; thứ tự chương trình không có tác dụng (lần thứ 5) | ao-e1/RESULT §2 |
| A19 store từ layout có pad | store y từ layout 30 kênh có pad (bỏ DmaStos) | 25,295 (+1,867) | CÓ HẠI | – | cùng luật với Q02 | ao-e1 §3 |
| AA6 ops3 (E6) | DMA 3 operand epilogue vào các tile của một tensor (liền nhau) | 23,218; không chạy | ĐÓNG BẰNG PHÂN TÍCH | – | static đặt ws trước w4 (+788) | ao_qkv_notes |
| AA8 ao8 opsmain / c_opsmain / c_resm / wsm | VRF của 3 operand (hoặc residual) ghi từ Main | 22,188 / 22,367 / 22,603; 41,132 / 40,844 / 41,049 | CÓ HẠI | – | pass VRF trên Main chậm hơn; page WAR (nw chờ ws_main HOÀN TẤT) | ao8_notes |
| AA9 ao8 **chain26 / chain24** | 26 (rồi 24) pass identity-copy (trên Main; 328 static / 445 HW mỗi pass) đặt trước activation split, để x_trf kết thúc (static) sau khi tile trọng số kế tiếp bắt đầu: w4/w6 được list TRƯỚC c26 và issue lúc StoTrf dispatch, nằm trong FIFO ngay sau tile đầu | 22,893; warm 40,415 (TB 40.1k) so với 41,081 | **CÓ LỢI** (cùng với 24+6) | w4 land 30.1–30.6k (v7 31.4–31.6k), c4 kết thúc sớm 0.85k | "steering bằng static schedule" | ao8_notes |
| AA11 ao8 c_t24 = cand1 (lab2 79582; harness thường 79639) | chain24 + tile 24+6 | warm 40,416 (TB 39.94k); harness thường AO 40,362 | CÓ LỢI | đuôi từ lúc gather xong → store xong 5.3k (v7 6.4k) | giờ kết thúc kernel do cluster 1 quyết định: end = max(c0, c1 + ~1k barrier message) | ao8_notes |
| AA12 ao8 agrep / agsub (lab3 79665) | switch all-gather relabel cluster `Dummy2`; all-gather trên Sub | 22,316 / 22,522; 39,303 / 40,499 so với cand1 39,657 | CÓ HẠI (bỏ) | – | coupling all-gather của c1 không do mapping `1 # 2`; pass switch trên Sub 1.19k so với 0.70k | ao8_notes |
| AA14 các hướng ao8 đã đóng | tách gather qua `cluster_tile`/`slice_tile` ("tiling a non-in-slice axis is not lowered yet"); pdma HBM→HBM gộp 3 operand (tuần tự trên engine DMA, đảo tile, chèn ExplicitSync); pass giả warm-up ở head | – | KHÔNG BIỂU DIỄN ĐƯỢC / CÓ HẠI | – | – | ao8_notes |
| AA18 ao9 tbi_ops3 (lab2 80296) | TB + INTER + 3 operand `to_dm_view` vào TILE của một DM tensor | 23,612; 41,027 **FAIL** (27 % trong tol, max 4.4) | CÓ HẠI (SAI KẾT QUẢ) | – | `to_dm_view` vào tile của một DM tensor (hoặc đọc chúng) hỏng trên HW | ao9_notes |
| AA19 ao9 i_t1_ov / ti_t1_ov / tbi_t1_ov_c30 | T1 residual trước; OV `to_dm_view` full-view vào buffer mới; chain 30 | 22,347 / 22,155 / 22,559; pooled ti_t1_ov − inter −227 / +132 / +26 (TB −23) | TRUNG TÍNH | – | TB+T1 dời gather c1 sớm 0.55–0.64k nhưng làm dài đuôi c0; page-0 WAR cũng áp cho load `to_dm_view`; "position bias": arm đầu block chậm hơn 0.1–0.3k → cần xoay thứ tự | ao9_notes |
| AA20 ao9 PI page isolation (lab5 80415) | phần tử `AoOperand` pad tới `m![H % 120 # 15360]` (30 KB/slice) để buffer operand nằm ở page khác | static giống hệt; inter 39,005, i_pi 39,633, i_t1_pi 40,035 | CÓ HẠI | – | allocator dùng lại dải đã chết → WAR chồng địa chỉ thật (residual chờ gather xong); `# 16384` lỗi "cannot factorize common axis H" | ao9_notes |
| AA22 / V12n **WSM** | ws VRF = ws + (res·0) trên Sub để residual là operand cần đầu tiên | static i_wsm 22,521, tbi_wsm 22,329 …; ao9 lab8: tbi_wsm 41,274, tbi_wsm_t26_c26 40,908 so với inter 38,534; V12: +2.4k | CÓ HẠI | – | pass đọc VRF do Sub ghi bị dài ra | ao9_notes; V12_X3 |
| E202 NWFIRST (V10) | issue DMA nw trước ws-VRF exec; 11 biến thể static | static 22,347–22,379; chuỗi giấy +0.7..+1.1k | KHÔNG BIỂU DIỄN ĐƯỢC (không chạy Arena) | – | trước gather chỉ vừa đúng một operand load + VRF (khe lane 680 static < 2 × 616) | V10-E202.md |
| thứ tự list đuôi T1/T2/OV/PI/WSM/OPS3/chain/26+4 (V10 tổng kết) | sắp lại thứ tự | ≤ 0.4k; WSM +2.4..+2.7k | ĐÓNG | – | – | CRITICAL_PATH_AO.md s4 |
| gather nửa của mình (ring-16 all-gather, begin_interleaved) | – | nw bị gate bởi ws VRF (page-0 WAR), không phải bởi gather | ĐÓNG BẰNG PHÂN TÍCH (trần ~0) | – | – | IDEAS_MOVEMENT.md s3 |
| P2 quét độ dài delay chain | 24 / 28 / 30 / 32+ steering copy | static p2c24 22,528 (c48 list trước w12), p2c28 22,446, p2c30 22,460, ≥ p2c32 tệ hơn | 28 được chọn | – | chain giữ trọng số thứ hai được list sau contraction đầu tiên | ao10_notes |
| aoN (V12 → ao13) | bỏ chain 16 copy (tile b list sau contraction a) | static +79, Main 13,734 → 9,254; screen 91304/91306 AO warm TB 35,749 so với 35,204; ao13lab1 91477: **+986 (115), 0/12** | CÓ HẠI | – | chain giữ tile b xếp ngay sau tile a (~1.0k: toàn bộ khởi động của lệnh trọng số 2 bị lộ) | V12-E781.md |
| c20 (ao13) | ws list trước contraction b (20 copy) | +911 (125), 0/12; static −416 | CÓ HẠI | – | static chỉ sai chiều | V12-E782.md |
| p5a (ao13, AO-5) | residual trước norm_weight trong source | static giống hệt | KHÔNG BIỂU DIỄN ĐƯỢC | – | thứ tự source bị bỏ qua | V12-E782.md |
| p5c (ao13) | nw + res trong một DM tensor | +2,397 (99), 0/12 | CÓ HẠI | – | – | V12-E782.md |
| memset-pin ở đuôi (ao13, V12-E784) | dời một sync ở đuôi bằng memset | các POST ở đuôi gắn với truyền DMA / đọc DRAM (POST2 sau wdma(gather), POST3 sau wdma(res), POST4 sau store), không gắn với lần ghi cục bộ dời được | KHÔNG BIỂU DIỄN ĐƯỢC | – | memset chỉ dời POST ra muộn hơn | V12-E784.md |
| AO-5 (ý tưởng X3) | cho mỗi operand một consumer rẻ sớm để ALAP đặt DMA của nó trước gather | ý tưởng (3 pass thêm ~0.4k HW mỗi pass) | ĐÓNG BẰNG PHÂN TÍCH (xem ao13 p5*) | – | – | V12_X3 §4 |
| aoP (X3 AO-1) | pool operand qua to_dm_view | 23,248 (+458); load vẫn sau gather (788 so với 616) | CÓ HẠI | – | – | V12_X3 §4 |
| AO-2 (X3) | dời đuôi sang cluster muộn | không biểu diễn được (pad chỉ ở đuôi; `asymmetric_cluster_slice` chọn 1 phần tử/cluster) | KHÔNG BIỂU DIỄN ĐƯỢC | – | – | V12_X3 §4 |
| c14 (ao15 lab2 94324) | 14 steering copy (list sạch) | c14_i −11 (SE 112) 7/12 | TRUNG TÍNH | – | – | ao15_progress |
| prw / pwr / prwn / yvc / c13 / c15 (ao15 lab2) | pool operand ưu tiên residual (DR bị bỏ; lane res → gather → ws → nw); pwr ưu tiên ws (DR giữ); yvc PE giống aoYG | prw_i +1,457 (SE 125) 0/12 và **FAIL** (max abs d 1.00, 36 lỗi: residual sai); còn lại static | CÓ HẠI (SAI KẾT QUẢ) | – | bỏ DramReuse bằng cách sắp lại thứ tự có thể phá tính đúng: DramReuse là cơ chế bảo vệ thứ tự đọc/ghi | ao15_progress 17:41-18:37 |
| ops2 / ops3 / opsr / opsn / msr (ao16) | operand trong một DM tensor; pin Mode00 residual trong ms | DMA vào tile chuyển toàn bộ extent có pad (15K/30K), đuôi bị xáo; opsn = p5c (23,184); msr visa "operator schedule base graph has a cycle" | ĐÓNG BẰNG PHÂN TÍCH / KHÔNG BIỂU DIỄN ĐƯỢC | – | – | ao16_progress |
| D2 dời đuôi (cloud R1 / triage) | res sau tile b; res_vrf, W(res), POST3 ngay sau POST2; cloud −0.28k | d2s và d2l compile ra PE program BYTE-IDENTICAL với aoYG (code_0.c md5 e8b12add) | KHÔNG BIỂU DIỄN ĐƯỢC (KILL) | – | thứ tự source không phải handle; mọi hiện thực tương đối đều thua (rzW +1,751, rpa +520, rpb +302, c20 +911, t90 +731, A5/AB +1.5k) | AO_REDESIGN_triage.md |
| E0 dời lúc dispatch tile b (cloud R1 / triage) | dispatch tile b sớm ≥ 1.21k trước khi dữ liệu tile a xong; cloud −0.41k | thí nghiệm tự nhiên: slope −0.151 ± 0.035 (n 258, 7 arm), −0.38 ± 0.06 (chuẩn hoá theo rate) | ĐÓNG BẰNG PHÂN TÍCH (KILL) | – | loại mô hình M1 (cần −1): dispatch sớm chỉ giấu được 15–38 % setup của một lệnh lớn; fe3 +1,257; c14 −11 | triage |
| danh sách đuôi IDEAL (cloud R1 / triage; ARCH_AO_B §7.1) | cùng số pass, các wait dời ra sau dispatch; −0.34k | cần handle của D2; NEW-cbfill lật đuôi ở 4/4 list static | ĐÓNG BẰNG PHÂN TÍCH (KILL) | – | scheduler static chạy Sub to_vrf eager ngay khi DMA của nó land | triage; ao16 |
| NEW-subswap (triage) | res_vrf, W(res), POST3 trước y_vrf sau WAIT2 | giá trị nếu làm được: ròng −0.1..−0.25k | KHÔNG BIỂU DIỄN ĐƯỢC | – | thứ tự Sub theo thời điểm land static | triage |
| NEW-cbfill (cbp1/2/3, cbf8..48) | filler identity-copy ping-pong WAW sau contraction a (trễ static) | cbp2 static 22,449, đuôi bị lật; cbf* filler đọc-chết bị dời ra sau contraction b (22,099) | CÓ HẠI khi build; **GIỮ CÔNG CỤ** | – | filler chỉ đọc không tạo WAR; ping-pong WAW trong cửa sổ Main idle làm trễ start static của một writer phía sau với chi phí HW ~0 | triage expr lens |
| N1 (n1z, n1p, n1pz; cloud R2 → ao18) | nw đặt cạnh ws qua pair-mode / pin Mode00 | n1z static 22,099, trace 101042 +1,244 (218) 0/6 (inv_sw pair trên Main 0.97k so với 0.51); n1p / n1pz static 22,869, tệ hơn | CÓ HẠI | – | pass VE bị pin nằm SAU sync gather; pair pass tốn ~2× | ao18_report |
| wsm (ao15c lab4 99768) | ws_vrf là Main collect → to_vrf(sub) KHÔNG qua stage VE | compile được; **TREO device** "rank 0 left request 6 unanswered for 7s" | CÓ HẠI (TREO) | – | LUẬT: Main to_vrf phải có chuỗi VE (vector_init .. vector_final) | ao15c_progress 07:04 |
| wsm2 (ao15c lab5 99772) | ws_vrf trên Main có chuỗi VE x·1.0 | +213 (pSE 121) 2/8; trace: ws_vrf Main 1.23k và DMA nw vẫn dispatch lúc nó kết thúc | CÓ HẠI | – | exec TU nào list trước một DMA cũng chặn dispatch DMA đó tới khi exec xong | ao15c_progress 07:08 |
| AO-E: mỗi slice giữ nguyên hàng (ao16 / cloud / triage) | bỏ gather 1.3k bằng cách để mỗi slice giữ trọn hàng của tile cuối | phân tích 0 ± 0.1k | ĐÓNG BẰNG PHÂN TÍCH (không build) | – | 1.3k là hop DM→DM cross-cluster tới 32 slice epilogue của c0, không phải gather K-split (K partial đã reduce trong contraction b); cần x đầy đủ (8 KB) trên mọi slice (all-gather lớn gấp 16 lần switch x ở head: 4871 static / 10.1k HW) + 256 piece DMA nhỏ mỗi cluster | ao16_progress; triage |
| xếp tile b sớm để giấu F_b (ao_gap) | – | trên giấy | ĐÓNG BẰNG PHÂN TÍCH | – | luật qkvMB: phần tuần tự 0.8–1.7k bất kể xếp hàng lúc nào | ao_gap §4 |
| bỏ fence ws_vrf / chuỗi inv_sw của c1 (ao_gap) | – | bằng cận POST2 + 1.05k | ĐÓNG BẰNG PHÂN TÍCH | – | lợi 0 nếu POST2 không dời | ao_gap §2.4 |
| "đua" một DMA trọng số với contraction sớm dùng alias (ao_gap) | – | trên giấy | ĐÓNG BẰNG PHÂN TÍCH | – | data race, không đảm bảo đúng | ao_gap §4 |
| pws (dự kiến) | chuỗi identity-copy trên Main sau contraction a (trễ static contraction b) để DMA ws list trước W(tile b) | chưa đo | – (xem §8) | – | – | ao15c_progress 07:08 |
| PHÁT HIỆN CHÍNH ao15c (lab2 aoyg_t, 24 launch) | ai quyết định WAIT2 | WAIT2 của c0 được thả đúng 1 tick sau khi inv_sw của c0 xong ở 14/24 launch → chuỗi drain trước WAIT2 của chính c0 (W(ws) T0+1.59 → Sub ws_vrf 1.1k HOL → nw dispatch → nw 1.15k → inv_sw 0.52k) quyết định nhiều hơn chuỗi gather của c1 (10/24) | phát hiện | – | đòn bẩy: rút ngắn chuỗi của c0 | ao15c_progress 07:02 |

#### 3.6 Nhóm F — Sync, DramReuse, trao đổi cross-cluster, bất đối xứng

| tên/id | thay đổi (cơ chế) | kết quả | kết luận | được / mất | lý do, bài học | nguồn |
|---|---|---|---|---|---|---|
| A16 8+22 một sync | gộp 2 sync bằng tile 8+22 | 24,498; +0.7 % (71671) | CÓ HẠI | – | các sync chồng nhau vốn đã được giải quyết cùng lúc | dma-docs/FINDINGS §4 |
| A21 AO-E2 | gather y trên chip qua switch thay cho `DmaStos` 952 cycle | ~17k (CustomBroadcast 64→1) hoặc ~5k (Broadcast1) | ĐÓNG (vì chi phí) | – | CustomBroadcast ~ ring_size cycle/bước + config DMA 719 | ao-e1 §4 |
| A23 probe round trip y (71613, ABBA ×6) | thêm một hop y f32 → HBM → f32 | +3,173 static; +2,485 HW (SE 760, median +1,924) | phát hiện | – | round trip HBM giữa kernel ~2.5k HW, không phải "rào 5–20k" như RESULTS_v11 (đó là chi phí sync đầu tiên / launch) | ao-e1/RESULT |
| A25 exchange qua dma_gather | trao đổi không sync | vẫn là `ExplicitSync` + `DramReuse` | ĐÓNG BẰNG PHÂN TÍCH | – | sync gắn với hazard RAW trên HBM; key phải là trục có tên, index phải là tham số HBM | ao-e1 §9 |
| A26 RMS cục bộ (73458, ABBA ×5) | mỗi cluster tự normalize nửa của mình: không hop y, không sync giữa kernel | 22,376 (−1,052); −431 ± 2,049 HW (thắng 2/5); max abs d 0.04688 so với atol 0.05 | CÓ HẠI (SAI HÀM) | – | chi phí sync đầu tiên dời xuống sync cuối (như QKV); RMSNorm 2 nhóm là một hàm khác (lệch 0.071 / 0.147 / 0.032 so với oracle toàn cục), không được nộp | ao-e1 §10; CRITICAL_PATH_AO.md s3.3 |
| P2 port v2 (0.8.1) | gather y của AO + FFN bằng `to_dm` DM→DM đổi Cluster (compiler chèn ClusterSync thay cho hop HBM) | AO chỉ ~−1k | CÓ LỢI (ít) | – | trên 0.8.1, sync host không còn là chi phí của AO | RES |
| A-H7 epilogue từng cluster + AllReduce qua cluster_swap (V10) | piece operand 16 slice × 240 B, pms + reduce cục bộ, swap 32 B, store từng cluster; EPS cộng từng số hạng bình phương | h7a static 22,535 (+156), sync giống v9, số học host bit-identical, emulator trao đổi đúng; mô hình đuôi 10.3–11.3k so với 8.97k (+1.3..+2.3k) | ĐÓNG BẰNG PHÂN TÍCH (đã build, không chạy) | – | thay gather 1:1 trên lane, thêm một DMA regroup, giữ nguyên mọi sync, đến đồng thời nên giữ L | CRITICAL_PATH_AO.md s3 |
| V12k AO H7 (V11/V12) | giữ shard 1,920 kênh của mỗi cluster; chỉ trao đổi tổng bình phương từng phần | không lợi | ĐÓNG | – | giữ mọi sync; c1 vẫn tới barrier cuối cùng lúc | V12_CONTINUATION_STATE §4 |
| A5 đuôi không DramReuse (a5) = V10-E222 | residual_vrf có thêm op VE (res × 1.0 = RZ) pin vào ws VRF (WSP = ws + 0·(ws·RZ)) để residual được tiêu thụ trước POST2 → compiler bỏ DramReuse; static 21,778 (−1,012), 3 sync | lab 85694: info +1,294 (SE 173), E[max2] +1,257; trace +700 | CÓ HẠI | – | RZ 1.23–1.47k và WSP 1.24k trên HW (static ~0.3k, tính thiếu 4×) | V10-E222.md |
| A5 + B (a5t106) = V10-E221 | cả hai | static 21,698; info **+1,499 (SE 166)**; trace +1,354 | CÓ HẠI (~9 SE) | – | tiết kiệm DR (0.38–0.40k) < một pass Sub HW thêm (1.24k), A5 cần hai pass; cổng WSP của c1 chưa giải thích được | V10-E221.md, ao11_notes.md |
| các biến thể static khác của aoS2 | wsm 21,891, a3 22,126, a4 22,089 (bỏ DR, resVRF bị pin làm tuần tự), a2 22,108 / v4 22,088 (giữ DR), a5p 22,008 (DR quay lại); t104c19 / t108c19; a5t108 21,682 | chỉ static | ĐÓNG BẰNG PHÂN TÍCH | – | chỉ pin ws_vrf mới bỏ được DR | AO_STRUCTURAL.md s3-4 |
| bác trên giấy của aoS2 | chia byte bất đối xứng (~0.1k ròng), thống kê trước gather, H7 trên K16, bỏ gate operand c0 / NWFIRST, tile cuối < 12 hàng / ba tile | mô hình | ĐÓNG BẰNG PHÂN TÍCH | – | – | AO_STRUCTURAL.md s3 |
| AO-R1b (ao14, V14-E781) + r1k / r1h / r1hk | bỏ DramReuse: một writer Sub của đích gather lấy từ residual (`y := 0·f32(res)`), đặt việc đọc residual trước readiness sync | static: DramReuse biến mất (3 sync) ở mọi dạng, NHƯNG scheduler kéo DMA residual lên head (static 1003, block PE 0, trước x); writer Sub chạy trước memset(gathered) → POST1 3252 < tile a 3334 (bệnh cũ của head V11.1); makespan 23,500 / 23,482 | ĐÓNG BẰNG PHÂN TÍCH (static kill, 0 Arena) | – | pass mới đọc một operand DMA và ghi vào đích gather sẽ bị đặt ở head, làm hỏng head của aoYG | ao14_progress.md |
| AO-S1 (điều kiện theo R1b) / AO-R1a (writer trên Main) | – | không chạy / không build | ĐÓNG | – | R1b chết; R1a nằm trên đường của c1 | V14_PLAN §4 |
| review AO-R1 (ao_r1_replay) | replay | P (res trước T0): median −381 …; Q: W 0.4k → −330, W 1.2k → +241; P(KEEP) ~0.15 | ĐÓNG BẰNG PHÂN TÍCH | – | – | ao_review §2 |
| AO H7 cục bộ / epilogue do c1 sở hữu (V14_PLAN; critic) | c1 không gửi gather nữa (POST2 −1.26k) | ước ~−0.2k: chuỗi sau sync bị operand giới hạn trên lane của c1, vẫn trả L | ĐÓNG BẰNG PHÂN TÍCH | – | – | V14_PLAN §4; critic §5 |
| ao15 lab1 94208: r1wg / r1w / r1wgt104; r1d, r1m / r1mk (static) | reader residual trên Main ở head + memset(g) thứ 2 (bỏ DR) / không có memset thứ 2 / + tile 104/16 | r1wg +740 (97) 0/12; r1w +699 (100) 0/12; r1wgt104 +1,682; A/A aoyg2 −97 (124) | CÓ HẠI | – | đường x gần găng trên HW (switch 10.1k HW, 16 copy 7.9k, x_trf lúc 23.06k, slack đường x ~1.4k); memset(g) thứ 2 tốn +2.4k HW vượt quá slack đó | ao15_progress 17:12-17:27 |
| rzW / rzW16 / rzW20 (ao15 lab3 95579) | ws_vrf dựng bằng pass Main mang pin chính xác Mode00 max(v,v) lên residual_vrf → gather + res không chờ ở giữa, POST3 trước WAIT2 trên cả 2 cluster; + tile 104/16; 100/20 | rzW **+1,751** (paired 117) 0/12; rzW16 +1,806; rzW20 +1,587; rzW_t +1,981 | CÓ HẠI | – | thiết kế đúng như ý (gather lúc contrB kết thúc +1.30, res land +3.52, POST3 +3.52) nhưng WAIT2 qua ở +6.27 thay +4.53: PE-core POST/WAIT drain mọi thứ list trước nó; chuỗi mới res_vrf (Sub) 1.07k → ws_vrf' (Main) 1.16k → inv_sw 0.52k | ARCH_AO_B §5-6 |
| rzI / rzIr / rzM | pin Mode00 trong inv_sw / biến thể / bình phương Mode00 trong ms | rzI: static POST3 trước WAIT2 nhưng ws sau gather; rzM visa "operator schedule base graph has a cycle" | ĐÓNG BẰNG PHÂN TÍCH / KHÔNG BIỂU DIỄN ĐƯỢC | – | – | ARCH_AO_B §5 |
| rpa (ao15c lab2 99736) | pin max Mode00 của residual_vrf (reshape [1#120]) trong contraction a → DMA residual lên head, **DramReuse BỊ BỎ** (3 sync), static 22,473 | +520 (pSE 201) 2/8 | CÓ HẠI | – | – | ao15c_progress |
| rpb (ao15c lab3 99741; ao20 103229) | pin Mode00 residual_vrf trong contraction b; list head GIỐNG aoYG; DMA res giữa tile a và b; DramReuse bị bỏ; static 22,497; dự −0.26k | ao15c +302 (pSE 192) 3/8; trace +2,218; ao20 103229 **+728 (pSE 154) 0/6** | CÓ HẠI | – | – | ao15c_progress; ao20_report |
| T-split (ARCH_AO_B §4; AO_R3 H1) | mỗi cluster tự xong 1,920 kênh của mình; trao đổi tổng bình phương từng phần | phân tích: c1 T0 + s + ~8.4k so với hiện tại c1 T0 + s + 7.6k; AO_R3: +2.21k E[max2] như đặc tả (+2.59k theo luật WAIT2) | ĐÓNG BẰNG PHÂN TÍCH | – | cấu trúc hiện tại là đúng: cluster về muộn thì kết thúc sớm (no-op 0 byte), cluster về sớm gánh đuôi | ARCH_AO_B §4; AO_R3 |
| epilogue trên cluster 1 (cluster về muộn) | – | sẽ tiết kiệm ~0.75k ở sync gather | KHÔNG BIỂU DIỄN ĐƯỢC | – | layout sống chỉ trên c1 thì cần một offset | ARCH_AO_B §4 |
| all-gather + cả hai cluster cùng tính | – | 0 | ĐÓNG BẰNG PHÂN TÍCH | – | c0 vẫn chờ nửa của c1 | ARCH_AO_B §4 |
| AO-B1 chia không đều cân skew (ao_gap đề xuất, review, BACKWARD) | c0 stream nhiều hàng hơn c1 qua mapping SPMD có pad (H = 30 × 128: c0 16 nhóm / c1 14 + 2 pad) | mô hình +0.3k ở skew median, −0.9k ở s ~2.7k, E[max2] ~0; BACKWARD: luật HBM chia chung → PE0–2 của c1 vẫn xong cùng lúc | ĐÓNG BẰNG PHÂN TÍCH | – | chia mịn hơn cần mapping không affine (không biểu diễn được) | ao_gap §4; ao_review; BACKWARD_AO |
| D1 "U3-b" (cloud R1 / triage) | tile cuối đối xứng nhỏ đi delta hàng/nhóm + tile đuôi c chỉ trên c0 (2·delta hàng), gather list trước contraction cuối của c0; cloud −1.8..−2.4k (M1) | không biểu diễn được (visa từ chối output reducer pass-through có pad, 3 cách viết; LIR từ chối layout x-local; DM→DM vào tile 118/120 bị từ chối; scheduler kéo tile c lên trước tile b +6.7k static hoặc ra sau sync gather +6.1k); tiền đề sai (48/48 launch: tường của c0 là chuỗi operand) | KHÔNG BIỂU DIỄN ĐƯỢC (KILL) | – | nếu ép làm thì +0.3..+2.0k | triage |
| NEW-c1path / NEW-syncsem (triage) | hiệu chỉnh mô hình | wait cuối kernel của c0 HAI MODE: 3–67 hoặc 787–1,051 tick (72/72 launch); c1 trả một round trip trên các POST của chính nó (0.85–1.34k ở WAIT2, 1.43–1.86k ở DramReuse) | phát hiện (sau ao20: ở INFO dead zone không tồn tại) | – | – | triage |
| NEW-AOD chia không đều mịn (triage) | – | chỉ chia nguyên nhóm là biểu diễn được (v14 AO-B1); chia 16/14 nhóm SIM +0.64/+0.41k | ĐÓNG BẰNG PHÂN TÍCH | – | – | triage |
| H1 one-way hop (h1k, h1kn1z; cloud R2 → ao18) | chỉ y của c1 đi qua; phần gather của c0 là no-op 0 byte; qua `cluster_tile` + `to_dm_view` | **visa TỪ CHỐI: "a resolved index step counts in-slice cells, but this tile targets Cluster; tiling a non-in-slice axis is not lowered yet"** | KHÔNG BIỂU DIỄN ĐƯỢC | – | kéo theo P1/P2/P3/P4 cũng chết | ao18_report |
| n1r / n1r2 (+ n1r4, rr, rn, r2, r2t, r3, r4) (ao18) | ws_vrf pin vào residual (pair mode, chính xác), inv_sw = zip của n1z → **DramReuse bị bỏ** (3 sync) | n1r lab1 101022 **+1,409 (146) 0/8**; trace +1,023; n1r2 +1,544 (138) 0/8; các biến thể khác static 21,857–23,055 | CÓ HẠI | – | twin res_vrf sớm 1.29k giữ gather của c1 thêm +1.25k; mọi pass thêm trước gather cũng rơi vào đường của c1 (SPMD twin) | ao18_report |
| P1 / P1' / P2 / P3 / P4 (g1a, g2a; cloud R2 → ao18) | bất đối xứng hàng mịn theo cluster (`weight.view().pad::<m![H # 3968, Qs]>()`, tile trên `m![H # 3968 % 124]`) | **visa: "tile index with non-empty packing is not supported"** | KHÔNG BIỂU DIỄN ĐƯỢC | – | index tile hàng phải là một trục view thuần | ao18_report |
| L1 (ao18) | ws/nw giữa tile a và b | mục đích không còn; SIM ~0 | ĐÓNG BẰNG PHÂN TÍCH | – | – | ao18_report |
| §4.1 dr41 (cloud R3 → ao19) | inv_sw = begin_interleaved(norm_weight, residual) → unzip → MulF(&ws_vrf, 0.0) → zip AddF → DivF Mode10 1.0; res_c = residual.to_dm(tdma) nuôi res_vrf | static 22,504: POST3 trước WAIT2 OK, nhưng nw và inv_sw pair ra SAU WAIT2 (ước +1.0k c0 / +0.9k c1 so với tiết kiệm DR ≤ 0.45k) | ĐÓNG BẰNG PHÂN TÍCH (static gate KILL) | – | chính sách ALAP cho load nhỏ; c0 cần 4 lệnh nối tiếp trước inv_sw | ao19_report |
| lf1 / lf2 / lf3 (loop trip-count-1; cloud R3 Q1 → ao19) | pass final / load residual / load ws trong vòng `for _ in 0..1` rolled làm hàng rào thứ tự list | lf1 static 23,667 (DR ra sau final); **lf2 static 22,394 (−475), loop bị kéo lên head, DramReuse BỊ BỎ; lab1 101887 +2,085 (pSE 76) 0/8**; trace +2,618; ao20 103229 ~+2.0k/block; lf3 22,636 → đuôi lật | CÓ HẠI | – | loop hạ xuống thành block TUC `for`, kèm drain toàn bộ DMA tag (wait_dma 0..31) ở trước và sau, cộng code CPU; đặt theo phụ thuộc dữ liệu; tốn ~+2.3k ở head trên cả 2 cluster | ao19_report |
| đuôi sạch không DR + không stretch (lf2_t, anchor) | – | c0 POST4 − WAIT2 pass 2,636 (aoYG 2,771), c1 1,927 (aoYG 2,456); y StoVrf 457; END − T0 c0 7,291 so với 7,653 (−0.36k), c1 7,084 so với 7,628 (−0.54k) | phát hiện | – | mô hình ao_r3 −0.49 khá đúng; chỉ thiếu một "phương tiện" rẻ hơn ~0.4k | ao19_report |
| H2 không có epilogue trên c1 (AO_R3) | – | không construct nào giữ được pass ngoài cluster 1 (không có scalar cluster-id, không có nhánh theo cluster; Device chỉ có main, sub, tdma, pdma) | KHÔNG BIỂU DIỄN ĐƯỢC | – | – | AO_R3_external.md |
| PR-S1 ao_dra1 / ao_dra2 / ao_dra5 (BACKWARD_AO) | bỏ DramReuse bằng view residual alias (`unsafe reshape` sang `m![H]` / `m![AoQ, AoR]`; store output qua HbmTensorViewMut đã reshape) | dra1 / dra2 static giống hệt 22,869 (vẫn còn DramReuse); dra5 "mir: expected a HbmTensorViewMut" | KHÔNG BIỂU DIỄN ĐƯỢC | – | hazard gắn với tensor HBM, không phải kiểu view | BACKWARD_AO.md |
| L2 lật dead zone cuối kernel (fcf18 → ao20) | cho c1 POST POST4 sớm ≥ 0.6k trước c0 | ao20 lab0 102800 / lab1 103229: ở INFO, **c1 kết thúc trước c0 0.82–1.05k trong 36/36 launch** → c0 luôn qua barrier cuối nhanh | ĐÓNG BẰNG PHÂN TÍCH (artefact trace) | – | dead zone ở cuối kernel AO chỉ có ở trace; rút ngắn phía c1 sau WAIT2 ~0 giá trị ở INFO | ao20_report |
| 64 × 60 hàng, chia 33/31 (fcf18 L1 → ao20) | chia c0 nặng hơn một cách mịn | không khả thi (33 nhóm × 8 K-chunk = 264 > 256 slice/cluster); sai dấu ở INFO | ĐÓNG BẰNG PHÂN TÍCH | – | đường của c0 mới quyết định | ao20_report |
| E204 skew kéo sang launch sau (V10) | một launch skew cao có làm launch sau bị c1 giới hạn? | INTER 5/21 so với 6/21; P2 4/15 so với 3/15; v9 2/12 so với 2/12 | ĐÓNG | – | không có tín hiệu | V10-E204.md |

#### 3.7 Nhóm G — Đo đạc, biến thiên, phân tích AO

| tên/id | nội dung | kết quả | kết luận | nguồn |
|---|---|---|---|---|
| A13 nghiên cứu biến thiên (0.6) | 5 lần chạy v6 giống hệt | 54,654 / 72,962 / 63,946 / 54,694 / 56,318; HW/static 1.93–2.93 qua 9 lần | AO CV 13–14 %; giả thuyết "độ trễ HBM" bị thay bằng "sync host đầu tiên" (0.6) | MEASUREMENT_VARIANCE |
| A28 AO trong cây r21n | AO v11 không đổi | 23,428; 49,614 (77046), 54,344 (77120); chính thức 53,068 / 56,736 / 45,292 / 58,138 | dải 45–58k là nhiễu sync (0.6) | LED |
| các lần AO "đóng băng" | stall hiếm trên toàn c0 | P2: 48,145 (1 trong 56 launch warm; c0 Sub resVRF 11.7k); V18 AO info 51,876 (1 trong 12; ~1/36 qua các job V18-AO); 0 trong 80 launch aoYG | rủi ro (P ~3 %/lần chấm, tốn ~−3.4 % điểm lần đó); nhiễu nền tảng | V10-E205.md, FINAL_SUBMISSION.md |
| E224 trace so với info (AO, lab 85694) | cùng arm ở 2 mức | trace − info warm: V +925 (SE 142), E[max2] +757; AB +780; A +330; cold +2.3..2.6k | xếp hạng arm giống nhau ở cả 2 mức | V10-E224.md |
| mô hình barrier v9 AO (CRITICAL_PATH_AO) | reported = max(c0arr + max(0, L − s), c1arr_abs + L) | tái lập 43,271 / 39,197 / 39,686 so với 43,274 / 39,200 / 39,690 | phát hiện (L ~0.92k v9; 1.03–1.11k V18) | CRITICAL_PATH_AO.md |
| giải phẫu kết thúc AO (ao_gap, 16 trace warm job 91244) | phân rã thời điểm kết thúc | END ≈ s 1.50k + c1_T0 27.43k + contrB 1.43k + gather 1.26k + syncRT 1.10k + chuỗi sau sync 2.40k + L 1.05k = 36.16k (đo 36.22k); c1 quyết định 15/16; skew trả ~1:1 | phân tích (mô hình trace, sau bị hiệu chỉnh ở INFO) | analysis/v14/ao_gap.md |
| giải phẫu đuôi ao15 (tail15.py, lab2 aoyg_t) | chuỗi găng ở đuôi, 24 launch warm | chuỗi c1 T0 → contrB 1.48 → gather 1.26 → POST2 → +1.1 → y_vrf 0.76 → POST3 (DR) → c0 WAIT3 +1.34 → store 0.87 → POST4 → +0.7; tb_eff 5.46–6.23k KHÔNG tương quan với độ sớm của dispatch | phân tích: bỏ DR chỉ đáng ~0.3k; ~0.9k phần dư engine-0 mỗi lệnh không giấu được bằng dispatch sớm | ao15_progress; ARCH_AO_B §2 |
| thước đo & giải phẫu (BACKWARD_AO) | metric = max(Task_c0, Task_c1) = trên thực tế cửa sổ của c0 (trace 52/60; INFO c1 lớn nhất chỉ 3/120) | lab trace 36.08, lab INFO ~35.02, NH ~35.43–35.50 → chính thức ≈ lab INFO + 0.4, trace ≈ lab INFO + 1.05 | phát hiện | BACKWARD_AO §1 |
| ao15 lab4 (fe3_t + aoyg_t + tb16c20_i) | chỉ để xem cơ chế | không chạy được lên Arena (WSL crash 2 lần, C: còn 0.22 GB) | không cần cho kết luận | ao15_progress 21:03 |
| K12 overhead trace so với info (ao_gap) | cùng job 91244 | +1,081 (aoYG) / +843 (v111); chính thức ≈ trung bình lab-info − 0.39k | phát hiện | ao_gap §0-1 |
| K13 dị thường wsVRF | HW 1.10–1.17k so với static 293 (3.8×), trong khi resVRF cùng shape 0.37k | về sau triage quy cho stretch của config Sub; ws_vrf 0 byte của c1 cũng 1.13k | phát hiện | ao_gap §2.5; triage |
| ngân sách khoảng cách tới #1 (triage §6) | trước T0: lane idle 0.28 + x trên engine 0 1.10; phần dư mỗi lệnh tile a 0.97, tile b 1.08; stream thuần 24.17; sau T0: contraction b 1.42, chuỗi operand c0 4.43–4.47, c0 sau WAIT2 2.88, wait cuối 0.6–0.75 | #1 thấp hơn ta 3.1–3.5k; mọi lãng phí gọi tên được chỉ ~1.1–1.3k → leader có gì đó khác (giả thuyết H1–H4) | phân tích | AO_REDESIGN_triage §6 |

---


## 4. Kernel FFN (`decoder_feedforward`)

### 4.0 Kiến trúc hiện tại (nd11gz, V17.3; static 102,031; NH ~199.3–199.5k)

**Dataflow.**
- Input RMSNorm → Up (15,360 × 3,840) và Gate, cả hai là trọng số NVFP4, kèm block scale E4M3 và global scale. Up và Gate mỗi ma trận 14.75 MB/cluster.
- GeGLU → Down (3,840 × 15,360; W11 10.8 MB + W4 3.93 MB mỗi cluster).
- Post-FFN RMSNorm, layer scalar 0.375, cộng residual.

**Up/Gate.**
- FP4 → FP8 lookup được fuse vào contraction (v5). Activation là FP8 hi/lo (v11, `AddF(−hi)` vì `SubF` tính `vrf − stream`).
- Mỗi ma trận là một tile 30 hàng/slice (v11).
- Đầu kernel:
  - x và rms_w nhân bản 4× + `Broadcast01{8,8,30}`, thay CustomBroadcast{256} và config DMA (E101 x4);
  - global scale nạp vào 16 slice mỗi cluster rồi fan-out bằng Main `Broadcast1{4,4}` (F-4a);
  - block scale Up/Gate áp dụng trong contraction engine ở bf16 (varA).
  - Ba thay đổi trên gộp lại thành **fxv** (V13).
- Seed block-scale của Up là octet 240 hàng (piece 256 B, 57.6 KB/leader), relayout bằng 2 InterTranspose trên Main (octk, V17.0).
- Scale của Gate cũng qua oct seed (nd11g).
- Việc chuyển đổi block scale của Gate/Up chạy trên Main (mce).

**Down: CC-1 (V15 cc1c).**
- Down chia theo L giữa 2 cluster, nên h (đầu ra GeGLU) ở lại cluster của nó.
- h được gather trên Main bằng ring-64 `Broadcast1{64,4}`, rồi Main `to_trf` (mtrf, V15.1).
- Chia hàng 11+4 (k11).
- Cả 15 hàng block scale của Down nằm trong MỘT DMA 1.84 MB (k15, V15.2).

**Không còn pass decode thuần (nd11g/nd11gz, V17.3).**
- Down piece A (11 hàng) là một fused lookup, ghi f32 block sums mà không qua VE, sau đó 3 pass VE áp scale. Số học giống hệt octk.
- Piece B (4 hàng) là fused4.
- Vì không còn decode pass, compiler **đưa mọi bảng lookup và seed lên trước trọng số của nó** (ND regime).
- Seed Up được issue đầu tiên và gate chuỗi RMSNorm bằng exact +0 (`x² + (+0) == x²`). Nhờ đó Up W bắt đầu lúc 11.53k thay 13.74k.

**Epilogue.**
- c1 đẩy partial f32 sang c0 (one-way AllGather `to_dm` với `Cluster = m![1 # 2]`).
- Sau đó: add, ms, final, store.
- Chuỗi đuôi sau pass Down cuối ~8.2k: regroup 1.19 + push 1.27 + cluster sync 2.44 + add 0.71 + ms 0.98 + final 0.75 + store 0.85 (nd11gzt).

**Nút thắt còn lại** (ffn20, không đòn bẩy nào > 0.6k dưới gate bit-identical):
- Main bị giới hạn 0.6k trước fused4.
- Ba pass VE scale 2.2–2.8k, VRF load nối tiếp; pass đầu bị StoTab của tabB giữ thêm 0.8k.
- Đuôi 8.2k.
- StoTab của bảng fused chặn các pass VE đồng thời (luật StoTab, §6).

**So với leader** (190.0k INFO, BACKWARD_FFN): leader nhanh hơn ~9.3k.
- Giả thuyết chính: Down chia theo H (không có penalty L-split), block scale Down áp bằng CE với **bf16 block sums**, đuôi ngắn hơn ~3k.
- bf16 block sums làm run-2 max|d| = 0.01172, vượt gate nội bộ 0.00781 nhưng vẫn qua tolerance của cuộc thi (~2× dư). Vì vậy hướng R5 đang chờ **user quyết định** (§8).

### Bảng thí nghiệm FFN
- Mã F01–F62 là thời 0.6 (LED §5).
- FA1–FA31 là 0.8.1 v7–v9 (`experiments/p081/ffn_down_notes.md`, `ffn_scale_notes.md`, `ffn8_notes.md`, `ffn9_notes.md`).
- Từ V10 trở đi: `V10-E1xx`, ffn10/11/13/14/15/18/19/20 progress/report, cùng `analysis/*`.

#### 4.1 Nhóm A — Kiến trúc tổng thể: decode FP4, GeGLU, cách chia Down giữa 2 cluster, pass decode

| tên/id | thay đổi (cơ chế) | kết quả | kết luận | được / mất | lý do, bài học | nguồn |
|---|---|---|---|---|---|---|
| F01 FFN-001 | P1-A: Up/Gate decode theo từng pass; Down decode packed-to-columns ngay tại đích | baseline static 1,693,200; HW 1,266,679 (31778) = 2.92× | CÓ LỢI | 3.70M → 1.27M | chuỗi lookup→cast hợp lệ (Deep_knowledge P1-A) | TSV:3 |
| F02 CDX-000 | partial dot theo block-16, scale trong Vector sau contraction | 218,563; 490,316 (33638) | CÓ LỢI | – | – | TSV:4 |
| F33 **FFN-020** (REV-AF) | GeGLU cục bộ trong slice: Up/Gate giữ kết quả f32 phân tán (30 hàng/slice, 256 slice, 2 cluster); một lần regroup DM→DM trong cluster về 120 phần tử/slice (64 slice sống), làm GeGLU tại đó, rồi 1 gather + 1 store HBM | 144,061; 329,214 (35641) = 11.25× | CÓ LỢI | bỏ 2 chuỗi (gather → store HBM → load → cast), Switch ghép cặp, 2 pass global-scale; DMA 135.7k → 129.7k | chọn 120 vì 8 f32 = một flit 32 B và 120 bf16 = 240 B; 30 bf16 = 60 B không chia hết 8 B | TSV:50, OL§2 |
| F35 FFN-023 | Down FP8×FP8: một lookup f4→f8, hai contraction thuần (hi, lo), cộng ở mức block; không reduce `Dummy2` | 147,036 (+2,975) | CÓ HẠI (sau bị v11 thay) | pass không lookup 3,868 so với 9,263 (2.4×, không phải 8×) | băng thông OUTPUT của block partial là nút thắt (240 block × 30 hàng = 4,320 f32/slice) | TSV:61, AUDIT §6 |
| F36 chunk K của Down rộng hơn | 2 × 7,680 thay 4 × 3,840 piece | chỉ phân tích | ĐÓNG BẰNG PHÂN TÍCH | trọng số ~+3k, scale ~+1.7k tốt hơn | nhưng nhân bản activation tăng gấp đôi 5,367 → ~10,700 | OL§6, AUDIT §8 |
| F46 bàn giao H3 | `DownCluster` chia theo L để h không bao giờ qua cluster khác (dự ~−6,000 static) | chưa build (thời 0.6) | ĐÓNG (bằng bằng chứng ngoài; sau này được hiện thực thành CC-1 trên 0.8.1) | – | cần DM→DM vừa nhân bản vừa chia; tác giả v11: Down chia theo L +5,938 | DIRECTION §3, RESULTS_v11 |
| F51 **v11 FFN** (chọn 09-21) | FP4 giải mã một lần thành FP8, stream với activation FP8 hi/lo (cho cả Up và Down); Up và Gate mỗi ma trận một tile 30 hàng; scale Up/Gate qua Switch + gather theo cặp; Down 24+6; pass scale 6 hàng; `AddF(−hi)` | 118,593 (−21.8k); 258,329 (71345), 257,629 (71414) (−22 % HW) | **CÓ LỢI** | – | thay F35 và F44: rút ngắn Main mở khoá được phần tiết kiệm hàng đợi mà FFN-032 đã có | RESULTS_v11, main mlp.rs |
| F56 Down K-split giữa 2 cluster (0.6) | bỏ hop HBM của x | 140,703 (+22k) | CÓ HẠI | – | seeding x bằng DM→DM trong cluster tốn 29k static | dma-docs |
| P2 port v2 | FFN Down 1×30; gather y bằng DM→DM đổi Cluster | FFN 126,140 static, HW 268,060 (77526) | CÓ LỢI | FFN −42k so với v1 | – | RES |
| P3 port v3 | hop GeGLU qua all-gather + replicate DM→DM | FFN 134,577 (+8.4k); không chạy | ĐÓNG BẰNG PHÂN TÍCH | – | replicate DM→DM tới 16 nhóm hàng × 2 cluster = 15,592 static so với đường HBM 611 + 1,978 | RES |
| P7 **port v5** | fuse lookup FP4→FP8 của Up/Gate vào pass contraction (api081 P1; hi/lo làm stream-adapter broadcast) | FFN 117,994 (Main 79.3k → 64.3k); HW 245,848 (77953) | **CÓ LỢI** | FFN −15.1k so với v4a | từ đây FFN bị DMA giới hạn (93 % DMA busy) | analysis-work/api081 |
| F4 sở hữu K-split GeGLU/Down (V10 probe #3) | mỗi cluster tiêu thụ nửa GeGLU của mình, trao đổi y | cần all-gather ring-256 (~7.7k static, ~15–17k HW trên Main trong lúc decode12) | ĐÓNG BẰNG PHÂN TÍCH (trần ≤ 0) | – | (sau này CC-1 làm được bằng {4,1} hai tầng + ring-64: số bước ring ít hơn ~12×) | CRITICAL_PATH_FFN.md F4 |
| ffn15 S1 switch gate (09-25 16:43) | chạy validator host cho các switch của CC-1 | control OK (gather_up_gate B1{4,1}, F-4a B1{4,4}, XG4 B1{32,4}); CC-1 OK: regroup B1{4,1} → [L/120%64, Dq4], fan-out gs B1{16,1}, gather B1{64,4}, gather nửa, regroup kết quả B1{16,1}; 5 ca âm tính FAIL đúng như yêu cầu | GATE PASS | – | mọi switch mới đều phải qua validator có ca âm tính trước khi lên Arena | ffn15_progress 16:43 |
| cc1 = CC-1 thuần (ffn15, mlp_cc1.rs 93d3fb9a) | Down chia theo L giữa 2 cluster (DownCluster = L/7680), h ở trong cluster, ring-64 gather B1{64,4} trên Main → DM [Gs, L%7680] → Sub to_trf; regroup kết quả B1{16,1} + transpose, d2d cả hai partial về c0, pass cộng | static 105,112 (fxv 108,286: −3,174); emulator 0/3,840; lab A 94088 **+1,339 ± 252 (0/6)**; trace 94091: gather 7,040 HW, StoTrf của x 4,173 vượt qua lúc W12 land và CHẶN issue in-order (decode12 +2.7k sau W12, lane idle 3.0k); tab3 StoTab (7.7k HW) chặn g0 | CÓ HẠI (như đã build) | – | StoTrf chặn + StoTab/g0 stall | ffn15_progress 16:45-17:12 |
| cc1s (32b3eb01) | cc1 + tách gather: hi B1{64,4} trên Main, lo trên Sub, vào các tile Gs của một DM tensor | static 105,193 (gather lo trên Sub 4,615, tuần tự sau Main qua WAW tile / PeCore); lab A +456 ± 331 | TRUNG TÍNH / tệ hơn | – | gather trên Sub chậm + tile WAW làm tuần tự | ffn15_progress |
| cc1b = CC-1b (97bf855b) | fxv + reload hẹp (DownQuarterRows 256 slice đích, 2 × 1,920 B mỗi slice) + Broadcast01{4,1,60} CỐ ĐỊNH Sub → TRF; không có config DMA CustomBroadcast (26 DMA) | static 108,139; lab C 94195 **+1,968 ± 172 (0/6)**; trace 94201 +1,450; reload 256 slice 7,612 (so với 5,838) | CÓ HẠI | – | luật "số slice đích" không đúng với piece 2 × 1,920 B; nạp TRF bằng ring-4 Broadcast01 là StoTrf CHẶN, còn ring CustomBroadcast{4} của fxv là TuExec không chặn | ffn15_progress |
| CC-2 như viết nguyên văn | B1 cố định cho seed ring-4 có pad | không build được: Broadcast1/01/TB1/IT dời cả trục slice có pad vào Time (chỉ CustomBroadcast trích được một phần) | KHÔNG BIỂU DIỄN ĐƯỢC | – | CC-1b CHÍNH LÀ dạng build được của CC-2 | ffn15_progress |
| cc1n (6ed64987) | cc1 với switch Transpose{1,1} ring-1 trong lần nạp TRF của x | static = cc1; lab C +1,394 ± 122; trace +1,851; vẫn là StoTrf (4,175) | CÓ HẠI | – | switch Transpose{1,1} KHÔNG biến Sub to_trf thành TuExec | ffn15_progress |
| **cc1c (b567fa93) → V15** | cc1 + nạp TRF của x bằng CustomBroadcast{ring_size: 1} | static 105,185 (compiler thêm config DMA 719 sau W12); emulator 0/3,840; lab E 94268 −1,121 ± 232 (6/6); lab F 94333 −1,522 ± 248 (8/8); **pooled −1.31k ± 0.17k (z −7.7)** | **CÓ LỢI** (V15) | được: bỏ x hop 9.2k lane; mất: penalty đọc nửa hàng ~+2.8k, cần exchange + sync | chỉ Sub to_trf nuôi bởi CustomBroadcast mới chạy như TuExec KHÔNG chặn (8,731 HW, song song với decode12) | ffn15_progress; V15_RESULT |
| FFN-A (thiết kế ngoài) trên k15 | cluster về sớm làm finalizer duy nhất, đẩy partial f32 | ĐÃ CÓ SẴN trong cấu trúc k15 từ cc1c (part.to_dm vào Cluster = m![1 # 2] = AllGather một chiều N2) | TRUNG TÍNH (đã làm) | – | – | ARCH_FFN_B §7 |
| FFN-B chia L cân skew | – | không biểu diễn được nếu không viết lại mọi mapping Up/Gate/GeGLU/regroup thành 31 hàng/slice; EV 0.6–1.0k ở P ~0.3 | ĐÓNG BẰNG PHÂN TÍCH | – | – | ARCH_FFN_B §7 |
| FFN-D | một pass final fuse chính xác | final của k15 vốn đã là một pass; pass đơn 3 lần làm tròn là bất khả (VE làm tròn một lần ở output) | ĐÓNG BẰNG PHÂN TÍCH | – | – | ARCH_FFN_B §7 |
| dcl / dcln / dce11 / dce11n / dcm / dcm15 / dcf11 / dcf12 / dcf13 (ffn18 họ D-CE) | block sums + scale bằng CE; giữ hoặc fuse decode; sums f32/bf16 | static 103,871–105,957; dcf13 đặt 4 load epilogue trước W2 (~+5.6k HW); dcl VE cast rơi vào cửa sổ StoTab | ĐÓNG BẰNG PHÂN TÍCH / tạm dừng | – | static StoTab 1.3k so với HW 5.4–8.2k: static không định giá được hiệu ứng bảng của D-CE | ffn18_report |
| odcf12 / odcf11 / odcf13 (ffn18) | octk + D-CE | lab C 100856 (control octk 204,603): odcf12 **+3,230 ± 562 (0/6)**; odcf11 +206 ± 188; trace odcf12t +4,758; numerics r2 max abs d **0.01172 > 0.00781** gate | **CÓ HẠI (họ D-CE KILLED)** | – | head seed-trước tạo khe trong lane; StoTab chồng lên fused12 (19.8k so với dự 16.5k); bf16 block sums trượt gate số học nội bộ | ffn18_report |
| odcf12g / odcf12gs | scale Gate qua oct seed trên odcf12 | static 101,946 | không còn ý nghĩa | – | xây trên odcf12 | ffn18_report |
| D-CE 12/3, D-DC 11/4 (cloud FFN_GAP) | block scale bằng CE với bf16 block sums | cloud dự −4.3k / −2.9k | ĐÓNG (chết dưới gate của ta) | – | bf16 block sums r2 max abs d 0.01172 | FFN_GAP_external.md |
| **nd11g (72a3a2b6)** (BACKWARD_FFN R2 → ffn20) | không còn decode (ND): Down piece A = fused lookup ghi f32 block sums, không qua VE + 3 pass VE scale; scale Gate qua oct seed OCT-b; compiler đưa mọi bảng/seed lên trước trọng số | static 102,008; so với octk: A 102602 −3,672 ± 244; B 102735 −4,114 ± 433; C 102775 −2,850 ± 202; D 102814 −4,096 ± 232; **pool −3,520 ± 124**; port NH 102760 FFN 202,139 | CÓ LỢI (KEEP-READY, bị nd11gz thay) | lane sớm hơn (Gate seed aligned, không chờ bảng Gate), chuỗi Main cân với lane sau W11 (27.6k so với 26.9k); mất: khe ở head +8.1k khi Up W bắt đầu 13.74k | closure D-CE không áp vì ND giữ số học f32 | ffn20_report |
| **nd11gz (54d1252b) → V17.3** | nd11g + sửa khe head R3: DMA seed Up issue đầu tiên trong feedforward, exact 0.0 từ seed đã land (view reshape, `vector_logic(BitAnd, 0.0)`, reduce trong slice, Sub to_vrf = công thức zero_from_q_contraction bên QKV) cộng vào pass mean-square của RMSNorm; normalize_input trả tuple (x, bản rms-w) | static 102,031; so với nd11g: C −2,033 ± 286 (−7.1 SE), D −1,833 ± 220; so với octk: C −4,883 ± 348, D −5,929 ± 260, **IV pool −5,553 (SE 208, z −26.7)**, E[max2] −5,265; NH 102873 FFN 199,370 | **CÓ LỢI** (V17.3) | được ~−5.5k FFN (≈ +0.9 % điểm); Up W bắt đầu 11.53k so với 13.74k; cold run0 −8.2..−9.4k; mất: không (bit-identical) | static chỉ −1,291 nhưng HW −5.6k | ffn20_report; V173_RESULT |
| nd11e9 (9bf498dc) (BACKWARD R1) | load scale Down 4 hàng muộn s4 sau W4 (dự −6.0k) | A 102602 +399 ± 422 (2/6); trace +1,509 | CÓ HẠI | – | s4 issue 4.0k sau khi W4 land (in-order sau EXEC của fusedB); thêm một pass 2.76k | ffn20_report |
| nd11 (3f9f6b3a) (R4) | ND không có Gate seed | static 103,334; không lab | ĐÓNG BẰNG PHÂN TÍCH | – | bị nd11g trội | ffn20_report |
| nd11gzs / nd11e9s | zero pass / pass scale trên Sub | LIR ICE sfr_impl.rs:868 | KHÔNG BIỂU DIỄN ĐƯỢC | – | – | ffn20; BACKWARD_FFN |
| R5 Down kiểu leader (BACKWARD_FFN) | ND chia H + block scale Down bằng CE với bf16 block sums + trao đổi h trong cluster | mô hình −9..−14k, P 0.15 | CHỜ USER QUYẾT ĐỊNH | – | trượt gate max abs d nội bộ nhưng qua tolerance của cuộc thi (~2× dư) | BACKWARD_FFN §7 |
| X1–X14 không gian thiết kế (BACKWARD_FFN) | ND, Gate seed, scale muộn, split 12/3 trong ND (X4 +2.5k), tách Gate W (X5), 240×32, Down chia H (X7), CE bf16 (X8), CE bf16 hi/lo 2 số hạng (X9 ~7.5k so với 8.1k, không lợi), cluster bất đối xứng c0 = Up, c1 = Gate (X10 ~0), lệch pha cluster (X11 ≤ 1.7k), ít bảng hơn (X12 bất khả), x/rms sau Up W (X13), push partial sớm (X14) | phân tích | ĐÓNG BẰNG PHÂN TÍCH (chỉ họ ND ≥ 3k dưới gate) | – | – | BACKWARD_FFN §4 |

#### 4.2 Nhóm B — Chia tile trọng số và pipeline (0.6 → 0.8.1)

| tên/id | thay đổi (cơ chế) | kết quả | kết luận | được / mất | lý do, bài học | nguồn |
|---|---|---|---|---|---|---|
| F03 FFN-002 | prefetch DMA trọng số + scale của Gate/Down theo thứ tự chương trình | 220,006 | TRUNG TÍNH | – | load trọng số vẫn chỉ bắt đầu khi Main của ma trận trước xong; DMA busy 146,497 không đổi; giả thuyết WAR page 16 MiB | TSV:5 |
| F04 FFN-003a | Up/Gate 5 tile × 12 hàng, prefetch 1 tile | 223,200 | CẦN ĐIỀU TRA | – | tile sau chờ lookup của tile trước (WAR theo page); +1.7k DMA/tile để nạp lại bảng; lần đầu `Option<DmTensor>` MIR ICE | TSV:8 |
| F05 FFN-003b | load scale sau tile 0 trong thứ tự chương trình | 223,200 giống hệt | NGUYÊN NHÂN GỐC | – | allocator cho mọi tile CÙNG địa chỉ (9,437,184) → DMA(t+1) chờ lookup(t) | TSV:9 |
| F06 FFN-003c | 2 buffer `to_dm_view` ping-pong, 5 tile/ma trận | 204,018 (−6.7 %); 511,151 (34133) FAIL max 0.72, 14.6 % trong tol | CÓ HẠI (SAI) | – | lần nạp lại `to_dm_view` không được sắp thứ tự theo hazard (3 phiên bản cùng địa chỉ bị đọc ngược) | TSV:10,12 |
| F07 FFN-003d (REV-B) | 5 tile tensor mới | 197,096; 470,944 (34181) | CÓ LỢI | – | tensor riêng → địa chỉ riêng → pipeline đúng | TSV:36 |
| F08 FFN-004 (REV-C) | Down thành 5 tile mới 12 hàng | 192,368; 481,405 (34230) | TRUNG TÍNH / CÓ HẠI | – | DMA 90 % busy (bảng 30 × 838 = 25k, scale 29k, tile 95k); không phải thắng trên HW so với REV-B | TSV:13, LED X26 |
| F09 FFN-006a | hàng liên tục 30 `m![L/30 % 256]` đủ K, DMA broadcast thuần cho x (18.4k), một load 28 MiB | 221,563 | CẦN ĐIỀU TRA | DMA trọng số 24.6k @0.88 (trước 29.4k @0.74), scale 4.9k @0.56 | x broadcast 18.4k @0.16; load Gate 28 MiB không được kéo lên | TSV:15 |
| F10 FFN-006b | x qua `Broadcast1{slice1:256}` + 2 load nửa trọng số | 267,059 | CÓ HẠI | – | Broadcast1 1→256 = 62k Main (bằng ring); nửa 14 MiB util 0.72, không được kéo lên | TSV:16 |
| F11 FFN-007 (REV-D) | 30 hàng + trao đổi cặp x `Broadcast1{slice1:2}` + 2 tile nửa | 196,651; 445,483 (34314) FAIL 15.4 % | CÓ HẠI (SAI) | – | Broadcast1 vào output `Dummy2` chỉ lấp slot 0; verifier KHÔNG kiểm luật trục-trong-cùng cho Broadcast1 (có kiểm cho CustomBroadcast) | TSV:14 |
| F12 FFN-008 (REV-E) | biến thể lấp tile, cùng exchange Broadcast1 | 481,048 (34339) FAIL 15.42 % | CÓ HẠI (SAI) | – | cùng lỗi | LED |
| F13 FFN-009 (REV-F) | 30 hàng + trao đổi ring-2 + lấp `to_dm_view` + 1 lookup + 1 contraction | 221,682; 491,223 (34360) PASS | CÓ HẠI (hiệu năng) | – | các lần lấp đều issue trước (trước cả load RMSNorm) → Main idle 89k; exchange x 504 cycle thì OK | TSV:17 |
| F14 FFN-010 (REV-G) | tile 6 hàng mới + lookup từng tile vào FP8 dùng chung + 1 contraction/ma trận | 240,261; 528,688 (34388) | CÓ HẠI (hiệu năng) | – | scheduler nạp + decode hết 15 tile (Down trước) rồi mới contraction | TSV:18 |
| F15 FFN-011 (REV-H) | lookup + contraction + pass theo từng tile, nạp từng tile một | 254,163; 607,856 (34401) | CÓ HẠI | – | chỉ một tile đang bay → cùng địa chỉ → 12.5k/tile nối tiếp | TSV:19 |
| F16 FFN-011b (REV-I) | Up/Gate 30 hàng liên tục + x ring-2 + 5 tile mới 6 hàng/ma trận issue hết từ đầu; Down 5 × 12 | 173,035 (−21 % so với codex); 436,262 (34417) | CÓ LỢI | DMA 89.5 % busy | PeCore 85k (VRF load Sub từng tile + pass xen kẽ) | TSV:20,21 |
| F17 FFN-011c (REV-J) | 5 pass vector sau khi xong mọi tile | 185,771; 454,400 (34458) | CÓ HẠI | – | static +12.7k (pass bị tuần tự) | TSV:22 |
| F18 FFN-012 (REV-K) | 3 × 10 hàng/ma trận (19 bảng), pass scale 6+4 hàng, Down liên tục `[H/30 % 64, L/3840]` | 162,524; 390,859 (35294) | CÓ LỢI | ít tile = ít bảng 838; scale DMA 9.8k → 6.6k | – | TSV:23 |
| F19 FFN-012b (REV-L) | tile 6+12+12 / 12+12+6 + fuse epilogue Down (gs, post-norm, residual, layer gate trên 8 slice × 480 kênh) | 153,680 (−8,844); 393,122 (35343) | TRUNG TÍNH (giữ) | – | DMA 140.2k / 54 op không đổi → HW bám theo hàng đợi DMA, không theo đuôi Main ẩn | TSV:24 |
| F20 FFN-013 (REV-R) | Gate 18+12, Down 18+12 (48 DMA op, 15 bảng) | 152,264; 364,036 (35464) | CÓ LỢI | bảng 31 → 19 → 15 | tile Down 18 hàng không được kéo lên trước GeGLU (load 115k–127k); tile 12 hàng thì có | TSV:30 |
| F21 FFN-013b | Down 12+12+6 | 152,234 (−30) | TRUNG TÍNH | – | – | TSV:31 |
| F23 FFN-015a | Gate là một load 30 hàng 29.5 MiB | 174,126 (+22k) | CÓ HẠI | – | load lớn đơn lẻ issue đầu tiên (3.6k–28.2k) trước các tile Up | TSV:34 |
| F34 FFN-021 | Down 18+6+6 để rút đuôi Main | 144,262 (+200) | TRUNG TÍNH | – | 2 bảng thêm (1.7k) triệt tiêu lợi ở đuôi | TSV:51 |
| F37 FFN-024 | Down 12+12+6 để tile cuối Main 11,326 → 5,926 | 144,493 (+432) | CÓ HẠI | đuôi −1,794 idle nhưng hàng đợi +2,226 | `down_weight_scale` di chuyển lên trước tile cuối mới → Luật L6: "DMA còn lại sau byte trọng số cuối" không phải hằng số dự trù được | TSV:71 |
| F39 FFN-026 | Up 6+12+12 → 18+12 (−1 op trọng số, −2 bảng) | 141,097 (−1,932); dmaBusy −2,225 (luật dự −2,226); 336,589 (39569) | CÓ LỢI (tranh cãi) | – | V8 revert sau 1 lần chạy (337,157); tác giả v11 báo "+10,760"; chưa từng A/B cùng job; không còn ý nghĩa với FFN v11 | TSV:75, LED X13 |
| F40 FFN-027 | issue 2 tile Down liền nhau theo thứ tự chương trình | 141,097 byte-identical | TRUNG TÍNH | – | thứ tự chương trình không có tác dụng (lần 4) | TSV:76 |
| F41 FFN-028 / FFN-029 | Down 20+10 / 16+14 | 141,927 (+830) / 142,608 (+1,511) | CÓ HẠI | – | 18+12 là tối ưu cục bộ thật | TSV:77,78 |
| F44 FFN-032 | một tile 30 hàng/ma trận: 12 bảng → 6, 8 op → 5 | 142,436 (+2,046); dmaBusy thấp nhất 119,628 nhưng idle +8,723 | CÓ HẠI (sau bị v11 thay) | – | Main không bắt đầu được trước khi cả load 24.5k land (khi đó Main bị lookup giới hạn) | TSV:83 |
| F45 FFN-033 | chỉ Up một tile 30 hàng | 144,517 (+4,127) | CÓ HẠI | – | tile đầu nhỏ của Up cho Main bắt đầu sớm ~9.5k | TSV:84 |
| F54 Down K-halves trên 0.6 (71629) | 15 hàng/slice, piece 3,840 B aligned | 118,593 → 115,198 static; +1.2 % cùng job | CÓ HẠI trên 0.6 (THẮNG trên 0.8.1 thành v7) | DMA −12k thật | x mỗi slice gấp đôi: seed + ring + TRF +6.6k trên đường găng sau sync | dma-docs/FINDINGS §4 |
| P1 port v1 | 0.8.1 từ chối tile hàng không chia hết 30 → Down 3×10 | FFN static 118,593 → 144,689; HW ~310k | CÓ HẠI (sửa ở v2) | – | tile Down 3×10 mới bị KÉO lên đầu DMA (tile mới nhỏ hơn mà consumer lookup không có input nào khác); một tile 30 hàng thì không | RES; MEM |
| P5 port v4a | Down 24+6 (+ scale 8,8,8,6) qua `unsafe reshape` sang trục riêng | FFN 120,903; HW 260,993 (77915) | CÓ LỢI | FFN −2.6k so với v3e (dự −7k) | – | analysis-work/toolchain081/ffn24 |
| FA4 **KH** (79025) | trọng số Down theo K-halves: layout [g32 = H/60 % 32, k2 = L/7680, q4 = H % 4], 15 hàng/slice × piece 3,840 B aligned, tile 12+3 qua reshape sang trục riêng (KhG/KhR/KhQ), nhóm 4 hàng (VRF 4×480 f32 = 7.68 KB), kết quả [value, pad], reduce = một pass Main TransposedBroadcast1{2,4} + inter-slice reduce + transpose cặp (445 static); switch kiểm trên host (swcheck) | 112,757 (−5.2k); median 231,144 (−14.2k so với v6, −10.1k so với X1s) | **CÓ LỢI** (v7) | W12 38.5k ở 613 B/c (trước W24 479 B/c); W3 540 B/c (trước 343) | trên 0.8.1, đường x gấp đôi được che dưới decode12 (trên 0.6 bị loại, F54) | ffn_down_notes.md |
| FA5 **E3** (KH_F3, 79061) | tile cuối 3 hàng: lookup FP4→FP8 fuse vào pass nhóm (không decode3, không group3) | 112,462 (−295); median 227,964 (−3.2k) | **CÓ LỢI** (v7) | fused3 5.2k (trước decode3 4.5k + g3 5.2k) | – | ffn_down_notes |
| FA23 ffn8 t13 (LAB6) | tile Down W13 decode + W2 fused cuối | 222,702 so với 216,563 | CÓ HẠI | – | StoTab của bảng fused2 list sau nhóm 1 hàng chạy 6k; load residual sau ClusterSync epilogue → DramReuse 3.8k | ffn8_notes |
| FA29 ffn9 static fa / fb / g2r | fused3 trước; (g0, fused3, g1, g2); sáu nhóm 2 hàng | 116,526 / 112,564 / 111,053 | ĐÓNG BẰNG PHÂN TÍCH | – | – | ffn9_notes |
| F-1a t114 (ffnA13) | Down cắt 11/4 (trên fxv; thiết kế cũ) | 91461 info, 80/80 PASS: +6,508 ± 206 | CÓ HẠI | – | 12/3 là tối ưu cục bộ (với body khi đó) | ledger V13 |
| F-1b t105 (ffnA13) | Down cắt 10/5 (ý tưởng từ repo công khai: ABBA 0.6 ×14 ~1.65k tốt hơn 12/3) | 91461: +17,403 | CÓ HẠI | – | thiết kế của ta khác | ledger V13 |
| F-1c (ffnA13) | Down cắt 13/2 | 91684: +5,222 (32 warm) | CÓ HẠI | – | 12/3 tối ưu cả hai phía (ở thời điểm đó) | ledger V13 |
| k11 (49e80d5c) (ffn15) | chia hàng Down 11+4 thay 12+3 (trọng số Down đầu land sớm hơn một hàng ~3.3k; decode11 ngắn hơn 1.35k; nhóm fused 4 hàng vừa VRF 8 KiB) | static 105,737; lab G 95840 −919 ± 407 (4/5) | TRUNG TÍNH đứng riêng (thành phần của k11mmce8) | – | cùng một thay đổi (11/4) mà ở fxv lại +6.5k: kết quả phụ thuộc thân kernel | ARCH_FFN_B §5 |
| k11p1 / k11p2 | hoán vị thứ tự chương trình các load Down | static giống k11 | TRUNG TÍNH | – | thứ tự chương trình không điều khiển list DMA | ARCH_FFN_B §5 |
| X4 split 12/3 trong ND (BACKWARD) | – | +2.5k | ĐÓNG BẰNG PHÂN TÍCH | – | 11/4 vẫn tối ưu | BACKWARD_FFN §4 |



#### 4.3 Nhóm C — Layout trọng số trên HBM và tốc độ stream

| tên/id | thay đổi | kết quả | kết luận | được / mất | lý do, bài học | nguồn |
|---|---|---|---|---|---|---|
| F52 layout bench1 (71568, chỉ-DMA) | probe | Up 28 MiB liên tục mỗi slice 59.5/62.0k so với strided 74.5/73.5k (−18 %); Down 3,840 B aligned (d_c2) 59.9/55.6k so với 1,920 B lệch nửa (d_c4) 77.5/72.5k (−23 %); seed scale 57.6 KiB aligned 18.8k so với 14.4 KiB 23.2k | phát hiện | – | trên HW, căn 256 B + run dài mỗi slice mới quan trọng; lệnh nhỏ thì không | dma-docs/FINDINGS |
| F53 probe "HBM-local" (71655) | layout xen kẽ (wps/wps2) | tệ hơn liên tục 4–11 % | CÓ HẠI | – | – | dma-docs |
| FA19 ffn8 r32 (LAB3 79494) | Up/Gate trên 240 slice sống × 32 hàng (piece scale 7,680 B aligned; reshape trục riêng UrC/UrG/UrQ/UrR) | 113,558; 230,936 (+12.6k) | CÓ HẠI | – | layout 240/256 slice nạp chậm ~10 % (DMN mất cân bằng); piece aligned 7,680 B chỉ đạt 434 B/c | ffn8_notes |
| F1 layout load/compute FP4 của Down (V10 probe #1) | phân bố lại FP8 đã decode trước khi commit | W12 đã 622–631 B/c với packet aligned 256 B; W3 nằm trên đường B (trần 0.4–0.7k) | ĐÓNG BẰNG PHÂN TÍCH (trần ≤ 0.5k) | – | – | CRITICAL_PATH_FFN.md F1 |
| F12 H-ROW (thứ tự trọng số so với plateau) | thứ tự packet kề nhau | chưa đo ở V10/V11 | (V12 covM trả lời: không có đòn bẩy) | – | – | CRITICAL_PATH_FFN.md F12 |
| covM mô hình coverage FFN + probe (job 90990) | mô hình phủ kênh pha ngẫu nhiên eff_x17; tìm block 30 hàng Up/Gate (252 lựa chọn bit: eff 0.47–0.58, hiện tại 0.558, tốt nhất 0.577 = +0.3 %) | chỉ-DMA B/t/cl: Up.U0 310.4 (chip 618.8), Up.best 309.1, Up.worst 308.7 (mô hình dự −2.5 %), Up.32 (2 lệnh) 246.5; scale: Sc.dir800 195.7, Sc.quad960 201.5, Sc.pair960 194.2, **Sc.oct256 262.9 (chip 447.5)**, Sc.32 187.3 | phát hiện: không đòn bẩy layout trọng số Up/Gate/W12 > ~0.5 %; layout scale căn 256 B stream nhanh hơn ~34 % | – | mô hình coverage sai cho Up (worst = best); căn chỉnh chứ không phải coverage quyết định tốc độ load scale | covM_notes.md |
| layout trọng số FFN (V14 ffn_gap) | – | trong kernel = chỉ-DMA (618.6 so với 618.8); covM ≤ 0.5 % | ĐÓNG BẰNG PHÂN TÍCH | – | – | ffn_gap §6 |
| ph / pl probe (ffn15 lab B 94091) | tập DMA Down chia H so với chia L, không tính toán | static 35,686 / 35,926; HW ph 67,262 / pl 69,161: pl − ph +1,899 ± 164 (4/4) | phát hiện | – | (P1 là số chuẩn: +2.8/+3.1k lane) | ffn15_progress |
| **P1 penalty nửa hàng của CC-1** (J1 94294, dn_l − dn_h) | W12 + W3 Down đọc theo nửa L mỗi cluster (mapping CC-1) so với chia H | lane +2,784 ± 129 (c0), +3,053 ± 98 (c1); info Task +2,418 / +2,368; W12 309 → 295 B/tick/cl (−4.6 %), W3 +0.81/+1.03k; số request mỗi engine như nhau | phát hiện: VỪA PHẢI (2.0k < penalty < 4.5k) → CC-1 ròng −3.8..−5.3k | – | chi phí nằm ở pattern truy cập HBM (hai cluster đọc xen kẽ hai nửa của cùng các hàng 7,680 B); khớp note SOTA 0.6.0 (+3.3k trên 29.5 MB) | analysis/v15/PROBES.md P1 |
| P1 dn_l_sw (đổi chỗ 2 nửa) | – | không biểu diễn được (cluster_tile làm đích DMA chưa lower; đại số mapping không có phép đảo) | KHÔNG BIỂU DIỄN ĐƯỢC | – | độ đối xứng đã được bao bởi dn_l c0 so với c1 | PROBES P1.1 |
| P1 sở hữu scale Down (ds_l − ds_h) | s12 + s3 dưới mỗi kiểu sở hữu | −33 ± 133 / +16 ± 73 | TRUNG TÍNH (không có penalty) | – | – | PROBES P1.2 |
| dcon / dpc / dh16 (cloud FFN_GAP → ffn19, lab 101064) | đổi sở hữu hàng Down (h = 15s + r; thân thiện regroup; kiểm chứng mô hình dự đoán tệ nhất) để bỏ aliasing hash kênh HBM XOR | dcon c0 W11 −558 ± 82 (12/12), c1 −170 ± 78; dpc c1 +346 ± 70; dh16 = hiện tại | TRUNG TÍNH / ĐÓNG | – | gate (≥ 300–305 B/tick/cl trên CẢ HAI cluster) fail ở mọi arm; mô hình XOR của cloud sai (dh16 bằng hiện tại); penalty L-split ~5 % đến từ chính việc xen nửa hàng, không từ hoán vị slice→hàng | ffn19_report |
| mlp_odpc (ffn19 S2/S3/S4) | – | chưa từng compile | tạm dừng | – | S1 fail | ffn19_report |
| layout Up/Gate 240×32 (cloud / BACKWARD / ffn18 C7 r32b) | scale aligned mà không cần relayout | phân tích: capped ~0.1–0.5k cho một lần viết lại toàn bộ mapping | ĐÓNG BẰNG PHÂN TÍCH | – | cần gather TRF có pad (fetch_mask là todo!()); slack P1 | ffn18_report; FFN_GAP_external |
| K16 tốc độ trọng số FFN (knowledge) | – | Up W trong kernel 618.6 so với chỉ-DMA 618.8; Gate 621, W12 625, W3 571 B/tick; luật thời gian trọng số FFN bytes/647 + 1.7k/lệnh | phát hiện: trọng số đã ở tốc độ chỉ-DMA | – | – | covM 90990; ffn_gap §0-2 |

#### 4.4 Nhóm D — Block scale (Up/Gate/Down): pass, DMA, alignment, engine rebalancing

| tên/id | thay đổi (cơ chế) | kết quả | kết luận | được / mất | lý do, bài học | nguồn |
|---|---|---|---|---|---|---|
| F38 **FFN-025** | pass block-scale 6 → 8 hàng (8+8+8+6) cho Up, Gate, Down; VRF 8×240×4 = 7,680 B vừa 8 KiB | 143,029 (−1,032), DMA byte-identical, nSub 42 → 39, nMain 44 → 41; cặp khớp −2,316 HW (39037/39534), dự −2,398 | CÓ LỢI | bớt một pass Main + một VRF load Sub mỗi ma trận | 8 hàng là tối đa vì operand VRF phải là f32 (`VeScalar` = i32/f32); cặp đo này BÁC BỎ event model (dự −9,044) | TSV:72, DMA_COST_LAW |
| F55 pass scale 8 hàng trên v11 (71688) | 4 pass thay 5 | 115,915; +0.5 % | CÓ HẠI (trên 0.6 v11) | – | trên HW thời gian pass tăng theo số hàng, không có chi phí cố định để tiết kiệm (trái với lợi static của FFN-025) | dma-docs |
| F57 seed scale aligned 57.6 KiB + InterTranspose ring-8 | – | 121,871 (+3.3k) | CÓ HẠI | – | InterTranspose ring-8 tốn 14.7k trên Main, 68.6k trên Sub | dma-docs |
| F58 gộp scale Down 4 → 1 | – | 121,694 (+3.1k), DMA −1.6k | CÓ HẠI | – | phá pipeline Main; tác giả v11 đo +10,956 HW | dma-docs, RESULTS_v11 |
| F60 F2a / F2b (0.6) | F2a: scale block Down thành một load 7,200 B liên tục/slice (so với 4 load 1,920 B lệch ~105 B/cycle); F2b: scale Up/Gate nạp thẳng từng slice | đã compile trong lab-dma, chưa đo (harness CF) | (mở; trên 0.8.1 "scale Up/Gate 7,200 B/slice nạp thẳng" được giữ ở v7) | – | – | qkv-lab/ffn_f2a_ab.log |
| FA13 ffn-scale LAB1 (q4 / f2b / s14) | q4: seed Up theo quad 120 hàng (28.8 KB/leader, nửa aligned 256 B, InterTranspose ring-4); f2b: seed thẳng 7,200 B/slice không InterTranspose; s14: hàng scale Down 16..29 thành một load | lab v0 247,236; q4 246,964 (−0.3k); f2b 248,833 (+1.6k); s14 247,638 (+0.4k) | q4 giữ về sau; f2b, s14 CÓ HẠI | – | lab launch luôn cold (+6.6k so với harness thường): chỉ dùng để so tương đối; f2b: to_vrf scale đầu chỉ phụ thuộc seed DMA → eager → list trước trọng số | ffn_scale_notes.md |
| FA14 ffn-scale LAB2..LAB4 (d0..d10) | d1 = d0 + E2v; d2 = + F2bm (seed thẳng, scale VRF từ Main qua VE với 1/16 gộp, không InterTranspose); d3 = + q4up; **d5 = d3 + u8 (pass block-scale Gate 8,8,8,6 hàng)**; d7 gate-gs store trở lại Sub; d9 gộp GeGLU (k1 = ggs/√2 trong gelu của Sub); d10 = d5 không E2v | d1 228,773 (+0.6k); d2 227,948; d3 227,191 (−1.0k); **d5 226,273 (−3.4k so với d0 229,656)**; d7 = d3; d9 không lợi; d10 227,199 | d5 CÓ LỢI; d9 TRUNG TÍNH | – | Main to_vrf thẳng từ collect làm treo; gộp scale Down s14 +0.4k | ffn_scale_notes |
| FA15 **ffn-scale d5 final** (79198, 79203) | ffn-down h5 + E2v + F2bm + q4up + u8 | 112,660; 224,530 / 225,152 (−0.6..−1.2k so với h5) | **CÓ LỢI** (mlp.rs của v7) | bit-identical với h5 theo cấu trúc (×1.0, gộp chính xác 2^-4) | tốc độ seed: cặp 14.4 KB / thẳng 7.2 KB ~360 B/c; quad 28.8 KB ~390 B/c; chỉ piece 240 hàng (57.6 KB) là aligned 256 B; phân phối ring-8 ~15k+ HW trên Main | ffn_scale_notes |
| FA17 ffn8 **s12** (LAB2 79477) | scale block Down hàng 4..11 trong MỘT DMA 8 hàng | 110,959; 219,781 (−4.3k so với g4) | **CÓ LỢI** (v8) | s0 nằm trong cửa sổ ClusterSync của x hop; W3 issue lúc g1 dispatch | – | ffn8_notes |
| FA21 ffn8 s123 (LAB4 79564) | hàng 12..14 gộp vào load 8 hàng (DMA 11 hàng) | 110,410; 219,191 (+1.2k) | CÓ HẠI | – | vị trí issue in-order của W3 quan trọng hơn số DMA scale; tile 7 hàng của KhR = 15 ở 0 bị từ chối ("PROG-719: 15 has no divisor in 7..14"), tile 8 hàng ở 0 và 7 thì compile được | ffn8_notes |
| FA22 ffn8 s12e16 / sR / sRe16 (LAB5) | sR: tile fused = hàng 0..2 để scale của nó dùng chung DMA với g0 | s12e16 217,356; sR 219,562; sRe16 218,252 | s12e16 CÓ LỢI; sR CÓ HẠI | – | DMA lớn trong cửa sổ sync không miễn phí (x reload dùng chung băng thông 9.4k so với 5.1k) | ffn8_notes |
| FA26 ffn9 pdma relayout scale Down (static) | HBM→HBM `DmaDtod` để có run scale liên tục mỗi slice | 243,710 static | ĐÓNG BẰNG PHÂN TÍCH | – | lower lên cùng tài nguyên DmaEngine + thêm ExplicitSync | ffn9_notes |
| FA28 ffn9 LAB2 s15 / st2n / s15st2 / s15st2y4 (80291) | cả 15 hàng scale Down trong MỘT DMA (trên v8) | 110,840; +4.7k; st2n +0.1k; s15st2 +3.1k; + y4 +4.0k | CÓ HẠI (ở v8/v9) | – | DMA scale gộp 10.5k (1.84 MB/cluster ở ~175 B/c: piece 480 B quyết định rate, không phải số lệnh); W3 cấp phát ở địa chỉ 0 page 0 đè lên bảng của decode12. (Cùng ý đó sau này THẮNG ở k15 trên thân CC-1: phụ thuộc thân kernel) | ffn9_notes |
| FA30 ffn9 LAB3 mv (80342) | VRF scale nhóm Down ghi từ Main (gộp 1/256) | 112,598; +6.3k | CÓ HẠI | – | 4 pass VRF trên Main 1.32k mỗi pass làm dài đuôi nối tiếp; W3 land 209.7k | ffn9_notes |
| F2 / P1, P1b: load scale Down aligned + InterTranspose ring-8 (V10) | nguyên nhóm 4 hàng (3,840 B liên tục, aligned 256 B) rồi hoán đổi slice/time | P1 static 110,077 (−812), mô hình −0.28k nhưng ước tay +1.3..+2.6k (StoTab tranh chấp); P1b +135 static, mô hình +1.7k | ĐÓNG BẰNG PHÂN TÍCH | – | tiết kiệm lane rơi vào đường B (trần 0.4–0.7k); pass ring-8 rơi vào đường A | CRITICAL_PATH_FFN.md F2 |
| F3 căn chỉnh block-scale Up/Gate (V10) | nhóm 16 hàng aligned | không có lời giải switch luỹ thừa 2 cho 30 hàng/slice; F57 ring-8 14.4k static Main; r32 +12.6k HW | ĐÓNG BẰNG PHÂN TÍCH (cấu trúc) | – | – | CRITICAL_PATH_FFN.md F3 |
| E102 seed scale Up list ngay sau Up W (pin exact-zero, V10) | – | thiết kế: bảng Up sẽ xếp sau seed (+9–10k), ~5k lane idle trước Gate W; dự +3..+4k | ĐÓNG BẰNG PHÂN TÍCH | – | seed vốn đã lấp cửa sổ table → StoTab → contraction; cái "mất" 4.0k của nó là do hình dạng packet (960 B lệch nửa, 369 B/c), không phải độ trễ issue | V10-E102.md |
| FFN varB (X3, V12) | block scale Gate qua seed4 + InterTranspose ring-4 như Up | static −607; Arena info 91210 / 91211: FFN 222,461 / 222,309 (so với V11.1 ~214–217k) | CÓ HẠI (+6–7k) | – | – | ledger dòng 60; v12self/varb_*.log |
| FFN varA trên V12 (X3 / cơ chế "F-2" của ffn7 công khai) | block scale áp bởi contraction engine (scale trong TRF, không có pass VE scale) | Main static −6.4k; info 91297 / 91299: FFN warm mean 214,932 so với 214,958; sai số 0.00391 / 0.00781 / 0.00781 (nhỏ hơn gate) | TRUNG TÍNH đứng riêng (thành phần của fxv) | – | FFN bị lane DMA giới hạn: Main −6.4k static không tới được HW khi đứng riêng | ledger dòng 62; v12x3/varA_*.log |
| varA đứng riêng (V13) | block scale Up/Gate trong CE ở bf16 | −270 ± 239 pooled | TRUNG TÍNH | – | – | ledger V13 |
| varAB (ffnB13) | biến thể varA (không có số trong nguồn; SUY LUẬN: áp cho nhiều tập scale hơn) | CÓ HẠI (không có số) | CÓ HẠI | – | V14 do-not-do: "varA cho Down: fused3 bị lookup giới hạn" | V13_RESULT; V14_PLAN §4 |
| FFN-OCT (ffn14, V14-E801) | seed scale Up theo octet 240 hàng (57.6 KB/leader, mỗi slice thứ 8, packet 256 B; chỉ-DMA 7.0k so với seed quad 9.15k) + InterTranspose{2,4,60} ring-2 stride-4 trên Main để dựng lại layout q4 | swcheck PASS; static: seed packet 256 B, nhưng stage 1 = 14,663 static; slack Gate chain 4,887 → 850; ước HW +3k | ĐÓNG BẰNG PHÂN TÍCH (0 Arena) | – | luật chi phí switch: ring-2 stride-4 tốn bằng ring-8 (cost = flits × slice1 × slice0 + 263) | ledger V14 |
| **FFN-OCT-b** (ffn14, V14-E803) | oct seed + relayout ring tối thiểu (A ring-4 stride-1 1,800 flit + B ring-2 stride-4 trên tile 450 flit), thay q4 | mọi gate static + emulator bitwise PASS; lab 93569 +98 ± 249, 4/10; trace: seed 7.41k so với 9.28k, Gate W land −2.14k, nhưng chuỗi Up (+14.7k Main HW) kết thúc 2.5k sau Gate W và DMA tabGate issue ở đuôi của chuỗi đó → 2.5k lane idle | TRUNG TÍNH (trên fxv; sau là CÓ LỢI trên k15 = octk) | – | ngân sách đăng ký trước dùng sai ràng buộc (StoTab start); ràng buộc thật: chuỗi Up kết thúc ≤ Gate W land − ~0.3k | ledger V14; ffn14_progress |
| FFN-OCT-c + fxvp (ffn14) | OCT-b + pin tile-tensor (cặp gs + cặp kết quả CE Up/Gate) để list tabGate trước đuôi Up | lab 93611: octc +976 ± 194, fxvp (chỉ pin) +845 ± 171 (0/8 mỗi cái); trace: DMA to_dm_view-vào-tile thêm lệnh PeCore giữ issue in-order tới khi các DMA tile xong → W12 issue muộn 5.9k | CÓ HẠI (đóng họ OCT ở V14) | – | LUẬT: không đặt DMA có đích là tile / pin commit tile gần DMA trọng số | ledger V14 |
| chi tiết swcheck OCT (ffn14) | validator switch trên host | ctl q4 IT{4,1,15} OK; OCT s1 IT{2,4,60} OK; các ca âm tính FAIL đúng như yêu cầu | GATE PASS | – | – | ffn14_progress 14:02 |
| review P2 OCT-UP (ffn_review) | oct seed + stage 1 ring-2 | covM oct 7,062/6,969 so với quad 9,558/8,744; tiết kiệm lane ~1.9–2.1k; Main idle trước Gate contraction trung bình 21,500 | SỐNG (P ~0.4) → sau bị static kill (stage 1 = 14,663) | – | mô hình chi phí của reviewer sai với stride-4 | ffn_review §2 |
| seed oct cho scale Gate (ffn_gap; critic kiểm lại) | – | giấy: ròng +2.2k (2 stage Sub ~2 × 12.8k vượt slack 3.46k) / +3.6k trước Gate W; trên Main ~2 × 7.5k = 15k HW nhưng chỉ còn 12.1k Main idle | ĐÓNG BẰNG PHÂN TÍCH | – | (sau này ở ND, Gate oct seed chạy được vì compiler đưa seed lên trước Gate W) | ffn_gap §6; critic §5 |
| seed oct cho scale Down | scale Down đúng hình dạng oct | giấy: ròng +3.8k hoặc tệ hơn | ĐÓNG BẰNG PHÂN TÍCH | – | không vị trí đặt nào có lợi | ffn_gap §6 |
| C6 đóng gói packet scale / CJ-6 (ffn18) | đọc scale theo run 256 B aligned không cần relayout | run 7,200 B mỗi slice của Up/Gate bắt đầu ở 32·s mod 256; muốn đi aligned cần offset đầu mỗi slice −32·(s mod 8), không biểu diễn được với mapping affine đồng nhất; piece 32 B = 1 request mỗi piece | ĐÓNG BẰNG PHÂN TÍCH | – | – | ffn18_report; CONSOLIDATED CJ-6 |
| **octk (9c3e6e7c) → V17.0** | k15 + seed scale Up OCT-b (piece 256 B, descriptor pk = 256) + stage A IT{4,1,15} + stage B IT{2,4,15} trên Main | static 103,322 (k15 103,924); seed DMA 3,557 static (q4 4,159); chuỗi Up HW 110.37k so với Gate W kỳ vọng 111.2k: dư 0.83k PASS; emulator 0/3,840; labA 100838 −1,814 ± 218 (6/6); labB 100847 −2,044 ± 304 (8/8); **pooled −1,892 ± 177** | **CÓ LỢI** (V17.0) | được −1.9k; Gate W land −2.49k; W11 −1.57k; fused4 kết thúc −1.65k | k15 có thêm ~3k slack chuỗi Up so với fxv (nơi OCT-b là NULL): cùng một ý tưởng, thân kernel khác thì kết quả khác | ffn18_report; V17_RESULT |
| Sub cho stage 1 của OCT (V14) | – | Sub chậm hơn Main 4.7× (F57) | ĐÓNG BẰNG PHÂN TÍCH | – | – | V14_PLAN §4 |
| varA cho Down (critic V14) | – | pass fused3 sau W3 bị lookup giới hạn (5.17k); phần tiết kiệm phía Main của varA đứng riêng là NULL trên HW (91297/91299) | ĐÓNG BẰNG PHÂN TÍCH (không đề xuất) | – | – | critic §2 |
| tách W12 quanh x hop hoặc GateSc (ffn_gap) | – | giấy | ĐÓNG BẰNG PHÂN TÍCH | – | +1 lệnh F, decode12 bị trễ | ffn_gap §6 |
| octs / octss (ffn18) | stage A (/ + B) của relayout trên Sub | static 109,622 / 109,596 (stage A trên Sub 39,849 static: fetch Sub chỉ 8 B) | ĐÓNG BẰNG PHÂN TÍCH | – | – | ffn18_report |
| C5 cắt chuỗi Gate / C9 stall StoTab g0 (ffn18, trên octk) | – | slack P1 1.86k; relayout scale Gate (octet +22.3k HW Main cho −2.9k lane; q4 +7.5k cho −0.5k; pair ~+4k cho −0.5k); tab4 luôn list sau DMA scale của g-chain (ALAP ngược); tách cần thêm một lệnh scale ~1.0–1.2k cho ≤ 1.64k; nhóm VRF 2 hàng mất ~1.2k cố định mỗi g-pass | ĐÓNG BẰNG PHÂN TÍCH | – | – | ffn18_report |
| mce (5d53f71b) (ffn15) | chuyển đổi block-scale Gate/Up + staging TRF của CE sang MAIN (chuỗi Gate static −2.2k) | lab G −141 ± 118 | TRUNG TÍNH đứng riêng | – | đúng như mô hình: chỉ P1 được thêm slack | ARCH_FFN_B §5 |
| mtrf (295f6615) (ffn15) | Main to_trf của h (không CustomBroadcast{1}, không cfg1; 23 DMA) | static 105,112 (thứ tự sau W12: s0, s3, s12, tab4, W3 — tab4 MUỘN hơn); emulator bit-identical; lab G +190 ± 496 | TRUNG TÍNH đứng riêng | – | – | ARCH_FFN_B §5 |
| s4x (2cb35a3d) | DMA scale đúng 4 hàng cho mỗi nhóm (25 DMA) | static 105,686 (tab4 muộn +1.7k); lab G +2,049 ± 443 (0/5) | CÓ HẠI | – | DMA scale đúng 4 hàng tệ hơn tile | ARCH_FFN_B §5 |
| k11m / k11mce / k11mmce (ffn15) | k11 + mtrf / + mce / + cả hai | static 105,185 / 105,737 / 105,185; k11mmce median lab G 210,282 (một outlier 262,740) | TRUNG TÍNH | – | – | ARCH_FFN_B §5 |
| gcc / gcc2 / mg / k11mg | chuyển đổi block-scale Gate f8 → bf16 qua commit_cast (ra khỏi VE) | lowering FAIL: "CommitUnit verify failed: shape packing should be done with no conversion"; "No valid fetch size found: pv=4 f8->f32" | KHÔNG BIỂU DIỄN ĐƯỢC | – | – | ARCH_FFN_B §0.6, §5 |
| k11mmce7 (12f0c352) | cửa sổ scale 7 hàng (hàng 8–14) trong một DMA | FAIL "PROG-719: 15 has no divisor in 7..14 to grow 7 into" | KHÔNG BIỂU DIỄN ĐƯỢC | – | – | ARCH_FFN_B §5 |
| **k11mmce8 (e5d25ada) → V15.1** | k11 + mtrf + mce + scale hàng 7–14 trong MỘT DMA 8 hàng (hai cửa sổ 8 hàng chồng nhau; tile g2 offset 1, fused4 offset 4); 22 DMA | static 104,765; lab G 95840 (TIMEOUT 70 s sau 136/162 launch) −3,296 ± 294 (5/5); lab H 95904 −2,958 ± 132 (6/6, −22 SE); lab J 96112 −3,003 ± 102; k11mce8 (không mtrf) −419 ± 191 | **CÓ LỢI** (V15.1) | chuỗi hậu Gate 141.0 → 159.6k (−4.3k nhờ pass block-scale trên Main); W11 land 162.3k (W12 165.1k) | CẦN đủ tổ hợp (mtrf riêng +190, mce −141, k11 −919, k11mce8 −419); bài học "cân bằng engine, giữ lane kín" | ARCH_FFN_B §3-5; V151_RESULT |
| k8s12 (5ac8d763) | scale nhóm 12 hàng + scale fused4 đúng | static 104,767; thứ tự sau W11 s4(fused), s12, tab4, W4 → bảng vẫn sau MỌI DMA scale | ĐÓNG BẰNG PHÂN TÍCH (không lab) | – | compiler luôn list bảng fused sau DMA scale Down CUỐI | ffn15_progress 21:16 |
| **k15 (4a0a502a) → V15.2** | k11mmce8 + cả 15 hàng scale Down trong MỘT DMA 1.84 MB; 21 DMA; thứ tự sau W11: s15, tab4, W4 | static 103,924; lab I 96040 −1,305 ± 248 (6/6); lab J 96112 −739 ± 346; lab K 96138 −1,003 ± 67 (8/8, −15 SE); **IV −1,014 (SE 64)** | **CÓ LỢI** (V15.2) | bảng tab4 land sớm hơn 1.6k | bảng fused land sớm hơn khi cả tập scale là một lệnh (luật K101) | ffn15_progress; V152_RESULT |
| k15v2 (f500e242) | nhóm 2 hàng để double-buffer VRF | static 104,007; VRF load k+1 vẫn bắt đầu lúc contraction k KẾT THÚC (không double buffer) | ĐÓNG BẰNG PHÂN TÍCH | – | – | ffn15_progress 21:49 |
| cc1cm / cc1c15 | scale Down hàng 0–11 một DMA / cả 15 hàng một DMA (trên cc1c) | static 105,112 / 105,191; emulator khác 2,319 / 3,044 trên 3,840 | bỏ (emulator lỗi; về sau biết emulator không chính xác khi đổi tập DMA scale Down) | – | – | ffn15_progress |
| P4 probe: DMA scale chậm lại trong kernel (CONSOLIDATED) | Gate scale 10.37–10.58k trong kernel so với 9.38–9.46k chỉ-DMA; tranh chấp bank so với pin share | chưa chạy (cần build micro-kernel có contraction Main) | MỞ | ước −1.5k (P 0.2) | – | CONSOLIDATED §5; PROBES |
| PB6 VRF bf16 (BACKWARD_FFN) | VrfTensor<bf16> để giảm một nửa chuỗi VRF | VeScalar chỉ có i32/f32 (scalar.rs:38,52) | KHÔNG BIỂU DIỄN ĐƯỢC | – | – | BACKWARD_FFN |
| X3 scale phần cuối muộn / X8 CE bf16 / X9 CE hi/lo bf16 2 số hạng (BACKWARD) | – | X3 = nd11e9 (+0.4k); X8 trượt gate (0.01172); X9 ~7.5k so với 8.1k, không lợi | ĐÓNG | – | – | BACKWARD_FFN |

#### 4.5 Nhóm E — Đầu kernel: x / rms / bảng lookup / global scale / warm-up / stack fxv

| tên/id | thay đổi (cơ chế) | kết quả | kết luận | được / mất | lý do, bài học | nguồn |
|---|---|---|---|---|---|---|
| F42 FFN-030 | broadcast activation ring_size 2 → 4 (mỗi slice nạp H/4 = 1,920 B) | 140,672 (−425): DMA ACTIVATION 10,208 → 9,062 (−1,146, đúng luật), idle +721 | CÓ LỢI | – | `Dummy8 % 4` dùng được làm trục slice đích ring-4 | TSV:79 |
| F59 trace head FFN (71633, 71689) | trace từng lệnh của FFN v11 | DMA bảng 98–2,054; StoTab (TU đầu, cold) dispatch 5,620; load x 5,674; reduce 7,918–11,859 (3.9k cho 462 static); Up weight issue 11,870; sync y-hop 9,731 / 9,265 / 9,331 | phát hiện | – | có thể lấy lại ~6k HW nếu Up được list trước (sau sửa bằng "head fix" v7) | MEM:rngd-hw-execution-model |
| F61 reduce trên Sub | dời reduce của input RMSNorm sang Sub để Up issue sớm | trace | ĐÓNG BẰNG PHÂN TÍCH | – | StoTab (5.7k HW) là lệnh Sub đầu tiên và chặn các lệnh Sub sau; DMA bảng luôn list đầu | qkv-r12/RESULT |
| FA6 **H5** (79091, 79164) | to_vrf rms-weight trên Sub list sau DMA Up weight bằng một pass copy trên Main (StoTab đầu của bảng decode12 được kéo lên 5.4–5.7k cold chặn Sub in-order và cả issue của Up W) | 112,462; median 225,703 / 225,732 (−2.3k; −19.6k so với v6) | **CÓ LỢI** (v7) | Up W issue 4.05k (trước 7.8k) | cả hai chuỗi giờ cân bằng (TU tới GeGLU 50.5k so với DMA tới W12 51.4k) | ffn_down_notes |
| FA9 H6 (79145) | H5 + bản copy Main của global scale Up/Gate để Sub to_vrf eager list sau trọng số kế | 112,450; median 224,595 | TRUNG TÍNH (revert) | – | Gate W issue sớm 6k nhưng land cùng lúc: bị băng thông DMA giới hạn; gate bằng Sub to_vrf eager chỉ tốn khi engine đáng lẽ idle | ffn_down_notes |
| FA10 ffn-scale E12 (78998) | E1 (input-norm weight gộp thành một Sub VRF w/rms qua pass vector Sub kết thúc bằng to_vrf) + E2 (VRF global-scale Up/Gate qua Main `collect().to_vrf(&mut device.sub)`) | 117,982; FFN launch TREO ("launch ran past its time limit") | CÓ HẠI (TREO) | – | Main `to_vrf` thẳng từ `collect` TREO device — phải đi qua vector engine | ffn_scale_notes |
| FA11 ffn-scale E1 riêng (79018) | – | PASS; median 243,265 (−2.1k so với v6) | CÓ LỢI (thay thế cho H5) | Up W issue 3,703 (trước 7,763) | – | ffn_scale_notes |
| FA12 ffn-scale **E12v** (79043) | E1 + E2v: VRF gs từ Main qua vector engine (collect → vector_init → narrow_trim → MulF 1.0 → widen_pad → vector_final → to_vrf) | 240,617 (−4.7k so với v6) | CÓ LỢI (một phần) | – | trên base ffn-down, E2v trung tính (store Main nằm trên chuỗi TU của Gate); ~0.5k trên d5 | ffn_scale_notes |
| F6 head x8w (P3 / P3b, V10) | x/rms nhân bản 8× + Broadcast01{4,8,30} | P3 static 110,460 (−429) nhưng list lỗi: pass rms chạy trên bảng D12 thô (page 0) → WAR với StoTab cold 5.4k, mô hình +1.9k; P3b (sqrt thẳng vào Sub VRF) −0.94k / +1.9k | → E101 (dạng x4) | – | – | CRITICAL_PATH_FFN.md F6 |
| E101 **head x4** (V10) | x và rms_w nhân bản 4× (`UpPairRows = [L/1920%4, 1#8, H/480]`) + `Broadcast01{8,8,30}` có tên; bỏ CustomBroadcast ring-256 và config DMA snoop-bitmap 1.22k | static 110,232 (−657); lab1 82802 −237 (SEd 369); lab3 82963 −746 (447); pooled −443 ± 285 (1.6 SE); Up W bắt đầu c0 6,001 → 5,256 | TRUNG TÍNH đứng riêng (sau đó THẮNG trong stack fxv) | – | load nhân bản ăn ~0.5k trong 1.2k tiết kiệm từ cfg; các đường đuôi hoà nhau hấp thụ phần còn lại; ngưỡng kill Up-W 5.0k bị hiệu chuẩn sai | V10-E101.md, ffn10/src/mlp_x4.rs |
| E101 x8 fallback | Broadcast01{4,8,30} | static 110,460, mô hình +1.93k | ĐÓNG BẰNG PHÂN TÍCH | – | lỗi list tái hiện | V10-E101.md |
| E113 E101 x4 xếp trên XG2+SQF+C2 | – | 85212 all3 +92 (block-paired ± 121); Up W start −0.72k / −0.53k nhưng x reload +0.89k / +0.64k; pooled với các lab trước −178 ± 180 | TRUNG TÍNH | – | lợi ở head bị reload chậm triệt tiêu; manh mối: đích DM của reload dời 0x4080000 p1 → 0x14a0000 p2 (chưa kiểm) | V10-E113.md |
| V12o E101 | – | −178 ± 180 (trace), −443 ± 285 (1.6σ) | TRUNG TÍNH | – | Up W issue sớm hơn nhưng x reload chậm 0.64–0.89k | V12 TONG_HOP §2 |
| F8 load gs trên 1 slice + switch fan-out (V10) | up gs 2.7–3.3k, gate gs 1.5–1.8k | trần 0.5–1.5k; không build | ĐÓNG (qua E104) | – | – | CRITICAL_PATH_FFN.md F8 |
| E104 dị thường up gs (V10) | chẩn đoán static: load 512 B tốn 2.65–3.3k so với gate gs 1.5–1.8k (cùng descriptor dma_23) | tensor 131 @ 8,576 KB page 0; không nguyên nhân cấp phát / page / WAR / list | ĐÓNG | – | có lẽ do tranh chấp DM với Up contraction đang chạy (SUY LUẬN) | V10-E104.md |
| site memset-pin FFN (audit #2 table 485, #3 scale12, #4 scale3, #8 head x/rms_w, #9 Up scale sau StoTab, #14, #16) | – | fT2 (memset DownResult): memset land ở head, StoTab đầu 1841 → 2041 (trễ x), makespan +200; fH1 (memset 10 KB/slice, WAW reduce x): đặt sau StoTab đầu, Up issue ~7.4k (+2k lane idle) | ĐÓNG BẰNG PHÂN TÍCH / KHÔNG BIỂU DIỄN ĐƯỢC (lợi kỳ vọng 0 ở mọi site) | – | memset không input luôn ở slot Sub rảnh đầu tiên; pin giữa kernel bất khả; Sub ở head FFN thuộc về StoTab decode của Down | MEMSET_PIN_AUDIT.md, V12-E602.md |
| fH2 (kế hoạch pinFFN) | memset TRƯỚC StoTab đầu | chỉ kế hoạch; không có kết quả | KHÔNG RÕ (không ghi nhận) | – | – | pinFFN_notes.md |
| S7 bảng lookup D12 ở head (V10) | lệnh lane đầu do compiler sinh (1.96k) | không handle | ĐÓNG BẰNG PHÂN TÍCH (≤ 0.3k) | – | – | IDEAS_SCHED.md s6 |
| F-3a wua / F-3b wub (ffnA13) | pass warm-up Sub ở head (ngắn / dài) (ý tưởng repo công khai: Up start 9.2k → 6.3k trên 0.6); wum bị loại static | 91461: wua +470 ± 281, wub +254 ± 256 | TRUNG TÍNH | – | head fix H5 đã đưa Up W issue về ~4.05k trên 0.8.1 | ledger V13 |
| F-3c wuL (ffnA13) | StoTab sau DMA(Up W) | 91501: +397 (bỏ outlier) | TRUNG TÍNH | – | – | ledger V13 |
| F-3d (ffnA13) | – | không build được | KHÔNG BIỂU DIỄN ĐƯỢC | – | – | V13_RESULT |
| E101 x4 riêng (V13) | – | +96 ± 191 | TRUNG TÍNH đứng riêng | – | chỉ có lợi trong stack | ledger V13 |
| F-4a riêng (V13) | global scale Up/Gate nạp vào 16 slice mỗi cluster, fan-out bằng Main Broadcast1{4,4} | không giữ được khi riêng lẻ | TRUNG TÍNH | – | – | V13_RESULT |
| F-4b (ffnB13) | gs trên 1 slice + Broadcast1{64,4} | 91647: +355 | CÓ HẠI | – | – | ledger V13 |
| **fxv = x4 + varA + F-4a (V12-E766, ffnB13) → V13** | stack cả ba (chỉ mlp.rs) | lab B 91676 −2,640 ± 319 5/5; C 91707 −3,367 ± 437 5/5; **pooled −3,004 ± 282, 10/10**; dE[max2] −2,963; trace D 91722 −4,184 ± 166; bỏ bất kỳ phần nào trong stack tốn +1.1..+2.1k | **CÓ LỢI** (V13) | trace: DMA 28 → 27 (mất config DMA ring), TuExec 61 → 50, TU busy trước 170k 121.5k → 93.3k, Gate W issue sớm 14k, W12 land sớm 2.8/3.0k, end −4.2/−4.0k | ví dụ kinh điển của **luật stack**: từng phần NULL, gộp lại giải phóng tài nguyên khác nhau trên cùng lane | V13_RESULT; ledger V13 |
| H-LUT (X3 / ffnB13) | bảng lookup Down (16 KiB/slice) đang ở offset lẻ 256 → dời | V12: hạ ưu tiên sau khi qkvOKV NULL; V13: pad memset bị cấp phát SAU LUT (seed@0, LUT@256 cố định) | KHÔNG BIỂU DIỄN ĐƯỢC | – | – | COMMON_v12; ledger V13 |
| FFN-LUT (ffn14, V14-E805) | W12 + W3 trong một DM tensor đóng gói (probe khử trùng LUT) | static: 4 DMA bảng + 4 StoTab (= fxv), makespan +2,384 | ĐÓNG BẰNG PHÂN TÍCH | – | LUT tính theo từng pass lookup, không khử trùng theo tensor nguồn | ledger V14 |
| review P3 dùng chung LUT (ffn_review) | – | sách nói bảng f4e2m1 là hằng lúc compile, không staging, nhưng mọi lịch 0.8.1 có một DMA bảng + StoTab mỗi pass; T252 (fused3) cấp phát đè lên T90 (decode12) đã chết, cùng nội dung, và được dựng lại | ĐÓNG (sau probe static) | – | – | ffn_review §5 |
| F10 W3 page-0 / vị trí StoTab fused3 / số bảng (decode15) (V10) | – | WAR của W3 với bảng decode12 chỉ găng nếu W3 issue trước ~183k (thực tế 189.8–191.2k); không có API đặt vị trí; decode15 sẽ dời 10.2k của W3 lên trước x hop (+5.5k) | ĐÓNG BẰNG PHÂN TÍCH | – | – | CRITICAL_PATH_FFN.md F10 |
| head FFN trọng số trước (V14) | – | – | ĐÓNG | – | Sub ở head thuộc về StoTab decode12 do compiler sinh (fH1/fH2) — (ở ND thì compiler tự đưa bảng/seed lên trước, xem nd11gz) | V14_PLAN §4 |
| dời tabD12, s0 hoặc x/rms giữa head, E và x hop (ffn_gap) | – | giấy: ≤ 0.85k trên V12, trung tính trên V13 | ĐÓNG BẰNG PHÂN TÍCH | – | không có handle cho bảng của compiler | ffn_gap §6 |
| P4 lệch pha cluster cho scale Up (ffn_gap + review) | lệnh seed chỉ-c1 trước Up W, chỉ-c0 sau | đề xuất 0.3–1.8k; review: load nhỏ chồng lên hoạt động của cluster kia chậm hơn (c1 up gs +0.71k 6/6, …) | ĐÓNG BẰNG PHÂN TÍCH (KILL) | – | tiền đề 387 B/tick là số thời 0.6 | ffn_gap §4; ffn_review §4 |



#### 4.6 Nhóm F — x hop (GeGLU → Down) và nạp TRF cho x

Từ V15 (CC-1) trở đi, h không còn rời cluster nên x hop cũ biến mất. Các dòng dưới đây ghi lại các cách đã thử trước đó.

| tên/id | thay đổi (cơ chế) | kết quả | kết luận | được / mất | lý do, bài học | nguồn |
|---|---|---|---|---|---|---|
| F22 FFN-014 | x cho Down qua cặp + ring | verifier TỪ CHỐI | KHÔNG BIỂU DIỄN ĐƯỢC | – | "VRU reduce axes must be innermost" | OL§3 |
| FA2 X1 (78987) | broadcast ring-4 của x thẳng vào TRF trên Main (bỏ DM x + Sub StoTrf); slot ring `H/30 % 4` đổi tên thành `Dummy8/2` | 117,997; median 245,095 so với 245,345 | TRUNG TÍNH (giữ làm base) | – | ring→TRF trên Main 3.3k so với ring 2.6k + StoTrf 2.55k | ffn_down_notes |
| FA3 **X1s** (79000) | như trên nhưng pass ring-4 → TRF chạy trên **Sub** (song song với decode24 trên Main) | 117,997; median 241,196 (−4.1k) | **CÓ LỢI** (v7) | ring 5.0k trên Sub được che dưới decode24 | – | ffn_down_notes |
| FA8 XDM (static) | x hop của GeGLU bằng DM→DM (không store HBM / sync / reload) | 28,648 static cho phần replicate (makespan 136,204) | ĐÓNG BẰNG PHÂN TÍCH | – | – | ffn_down_notes |
| FA24 replicate x DM→DM trong cluster (static) | x đã gather (15 KB) vào 64 slice seed | 27,380 static (reload HBM 3,411) | ĐÓNG BẰNG PHÂN TÍCH | – | đường x không HBM / K-split không khả thi | ffn8_notes |
| FA27 ffn9 st2 / st2y4 / st4y4 (80046) | st2: x của GeGLU gather về 2 slice (piece 3,840 B); y4: y Down về 4 slice; st4: x về 4 slice | median c0 v8 217,069, st2 216,796, st2y4 215,807, st4y4 220,004 | CÓ HẠI / TRUNG TÍNH | – | st2: reload chậm hơn ăn hết lợi Stos + store; st4y4 +2.6–2.9k (cơ chế: guard MOV-6 queue-block của PE) | ffn9_notes |
| FA29 xd / xd8 (static) | hop x cross-cluster DM→DM; xd8 GeGLU trên 32 slice × 240, piece 240 B | 118,557 / 113,559 | ĐÓNG BẰNG PHÂN TÍCH | – | – | ffn9_notes |
| FA30 mvx / xd8 (lab3 80342) | – | +23k / +23k | CÓ HẠI | – | replicate DM→DM của xd8 tốn 30.9k HW (4× static) | ffn9_notes |
| F11 x hop trên 2 slice (st2, V10) | – | lab 80046 / 80291: st2 −0.3 / −1.0k, st2n +0.1 / +0.4k so với v8; thời gian W3 → tới sync(256) giống hệt | ĐÓNG (nhiễu) | – | phần tiết kiệm store bị hấp thụ ở hạ nguồn | CRITICAL_PATH_FFN.md F11 |
| E105 chẩn đoán endpoint 1 slice của x hop | – | không chạy dạng đó | bị thay thế | – | IDEAS_MOVEMENT s1.5: reload chậm không chỉ riêng st2 (e1 cũng 5.81/5.64k so với v8 5.16/5.19k) | IDEAS_MOVEMENT.md |
| MOV-1 xcopy (V10) | bản copy HBM riêng cho từng cluster `[Cpy2, Gs, L]` cho reload x nhân bản ×32 | static 110,445 (−444), reload 2,934 so với 3,411 static | chưa chạy ở V11 (GATE_FIRST) | – | judge: reload đã ~770 B/c ở mức chip | IDEAS_MOVEMENT.md MOV-1 |
| MOV-1 prpc so với prsh (ffnB13) | nguồn seed Down theo từng cluster | labP 91457: −185 ± 314, E[max2] −38; gate cần ≤ −0.6k | CÓ HẠI (FAIL gate) | – | – | ledger V13 |
| MOV-1 xcopy arm FFN (ffnB13) | – | 91647: +3,065 | CÓ HẠI | – | – | ledger V13 |
| MOV-2 st4a / st4b / st4c | store x từ cả 4 engine với run aligned 256 B | st4a static 111,202 (+313); st4c 110,769 (−120) nhưng reload lệch 960 B và block PE sau reload khác | ĐÓNG BẰNG PHÂN TÍCH (DROP) | – | cùng lớp hazard với st4y4 | IDEAS_MOVEMENT.md MOV-2 |
| MOV-3 X1 / X1c | store x thẳng từ 64 slice GeGLU (không Stos) | X1: packet store partial-unit 120 B tốn 4,456 static, makespan +3,267; X1c: store 2,879 nhưng reload packet 120 B 7,893, makespan 117,061 | ĐÓNG BẰNG PHÂN TÍCH | – | cách định giá static cho ghi partial (RMW) khớp HW; xác nhận Q02 thời 0.6 trên 0.8.1 | IDEAS_MOVEMENT.md MOV-3 |
| MOV-5 gather TU thay Stos 64→1 | fetch → switch → commit | run 120 phần tử buộc packet 8 B; ring-64 30 × 1 = 1,920 static ~3.8k HW > cửa sổ 2.1–3.0k | ĐÓNG BẰNG PHÂN TÍCH | – | (XG2 sau đó làm được với packet 24 B trên 2 leader) | IDEAS_MOVEMENT.md MOV-5 |
| **S1-XG / E112 XG2 (+SQF + C2) → V11.1** | x của GeGLU gather trên TU: Main fetch (packet 24 B) → `Broadcast1{32,4}` → commit lên 2 slice leader mỗi cluster (run aligned 3,840 B); bỏ DMA Stos 64→1 (2.18k), chia store x cho 2 engine | static 109,660 (−641 so với B; −1,229 so với V11); validator host OK; emulator bit-identical; C − B 85212 −766 (279); 85243 −722 (301); **pooled C − B −742 ± 222 (3.3 SE); C − A −922 ± 227 (4.1 SE)** | **CÓ LỢI** (V11.1) | store 2.64–2.74k → 1.55–1.57k; W3 land −1.25k (c0) / −0.42k (c1); gather tốn 3,643 HW (static 1,673, 2.2×) | chỉ ~0.75k của ~3.3k tiết kiệm lane tới được điểm kết thúc: gather làm decode12 trễ ~1–1.3k (proxy kill D > 0.8k đã "bắn" nhưng bị luật 2 job > 2 SE đè lại) | V10-E112.md, V10-E112-verify.md, ffn11/src/mlp_xg2sqf.rs |
| XG2 trên Sub | cùng gather nhưng chạy trên Sub | static 4,231 (truy cập Sub 8 B), ~8.5k HW | ĐÓNG BẰNG PHÂN TÍCH | – | không xong trước khi W12 land | IDEAS_SCHED.md s1 |
| E114 XG4 | `Broadcast1{16,4}`, 4 slice leader, piece 1,920 B | static 110,985 (+684): store bị định giá 1,870 (RMW trên piece 1,920 B); **MOV-6 FAIL** | ĐÓNG BẰNG PHÂN TÍCH | – | lớp hazard st4y4 (+2.6–2.9k HW); piece store lệch làm fail MOV-6 | V10-E114.md |
| E103p thay thế ring-4 của Down bằng dạng có pad | `Broadcast01{4,1,480}`, TRF [1#4, Gs, L%7680] | static 112,829 (+1,940): ring Sub 7,805 so với 2,186 static; mô hình +12.6k | ĐÓNG BẰNG PHÂN TÍCH | – | config có tên đều là all-gather (tốn theo khối lượng × nhóm) | V10-E103.md |
| E103q reload một phần tư + all-gather có tên | HBM x [L/1920, Gs, L%1920], mỗi slot ring reload phần tư của nó, `Broadcast01{4,1,120}` vào TRF | static 109,693 (−1,196): mất config DMA, reload 2,934 so với 3,411; nhưng W3 list sau Sub scale-VRF của g2 (race static 58 cycle) | ĐÓNG (hazard list) | – | chuỗi TU ngắn hơn làm lật race W3 / Sub-VRF | V10-E103.md |
| E103qm (steer: VRF scale g1 ghi từ pass VE ×1.0 trên Main) | – | static 109,693, mô hình −1.65k; lab2 82925 −417 (SEd 584); x4q −837 (515); HW: cfg biến mất (s0 land sớm 0.96k), reload một phần tư KHÔNG nhanh hơn (5.43k so với 5.19k), VRF g1 ghi từ Main 2.03k HW (Sub 1.10k) → g1 dispatch +0.93k; lab3 x4q so với x4 +815 | CÓ HẠI | – | phí steer ngang với phần tiết kiệm; trần khi steer miễn phí ~0.76k | V10-E103.md |
| E103qg (VRF g0 trên Main sau decode12) | – | static 110,378, list bị xáo | ĐÓNG BẰNG PHÂN TÍCH | – | VRF trên Main bị gate bởi lúc decode12 xong | V10-E103.md |
| E103 bỏ cfg4 (V12p, V14) | bỏ config DMA ring-4 | lab 82925 / 82963: thua | CÓ HẠI | – | hiệu ứng phụ lên list của W3 | V14_PLAN §4 |
| FFN-RLD (ffn14, V14-E804) | chẩn đoán x reload +0.49k của V13 (91722) | reload issue ở lúc sync(208) thả + 1 trên cả 2 cluster ở mọi arm; cửa sổ nằm trọn trong decode12; overlap TU giống nhau; địa chỉ DM giống hệt cho v12 và f4a (f4a vẫn +630); reload service v12 5,332 / f4a 5,962 / x4 5,950 / vara 5,741 / fxv 5,819 | ĐÓNG (không có handle trong source) | – | – | ledger V14; ffn14_progress |
| giả thuyết địa chỉ HBM của x reload V13 (reviewer → critic) | giải mã descriptor reload | reload 983,040 B/cl, packet 256 B, nhân bản, offset nguồn 0 lấy từ tham số runtime; vòng lặp giống hệt ở V12 / fxv / f4a / fx / fv; đích DM V12 132,096 / f4a 132,096 / fxv 16,640 | ĐÓNG (không rõ nguyên nhân) | – | – | critic §5 |
| pad địa chỉ reload | – | – | ĐÓNG | – | địa chỉ nguồn không tĩnh; f4a thụt lùi với đích của V12 | V14_PLAN §4 |
| reload nhân bản với nhiều / ít seed hơn (E103q quarter reload) | – | 5.43k, không nhanh hơn | CÓ HẠI | – | ít seed thì việc dồn lên ring của đường A | ffn_gap §6 |
| P1 CC-1b reload hẹp (J1 94294) | reload 1× 30 KB thay 32× nhân bản 983 KB | xr_1x64 1,325 / 1,323; xr_1x16 1,293 / 1,294; so với xr_rep −4,571 / −5,097; xr_norep (cùng 983 KB, dữ liệu khác nhau) 4,491 / 4,363 = đúng luật request | phát hiện: GO theo gate DMA; phần dư +1.5k của FREECHECKS chính là do NHÂN BẢN | – | (trong kernel CC-1b vẫn thua +1,968, xem 4.1) | PROBES P1 |

#### 4.7 Nhóm G — Epilogue và đuôi (RMSNorm sau FFN, residual, sync y, store)

| tên/id | thay đổi (cơ chế) | kết quả | kết luận | được / mất | lý do, bài học | nguồn |
|---|---|---|---|---|---|---|
| F43 FFN-031 | gộp sqrt rms vào reduce giữa các slice của `down_epilogue` | 140,390 (−282); 331,094 (39615) | CÓ LỢI | – | `Clip(Add,EPS)` → `AddF(EPS)` | TSV:80 |
| F47 epilogue E2 (41489) | xoá global scale 1/9600 (Up), 1/12928 (Down) vì nghĩ chúng triệt tiêu qua RMSNorm | FAIL: 23.98 % trong tol, 70.55 % so với kỳ vọng | CÓ HẠI (SAI) | mất 1 job | normalize(c·v) ≠ normalize(v) khi EPS = 1e-6 cỡ bằng mean-square (~1e-3 của tích đã lượng tử) | ffn-next/arena_e2.log; MEM:rmsnorm-scale-does-not-cancel |
| F48 epilogue E1b (41499) | cùng việc xoá, dạng pass khác | FAIL, sai số giống hệt | CÓ HẠI (SAI) | mất thêm 1 job | sai số byte-identical ở 2 dạng pass chứng minh lỗi số học, không phải thứ tự ALU | ffn-next |
| F49 **epilogue FFN 5 → 4** (E3, 41516) | gộp global scale vào các consumer (`Mul0(scale) → stash → Mul1(stash) → reduce` cho ms; `Mul(scale) → Div(rms) → Mul(weight)` cho normalise) | −401; 324,533 (41516) PASS max 0.00781 | CÓ LỢI | – | đuôi FFN lộ hoàn toàn (~12k trên 140,390 là đuôi) | guide-fork/RESULT |
| P4 port v3e (FFN) | mean-square epilogue: partial trong slice + Broadcast1{8,1} + reduce trong slice | FFN −4.5k so với v2 | CÓ LỢI | – | sai số bit-identical | RES |
| FA7 Y1 (79125) | một lần relayout y DM→DM (layout Down đã reduce → 8 slice epilogue) | 111,759 (−703); 225,887 | CÓ HẠI (revert) | – | compiler thêm op Core `DramReuse` (1,097 static) = một span Cluster thứ 2 4.5k HW trước store cuối | ffn_down_notes |
| FA20 ffn8 y1 / gf (static) | y1 một gather epilogue DM→DM; gf gộp gate_scaled + gelu Sub + StoVrf Sub thành một pass VE→VRF trên Main | y1 DramReuse +1,097; gf "13 is not available for op Binary(MulF)" (3 phép nhân, chỉ có 2 ALU nhân) | ĐÓNG BẰNG PHÂN TÍCH / KHÔNG BIỂU DIỄN ĐƯỢC | – | – | ffn8_notes |
| FA21 ffn8 **e16** (LAB4 79564) | epilogue 16 slice × 240 kênh (Broadcast1{16,1}) | −1.5k so với s123 | **CÓ LỢI** (v8) | – | – | ffn8_notes |
| FA27/FA31 **ffn9 e1** (80046, 80408) | ver15: reduce inter-slice KHÔNG pad (ms + tổng 16 slice trong một pass, `m![Dummy256/16, Epi16]`) | 110,889; lab median +1.5k (nhiễu) nhưng chuỗi sau ClusterSync 2,911 ở cả 3 run so với 3,388–3,954 ở v8 (−0.48k, tất định); FFN 216,770 (80408) | **CÓ LỢI** (v9) | – | nhiễu lane ±2k của median lab che mất; phải đánh giá theo chuỗi sau sync của từng launch | ffn9_notes |
| C2 (pha 3a) → E111 SQF + C2 | operand epilogue đã gate (`residual·gate`, `norm_weight·gate`) kết thúc bằng Sub `.vector_final().to_vrf()`; bỏ 2 pass Sub StoVrf; + gộp sqrt vào pass final (EPS/16 mỗi partial trong pass ms → VRF; Mode021 / Mode10) | static 110,301 (−588); emulator 0/3,840 khác; lab1 85212 −461 (SEd 405); lab2 85243 +147 (438); **pooled −181 ± 297**; span đúng như dự: chuỗi sau sync c0 −0.34..−0.36k, c1 −0.47..−0.52k | CÓ LỢI (giữ như một phần của E112; riêng lẻ không đáng kể) | – | thay đổi đuôi tất định: giữ nhờ span được xác nhận + E[max2] không tệ đi | V10-E111.md, V10-E112-verify.md |
| F5 epilogue FFN cục bộ H7 (V10 P2) | chỉ partial mean-square đi qua cluster (cluster_swap 32 B) | static 111,034 (+145); mô hình 216,232 (+0.65k) | ĐÓNG BẰNG PHÂN TÍCH | – | barrier không bỏ được; kết thúc bị c1 gate kiểu gì cũng vậy | CRITICAL_PATH_FFN.md F5 |
| F9 epilogue trên c1 / tính trước trên Sub | post của c1 bị trễ bởi Sub 544/549/557 | trần 0.3–0.5k | → C2 (E111) xử lý trễ post của c1; "epilogue trên cluster về muộn" ĐÓNG | – | barrier đối xứng | CRITICAL_PATH_FFN.md F9 |
| MOV-4 Y1 | một gather y DM→DM thay Stos y + d2d | static 110,390 (−499) nhưng DramReuse(247) được chèn lại (1,600 static); đo ở ffn-down 4.5k HW | ĐÓNG BẰNG PHÂN TÍCH | – | Stos y (1.24k) không đáng một barrier thứ hai | IDEAS_MOVEMENT.md MOV-4 |
| S6 Stos y → gather TU về 4 leader | – | đuôi nối tiếp trên Main: pass ring-64 ~1.3k thay DMA 1.24k 1:1 | ĐÓNG BẰNG PHÂN TÍCH (≤ 0.3k) | – | – | IDEAS_SCHED.md s6 |
| "lit" offset tile là literal | viết `tile(r0)` bằng literal để bỏ section code CPU | số PE giống hệt (102 code section, 126 tuc_sync_tail ở FFN V10) | TRUNG TÍNH | – | offset tile luôn lower thành `tidx_dynamic` lúc runtime | IDEAS_MOVEMENT.md s1.6 |
| FFN-L barrier L cuối (~0.9k ở 3/4 launch warm) | quan sát: c0 chờ 921 / 901 / 49 / 913 | c1 chỉ đi trước 0.68k < L | ĐÓNG BẰNG PHÂN TÍCH (không đòn bẩy) | – | – | IDEAS_SCHED.md s2 |
| FFN-CLK "bước +0.83k" / phần dư fused3 | giả thuyết stall control_tu_clock | 500 cluster-run lab: fused3 median 5,169 (5.12–5.20k) bất kể TU idle trước đó | ĐÓNG (nghiên cứu trace) | – | độ dài fused3 là bản chất; độ dài g1 không dời W3 (W3 issue lúc g1 dispatch) | IDEAS_SCHED.md s3 |
| fetch-axis lifting làm thay switch / DMA | – | kiểm kê: mọi site cần nhân bản hoặc gather; lift không làm được | ĐÓNG BẰNG PHÂN TÍCH | – | – | IDEAS_SCHED.md s5 |
| giảm seed nhân bản gs; mẹo zero-padding; asymmetric_cluster_slice; pdma cho load nhỏ; reload trước s0 (V10) | – | config có tên là all-gather (replicate 1→N cần ~1.1k cfg DMA); to_dm_view vào tile fail trên HW; chỉ chọn phần tử từng slice; pdma chỉ host↔HBM; chỉ đường A | không theo đuổi | – | – | IDEAS_MOVEMENT.md s3 |
| C6 tách reduce K-half của Down (V10) | reduce hàng 0–11 trong cửa sổ ~4.5k Main idle trước W3 | chưa build; trần 0.15–0.25k | ĐÓNG BẰNG PHÂN TÍCH (DROP) | – | pass nhỏ bị giới hạn chi phí cố định; rủi ro xáo đuôi kiểu t13/s123 | IDEAS_COMPUTE.md C6 |
| FFN-GSU (ffn14, V14-E802) | gộp up_global_scale qua Down tuyến tính vào epilogue (gsu × gsd lên y trước mean square; EPS chính xác); DOWN_ACTIVATION_SCALE 2^-10 | sim số học host PASS (d/tol xấu nhất 0.656); static: Gate W −583 nhưng DMA residual dời ra sau sync và compiler thêm DramReuse(160) cross-cluster trước store in-place (đuôi +382 static, makespan chỉ −201); gate (ii) của critic fail (slack sau W3 của c0 trung bình 506) | ĐÓNG BẰNG PHÂN TÍCH | – | một DMA epilogue mới bị lộ trên c0 + thêm DramReuse | ledger V14; ffn14_progress |
| review P1 gsu-late | – | global scale KHÔNG phải hằng số của fixture: harness rút theo từng run từ U(2048, 16384); hằng số trên V13 phải là 2^-10; 2^-6 của analyst sẽ làm bão hoà f8 → FAIL | gộp vào P2 → kill ở static | – | – | ffn_review §3 |
| GSU (C4) trên octk (ffn18) | gộp up gs vào scale epilogue | sim host: r2 max abs d 0.01172 cho mọi S = 2^-10..2^-6 | CÓ HẠI (KILLED theo gate số học) | – | output GeGLU được làm tròn bf16 mà không có hệ số gsu (không phải luỹ thừa 2) → kiểu làm tròn khác | ffn18_report |
| fa1 / fa1b / fa1c / fa1f / fa1g (FFN-A, ffn15) | gộp pass cộng partial sau sync (0.73k) vào pass ms và final | fa1: lowering FAIL (AddF và FmaF của pass final cùng cần ALU FpFma); fa1b/fa1c: "operator schedule base graph has a cycle"; fa1g static 104,303, load P_E sau sync exchange → chuỗi sau sync 1,588 so với 1,215 (+373); lab L 99728 +681 ± 222 (1/6) | CÓ HẠI / KHÔNG BIỂU DIỄN ĐƯỢC (FFN-A ĐÓNG) | – | không có cách rút ngắn bit-identical add 0.73 + ms 0.97 + final 0.74 + store 0.85k trên 0.8.1 | ARCH_FFN_B §7 |
| Pcie d2d (đuôi FFN) | Dma::Pcie cho copy HBM→HBM | k15 không có copy HBM→HBM (exchange là DM→DM); Pcie vẫn lower thành tensor DMA | ĐÓNG (DEAD) | – | – | ffn15_progress; AUDIT_L1 §6 |
| rga (8cb64df0) (Lever 3) | tách regroup: hàng 0–10 regroup trên Sub trong lúc fused4 | LIR ICE "no entry found for key" (sfr_impl.rs:868) | KHÔNG BIỂU DIỄN ĐƯỢC | – | Sub không chạy được regroup switch + transpose | ARCH_FFN_B §8 |
| rgm (53dc8c21) | cả hai nửa regroup trên Main | static 104,303 (+379); lab M 99764 +355 ± 169 (2/6) | CÓ HẠI | – | nửa đầu nằm trên Main trước fused4 (Main đang là nút thắt) | ARCH_FFN_B §8 |
| rgb / rgb2 (13a33bf4) | regroup API cặp + cộng trên c0 | rgb FAIL "VRU reduce axes must be innermost in Partitioning"; rgb2 compile 112,271 static (pass regroup + add dạng cặp 8,457 static: VE trên packet 1 giá trị mỗi flit `[1 # 8]`); lab M +19,043 ± 3,270 | CÓ HẠI | – | VE trên packet 1 giá trị/flit chậm ~20× | ARCH_FFN_B §8 |
| gộp pass cộng vào ms / final (ffn20) | – | bị chặn bởi thứ tự stage VE (stage/markers.rs: không có stage Fp sau IntraSliceReduce; zip clip dạng cặp chỉ gộp ở cuối; p0·k + p1·k không bit-identical) | KHÔNG BIỂU DIỄN ĐƯỢC | – | – | ffn20_report |
| X14 push partial sớm (BACKWARD) | push sớm các hàng của piece A | rgm +355 áp dụng khi Main là nút thắt | ĐÓNG BẰNG PHÂN TÍCH | – | – | BACKWARD_FFN §4 |

#### 4.8 Nhóm H — Contraction, packet, lookup FP4

| tên/id | thay đổi | kết quả | kết luận | được / mất | lý do, bài học | nguồn |
|---|---|---|---|---|---|---|
| F31 FFN-019 | 32 hàng × 240 slice để hàng bf16 là 64 B | CF | KHÔNG BIỂU DIỄN ĐƯỢC | – | "a DM allocation must span the whole device"; pad `# 256` → "snoop bitmap disagrees at output slice 240" | OL§3 |
| F32 FFN-020 @60 (REV-AE) | GeGLU cục bộ trong slice ở 60 phần tử/slice | 334,055 (35640) FAIL 10.8 % | CÓ HẠI (SAI) | – | packet collect có pad (4 sống / 8) làm lệch lane VRF so với stream đã thu hẹp | TSV:50 |
| FA1 ffn-down F1 (khởi động lại) | lookup FP4 fuse theo từng nhóm trong 4 pass nhóm của Down | 119,359 (+1.4k): 4 DMA bảng + 4 StoTab | ĐÓNG BẰNG PHÂN TÍCH | – | mỗi pass lookup một bảng | ffn_down_notes |
| FA16 ffn8 **g2** / g3 / **g4** (LAB1 79382) | g2: các nhóm Down + contraction fused3 dùng PackSize 2 (packet 64 B) + LaneMode::Sequential (4 block sum mỗi flit output thay 1); g3: như vậy cho Up/Gate; g4: g3 + sqrt epilogue → VRF từ Main | 112,797 / 112,797 / 112,529; 223,897 (−0.8k) / 224,081 / 223,049 (−1.6k) | **CÓ LỢI** (g2, g4; g3 trung tính nhưng vẫn gộp) | pass nhóm 5.7/6.5/8.0k → 2.8/3.6/3.6k | nút thắt là lane folder Interleaved với Lane = 1; sau đó đuôi chờ W3 (W3 cấp phát cùng page 16 MiB với DownResult: DMA W3 chỉ bắt đầu khi g2 XONG); contraction Up/Gate vẫn 21.0k: bị lookup giới hạn (3,600 flit × 5.83 c) | ffn8_notes |
| FA18 ffn8 lk64 / lk128 | packet fetch lookup Up/Gate 32 B / 64 B | 224,613 (+0.5k) / 223,880 (−0.2k) | TRUNG TÍNH (đóng) | – | throughput lookup không phụ thuộc packet fetch (~5.5 phần tử decode/cycle/slice) | ffn8_notes |
| C8 decode FP4 nhanh hơn / PB5 phụ thuộc dữ liệu của rate lookup | – | cap công khai 16 B FP4/cycle (~3× cao hơn đo); giả thuyết: scratchpad bảng (banked, table_dup_count < 16) là giới hạn; chưa kiểm all-zero so với code ngẫu nhiên | MỞ | – | – | FFN_GAP_external §3; BACKWARD_FFN §6 |

#### 4.9 Nhóm I — Số học (FP8 hi/lo, scale, gate số học nội bộ)

| tên/id | thay đổi | kết quả | kết luận | được / mất | lý do, bài học | nguồn |
|---|---|---|---|---|---|---|
| F24 FFN-016 / 016b | contraction FP8×FP8 với x tách hi/lo = f8(4x) + f8(4x − hi) trên trục time `Hl`, reduce bằng `contract_time`; 1/4 gộp vào pass scale | 160,854 | CÓ HẠI (tạm) | Main 101.6k → 76.3k, DMA 135.7k → 129.8k | list DMA đặt chuỗi activation lên trước tile Up (+12k ở đầu), load Down muộn; thứ tự chương trình không sửa được | TSV:35 |
| F25 FFN-016c (REV-U) | hi/lo trên K-half trước ring exchange (1 pass lo, exchange f8) | 154,247; 340,487 (35508) FAIL max 0.0303 (12.7 %), 96 % | CẦN ĐIỀU TRA → bỏ | – | giả thuyết "lo sai dấu (SubF = vrf − stream)" là ĐÚNG (xác nhận bởi v11 `shared/mlp.rs:52-53`) nhưng lúc đó đã bị bỏ | TSV:37,38; LED X11 |
| F26 FFN-016e (REV-V) | pre-scale 2×, lo = 2x + (−hi) qua AddF | 155,096; 347,250 (35530) FAIL max 0.0537 (73 %), 80 % | CÓ HẠI (SAI) | – | lúc đó nghi tràn e4m3 | TSV:39 |
| F27 FFN-016f (REV-W) | pre-scale theo tensor (Up/Gate 128, Down 1024) | 155,096; 353,222 (35534) FAIL max 0.039 (21 %), 89 % | CÓ HẠI (SAI) | – | dấu hiệu = "lo bị bỏ" trong sim host | TSV:40 |
| F28 REV-Z / FFN-017 | chạy lại 016f (script patch hỏng) | 341,768 (35555) FAIL giống hệt | TRÙNG | mất 1 job | tất định | TSV:43 |
| F29 FFN-018 (REV-AC) | hi/lo 016f với buffer block theo tile | 202,798; 435,849 (35568) FAIL cùng thống kê | CÓ HẠI (SAI) | – | tất định, không phải hazard lịch | TSV:44 |
| F30 sim hi/lo trên host | `tools/ffn_fp8_sim.py` trên fixture thật (`scripts/fixture_prng.py`, torch float8_e4m3fn) | BF16 0.00781/0.000487/100 %; hi/lo 128/1024 100 %; theo nhóm 16/64/960 100 %; một f8 block-16 0.01953, 98.93 % FAIL; "lo lật dấu" sim 0.0593/76 % ≈ REV-V; "lo bị bỏ" 0.0352/94 % ≈ REV-W | phát hiện | – | hi/lo chính xác về mặt số học; một thành phần là không đủ; lỗi device là do trục/packet (v11 reduce trục hi/lo `Gs` trong contract_time trong khi giữ block và chạy đúng) | OL§3c |
| F50 activation FP8 theo nhóm 960 | – | tạm dừng | ĐÓNG BẰNG PHÂN TÍCH | – | rủi ro cao (tol FFN chặt); v11 dùng theo tensor 16 / 256 | guide §15-16 |
| activation Down f8 đơn (ffn_gap) | tương tự Q-6 của QKV | giấy: sai số tương đối ~3.6 % RMS so với atol 0.01 + 1 % của FFN | ĐÓNG BẰNG PHÂN TÍCH | – | rủi ro FAIL | ffn_gap §6 |
| số học varA | – | sai số FFN nhỏ hơn: 0.00391 / 0.00781 / 0.00781 so với 0.00488 / 0.00781 / 0.01172 (V12); mean abs d +~7 % | phát hiện (làm đổi bit output, trong gate) | – | từ V13, số học của mọi KEEP giống hệt (bit-identical) | V13_RESULT |
| gate số học FFN nội bộ | per-run max abs d ≤ 0.00391 / 0.00781 / 0.00781, mean ≤ 1.10 × control; các arm có cùng cấu trúc phải khớp tuyệt đối (tools/numchk.py) | chặt hơn tolerance cuộc thi (~2×): chặn D-CE, GSU, R5 | QUY TẮC (user có thể đổi, xem §8) | – | – | ARCH_FFN_B §5; BACKWARD_FFN §7 |
| K38 global scale FFN được rút ngẫu nhiên | harness rút theo run từ U(2048, 16384); raw up của 3 run: 9,508.6 / 3,429.6 / 4,642.2 | mọi phép gộp giả định hằng số đều sai | QUY TẮC | – | – | test_kernels.rs:153,179-190; generate_references.py |
| K102 emulator CPU không chính xác cho FFN | cc1c tự fail trên emulator (max abs d 1.043) trong khi HW max 0.0039; arm cùng cấu trúc lệch nhau trên emulator khi tập DMA scale Down đổi | emulator chỉ dùng để so bit-identical với base, không dùng làm oracle | QUY TẮC | – | – | ARCH_FFN_B §5 |

#### 4.10 Nhóm J — Đo đạc và giải phẫu FFN

| tên/id | nội dung | kết quả | kết luận | nguồn |
|---|---|---|---|---|
| F62 FFN trong cây r21n | FFN v11 không đổi | 118,593; 253,629 (77046), 254,775 (77120); chính thức 258,457 / 259,679 / 256,187 / 260,184 | dải FFN 253–261k (0.6) | LED |
| FA25 **ffn8 V1** (79621, 79666) | s12e16 trong harness thường | FFN 216,743 / 217,367 (−7.9k, −3.5 % so với v7) | END ≈ W3 land + fused3 5.2k + reduce 0.8k + epilogue 8.6k; lane DMA bận 96.8 %; mô hình chi phí mỗi DMA ~2k cố định (4.4k nếu lệch) + bytes/~620 B/c | ffn8_notes |
| giải phẫu critical path v9 (CRITICAL_PATH_FFN) | lane bận 96.8 % c0 / 95.2 % c1; 3 đường kết thúc hoà nhau | thời gian lane mất trước W3 ~34–37k: sàn lệnh nhỏ (bảng, gs, cfg, x/rms ~17k), packet scale lệch (~12–13k), x hop (~9.7k) | phát hiện | CRITICAL_PATH_FFN.md |
| phân rã lane V12 (ffn_gap, 12 cluster-run warm) | head 5,943 / Up W 47,674 (618.6 B/t) / Up nhỏ 13,770 / Gate W 47,469 / Gate nhỏ 14,075 / W12 37,742 / x hop 12,370 / scale Down + bảng 13,176 / W3 10,333 / tail 13,271 = 215,823 | phần dư không phải trọng số ~46k; #1 cần bớt ≥ 10.2k thời gian không-trọng-số | phân tích | ffn_gap §2 |
| ghép pha (ffn_gap) | lane so với chuỗi TU theo pha | V12 pha Gate: chuỗi TU Gate găng ~0.8k → cắt lane được 0 (vì sao varA riêng lẻ NULL); V13 fxv: chuỗi TU có 3.46k slack → lane găng (vì sao fxv siêu cộng); nhánh Down: lane chỉ găng hơn đường A 0.37–0.50k | TRI THỨC: giải thích vì sao các đòn bẩy một phía NULL | ffn_gap §3 |
| V13 so với V12 trong 91722 | cùng job trace | Task 215,265 → 211,163; head 5,997 → 5,250; up gs 2,326 → 1,026; gate gs 2,018 → 1,010; W12 land 166,120 → 163,221; W3 201,995 → 198,612; tail 13,270 → 12,551; x reload +0.49k (không giải thích được) | phát hiện | ffn_gap §3 |
| FREECHECKS bảng DMA FFN fxv | 27 lệnh/cluster | tabD12 1,958; x/rms_w 1,368; Up W 47,538; tabUp 1,632; seed Up q4 9,856/9,050 (+0.9..1.7k); up gs 966; Gate W 47,446; tabGate 1,615; Gate scale direct 800 B 10,371/10,584 (+1.75..1.96k); W12 38,235/37,337; storeX 1,547; cfg 1,394; s0 3,528; x reload 5,747/5,864 (+1.45..1.56k); s12 6,203; s3 2,880; tab3 1,975; W3 10,576; load epilogue chỉ-c0 1,035–1,573 | phát hiện (luật request) | FREECHECKS §2-4 |
| giải phẫu cc1c (ARCH_FFN_B, cc1ct 94268 c0) | 24 DMA/cluster; lane 95 % bận | mọi wait còn lại trước W12 giữa 2 DMA đều là StoTab do compiler chờ input (không handle); 3 đường găng: P2 (găng) lane tới W12 land 165.1k + chuỗi Down Main 37.3k + tail 7.7k; P1 slack 1.2k; P3 slack 3.3k | phân tích: FFN bị lane giới hạn tới khi W12 land, sau đó bị Main giới hạn | ARCH_FFN_B §0-2 |
| GAP_FFN (k15, trace 96040) | 3 đường cùng giới hạn | P2 găng: lane tới W11 land 162.20k → chuỗi Down Main 36.45k → tail 7.86k = 206.51k; lớp lane: trọng số 146.22k (sàn 135.90k), block scale 30.85k (sàn 16.99k), 4 bảng 6.74k, x+rms 2.73k, gs 2.00k; lookup 5.50 /tick/slice | phân tích | analysis/v17/GAP_FFN.md |
| giải phẫu octk (ffn18 trace, BACKWARD_FFN) | lane tab1 1.96, x 1.36, rms 1.36, Up W 47.42, tab2 1.60, seed 7.59, upgs 0.99, Gate W 47.81, tab3 1.93, GateS 10.37, gategs 0.93, W11 36.33, s15 10.63, tab4 1.67, W4 14.95; Main sau T0: chờ StoTab Gate 7.45k, Gate contraction 20.96k, Gate chain 19.61k, … fused4 6.49k, tail 8.27k | sàn Main của họ này 195.6k trace; sàn lane ~196.5–198k; sàn ND ~192.3–194k | phân tích | BACKWARD_FFN §1-3 |
| giải phẫu nd11e9t / nd11gt / nd11gzt (ffn20) | – | nd11gzt: seed 0.10–6.75k, bảng 7.96–9.96k, x 10.01–11.52k, Up W bắt đầu 11.53k (nd11gt 13.74k), land 58.94k; W11 156.87k; fusedA 156.88–172.55k; pass 1 bị giữ tới lúc tabB StoTab xong 176.49k; fused4 tới 190.80k; tail 8.2k → 198.99k; **fusedB không VE 6.42k = fused4 có VE 6.49k: VE miễn phí bên trong một fused pass** | phân tích | ffn20_report |
| c1-light asymmetry L1 (fcf18) | dời việc từ c1 sang c0 ở sync găng (tax 2.2–2.5k ở y hop) | chia L thì mỗi slice phải dời ≥ 1 hàng L = 256 hàng × 5.76 KB = 1.47 MB (E 2.95 MB); dời cột Down qua hop h tốn thêm 3 lệnh (+3k) | ĐÓNG BẰNG PHÂN TÍCH (KILL: độ mịn quá thô) | fcf18_report |
| lab G môi trường (ffn15) | 9 arm × 6 × 3 kèm 2 arm trace | Arena 95840 FAIL = scheduler TIMEOUT (70 s) sau 136/162 launch | QUY TRÌNH: lab ≤ ~72 launch; arm trace chậm | ffn15_progress 21:12 |

---


## 5. Kiến trúc chung và các thí nghiệm liên kernel

Chương này gom các probe, audit và phân tích không gắn với một kernel riêng. Chúng đo skew, sync, TU-clock, front-end DMA, bất đối xứng giữa cluster, engine, bảng lookup, v.v. Luật rút ra từ các thí nghiệm này được hệ thống hoá ở §6.

### 5.1 Chuyển toolchain và runtime

| tên | nội dung | kết quả | kết luận | nguồn |
|---|---|---|---|---|
| port 0.8.1 (v1) | port máy móc | QKV static byte-identical (39,969) nhưng HW 77.8–79.0k so với ~89k trên 0.6 → **riêng runtime 0.8.1 đã −11k cho QKV**; device program QKV byte-identical giữa 0.6.0 và 0.8.1 (11 descriptor blob, 22 run_dma, 24 wait_dma) | CÓ LỢI (SUY LUẬN: skew nhỏ hơn / barrier rẻ hơn) | RES; notes_early §3.5 |
| sync trên 0.8.1 | mọi sync do compiler sinh (ExplicitSync, DramReuse, ClusterSync, cuối kernel) là barrier phía device giữa 2 cluster: `tuc_sync_tail` + ghi IPC + poll; 1.2–2.7k HW (0.6: 9–14k) | sync host 2–30k của thời 0.6 biến mất | phát hiện | notes_early §3.5; MEM:furiosa-opt-081-port |
| compiler 0.8.1 | cách compiler lập lịch và các knob | beam search mã đóng (width 500, FFN 276 bước, 10 s); `SKIP_BEAM_SEARCH=1` kéo tile Down 30 hàng lên đầu DMA (126,140 → 148,604); `TACTIC_SORTING_SEED` không tác dụng; chỉ có `#[unroll]`; `DUMP_PE_PROGRAM=<dir>` dump program C của từng cluster | các knob ẩn chạy được ở local nhưng KHÔNG tới grader | notes_early §3.2 |

### 5.2 Microbench và probe (thời 0.6 và bắt đầu 0.8.1)

| tên / job | đo cái gì | kết quả | kết luận | nguồn |
|---|---|---|---|---|
| M01 cùng baseline ở 2 job (31766 so với 33853) | độ lặp lại | QKV −1.2 %, AO +3.0 %, FFN ±0.01 % | AO nhiễu ngay từ ngày đầu | TSV:7 |
| M03 event model | `HW = 1.99·static + 233·(nDma + nSub + nCore)` từ 115 lần đo / 31 thiết kế PASS (R² 0.9994, LOO MAPE 4.4 %) | FFN-025 dự −9,044, đo −2,316 (cặp khớp 39037/39534) | BỊ BÁC 09-16: fit gộp tạo ra số hạng "mỗi event" giả khi static và số event cùng thay đổi; kiểm mô hình trên một cặp khớp sạch | reverse/REPORT.md |
| M05 luật chi phí DMA | `550 + bytes/rate` (startup 544/550/558/550; phần dư Up 3 điểm 2 cycle; rate 1229 / 1025 / 615–855 / 578–818) | dự FFN-024 +2,226 và FFN-026 −2,225 đúng tới cycle | LUẬT (chỉ cho mô hình static) | DMA_COST_LAW.md |
| M06 `HW = c × static` | c lấy từ kernel không đổi trong cùng job (FFN 2.32–2.41, QKV 2.25–2.47, AO 2.20–2.67) | sai với thay đổi list/sync (r12: static tệ hơn mà HW −4.6 %) | một phần bị thay (09-22) | same |
| M07 O weight AO trên 1 cluster (probe static) | 24,902 → 36,562 | rate = số slice sống × ~2.40 B/cycle/slice; HW: 1 cluster 366 B/c so với 620–650 | ĐÓNG: không bao giờ dồn một load lớn lên ít slice | DMA_COST_LAW |
| M10 profiler HW theo từng lệnh (`TUC_PROFILE_LEVEL=trace` 0.6 / `FURIOSA_OPT_PROFILE=trace` 0.8.1) | – | DMA khối 1.85×; decode/contraction 2.4–2.5×; pass vector nhỏ 3–8× (0.4–1.5k mỗi pass); lệnh TU đầu 3.5–4k cold; StoTab 6–9k so với 1,293 static; span khớp 1:1 với static (FFN 31 DMA + 79 TU) | CÔNG CỤ: HW issue theo thứ tự list static, bị chặn head-of-line | dma-docs/FINDINGS §2.1 |
| M11 microbench DMA bench1 (71568) | 1/2/4/8 load nhỏ 7.5 KiB | 17.5/16.2/20.1/13.9k rồi 13.6/11.5/14.6/9.6k: độ dốc ~0; kernel 1 DMA tốn cố định 12–15k | phát hiện (0.6): số DMA nhỏ gần như miễn phí trên HW, VỊ TRÍ của chúng thì không | dma-docs/hw/bench1_job71568.log |
| M12 microbench sync barrier1–3 (71648, 71664, 71698) | – | sync `Cluster` đầu tiên 2–30k (probe t1 18,499 rep0 / 5,539 rep1 / 3,723 rep2); các sync 1 cluster sau 0–1.6k | phát hiện (0.6): sync đầu tiên của một launch = nguồn nhiễu chính | logs |
| M19 audit mức song song (09-15) | – | lane dùng 1/8; lookup ~12 B/cycle/slice; MAC idle > 95 % | phân tích: nới lane/packet vô ích khi lookup giới hạn stream | parallel_utilization_audit.md |
| M20 mức dùng từng lane trên HW (trace v11) | – | QKV DMA 81 % / TU 82 %; AO 77 % / 59 %; FFN 87 % / 85 %; "Main chạy khi DMA idle" tĩnh chỉ 1.2k / 1.9k / 2.5k | QKV cùng giới hạn bởi TU (vì sao r11 NULL) | MEM:rngd-hw-execution-model |
| AUDIT 09-15 phân loại DMA | QKV 35 op / 52,890; AO 10 op / 20,548; FFN 45 op / 129,676 (chia A trọng số / B round-trip / C hằng số / D output) | – | phân tích | AUDIT |
| probe static analysis-work (09-23) | cp-ao (L_rep*, xag256 → v4b), api081 (P1 lookup fuse → v5, P2 vector_final→to_vrf, P3 packet 64 B), secA (A1 packet 64 B + TimeBroadcast, A2 lane fold Sequential, downB −3,453), cp-ffn (hwsim --rt 081) | makespan tái lập được 26,189 / 126,140 / 39,969 | CÔNG CỤ / NGUỒN GỐC của v4b, v5, A1, A2 | analysis-work/ |
| dự đoán trước chiến dịch K1–K8 (Deep_knowledge v5, 09-14) | P1-A decode theo pass; R2 fuse; dùng cả 2 cluster; xen Up/Gate với Sub; quét AO CHUNK; dùng chung x_trf; đo ring-256; sàn HBM | K1 → 2.92× (31778); K2 → CDX (33638); K3 → mọi thiết kế 2 cluster; K4: lookup chỉ Main, StoTab chặn Sub; K5 không còn ý nghĩa; K6 K/V dùng chung; K7 ring-256 8,455 static / 17,128 HW → thay bằng ring-8 / Broadcast1{8,1}; K8 sàn thật ~2.2× dự đoán (DMA 2 cluster ~600–650 B/cycle) | đối chiếu | notes_early §2.9 |

### 5.3 Profile trace so với INFO, và nhiễu

| tên / job | đo cái gì | kết quả | kết luận | nguồn |
|---|---|---|---|---|
| trace so với info (QKV, lab 85630) | cùng code ở hai mức | QKV warm A_i − A_t −713 (SE 184, 7/8); B −601 (151, 8/8); cold ~3.3k; c1 đẩy 681 lệnh TUC ở trace so với 496 ở info | LUẬT: quyết KEEP/kill trên arm INFO; arm trace chỉ để xem span | V10-E351.md; memory trace-profile-overhead |
| trace so với info (AO, FFN; lab 85694; V11.1 85677/85678 so với 85340/85350) | – | AO +925 (142) warm, cold +2.3..2.6k; "trace tốn ~1k ở AO, ~0.5–1.5k ở FFN, ~0–1k ở QKV" | LUẬT: cây làm việc mặc định trace, snapshot nộp mặc định info → median trên cây làm việc đều bi quan | V10-E224.md, V11_RESULT.md |
| TU-clock probe (99849, 99881, 99892, 99898, 99902) | control_tu_clock "slow contraction" | trace +2.58k, INFO −0.29k | ĐÓNG: artefact profile trace (backlog PERT); 2.21 slope ≈ 2.0 do đổi đơn vị (compiler 1 GHz, bộ đếm 2 GHz) + ~10 % sai mô hình + ~0.51k cố định mỗi span | TUCLOCK.md; TUCLOCK_external.md |
| profile debug (qkv19 lab5) | `FURIOSA_OPT_PROFILE=debug` | chỉ ghi span Task; chi phí bằng INFO (+48 SE 149) | không cung cấp thông tin cơ chế | qkv19_report |
| outlier host launch-skew (V10-E112 verify) | xg2 b2k3 264,506 (c1 launch trễ ~49k) | skew abs(T0 − T1) > 20k = 5 / 2,965 launch trên mọi lab, đều do c1 trễ, không phụ thuộc arm | loại bỏ, có lý do từ span | V10-E112-verify.md |
| outlier host-sync (qkv19 lab8/lab10) | ~+19–20k mỗi launch | ~1 lần mỗi lab 72 launch | screen 3.5-MAD chỉ cho lab có outlier > 10k | qkv19_report |
| freeze AO | stall hiếm trên toàn c0 | ~1/36–1/56 launch warm | rủi ro nền tảng | §3.7 |

### 5.4 Front-end DMA: số lệnh, lệnh nhỏ, alignment, engine

| tên / job | đo cái gì | kết quả | kết luận | nguồn |
|---|---|---|---|---|
| qkvMB (trace mining) | DMA nhỏ xếp sau trọng số | xếp hàng giấu được 0.2–0.6k (12–35 %); 0.8–1.7k vẫn tuần tự | LUẬT | qkvMB_notes.md |
| MOV-6 guard queue-block PE (`p081-v10-rdJ/tools/peblk.py`) | so cấu trúc queue-block của program PE ở mỗi lần chuyển TU/DMA | giải thích st4y4 +2.6–2.9k: ring bắt đầu 398–402 sau decode12 ở 12/12 launch, dù list static giống hệt | CHỌN LÀM gate trước lab | IDEAS_MOVEMENT.md MOV-6 |
| MOV-7 mô hình rate load lệch | rate chip ≈ 2.45 lần chạm unit/cycle × byte mỗi lần chạm | dự / đo: trọng số 627 / 620–645; scale Down (480 B) 314 / 285–326; seed q4 Up (960 B) 348 / 369; seed Gate (800 B) 301 / 358 | chỉ căn đầu 256 B mới có ích; packet dài hơn / gộp lệnh thì không | IDEAS_MOVEMENT.md MOV-7 |
| audit memset-pin (cả 3 kernel) | xếp hạng mọi DMA bị TU gate trong V11.1 (16 site) | chỉ đầu AO có giá trị (aoYG); chi phí static memset = 266 + bytes/slice/8 | CÓ LỢI (aoYG) / phần còn lại 0 | MEMSET_PIN_AUDIT.md |
| nghiên cứu plateau DMA "H-ROW" (V12; X1 bên ngoài, qkvRate, covM) | – | X1: tick ~0.5 ns; trần HBM ~665.6 B/tick ở 5.2 Gbps; probe tốt nhất 651 (97.8 %); lệnh trọng số trong kernel 572–618 B/tick chủ yếu vì chi phí cố định F ~1.0–1.9k mỗi lệnh; 2 cluster = 1.68× một cluster; số "Down 510 B/tick" là so sánh chéo job sai | TRI THỨC: đòn bẩy còn lại là ít lệnh hơn / lệnh nhỏ hơn / nhiễu, và ≤ 2–5 % từ layout | COMMON_v12.md |
| FREECHECKS (V15, trên trace đã có) | GL-3 (Sub store so với contraction), KR-1 (độ phủ kênh HBM), CJ-3 (độ phủ engine), CJ-4 (DMA bảng), KR-4 (contraction sau stream) | **luật request: service = 1,212 + 3.217 × request mỗi engine** (8 engine / 3.217 tick × 256 B = 637 B/tick chip = giới hạn pin); GL-3 DEAD (0 tick; store Sub bị chậm ~50 % dưới Main, Main không chậm); KR-1 DEAD (scale phủ NHIỀU kênh hơn trọng số; cái mất là thuế request của piece lệch ~5.5k/cluster); CJ-3 0 tick; CJ-4 không có handle (bảng 2 KB/cluster, 64 packet 8 B/engine, 1.61–2.09k); KR-4 đã thoả (đúng 1 contraction sau byte cuối) | FACT | analysis/v15/FREECHECKS.md |
| **PROBES J1 = 94294** (16 arm × 8 block) | P2 GL-1 (overlap giữa các engine rời nhau) + P1 hình học CC-1 / CC-1b | **P2: descriptor xếp hàng THEO TỪNG ENGINE (per-PE) trên HW** (q_AB: A kéo dài 11.2k → 14.4k/14.7k ở 16/16 launch; lane so với control nối tiếp −1,108 ± 169 / −2,848 ± 282); nhưng **KHÔNG biểu diễn được trong 0.8.1**: mọi DMA biểu diễn được đều chứa engine 0 (grammar mapping không có offset; tile slice/cluster làm đích DMA chưa được lower) → trần 10.2× → 10.6× không mở được; phụ: dispatch sớm giấu ~0.8k mỗi load nhỏ khi không có wait/TU gate ở giữa; load nhỏ rộng (4 engine) tốn như load 1 engine | FACT (H) | analysis/v15/PROBES.md |
| P2.2 cách TUC dispatch DMA | compiler đặt `wait_dma` theo tag hoàn tất của DMA ở consumer ĐẦU TIÊN của nó; DMA không có consumer thì wait ở static end của nó → lab chỉ-DMA luôn cho gap = +2 (artefact của compiler) | trong kernel, các DMA không có wait/TU hoà giữa chúng được dispatch liền nhau (fxv: tabUp span bắt đầu 34.4k trước khi Up W kết thúc …) | FACT (rút lại "luật tag" tạm thời) | PROBES P2.2 |
| P5 độ sâu pipeline DMA (static, V15-B) | 33 arm | giới hạn runtime TUC: hàng đợi 64, 64 thanh ghi, 32 completion id; `wait_dma(id)` pop mọi id push trước nó (runtime giả định hoàn tất theo thứ tự) → một wait ở DMA cuối của burst bao cả burst; compiler tách queue block ở 32 thanh ghi = 8 run_dma; DMA vào tile trong slice của tensor lớn chép TOÀN BỘ extent có pad → tuần tự; start tile ở biến runtime → DMA động và tuần tự | FACT | PROBES P5; probeP5.progress.md |
| fcf18 J2 / J4 (102567, 102723): burst DMA nhỏ | aS16 nối tiếp 1.24–1.29k/lệnh; aP16 liền nhau 0.82–0.91k; qz16 sau trọng số: lệnh đầu 0.57–0.64k rồi ~1.00k; aS4 1.23–1.26k, aP4 #3–#4 0.82–0.85k | **H-A (burst 0.3k/load) SAI**: burst chỉ tiết kiệm 0.33–0.42k mỗi load; sàn ~0.8k + ~3.2 tick mỗi request 256 B trên engine 0 | FACT | fcf18_report |
| AUDIT L1 (engine set) | có thể bỏ engine 0 khỏi DMA từ source chấm điểm không? | HW/runtime hỗ trợ (slot descriptor 0xffffffff, `run_parallel_dma`) nhưng compiler: mọi DM allocation phủ cả device với cell sống từ slice 0; tile view bị visa từ chối làm đích (d1) VÀ nguồn (s1, s2); alias legacy `HbmTensor::to_dm` 3 đối số / `to_dm_at` compile được nhưng địa chỉ bị BỎ QUA im lặng; `run_dma_check_eq_zero/le_zero` chỉ có ở đường attention production (0 call site trong 40 dump) | KHÔNG gỡ được từ source | AUDIT_L1.md |
| AUDIT L1 Dma::Pcie (p1–p4) | pdma trong device code | lower thành tensor DMA bình thường (run_dma_static, 4 descriptor, PE dump byte-identical với tdma) | DEAD END (vd. d2d đuôi FFN); DMA có đích là output HBM kernel được hoãn wait tới cuối kernel (fire-and-forget) | AUDIT_L1 §6 |
| AUDIT L2/L4 (đặt wait / thứ tự list) | ~200 primitive; knob compiler | không primitive nào tên wait / sync / fence / barrier / priority / order / prefetch / DMA async; Config driver (wait_before_every_instruction, dma_grace_period, force_wait_targets, enable_auto_pipeline, …) không với tới từ source chấm; view op trên input consumer không dời start static; với hàng đợi rảnh, trọng số ASAP (không phải ALAP); loop rolled không pipeline | không có điều khiển nào ngoài pin/swap | AUDIT_L24.md |
| AUDIT L3 (bảng LUT) | lut1, lut2, lut2s, lut64, lut64p, lutbf, lutloop, lutunr | mỗi pass lookup có DMA bảng riêng (2 KB/cluster, static 838) + StoTab Sub (1,293); KHÔNG khử trùng (2 lookup cùng input → 2 + 2); không hoist khỏi loop rolled; `#[unroll]` với lookup → LIR ICE | không có handle; đòn bẩy duy nhất = ít pass lookup hơn | AUDIT_L35.md §1 |
| AUDIT L5 (sync) | sylocal, syc0, syone, sygather, syswap, syhbm, syip0, syip2 | ExplicitSync cuối kernel là vô điều kiện; DramReuse chèn cho mọi đọc-rồi-ghi in-place HBM kể cả chỉ c0; exchange DM→DM rẻ hơn round trip HBM 682 static; kernel 1 cluster / 64 slice bị từ chối | không có handle | AUDIT_L35.md §2 |

### 5.5 Sync, skew, bất đối xứng (fcf18, V18)

| tên / job | đo cái gì | kết quả | kết luận | nguồn |
|---|---|---|---|---|
| đồng hồ chung (fcf18) | offset giữa bộ đếm của 2 cluster | một đồng hồ với offset cố định mỗi phiên (−290,346,962 ± ~150 ở J1; −319.02M J2; −315.51M J3) → đo được skew từng launch bằng print SB | FACT (công cụ) | fcf18_report H0 |
| J1 = 102420 skew launch | – | skew s warm median ~0.95k (IQR 0.70–1.16k, max 2.70k); cold 0.42k; không phụ thuộc arm (1–9 DMA, có/không sync giữa); tự tương quan lag-1 −0.16 | FACT: không kernel nào điều khiển được (host submit lần lượt theo rank) | fcf18 |
| J1 thuế barrier | – | pass nhanh 2–3 tick; một bên chậm: qua ~0.69k (c0 chờ) / ~0.50k (c1 chờ) sau khi bên kia tới; cả hai ngủ: mỗi bên 0.9–1.1k; thuế trên metric của kernel đối xứng = s + l0 (e1 1.75k, i0 1.67k, h1 1.74k) | FACT | fcf18 |
| J1 quét mất cân bằng (iK) | c1 làm K DMA no-op | wait cuối của c0 sụp từ ~1.7k xuống 2 tick khi c1 xong sớm ≥ ~1.9k (own-frame); lợi metric −1.6..−2.0k ở INFO (i0 12,114 → i2..i5 10,045–10,502) | FACT | fcf18 |
| J2 = 102567 kích thước frame | 2 so với 14 đối số | ~+20 tick mỗi word đối số launch | không phải đòn bẩy | fcf18 |
| PR-1 (81 launch warm) | độ trễ release theo độ lệch lúc tới | waiter tới ≥ ~0.6k sau POST của bên kia → qua trong 2–3 tick; waiter tới sớm → được thả ~0.65k sau khi bên kia tới; **DEAD ZONE**: tới 0–0.5k sau bên kia → ngủ 0.9–1.0k (TỆ HƠN tới sớm) | LUẬT: đặt ≥ ~0.6k việc độc lập trước một WAIT găng, hoặc không đặt gì | fcf18 H4 |
| J3 = 102670 bất đối xứng trong list bị stream giới hạn | c0-only thêm E byte, c1 no-op | E 393 KB −0.97k ± 0.31; 786 KB **−2.42k ± 0.30**; 1.18 MB −1.26k ± 0.17; 1.97 MB +0.58k ± 0.18; stream W của c1 chậm hơn 0.2–1.9k (tranh chấp với lệnh kế của c0) | FACT: bất đối xứng mịn c0-nặng hơn có lợi (~3× so với mô hình fluid V14), nhưng kernel của ta chỉ dời được khối thô (QKV head 1.97 MB, FFN hàng L 2.95 MB, nhóm AO ~0.98 MB) | fcf18 H5 |
| các đòn bẩy L1–L4 (fcf18 STEP 2) | L1 c1-light: QKV / FFN KILL, AO biên; L2 lật dead zone AO −0.45..−0.55k (→ ao20: chỉ có ở trace); L3 burst −0.33..−0.42k mỗi ranh giới; L4 cửa sổ sync giữa kernel ≥ s + 0.7k che được sync (FFN y hop và QKV cuối không có việc độc lập) | tổng: QKV 0, AO ~0 ở INFO, FFN 0 | ĐÓNG | fcf18 H8 |
| ao20 chữ ký cuối kernel ở INFO | c1 end − c0 end (wall) | 0.82–1.05k ở 36/36 launch warm INFO (aoyg_i, lf2_i, rpb_i) | FACT: dead zone là artefact trace; phải kiểm lại thuế trace của FFN y hop bằng chữ ký này | ao20_report |
| metric thực tế | – | metric = max(Task_c0, Task_c1) = c0 ở 142/150 run warm NH; dT = c0 − c1: QKV 2,238, AO 1,218, FFN 369 | FACT | fcf18 STEP 0 |

### 5.6 Nghiên cứu công khai, cloud và bảng xếp hạng

| tên | nội dung | kết quả | kết luận | nguồn |
|---|---|---|---|---|
| 5 sweep nghiên cứu V15 (09-25) | official (patent, paper, forum, docs), korean, global, cnjp_analog, contest_code | ý tưởng OFF-0/2/3/4/7/11, KR-1..6, GL-1..5, CJ-1..9, CC-1..4 → sinh ra CC-1 (→ cc1c V15) và GL-1 (→ tồn tại trên HW, không biểu diễn được) | CÓ LỢI (CC-1) | analysis/v15/sweep_*.md; CONSOLIDATED |
| số phận các ý tưởng của sweep | OFF-2 audit tách DMN ranh giới DEAD (mỗi to_dm = 1 lệnh: FFN 27 = 27, AO 8 = 8, QKV 14 = 14); OFF-3 gộp vào CJ-6; OFF-4 (P4) MỞ; OFF-7 KILL; OFF-11 ~0; KR-1 DEAD; KR-3 xong (lane.py + luật request); KR-4 đã thoả; KR-5 (≤ 16 MiB/lệnh) QUY TẮC; KR-6 (assert span::npu) QUY TẮC; GL-1 tồn tại trên HW / không biểu diễn được; GL-2 → CC-1 / k11mmce8 / k15; GL-3 DEAD; CJ-1 một phần; CJ-2 KILL (trừ CJ-6); CJ-3, CJ-4, CJ-5 DEAD; CJ-6 chỉ gate static (EV 0.02 %); CJ-7 KILL; CJ-8/9 QUY TẮC; CC-1 KEEP; CC-2 không build được (dạng build được CC-1b CÓ HẠI); CC-4 → QKV-A (chưa đạt) | – | ghi sổ | CONSOLIDATED §4 |
| CONSOLIDATED U1–U10 + roofline (09-25) | tầng A stream thuần 13.32× @651; tầng B sàn vật lý FIFO 63.9k / 32.7k / 179.2k = **10.01×** (10.18× @665.6); tầng C sàn bằng chứng 64.0k / 34.1k / 196.6k = 9.56×; B' nếu GL-1 = 10.51× | #1 (9.787 lúc đó) ở 97.8 % tầng B; 12.9× của user không đạt được về vật lý; 11× không đạt với mọi cơ chế đã biết | phân tích | CONSOLIDATED §0, §2 |
| sổ ý tưởng V15 (EV) | 1 GL-1a; 2 CC-1; 3 AO-T; 4 GL-1b; 5 QKV-A; 6 CC-1b; 7 OFF-4/P4; 8 CJ-6; 9 OFF-11 | EV slate ~+1.2 % (~9.04); thực tế V15.2 9.021, V16 9.049 (sát EV) | kế hoạch | CONSOLIDATED §4.1 |
| vòng kế hoạch V14 | FFN-OCT, FFN-GSU, QKV-H, AO-R1, FFN-LUT, ENV-1, FFN-RLD; kỳ vọng +0.30 % | mọi mục KILLED / NULL / ĐÓNG; V14 = V13.1; chỉ tốn 2 job Arena nhờ gate static | VÒNG NULL (gate static tiết kiệm quota) | V14_PLAN |
| ENV-1 (đọc sysfs NPU, ghép vào lab 93569) | – | npu_clocks 2,000,000,000 trên cả 8 PE; governor "performance"; không có thuộc tính giới hạn tần số / slowdown | FACT (entrypoint lab đã sửa rồi khôi phục 19ce94ce) | ffn14_progress |
| INTEL sweep (V18, 09-26) | tư liệu công khai về leader | không có gì mới từ 09-25; board tái lập được ~1 % (không phải may); lead IA vs ta: AO 3.5k (40 % lead log-score), QKV 5.4k (31 %), FFN 14.7k (29 %); thứ hạng kernel của ta AO #3, QKV #7, FFN #10; tốt nhất ngoài IA ghép lại = 9.485×; lần nộp 953875dc: QKV −7,868 / AO −1,153 / FFN −6,587 trong MỘT lần | FACT: hồ sơ công khai không trả lời được "làm thế nào" | analysis/v18/INTEL |
| repo công khai (X2, chỉ đọc ý tưởng) | hyunjun1234 (qua hoanganh0106/gemma) 0.6.0 với host yên tĩnh (process ghim 1 core) 74,651 / 34,785 / 192,869 so với chấm chính thức 81,909 / 38,271 / 210,595 | hệ số môi trường tỉ lệ ~0.91 trên cùng code; tỷ lệ board #1 / ta gần đồng đều (AO 0.900 / QKV 0.911 / FFN 0.908 so với V11.1) | giả thuyết môi trường host (chưa kiểm, không điều khiển được); hàng 09-23 của cùng đội không đồng đều → có thay đổi code thật | V12_X2; ao_gap §3.3; critic §4 |
| các báo cáo của cloud agent | TUCLOCK_external, AO R1/R2/R3, QKV_GAP, FFN_GAP, EPILOGUE_DESIGN, API_RECIPES | mọi thiết kế AO chết ở triage / lowering; QKV T2-1 → q19n2 KEEP; FFN OCT-b chạy lại → octk (độc lập); mô hình HBM XOR và mô hình đặt bảng của cloud bị đo bác; dòng Pcie của API_RECIPES được sửa | QUY TẮC: mọi câu trả lời cloud phải được triage theo nguồn 0.8.1 + một probe rẻ trước khi build | memory external-cloud-agent |
| bảng xếp hạng 09-25 và 09-26 | – | xem §1.6; FFN là kernel yếu nhất của ta so với field (6 trong top 8 ≤ 203k so với ~212k của ta lúc đó) | bối cảnh | sweep_contest_code §2.1 |

---


## 6. Tri thức về phần cứng và toolchain (mô hình máy, luật và bằng chứng)

Mỗi dòng là một luật đã được kiểm chứng. Cột "bằng chứng" ghi job hoặc file. Nhãn (0.6) nghĩa là luật chỉ đúng ở toolchain 0.6.0, có thể không còn đúng ở 0.8.1. Nguồn chính: `notes_knowledge.md` PART A–J và mục §3 của các notes khác.

### 6.0 Luật cuộc thi, cách chấm, cách nộp

| luật | bằng chứng / nguồn |
|---|---|
| Score = geomean(baseline/cycles) của 3 kernel. Mỗi phần tử phải thoả `abs(want − got) ≤ atol + rtol·abs(want)`; gặp giá trị non-finite là fail. QKV atol 0.04 (q, k, v tính riêng), AO 0.05, FFN 0.01; rtol 1e-2 cho tất cả. | knowledge/competition_rules.md §3a |
| Cycles = cửa sổ span `span::npu` của mỗi cluster (từ marker 0x0 tới 0x1), lấy max qua 2 cluster (`Collector::window_cycles`, test_kernels.rs:712-731). Trên thực tế đó là cửa sổ của c0. KHÔNG BAO GIỜ đặt `GEMMA4_PROFILE_SETTLE_MS` (ví dụ settle = 3000 cho AO 392,841 thay vì 51,973). | toolchain_081_notes §e; M09; BACKWARD_AO |
| Grader đã siết từ 09-18: mỗi kernel 3 run có seed trên 0.8.1, lấy median, run nào cũng phải PASS. AO x = bf16 uniform(−1,1) (trước đây ±1). Vị trí RoPE ngẫu nhiên mỗi run trong 1..2047. Global scale NVFP4 uniform [2048, 16384) theo từng run. LAYER_SCALAR 0.375 cố định. Tối ưu riêng cho tính chất của fixture coi như bị cấm. | competition_rules §3b; K38 |
| **Xếp hạng cuối chạy lại bài nộp CUỐI CÙNG** của mỗi đội trước 2026-10-01 11:59 UTC. Board chỉ để hiển thị bài tốt nhất. (Trước 09-23 board giữ bài tốt nhất, nên nộp nhiều lần chỉ có lợi.) | competition_rules; INTEL; M14 |
| Chỉ upload `src/ops.rs` (thân hàm) và mọi file dưới `src/device/**`: tối đa 1000 file, 10 MiB, không có file rỗng, symlink bị bỏ qua. `src/axes.rs` không được upload → lần nộp cd8cad57 FAIL vì `unresolved import crate::axes::Hl`. Dùng `Dummy2/Dummy8/Dummy256` của skeleton hoặc khai báo `axes!` bên trong `src/device/**`. File mới dưới `src/device/**` được chấp nhận. | P01; v131V_legal |
| Chữ ký `#[device]` được so NGUYÊN VĂN với skeleton. Đổi tên tham số không dùng thành `_cos` bị từ chối (6e48babf, 22b3e0a0). Thay vào đó dùng `let _ = cos;`. 0.8.1 dùng chữ ký `device: &mut Device`. | P02 |
| Grader tự biên dịch trong môi trường riêng: các knob ẩn (`SCHEDULER_MANUAL_ORDERING_PATH`, `SKIP_BEAM_SEARCH`, `TACTIC_*`, `DUMP_PE_PROGRAM`, `NPU_COMPILER_CONFIG_PATH`, …) chạy được ở local nhưng không bao giờ tới grader. | toolchain_081_notes §0.6; AUDIT L24 |
| Mỗi đội chỉ có 1 bài nộp đang chấm; độ trễ 13–40 phút. Arena: tối đa 2 job active mỗi tài khoản, timeout job tối đa 70 s. | P06; K107 |
| Lệnh cấm chia sẻ code công khai (upstream 1caa5d2, 09-22): nếu code chia sẻ công khai bị copy thì cả hai đội bị loại khỏi giải. Repo team TruongNVMM/MicroNPU2026 là private. Được phép đọc repo công khai của đội khác để học cơ chế (user đồng ý 09-24), với điều kiện không copy, tự hiện thực độc lập, và ghi provenance. | competition_rules §8; memory contest-code-sharing-ban, public-code-learning-policy |
| Board chính thức ≈ 0.99151–0.99152 × geomean theo baseline task (trước 09-23). Từ 09-23 dùng baseline board 250,514 / 404,633 / 3,703,473 (tái lập tới 4 chữ số thập phân). Ở v6, chính thức cao hơn Arena ~0.5 % (tỷ lệ theo kernel QKV 0.981, AO 0.979, FFN 1.016). AO chính thức ≈ trung bình lab-info − 0.39k. | LED §1; v7_notes; K40 |

### 6.1 Topology, bộ nhớ, giới hạn thanh ghi

| luật | bằng chứng / nguồn |
|---|---|
| Chip có 2 cluster × 4 PE × 64 slice = 256 slice/cluster. DM 512 KB/slice (32 MB/PE, 256 MB/chip). VRF 8 KiB/slice chia 2 bank; `vector_stash` giữ 1 KiB. TRF 64 KiB payload (80 KB vật lý). SPM mỗi PE 4 KB cho danh sách index gather. SRAM CPU của PE 3.5 MB riêng. | CONSOLIDATED U5; forum #451 |
| Operand VRF ≤ 8,192 B tính cả biên. Operand đúng 8 KiB fail "Cannot find evict target for Vrf" nếu còn VRF khác đang sống → chia đôi. Accumulator MAC 1024 phần tử. Packet contraction 32 hoặc 64 B. Output Collect đúng 32 B (pad `m![X # 8]` cho f32). Extent mỗi slice trong DM phải là bội của 8 B (30 bf16 = 60 B bị từ chối). DM allocation phải phủ đủ 256 slice. `Slice::SIZE` chỉ nhận {64, 128, 256}. | L-1, L-2, L-4; notes_early §3.2 |
| Page 16 MiB là đơn vị hazard: một DMA ghi vào page p phải chờ mọi reader của page p list trước nó, cả static lẫn HW. Allocator dùng lại page thấp rất hăng, nên các tile liên tiếp trùng địa chỉ và bị tuần tự hoá (FFN-003a/b: cùng địa chỉ 9,437,184). Mỗi lệnh nên ≤ 16 MiB/cluster (rngd-burn: một lệnh copy 64 MB chạy nửa tốc độ). | S-c; KR-5; U5 |
| HBM: 2 stack SK hynix HBM3, 32 channel × đơn vị 256 B, 5.2 Gbps → 1,331 GB/s = **665.6 B/tick** ở 2 GHz (nhân viên FuriosaAI, forum #435). Interleave: stack = bit 8, channel = bit 9–12 XOR-fold. Plateau đo được **651 B/tick** chip (325.5/cluster). Một cluster chạy một mình ~360–387 B/tick. Mô hình compiler giả định 680 B/tick. | CONSOLIDATED U2; X1 |
| Arena chạy NPU ở 2 GHz, governor "performance" trên cả 8 PE. 1 tick ≈ 0.5 ns; compiler lập lịch theo mô hình 1 GHz (`profiling_frequency_in_mhz: 1000`). Vì vậy độ dốc HW/static ~2.21 gồm ~2.0 do đổi đơn vị + ~10 % sai mô hình + ~0.51k cố định mỗi span. | ENV-1 (93569); TUCLOCK_external |
| Tensor unit: 8 DPE/slice. Fetch Main 32 B/cycle/slice, Sub 8 B. Kiểu contraction chỉ có i4/i5/i8/i9/f8e4m3/f8e5m2/bf16: KHÔNG có MAC FP4, không có HW MX-scale, KHÔNG có MAC f32. FP4 chỉ decode được qua bảng fetch-adapter của Main context. | U6; cast.rs:365-405 |

### 6.2 DMA: luật chi phí, engine, FIFO, alignment

| luật | bằng chứng / nguồn |
|---|---|
| **Mô hình static:** thời lượng DMA = 550 + bytes/rate. rate 1229 cho run liên tục, 1025 cho hàng stride không phải luỹ thừa 2, 615–855 cho load scale, 578–818 cho activation nhân bản. Mỗi DMA bảng 838, mỗi config switch 719. Đây CHỈ là mô hình static. | D1; M05 |
| **Luật request trên HW:** service ≈ 1,212 + 3.217 × (số request 256 B mỗi engine). Một request là một đơn vị 256 B mà piece chạm vào. Luật cho 637 B/tick chip = giới hạn pin: mỗi request, dù đủ hay thiếu, đều chiếm một slot pin 256 B. | FREECHECKS §3 (fit trên UpW/GateW/W3; W12, s0, s3, cfg, x/rms_w, QKV Q sai lệch trong ±0.13k) |
| **Luật F:** lệnh lớn (≥ 64 KiB/cluster) ≈ 1.6k + bytes_cl/323.5; lệnh nhỏ ≈ 1.0k; no-op 0 byte ≈ 0.17k. Luật tái lập lane trong ±10 % (QKV −2.7..+4.5 %, AO −4.6..−1.0 %, FFN +7.3..+8.5 %, phần dư của FFN là piece scale lệch). | CONSOLIDATED §0.3 (lane_out.txt) |
| **DMA nhỏ trên 0.8.1:** sàn ~0.8k + ~3.2 tick/request trên engine 0, kể cả khi đã xếp hàng hoàn hảo. Nối tiếp 1.10k (engine 0) / 1.24k (4 engine). Burst liền nhau 0.82–0.91k. Xếp sau một trọng số: lệnh đầu 0.57–0.64k, các lệnh sau ~1.0k. Trong kernel, load nhỏ đã xếp hàng tốn 0.76–1.08k. **Trên lane nối tiếp, một load nhỏ tốn ~1.0–1.2k ở mọi vị trí.** | fcf18 J2/J4; BACKWARD_QKV; qkv20 |
| (0.6) Số DMA nhỏ gần như miễn phí trên HW (độ dốc ~0), nhưng VỊ TRÍ và việc nhân bản thì không. Kernel chỉ có 1 DMA vẫn tốn 12–15k cố định. | M11 (71568) |
| Load HBM→DM có nhân bản ~130 B/cycle (240 KiB = 2.7k; 512 KiB = 4.0k); chi phí ≈ 1.2k + 6.4 cycle/KiB. Nhân bản một tensor 4 B ra 256 slice tốn 7.4k HW (giới hạn bởi số descriptor). Load nhân bản vào đích là tile còn chậm hơn nữa. Ở 0.8.1: x reload 983 KB ×32 dư +1.6–2.1k so với luật; cùng số byte nhưng không nhân bản thì đúng luật. | D9; QA10; K70 (J1 xr_*) |
| **Alignment:** piece căn 256 B và run dài trên từng slice là quan trọng. Down 1,920 B → 3,840 B aligned −23 %; seed scale aligned 57.6 KiB 18.8k so với 14.4 KiB 23.2k; oct256 7.0k so với dir800 9.38k / quad960 8.74–9.56k mỗi 1.84 MB. Luật chi phí đoán thiếu cho mọi lệnh lệch 0.9–1.8k. Compiler chỉ chọn piece 256 B khi run liên tục của mỗi slice là bội của 256 B; nếu không, nó lấy ước số bội-32 lớn nhất ≤ 1,024 (7,200 → 800, 28,800 → 960, 480 → 480, 1,920 → 960). | D4; F52; covM; FFN_GAP_external §4.1; CRITICAL_PATH_FFN s1.4 |
| **Các slice đọc đồng thời nên chạm vào các hàng HBM KỀ NHAU** (QKV hàng strided theo slice +20 % rate). Run 8 hàng trải trên các hàng của engine (far2) tốt hơn 4/16/32/64 hàng và tốt hơn hàng liên tục theo slice; khoảng hop DMN không quan trọng. Access pattern chứ không phải tensor quyết định rate (hoán đổi XQ/XO). Hai cluster đọc xen kẽ nửa của CÙNG các hàng thì mất ~5 % (penalty L-split). | Q49/Q50; E301/E303; qkvRate; P1; ffn19 |
| Rate tăng theo số slice sống: 465 / 259 / 150 B/tick cho 256 / 128 / 64 slice. Dồn một load lớn lên một cluster: +11.7k static / +23.6k HW. Layout 240/256 slice chậm ~10 % khi DMN mất cân bằng (r32). | K36; M07; FA19 |
| **Engine:** 8 engine tensor-DMA/chip, mỗi PE một engine (4 mỗi cluster). Một lệnh có một descriptor cho mỗi engine nó chạm vào (≤ 4). Descriptor xếp hàng theo từng engine trên HW (GL-1, J1 94294). Nhưng mọi DMA biểu diễn được đều chứa engine 0, và engine 0 gánh phần lớn nhất → engine 0 thành một lane nối tiếp duy nhất mang F của mọi lệnh. Đã quan sát 0 lần hoàn tất sai thứ tự trong 64,350 cặp. | U1; PROBES P2; AUDIT L1 |
| Vài DMA có thể cùng outstanding (dma_tag_id 0–31) nhưng chia chung băng thông. Khi đã bị băng thông giới hạn, issue sớm hơn vẫn land cùng lúc (FFN H6). Lệnh DMA thứ 2 của cùng một trọng số tốn ~+1.3k ngay cả trong lab chỉ-DMA; dispatch sớm chỉ giấu 15–38 % setup của lệnh lớn. | FA9; ao19 101926; triage E0 |
| DMA có đích là output HBM của kernel được hoãn wait tới cuối kernel (fire-and-forget), nên không chặn DMA kế. DMA không có consumer thì wait ở static end của chính nó (vì vậy lab chỉ-DMA luôn cho gap +2). | AUDIT L1 §6; PROBES P2.2 |
| Bảng FP4 của compiler: mỗi pass lookup có đúng MỘT DMA bảng (2 KB/cluster, static 838, HW 1.6–2.1k) và một StoTab trên Sub (static 1,293, HW 5.4k, dưới Main 7.0–8.98k). Không khử trùng (kể cả 2 lookup cùng input), không hoist khỏi loop. Mọi pass contraction/lookup thêm vào đều tốn thêm một bảng. | D16; AUDIT L3; 77/77 lịch |
| `Dma::Pcie` trong device code bị lower thành tensor DMA thường. Không có API chọn engine (sách có nói nhưng API không có). | AUDIT L1; API_RECIPES N6 |
| Một DMA nhỏ chạy đồng thời trên engine 0 trong lúc stream làm stream mất đúng bằng thời lượng của nó (sd1x +0.85k). Một switch Sub chạy cạnh DMA làm DMA chậm lại (x 2.2k so với 1.67k). DMA có bank priority thấp nhất nhưng không đo được stream trọng số chậm đi; riêng scale-class DMA trong kernel +0.3..+1.1k so với chỉ-DMA. | ao19; qkv17; U9 |

### 6.3 Scheduler static (cách list được dựng)

| luật | bằng chứng / nguồn |
|---|---|
| **S1.** DMA nằm trên MỘT hàng đợi static nối tiếp, được đặt theo THỨ TỰ CỦA CONSUMER (beam search trên DFG). Thứ tự chương trình không có tác dụng: hơn 10 lần kiểm chứng cho lịch byte-identical (FFN-002, FFN-003b, QKV-004..006, FFN-027, AO-E1a, source reorder của ao-qkv, qkvD2 po3, qkv12 biến thể d, ao13 p5a, d2s/d2l, ps*/pz*, k11p1/p2, bhv2). | G1 S1; §2–4 |
| **S2.** Khi hoà với begin static của một DMA, pass TU thắng và được list trước. **S3.** `wait_dma` đặt ngay trước consumer ĐẦU TIÊN của DMA. **S4.** Hệ quả: consumer nào bắt đầu được đúng lúc DMA của nó xong sẽ hoà với DMA kế → `run i, wait i, pass, run i+1`, nghĩa là DMA i+1 chỉ dispatch khi DMA i đã land. DMA kế chỉ được list trước wait nếu consumer bắt đầu chậm hơn hẳn. | G1 S2–S4; P5 |
| Load lớn đặt ALAP (xong đúng lúc contraction của nó bắt đầu) khi hàng đợi / page bị tranh chấp. Trên hàng đợi rảnh thì trọng số lại ASAP. Load nhỏ đặt ALAP ngay trước consumer đầu tiên; nếu vị trí đó đụng load lớn thì bị đẩy SỚM hơn, không bao giờ muộn hơn. Sub to_vrf không có input nào khác được đặt eager ngay sau DMA của nó. | S-g, S-h; AUDIT L24 Group E |
| Pass Main/Sub độc lập KHÔNG được đặt ASAP mà nằm cạnh consumer của nó ("touch pass" không kéo load lên được). Chuỗi pin chỉ ASAP nếu nó găng, ngược lại ALAP cạnh consumer. Thứ tự trọng số (K, Q, V) đi theo độ dài chuỗi consumer. | S-i, S-j; qkvD2 v2qa |
| **Công cụ steering duy nhất** là phụ thuộc dữ liệu: exact-zero pin (`+0·x`, `x·1.0`, BitAnd 0, Mode00 max(v,v)), memset pin (memset không input đặt ở đầu kernel), operand swap, identity-copy chain / ping-pong WAW, tách/gộp pass. Chỗ tiêm zero quyết định tất cả: zero lấy từ kết quả của một cây SAU thì sắp lại thứ tự mà không gate; zero lấy từ trọng số thì gate Main tới khi mọi trọng số land. | S-f; G1 S5; K82 |
| Pin exact-zero có nguồn zero phủ đủ 256 slice có thể khiến compiler sinh thêm một DMA hằng số 8 KB + wait ở head (qf1, q19z, qz, s2 zero15_512 cfg 719). | G4; qkv18; qkv19 |
| Memset pin chỉ có tác dụng ở đầu kernel: memset không input luôn nằm ở slot Sub rảnh đầu tiên và đẩy mọi pass Sub sau nó. Memset static = 266 + bytes/slice/8; trên HW chiếm Sub ~4× static. Các POST ở đuôi gắn với truyền DMA / đọc DRAM nên không dời được. POST readiness phía nhận của một sync cross-cluster được đặt ngay sau lần ghi cục bộ CUỐI vào buffer đích (aoY 1164 → 1329; aoYG → 2619). | MEMSET_PIN_AUDIT; V12-E784; K31, K75 |
| DramReuse được chèn khi không có barrier nào đã sắp đọc-HBM trước ghi-HBM của cùng một tensor (residual in/out). Nó gắn với tensor HBM, không gắn với kiểu view; bỏ được chỉ khi đọc residual được xếp trước một sync có sẵn. Đây là cơ chế giữ đúng thứ tự (prw bỏ nó → residual sai). | AUDIT L5; PR-S1; K87 |
| **ND regime:** khi không có pass decode thuần, compiler đưa mọi bảng lookup và mọi seed lên trước trọng số của nó. Khi có decode11, bảng decode chiếm head còn bảng Up/Gate nằm sau trọng số (Gate chờ 6.98k). Bảng của fused lookup luôn được list sau DMA scale Down CUỐI, nên gộp mọi scale Down vào một DMA là cách duy nhất để bảng land sớm (k15). | ffn18; BACKWARD_FFN; K101 |
| Tile nhỏ mới tạo mà consumer là một lookup không có input khác sẽ bị KÉO lên đầu DMA (0.8.1 v1). Tensor mới 14–28 MiB không được nạp trước khi ma trận trước xong. Muốn overlap, mọi tile của một ma trận phải là các tensor `to_dm` mới riêng biệt và phải được issue trước khi tiêu thụ. | S-n; S-d |
| Offset tile (`tile(r0)`) luôn lower thành code CPU `tidx_dynamic` lúc runtime (kể cả literal). Start tile nằm trong biến runtime làm DMA thành động và tuần tự. Compiler tách queue block ở 32 thanh ghi TUC = 8 run_dma mỗi block. | MOV "lit"; P5.1 |
| **Mô hình static không định giá:** layout HBM (mọi layout coi như cùng băng thông), issue in-order, sync host/cluster (600 phẳng), chi phí cố định mỗi lệnh. Pass Sub VE bị định giá thiếu ~4×; gather switch ~2.2×; endpoint một slice thiếu 4–5×; StoTab 1.3k static so với 5.4–8.2k HW. Riêng store partial-unit (RMW) được định giá khớp HW. | K3; notes_v10v12 §3.1 |



### 6.4 Thực thi trên HW (TUC, context, pass, contraction, StoTab)

| luật | bằng chứng / nguồn |
|---|---|
| **Mỗi cluster có MỘT list lệnh TUC in-order**, do firmware (PERT) của PE lead nạp vào. List chứa DMA dispatch, `wait_dma`, launch pass TU (Main + Sub dùng chung một VE), POST/WAIT cross-cluster, và block "CODE" CPU của PE. Mỗi context (Main, Sub, DMA) cũng in-order. Một lệnh TU dispatch khi lệnh trước của cùng context xong. Một DMA issue khi mọi lệnh TU list trước nó đã dispatch. Hàng đợi dừng ở lệnh TU của một context đang BẬN cho tới khi pass hiện tại xong: đây là head-of-line blocking. | H2; U4; ao9 hs6 |
| DMA list ngay sau một pass Sub VRF chỉ issue khi pass đó HOÀN TẤT (StoTrf thì issue lúc dispatch). Exec TU nào list trước một DMA cũng chặn dispatch của DMA đó tới khi exec xong (wsm2). Mọi pass Sub list trước một lần chạy DMA đều gate nó (FS trước khi nạp program) (g10z). DMA list sau một contraction issue ~290–310 tick sau khi contraction dispatch. | ao9 lab7; K86; qkv17; notes_v10v12 §3.1 |
| Fence TUC: X(0x22000000) chờ Main idle, X(0x23000000) chờ Sub idle. StoTab = R2(0x16000000), memset = GEN(0x10000000). Mọi lệnh Sub / preload program Sub sau một memset đều bị fence X23. Đây là dạng thật của các thất bại hs2/hs6 và qM1. | notes_v10v12 §3.2 |
| Runtime PE (tuc.o đã dịch ngược): CPU dành từng queue block trên hàng đợi TUC 64 entry (≤ 62 lệnh), 64 thanh ghi TUC, 64 slot descriptor DMA. Đầu hàng đợi chỉ tiến khi một commit block (≥ 16 lệnh, hoặc một core interrupt như control_tu_clock) xong. Scatter động = DMA fetch index + DMA scatter = 8 bản descriptor + 4 danh sách gián tiếp. Mỗi khe TUC ~400 cycle trên QKV c1 là một lần nạp program TU 1,320 B bị lộ ra. | qkv12 |
| POST/WAIT là lệnh của PE core. PE drain hết hàng đợi TUC trước chúng, nên pass list trước một WAIT sẽ làm trễ đuôi, trừ khi vùng đó vốn đã phải chờ sync lâu hơn (rzW +1.75k). | K77, K81 |
| Pass TU nhỏ tốn ~0.5–1.2k trên HW bất kể kích thước static (pass Main VE 0.3–1.3k + ~0.4k khe nạp program). Sub to_vrf 0.35–1.0k: 0.67–1.07k nếu program của nó nạp trong cùng queue block, 0.37k nếu đã nạp trước. Pass chế độ cặp (begin_interleaved) chính xác nhưng tốn ~2× pass thường. Pass switch 1.1–1.7k. VE trên packet 1 giá trị/flit chậm ~20×. | K84; T12; rgb2 |
| Contraction HW ≈ **0.51k + 2.21 × static** trên cả 26 span; không có ~2k cố định mỗi lần launch contraction. Thời lượng danh định: V nhỏ 1.21–1.30k, V/K 2.50–2.66k, Q 3.50–3.54k, fp4 LUT 6 hàng 5.24k. Nửa contraction còn 74 % bản đầy đủ (~1.2k cố định mỗi contraction). f8×f8 không bảng nhanh hơn contraction fuse bảng ~3.5×. | FREECHECKS §5; TUCLOCK; q19ks; U6 |
| Rate lookup ~5.5 giá trị/tick/slice (Up/Gate); decode 5.69; fused4 có VE 4.73. Không phụ thuộc hình dạng packet. Giới hạn nằm ở scratchpad của bảng (SUY LUẬN). | GAP_FFN; ffn8 lk64/lk128; FFN_GAP_external |
| **Luật StoTab:** một StoTab chặn mọi pass VE chạy đồng thời. Contraction chồng lên một StoTab không thể xong trước StoTab đó. Pass decode thuần (không có stage VE) thì miễn nhiễm. Contraction sống lâu hơn StoTab thì không bị chậm. VE miễn phí bên trong một fused lookup pass. | K100; BACKWARD_FFN; ffn20 |
| Main và Sub chạy đồng thời nhưng dùng chung VE. Main có priority bank DM: Sub store dưới Main chạy ~50 % tốc độ, Main không chậm. rngd-burn nói "TRF store tuần tự với contraction", nhưng điều này KHÔNG xảy ra ở kernel của ta vì store luôn được sắp dữ liệu trước contraction. Main pass không overlap với một Sub StoTrf đang chạy; Main VE pass chờ Sub StoVrf đang chạy. Một pass Main kết thúc bằng `to_vrf(&mut device.sub)` cũng chiếm Sub context; VRF ghi từ Main tốn 2.03k HW so với Sub 1.10k. | GL-3 (FREECHECKS §1); qkv8 x4g; E103qm |
| **Luật nạp TRF:** Sub `to_trf` nuôi bởi CustomBroadcast (kể cả ring_size 1) chạy như TuExec KHÔNG CHẶN. StoTrf thường, switch Transpose{1,1} hay nạp TRF qua ring Broadcast01 đều là StoTrf CHẶN issue in-order. CustomBroadcast kèm thêm config DMA 719 static (1.1–1.6k HW). Main `to_trf` (743 static) tránh được cả hai. Sub fetch `read_size` 8 B nên Main `to_trf` ngắn hơn 2.4× về static. | K99 (ffn15 94088/94195/94201/94268); api081 P4 |
| Mỗi contraction có một core interrupt `control_tu_clock` ngay trước EXEC. "Slow contraction" (+1.0..2.5k, TUC đứng 2.1–3.5k) chỉ có ở trace (backlog PERT), KHÔNG có ở INFO. | TUCLOCK (J4 99898) |
| Head: từ lúc Task bắt đầu tới DMA đầu tiên mất 0.10–0.48k (code CoreImp tidx của CPU PE). PE CPU dựng queue block 8 DMA một lần (khe lane 1.1k ở head của lab chỉ-DMA). Launch cold +1.5k (không tính điểm). | fcf18 F10–F11 |
| Dính cặp 2 cluster: kernel ≈ max(đường c0, skew + đường c1 + latency barrier). c1 chạy mọi pass TU chỉ dành cho c0 trên phần pad, với chi phí đầy đủ (SPMD twin). DMA chỉ dành cho c0 được phát trên c1 như một no-op 0.16k. Không construct nào giữ được pass ngoài cluster 1. | notes_early §3.4; fcf18; AO_R3 Q2 |

### 6.5 Sync, skew, barrier

| luật | bằng chứng / nguồn |
|---|---|
| (0.6) Sync host (span `Cluster`, Core static 600/708) được chèn sau DmaStore của tensor HBM sẽ reload, sau load cuối của tensor HBM sẽ ghi, và ở cuối kernel. Sync ĐẦU TIÊN của một launch tốn 2–30k ngẫu nhiên (cold hơn thì dài hơn); đây là nguồn nhiễu chính thời 0.6. Round trip HBM thêm giữa kernel ~2.5k HW. | H4; A23 |
| (0.8.1) Mọi sync là barrier phía device: 1.2–2.7k HW. ExplicitSync (gather xong) kết thúc ~1.2–2.0k sau DMA CUỐI issue trước POST. DramReuse kết thúc ~2.25k sau khi residual land. | H7; notes_early §3.5 |
| Hai bộ đếm cluster chạy chung một đồng hồ, offset cố định mỗi phiên, nên đo được skew từng launch (print SB). **Launch skew**: c1 bắt đầu sau c0 ~0.95–1.5k (median warm, max ~3.0k), ngẫu nhiên theo launch và theo phiên; không phụ thuộc kích thước program / số DMA / số sync; +~20 tick mỗi word đối số launch. Kernel không điều khiển được. | fcf18 J1–J3 |
| **Luật release của barrier:** waiter tới ≥ ~0.5–0.75k sau POST của bên kia thì qua trong 2–3 tick. Waiter tới sớm được thả ~0.65k sau khi bên kia tới, dù sớm tới đâu. **Dead zone:** tới 0–0.5k sau bên kia thì ngủ 0.9–1.0k (tệ hơn tới sớm). → Hoặc đặt ≥ ~0.6k việc độc lập trước một WAIT găng, hoặc không đặt gì. Một cửa sổ POST→WAIT ≥ s + 0.7k (~1.7k) che được một sync giữa kernel. | fcf18 PR-1, h1 |
| **Thuế skew** ở sync đối xứng găng ~2k mỗi kernel: QKV 2.24k (cuối), FFN 2.2–2.5k (y hop, trace), AO 1.74k ở WAIT2 + 0.9k ở cuối (trace). Ở INFO, AO không có dead zone ở cuối vì c1 luôn xong trước. Stream của c1 chậm 0.2–1.9k mỗi 3.93 MB vì tranh chấp với lệnh kế của c0. | fcf18; ao20 |
| Bất đối xứng mịn, c0 nặng hơn, có lợi −0.97 / −2.42 / −1.26 / +0.58k cho E = 393 KB / 786 KB / 1.18 MB / 1.97 MB. Nhưng độ mịn của kernel ta quá thô: QKV head 1.97 MB, hàng L của FFN 2.95 MB, nhóm AO ~0.98 MB. Hàng có packing và tile cluster đều bị từ chối. | fcf18 J3; ao18 |
| Ở mức INFO: QKV có cửa sổ c0 = cửa sổ riêng của c1 + 2.0–2.3k; lợi TU sau V trên c1 tới được metric ~40 %. AO bị c0 giới hạn khi s < ~2.3k, cắt đường c0 được ~1:1 (tới ~0.8k). Barrier sync 2 của V18 AO không khớp một hằng số L duy nhất. c1 trả một round trip cho chính các POST của nó; thêm một POST trước WAIT trên c1 tốn một round trip. | qkv19; ao20; ao19 check 0 |

### 6.6 Tensor unit, vector engine, switch

| luật | bằng chứng / nguồn |
|---|---|
| **Thứ tự stage của VE** (isc.md:62-75): Tag/unzip → Logic → Fxp → FxpToFp → Narrow → Float (unary: Exp, NegExp, **Sqrt**, Tanh, Sigmoid, Erf, Log, Sin, Cos) → **IntraSliceReduce** → FpDiv → Widen → FpToFxp → Clip (Min, Max, AbsMin, AbsMax, Add) → Filter (todo!). Không có unary nào sau reduce, nên RMSNorm cần ≥ 3 pass (Σy², sqrt → VRF, normalize), hoặc 2 pass với mẹo SQF (Mode021 FmaF(&vrf, 0.0) đưa VRF vào stream + Sqrt + FpDiv Mode10 trên stash). | API_RECIPES F2; E311; q19n2 |
| Mỗi ALU dùng tối đa MỘT lần mỗi pass. Cụm Float có 5 ALU: FpFma, FpFpu (Sqrt, DivF), FpExp, FpMul0, FpMul1. Tối đa 2 MulF ("13 is not available for op Binary(MulF)"). Hai AddF không được ("10 is not available"). AddF và FmaF tranh cùng FpFma. Inter-slice reducer là InterFirst HOẶC IntraFirst, không cả hai. VE chỉ i32/f32. Một op không nhận được 2 operand VRF. Mỗi pass có đúng một lần làm tròn bf16, ở output. Không pass nào cho 2 output. Không có hoán vị lane trong VE (rotate-half phải qua stream fetch thứ 2). | F2; FA20 gf; fa1 |
| Sau `vector_intra_slice_reduce` chỉ còn `vector_fp_div` / `vector_clip`. Sau `vector_inter_slice_reduce` chỉ còn `vector_clip` (không có phép nhân), nên per-channel scale không bao giờ fuse được vào contraction. `Clip` không chuyển sang `Narrow` được, nên Clip(Add, EPS) phải viết thành AddF(EPS). | T16 |
| `vector_inter_slice_reduce` với trục dummy output SỐNG (`m![.., Dummy256 % 4]`) để kết quả ở mọi slice của nhóm, tức broadcast miễn phí. 0.8.1 định giá reduce trên 8 slice sống pad thành `1 # 32` ở ~2.3k static / ~4.5k HW. Reduce trên view regroup KHÔNG pad thì rẻ (trục có pad đắt ~5.8×). "VRU reduce axes must be innermost". | T6, T7 |
| **Ngữ nghĩa switch:** `Broadcast1{slice1, slice0}` là ALL-GATHER dọc slice1 (không phải replicate 1→N). `TransposedBroadcast1{slice1, slice0}`: slice1 là trục NGOÀI của cặp nguồn. Config có tên (Broadcast1/01, Transpose, InterTranspose, TransposedBroadcast1) không cần config DMA. CustomBroadcast cần config DMA snoop-bitmap 16 KiB (719 static, 1.1–1.4k HW) và chiếm cả DMA lẫn Sub context. Switch không broadcast được vào mapping có pad bên trong ("snoop bitmap disagrees"). | T1–T4; U7 |
| **Chi phí switch:** static ≈ flits × slice1 × slice0 + 263, nên ring-2 stride 4 đắt bằng ring-8. CustomBroadcast ~ring_size cycle mỗi bước time (ring-4 ~5, ring-64 ~68, ring-256 ~282). Mốc HW: ring-256 17.1k (0.6) / 29.5k; XG2 ring-32 3.64k; ring-64 Broadcast1 fan-out 30 KB ~1.7k trên Sub; InterTranspose ring-8 14.7k trên Main / 68.6k trên Sub; `Broadcast1{256}` 1→256 = 62k Main. | K18, K37; U7 |
| **Compile KHÔNG kiểm switch config:** config sai vẫn compile, vẫn có lịch static, rồi TREO device (71578). Pre-flight bằng host `furiosa_opt_lower::config_switch` (swcheck, vài giây) với cả ca âm tính, hoặc CPU emulator. | T19; M14 |
| Transpose Engine: ≤ 4 hàng input mỗi packet cho 16 bit (2 cho 32 bit, "f32 Transpose merges at most 2 rows"). Không phân rã được trục bị giới hạn. Trục tile sau `tile::<Axis,N,..>` chuẩn hoá thành `Axis = N # M`. | T9; bvs2 |
| `fetch_table_lookup` chỉ chạy trên Main, chỉ f4→f8 và f8→bf16 (không có đích f32), không nhận đối số bảng. Không có fetch cast f8→bf16. FetchCast f8e4m3→f32 và f32→bf16 có sẵn (không cần VE). `.cast::<bf16>()` đưa pass vào context VE. Contraction float chỉ ghép cùng kiểu. `LaneMode::Sequential` + PackSize 2 gói 4 block sum mỗi flit (A2; nhóm Down nhanh 2×). Luật contraction static: T ≈ 280 + max(flit fetch, packet MAC, flit output). | T8; ffn18; FA16 |

### 6.7 Danh sách lowering bị từ chối / ICE (device code, 0.8.1 trừ khi ghi khác)

| construct | lỗi | nguồn |
|---|---|---|
| `Option<DmTensor>`, struct literal, device struct, array literal (0.6) | ICE "not yet implemented … Option::Some"; "expected ConstKind::Value"; "mir: Unknown type: std::mem::MaybeUninit" | R1–R3 |
| tile trên trục Cluster (`cluster_tile`), `slice_tile` làm đích hoặc nguồn DMA | visa "a resolved index step counts in-slice cells, but this tile targets Cluster/Partitioning; tiling a non-in-slice axis is not lowered yet" | ao18; AUDIT L1 |
| index tile hàng có packing / pad | "tile index with non-empty packing is not supported" | ao18 g1a/g2a |
| số hàng tile không chia hết cho số hàng của digit; tile 7 hàng trên 15 | "cannot factorize common axis H"; "PROG-719: 15 has no divisor in 7..14" | L-9; FA21; k11mmce7 |
| `HbmTensorViewMut::tile`, view mutable đã reshape; `DmTensorViewMut::reshape` gán qua let | "mir: expected a HbmTensorViewMut / DmTensorViewMut" (reshape theo giá trị thì được) | qkv18 qs1; ao_dra5; P5 |
| `&raw const` alias đọc | "mir: unsupported &raw const" | qkv16 rip |
| output reducer có slice pad pass-through; LIR với factor ngoài có pad; DM→DM vào tile 118/120 | visa / LIR từ chối; "cannot factorize common axis H" | triage expr |
| DMA piece 2 byte (per-channel vào layout contraction) | "strides(...) is not aligned by 8" / "visa: while lowering Dma" | qkv20 t22p; Q65 |
| commit_trim output < 8 B | "output packet must be one of [8, 16, 24, 32] bytes" | bvs |
| switch trên Main commit vào buffer DMA-written trên slice có pad | "owner mismatch" | bqvr/bqvs |
| commit_cast f32→bf16 có packing | "CommitUnit verify failed: shape packing should be done with no conversion"; "No valid fetch size found: pv=4 f8->f32" | gcc |
| pass Sub stream DM tile có op vector; một số pass VE scale/zero trên Sub; regroup switch + transpose trên Sub | LIR ICE "no entry found for key" (sfr_impl.rs:868) | bfnr; nd11e9s/nd11gzs; rga |
| `vector_clip` trên Sub; `zero_q` trên Sub; `.transpose()` trong đường commit; commit_view vào tile trên Sub | lir ICE; "visa: while lowering TensorUnit" | r19/r20; T15 |
| chu trình phụ thuộc trong đồ thị op (pin Mode00 trong ms, fa1b/c) | "operator schedule base graph has a cycle" | rzM, msr, fa1b |
| pass VE vượt quá số slot | "this node needs more slots than a VE pass has" | bzr lần compile đầu |
| gate chuỗi 2/4 load nhỏ cùng lúc | "Cannot find evict target for Vrf" | BACKWARD_QKV r1a/r1b |
| lookup trong vòng `#[unroll]` | LIR ICE "expected at least one node" | AUDIT L3 |
| seed ramp gate theo trọng số K (h1k) | visa V1-Modulo "Insufficient resources … Sram AllocRequest size 0" | qkv14 |
| `Slice::SIZE = 64`, `Cluster::SIZE = 1` | "slice/cluster extent … does not match the device config" | AUDIT L3/L5 |
| Main `to_vrf` thẳng từ `collect` (không có chuỗi VE) | **compile được nhưng TREO device** (78998, 99768) | FA10; wsm |
| SwitchConfig sai | **compile được nhưng TREO device** (71578) | A22 |
| API todo!() | `fetch_mask`, `commit_valid_count_pack`, `dma_scatter_unscaled`, `hbm_cluster_shuffle`, `vector_filter`; ghi có index qua slice_tile/cluster_tile | API_RECIPES F3 |
| địa chỉ trong alias legacy `HbmTensor::to_dm` 3 đối số / `to_dm_at` | compile được, địa chỉ bị BỎ QUA im lặng | AUDIT L1 x3/xat |
| `to_dm_view` nạp lại cùng buffer (0.6); `to_dm_view` vào tile của một DM tensor (0.8.1) | chạy nhưng SAI KẾT QUẢ (34133; 80296) | L-12; AA18 |
| trục cluster đặt theo trục mà tensor không có | DMA nhân bản thay vì chia → rác im lặng ở nửa output (35403, 35421, 35700: "58 % within tol, max error 100 %") | L-5 |
| DMA vào tile trong slice của tensor lớn hơn | chép TOÀN BỘ extent có pad → WAW, bị tuần tự | P5.1 |

### 6.8 Số học

| luật | bằng chứng / nguồn |
|---|---|
| Tách activation FP8 hi/lo (`hi = f8(kx)`, `lo = f8(kx − hi)`) là chính xác (sim host 100 % trong tol; QKV-008 bit-identical trên device với lo ≠ 0). Một thành phần FP8 là KHÔNG đủ (FFN 98.93 % FAIL; QKV chỉ "pass" nhờ fixture co về code 448.0). | N2, N3 |
| **Scale KHÔNG triệt tiêu qua RMSNorm khi có EPS:** normalize(c·v) ≠ normalize(v) khi EPS (1e-6) cỡ bằng mean-square. Xoá global scale FFN → chỉ còn 70.55 % so với kỳ vọng (41489, 41499; mất 2 job). 1/k (k = 128) của AO thì triệt tiêu được vì mean square >> EPS. Luật: kiểm độ lớn so với EPS; nếu không xoá được thì GỘP hằng số vào các consumer (vẫn bỏ được một pass). Không bao giờ dời một scale qua một norm. | N1; EPILOGUE_DESIGN R1 |
| `SubF` với operand VRF tính `vrf − stream` trên device → dùng `AddF(−hi)` (gốc rễ thất bại của FFN-016c). | v11 mlp.rs:52-53 |
| Khớp đúng các điểm làm tròn của reference thì cho kết quả chính xác (AO-004 = 0.0). Bỏ một lần làm tròn bf16 trung gian có thể làm CHÍNH XÁC hơn (V scale fold: v max 0.03125 → 0.01562). Mỗi pass VE làm tròn đúng một lần ở output; không có RNE giữa chừng. | N4; F2 |
| Cộng EPS vào từng phần tử trước reduce = mean + EPS (dùng khi `vector_clip` không chạy được trên Sub). RMSNorm phải là toàn cục qua 2 cluster; RMS 2 nhóm là một hàm khác. Partial cục bộ chỉ chính xác nếu EPS được cộng một lần (EPS·128² theo từng số hạng; EPS/16 mỗi partial trong FFN SQF). | T15; A26; notes_v10v12 §3.3 |
| Mọi KEEP từ V13 tới V17.3 đều bit-identical (max/mean abs d từng run giống hệt; emulator 0/3840). Viết lại chính xác (x·1.0, x + 0.0, bits AND 0, relayout thuần dữ liệu, cùng chuỗi op f32) là không gian thiết kế an toàn. Pad-slot partial ép +0 bitwise trước mọi op FP (NaN không lọt). | V13–V17.3 RESULT; v131V_legal |
| Gate số học FFN nội bộ (0.00391 / 0.00781 / 0.00781; mean ≤ 1.10×) chặt hơn tolerance của cuộc thi. bf16 block sums cho Down → r2 max abs d 0.01172 (FAIL gate nội bộ, PASS cuộc thi ~2×). | ffn18 100856 |
| cos/sin trên chip cần range reduction `fxp_to_fp(24)(turns)·(−2π·128) + angle` trong [−π, π]; tính góc ngây thơ ra non-finite. Ramp index dựng bằng nhân đôi; `one = sum(0·k + 1/256) = 1.0` dùng làm gate. Fixture đã đảo dấu trước nửa thấp của sin; device làm half-swap thường. | notes_early §3.6 |
| CPU emulator không đáng tin về số học (v11 QKV k/v FAIL 16 %/9 % trên emulator nhưng PASS trên HW; FFN cc1c max abs d 1.043 trên emulator). Chỉ dùng emulator để kiểm switch config và bit-identity so với base. Sim host (`scripts/fixture_prng.py` + `generate_references.py`, torch float8_e4m3fn) tái lập đúng fixture, dùng để quyết định các câu hỏi về độ chính xác trước khi tốn Arena. | T19; K102; N9 |

### 6.9 Các khẳng định mô hình đã bị sửa (bài học đắt nhất)

| khẳng định trước | bị sửa bởi | cách hiểu hiện nay |
|---|---|---|
| HW ≈ 2.2–2.3 × static, xếp hạng theo static | ≥ 5 thiết kế QKV "static tốt hơn" thua trên HW; đuôi AO static ngược chiều; x15k2 static lỗ / INFO lãi | static chỉ xếp hạng được trong cùng một họ cấu trúc |
| event model 233 / event | cặp FFN-025 (−2,316 so với −9,044 dự đoán) | dùng c × static trong cùng job, rồi lab INFO cùng job |
| "mỗi DMA op tốn 550" → cắt số DMA | 0.6: độ dốc ~0; 0.8.1: sàn ~0.8k + 3.2/request, burst chỉ tiết kiệm 0.33–0.42k | vị trí, nhân bản, và F của lệnh lớn mới là cái đáng tiền |
| DMA engine là một hàng đợi nối tiếp (0.6) / "8 transfer song song" (sách) | có hàng đợi descriptor theo từng engine trên HW, nhưng mọi lệnh đều chứa engine 0 | trên thực tế là một lane |
| "mọi DMA dùng engine 0" là giới hạn HW | runtime/HW hỗ trợ tập con bất kỳ | giới hạn nằm ở compiler (không có offset, tile không được lower) |
| trọng số ALAP là luật | trên hàng đợi rảnh thì ASAP | "ALAP" là hệ quả của tranh chấp hàng đợi/page |
| tile view làm trễ consumer +79 (P5.2b) | Group A (9 arm) không đúng | view op không dời start static |
| rngd-burn: ~2k cố định mỗi contraction; TRF store tuần tự với contraction | contraction ≈ 0.51k + 2.21 × static; store đã được sắp dữ liệu trước | sai với kernel của ta |
| sync host đầu tiên 2–30k là nguồn nhiễu (0.6) | 0.8.1 sync là barrier device 1.2–2.7k | nay dùng luật skew/dead-zone của fcf18 |
| latency sync l ~0.8–1.05k, "thả cùng lúc" | mô hình penalty (fast / early / dead zone) | fcf18 PR-1 |
| chia cluster bất đối xứng đã đóng vì độ mịn (U8, OFF-7) | bất đối xứng mịn có lợi −1..−2.4k | vẫn bị chặn vì độ mịn thô |
| TU-clock slow contraction / "cho Q ≥ 1k Main idle" / "cửa sổ V bị Main giới hạn" | artefact trace | không có ở INFO |
| "plateau ~620–645 B/c là giới hạn HW"; leader có một cơ chế DMA nhanh hơn | stream đã ở 96–98 % pin rate | leader cắt thời gian không-stream |
| Q-first đã chết (qsim +2.7k/+9.6k; V14 static kill) | qf7 −524 (SE 84) trên thân V16 | kill static phụ thuộc thân kernel |
| "head-per-slice là tệ" (E8) | qkv8 h4 −2.55k | không đúng |
| "barrier sau to_hbm = 5–20k" | +2.5k cho một round trip | 5–20k là sync đầu tiên |
| PCIe DMA là engine riêng dùng được cho d2d đuôi | lower thành tensor DMA | dead end |
| mô hình HBM-channel và đặt bảng của cloud | bị đo bác (KR-1, dh16) | triage trước khi build |
| "board giữ điểm cao nhất" | xếp hạng cuối chạy lại bài nộp CUỐI | M14 |
| "FFN bị lane DMA giới hạn: tiết kiệm Main không tới HW" | chỉ đúng cho thay đổi đơn lẻ; stack (fxv) thắng | luật stack |
| "đạt không được; tốt nhất ~9.07–9.10" (09-26 09:05) | V17.2 ~9.110, V17.3 ~9.19 nhờ thay đổi cấu trúc FFN | trần mô hình phải hoài nghi |
| "honest ceiling 6.45×" (0.6) | đã đạt 7.08 / 7.43 / 8.4+ / 9.19 | trần static không phải trần HW |
| "M1 đóng hẳn" (QKV-030) | V8-MERGE full-K 51,098 / 117k | vấn đề là đường của activation |
| "on-device cos/sin bất khả" | v11 range reduction | sai |
| "FFN FP8 một lookup đã đóng; lỗi là do bug trục Dummy2" | v11 FFN FP8 hi/lo chạy được | cách ta hiện thực bị lỗi (dấu + lỗi khác), không phải hướng đi |
| "một tile mỗi ma trận thua" (FFN-032/033) | v11 dùng một tile 30 hàng cho Up/Gate | chỉ đúng khi Main bị lookup giới hạn |
| "Down K-halves đã đóng" (0.6 F54) | KH trên 0.8.1 −10.1k | thay đổi toolchain/runtime mở lại hướng này |
| "đầu AO ~4k với mọi fan-out" (0.6) | 0.8.1 v4b seed all-gather: issue trọng số 3.9k → 2.1k | phụ thuộc cơ chế |
| "cổng head AO không biểu diễn được" (aoS2 / aoS2R) | aoYG memset-pin −756 | POST dời được bằng cách ghi cục bộ |
| "s15 (15 hàng scale một DMA) thua +4.7k" (ffn9) | k15 −1,014 trên thân CC-1 | phụ thuộc thân kernel |
| "E311 SQF thua" (V10) | q19n2 −264 trên thân qf7 | phụ thuộc thân kernel |
| "OCT-b NULL" (fxv) | octk −1,892 trên k15 (thêm ~3k slack) | phụ thuộc thân kernel |

### 6.10 Trần hiệu năng (cập nhật 09-26)

| tầng | giả định | QKV / AO / FFN | điểm | nguồn |
|---|---|---|---|---|
| A stream thuần | trọng số ở R/2 | 48.3k / 24.2k / 135.9k (152.9k tính cả scale) | 13.32× (12.81× có scale) | CONSOLIDATED B2 |
| B sàn vật lý FIFO | mọi lệnh trước byte cuối tốn F; contraction sau stream + exchange + s + l | 63.9k / 32.7k / 179.2k | **10.01×** @651 (10.18× @665.6) | same |
| C sàn bằng chứng | chi phí từng lớp tốt nhất đã đo trên 0.8.1 | 64.0k / 34.1k / 196.6k | 9.56× (9.76× với scale aligned) | same |
| B' nếu GL-1 dùng được | lệnh nhỏ trên một engine xoay vòng | 58.7k / 31.9k / 172.4k | 10.51× | same |
| BACKWARD (09-26) | lane QKV hoàn hảo; AO không lãng phí; FFN họ ND | QKV 64.6–66.2k; AO ~34.2–34.4k chính thức (sàn 6.4k sau byte cuối); FFN ~192.3–194k trace | – | analysis/v18/BACKWARD_*.md |
| thực tế V17.3 | – | 68.8–69.2k / 35.4–35.6k / 199.3–199.5k | 9.14–9.17 NH, ~9.19 ước | §1 |
| board #1 (09-26) | – | 63,704 / 31,966 / 190,026 | 9.8995 | INTEL |

Nhận xét:
- Audit kết luận không điều khiển nào từ source chấm điểm gỡ được các giới hạn engine-0 / wait / LUT / sync. Do đó trần ~10.0–10.2× của 0.8.1 vẫn đứng (memory furiosa-opt-081-audit-limits).
- Leader đã thấp hơn tầng C của ta ở cả 3 kernel. Như vậy họ có ít nhất một chi phí theo lớp tốt hơn ở mỗi kernel.
- Không cấu trúc nào biểu diễn được mà ta tìm ra đạt được AO 31,966 hay QKV 63.7k với các primitive đã đo.

---


## 7. Phương pháp đo và quy trình

### 7.1 Metric và nhiễu

- **Metric của grader.** Median của 3 run có seed ở mức INFO. Run 0 luôn cold, là run chậm nhất ở mọi job normal-harness, nên median ≈ max(run1, run2). Vì vậy khi đánh giá một arm phải xem **E[max2] cùng với mean**. Tối ưu run cold thì không được gì. Nguồn: V10 phase 1, K27.
- **Nhiễu thời 0.6.**
  - Cùng source mà dải QKV 11 %, AO 46 %, FFN 4.2 %. Cùng binary ở 2 job lệch tới 9.8 %. Trong một job ~0.5 %.
  - Một lần chạy chỉ phân giải được ≥ 2 % FFN, ≥ 5 % QKV, và không gì dưới ~12 % ở AO.
  - Luật L7: một cặp đo không phân giải được < 4 % trên QKV. Ví dụ E8 cho dấu ngược nhau ở 2 cặp.
  - FFN là thước đo tốc độ máy (CV 1.0–1.3 %). Nguồn: M02, M08, MEASUREMENT_VARIANCE.
- **Nhiễu thời 0.8.1.**
  - Trôi giữa các job với code byte-identical: QKV ~0.2–0.8k, AO ~1k, FFN ~1–1.5k. Trung bình 4 job NH lệch ±0.02–0.05 điểm chỉ vì trôi. Ví dụ: V17.0 QKV +307 / AO +294 so với V16; V17.1 FFN +728; V17.2 AO −223 / FFN −772; V15.2 có một job AO 36,573 → 8.8844.
  - **Median NH của các job khác nhau KHÔNG phải là bằng chứng.** Nguồn: K2, int17–int173.
- **Noise yardstick.** SE block-paired của một lab: QKV 0.09–0.22k, AO 0.11–0.25k, FFN 0.07–0.35k. Lab 24 warm/arm có Welch SE 99–149. Lợi dưới ~2.5 SE của một lab (QKV/AO < 0.4k, FFN < 0.6k) thì không chạy riêng. Ý tưởng có trần < ~0.2k không bao giờ qua được luật KEEP. Nguồn: V17_PLAN §1, K29.
- **Outlier.**
  - Host-sync ~+19–20k: ~1 lần mỗi lab 72 launch.
  - Launch skew cực đoan (c1 trễ ~49k): 5 / 2,965 launch.
  - Freeze AO ~1/36–1/56 launch warm.
  - Cách xử lý: screen 3.5-MAD, chỉ áp cho lab có outlier > 10k. Báo cả số đã screen lẫn chưa screen. Loại outlier chỉ khi có cơ chế giải thích từ span.
- **Trace so với INFO.**
  - Trace tốn thêm ~0.6–0.7k warm cho QKV, ~0.9–1.0k cho AO, ~0.5–1.5k cho FFN; cold +2.3–3.3k.
  - Trace sinh artefact riêng: TU-clock slow contraction, TUC hold, dead zone cuối kernel AO. Có khi dấu còn ngược: x15k2 trace +1,115 so với INFO −570.
  - **Chỉ quyết định trên arm INFO**; arm trace chỉ dùng để đọc span.
  - Cây làm việc mặc định trace, snapshot nộp bài mặc định info. Harness đọc `FURIOSA_OPT_PROFILE` ở mỗi launch, nên chọn mức được theo từng arm.
  - Mức `debug` chỉ ghi span Task và tốn bằng INFO.

### 7.2 Thiết kế lab (Arena)

- **Lab cùng job, xoay vòng block.** Mỗi block chạy tất cả arm liên tiếp, và thứ tự arm được xoay giữa các block. Lý do là "position bias": arm đứng đầu block chậm hơn 0.1–0.3k (xem patch_harness2.py, patch_harness3.py). Cần ≥ 12–16 launch warm mỗi arm. Báo riêng c0 và c1.
  - Dạng block thường dùng: `[trace k0 (cold), trace k1, info k2, trace k3]`, hoặc `[trace, info, info, trace]`, hoặc toàn info.
  - Mỗi launch trong lab đều cư xử như run 0 cold (+6.6k FFN so với harness thường), nên lab chỉ dùng để so sánh TƯƠNG ĐỐI.
- **Giới hạn Arena.**
  - Mỗi job tối đa 70 s. Lab 9 arm × 6 × 3 có 2 arm trace đã TIMEOUT sau 136/162 launch (95840).
  - Giữ lab ≤ ~72 launch, settle 50–100 ms. Mốc tham khảo: 72 launch info ~11.6 s; 80 launch FFN ~43 s.
  - Tối đa 2 job active mỗi tài khoản, dùng chung cho mọi agent. Retry khi hết quota (20 s → 120 s).
- **Thống kê.**
  - Paired delta tính theo block. Pooling dùng IV (inverse-variance).
  - Lab không đồng nhất thì dùng random-effects (DerSimonian–Laird), như ở int173 với lab C và D.
  - E[max2] tính bằng U-statistic, SE bằng block bootstrap.
  - Thêm sign test / exact sign-flip. Ví dụ x15k2: 15/18 block, p = 0.0038.
  - Khi verifier không đồng ý, tính lại từ các dòng LABCYC/LABSUM thô bằng một parser độc lập (vstats.py, labre*.py).
- **Kill threshold** phải hiệu chuẩn trên control cùng job, không dùng trace chính thức (E101: ngưỡng Up-W 5.0k là sai, vì control đã 5.8–6.0k). Proxy kill chỉ là thay thế cho "không có lợi ròng", và có thể bị đè bởi kết quả toàn kernel > 2 SE ở 2 job (E112 XG2).
- **Thay đổi đuôi tất định** có thể giữ nếu span đã được xác nhận và E[max2] không tệ đi (E111 SQF+C2). Lý do: SD toàn kernel (FFN 1.0–1.5k) che mất lợi 0.3–0.5k. Muốn thấy hiệu ứng 0.3k trên FFN cần ~6 lab cùng job. Với thay đổi epilogue nhỏ, hãy đánh giá theo chuỗi sau sync của từng launch (FFN e1) thay vì median.
- **Chữ ký sai số / md5.** Thiết kế thay đổi số học sao cho nó để lại dấu vết (v max 0.03125 → 0.01562 đã nhận diện được build E8). In md5 source lúc build (arena_run_iso.sh). Kiểm "SOURCE MD5 header" trong log lab.
- **Đồng hồ.** Thêm print `SB` (base của bộ đếm từng cluster) vào harness lab để đo skew và thời điểm release của sync theo từng launch (leadlam.py). Ở INFO, chữ ký cuối kernel (c1 end − c0 end) phân biệt được ba trường hợp: "c0 nhanh" +0.8..+1.0k / "cả hai ngủ" ~0..+0.3k / "c1 về sau" âm.

### 7.3 Luật KEEP và kiểm chứng

- **KEEP** khi thoả tất cả:
  - paired delta < 0 và ≥ 2.5 SE;
  - E[max2] < 0;
  - mọi run PASS;
  - số học giống hệt (max/mean abs d từng run; với FFN: arm cùng cấu trúc phải khớp tuyệt đối, tools/numchk.py);
  - được xác nhận bởi **lab thứ hai độc lập**.
- **Lăng kính kiểm chứng** (dùng cho x15k2 ở V13.1): ba reviewer đối kháng.
  - THỐNG KÊ: parser riêng, estimator pooled/robust, sign test / permutation test, confound.
  - HỢP LỆ + SỐ HỌC: chữ ký, luật upload, an toàn NaN/pad, sai số so với baseline.
  - CƠ CHẾ: span đã loại skew, lab trace riêng, run cold.

  Lăng kính cơ chế đã tìm ra một chỗ sửa: V issue lúc Q-contraction kết thúc.
- **Mô hình chỉ là bộ lọc để kill.** qsim, hwsim, replay và mô hình cloud đã đoán sai hiệu ứng list 0.7–3k nhiều lần:
  - e4: −0.35 → +0.43
  - ksq2: −0.32 → +1.0k
  - mp: +0.4..1.0 → +2,027
  - qe: −0.9..−2.0 → +1.2k
  - r3m0: −0.8k → +3.2k
  - nd11e9: −6.0k → +0.4k

  Mô hình cloud đoán thiếu cho các arm bỏ DramReuse 0.35–0.65k. Coi lợi mô hình dự là cận trên; không bao giờ KEEP dựa trên mô hình hay arm trace.

### 7.4 Gate trước Arena (tiết kiệm quota)

Làm theo thứ tự, với mỗi biến thể:
1. **Compile static** (`tools/sc.sh`, dùng `flock` build lock). Xem makespan, **list DMA** và vị trí các wait. Gate chính là list chứ không phải makespan. Ví dụ, gate cho Q-first là: Q đi trước, ≤ 3 wait trước V, không có scale k/v ở head.
2. **`DUMP_PE_PROGRAM`** rồi so cấu trúc queue-block của PE (peblk.py, tailblk.py, pe13.py, peseq.py). Bước này bắt được lỗi mà list static không thấy (MOV-6: st4y4, XG4).
3. **Validator switch trên host** (`furiosa_opt_lower::config_switch`, swcheck) cho mọi switch mới, **kèm ca âm tính**. Switch sai sẽ treo device.
4. **CPU emulator bit-identity** so với base (không dùng làm oracle; tự cô lập sub-kernel khi kernel đầy đủ panic, ví dụ V-tail lab, H7 exchange lab). Kiểm số học trên host bằng sim (gsu_sim.py, ffn_fp8_sim.py, qkv_fp8_single_sim.py).
5. **Định giá bằng luật request / lane.py** trước khi build lại DMA. Kiểm engine/piece của descriptor bằng `freechecks_tools/pe_dma.py` trên PE dump. Không dùng `qkvRate/tools/descmap.py` cho FFN 0.8.1 vì nó tách store x và làm mất base động.
6. **Căn trace với list static** (align.py): các span (trừ Task/Cluster) sắp theo begin phải trùng đúng list static.
7. Gate static đã giúp vòng V14 chỉ tốn 2 job Arena, và giúp qkv14/ao14 tốn 0 job.

### 7.5 Quy trình tích hợp một KEEP (int131 / int15 / int151 / int152 / int16 / int17 / int171 / int172 / int173 — đã chạy 9 lần không lỗi)

| bước | nội dung |
|---|---|
| T1 | Kiểm port == arm lab: đối chiếu prefix md5 / từng mục / macro expansion (expand19.py) / chạy lại transform của generator (labarm.py). Kiểm static bằng so multiset toàn bộ lệnh + timing (cmpstatic.py). Tính lại thống kê lab từ LABCYC/LABSUM thô (labre*.py). Xử lý đổi tên trục của gen_lab (labeq.py). |
| T2 | Tạo cây mới bằng WSL `tar` copy của base trước (không kèm target/), rồi byte-copy các file KEEP. Kiểm: `diff -rq`; sigcheck 10/10 chữ ký; axes.rs == p081-base; quét byte LF-only bằng Python (Git Bash `grep $'\r'` báo CR giả); không có `ops_lab` / `lab_` / `println` / `eprintln` / `dbg!`; entrypoint md5 19ce94ce (INFO). |
| T2b | Build static 3 kernel dưới build lock. |
| T3 | Chạy 4 job normal-harness INFO, **lần lượt từng job**, ghi md5 binary mỗi job. Mục đích là kiểm PASS + sai số giống hệt, KHÔNG phải để đo delta. |
| T4 | Snapshot = `cp -r` của snapshot trước + các file đã đổi; tạo lại MD5SUMS; tính hash cây WSL; presubmit.sh PASS (chạy từ Git Bash: WSL không có `python`); build kiểm tra thật với `CARGO_TARGET_DIR` ngoài snapshot. Phải có dòng "Compiling <đường dẫn snapshot>", nếu không thì bước kiểm tra là rỗng (V15: `touch` src rồi build lại). |
| T5 | Viết `Vxx_RESULT.md` + dòng ledger + `RESUME.md`. **Không bao giờ sửa snapshot tại chỗ.** Port tree thành read-only một khi integrator bắt đầu. |

### 7.6 Quy trình nộp bài

- **Chỉ nộp khi user nói rõ "nộp".** Không nhắc nộp. User chỉ nộp khi ≥ 10×; các phiên bản khác báo điểm Arena normal-harness. Thời 09-23 có luật riêng: dừng ở ~7.5–7.6, sau đó "nộp thêm 9 lần, dừng nếu ≥ 8.0".
- **Nộp từ snapshot đông cứng** (`submit-snap-*`):
  - `submit_loop.sh` huỷ khi md5 đổi.
  - `submit_loop_safe2.sh <snap> <log> 1 <fallback>` nộp, chờ kết quả, và nếu kết quả FAIL thì nộp bản fallback, để bài nộp cuối luôn PASS.
- **Presubmit** kiểm:
  - 10/10 chữ ký `#[device]` khớp skeleton 0.8.1;
  - axes.rs = skeleton;
  - chỉ ops.rs + device/** khác so với bản đã được chấp nhận;
  - LF-only;
  - MD5SUMS mới;
  - số file (30 ở V11.1).
- Trên Windows, detach bằng `Start-Process wsl.exe -ArgumentList '-e bash -lc "..."'` (toàn bộ lệnh trong MỘT chuỗi có nháy).
- **Luồng repo team** (thời 09-23/24): git worktree tách từ origin/main → thêm thư mục + RESULTS.md tiếng Việt → push branch → PR (Ver13 = PR #1 … ver19 = V11.1 commit 882b705). Lấy thư mục team bằng `git fetch origin main` + `git archive origin/main <folder>` (file ra CRLF, phải chuẩn hoá). Git trong WSL treo ở prompt credential.

### 7.7 Rủi ro vận hành và cách xử lý

| rủi ro | chuyện đã xảy ra | cách xử lý |
|---|---|---|
| agent khác sửa cây nguồn | Codex trong VS Code revert 5 file QKV lúc 09-16 01:26; job 41088 build ra M3b thay vì E8 | build Arena từ bản sao cô lập (qkv-tt-run/, ao-e1-run/, lab-dma/, cây p081-*), in md5; đóng băng kernel không đổi; không copy cả file khi merge |
| ổ C: đầy vì WSL | ext4.vhdx 191.7 GB làm đầy C: (09-25 08:36; lặp lại 3–4 lần, crash lúc 21:04 khi C: còn 0.22 GB) → WSL E_FAIL / "Catastrophic failure E_UNEXPECTED" | `wsl --shutdown` + xoá thư mục target t081_* cũ (40 thư mục ~63 GB), swap.vhdx cũ trong %TEMP%, log .etl của WSLg DiagOutputDir (460–527 MB mỗi lần), clone thừa của các sweep; tạm dừng build khi C: < 2 GB; dùng lại ~/t081_*; sửa lâu dài (dời distro sang E:) còn chờ user duyệt |
| WSL hết RAM | VM ~7–7.5 GB; mỗi rustc ~5.7 GB; 4 build song song bị OOM-kill (exit 137) | tuần tự hoá bằng `flock -w 5400 /tmp/rngd_build.lock <cmd>` |
| `wsl --shutdown` giết agent khác | – | không bao giờ shutdown từ bên trong một agent; sau restart, lấy lại job đang chạy bằng `furiosa-arena logs <job>` |
| job chạy nền bị giết | `wsl -e bash -lc 'nohup … &'` bị giết khi WSL thoát; VM restart giết cả job setsid-nohup | chạy foreground bên trong một lời gọi được đẩy ra nền (Bash run_in_background, giữ phiên WSL mở) |
| user interrupt giết workflow agent | 09-23 15:35 (mất 1 giờ), 09-26 11:51 | sau mỗi interrupt kiểm journal / mtime của agent-*.jsonl rồi relaunch với resumeFromRunId + epoch mới; không SendMessage tới id agent workflow còn sống (sẽ sinh writer trùng, sự cố 09-25 08:40) |
| hết token / đổi tài khoản | nhiều agent chết (V14, V15, qkv19 429, int173) | **RESUME_PROTOCOL**: progress file có tên; đọc khi bắt đầu; ghi job Arena NGAY sau khi submit; kiểm filesystem trước bước không đảo ngược được; báo cáo tăng dần; `run()` helper retry null kèm ghi chú RESTART + epoch, trả `{incomplete: true}` thay vì tự quyết; phát hiện writer trùng thì ghi HANDOVER rồi dừng |
| API rate limit | 10 agent song song → 8/10 chết | pool ≤ 3 |
| CRLF | git-for-Windows clone ra CRLF làm script lỗi "required file not found"; lab tree CRLF làm snippet KEEP-READY dính CRLF (int171) | `core.autocrlf false`; script chỉ LF bên trong thư mục làm việc; integrator chuẩn hoá (nội dung không đổi, kiểm bằng multiset static); quét CR bằng Python |
| chuyển đổi đường dẫn MSYS | Git Bash làm hỏng `/mnt/e/...` | `MSYS_NO_PATHCONV=1` hoặc tiền tố `cd ~ &&`; dùng script launcher bên trong WSL |
| encoding console Windows | cp1258 cắt cụt file báo cáo (ao20) | ghi file với UTF-8 tường minh |
| script patch hỏng im lặng | REV-Z là chạy lặp của 016f; rev-AD/FINAL không ra cycles | kiểm md5 / diff trước khi submit |
| target cũ | rustc ICE `uninterned StableCrateId` | `CARGO_TARGET_DIR` mới trên ext4 (build 1m29s); không compile và test cùng lúc trên một target |
| git fetch nhầm chỗ | clone bare cũ không có refs/ nên git đi ngược lên repo cha (doan-duc/Euclidean) | đặt `GIT_CEILING_DIRECTORIES`; clone mới trong scratch |
| sửa entrypoint | ENV-1 đã sửa `remote_entrypoint.sh` của lab | khôi phục md5 19ce94ce; không để entrypoint đã sửa lọt vào snapshot |
| tên trùng giữa các lab | tên trong ops_lab trùng giữa các engineer (lab_qkv_qs2, lab_qkv_qv3) → 2 lần compile fail | dùng hậu tố duy nhất |

### 7.8 Bộ công cụ còn lại (đường dẫn tính từ `p081-v10/v10` nếu không ghi khác)

- **Trace / lane:**
  - `experiments/p081/trace_summary.py`; `analysis/v15/freechecks_tools/{align,table,gl3,pe_dma,dmageo,kr4}.py`
  - `analysis/v15/consolidated_tools/{lane,roofline}.py`; `analysis/v15/probes/{pan,paired,tagalign}.py`
  - `experiments/fcf18/tools/{mine0,bothslow,info0,fcfan,leadlam}.py`
  - qkv15 `chip15.py`, `tr15.py`; ao15 `tail15.py`
- **Mô hình** (chỉ dùng làm bộ lọc kill):
  - FFN `analysis-work/cp-ffn/p081/py/hwsim.py`, `experiments/p081/ffnR/hwmodel.py`
  - QKV `hwsim.py` (qkv9), qsim (qkvD)
  - AO `ao_sim17.py`, `ao_r1_replay.py`, `ao_tick_model*.py`
  - FFN `ffn18src/ndmodel.py`
- **Kiểm tra:**
  - swcheck (validator switch trên host); `tools/numchk.py`; `labeq.py`, `labre*.py`, `cmpstatic.py`, `expand19.py`, `labarm.py`
  - `presubmit.sh`, `submit_loop_safe2.sh`, `tools/sc.sh`
  - `v12okv/align_audit.py`, `dmmap.py`; `p081-v10-rdJ/tools/peblk.py`
- **Sim số học:** `tools/ffn_fp8_sim.py`, `tools/qkv_fp8_single_sim.py`, `gsu_sim*.py`; fixture `scripts/fixture_prng.py` + `generate_references.py` (các đường dẫn này tính từ `E:\2026_Macro_NPU`).

---


## 8. Nên làm gì tiếp và những gì KHÔNG nên làm lại

Tình trạng cuối ngày 09-26:
- **V17.3** là bản tốt nhất (`submit-snap-p081v173`, hash 1ef19c8cd9bd), đã tích hợp vào `furiosa-opt-gemma4-12B-main`.
- Điểm NH: 9.139 ± 0.047 (5 job) và 9.173 (4 job). Ước lượng chính thức ≈ 9.19.
- **V17.3 chưa nộp.** Bài nộp chính thức hiện tại vẫn là V11.1 = 8.8195. Hạn: bài nộp CUỐI CÙNG trước 2026-10-01 11:59 UTC được chạy lại. Theo kế hoạch V17_PLAN, sau 09-29 12:00 không mở lab cho cơ chế mới nữa. Nếu user quyết nộp, nộp đúng snapshot V17.3 qua `submit_loop_safe2.sh`, dùng một snapshot đã PASS làm fallback.

Quy đổi điểm: 1k cycles ≈ QKV 0.46–0.48 %, AO 0.94–0.96 %, FFN 0.15–0.16 %.

### 8.1 Ý tưởng còn mở, xếp theo giá trị kỳ vọng

| hạng | kernel | ý tưởng | lợi kỳ vọng | trạng thái / điều đang chặn | thử đầu tiên (rẻ nhất) | nguồn |
|---|---|---|---|---|---|---|
| 1 | FFN | **R5 — Down kiểu leader:** họ ND chia Down theo H (bỏ penalty L-split ~2.2k và scale Down lệch 3.6–3.9k) + block scale Down áp bằng CE với **bf16 block sums** + trao đổi h trong cluster | mô hình −9..−14k FFN (≈ +1.4..+2.2 % điểm), P ~0.15 | chỉ bị chặn bởi **gate số học nội bộ** của ta (bf16 block sums → r2 max abs d 0.01172 > 0.00781). Vẫn qua tolerance cuộc thi (~2× dư; lab 100856 PASS). **User phải chọn:** A = giữ gate chặt; B = theo tolerance cuộc thi | (1) user quyết A/B; (2) nếu B: kiểm số học trên nhiều seed hơn bằng sim host (gsu_sim-style) cho `atol 0.01 + 0.01·abs(ref)`; (3) build static H-split ND + CE, kiểm list; (4) lab INFO cùng job so với nd11gz | BACKWARD_FFN §4, §7; ffn18 100856 |
| 2 | AO | **Bỏ gather DMA cục bộ của c0:** hàng của c0 ở lại slice contraction 16g, chỉ hàng của c1 chuyển vào slice 16g+1 của c0; epilogue layout `m![H/120 % 16, H/1920 # 16]`, reduce ms 256 slice | chưa rõ (ở INFO, cắt đường c0 được ~1:1 tới ~0.8k) | nhiều ngày công; chưa biết có biểu diễn được không (cần relabel cluster cho buffer kết quả) | compile static một kernel nhỏ để thử relabel cluster + reduce 256 slice | ao20 §3.4 |
| 3 | AO | **pws:** chuỗi identity-copy trên Main sau contraction a (trễ static contraction b), để DMA ws được list trước W(tile b). Ws land ở T0 + 0.3..0.8 thay 1.59 và rút ngắn chuỗi drain trước WAIT2 của c0 (chuỗi này quyết định 14/24 launch) | ~0.3–0.8k | kế hoạch, chưa đo; rủi ro xáo đuôi (NEW-cbfill từng lật đuôi ở 4/4 list) | list static + peseq.py (tail list giữ nguyên aoYG trừ vị trí ws), rồi lab INFO 8×3 | ao15c_progress 07:08 |
| 4 | AO | "phương tiện" bỏ DramReuse / Sub-stretch mà tốn < ~0.2–0.4k | ≤ −0.2k INFO (−0.36..−0.54k own-frame ở trace) | mọi phương tiện đã biết đều đắt hơn (rpb +728, rzW +1,751, n1r +1.4k, lf2 +2.1k, dr41 / dra* đã đóng) | chỉ thử nếu có ý tưởng mới làm được dưới 0.2k HW | ao20; BACKWARD_AO |
| 5 | AO | danh sách đuôi "IDEAL" / tile cuối nhỏ hơn giữ list của aoYG | −0.34k / −0.2..−0.27k | cần một pin làm trễ CHỈ start static của ws_vrf mà không tốn HW | thử công cụ ping-pong WAW (NEW-cbfill) nhắm đúng ws_vrf, xem list | ARCH_AO_B §7.1-7.2 |
| 6 | FFN | các phần dư sau nd11gz: Main-bound 0.6k trước fused4; 3 pass VE scale 2.2–2.8k với VRF load nối tiếp (pass đầu bị StoTab của tabB giữ 0.8k); đuôi 8.2k (cluster sync 2.44k) | mỗi cái ≤ 0.6k dưới gate bit-identical | đưa tabB lên trước s15 thì VRF load đầu phải sau fusedA (+1.2k) cho lợi tối đa 0.6k | probe static | ffn20_report |
| 7 | FFN | kiểm lại thuế sync của y-hop FFN ở INFO bằng chữ ký SB (thuế trace 2.2–2.5k có thể khác ở INFO, như đã thấy với AO) | thông tin (nếu thuế ở INFO thấp thì các đòn bẩy sync FFN mất giá trị; nếu cao thì mở lại L4) | chưa chạy | thêm print SB vào harness lab FFN, 1 job INFO 12 launch | ao20 §3.5 |
| 8 | FFN | PB3 probe đuôi (phần exchange + sync trong 8.2k; đuôi kiểu leader ≤ 5k), PB4 micro-probe StoTab, PB5 tốc độ lookup phụ thuộc dữ liệu (code toàn 0 so với code ngẫu nhiên) | thông tin | chưa chạy | 1 job lab chỉ-TU nhỏ | BACKWARD_FFN §6 |
| 9 | FFN | P4: DMA loại scale chậm lại trong kernel (Gate scale 10.37–10.58k trong kernel so với 9.38–9.46k chỉ-DMA): do tranh chấp bank DM hay do pin share | −1.5k (−0.5..−2.5) ở P 0.2 | cần build micro-kernel có contraction trên Main | 4 arm gsc / gsc_main / gsc_main_dis / gsc_c0w | CONSOLIDATED §5 P4 |
| 10 | FFN | X5 tách Gate W thành 2 lệnh (chỉ có lợi nếu P1 găng); X13 x/rms sau Up W | −10.5k trên P1 (có điều kiện) / −1.5k (mong manh) | chưa build | static: kiểm P1 có găng trên nd11gz không | BACKWARD_FFN §4 |
| 11 | FFN | phần dư mỗi lệnh chưa giải thích so với luật request (tables +0.2–0.67k; trước đây seed Up q4 +0.9–1.7k và Gate direct +1.75–1.96k) | ~4–5k tổng (một phần đã biến mất nhờ oct seed / ND) | không giải thích được bằng coverage / engine / số slice | đo lại lane.py trên trace nd11gz | FREECHECKS §8 |
| 12 | FFN | luật 64 lần truy cập liên tiếp cùng bank (op TU vượt ngưỡng bị lập lịch như chiếm DMA → tuần tự ẩn) | chưa rõ | audit được gợi ý, chưa làm | grep lịch static nd11gz tìm op TU bị đánh dấu chiếm DMA | API_RECIPES N6 |
| 13 | QKV | đuôi V thành 2 pass switch + VE (gather + ms → VRF; gather + normalize → commit), VRF v_scale theo thứ tự hoán vị, nạp v_scale đầu tiên trong đuôi | ≤ ~0.2k metric | chưa build (C4 thời V10 fail vì fetch không liên tục; cần layout khác) | compile static | qkv19_report |
| 14 | QKV | khoảng cách tới leader ~0.9–2.5k dưới sàn lane hoàn hảo (65.4k): đuôi ~5.5–6k và/hoặc stream nhanh hơn vài % | – | chưa tìm được cơ chế biểu diễn được (burst nhỏ đã bị bác; T2-2 không biểu diễn được) | – | BACKWARD_QKV §7 |
| 15 | QKV | PR-6: skew so với kích thước program (QKV 90 program TU / 651 KB code so với AO 21 / 180 KB) | xác suất thấp | chưa chạy | 1 job fcf-style với program phình to giả | BACKWARD_QKV §6 |
| 16 | tất cả | **GL-1** (hàng đợi DMA theo từng PE; overlap giữa các engine rời nhau): có trên HW (−1.1..−2.9k trong probe) | có điều kiện: QKV −3.5k / FFN −3.0k / AO −0.5k (~30 % khoảng cách điểm tới leader) | KHÔNG biểu diễn được trên 0.8.1 (mọi DMA chứa engine 0; tile slice/cluster làm đích DMA chưa được lower) | đề nghị Furiosa / toolchain sau lower `DmTensorViewMut::slice_tile` làm đích DMA | PROBES P2; AUDIT_L1; INTEL §4 |
| 17 | tất cả | bất đối xứng mịn c0 nặng hơn (~0.4–0.8 MB): cơ chế chi phí cố định duy nhất đã đo khớp với khoảng cách đồng đều 1–2.4k mỗi kernel của leader | −1..−2.4k/kernel nếu biểu diễn được | độ mịn thô (hàng có packing và tile cluster đều bị từ chối) | tìm layout mà một khối ~0.4–0.8 MB có thể chuyển c1 → c0 | fcf18 J3 / L1 |
| 18 | tất cả | ghép host/firmware (đội khác: ghim host CPU 1 core −7..−17k trên 0.6) | chưa rõ | không điều khiển được từ kernel; frame +20 tick/đối số không phải đòn bẩy | – | V12_X2; fcf18 J2 |
| 19 | tất cả | double buffering tường minh trong một loop rolled | chưa rõ | chưa thử (kernel không có loop; loop rolled thêm drain DMA đầy đủ +2.1k) | xác suất thấp | AUDIT L24 |
| 20 | vận hành | dời distro WSL sang E: (`wsl --manage Ubuntu --move E:\wsl`) | loại bỏ crash do C: đầy | chờ user duyệt | – | memory wsl-disk-full-hazard |
| 21 | vận hành | câu hỏi cloud còn treo: FFN follow-up, QKV addendum, paradigm prompt (GL-1 + asymmetric) | – | – | triage mọi câu trả lời theo nguồn 0.8.1 + một probe rẻ trước khi build | RESUME 16:15–18:35 |

### 8.2 Những gì KHÔNG nên làm lại (đã đóng, có bằng chứng)

**Chung / toolchain**
1. Dời DMA bằng thứ tự chương trình / thứ tự statement / thứ tự store (≥ 10 lần byte-identical). Chỉ phụ thuộc dữ liệu mới dời được list.
2. Nạp lại cùng buffer bằng `to_dm_view` để ping-pong (lỗi WAR im lặng, 34133). `to_dm_view` / DMA vào TILE của một DM tensor (HW FAIL 80296; chép toàn bộ extent có pad; thêm PeCore giữ issue: OCT-c +976).
3. `Option<DmTensor>`, struct/array literal, device struct (MIR ICE).
4. Dời pass từ Main sang Sub để rút hàng đợi Main (r19, r20s, r23, x-bcast trên Sub, XG2 trên Sub, octs, cc1s: Sub cũng in-order, fetch Sub chỉ 8 B, StoTab chặn Sub). Fuse pass thay vì dời.
5. Dùng lại bảng decode / khử trùng LUT / hoist LUT khỏi loop (1 bảng mỗi pass lookup, AUDIT L3; FFN-LUT +2,384; H-LUT không build được).
6. Activation FP8 một thành phần (QKV chỉ là artefact fixture, Q-6 +0.7..+1.3k; FFN 98.93 %; AO sai ~0.14; Down ~3.6 % RMS).
7. Nhân bản tensor tí hon ra nhiều slice (4 B → 256 slice = 7.4k).
8. Dồn một load lớn lên ít slice / một cluster (rate ~ số slice sống).
9. SwitchConfig chưa qua validator host / emulator (treo device). Main `to_vrf` thẳng từ `collect` không có chuỗi VE (treo, 2 lần).
10. Xoá một scale "vì nó triệt tiêu qua RMSNorm" mà không kiểm độ lớn so với EPS (FFN 23.98 %, 2 lần). Dời scale qua một norm.
11. Chiều rộng packet fetch của contraction/lookup (byte-identical; tốc độ lookup không phụ thuộc packet).
12. `Dma::Pcie` như một engine riêng (lower thành tensor DMA). Alias legacy `to_dm` / `to_dm_at` với địa chỉ (bị bỏ qua).
13. Loop rolled trip-count-1 làm hàng rào thứ tự (block TUC `for` với drain toàn bộ, +2.1k).
14. Knob compiler ẩn / Config driver (không tới grader). Tile Cluster / slice làm đích DMA (visa từ chối). Index tile hàng có packing.
15. Quyết KEEP dựa trên static makespan, arm trace, qsim/hwsim/replay, mô hình cloud, hoặc median NH so chéo job.
16. Mọi mẹo "cho contraction có Main idle" (TU-clock là artefact trace). Tối ưu cold run 0.
17. Probe tốc độ stream / layout > 651 B/tick mà không có layout ứng viên (mọi layout lớp nhanh bão hoà ở 305–312 B/tick/cluster; covM ≤ 0.5 %).
18. Audit độ phủ kênh HBM (KR-1 DEAD), độ phủ engine (CJ-3), độ rộng load nhỏ (CJ-5), hình học DMA bảng (CJ-4), Sub store so với contraction (GL-3), tách DMN ở ranh giới (OFF-2).
19. Burst DMA nhỏ để giấu độ trễ (sàn 0.82–0.91k; tiết kiệm ≤ 0.4k mỗi ranh giới).
20. Bất đối xứng cluster ở độ mịn thô (QKV 1 head, FFN 1 hàng L, AO 16/14 nhóm).

**QKV**
21. Hậu xử lý head-local dựng từ nhiều pass nối tiếp nhỏ (004, 012b, 011/013/014/015). Head-per-slice / mỗi PE một head thì KHÔNG đóng (h4 thắng).
22. Tách H 8 đường (run 480 B); trọng số cả hàng qua fan-out HBM thuần; layout M1 đứng riêng; M2 DM→DM replicate; M4 fuse K+V.
23. Remap HBM theo L3 p1–p4, far/farq, run 4/16/32/64 hàng, hàng liên tục theo slice (f4), parity/offset K/V 512 (qkvOKV NULL), QKV-MIX, K15 riêng lẻ không có ksq2 (x15 +1.4k).
24. Giữ layout contraction strided xuyên suốt norm/RoPE/store (piece 2 byte); V norm không gather (T2-2).
25. A1 packet 64 B trên K; steering list static qd/g1/g2/g12/bk/a1vk; e4; K/Q trên PE0 + trải V; Q normalize với q làm stream; sqrt → VRF từ Main trong cửa sổ Sub bận; nhân bản x 4× (x4g); load head vào đích tile.
26. Dời load nhỏ lên head (s1a/h5a/h5b +1.3..+2.3k), ép k_scale/rope_off vào head, HEAD-1/h0/h2 (15 dạng static), Route 1 bqkk* (6 dạng), mask-first head (q19h +1.4k), r3m0/a2 (+3.2k/+3.4k), qf1–qf6/qf9–qf11, memset pin ở head QKV (mp +2.0k, qM1/qM2), zero-pin 256 slice (hằng số 8 KB), 5 bản copy nhẹ Sub (g10z +0.84k).
27. Tách V thành 2 tile / 2 nửa head / 2 nửa hàng (bvs2 +1.9k, bhv2 +3.95k, vr18–21), K contraction 2 nửa (q19ks +1.05k), tách Q (qs1/qsb2), Q/K 1 head mỗi slice (E9).
28. Fuse normalize + RoPE với hệ số dựng trên Sub (bfnr3 +1.4k), ramp gieo từ chunk mask (brmp +1.06k), RoPE row nửa độ rộng (hz/hzg/hs +0.85..+1.9k), ramp in-place (`&raw const`).
29. Đòn bẩy queue-block của v-scatter (E351), steering v_scale (E302), GP packing 8 B (E312), gộp sqrt vào q/k norm (q19fqk), VRF q-scale bằng pass Main (qe +1.2k), swap q_rms/q_scale (~0.05k ở INFO), page-pad q_scale (list không đổi).

**AO**
30. Tile trọng số khác 96+24 (t100 +762, t90 +731, t3 +2,215, tb16 +942, fe3 +1,257, 104/106 + chain, 22+8 thời v8, 8+22 một sync); một lệnh trọng số (+2.1k); bỏ steering chain (aoN +986).
31. Layout trọng số O cả hàng / ít slice hơn; probe layout khác K16 (covM: K16 nhanh nhất).
32. Steering head bằng pass giả (hs/hs2/hs6, zb), Broadcast1{16,4} để fan-out (treo), replicate x bằng DMA (xrep), X-PUSH, operand ở head, AO-H4 seed-first (≤ 0.18k).
33. Mọi thay đổi thứ tự list đuôi đặt POST3 trước WAIT2 hoặc ws_vrf sau gather (c20, t90, p5c +2.4k, WSM +2.4k, NWFIRST, ops2/ops3/opsn, T1/OV/PI, prw — SAI kết quả, pwr); VRF operand ghi từ Main (opsmain, wsm TREO, wsm2 +213); D2 / IDEAL / subswap / cbfill (không có handle).
34. Mọi phương tiện bỏ DramReuse đã biết: A5 (+1.29k), A5+B (+1.5k), R1b (static kill), r1wg/r1w (+0.7k), rzW (+1.75k), rpa (+520), rpb (+302/+728), n1r/n1r2 (+1.4/+1.5k), dr41 (static kill), lf2 (+2.1k), dra1/dra2 (view alias không có tác dụng).
35. Epilogue AO từng cluster / T-split / H7 / epilogue trên c1 / all-gather rồi cả hai cùng tính / AO-B1 / chia mịn P1–P4 / one-way hop cluster_tile / H2 không epilogue trên c1 / RMS 2 nhóm (SAI hàm).
36. Epilogue 64 slice (E64), final stream y (yst2 +1.09k, rpy +1.18k), inv_sw trên Sub (ics −46 ± 70) / Main commit (ic), Gram-contraction (D3), AO-E cả hàng mỗi slice, pre-gather RMS partial, hi/lo trên lane MAC, TB contraction, AO SQF (C1a), WSFOLD.
37. Lật dead zone cuối kernel AO (chỉ có ở trace; ở INFO c1 luôn xong trước).

**FFN**
38. Pipeline tile ở mức 0.6 (FFN-003..011, 015a, 021, 024, 027–029, 032/033) — đã bị v11/0.8.1 thay; tile Down 11/4 / 10/5 / 13/2 trên thân fxv (+5.2..+17.4k) — (11/4 chỉ có lợi trên thân CC-1 với mtrf + mce).
39. Hop x DM→DM / fan-out (XDM 28.6k, FA24 27.4k, xd8 30.9k HW, v3 15.6k), x về 2/4 slice (st2 / st4y4 / MOV-2 / XG4 E114), store x thẳng từ slice GeGLU (MOV-3), xcopy (+3,065), CC-1b reload hẹp (+1,968), CC-2 nguyên văn (không build được), ring-4 thay bằng all-gather có pad (E103p), E103q/qm/qg (hazard list), bỏ cfg4, pad địa chỉ reload.
40. Một DM→DM relayout y (Y1 / y1 / MOV-4: DramReuse +4.5k HW), epilogue cục bộ H7, gộp pass cộng vào ms/final (fa1*; thứ tự stage VE), regroup trên Sub (rga ICE) / trên Main (rgm +355) / theo cặp (rgb2 +19k), FFN-D, FFN-B, Pcie d2d.
41. Họ D-CE với bf16 block sums DƯỚI GATE HIỆN TẠI (odcf12 +3.2k và trượt gate số học), GSU (dời scale qua Down: đổi kiểu làm tròn bf16 / thêm DramReuse), varA cho Down, varB (+6–7k), commit_cast cho scale Gate (không lower được), VRF bf16 (không có VeScalar).
42. Gộp scale Down theo cách tạo piece 480 B ở chỗ lane không găng (s14, s123, s15 trên v8/v9; P1/P1b aligned + ring-8; Down-scale oct; seed oct cho scale Gate ở chế độ có decode), DMA scale đúng 4 hàng (s4x +2,049), cửa sổ 7 hàng (PROG-719), nhóm VRF 2 hàng (k15v2), VRF scale Down ghi từ Main (mv +6.3k), pdma relayout (243,710 static).
43. Họ OCT trên fxv (OCT stage 1 kiểu ring-8; OCT-c pin tile); relayout trên Sub (octs); layout 240×32 / r32 (+12.6k); đổi sở hữu hàng Down để né hash kênh HBM (ffn19: mô hình XOR sai); probe coverage (covM ≤ 0.5 %).
44. Warm-up Sub ở head (F-3a/b/c NULL), F-4b (+355), memset pin ở head FFN (mọi site 0), nd11e9 (scale muộn +0.4k), pass zero/scale trên Sub (ICE), Main `to_vrf` thẳng từ collect (treo).

---

### Phụ lục: chỉ mục job Arena theo giai đoạn (để tra lại log)

| giai đoạn | job |
|---|---|
| 0.6 (09-14 → 09-23) | 31766, 31778, 33638, 33853, 34133, 34181, 34230, 34314, 34339, 34360, 34388, 34401, 34417, 34458, 35294, 35343, 35403, 35421, 35431, 35441, 35448, 35464, 35471, 35476, 35508, 35530, 35534, 35543, 35549, 35555, 35568, 35640, 35641, 35647, 35700, 35707, 35710, 37513, 37852, 38125, 38188, 38261, 39037, 39534, 39569, 39600, 39615, 39644, 39654, 39661, 39667, 41088, 41107, 41109, 41112, 41114, 41120, 41121, 41475, 41489, 41499, 41516, 41538, 71345, 71414, 71568, 71578, 71613, 71629, 71633, 71648, 71655, 71664, 71671, 71688, 71689, 71698, 71703, 71718, 72017, 72064, 72065, 72167, 73458, 73542, 73583, 73595, 73671, 74981, 75131, 75191, 76874, 76912, 76955, 76963, 76998, 77009, 77036, 77046, 77120 |
| port 0.8.1 + v7–v9 (09-23) | 77526, 77588, 77915, 77919, 77953, 77990, 78987, 78998, 79000, 79018, 79019, 79025, 79043, 79061, 79091, 79106, 79125, 79145, 79164, 79167, 79171, 79198, 79203, 79226, 79254, 79255, 79382, 79428, 79473, 79477, 79483, 79494, 7951x, 79559, 79564, 79582, 79589, 79618, 79621, 79637, 79639, 79660, 79665, 79666, 79682, 79688, 79705, 79712, 79720, 79731, 79743, 79793, 79795, 79939, 79963, 80046, 80291, 80296, 80316, 80323, 80342, 80375, 80381, 80403, 80408, 80412, 80415, 80441, 80457, 80462, 80799 |
| V10 → V12 (09-24/25) | 82802, 82835, 82904, 82925, 82931, 82963, 82964, 82986, 83007, 83021, 83063, 83098, 83170, 83174, 85106, 85109, 85174, 85212, 85243, 85260, 85340, 85350, 85630, 85677, 85678, 85694, 90846, 90990, 91165, 91167, 91191, 91198, 91210, 91211, 91219, 91220, 91221, 91244, 91252, 91297, 91299, 91304, 91306 |
| V13 → V15.2 (09-25/26) | 91457, 91461, 91477, 91501, 91595, 91647, 91675, 91676, 91684, 91691, 91707, 91722, 91738, 91739, 91771, 91776, 91814, 91889, 91909, 92075, 92953, 92963, 92968, 93569, 93611, 94088, 94091, 94195, 94201, 94208, 94235, 94261, 94268, 94294, 94324, 94333, 94546, 94552, 95579, 95840 (TIMEOUT), 95861, 95894, 95904, 96040, 96112, 96127, 96128, 96138, 96288, 96311, 96314, 99694, 99708, 99723, 99728, 99736, 99741, 99742, 99745, 99764, 99768 (TREO), 99772 |
| V16 → V17.3 + V18 (09-26) | 99849, 99864, 99873, 99881, 99884, 99885, 99892, 99898, 99902, 100086, 100103, 100136, 100151, 100168, 100190, 100204, 100226, 100276, 100292, 100316, 100323, 100360, 100373, 100413, 100418, 100427, 100436, 100838, 100847, 100856, 100887, 100890, 100891, 100892, 100939, 100958, 100979, 101005, 101022, 101042, 101064, 101114, 101188, 101193, 101200, 101205, 101221, 101887, 101926, 101974, 102016, 102056, 102062, 102090, 102103, 102114, 102121, 102138, 102254, 102266, 102267, 102281, 102287, 102302, 102420, 102567, 102602, 102670, 102723, 102735, 102755, 102760, 102775, 102780, 102800, 102814, 102873, 103183, 103184, 103186, 103187, 103229 |
| đóng gói cuối (furiosa-opt-gemma4-12B-main) | 103215, 103216, 103221, 103225, 103231 |

### Phụ lục: danh sách bài nộp chính thức (moa-submitter)

| id | phiên bản | điểm | ghi chú |
|---|---|---:|---|
| cd8cad57 | REV-AK (0.6) | FAIL | `unresolved import crate::axes::Hl` |
| ca54f80c / ccaf0508 | 09-15 (0.6) | 5.1324 / 5.6453 | board của team |
| aeea0c97, a4e96a99, d0bff8ea, d38ebfe4, f728dd57, 7332b8b0 | v11 + trì hoãn RoPE (0.6) | 6.3057, 6.8860, 6.6209, 6.5545, 6.5950, 6.6794 | – |
| 9c7a97cd, e79ed4a4, **92fdd139**, cfc3e216 | r21n (0.6) | 6.7434, 6.4623, **7.4345**, 6.5259 | 6e48babf, 22b3e0a0 bị từ chối (đổi tên tham số) |
| c1928278, 46c20165 | P081-v1 | 6.7089, 6.7381 | – |
| 8225589c … e1748f19 (11 lần) | P081-v6 | 7.6453–7.7250 (median ~7.71) | – |
| **a0af0cea** | P081-v7 | **8.0337** | – |
| 5f9689dd | ver15 (teammate) | 8.1527 | – |
| **16b602fe**, b1e5c347, … | P081-v8 | **8.4051**, 8.3711, 8.3601 | – |
| **563b0e6e**, 9838fdd4, c1de64c3 | P081-v9 | **8.4708**, 8.4210, 8.4758 | – |
| (Version_18 của team) | V18 | 8.6744 | teammate |
| **bc26edb1** | **V11.1** | **8.8195** (71,874 / 35,402 / 215,068) | **bài nộp chính thức hiện tại (cuối cùng)** |

*Hết log.*

## Bổ sung 2026-09-28: kiểm tra hướng thay đổi kernel để tiến tới 10×

- **Mốc hiện tại:** V17.5 đã đạt **9.2337× chính thức** (QKV 68,124 / AO 35,079 / FFN 199,539). `submit-snap-p081v176` là ứng viên tách biệt, chưa nộp; chỉ `src/device/shared/mlp.rs` thay so với cây V17.5 tương ứng. Ba lab INFO cùng job (118539, 118574, 118604) đo FFN giảm **650 ± 131 cycles** khi ghép cặp; tính từ điểm chính thức V17.5 cho ước lượng **~9.244×**, không phải điểm chính thức. Bốn job normal harness 118938/118962/118982/118994 đều PASS 15/15, điểm riêng 9.2572/9.1434/9.2118/9.1611; khác phiên nên không dùng trung bình của chúng để kết luận hơn V17.5. Chi tiết: `p081-v10/v10/state/V176_RESULT.md`.
- **Ngân sách 10×:** từ V17.5, tích ba số cycles phải giảm xuống **0.7873** lần. Nếu chia đều mức giảm tương đối giữa ba kernel, đích là khoảng **62,904 / 32,391 / 184,248** (mỗi kernel giảm 7.66%). Để bằng #1 trong ảnh user cung cấp (61,555 / 30,421 / 183,835; 10.293×), chênh lệch riêng từng kernel là **6,569 / 4,658 / 15,704 cycles**. V17.6 chỉ cải thiện riêng FFN ~0.33%, tương ứng điểm ước lượng +0.010×. Đây là lý do các chỉnh sửa nhỏ không đủ để đạt yêu cầu.
- **FFN H-split Down (thay đổi cơ cấu) dừng ở gate chi phí:** đọc trọn hàng H và scale căn 256 B có thể tiết kiệm 4.6–6.1k DMA, nhưng phải đưa cả 15,360 phần tử GeGLU tới các slice sở hữu hàng H. Nửa activation mới cần thêm ~7.14k Main broadcast và ~1.27k TRF. Mô hình từ các primitive đã đo cho lower bound ngay cả khi DMA và exchange miễn phí vẫn **+4.2k** với CE đã đo; giả thiết CE lý tưởng vẫn **+1.5k**. Chưa tạo kernel hay chạy Arena cho biến thể H-split; không coi đó là kết quả phần cứng. Chi tiết: `p081-v10/v10/experiments/r5h_report.md`, `p081-codex-hsplit/RESULT.md`.
- **FFN R5B (đã custom kernel, chỉ compile):** đổi 4 hàng Down cuối sang CE, compiler 0.8.1 PASS nhưng static makespan **102,866** so với R5 **102,422**; không rút ngắn đường găng, dừng trước Arena và kiểm số học. Chi tiết: `p081-codex-r5b/RESULT.md`.
- **AO n1r là cảnh báo về static:** makespan **21,857** so với aoYG **22,869**, nhưng lab INFO **101022** đo biến thể này chậm khoảng **+1,409 cycles** vì DMA và Sub pass bị đẩy vào đường găng. Không hồi sinh biến thể chỉ dựa vào static. Chi tiết: `p081-v10/v10/experiments/ao18_report.md`.
- **Giới hạn còn mở:** chưa tìm được cách biểu diễn bằng 0.8.1 để tránh DMA engine 0 hoặc đưa nửa GeGLU ở cluster kia tới toàn bộ slice Down mà không trả phí broadcast. Probe engine-lane job 118865 không thấy K16 weight được lợi từ việc cho engine 1–3 chạy trước; các chuyển đổi expressible dự kiến <1k/kernel. Đây là kết luận theo các probe hiện có, không phải chứng minh 10× bất khả thi. Chi tiết: `p081-v10/v10/analysis/v19/ENGINE_LANE_REDESIGN.md`.

Giữ nguyên V17.5 trong cây chính, V17.6 trong snapshot riêng. Khi có một cơ chế mới đủ giảm vài nghìn cycles trên đường găng, cần qua thứ tự: biểu diễn/lowering 0.8.1 → kiểm số học → lab INFO ghép cặp cùng job → normal harness → điểm chính thức. Không dùng một điểm Arena khác job hoặc static schedule để tuyên bố đã đạt 10×.

---

## 9. Bổ sung 2026-10-03: V17.5 → V17.8 và kết thúc stage 1

Mọi đường dẫn trong mục này tính từ `E:\2026_Macro_NPU\stage1_archive\`. Báo cáo của từng track nằm ở `p081-v10/v10/experiments/<track>_report.md`, log từng bước ở `<track>_progress.md`. Hồ sơ phiên bản nằm ở `p081-v10/v10/state/V17x_RESULT.md`.

### 9.1 Chuỗi phiên bản và điểm chính thức

| phiên bản | thay đổi so với bản trước | bằng chứng | điểm |
|---|---|---|---|
| V17.5 | QKV `qkv63_vtail` (x4 fan-out on-chip, đuôi V gồm 2 pass switch+VE) và `normalize_input_k15x4f`; FFN fsi16 (các hàng Down xen kẽ theo 16) | `V17.5_NOTES.md` | bảng **9.2337** (`e7048c3f`) |
| V17.6 của đồng đội (`tm176`, thư mục `V17.6_9.269x/`) | Chỉ đổi FFN. (1) Bảng tra Up xếp hàng sau lệnh nạp rms weight; pass tạo số 0 có thêm một reduce giữa các slice. (2) usf: global scale của Up ra khỏi GeGLU và được gộp vào global scale của Down. | lab −748 [−1,049, −299] so với V17.5 | bảng **9.2691** (`74c8d610`) |
| V17.6 / V17.7 của mình (`submit-snap-p081v176/177`) | FFN r5a11g (X8: scale block của Down áp bằng CE trên block sum bf16, chia 11/4) rồi gh (gq + h1, chỉ đổi thứ tự list) | lab −650 ± 131, sau đó −912 ± 136 | **không nộp**. ffnmerge: gh chậm hơn tm176 +265 (pSE 147). nhaudit f177: chậm hơn V17.8. |
| **V17.8** (`submit-snap-tm176p68` = root repo) | tm176 + FFN **p68**: DMA seed scale của Gate lên đầu lane, giữ chỗ bằng một exact-zero pin (AND 0 của một tile seed đã về, cộng +0.0 vào bản sao rms weight trên Main). Lệnh nạp rms weight về sau Up W, nên chuỗi x chạy trong cửa sổ Up W. | lab −633 (z −8.9, 6 lab). NH 11 vòng xen kẽ: FFN 197,276 so với 198,013 (−737 ± 230). Số học giống tm176 từng bit. | bảng **9.3217** (`561a3580`), cao nhất **9.3257** (`424e4432`), **bài nộp cuối `2fe86ca1` = 9.3192** (QKV 67,675 / AO 34,680 / FFN 197,630) |
| V17.9 (`submit-snap-v179`) | V17.8 + FFN dcv12p (scale của Down nạp sau cùng, áp bằng CE) | lab −0.8..−1.1k nhưng NH +5 ± 245; run0 chậm thêm +1.0..+1.8k | **không nộp** (`ab179_report.md`) |

**Thống kê nộp bài.** Mã nguồn V17.8 được chấm 132 lần trong 4 đợt nộp tự động (`experiments/merge19/`, các log đã xoá). Kết quả: trung bình 9.217, SD 0.054, min 8.844, max 9.3257. Chỉ 6 lần vượt 9.30. Một lần chấm dao động khoảng ±0.05 với cùng mã nguồn. Bảng xếp hạng cuối chấm lại **bài nộp cuối cùng** của team.

**Khoảng cách tới top.** Ngày 09-26, hạng 1 đạt 9.899× (63.7k / 32.0k / 190.0k). Ngày 09-28, có ảnh cho thấy 10.293× (61,555 / 30,421 / 183,835). V17.8 đang ở khoảng 67.7k / 34.8k / 197.3k.

### 9.2 Các KEEP sau V17.5 và việc chúng có đứng vững hay không

- **p68** (fgs) đứng vững cả ở lab lẫn NH và được nộp.
- **dcv12p** (fsl) thắng ở lab nhưng hoà ở NH. Nguyên nhân là run0 chậm hơn mỗi khi đổi chương trình, và lab bỏ qua run0. Không dùng.
- **r5a11g + gh** (bản V17.6/V17.7 của mình) thắng V17.5 ở lab, nhưng thua tm176 của đồng đội. Đã thử ghép hai nhánh (ffnmerge, 11 arm, 6 lab, 432 launch) và không arm nào thắng tm176.
- **nhaudit** đã đo lại chuỗi KEEP V17.x bằng NH (60 job, mỗi lần hoàn tác một kernel về bản cũ). Mọi bản cũ đều chậm hơn, gồm QKV x4/vtail, q19n2, qf7, AO trước aoYG và FFN V17.5. Vậy các KEEP trước đó là thật. AO không đổi từ V12 (aoYG).

### 9.3 KHÔNG làm lại (09-26 tối → 10-01, đều có số đo)

**QKV**

| ý tưởng | track | kết quả | kết luận |
|---|---|---|---|
| Dời DMA nhỏ (q_scale, k_scale, k_rms, q_rms) bằng exact-zero / WAW pin: thêm 2 pass Main mới (h21d) hoặc gộp vào pass có sẵn (h21e = r3m0 của BACKWARD_QKV) | qkv21 | +2,739 (+4.0 %, job 122277); +4,537 (+6.6 %, job 122460). Hai tập 24 mẫu không chồng lên nhau. | CÓ HẠI. Đóng **cả họ "pin để dời small"** cho QKV: pass pin trên Main đắt hơn phần lane tiết kiệm được. |
| Memset spacer (h21a) | qkv21 | static +684, HW memset dài ~6× static | ĐÓNG BẰNG PHÂN TÍCH |
| Đổi vai toán hạng (operand swap) trong `mean_square` để dời q_scale/k_scale (h22a) | qkv22 | 0 lệnh thêm, list DMA không đổi, static +311 | ĐÓNG BẰNG PHÂN TÍCH |
| Rút pass khỏi chuỗi pin Q→K (kz0/kz): h23a, h23b và 2 cơ chế khác | qkv23 | 3 arm chết ở static, 1 arm thua trên HW | ĐÓNG |
| Chia cluster theo projection: Q trên c0, K+V trên c1 (E1) | q20, q20b | visa từ chối: "tiling a non-in-slice axis is not lowered yet" | KHÔNG BIỂU DIỄN ĐƯỢC |
| Mean square của V không qua gather, trên layout KvPart4 (T2-2) | qkvT22 | job 135173 thua | CÓ HẠI |
| Small loads chỉ trên PE0, fan-out bằng Transpose{64,4} | qkvPE0 | a5 static 82,996 (control 38,008); job 135325 thua | CÓ HẠI |
| Layout trọng số K3 = 3 × 1280 B | qpad | DMA-only +1,807 (SE 89, job 138616): 568 B/tick so với 586 | CÓ HẠI |
| K15-PAD (reduce có padding) | qpad | a1 +755, a2 +2,486 (job 138668) | CÓ HẠI |
| Đặt mỗi cluster lên một HBM stack riêng (QKV-S, AO-S) | p0s (P0) | +14.7k (job 117951). Một cluster đọc một stack chỉ đạt ~205 B/tick; chip đạt 651–655 B/tick khi cả hai cluster đọc cả hai stack. | CÓ HẠI. Đóng mọi layout theo stack. |

**AO** (vẫn là aoYG từ V12)

| ý tưởng | track | kết quả | kết luận |
|---|---|---|---|
| Rút ngắn đuôi c1 hoặc canh thời điểm sync ở cuối kernel | ao20 | Ở INFO, c0 qua barrier nhanh trong mọi launch; c1 xong trước c0 0.82–1.05k trong 48/48 launch | Không còn gì để lấy (c1 không nằm trên đường găng) |
| ws + nw xếp sau tile b, rồi gather, rồi res (A1) | ao22 | job 122199: +1.1..+3.6k, thua 12/12 block | CÓ HẠI |
| Bỏ gather cục bộ của c0 (A2); tách gather sau contraction theo H%120 (G2) | ao22 | "cannot factorize common axis H" | KHÔNG BIỂU DIỄN ĐƯỢC |
| Cho các DMA engine chạy chồng nhau ở đầu kernel (H4/GL-1) | ao22 | Có trên HW, nhưng mọi lệnh DMA trong 0.8.1 đều mang engine 0 | KHÔNG BIỂU DIỄN ĐƯỢC |
| Filler chain sau contraction a, sweep ws/tile b (54 biến thể) | ao21 | Không có KEEP; một DMA nhỏ bất kỳ đặt giữa hai tile tốn ~1k của T0 | ĐÓNG |
| AO-S (fan-out theo seed nhỏ hơn) | aos | Dừng theo kết quả P0 | ĐÓNG |
| Cỡ tile O: 104+16 hàng, 100+20 hàng | V17.6 notes | +1,435; +906 (0/20) | CÓ HẠI |
| Chia epilogue cho 2 cluster, đổi chéo tổng bình phương bằng DM→DM 2 chiều | V17.6 notes | Thiết bị treo ("rank 0 left request unanswered") | CÓ HẠI. Không dùng DM→DM hai chiều giữa hai cluster. |

**FFN**

| ý tưởng | track | kết quả | kết luận |
|---|---|---|---|
| H-split Down (mỗi slice đọc trọn hàng H, scale căn 256 B) | r5h, codex-hsplit, downrw (b), hsp | Mô hình: +2.4..+9.9k (cần broadcast h lần 2, 15 KB/slice, ~7.1k Main). DMA-only: dH6 +3,213 và +3,137 (job 141641/141672); dHw +6,210. Chỉ riêng phần DS căn hàng được −1,160. | CÓ HẠI. Static báo ngược dấu (−1.4k). Hàng 7,680 B có stride chẵn nên hai cluster cùng đọc một stack. |
| H-split với activation f8 một lớp | downrw (a) | Tỉ lệ sai số 1.61 / 2.45 / 2.38 (ngưỡng 1.0) | CÓ HẠI (sai số học) |
| H-split với activation bf16 một lớp | downrw (a') | Không có lookup bf16 × f4 | KHÔNG BIỂU DIỄN ĐƯỢC |
| CE áp scale cho 4 hàng Down cuối | codex-r5b | static 102,866 so với 102,422 | ĐÓNG BẰNG PHÂN TÍCH |
| Chia trọng số Gate thành 2 tile | codex-x5 | 12+18: "cannot factorize common axis L"; 15+15: static +6.2k | ĐÓNG |
| X8 với chia 12/3, 13/2; pin ở đầu kernel | r5 | +1,134; +4,946; pin +0.4k | CÓ HẠI |
| gq một mình (global scale lên đầu) | ffn21 | −0.3k, z 1.63 | TRUNG TÍNH |
| Ghép R5 / h1 / gq / sg-pin lên tm176 (m1, m2n, m3n, t1g, t1h, t1s, t1b) | ffnmerge | Không arm nào thắng tm176 | ĐÓNG |
| Đổi layout đọc Down weight / DS dưới L-split | ffndeep | DS đã ở sàn số request (1,212 + 3.2 × request mỗi engine) | ĐÓNG BẰNG PHÂN TÍCH |
| DSL: scale Down nạp sau cùng nhưng dùng bảng tra thứ 3; các cách chia piece 12/3, 10/5 | downrw, ffnx | +2.0..+2.3k; dsl12 +3,197; dsl10 +280. Piece phải chia hết 15 (PROG-719). | CÓ HẠI |
| Up + Gate trong một pass lookup (ug) | ffnx | Ra non-finite (FAIL) | CÓ HẠI |
| Thêm tile Down chỉ cho c0 (c1 nhẹ hơn) | ffnx | visa từ chối view và cluster tile chỉ của c1 | KHÔNG BIỂU DIỄN ĐƯỢC |
| Chia piece Down cuối thành 2+2, 3+1; bỏ pass add | ffnx | k22 +6,256; k31 +6,439; na lỗi compile | CÓ HẠI |
| Tái dùng bảng tra, bỏ StoTab, partial wait, fence | fbyp | Không có API nào trong 0.8.1 (bảng tra chọn theo kiểu dữ liệu; lowering đóng) | KHÔNG BIỂU DIỄN ĐƯỢC |
| Độ rộng epilogue e32, e64 | f20 | +405; +390; +4,354 | CÓ HẠI |
| Các biến thể đầu kernel: hgc, hzn, hnc; pin trên pass sqrt | f20 | +1,175; +442; +148; lỗi compile | CÓ HẠI |
| Gộp global scale của Gate (gf, hgagf); add c1 qua VRF (ea) | f20 | +1,110; +1,434; ea lỗi compile (pass final chỉ chứa một trong AddF/FmaF) | CÓ HẠI |
| GELU ghi thẳng VRF (geluvrf); gộp gate-scale vào switch (gatefuse); gvdirect | ffnGG, codex_ffn30 | Không thắng (job 135253); gvdirect: "commit_trim packet must be 8/16/24/32 B"; gathertrf lỗi compile | ĐÓNG |
| Các vị trí khác cho Gate-seed pin: p87, p60, p106, q68, p68b, l/lb/g/w | fgs | Chỉ p68 thắng; q68 +223; p68b −253 (null) | Chỉ giữ p68 |
| Tìm có hệ thống vị trí exact-zero pin (31 arm, 9 lab) | fpl | Không arm nào thắng p68; tốt nhất s2 −276 (2/4) | **p68 là cực trị địa phương** |
| Đẩy raw partial ở đuôi | fpl | +0.6k sau sync | CÓ HẠI |
| SA256 (sở hữu theo atom scale 256) | fsa | DMA-only +6,502 (job 137629) | CÓ HẠI |
| Chuyển vị theo phase P8 | fsa | +1.3k | CÓ HẠI |
| r2k16 | fsa | static +58 % | ĐÓNG BẰNG PHÂN TÍCH |
| commit_cast (ccA, ccB) | fsa | −0.2..−0.35k, chưa đạt luật KEEP | TRUNG TÍNH |
| Tất cả scale nạp sau cùng (ASL); dcv13p, dcf10p | fsl | ASL không biểu diễn được (+13..+36k static); dcv13p, dcf10p chết | ĐÓNG |
| dcv12p / dcv10p (scale Down sau cùng, áp bằng CE) | fsl, int179, ab179 | lab −0.8..−1.1k; NH +5 ± 245 (11 vòng) | TRUNG TÍNH trên NH. Không dùng. |
| Gộp hằng số giữa các toán tử (b1, g2, g12, gd, b1gd) | alg | g2 −517 rồi +540; gd −375 rồi −4; b1 làm hỏng bit (r0 max 0.0078) | TRUNG TÍNH / CÓ HẠI |
| DMA vào một tile của DM tensor dùng chung (write-hole `#{!}`) | dmat | Hoặc chạy nối tiếp và chép trọn extent, hoặc panic (`dma_layout.rs:53`), hoặc hỏng dữ liệu nếu thứ tự ghi không may | KHÔNG BIỂU DIỄN ĐƯỢC |

### 9.4 Bài học mới (bổ sung §6.9 và §7)

1. **Lab cùng job đánh giá quá cao các biến thể FFN đổi chương trình** (ab179).
   - Run0 sau khi đổi chương trình chậm thêm +1.0..+1.8k, nhưng lab bỏ run0.
   - Với thay đổi FFN, quyết định bằng NH xen kẽ, ghép cặp, **≥ 10 job mỗi arm**.
2. **Pin trên Main đắt trong QKV**, kể cả khi static sạch và số học đúng từng bit (qkv21: +4 %, +6.6 %).
3. **Static model sai dấu với layout DMA đọc trọn hàng** (hsp: static −1.4k, HW +3.2k). Parity của bit 8 (stack HBM) quyết định tốc độ stream:
   - L-split đặt hai cluster lên hai stack khác nhau, vì offset nửa hàng là 15 unit (lẻ);
   - layout đọc trọn hàng luôn có stride chẵn, nên chậm hơn.
4. **Giới hạn biểu diễn của 0.8.1 gặp lại trong vòng này:**
   - không `cluster_tile` trên trục không nằm trong slice;
   - "cannot factorize common axis";
   - mỗi pass VE chỉ một AddF, và pass final chỉ chứa một trong AddF/FmaF;
   - commit_trim chỉ nhận packet 8/16/24/32 B;
   - piece DSL phải chia hết 15;
   - không có API cho bảng tra, StoTab, partial wait hay fence;
   - mọi lệnh DMA đều mang engine 0.
5. **Điểm chính thức dao động khoảng ±0.05 mỗi lần chấm** với cùng mã nguồn (132 lần, SD 0.054). So sánh phiên bản bằng lần chấm đơn lẻ là vô nghĩa.

### 9.5 Ý tưởng còn mở cho vòng sau (chưa ai đo)

- **QKV.** Cần giảm số lệnh DMA trên cluster găng, không phải dời chúng. Mỗi small tốn khoảng 1k dù đặt ở đâu (q20 §Gap budget). Muốn tách cluster theo projection thì cần một toolchain cho phép `cluster_tile` trên trục cluster.
- **AO.** Chồng engine ở đầu kernel và tách gather đều cần API mới.
- **FFN.** Khoảng 2.3k StoTab trên g0, cùng fused4 chờ Main khoảng 2.8k, vẫn chưa có handle nào trong mã nguồn.
- Nếu vòng 2 dùng toolchain mới, đọc lại §6.7 và §9.4.4 trước: các giới hạn ở đó có thể đã được gỡ.
