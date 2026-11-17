# Thử nghiệm: RFM theo cửa sổ 180 ngày (phân tích độ nhạy)

Thư mục này chạy lại toàn bộ Phase 1–6 với **một thay đổi duy nhất** so với thiết kế chính:
Frequency và Monetary (dùng để tạo trạng thái và làm feature dự đoán) được tính trong **180 ngày gần nhất**
trước mỗi snapshot, thay vì cộng dồn từ đầu. Mọi thứ khác giữ nguyên: snapshot, xử lý hủy đơn theo snapshot,
quy tắc chọn k (Elbow/Kneedle theo đa số 6 lần chạy), cách cố định trạng thái trên S1–S5, cấu hình A/B,
lưới tham số, fold tinh chỉnh, bootstrap 5.000 lần với hiệu chỉnh Holm, và mọi giả định của Phase 6.

**Pipeline chính, notebook 01–06, kết quả và report KHÔNG bị thay đổi.**

## Cách chạy (từ thư mục này)

```bash
python step1_panel.py                 # panel với F, M trong 180 ngày
python step2a_kselect.py 0 ... 5      # chỉ số chọn k cho 6 lần chạy (có cache)
python step2b_states.py               # quyết định k, fit và cố định trạng thái, mô tả, drift
python step3_transitions.py           # ma trận chuyển, kiểm định, bộ dữ liệu dự đoán, fold tinh chỉnh
python step4a_tune.py <Model> <A|B>   # tinh chỉnh từng model (LogReg, RandomForest, XGBoost) × (A, B)
python step4b_test.py                 # fit cuối, đánh giá một lần trên test S6–S7
python step5_bootstrap.py             # H1, H2, so với persistence
python step6_decision.py              # rủi ro × giá trị, mô phỏng ngân sách
python step7_compare.py               # bảng + hình so sánh với thiết kế chính
python step8_compare_k.py             # bảng so sánh 3 thiết kế (cần chạy xong cả k = 4 và k = 3)
```

### Chạy lại với k = 3

Đặt biến môi trường `WINDOW_K=3` rồi chạy lại từ `step2b` đến `step6` (bước 1 và 2a dùng chung, không cần chạy lại):

```bash
export WINDOW_K=3
python step2b_states.py
python step3_transitions.py
python step4a_tune.py <Model> <A|B>   # 6 tổ hợp
python step4b_test.py
python step5_bootstrap.py
python step6_decision.py
unset WINDOW_K
python step8_compare_k.py
```

Với `WINDOW_K=3`, kết quả ghi vào `outputs_k3/`, `models_k3/`, `data/k3/` và cache tinh chỉnh riêng, nên không ghi đè kết quả k = 4.
Ba trạng thái được đặt tên: Inactive (không mua trong 180 ngày), Occasional, Active.

## Đầu ra
- `outputs/tables/comparison_main_vs_window180.csv` — bảng so sánh 24 chỉ số giữa hai thiết kế
- `outputs/tables/comparison_main_window_k4_k3.csv` — bảng so sánh ba thiết kế (cộng dồn k = 4, cửa sổ k = 4, cửa sổ k = 3)
- `outputs_k3/`, `models_k3/` — toàn bộ kết quả và model của lần chạy k = 3
- `outputs/figures/compare_state_shares.png`, `compare_vs_persistence.png` — hình so sánh
- `outputs/tables/step*_*.csv`, `outputs/figures/step*_*.png` — kết quả từng bước (cùng cấu trúc với Phase 2–6)
- `data/`, `models/`, `cache/` — dữ liệu trung gian, model đã huấn luyện, cache (tạo lại được)

## Kết quả chính (so với thiết kế chính)

| Khía cạnh | RFM cộng dồn (chính) | Cửa sổ 180 ngày |
|---|---|---|
| Nhóm Active High-Value S1 → S8 | 7% → 23% (tăng cơ học) | 17% → 14% (dao động theo mùa) |
| Silhouette trạng thái cố định, S6–S8 | 0,34–0,38 | 0,58–0,65 |
| Ô bằng 0 trong ma trận chuyển | 2 | 0 |
| Khách giữ nguyên trạng thái mỗi quý | 72% | 58% |
| Persistence: Macro-F1 test | 0,649 | 0,561 |
| Model tốt nhất − persistence | +0,008 [−0,004; +0,020], không có ý nghĩa | **+0,027 [+0,015; +0,039], có ý nghĩa** |
| Số model vượt persistence có ý nghĩa | 0/6 | 4/6 (RF, XGBoost cả A và B) |
| Khách đổi trạng thái dự đoán đúng | 17–27% | 38–41% |
| H1 / H2 | H1 ủng hộ một phần (LR) / H2 không | H1 không / H2 ủng hộ (Δ RF ≠ Δ LR) |
| Doanh thu mất từ khách tụt trạng thái | 19% | 32% |
| L_stop bắt được, top 10%: rủi ro trạng thái × giá trị | 43,0% | 45,3% (không hơn "chi nhiều nhất" có ý nghĩa) |
| L_stop bắt được, top 10%: model ngừng mua × giá trị | 60,5% | 61,0% (+17,2 điểm, có ý nghĩa) |

**Tóm tắt:** cửa sổ 180 ngày cho trạng thái phân tách rõ hơn, ổn định hơn ngoài mẫu, không còn tăng cơ học,
và model dự đoán trạng thái vượt persistence có ý nghĩa. Tuy nhiên kết luận về ra quyết định không đổi:
model ngừng mua trực tiếp vẫn là cách chọn khách tốt nhất, còn rủi ro dựa trên trạng thái vẫn không hơn
cách "chọn khách chi nhiều nhất".

## Kết quả với k = 3

| Khía cạnh | Cửa sổ 180 ngày, k = 4 | Cửa sổ 180 ngày, k = 3 |
|---|---|---|
| Trạng thái | Dormant / Lapsing / Recent / Active | Inactive 18% / Occasional 55% / Active 27% (Recent gộp vào Active) |
| Nhóm Active S1 → S8 | 17% → 14% | 36% → 24% |
| Silhouette trạng thái cố định, S6–S8 | 0,58–0,65 | 0,62–0,66 |
| ARI cố định so với fit lại | 0,84–0,97 | 0,93–0,98 |
| Khách giữ nguyên trạng thái mỗi quý | 58% | 65% |
| Persistence: Macro-F1 test | 0,561 | 0,681 |
| Model tốt nhất − persistence | +0,027 [+0,015; +0,039] | +0,021 [+0,011; +0,031] (LogReg B) |
| Số model vượt persistence có ý nghĩa | 4/6 | 5/6 (trừ XGBoost A) |
| Khách đổi trạng thái dự đoán đúng | 38–41% | 34–38% |
| H1 / H2 | không / ủng hộ (RF − LR) | ủng hộ với XGBoost (Δ = +0,005, p Holm 0,019) / không |
| Doanh thu mất từ khách tụt trạng thái | 32% | 26% |
| L_stop top 10%: rủi ro trạng thái × giá trị | 45,3% | 40,0% |
| L_stop top 10%: model ngừng mua × giá trị | 61,0% | 59,9% (+16,1 [11,9; 20,2] so với chi nhiều nhất) |

**Tóm tắt k = 3:** trạng thái ổn định nhất trong ba thiết kế và hầu hết model vượt persistence; kết luận về ra quyết định
giữ nguyên. Report giữ k = 4 làm thiết kế độ nhạy chính vì đúng quy tắc đã định và giữ được trạng thái trung chuyển Recent.

## Lưu ý khi đưa vào report
1. **Chọn k kém rõ ràng hơn:** Kneedle chọn k = 4 (6/6) nhưng sai phân bậc hai và Silhouette chọn k = 3 (6/6). Thử nghiệm giữ k = 4 theo quy tắc đã định và để so sánh công bằng, đồng thời chạy lại toàn bộ với k = 3 (xem mục trên); kết luận không đổi.
2. **S1 bị cắt:** dữ liệu bắt đầu 01/12/2009 nên S1 chỉ có 121 ngày trong cửa sổ; không có khách Dormant ở S1.
3. **Nhóm Dormant tăng dần (0% → 41–46%)** vì khách cũ không bao giờ rời panel — đây là khách bỏ đi tích lũy thật, không phải lỗi.
4. **Không so trực tiếp Macro-F1 giữa hai thiết kế**: hai cách định nghĩa trạng thái tạo hai bài toán khác nhau. Hãy so *mức hơn persistence* và *doanh thu giữ lại ở Phase 6* (cùng thước đo cho cả hai).
5. **Độ dài cửa sổ 180 ngày** được chọn sau một lần sàng lọc nhanh 90/180/365 ngày (90 ngày quá nhiễu, 365 ngày làm một cụm < 5% và các snapshot đầu chưa đủ một năm). Đây là phân tích độ nhạy, nên ghi rõ như vậy.
