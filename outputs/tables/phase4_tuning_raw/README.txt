phase4_tuning_raw/ — kết quả tuning Phase 4 (notebook 04_predictive_modeling.ipynb)
=====================================================================================

FILE ĐANG DÙNG (có mã đuôi 8 ký tự):
  LogReg_A_14c32a35.csv        LogReg_B_39c7f2e8.csv
  RandomForest_A_c90ddba2.csv  RandomForest_B_617e3dbd.csv
  XGBoost_A_58143df3.csv       XGBoost_B_6afb8dce.csv
  Mỗi dòng = 1 điểm trong lưới tham số × 1 fold (Macro-F1, accuracy trên validation).
  Mã đuôi được tính từ lưới tham số + danh sách feature trong src/modeling.py;
  nếu lưới thay đổi, notebook tự tạo file mới với mã khác.

FILE CŨ — KHÔNG CÒN DÙNG, chỉ giữ để tham khảo:
  LogReg_A.csv, LogReg_B.csv
  Lần tuning đầu tiên, khi class_weight="balanced" còn bị cố định và lưới C chỉ có
  {0.01, 0.1, 1, 10}. Kết quả cho thấy cân bằng lớp làm Logistic Regression kém đi
  (accuracy fold 1 ≈ 0.59), nên class weighting được chuyển thành tham số tuning.
  Đã được thay thế bởi LogReg_A_14c32a35.csv và LogReg_B_39c7f2e8.csv.
  Notebook không đọc hai file này. Xem báo cáo tiến độ, mục 4.3.
