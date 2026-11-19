# Báo cáo Day 09 — LeTrungSang, 2A202603125

Mỗi số liệu dưới đây lấy từ các báo cáo trong `submission/reports/`.

## 1. Kết luận trước
Pipeline kết luận **Hold**. Có 224 ảnh đầu vào, 193 ảnh còn lại sau intake và 193 ảnh được ghi trong bản xuất nháp `clean_v1`. Hold vì còn 34/193 ảnh có cảnh báo chưa được người duyệt, tương đương 17.62%, cao hơn ngưỡng 10%.

## 2. Lỗi và phân việc
Lấy từ `02_error_list_v1` (gom theo mã lỗi). Cột "ai sửa": **máy** (autofix) hay **người** (hàng đợi); chỉ ghi những mã có mặt trong lô của bạn.

| Mã lỗi | Số ảnh/dòng | Ai sửa (máy/người) | Vì sao |
|---|---:|---|---|
| MISSING_LABEL_FILE | 6 ảnh | máy chuyển người | Máy không tự bịa nhãn |
| BAD_CLASS_ID | 6 dòng | máy | Xóa dòng có ID ngoài phạm vi |
| BOX_OUT_OF_IMAGE | 8 dòng | máy | Clip box về trong ảnh |
| BOX_ZERO | 4 dòng | máy | Xóa box có kích thước bằng 0 |
| BAD_FORMAT | 3 dòng | máy | Xóa dòng sai định dạng |
| DUP_BOX | 7 dòng | máy | Xóa box trùng |
| BAD_CLASS_NAME | 2 lỗi | máy | Ghi lại danh sách class chuẩn |
| POSSIBLE_MISSING | 53 cảnh báo | người | Model phát hiện vật không có box |
| POSSIBLE_LOOSE_BOX | 27 cảnh báo | người | IoU của box thấp |
| POSSIBLE_WRONG_CLASS | 17 cảnh báo | người | Model và nhãn khác class |
| POSSIBLE_FALSE_POS | 28 cảnh báo | người | Box nhãn không khớp model |

`05_review_queue.csv` xếp theo priority giảm dần, dùng trọng số trong config; `NEEDS_LABEL` và các lỗi ngữ nghĩa nặng được ưu tiên hơn cảnh báo nhẹ. Hàng đợi có 121 ảnh, nhưng ngân sách chỉ 80 ảnh nên mô phỏng đã duyệt 80 ảnh.

## 3. Calibration (2 ngưỡng)
| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | 16 | precision/recall duplicate gần: 1.00/0.50 → 1.00/0.83 | Chọn F1 tốt hơn; F1 tăng từ 0.67 lên 0.91 |
| `second_opinion.conf` | 0.25 | 0.6 | missing: 0.50/1.00 → 0.57/1.00; wrong class: 0.56/0.83 → 0.62/0.83 | Cân bằng precision tốt hơn mà vẫn giữ recall missing |

Calibration chỉ có 39 ảnh nên có thể không đại diện cho dữ liệu thật. Với `phash_dist=20`, precision vẫn 1.00 nhưng recall duplicate gần giảm còn 0.50; tối ưu một chỉ số trên calib không đảm bảo ít false positive hoặc tốt hơn trên raw.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
Với `python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb`, pipeline kết luận **Rework**, mã thoát là **20**, và `work/exp_nb/clean_v1/` không được tạo vì còn 6 lỗi bắt buộc. Gate dùng lỗi bắt buộc để chặn trước khi xét Hold.

Điểm mù: model checker có thể cùng sai với nhãn nên nhầm lớp không bị cảnh báo; box có IoU nằm trên ngưỡng vẫn có thể lỏng về mặt ngữ nghĩa; perceptual hash có thể bỏ sót ảnh bị biến đổi mạnh.

## 5. Train: raw vs clean (tùy chọn, chấm riêng)
Chưa train: ghi "chưa chạy" và lý do, rồi bỏ qua phần còn lại của mục này. Nếu đã train: dán bảng từ `submission/results/comparison.md` (model, epoch, seed, số ảnh train mỗi biến thể).

Chưa train; phần so sánh raw/clean sẽ thực hiện sau trên Colab với hai seed. Vì chưa có mAP50 và độ lệch chuẩn nên chưa kết luận clean tốt hơn raw.

## 6. Quyết định và ghi chú AI
Quyết định hiện tại là **Hold**. Để thành Release cần người duyệt thêm các ảnh còn lại trong hàng đợi hoặc giảm cảnh báo bằng kiểm tra nhãn thực tế, sau đó chạy lại pipeline và đạt warning ratio dưới 10%.

Có dùng AI để triển khai ba module còn thiếu theo docstring và test. Code được kiểm bằng test module, test e2e và toàn bộ 79 test; một lỗi môi trường ban đầu là dependency `imagehash` chưa được cài, đã cài theo `requirements.txt` rồi chạy lại thành công.

*(Thưởng +2, tùy chọn)* CVAT: lọc `width < 20`, ảnh/số box khớp, đối chiếu `BOX_TINY` (xem docs/CVAT_FILTER.md), 1 câu về `can_review`.
