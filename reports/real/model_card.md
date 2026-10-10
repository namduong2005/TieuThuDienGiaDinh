# Thẻ mô hình



## Mục đích

Phân nhóm các ngày sử dụng điện của hộ gia đình dựa trên hồ sơ công suất 24 giờ. Mô hình chỉ mang tính mô tả, không chứng minh quan hệ nhân quả và không đại diện cho hành vi của mọi hộ gia đình.

## Cấu hình được chọn

- Chế độ tiền xử lý: `absolute`
- Số cụm: `2`
- Nguồn dữ liệu: `data\raw\household_power_consumption.txt`
- Silhouette trên test: `0.1825`
- Khoảng cách trung bình tới tâm cụm trên test: `3.6114`
- Có dùng test để chọn mô hình: `False`

## Kích thước cụm

- Cụm 0: 432 ngày
- Cụm 1: 988 ngày

## Độ biến thiên qua seed

- Số seed dùng khi chọn cấu hình: `10`
- Silhouette validation: trung bình `0.2625`, độ lệch chuẩn `0.0000`.
- Khoảng silhouette validation qua seed: `0.2625` đến `0.2625`.
- ARI giữa các seed: trung bình `1.0000`, độ lệch chuẩn `0.0000`.

## Giới hạn

- Dữ liệu chỉ từ một hộ tại Pháp, không được khái quát cho toàn bộ dân số.
- Số hiệu cụm chỉ là mã định danh, không phải thứ hạng.
- Chuẩn hóa hình dạng sẽ loại bỏ thông tin về mức tiêu thụ tuyệt đối nếu được chọn.
- Chỉ giữ các ngày thiếu không quá ngưỡng công bố; giá trị thiếu được điền bằng tham số học từ train.
- Mô hình mô tả độ tương đồng, không chứng minh quan hệ nhân quả.
