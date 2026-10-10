# Thẻ dữ liệu

## Nguồn và phạm vi

- Bộ dữ liệu: Individual Household Electric Power Consumption.
- Nguồn: UCI Machine Learning Repository; DOI: 10.24432/C58K54; giấy phép CC BY 4.0.
- EDA mô tả chỉ dùng tập train trong khoảng 2009-01-01 đến 2009-05-24 để tránh nhìn trước validation/test.
- Tệp nguồn trong dự án: `synthetic_demo`.

## Quy mô và chất lượng

- Số dòng đo theo phút ban đầu: 5760.
- Số ngày theo lịch: 240; giữ lại: 240; loại: 0.
- Dòng thiếu Global_active_power: không có.
- Timestamp trùng: 0; công suất âm: 0.
- Ngày ngoại lệ theo IQR của công suất trung bình: 11.
- Ngày bất thường theo khoảng cách tới tâm cụm: 11 (4.6%).

## Biến dùng trong học máy

- Mỗi ngày là 24 đặc trưng `hour_00` đến `hour_23`, là Global_active_power trung bình theo giờ, đơn vị kW.
- Ba biến sub-metering được đổi từ Wh sang kWh/ngày và chỉ dùng để mô tả cụm, không dùng huấn luyện.
- Ngày trong tuần, cuối tuần, tháng và mùa chỉ dùng cho baseline hoặc phân tích sau phân cụm.

## Chia dữ liệu và tiền xử lý

- Train: 144 ngày; validation: 48 ngày; test: 48 ngày.
- Chia theo thứ tự thời gian. Imputer và scaler được học từ train khi chọn mô hình.
- Test không được dùng để chọn K hay chế độ chuẩn hóa.

## Giới hạn sử dụng

- Dữ liệu chỉ từ một hộ nên không đại diện cho mọi hộ gia đình hoặc quốc gia.
- K-means mô tả các mẫu giống nhau, không chứng minh nguyên nhân tiêu thụ điện.
- Nhãn cụm là mã định danh; cần đọc biểu đồ tâm cụm để diễn giải.
- Ngày bất thường là ngày xa tâm cụm về mặt thống kê, không mặc nhiên là lỗi hay gian lận.
