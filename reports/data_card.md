# Thẻ dữ liệu

## Nguồn và phạm vi

- Bộ dữ liệu: Individual Household Electric Power Consumption.
- Nguồn: UCI Machine Learning Repository; DOI: 10.24432/C58K54; giấy phép CC BY 4.0.
- Dữ liệu mô tả một hộ gia đình tại Pháp trong khoảng 2006-12-17 đến 2010-11-26.
- Tệp nguồn trong dự án: `data\raw\household_power_consumption.txt`.

## Quy mô và chất lượng

- Số dòng đo theo phút ban đầu: 2075259.
- Số ngày theo lịch: 1442; giữ lại: 1420; loại: 22.
- Dòng thiếu Global_active_power: 25979.
- Timestamp trùng: 0; công suất âm: 0.
- Ngày ngoại lệ theo IQR của công suất trung bình: 32.
- Ngày bất thường theo khoảng cách tới tâm cụm: 67 (4.7%).

## Biến dùng trong học máy

- Mỗi ngày là 24 đặc trưng `hour_00` đến `hour_23`, là Global_active_power trung bình theo giờ, đơn vị kW.
- Ba biến sub-metering được đổi từ Wh sang kWh/ngày và chỉ dùng để mô tả cụm, không dùng huấn luyện.
- Ngày trong tuần, cuối tuần, tháng và mùa chỉ dùng cho baseline hoặc phân tích sau phân cụm.

## Chia dữ liệu và tiền xử lý

- Train: 852 ngày; validation: 284 ngày; test: 284 ngày.
- Chia theo thứ tự thời gian. Imputer và scaler được học từ train khi chọn mô hình.
- Test không được dùng để chọn K hay chế độ chuẩn hóa.

## Giới hạn sử dụng

- Dữ liệu chỉ từ một hộ nên không đại diện cho mọi hộ gia đình hoặc quốc gia.
- K-means mô tả các mẫu giống nhau, không chứng minh nguyên nhân tiêu thụ điện.
- Nhãn cụm là mã định danh; cần đọc biểu đồ tâm cụm để diễn giải.
- Ngày bất thường là ngày xa tâm cụm về mặt thống kê, không mặc nhiên là lỗi hay gian lận.
