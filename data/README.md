# Hồ sơ dữ liệu

- Tên: Individual Household Electric Power Consumption
- Nguồn: UCI Machine Learning Repository
- Trang dữ liệu: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
- DOI: https://doi.org/10.24432/C58K54
- Giấy phép tại thời điểm viết dự án: CC BY 4.0; cần kiểm tra lại trên trang UCI khi nộp.
- Tệp sử dụng: `household_power_consumption.txt`
- Biến chính: `Date`, `Time`, `Global_active_power`
- Biến mô tả bổ sung: `Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3`
- Ngày tải cho lần chạy hiện tại: 2026-09-29
- Nguồn tải dự phòng đã dùng: `https://raw.githubusercontent.com/jbrownlee/Datasets/master/household_power_consumption.zip`
- SHA-256 của ZIP: `9F84B46ADE8A2D8E1286EC4B2B6C2987A45A755C59F263BE3B3B3D10DFBDA3FF`
- Kiểm tra đối chiếu: 2.075.259 dòng, 9 thuộc tính, khớp mô tả UCI.

Không commit dữ liệu thô hoặc dữ liệu đã xử lý lên Git. Dùng `py -m src.data --download --prepare` để tái tạo.

## Data dictionary của dữ liệu đã xử lý

- `date`: ngày quan sát.
- `hour_00` đến `hour_23`: công suất tác dụng trung bình theo từng giờ, đơn vị kW.
- `missing_hours`: số giờ thiếu trước bước impute.
- `day_of_week`: thứ trong tuần, chỉ dùng mô tả sau phân cụm.
- `is_weekend`: 1 nếu thứ bảy/chủ nhật, chỉ dùng baseline và mô tả.
- `month`: tháng.
- `season`: mùa quy ước phục vụ mô tả.
- `sub_metering_1_kwh` đến `sub_metering_3_kwh`: điện năng từng nhóm thiết bị,
  đơn vị kWh/ngày; chỉ dùng để mô tả cụm, không dùng huấn luyện.

`reports/data_quality.json` ghi số dòng thiếu, timestamp trùng, giá trị âm và số
ngày ngoại lệ. `reports/removed_days.csv` liệt kê các ngày bị loại cùng lý do.
