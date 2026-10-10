# Phân nhóm ngày theo hồ sơ tiêu thụ điện

Dự án xử lý dữ liệu Individual Household Electric Power Consumption, biến mỗi ngày thành vector 24 giá trị công suất trung bình theo giờ, rồi phân cụm bằng K-means.

## Cấu trúc

```text
project_24_hoc_may/
├── data/
│   ├── raw/                 Dữ liệu tải từ UCI, không đưa lên Git
│   ├── processed/real/      Hồ sơ ngày từ UCI
│   ├── processed/demo/      Hồ sơ mô phỏng
│   └── README.md
├── models/real|demo/        Pipeline K-means tách theo nguồn
├── notebooks/               Nơi đặt notebook EDA nếu cần
├── reports/
│   ├── real/                Báo cáo và 12 biểu đồ dùng nộp bài
│   └── demo/                Kết quả chỉ dùng thử chương trình
├── src/
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── eda.py
├── tests/
├── requirements.txt
├── run_demo.bat
├── run_api.bat
└── run_real.bat
```

## Chạy thử ngay bằng dữ liệu mô phỏng

Mở PowerShell tại thư mục dự án và chạy:

```powershell
py -m pip install -r requirements.txt
$env:POWER_PROJECT_VARIANT="demo"
py -m src.data --demo
py -m src.train
py -m src.evaluate
py -m src.eda
```

Hoặc nhấp đúp `run_demo.bat`.

Kết quả demo được ghi riêng vào `data/processed/demo`, `models/demo` và
`reports/demo`. Kết quả thật nằm trong các thư mục `real` tương ứng. Vì vậy chạy
demo không thể ghi đè mô hình hay báo cáo thật. Biến môi trường
`POWER_PROJECT_VARIANT` được các file BAT đặt tự động.

Kết quả chính:

- `data/processed/<variant>/daily_profiles.csv`: mỗi ngày và 24 cột giờ.
- `models/<variant>/kmeans_pipeline.joblib`: pipeline tiền xử lý và K-means.
- `reports/<variant>/experiments.csv`: kết quả K=2 đến K=8 và biến thiên qua seed.
- `reports/<variant>/summary.json`: cấu hình và metric cuối.
- `reports/<variant>/figures/`: 12 biểu đồ.
- `reports/<variant>/error_analysis_by_season.csv`: lỗi theo mùa.
- `reports/<variant>/error_analysis_by_period.csv`: lỗi theo quý.
- `reports/<variant>/data_dictionary.md`: từ điển dữ liệu đầy đủ.

Kết quả tạo bởi lệnh `--demo` chỉ dùng để kiểm tra chương trình. Không đưa metric hoặc biểu đồ demo vào báo cáo cuối.

## Chạy với dữ liệu UCI thật

```powershell
$env:POWER_PROJECT_VARIANT="real"
py -m src.data --download --prepare
py -m src.train
py -m src.evaluate
py -m src.eda
```

Hoặc nhấp đúp `run_real.bat`. Quá trình tải và xử lý dữ liệu thật có thể mất vài phút.

Nếu máy không tải tự động được, mở trang UCI trong trình duyệt, tải tệp ZIP và đổi tên thành:

```text
data/raw/individual_household_electric_power_consumption.zip
```

Sau đó chạy lại:

```powershell
$env:POWER_PROJECT_VARIANT="real"
py -m src.data --download --prepare
py -m src.train
py -m src.evaluate
py -m src.eda
```

`run_real.bat` thực hiện toàn bộ quy trình: chuẩn bị dữ liệu, chọn K, huấn luyện,
đánh giá, EDA chỉ trên train và tạo 12 biểu đồ tiếng Việt. Nếu chỉ muốn xem kết
quả đã tạo thì mở thư mục `reports/real/figures`, không cần chạy lại.

## Môi trường tái lập

- Python được khóa trong `.python-version`.
- Toàn bộ thư viện trong `requirements.txt` dùng phiên bản chính xác với `==`.
- Sau khi đổi môi trường, chạy lại kiểm thử trước khi huấn luyện.

## Quy trình chống rò rỉ dữ liệu

1. Resample và pivot là phép biến đổi xác định, tạo một dòng cho mỗi ngày.
2. Các ngày được sắp theo thời gian rồi chia 60% train, 20% validation và 20% test.
3. Imputer và scaler chỉ fit trên train khi chọn K.
4. Chọn chế độ chuẩn hóa và K bằng validation.
5. Sau khi chốt cấu hình, fit lại trên train + validation và chỉ đánh giá test một lần.

## Hai thí nghiệm chuẩn hóa

- `absolute`: giữ thông tin ngày dùng nhiều hay ít điện.
- `shape`: chia hồ sơ từng ngày cho trung bình của ngày, tập trung vào hình dạng và làm mất mức tiêu thụ tuyệt đối.

## Ghi chú

Nhãn cụm 0, 1, 2... chỉ là mã định danh, không thể hiện cụm nào tốt hơn. Phải xem đường centroid để đặt tên có ý nghĩa cho từng cụm.

## Kiểm thử

```powershell
py -m unittest discover -s tests -v
```

Các kiểm thử bao gồm chia dữ liệu theo thời gian, chống rò rỉ ở imputer, hai chế
độ chuẩn hóa, lưới K và một lượt huấn luyện–dự đoán từ đầu đến cuối.

## Chạy API

Sau khi đã có mô hình thật, nhấp đúp `run_api.bat` hoặc chạy:

```powershell
py -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Mở `http://127.0.0.1:8000/docs` để thử API trực tiếp. Endpoint chính là
`POST /predict`, nhận đúng 24 giá trị công suất trung bình theo giờ (kW). Có thể
để trống tối đa 2 giờ; pipeline sẽ điền bằng thống kê đã học từ dữ liệu train.

Ví dụ JSON:

```json
{
  "hourly_power": [
    0.45, 0.42, 0.39, 0.38, 0.37, 0.40,
    0.55, 0.82, 0.76, 0.63, 0.58, 0.56,
    0.54, 0.53, 0.55, 0.61, 0.72, 0.95,
    1.35, 1.72, 1.84, 1.50, 1.02, 0.68
  ]
}
```

Các endpoint:

- `GET /health`: kiểm tra API và mô hình.
- `GET /model-info`: thông tin cấu hình mô hình.
- `POST /predict`: phân cụm một ngày mới và kiểm tra mức bất thường.
- `GET /api/days?season=winter`: danh sách ngày để dựng lịch cụm.
- `GET /api/day-profile?date=2010-11-26`: hồ sơ 24 giờ của một ngày.
- `GET /api/dashboard`: số liệu đánh giá và model card.
- `GET /api/data-quality`: chất lượng dữ liệu và các ngày bị loại.

## Giao diện web

Giao diện web được chạy chung với API nên không cần mở thêm chương trình. Sau
khi nhấp đúp `run_api.bat`, mở `http://127.0.0.1:8000` để sử dụng. Web gồm:

- **Giới thiệu:** mục tiêu, phạm vi và quy trình xử lý.
- **Phân tích:** 24 ô theo giờ, kiểm tra đầu vào, dự đoán cụm, cảnh báo miền dữ
  liệu và biểu đồ so sánh với tâm cụm.
- **Lịch cụm:** lọc theo mùa/tháng, tô màu cụm và xem hồ sơ từng ngày.
- **Đánh giá:** silhouette, độ ổn định, baseline, kích thước cụm, tâm cụm và
  model card.
- **Dữ liệu:** chất lượng dữ liệu, quy tắc làm sạch, data dictionary và danh sách
  ngày bị loại.
