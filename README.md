# Phân nhóm ngày theo hồ sơ tiêu thụ điện

Dự án xử lý dữ liệu Individual Household Electric Power Consumption, biến mỗi ngày thành vector 24 giá trị công suất trung bình theo giờ, rồi phân cụm bằng K-means.

## Cấu trúc

```text
project_24_hoc_may/
├── data/
│   ├── raw/                 Dữ liệu tải từ UCI, không đưa lên Git
│   ├── processed/           daily_profiles.csv
│   └── README.md
├── models/                  Pipeline K-means đã lưu
├── notebooks/               Nơi đặt notebook EDA nếu cần
├── reports/
│   ├── figures/             Biểu đồ đánh giá
│   ├── model_card.md
│   └── summary.json
├── src/
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
├── tests/
├── requirements.txt
├── run_demo.bat
└── run_real.bat
```

## Chạy thử ngay bằng dữ liệu mô phỏng

Mở PowerShell tại thư mục dự án và chạy:

```powershell
py -m pip install -r requirements.txt
py -m src.data --demo
py -m src.train
py -m src.evaluate
```

Hoặc nhấp đúp `run_demo.bat`.

Kết quả chính:

- `data/processed/daily_profiles.csv`: mỗi ngày và 24 cột giờ.
- `models/kmeans_pipeline.joblib`: pipeline tiền xử lý và K-means.
- `reports/experiments.csv`: kết quả K=2 đến K=8.
- `reports/summary.json`: cấu hình và metric cuối.
- `reports/figures/`: các biểu đồ.

Kết quả tạo bởi lệnh `--demo` chỉ dùng để kiểm tra chương trình. Không đưa metric hoặc biểu đồ demo vào báo cáo cuối.

## Chạy với dữ liệu UCI thật

```powershell
py -m src.data --download --prepare
py -m src.train
py -m src.evaluate
```

Hoặc nhấp đúp `run_real.bat`. Quá trình tải và xử lý dữ liệu thật có thể mất vài phút.

Nếu máy không tải tự động được, mở trang UCI trong trình duyệt, tải tệp ZIP và đổi tên thành:

```text
data/raw/individual_household_electric_power_consumption.zip
```

Sau đó chạy lại:

```powershell
py -m src.data --download --prepare
py -m src.train
py -m src.evaluate
```

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
