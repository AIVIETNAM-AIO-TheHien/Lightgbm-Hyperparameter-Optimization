# LightGBM Hyperparameter Optimization - Dataset Builder

Repo này chuẩn bị đúng hai bảng dữ liệu dùng cho notebook so sánh LightGBM:

- `data/processed/dev.csv`: 9 năm, từ `2013-01-01` đến `2021-12-31`.
- `data/processed/test.csv`: 1 năm khóa để đánh giá, từ `2022-01-01` đến `2022-12-31`.

Cả hai file dùng cùng một schema và chứa:

- `Date`;
- các feature calendar/Fourier/holiday/Tết/promotion từ pipeline Datathon;
- hai target gốc `Revenue` và `COGS`.

Pipeline không tạo lag từ target và không dùng dữ liệu web traffic/inventory cùng ngày, vì các biến này có thể làm rò rỉ thông tin khi đánh giá dự báo cho toàn bộ năm 2022.

## Chạy pipeline

Đặt `sales.csv` tại:

```text
data/raw/sales.csv
```

Cài dependency và sinh hai file:

```powershell
python -m pip install -r requirements.txt
python -m src.build_dev_test
```

Có thể chỉ định đường dẫn khác:

```powershell
python -m src.build_dev_test --input path\to\sales.csv --output-dir path\to\output
```

Chạy kiểm tra:

```powershell
python -m unittest discover -s tests -v
```

## Đọc trên Google Colab

```python
import pandas as pd

dev = pd.read_csv(DEV_URL, parse_dates=["Date"])
test = pd.read_csv(TEST_URL, parse_dates=["Date"])

target = "Revenue"
feature_cols = [
    column
    for column in dev.columns
    if column not in {"Date", "Revenue", "COGS"}
]

X_dev = dev[feature_cols]
y_dev = dev[target]
X_test = test[feature_cols]
y_test = test[target]
```

Optuna và Random Search chỉ được tune bằng `dev`. File `test` chỉ dùng để đánh giá kết quả cuối cùng.

