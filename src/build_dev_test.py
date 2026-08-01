"""Build leakage-safe development and test datasets for the benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_prep import build_features


MODEL_START = pd.Timestamp("2013-01-01")
DEV_END = pd.Timestamp("2021-12-31")
TEST_START = pd.Timestamp("2022-01-01")
TEST_END = pd.Timestamp("2022-12-31")

REQUIRED_COLUMNS = ["Date", "Revenue", "COGS"]
TARGET_COLUMNS = ["Revenue", "COGS"]
EXPECTED_DEV_ROWS = 3_287
EXPECTED_TEST_ROWS = 365


def _validate_raw_sales(sales: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in sales.columns
    ]
    if missing_columns:
        raise ValueError(
            "sales.csv is missing required columns: "
            + ", ".join(missing_columns)
        )

    clean = sales[REQUIRED_COLUMNS].copy()
    clean["Date"] = pd.to_datetime(clean["Date"], errors="coerce")

    invalid_dates = int(clean["Date"].isna().sum())
    if invalid_dates:
        raise ValueError(f"sales.csv contains {invalid_dates} invalid Date values")

    for target in TARGET_COLUMNS:
        clean[target] = pd.to_numeric(clean[target], errors="coerce")

    invalid_targets = clean[TARGET_COLUMNS].isna().sum()
    invalid_targets = invalid_targets[invalid_targets > 0]
    if not invalid_targets.empty:
        details = ", ".join(
            f"{column}={int(count)}"
            for column, count in invalid_targets.items()
        )
        raise ValueError(f"sales.csv contains missing/non-numeric targets: {details}")

    target_values = clean[TARGET_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(target_values).all():
        raise ValueError("sales.csv contains infinite target values")
    if (target_values < 0).any():
        raise ValueError("Revenue and COGS must be non-negative")

    duplicate_dates = clean["Date"].duplicated(keep=False)
    if duplicate_dates.any():
        examples = (
            clean.loc[duplicate_dates, "Date"]
            .dt.strftime("%Y-%m-%d")
            .drop_duplicates()
            .head(5)
            .tolist()
        )
        raise ValueError(
            "sales.csv contains duplicate dates, for example: "
            + ", ".join(examples)
        )

    return clean.sort_values("Date").reset_index(drop=True)


def _validate_complete_daily_range(model_data: pd.DataFrame) -> None:
    expected_dates = pd.date_range(MODEL_START, TEST_END, freq="D")
    actual_dates = pd.DatetimeIndex(model_data["Date"])

    missing_dates = expected_dates.difference(actual_dates)
    unexpected_dates = actual_dates.difference(expected_dates)
    if len(missing_dates) or len(unexpected_dates):
        details = []
        if len(missing_dates):
            details.append(
                "missing dates: "
                + ", ".join(date.strftime("%Y-%m-%d") for date in missing_dates[:5])
            )
        if len(unexpected_dates):
            details.append(
                "unexpected dates: "
                + ", ".join(
                    date.strftime("%Y-%m-%d") for date in unexpected_dates[:5]
                )
            )
        raise ValueError("Incomplete modeling period; " + "; ".join(details))


def _validate_model_table(model_table: pd.DataFrame) -> None:
    if model_table.columns[0] != "Date":
        raise ValueError("Date must be the first output column")
    if model_table.columns[-2:].tolist() != TARGET_COLUMNS:
        raise ValueError("Revenue and COGS must be the final output columns")
    if model_table.isna().any().any():
        null_columns = model_table.columns[model_table.isna().any()].tolist()
        raise ValueError(
            "Model table contains missing values in: " + ", ".join(null_columns)
        )

    feature_columns = [
        column
        for column in model_table.columns
        if column not in {"Date", *TARGET_COLUMNS}
    ]
    feature_values = model_table[feature_columns].to_numpy(dtype=float)
    if not np.isfinite(feature_values).all():
        raise ValueError("Model table contains infinite feature values")

    forbidden_features = {
        "Revenue_lag_7",
        "Revenue_lag_14",
        "Revenue_lag_30",
        "sessions",
        "unique_visitors",
        "page_views",
        "start_stock_on_hand",
    }
    leaked_columns = sorted(forbidden_features.intersection(feature_columns))
    if leaked_columns:
        raise ValueError(
            "Model table contains disallowed leakage-prone features: "
            + ", ".join(leaked_columns)
        )


def prepare_datasets(input_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read raw sales and return validated 9-year dev and 1-year test tables."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Raw sales file not found: {input_path}")

    sales = pd.read_csv(input_path)
    sales = _validate_raw_sales(sales)

    model_sales = sales.loc[
        sales["Date"].between(MODEL_START, TEST_END, inclusive="both")
    ].reset_index(drop=True)
    _validate_complete_daily_range(model_sales)

    model_table = build_features(model_sales["Date"].copy())
    model_table["Revenue"] = model_sales["Revenue"].to_numpy()
    model_table["COGS"] = model_sales["COGS"].to_numpy()
    _validate_model_table(model_table)

    dev = model_table.loc[
        model_table["Date"].between(MODEL_START, DEV_END, inclusive="both")
    ].reset_index(drop=True)
    test = model_table.loc[
        model_table["Date"].between(TEST_START, TEST_END, inclusive="both")
    ].reset_index(drop=True)

    if len(dev) != EXPECTED_DEV_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_DEV_ROWS} dev rows, found {len(dev)}"
        )
    if len(test) != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_ROWS} test rows, found {len(test)}"
        )
    if dev.columns.tolist() != test.columns.tolist():
        raise ValueError("Dev and test schemas do not match")
    if dev["Date"].max() >= test["Date"].min():
        raise ValueError("Dev and test date ranges overlap")

    return dev, test


def _write_csv_atomic(data: pd.DataFrame, output_path: Path) -> None:
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    data.to_csv(
        temporary_path,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.15g",
    )
    temporary_path.replace(output_path)


def write_datasets(
    dev: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write dev.csv and test.csv atomically."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dev_path = output_dir / "dev.csv"
    test_path = output_dir / "test.csv"
    _write_csv_atomic(dev, dev_path)
    _write_csv_atomic(test, test_path)
    return dev_path, test_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build 2013-2021 dev.csv and 2022 test.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/sales.csv"),
        help="Path to raw sales.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for dev.csv and test.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dev, test = prepare_datasets(args.input)
    dev_path, test_path = write_datasets(dev, test, args.output_dir)

    feature_count = len(dev.columns) - len(TARGET_COLUMNS) - 1
    print(
        f"[OK] {dev_path}: {len(dev):,} rows, "
        f"{dev['Date'].min():%Y-%m-%d} -> {dev['Date'].max():%Y-%m-%d}"
    )
    print(
        f"[OK] {test_path}: {len(test):,} rows, "
        f"{test['Date'].min():%Y-%m-%d} -> {test['Date'].max():%Y-%m-%d}"
    )
    print(
        f"[OK] Shared schema: {feature_count} features + Date + "
        f"{', '.join(TARGET_COLUMNS)}"
    )


if __name__ == "__main__":
    main()

