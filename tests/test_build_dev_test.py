from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.build_dev_test import prepare_datasets, write_datasets


RAW_SALES = Path("data/raw/sales.csv")
TARGETS = {"Revenue", "COGS"}


class BuildDevTestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dev, cls.test = prepare_datasets(RAW_SALES)

    def test_expected_time_split(self) -> None:
        self.assertEqual(len(self.dev), 3_287)
        self.assertEqual(len(self.test), 365)
        self.assertEqual(self.dev["Date"].min(), pd.Timestamp("2013-01-01"))
        self.assertEqual(self.dev["Date"].max(), pd.Timestamp("2021-12-31"))
        self.assertEqual(self.test["Date"].min(), pd.Timestamp("2022-01-01"))
        self.assertEqual(self.test["Date"].max(), pd.Timestamp("2022-12-31"))
        self.assertLess(self.dev["Date"].max(), self.test["Date"].min())

    def test_schema_is_shared_and_model_ready(self) -> None:
        self.assertListEqual(self.dev.columns.tolist(), self.test.columns.tolist())
        self.assertEqual(self.dev.columns[0], "Date")
        self.assertListEqual(
            self.dev.columns[-2:].tolist(),
            ["Revenue", "COGS"],
        )

        feature_columns = [
            column
            for column in self.dev.columns
            if column not in {"Date", *TARGETS}
        ]
        self.assertGreater(len(feature_columns), 0)
        self.assertTrue(
            np.isfinite(self.dev[feature_columns].to_numpy(dtype=float)).all()
        )
        self.assertTrue(
            np.isfinite(self.test[feature_columns].to_numpy(dtype=float)).all()
        )

    def test_daily_dates_are_complete_and_unique(self) -> None:
        expected_dev_dates = pd.date_range("2013-01-01", "2021-12-31", freq="D")
        expected_test_dates = pd.date_range("2022-01-01", "2022-12-31", freq="D")
        self.assertTrue(self.dev["Date"].is_unique)
        self.assertTrue(self.test["Date"].is_unique)
        self.assertTrue(self.dev["Date"].equals(pd.Series(expected_dev_dates)))
        self.assertTrue(self.test["Date"].equals(pd.Series(expected_test_dates)))

    def test_target_and_feature_values_are_complete(self) -> None:
        self.assertFalse(self.dev.isna().any().any())
        self.assertFalse(self.test.isna().any().any())
        self.assertTrue((self.dev[["Revenue", "COGS"]] >= 0).all().all())
        self.assertTrue((self.test[["Revenue", "COGS"]] >= 0).all().all())

    def test_targets_match_the_raw_source(self) -> None:
        raw = pd.read_csv(RAW_SALES, parse_dates=["Date"])
        expected = raw.loc[
            raw["Date"].between("2013-01-01", "2022-12-31", inclusive="both"),
            ["Date", "Revenue", "COGS"],
        ].sort_values("Date", ignore_index=True)
        actual = pd.concat(
            [
                self.dev[["Date", "Revenue", "COGS"]],
                self.test[["Date", "Revenue", "COGS"]],
            ],
            ignore_index=True,
        )
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)

    def test_leakage_prone_legacy_features_are_absent(self) -> None:
        forbidden = {
            "Revenue_lag_7",
            "Revenue_lag_14",
            "Revenue_lag_30",
            "sessions",
            "unique_visitors",
            "page_views",
            "start_stock_on_hand",
        }
        self.assertTrue(forbidden.isdisjoint(self.dev.columns))

    def test_csv_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            dev_path, test_path = write_datasets(
                self.dev,
                self.test,
                temporary_dir,
            )
            loaded_dev = pd.read_csv(dev_path, parse_dates=["Date"])
            loaded_test = pd.read_csv(test_path, parse_dates=["Date"])

        self.assertEqual(loaded_dev.shape, self.dev.shape)
        self.assertEqual(loaded_test.shape, self.test.shape)
        self.assertListEqual(loaded_dev.columns.tolist(), self.dev.columns.tolist())
        self.assertListEqual(
            loaded_test.columns.tolist(),
            self.test.columns.tolist(),
        )


if __name__ == "__main__":
    unittest.main()
