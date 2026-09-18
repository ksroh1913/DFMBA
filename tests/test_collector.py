import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "collect_seoul_monthly.py"
SPEC = importlib.util.spec_from_file_location("collector", SCRIPT)
collector = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(collector)


class CollectorTests(unittest.TestCase):
    def test_secret_redaction_plain_and_encoded(self):
        client = collector.ApiClient()
        client.register_secret("abc+/= secret")
        text = client.redact("abc+/= secret and abc%2B%2F%3D%20secret")
        self.assertNotIn("abc", text)
        self.assertEqual(text.count("[REDACTED]"), 2)

    def test_kosis_normalization_keeps_only_25_districts(self):
        rows = [
            {"C1_NM": "서울특별시 종로구", "PRD_DE": "2024.01", "DT": "150000"},
            {"C1_NM": "서울특별시", "PRD_DE": "202401", "DT": "9000000"},
            {"C1_NM": "부산광역시 중구", "PRD_DE": "202401", "DT": "40000"},
        ]
        result = collector.normalize_kosis(rows)
        self.assertEqual(result[0]["district"], "종로구")
        self.assertEqual(len(result), 1)

    def test_duplicate_is_rejected(self):
        row = {"month": "202401", "district": "종로구"}
        with self.assertRaises(collector.CollectionError):
            collector.deduplicate([row, row], "test")

    def test_incomplete_month_is_rejected(self):
        rows = [{"month": "202401", "district": name} for name in collector.DISTRICTS[:-1]]
        with self.assertRaises(collector.CollectionError):
            collector.validate_coverage(rows, "test")

    def test_xlsx_is_valid_zip_with_three_sheets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.xlsx"
            collector.write_xlsx(path, [("첫째", [["a"], [1]]), ("둘째", [["b"]]), ("셋째", [["c"]])])
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(archive.testzip(), None)
                self.assertIn("xl/worksheets/sheet3.xml", archive.namelist())


if __name__ == "__main__":
    unittest.main()
