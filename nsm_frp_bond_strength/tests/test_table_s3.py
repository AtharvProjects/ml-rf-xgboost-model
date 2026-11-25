"""Tests for Table S3 parsing, structured CSV/JSON outputs, and bounds verification."""

import os
import unittest
import pandas as pd

from data.parse_table_s3 import PARAM_SCHEMA, parse_raw_markdown
from src.table_s3 import (
    INPUT_FEATURE_COLUMNS,
    audit_dataset_against_table_s3,
    get_global_bounds,
    load_table_s3_csv,
    load_table_s3_json,
    match_record_to_studies,
    verify_dataset_global_bounds,
)


MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(MODULE_DIR, "data")


class TestTableS3(unittest.TestCase):
    def test_raw_parse_20_studies(self):
        raw_md = os.path.join(DATA_DIR, "table_s3_raw.md")
        studies, payload = parse_raw_markdown(raw_md)
        self.assertEqual(len(studies), 20)
        self.assertEqual([s["ref_num"] for s in studies], list(range(21, 41)))
        self.assertEqual(payload["metadata"]["year_span"], [1999, 2023])

        for st in studies:
            self.assertTrue(st["short_citation"])
            self.assertTrue(st["full_citation"])
            for _, col_name, _, _, _, _ in PARAM_SCHEMA:
                p = st["parameters"][col_name]
                self.assertLessEqual(p["min"], p["max"])

    def test_ref37_frpst_quirk_preserved(self):
        payload = load_table_s3_json()
        ref37 = next(s for s in payload["studies"] if s["ref_id"] == "[37]")
        frpst = ref37["parameters"]["FRP_ST"]
        self.assertEqual(frpst["raw"], "1.5")
        self.assertEqual(frpst["min"], 1)
        self.assertEqual(frpst["max"], 5)
        self.assertEqual(frpst["allowed_values"], [1, 5])
        self.assertIn("1.5", ref37["notes"])

    def test_csv_and_json_parity(self):
        df_csv = load_table_s3_csv()
        payload = load_table_s3_json()
        self.assertEqual(len(df_csv), len(payload["studies"]))

        for row, st in zip(df_csv.to_dict(orient="records"), payload["studies"]):
            self.assertEqual(row["ref_id"], st["ref_id"])
            self.assertEqual(int(row["year"]), st["year"])
            self.assertEqual(row["short_citation"], st["short_citation"])
            for _, col_name, _, _, _, _ in PARAM_SCHEMA:
                self.assertEqual(str(row[f"{col_name}_raw"]), st["parameters"][col_name]["raw"])
                self.assertAlmostEqual(float(row[f"{col_name}_min"]), float(st["parameters"][col_name]["min"]))
                self.assertAlmostEqual(float(row[f"{col_name}_max"]), float(st["parameters"][col_name]["max"]))

    def test_global_bounds_match_published_ranges(self):
        gb = get_global_bounds()
        expected = {
            "fc_MPa": (23.0, 66.25),
            "Lb_mm": (14.4, 457.2),
            "Df_mm": (5.0, 24.0),
            "Ef_GPa": (37.17, 234.43),
            "ff_MPa": (749.0, 4826.5),
            "Wg_mm": (6.0, 30.0),
            "Dg_mm": (10.0, 40.0),
            "FRP_BT": (1, 3),
            "FRP_ST": (1, 7),
            "Pu_kN": (7.17, 127.52),
            "FN": (1, 16),
        }
        for col, (lo, hi) in expected.items():
            self.assertAlmostEqual(gb[col]["min"], lo)
            self.assertAlmostEqual(gb[col]["max"], hi)

    def test_dataset_within_table_s3_global_bounds(self):
        df = pd.read_csv(os.path.join(DATA_DIR, "nsm_frp_bond_dataset.csv"))
        for col in ("Ref_ID", "Year", "Study"):
            self.assertIn(col, df.columns)
        check = verify_dataset_global_bounds(df)
        self.assertTrue(check["all_within_bounds"])
        self.assertTrue(check["exact_extremes_match"])

        audit = audit_dataset_against_table_s3(df)
        self.assertEqual(audit["total_records"], 416)
        self.assertEqual(audit["assigned_ref_matches"], 416)
        self.assertEqual(audit["input_space_exact_study_matches"], 416)
        self.assertEqual(audit["full_space_exact_study_matches"], 416)

    def test_study_envelope_matching(self):
        # Specimen explicitly inside [34] Yan et al. (1999)
        rec = {
            "fc_MPa": 35.0,
            "Lb_mm": 100.0,
            "Df_mm": 9.5,
            "Ef_GPa": 230.0,
            "ff_MPa": 4000.0,
            "Wg_mm": 14.29,
            "Dg_mm": 19.05,
            "FRP_BT": 2,
            "FRP_ST": 5,
        }
        res = match_record_to_studies(rec, columns=INPUT_FEATURE_COLUMNS)
        matched_ids = [m["ref_id"] for m in res["exact_matches"]]
        self.assertIn("[34]", matched_ids)


if __name__ == "__main__":
    unittest.main()
