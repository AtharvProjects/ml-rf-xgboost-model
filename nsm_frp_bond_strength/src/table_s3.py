"""
Table S3 Source Metadata, Bounds Verification, and Specimen Envelope Matching
for NSM FRP Rod-to-Concrete Bond Strength (Kumar, Arora & Nehdi, 2024).

Grounded in data/table_s3_sources.json and data/table_s3_sources.csv parsed from
Supplementary Table S3 (Refs [21]-[40]).
"""

import json
import os
from typing import Any, Dict, Iterable, List, Optional
import pandas as pd


ALL_PARAM_COLUMNS = [
    "fc_MPa",
    "Lb_mm",
    "Df_mm",
    "Ef_GPa",
    "ff_MPa",
    "Wg_mm",
    "Dg_mm",
    "FRP_BT",
    "FRP_ST",
    "Pu_kN",
    "FN",
]

INPUT_FEATURE_COLUMNS = ALL_PARAM_COLUMNS[:9]


def _get_data_dir() -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "data")


def load_table_s3_json(json_path: Optional[str] = None) -> Dict[str, Any]:
    """Load structured Table S3 metadata, global bounds, and 20 study entries."""
    if json_path is None:
        json_path = os.path.join(_get_data_dir(), "table_s3_sources.json")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_table_s3_csv(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Load tabular Table S3 per-study bounds and citations."""
    if csv_path is None:
        csv_path = os.path.join(_get_data_dir(), "table_s3_sources.csv")
    return pd.read_csv(csv_path)


def get_global_bounds(json_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Return global [min, max] bounds and boundary provenance derived from Table S3."""
    return load_table_s3_json(json_path)["global_bounds"]


def verify_dataset_global_bounds(
    df: pd.DataFrame,
    json_path: Optional[str] = None,
    atol: float = 1e-6,
) -> Dict[str, Any]:
    """
    Verify that all dataset columns stay within the global [min, max] bounds of Table S3
    and check whether observed min/max align with the Table S3 boundaries.
    """
    global_bounds = get_global_bounds(json_path)
    report: Dict[str, Any] = {}
    all_within_bounds = True
    exact_extremes_match = True

    for col, spec in global_bounds.items():
        if col not in df.columns:
            continue
        vals = df[col].astype(float)
        obs_min = float(vals.min())
        obs_max = float(vals.max())
        s3_min = float(spec["min"])
        s3_max = float(spec["max"])

        within = (obs_min >= s3_min - atol) and (obs_max <= s3_max + atol)
        min_match = abs(obs_min - s3_min) <= atol
        max_match = abs(obs_max - s3_max) <= atol

        if not within:
            all_within_bounds = False
        if not (min_match and max_match):
            exact_extremes_match = False

        report[col] = {
            "table_s3_min": s3_min,
            "table_s3_max": s3_max,
            "observed_min": round(obs_min, 4),
            "observed_max": round(obs_max, 4),
            "within_table_s3_bounds": within,
            "exact_min_match": min_match,
            "exact_max_match": max_match,
            "min_refs": spec["min_refs"],
            "max_refs": spec["max_refs"],
        }

    return {
        "all_within_bounds": all_within_bounds,
        "exact_extremes_match": exact_extremes_match,
        "parameters": report,
    }


def _param_distance(
    val: float,
    bounds: Dict[str, Any],
    span: float,
    strict_allowed: bool = False,
) -> float:
    """Normalized deviation outside a study's parameter interval (0.0 if inside)."""
    lo = float(bounds["min"])
    hi = float(bounds["max"])
    if strict_allowed and "allowed_values" in bounds:
        allowed = [float(v) for v in bounds["allowed_values"]]
        if any(abs(val - a) <= 1e-6 for a in allowed):
            return 0.0
        return min(abs(val - a) for a in allowed) / max(span, 1e-9)

    if val < lo:
        return (lo - val) / max(span, 1e-9)
    if val > hi:
        return (val - hi) / max(span, 1e-9)
    return 0.0


def evaluate_record_against_study(
    record: Dict[str, float],
    study: Dict[str, Any],
    global_bounds: Dict[str, Dict[str, Any]],
    columns: Optional[Iterable[str]] = None,
    strict_allowed: bool = False,
) -> Dict[str, Any]:
    """Check whether a specimen record falls inside a single Table S3 study envelope."""
    cols = [c for c in (columns or ALL_PARAM_COLUMNS) if c in record]
    out_of_bounds: List[str] = []
    sq_dist = 0.0

    for col in cols:
        val = float(record[col])
        p_bounds = study["parameters"][col]
        span = float(global_bounds[col]["max"]) - float(global_bounds[col]["min"])
        d = _param_distance(val, p_bounds, span, strict_allowed=strict_allowed)
        if d > 1e-9:
            out_of_bounds.append(col)
            sq_dist += d * d

    return {
        "ref_id": study["ref_id"],
        "year": study["year"],
        "short_citation": study["short_citation"],
        "in_bounds": len(out_of_bounds) == 0,
        "out_of_bounds_params": out_of_bounds,
        "normalized_distance": round(float(sq_dist ** 0.5), 4),
    }


def match_record_to_studies(
    record: Dict[str, float],
    columns: Optional[Iterable[str]] = None,
    top_k: int = 3,
    json_path: Optional[str] = None,
    strict_allowed: bool = False,
) -> Dict[str, Any]:
    """Find matching Table S3 studies (and closest matches) for a specimen record."""
    payload = load_table_s3_json(json_path)
    global_bounds = payload["global_bounds"]
    studies = payload["studies"]

    evaluations = [
        evaluate_record_against_study(
            record, st, global_bounds, columns=columns, strict_allowed=strict_allowed
        )
        for st in studies
    ]
    exact_matches = [ev for ev in evaluations if ev["in_bounds"]]
    ranked = sorted(evaluations, key=lambda x: (x["normalized_distance"], len(x["out_of_bounds_params"])))

    return {
        "exact_match_count": len(exact_matches),
        "exact_matches": exact_matches,
        "closest_studies": ranked[:top_k],
    }


def audit_dataset_against_table_s3(
    df: pd.DataFrame,
    json_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Audit a specimen dataframe against Table S3 global bounds and per-study [21]-[40]
    envelopes across both 9-input feature space and full 11-parameter (input + output) space.
    """
    payload = load_table_s3_json(json_path)
    global_bounds = payload["global_bounds"]
    studies = payload["studies"]

    global_check = verify_dataset_global_bounds(df, json_path=json_path)

    input_exact_hits = 0
    full_exact_hits = 0
    assigned_ref_hits = 0
    has_ref_col = "Ref_ID" in df.columns
    study_by_ref = {st["ref_id"]: st for st in studies}
    nearest_study_counts = {st["ref_id"]: 0 for st in studies}
    records = df.to_dict(orient="records")

    for rec in records:
        evals_in = [
            evaluate_record_against_study(rec, st, global_bounds, columns=INPUT_FEATURE_COLUMNS, strict_allowed=True)
            for st in studies
        ]
        if any(ev["in_bounds"] for ev in evals_in):
            input_exact_hits += 1

        evals_full = [
            evaluate_record_against_study(rec, st, global_bounds, columns=ALL_PARAM_COLUMNS, strict_allowed=True)
            for st in studies
        ]
        if any(ev["in_bounds"] for ev in evals_full):
            full_exact_hits += 1

        if has_ref_col and rec["Ref_ID"] in study_by_ref:
            ev_assigned = evaluate_record_against_study(
                rec, study_by_ref[rec["Ref_ID"]], global_bounds, columns=ALL_PARAM_COLUMNS, strict_allowed=True
            )
            if ev_assigned["in_bounds"]:
                assigned_ref_hits += 1

        closest = min(evals_full, key=lambda x: (x["normalized_distance"], len(x["out_of_bounds_params"])))
        nearest_study_counts[closest["ref_id"]] += 1

    return {
        "total_records": len(df),
        "global_bounds_check": global_check,
        "has_ref_provenance": has_ref_col,
        "assigned_ref_matches": assigned_ref_hits if has_ref_col else None,
        "assigned_ref_match_pct": round(assigned_ref_hits / max(len(df), 1) * 100.0, 2) if has_ref_col else None,
        "input_space_exact_study_matches": input_exact_hits,
        "input_space_match_pct": round(input_exact_hits / max(len(df), 1) * 100.0, 2),
        "full_space_exact_study_matches": full_exact_hits,
        "full_space_match_pct": round(full_exact_hits / max(len(df), 1) * 100.0, 2),
        "nearest_study_distribution": nearest_study_counts,
    }


def main() -> None:
    data_dir = _get_data_dir()
    dataset_path = os.path.join(data_dir, "nsm_frp_bond_dataset.csv")
    custom_path = os.path.join(data_dir, "sample_custom_frp_entries.csv")

    print("=" * 78)
    print("TABLE S3 GROUNDING & CONSISTENCY AUDIT (Refs [21]-[40])")
    print("=" * 78)

    df = pd.read_csv(dataset_path)
    audit = audit_dataset_against_table_s3(df)
    gcheck = audit["global_bounds_check"]
    print(f"Dataset: {os.path.basename(dataset_path)} ({audit['total_records']} records)")
    print(f"  All records within Table S3 global bounds : {gcheck['all_within_bounds']}")
    print(f"  Exact global min/max parity with Table S3 : {gcheck['exact_extremes_match']}")
    if audit["has_ref_provenance"]:
        print(
            f"  Assigned Ref_ID study compliance          : "
            f"{audit['assigned_ref_matches']}/{audit['total_records']} "
            f"({audit['assigned_ref_match_pct']:.2f}%)"
        )
    print(
        f"  9-input envelope matches (>=1 study)      : "
        f"{audit['input_space_exact_study_matches']}/{audit['total_records']} "
        f"({audit['input_space_match_pct']:.2f}%)"
    )
    print(
        f"  11-param envelope matches (>=1 study)     : "
        f"{audit['full_space_exact_study_matches']}/{audit['total_records']} "
        f"({audit['full_space_match_pct']:.2f}%)"
    )

    if os.path.exists(custom_path):
        print("\nCustom Sample Entries vs Table S3 Studies (9 Input Features):")
        custom_df = pd.read_csv(custom_path)
        for idx, rec in enumerate(custom_df.to_dict(orient="records"), start=1):
            res = match_record_to_studies(rec, columns=INPUT_FEATURE_COLUMNS, top_k=2)
            top = res["closest_studies"][0]
            status = (
                f"MATCHES {','.join(m['ref_id'] for m in res['exact_matches'])}"
                if res["exact_match_count"] > 0
                else f"Closest {top['ref_id']} {top['short_citation']} (dist={top['normalized_distance']}, outside={top['out_of_bounds_params']})"
            )
            print(f"  Sample #{idx}: {status}")
    print("=" * 78)


if __name__ == "__main__":
    main()
