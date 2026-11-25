"""
Parser for Table S3 (Collected Experimental Dataset, Refs [21]-[40]) of
Kumar, Arora & Nehdi (2024), Journal of Building Engineering 90, 109418.

Reads data/table_s3_raw.md and generates:
  - data/table_s3_sources.csv   (flat per-study bounds + bibliographic metadata)
  - data/table_s3_sources.json  (structured JSON with per-study bounds & global provenance)
"""

import csv
import json
import os
import re
from typing import Any, Dict, List, Tuple

# Mapping from Table S3 column headers to canonical schema names in nsm_frp_bond_strength
PARAM_SCHEMA = [
    ("f’c (MPa)", "fc_MPa", "float", "MPa", "input", "Compressive strength of concrete"),
    ("Lb (mm)", "Lb_mm", "float", "mm", "input", "Bonded length of FRP bar"),
    ("Df (mm)", "Df_mm", "float", "mm", "input", "Diameter of FRP bar"),
    ("Ef (GPa)", "Ef_GPa", "float", "GPa", "input", "Elastic modulus of FRP rod"),
    ("ff (MPa)", "ff_MPa", "float", "MPa", "input", "Tensile strength of FRP rod"),
    ("Wg (mm)", "Wg_mm", "float", "mm", "input", "Width of the epoxy groove"),
    ("Dg (mm)", "Dg_mm", "float", "mm", "input", "Depth of the epoxy groove"),
    ("FRPBT", "FRP_BT", "int", "category (1-3)", "input", "FRP bar type (1: Basalt, 2: Carbon, 3: Glass)"),
    ("FRPST", "FRP_ST", "int", "category (1-7)", "input", "Surface treatment of FRP bar (1-7)"),
    ("Pu (kN)", "Pu_kN", "float", "kN", "output", "Ultimate pull-out bond strength"),
    ("FN", "FN", "int", "class (1-16)", "output", "Failure mode number (1-16)"),
]

SOURCE_QUIRKS = {
    ("[37]", "FRP_ST"): {
        "min": 1,
        "max": 5,
        "allowed_values": [1, 5],
        "note": 'Source Table S3 prints "1.5" for categorical FRPST in [37]; treated as discrete surface codes [1, 5] (min=1, max=5).',
    }
}


def _parse_bound_cell(raw_value: str, ref_id: str, col_name: str, dtype: str) -> Tuple[Dict[str, Any], str]:
    """Parse a single Table S3 numeric or range cell into {raw, min, max, is_constant}."""
    raw = raw_value.strip()
    if (ref_id, col_name) in SOURCE_QUIRKS:
        q = SOURCE_QUIRKS[(ref_id, col_name)]
        return {
            "raw": raw,
            "min": q["min"],
            "max": q["max"],
            "is_constant": False,
            "allowed_values": q["allowed_values"],
        }, q["note"]

    cast = int if dtype == "int" else float
    if "-" in raw:
        parts = raw.split("-")
        if len(parts) != 2:
            raise ValueError(f"Unexpected range format '{raw}' in {ref_id} {col_name}")
        lo, hi = cast(parts[0].strip()), cast(parts[1].strip())
    else:
        lo = hi = cast(raw)

    if lo > hi:
        raise ValueError(f"Invalid bounds {lo} > {hi} in {ref_id} {col_name}")

    info: Dict[str, Any] = {
        "raw": raw,
        "min": lo,
        "max": hi,
        "is_constant": lo == hi,
    }
    if dtype == "int":
        info["allowed_values"] = list(range(lo, hi + 1))
    return info, ""


def _extract_short_citation(full_citation: str, year: int) -> str:
    """Derive a concise 'FirstAuthor et al. (YYYY)' label from a bibliography entry."""
    marker = f". {year}."
    author_block = full_citation.split(marker, 1)[0]
    # Remove trailing ", and ..." vs first surname before first comma
    first_surname = author_block.split(",", 1)[0].strip()
    has_multiple = ", and " in author_block or " and " in author_block
    if not has_multiple:
        return f"{first_surname} ({year})"

    # Distinguish exactly 2 authors (only one initials/and break) vs 3+ authors
    if " and " in author_block:
        before_and, after_and = author_block.rsplit(" and ", 1)
        # In 2-author style: "Novidis, D. G., and S. J. Pantazopoulou" -> before_and has 1 surname
        # Last token of after_and is the 2nd author's surname (or multi-word surname after initials)
        # Check if before_and contains only one author (i.e., at most one comma or initials after first comma)
        parts_before = [p.strip() for p in before_and.split(",") if p.strip()]
        if len(parts_before) <= 2:
            # Extract second author's surname (strip leading initials like "L." or "S. J.")
            tokens = after_and.strip().split()
            surname_tokens = [t for t in tokens if not re.fullmatch(r"[A-Z]\.(?:-[A-Z]\.)?", t)]
            second_surname = " ".join(surname_tokens) if surname_tokens else tokens[-1]
            return f"{first_surname} & {second_surname} ({year})"
    return f"{first_surname} et al. ({year})"


def parse_raw_markdown(raw_md_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Parse Table S3 rows and References [21]-[40] from table_s3_raw.md."""
    with open(raw_md_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]

    # 1. Parse bibliography entries [21]..[40]
    refs: Dict[str, Dict[str, Any]] = {}
    ref_pattern = re.compile(r"^(\[\d+\])\s+(.*)$")
    doi_pattern = re.compile(r"(https?://doi\.org/\S+)")
    for line in lines:
        m = ref_pattern.match(line)
        if not m:
            continue
        ref_id, body = m.group(1), m.group(2).strip()
        # Normalize accidental double comma in source reference [32]
        body_clean = re.sub(r",\s*,", ",", body)
        doi_match = doi_pattern.search(body_clean)
        doi = doi_match.group(1).rstrip(".") if doi_match else ""
        refs[ref_id] = {"full_citation": body_clean, "doi": doi}

    # 2. Parse markdown table rows
    table_lines = [ln for ln in lines if ln.startswith("|") and ln.endswith("|")]
    if len(table_lines) < 3:
        raise ValueError(f"Could not find markdown table in {raw_md_path}")

    headers = [h.strip() for h in table_lines[0].strip("|").split("|")]
    expected_headers = ["References", "Year"] + [cfg[0] for cfg in PARAM_SCHEMA]
    if headers != expected_headers:
        raise ValueError(f"Header mismatch: got {headers}, expected {expected_headers}")

    studies: List[Dict[str, Any]] = []
    for row_line in table_lines[2:]:
        cells = [c.strip() for c in row_line.strip("|").split("|")]
        ref_id = cells[0]
        ref_num = int(ref_id.strip("[]"))
        year = int(cells[1])

        if ref_id not in refs:
            raise KeyError(f"Missing bibliography entry for {ref_id}")

        params: Dict[str, Any] = {}
        notes: List[str] = []
        for idx, (_, col_name, dtype, _, _, _) in enumerate(PARAM_SCHEMA, start=2):
            parsed, note = _parse_bound_cell(cells[idx], ref_id, col_name, dtype)
            params[col_name] = parsed
            if note:
                notes.append(note)

        full_citation = refs[ref_id]["full_citation"]
        studies.append(
            {
                "ref_id": ref_id,
                "ref_num": ref_num,
                "year": year,
                "short_citation": _extract_short_citation(full_citation, year),
                "full_citation": full_citation,
                "doi": refs[ref_id]["doi"],
                "notes": " ".join(notes),
                "parameters": params,
            }
        )

    # 3. Derive global bounds and boundary provenance across all studies
    global_bounds: Dict[str, Any] = {}
    for _, col_name, dtype, unit, role, desc in PARAM_SCHEMA:
        mins = [s["parameters"][col_name]["min"] for s in studies]
        maxs = [s["parameters"][col_name]["max"] for s in studies]
        g_min, g_max = min(mins), max(maxs)
        min_refs = [s["ref_id"] for s in studies if s["parameters"][col_name]["min"] == g_min]
        max_refs = [s["ref_id"] for s in studies if s["parameters"][col_name]["max"] == g_max]
        global_bounds[col_name] = {
            "min": g_min,
            "max": g_max,
            "min_refs": min_refs,
            "max_refs": max_refs,
            "dtype": dtype,
            "unit": unit,
            "role": role,
            "description": desc,
        }

    metadata = {
        "source_paper": "Kumar, A., Arora, H. C., & Nehdi, M. L. (2024). XGBoost algorithm based estimation of near surface mounted FRP rod-to-concrete bond strength and failure mode. Journal of Building Engineering, 90, 109418.",
        "table_title": "Table S3. Collected experimental dataset",
        "study_count": len(studies),
        "ref_range": f"{studies[0]['ref_id']}–{studies[-1]['ref_id']}",
        "year_span": [min(s["year"] for s in studies), max(s["year"] for s in studies)],
        "total_specimens_in_paper": 416,
    }
    payload = {
        "metadata": metadata,
        "global_bounds": global_bounds,
        "studies": studies,
    }
    return studies, payload


def write_csv_and_json(studies: List[Dict[str, Any]], payload: Dict[str, Any], out_csv: str, out_json: str) -> None:
    """Write structured Table S3 CSV and JSON artifacts."""
    fieldnames = ["ref_id", "ref_num", "year", "short_citation"]
    for _, col_name, _, _, _, _ in PARAM_SCHEMA:
        fieldnames.extend([f"{col_name}_raw", f"{col_name}_min", f"{col_name}_max"])
    fieldnames.extend(["doi", "full_citation", "notes"])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for st in studies:
            row: Dict[str, Any] = {
                "ref_id": st["ref_id"],
                "ref_num": st["ref_num"],
                "year": st["year"],
                "short_citation": st["short_citation"],
                "doi": st["doi"],
                "full_citation": st["full_citation"],
                "notes": st["notes"],
            }
            for _, col_name, _, _, _, _ in PARAM_SCHEMA:
                p = st["parameters"][col_name]
                row[f"{col_name}_raw"] = p["raw"]
                row[f"{col_name}_min"] = p["min"]
                row[f"{col_name}_max"] = p["max"]
            writer.writerow(row)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    data_dir = os.path.dirname(os.path.abspath(__file__))
    raw_md = os.path.join(data_dir, "table_s3_raw.md")
    out_csv = os.path.join(data_dir, "table_s3_sources.csv")
    out_json = os.path.join(data_dir, "table_s3_sources.json")

    studies, payload = parse_raw_markdown(raw_md)
    write_csv_and_json(studies, payload, out_csv, out_json)

    print(f"Parsed {len(studies)} Table S3 studies ({payload['metadata']['ref_range']}, "
          f"{payload['metadata']['year_span'][0]}–{payload['metadata']['year_span'][1]})")
    print(f"  -> CSV : {out_csv}")
    print(f"  -> JSON: {out_json}")
    print("\nGlobal Parameter Bounds Derived from Table S3:")
    for col_name, info in payload["global_bounds"].items():
        print(f"  {col_name:<8}: [{info['min']}, {info['max']}] ({info['unit']}) "
              f"| min={','.join(info['min_refs'])} | max={','.join(info['max_refs'])}")


if __name__ == "__main__":
    main()
