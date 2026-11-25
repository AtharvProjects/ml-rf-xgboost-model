#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

if [ -f "$REPO_ROOT/venv/bin/activate" ]; then
    source "$REPO_ROOT/venv/bin/activate"
fi

echo "========================================================================="
echo "🏗️  NSM FRP BOND STRENGTH & FAILURE MODE PIPELINE (Kumar et al., 2024)"
echo "========================================================================="

echo -e "\n[Step 1/5] Parsing & verifying Table S3 literature sources ([21]-[40])..."
python3 "$SCRIPT_DIR/data/parse_table_s3.py"

echo -e "\n[Step 2/5] Generating 416-record NSM FRP datasheet..."
python3 "$SCRIPT_DIR/data/generate_data.py"
PYTHONPATH="$SCRIPT_DIR" python3 -m src.table_s3

echo -e "\n[Step 3/5] Training 6 ML models (LR, DT, GBT, RF, XGB, ANN) & formulas..."
PYTHONPATH="$SCRIPT_DIR" python3 "$SCRIPT_DIR/src/train.py"

echo -e "\n[Step 4/5] Evaluating Table 3, 4, 5 & generating all paper diagrams..."
PYTHONPATH="$SCRIPT_DIR" python3 "$SCRIPT_DIR/src/evaluate.py"

echo -e "\n[Step 5/5] Verifying single & batch custom predictions..."
PYTHONPATH="$SCRIPT_DIR" python3 "$SCRIPT_DIR/predict_custom.py" --fc 48.5 --lb 320.0 --df 12.0 --ef 205.0 --ff 3100.0 --wg 18.0 --dg 20.0 --bt 2 --st 7
PYTHONPATH="$SCRIPT_DIR" python3 "$SCRIPT_DIR/predict_custom.py" --csv "$SCRIPT_DIR/data/sample_custom_frp_entries.csv" --output "$SCRIPT_DIR/data/sample_custom_predictions.csv"

echo -e "\n🎉 PIPELINE COMPLETED SUCCESSFULLY!"
