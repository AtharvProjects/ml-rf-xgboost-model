#!/usr/bin/env bash
set -e

# Determine script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Activate repository venv
if [ -f "$REPO_ROOT/venv/bin/activate" ]; then
    echo "Activating virtual environment at $REPO_ROOT/venv"
    source "$REPO_ROOT/venv/bin/activate"
else
    echo "Virtual environment not found at $REPO_ROOT/venv, using current python"
fi

echo "=========================================================="
echo "🎯 RUNNING END-TO-END REGRESSION PIPELINE (RF, ANN, LR)"
echo "=========================================================="

echo -e "\n[Step 1/4] Generating 100+ dataset entries..."
python3 "$SCRIPT_DIR/data/generate_data.py"

echo -e "\n[Step 2/4] Training Linear Regression, Random Forest, and ANN..."
PYTHONPATH="$SCRIPT_DIR" python3 "$SCRIPT_DIR/src/train.py"

echo -e "\n[Step 3/4] Evaluating models and generating analytical plots..."
PYTHONPATH="$SCRIPT_DIR" python3 "$SCRIPT_DIR/src/evaluate.py"

echo -e "\n[Step 4/4] Testing sample property valuation inference..."
PYTHONPATH="$SCRIPT_DIR" python3 "$SCRIPT_DIR/predict.py" \
  --square_footage 2600 \
  --bedrooms 4 \
  --bathrooms 3.5 \
  --lot_size_sqft 7200 \
  --property_age_years 8 \
  --distance_to_center_km 6.5 \
  --neighborhood_quality_score 8.8 \
  --has_garage 1 \
  --has_pool 1 \
  --annual_tax 8900.0

echo -e "\n=========================================================="
echo "🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!"
echo "=========================================================="
