#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "=== [1/4] Generating Dataset (100+ entries) ==="
./venv/bin/python data/generate_data.py

echo -e "\n=== [2/4] Training Random Forest and XGBoost Models ==="
export PYTHONPATH=.
./venv/bin/python src/train.py

echo -e "\n=== [3/4] Evaluating Models and Generating Plots ==="
./venv/bin/python src/evaluate.py

echo -e "\n=== [4/4] Running Sample Prediction Inference ==="
./venv/bin/python predict.py
