# Structural Break Detection

This repository contains code for detecting structural breaks in time series data.

## Directory Structure

- `data/`: Contains training and testing datasets
- `notebooks/`: Jupyter notebooks for exploration and analysis
- `scripts/`: Python scripts for the modeling pipeline
- `outputs/`: Model predictions and submission files

## Getting Started

1. Set up the virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the baseline model:
   ```
   python scripts/baseline.py
   ```

## Baseline Model

The baseline model implements a simple approach to detecting structural breaks in time series data. See `notebooks/baseline.ipynb` for details. 