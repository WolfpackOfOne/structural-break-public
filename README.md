# Structural Break Detection - ADIA Lab Challenge

## 🔍 Project Overview

This repository contains our solution for the [ADIA Lab Structural Break Challenge](https://www.crunchdao.com/competitions/structural-break) hosted by CrunchDAO. The goal is to identify structural breaks in time series data using a hybrid approach that combines statistical methods, machine learning, and deep learning techniques.

Structural breaks are sudden changes in time series data that represent significant shifts in underlying patterns or relationships. Detecting these breaks is crucial in financial markets, economic forecasting, and risk management. Our approach aims to provide robust and accurate detection methods that can be applied across various domains.

## 📁 Directory Structure

```
structural-break/
│
├── data/                         # Data files
│   ├── train.csv                 # Training dataset
│   ├── test.csv                  # Test dataset
│   └── metadata.csv              # Additional information about the data
│
├── notebooks/                    # Jupyter notebooks
│   ├── exploratory_analysis.ipynb    # Initial data exploration
│   ├── feature_engineering.ipynb     # Feature creation and processing
│   ├── statistical_methods.ipynb     # Implementation of statistical approaches
│   ├── ml_models.ipynb               # Machine learning models
│   ├── deep_learning.ipynb           # Deep learning implementations
│   └── baseline.ipynb                # Simple baseline approach
│
├── src/                          # Source code
│   ├── data_processing/          # Data processing modules
│   │   ├── __init__.py
│   │   ├── preprocessing.py      # Data preprocessing functions
│   │   └── feature_engineering.py # Feature creation
│   │
│   ├── models/                   # Model implementations
│   │   ├── __init__.py
│   │   ├── statistical.py        # Statistical methods (CUSUM, PELT, etc.)
│   │   ├── ml_models.py          # Machine learning models
│   │   └── deep_learning.py      # Deep learning implementations
│   │
│   ├── evaluation/               # Evaluation code
│   │   ├── __init__.py
│   │   ├── metrics.py            # Performance metrics calculations
│   │   └── visualization.py      # Result visualization
│   │
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       └── helpers.py            # Helper functions
│
├── scripts/                      # Execution scripts
│   ├── train.py                  # Model training script
│   ├── predict.py                # Prediction script for new data
│   ├── evaluate.py               # Evaluation script
│   └── baseline.py               # Baseline model implementation
│
├── outputs/                      # Model outputs
│   ├── models/                   # Saved model files
│   ├── predictions/              # Model predictions
│   ├── visualizations/           # Generated plots and visualizations
│   └── submission.csv            # Competition submission file
│
├── tests/                        # Unit tests
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_evaluation.py
│
├── configs/                      # Configuration files
│   ├── model_config.yaml         # Model hyperparameters
│   └── training_config.yaml      # Training configuration
│
├── README.md                     # Project documentation
├── requirements.txt              # Project dependencies
└── setup.py                      # Package setup script
```

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.9+
- pip package manager
- Virtual environment tool (optional but recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/structural-break-detection.git
   cd structural-break-detection
   ```

2. Create and activate a virtual environment (optional):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the competition data and place it in the `data/` directory.

## 🚀 How to Train and Evaluate Models

### Data Preprocessing

```bash
python scripts/preprocessing.py --input data/train.csv --output data/processed/train_processed.csv
```

### Training Models

To train the model with default parameters:

```bash
python scripts/train.py --data data/processed/train_processed.csv --model-type ensemble
```

For specific models:

```bash
# Statistical methods
python scripts/train.py --data data/processed/train_processed.csv --model-type statistical --method cusum

# Machine learning models
python scripts/train.py --data data/processed/train_processed.csv --model-type ml --method lightgbm

# Deep learning models
python scripts/train.py --data data/processed/train_processed.csv --model-type dl --method lstm
```

### Making Predictions

```bash
python scripts/predict.py --model outputs/models/best_model.pkl --data data/test.csv --output outputs/submission.csv
```

### Evaluation

```bash
python scripts/evaluate.py --predictions outputs/predictions/prediction.csv --ground-truth data/validation.csv
```

## 🧠 Methodology and Techniques

Our approach integrates three categories of methods:

### Statistical Methods
- **CUSUM (Cumulative Sum)**: Detects changes in the mean of a process
- **Bayesian Change Point Detection**: Probabilistic approach to identifying changes in data distributions
- **PELT (Pruned Exact Linear Time)**: Efficient algorithm for multiple change point detection

### Machine Learning Models
- **LightGBM**: Gradient boosting framework using tree-based algorithms
- **XGBoost**: Scalable gradient boosting implementation
- **Feature Engineering**: Extensive time series features including lags, rolling statistics, and spectral features

### Deep Learning Models
- **LSTM (Long Short-Term Memory)**: Recurrent neural networks for capturing temporal dependencies
- **CNN (Convolutional Neural Networks)**: For detecting localized patterns in time series
- **Autoencoders**: For anomaly detection and representation learning

### Explainability
- **SHAP (SHapley Additive exPlanations)**: For interpreting model predictions and understanding feature importance

## 📊 Evaluation Metrics

We evaluate our models using:

- **ROC AUC**: Area under the Receiver Operating Characteristic curve
- **Precision**: Ratio of correctly predicted positive observations to total predicted positives
- **Recall**: Ratio of correctly predicted positive observations to all actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **Time-to-Detection**: How quickly breaks are detected after they occur

## 🤝 Contribution Guidelines

We welcome contributions to improve our structural break detection methods:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Submit a pull request

Please ensure your code follows our coding standards:
- PEP 8 compliant
- Properly documented with docstrings
- Includes appropriate unit tests

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- ADIA Lab and CrunchDAO for hosting the competition
- All contributors who have helped improve this codebase
- Open-source libraries used in this project