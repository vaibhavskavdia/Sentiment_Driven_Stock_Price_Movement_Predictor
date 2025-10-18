# Sentiment Driven Stock Price Movement Predictor

A project that predicts short-term stock price movement using sentiment analysis from news and social media combined with historical market data. This repository contains data-preprocessing pipelines, sentiment extraction, feature engineering, modeling experiments, and evaluation tools to help explore how sentiment influences price movements.

## Table of Contents
- Project Overview
- Key Features
- Repository Structure
- Requirements
- Installation
- Data
- How to Run
  - Quick start (inference)
  - Training models
- Modeling & Evaluation
- Examples
- Notes & Tips
- Contributing
- License & Contact

## Project Overview
This project demonstrates a pipeline to predict stock price direction (up/down/flat) by combining sentiment signals (from news articles, tweets, etc.) with technical and fundamental features from historical price data. The goal is to provide reproducible experiments to evaluate the incremental value of sentiment features for short-term movement prediction.

## Key Features
- Data ingestion utilities for market data (e.g., Yahoo Finance, Alpha Vantage) and text sources (News API, Twitter)
- Text preprocessing and sentiment extraction (VADER, transformer-based classifiers)
- Feature engineering combining sentiment, technical indicators and lagged returns
- Modeling experiments: baseline (logistic regression), tree-based (RandomForest, XGBoost), and sequence models (LSTM/Transformer)
- Evaluation scripts with classification metrics and backtesting-ready outputs

## Repository Structure
- data/ (raw and processed dataset placeholders)
- notebooks/ (exploratory analysis and experiments)
- src/
  - data/ (download and preprocessing scripts)
  - features/ (feature engineering scripts)
  - models/ (training and inference code)
  - utils/ (helpers: logging, config)
- models/ (saved checkpoints and artifacts)
- tests/ (unit/integration tests)
- requirements.txt
- README.md

Adjust paths to the actual repository layout if files are organised differently.

## Requirements
- Python 3.8+
- pip or conda
- Recommended packages (see requirements.txt): pandas, numpy, scikit-learn, matplotlib, seaborn, nltk, vaderSentiment, transformers, torch or tensorflow, xgboost, yfinance, requests
- Optional: GPU with CUDA for transformer/LSTM training

## Installation
1. Clone the repository:
```bash
git clone https://github.com/vaibhavskavdia/Sentiment_Driven_Stock_Price_Movement_Predictor.git
cd Sentiment_Driven_Stock_Price_Movement_Predictor
```
2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```
3. (Optional) If using transformers/pytorch, install the appropriate torch build for your platform. See https://pytorch.org/

## Data
This repository does not include proprietary data. Example sources and how to fetch them:
- Historical prices: yfinance (Yahoo Finance) or Alpha Vantage (API key required)
- News: NewsAPI, GDELT, or RSS feeds (may require API keys)
- Social media: Twitter API (developer account required)

Place raw files under data/raw/ and processed datasets under data/processed/. Example exploratory notebooks expect these locations.

## How to Run
Quick start (inference with a saved model):
```bash
python src/models/infer.py --model models/baseline.pkl --input data/processed/sample_features.csv --output predictions.csv
```
Training a model (example for a scikit-learn classifier):
```bash
python src/models/train_sklearn.py --config config/train_sklearn.yaml
```
Run notebooks for exploratory analysis: open notebooks/*.ipynb with Jupyter or VS Code.

## Modeling & Evaluation
- Target: next-day/next-period price direction (binary or multi-class)
- Metrics: accuracy, precision, recall, F1, ROC-AUC (for binary), confusion matrix
- Use cross-validation and time-series-aware splits (e.g., expanding window CV) to avoid leakage.
- For backtesting, convert predictions into position sizing and simulate P&L over hold periods.

## Examples
See notebooks/ for worked examples: preprocessing_and_features.ipynb, modeling_experiments.ipynb, backtest_simulation.ipynb.

## Notes & Tips
- Align sentiment timestamps with market timestamps carefully (e.g., use only information available before market open for intraday predictions).
- Normalize or scale features appropriately and consider class imbalance techniques if the dataset is skewed.
- Start with simple baselines before using heavy models to measure incremental benefit of sentiment features.

## Contributing
Contributions are welcome. Suggested workflow:
1. Fork the repo
2. Create a branch for your feature/fix
3. Add tests and documentation where appropriate
4. Open a pull request describing your changes

Please follow the code style used in the repository and include reproducible steps for experiments.

## License & Contact
Suggested license: MIT — update this file or add a LICENSE file to specify the project license.

Maintainer: @vaibhavskavdia

