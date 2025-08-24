# Churn Prediction - Take Home Assignment

## Overview

This project contains the solution to a churn prediction task as part of the SafetyCulture Data Scientist interview process.

The goal is to build a robust, interpretable pipeline that predicts whether a customer is likely to **churn** based on historical and behavioral data. The solution includes traditional machine learning models and deep learning approaches to cover a wide range of patterns.

---

## Objective

Predict whether a customer will churn (`is_churned = 1`) or stay (`is_churned = 0`)** based on provided features.


## Project Structure
```bash
CHURN-PREDICTION/
│
├── notebooks/                            # Jupyter notebooks for analysis and reporting
│   ├── report_notebook.ipynb
│   └── output/                           # Model and evaluation output
│
├── outputs/                              # Processed data and PCA results
│
├── resources/                                 # Raw and reference data
│   ├── churn_orgs.csv
│   ├── data_commercial.csv
│   ├── data_product.csv
│   └── Task_README.md
│
├── src/                                  # Source code for the pipeline
│   ├── __init__.py
│   ├── data_loader.py                    # Data loading
│   ├── data_analysis.py                  # EDA and visualizations
│   ├── data_preprocessor.py              # Preprocessing pipeline
│   ├── model_prediction.py               # Model training and evaluation
│   ├── model.py                          # Model classes 
│   └── result_evaluation.py              # Evaluation logic and reporting
│
├── README.md                             # Project overview and instructions
└── requirements.txt                      # Required Python packages
```
## Key Components

### 1. Data Preprocessing
- Feature selection using prior importance analysis
- Handling missing values, outliers, and duplicates
- Standard scaling of numeric features
- Optional PCA for dimensionality reduction
- Time-based train/test splitting to simulate production scenario

### 2. Modeling
- **Traditional Models**:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- **Deep Learning Models** (optional, PyTorch-based):
  - LSTM (captures sequence behavior)
  - 1D CNN (detects feature pattern signals)

### 3. Evaluation
- ROC AUC, accuracy, and F1 score
- Classification reports

---

## How to Run

1. Download the folder and install dependencies (will be uploaded to GitHub if approved): 
   ```bash
   python -m venv churn_env
   ```

2. Activate environment:  
   **Windows:**
   ```bash
   churn_env\Scripts\activate
   ```
   **macOS/Linux:**
   ```bash
   source churn_env/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run project:
   ```bash
   cd notebooks
   jupyter notebook
   ```