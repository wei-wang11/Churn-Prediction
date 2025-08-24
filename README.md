# Churn Prediction - Wei Wang

## Overview

This project contains the solution to a churn prediction task.

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
├── resources/                                 # Raw and Reference Data (removed due to data sensitivity)
│   ├── churn_orgs.csv
│   ├── data_commercial.csv
│   ├── data_product.csv
│
├── src/                                  # Source code for the pipeline
│   ├── __init__.py
│   ├── data_loader.py                    # Data loading
│   ├── data_analysis.py                  # EDA and visualisations
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
- Time-based train/test splitting to simulate a production scenario

### 2. Modeling
- **Traditional Models**:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- **Deep Learning Models** (optional, PyTorch-based):
  - LSTM (captures sequence behaviour)
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
5. Run Streamlit for better visualisation analysis
   ```bash
   streamlit run streamlit_presentation.py
   ```
   <img width="1912" height="954" alt="image" src="https://github.com/user-attachments/assets/c36e016c-14aa-4b8f-bcf7-2d7ea1d3d24f" />
