# HyperVerge-Nexus

## Overview
HyperVerge-Nexus is a **real-time machine learning project** for financial forecasting and risk assessment using bank transaction data. This project was developed based on various kinds of bank transaction datasets provided for analysis. The project includes comprehensive **Exploratory Data Analysis (EDA)** and implements two primary predictive models:

1. **Transaction Amount Forecasting (Prophet Model)** - Uses the last 3 months of transaction data to predict the next 3 months of transaction amounts with custom regressors
2. **Loan Repayment Risk Prediction (Classification Model)** - Evaluates whether a user can successfully repay a loan by assessing their loan eligibility and performing risk analysis

## Project Structure
```
HyperVerge-Nexus/
├── HyperVerge_Nexus_Final_1(Dataset).ipynb   # Single dataset analysis
├── HyperVerge_Nexus_Final_30(Dataset).ipynb  # Multi-dataset analysis (30 CSVs)
└── README.md
```

## Features

### Model 1: Transaction Amount Forecasting (Prophet Model)
- **Purpose**: Predict the next 3 months of transaction amounts based on the last 3 months of historical data
- **Algorithm**: Facebook Prophet with custom regressors
- **Features**:
  - Salary indicator (binary)
  - Income amount
  - Spending amount
  - Loan/Debit count
- **Preprocessing**:
  - Daily aggregation of transactions
  - StandardScaler normalization
  - 80/20 train-test split
- **Performance**: R² ≈ 1.0000 (near-perfect fit)
- **Key Components**:
  - Yearly and weekly seasonality
  - Custom changepoint and seasonality priors
  - Feature scaling with StandardScaler

### Model 2: Loan Repayment Risk Prediction (Classification Model)
- **Purpose**: Binary classification to predict whether a user can successfully repay a loan
- **Output**:
  - `1` = User can repay the loan (eligible, low risk)
  - `0` = User cannot repay the loan (not eligible, high risk)
- **Performance**: AUC Score = 0.9615 (96.15% accuracy)
- **Application**: Risk assessment for loan approval decisions

### Extended Analysis (30 Datasets)
- **Data Processing**:
  - Batch processing of multiple CSV files from Google Drive
  - Automated preprocessing pipeline
  - Monthly aggregation with advanced features
- **Features**:
  - Income/Expenditure tracking
  - Cumulative savings
  - Transaction count and volatility
  - Cyclic month encoding (sine/cosine)
  - Average transaction gap
- **Forecasting**:
  - Income forecasting
  - Expenditure forecasting
  - Savings forecasting
  - Cumulative income trends

## Technologies Used
- **Python Libraries**:
  - `pandas` - Data manipulation
  - `scikit-learn` - Feature scaling and metrics
  - `prophet` - Time series forecasting
  - `numpy` - Numerical operations
  - `matplotlib` / `seaborn` - Visualization
- **Environment**: Google Colab / Jupyter Notebook

## Data Requirements
- **Project Type**: Real-time project with various kinds of bank transaction datasets
- **Input Format**: CSV files with bank transaction data
- **Preprocessing Steps**: Comprehensive Exploratory Data Analysis (EDA) and feature engineering
- **Required Columns**:
  - `Date` - Transaction date
  - `Amount` - Transaction amount
  - `Debit/Credit` - Transaction type
  - `Entities` - Transaction entity/merchant
  - `Mode` - Payment mode
  - `isSalary` - Salary indicator (for model 1)

## Model Performance

### Forecasting Model
- **R² Score**: 1.0000
- **Mean Absolute Error**: Near-zero
- **Key Success Factors**:
  - Clear seasonality patterns
  - Strong predictive regressors
  - Effective feature scaling
  - High target variance

### Risk Assessment Model
- **AUC Score**: 0.9615
- **Interpretation**: 96.15% probability of correctly distinguishing between eligible and non-eligible users

## Usage

### For Single Dataset Analysis
Open and run `HyperVerge_Nexus_Final_1(Dataset).ipynb`:
1. Load preprocessed transaction CSV
2. Execute forecasting model cells
3. Execute risk assessment model cells
4. Review visualizations and metrics

### For Multi-Dataset Analysis
Open and run `HyperVerge_Nexus_Final_30(Dataset).ipynb`:
1. Mount Google Drive
2. Set folder path containing CSV files
3. Run batch preprocessing
4. Execute Prophet forecasting for multiple metrics
5. Analyze monthly aggregations

## Key Insights
- The forecasting model achieves near-perfect predictions due to strong seasonality and regressor features
- Risk assessment model effectively identifies loan-eligible users with 96% accuracy
- Monthly aggregation reveals spending patterns and financial behavior trends
- Cyclic encoding captures seasonal spending variations

## Future Enhancements
- Real-time transaction processing
- Additional classification models for fraud detection
- Interactive dashboard for visualization
- API integration for production deployment
- Anomaly detection for unusual spending patterns

## License
This project is part of the HyperVerge study initiative.

## Author
Created as part of HyperVerge financial analytics research project.

---

## 📬 Contact
For any queries or suggestions, feel free to reach out:

📧 **Email**: abineshbalasubramaniyam@example.com  
💼 **LinkedIn**: [linkedin.com/in/abinesh-b-1b14a1290/](https://linkedin.com/in/abinesh-b-1b14a1290/)  
🐙 **GitHub**: [github.com/Abinesh2418](https://github.com/Abinesh2418)