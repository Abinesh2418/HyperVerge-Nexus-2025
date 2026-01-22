# HyperVerge-Nexus

## Overview
HyperVerge-Nexus is a **real-time machine learning project** for financial forecasting and risk assessment using bank transaction data. This project was developed based on various kinds of bank transaction datasets provided for analysis. The project includes comprehensive **Exploratory Data Analysis (EDA)** and implements two primary predictive models:

1. **Transaction Amount Forecasting (Prophet Model)** - Uses the last 3 months of transaction data to predict the next 3 months of transaction amounts with custom regressors
2. **Loan Repayment Risk Prediction (Classification Model)** - Evaluates whether a user can successfully repay a loan by assessing their loan eligibility and performing risk analysis

## 💰 FinSight - Streamlit Dashboard

**NEW:** This project now includes a complete Streamlit web application for interactive financial analytics!

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run the notebook to generate models and data
# Then launch the Streamlit app
streamlit run app.py
```

See the complete FinSight documentation below for full setup and usage instructions.

## Project Structure
```
HyperVerge-Nexus/
├── data/                                    # Transaction data
│   └── preprocessed_transactions.csv
├── models/                                  # Trained ML models
│   ├── prophet_forecasting_model.pkl
│   ├── loan_classifier_model.pkl
│   ├── scaler.pkl
│   └── feature_names.json
├── notebooks/                               # Analysis notebooks
│   ├── HyperVerge_Nexus_Final_1(Dataset).ipynb
│   └── HyperVerge_Nexus_Final_30(Dataset).ipynb
├── app.py                                   # Streamlit dashboard
├── utils.py                                 # Helper functions
├── requirements.txt                         # Python dependencies
├── .gitignore                              # Git ignore rules
└── README.md                               # This file
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

### Option 1: Interactive Streamlit Dashboard (Recommended)

**Step 1:** Install dependencies
```bash
pip install -r requirements.txt
```

**Step 2:** Generate models and data
- Open and run all cells in `HyperVerge_Nexus_Final_1(Dataset).ipynb`
- This will create the required files in `data/` and `models/` directories

**Step 3:** Launch the Streamlit app
```bash
streamlit run app.py
```

**Step 4:** Explore the dashboard
- 📈 **Dataset Overview** - View comprehensive statistics
- 🔍 **EDA & Trends** - Analyze spending patterns and detect anomalies
- 🔮 **Forecasting** - Predict future transactions (7-90 days)
- ⚠️ **Loan Risk Assessment** - Evaluate loan repayment capability

### Option 2: Jupyter Notebook Analysis

#### For Single Dataset Analysis
Open and run `HyperVerge_Nexus_Final_1(Dataset).ipynb`:
1. Load preprocessed transaction CSV
2. Execute forecasting model cells
3. Execute risk assessment model cells
4. Review visualizations and metrics

#### For Multi-Dataset Analysis
Open and run `HyperVerge_Nexus_Final_30(Dataset).ipynb`:
1. Mount Google Drive
2. Set folder path containing CSV files
3. Run batch preprocessing
4. Execute Prophet forecasting for multiple metrics
5. Analyze monthly aggregations

---

# FinSight Dashboard - Complete Documentation

## 🎯 Features

### 1. Dataset Overview 📊
- Total records and date range analysis
- Key column information and data quality metrics
- Transaction breakdown (Credit/Debit, Salary/Non-salary)
- Missing values summary
- Sample data preview

### 2. EDA & Trend Analysis 🔍
- Monthly income vs spending trends with interactive charts
- Category-wise expense distribution (pie charts)
- Credit vs debit timeline visualization
- **Anomaly Detection** with configurable sensitivity
- Transaction amount distribution analysis
- Financial health indicators

### 3. Time-Series Forecasting 🔮
- **Model:** Facebook Prophet with regressors
- **Features:** Income, Spending, Salary flag, Loan debit count
- **Forecast Horizon:** Configurable (7-90 days)
- **Visualization:** Historical data + forecasted values + confidence intervals
- **Output:** Average, total, min/max predictions with detailed tables
- Clear interpretation for non-technical stakeholders

### 4. Loan Repayment Risk Assessment ⚠️
- **Model:** Random Forest Classifier (96.15% AUC)
- **Input Features:**
  - Average monthly income
  - Average monthly spending
  - Average account balance
  - Salary income flag (Yes/No)
  - Existing loan debit count
- **Output:**
  - Risk category (Low Risk / High Risk)
  - Confidence score with progress bar
  - Feature importance visualization
  - Detailed explanation with personalized recommendations
- **Metrics:** Expense ratio, savings rate, net monthly savings

## 🛠️ Installation Guide

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook (for model training)

### Installation Steps

1. **Navigate to project directory**
```bash
cd d:\Projects\Hyperverge-Nexus\HyperVerge-Nexus-2025
```

2. **Create virtual environment (recommended)**
```powershell
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Prepare data and models**

   **Important:** Execute the notebook first!
   ```bash
   jupyter notebook HyperVerge_Nexus_Final_1(Dataset).ipynb
   ```
   
   Run all cells to:
   - Load and preprocess transaction data
   - Train the Prophet forecasting model
   - Train the Random Forest classifier
   - Save models to `models/` directory
   - Save processed data to `data/` directory

5. **Verify files are created**
   - `data/preprocessed_transactions.csv`
   - `models/prophet_forecasting_model.pkl`
   - `models/loan_classifier_model.pkl`
   - `models/scaler.pkl`
   - `models/feature_names.json`

6. **Launch Streamlit app**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📋 Usage Guide

### Home Page 🏠
- Overview of all features
- Quick statistics dashboard
- Navigation instructions

### Dataset Overview 📈
1. View comprehensive dataset statistics
2. Check data quality and completeness
3. Explore transaction breakdowns by type
4. Review sample transactions

### EDA & Trends 🔍
1. Analyze monthly income vs spending trends
2. Explore credit/debit distribution (pie chart)
3. View credit/debit timeline (bar chart)
4. **Detect anomalies:**
   - Use sensitivity slider (1.5-4.0 standard deviations)
   - Review flagged unusual transactions
   - Higher threshold = fewer anomalies
5. Analyze transaction amount distribution

### Forecasting 🔮
1. Select forecast horizon using slider (7-90 days)
2. Click "Generate Forecast" button
3. Review interactive forecast chart with:
   - Historical data (last 90 days)
   - Forecasted values (dashed line)
   - Confidence intervals (shaded area)
4. Analyze forecast summary metrics
5. Export detailed forecast table

### Loan Risk Assessment ⚠️
1. **Enter financial information:**
   - Average monthly income (₹)
   - Average monthly spending (₹)
   - Average account balance (₹)
   - Regular salary income? (Yes/No)
   - Number of existing loan debits (0-10)

2. **Review financial health indicators:**
   - Expense ratio (should be < 60%)
   - Savings rate (should be > 20%)
   - Net monthly savings

3. **Click "Assess Loan Risk"**

4. **Review results:**
   - Risk category (Low/High)
   - Confidence score
   - Detailed analysis
   - Feature importance chart
   - Personalized recommendations

## 🧪 Technical Details

### Forecasting Model
- **Algorithm:** Facebook Prophet
- **Regressors:** isSalary, Income, Spending, Loan_Debit_Count
- **Seasonality:** Yearly + Weekly
- **Preprocessing:** StandardScaler normalization
- **Performance:** R² ≈ 1.0

### Classification Model
- **Algorithm:** Random Forest
- **Parameters:** 100 estimators, max_depth=3
- **Features:** Same as forecasting (scaled)
- **Target:** Binary eligibility (1=eligible, 0=high risk)
- **Performance:** 97% accuracy, 96.15% AUC

### Tech Stack
- **Frontend:** Streamlit
- **Visualization:** Plotly, Matplotlib, Seaborn
- **ML:** Scikit-learn, Prophet
- **Data:** Pandas, NumPy
- **Persistence:** Joblib

## 🐛 Troubleshooting

### Error: Models Not Found
**Solution:** Run the Jupyter notebook first to train and save models.

### Error: Data File Not Found
**Solution:** Ensure `data/preprocessed_transactions.csv` exists after running the notebook.

### Prophet Installation Issues
```bash
# Try these commands
pip install --upgrade pip
pip install pystan
pip install prophet
```

### Streamlit Won't Start
```bash
streamlit --version  # Check installation
pip install --upgrade streamlit
```

## 📊 Sample Workflow

1. **Data Analyst:** Use EDA section to identify spending patterns and anomalies
2. **Finance Manager:** Review forecasts for budget planning
3. **Loan Officer:** Assess customer loan eligibility using risk assessment tool
4. **Stakeholder:** Review home page dashboard for quick insights

## 🎨 Design Philosophy

- **Clarity over complexity** - Clean, professional interface
- **Stakeholder-friendly** - Non-technical explanations
- **Interactive visualizations** - Plotly charts with hover details
- **Modular code** - Separate utils.py for maintainability
- **Production-ready** - Error handling and validation

## 📄 License

This project is part of the HyperVerge Nexus program.

## 🙏 Acknowledgments

- **HyperVerge Nexus** for providing financial transaction datasets
- **Facebook Prophet** for time-series forecasting
- **Streamlit** for the dashboard framework
- **Scikit-learn** for machine learning tools

---

**Status:** ✅ Production Ready  
**Last Updated:** January 2026  
**Version:** 1.0.0

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