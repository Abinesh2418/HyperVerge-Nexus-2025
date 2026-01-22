"""
Utility functions for FinSight - Financial Data Analytics Dashboard
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from typing import Tuple, Dict, Any
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta


def load_data(file_path: str = "../data/preprocessed_transactions.csv") -> pd.DataFrame:
    """
    Load transaction data from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with transaction data
    """
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'])
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {file_path}. Please run the notebook first to generate data.")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def load_models() -> Dict[str, Any]:
    """
    Load all trained models and scaler.
    
    Returns:
        Dictionary containing models and metadata
    """
    models = {}
    
    try:
        # Load Prophet forecasting model
        models['prophet'] = joblib.load('../models/prophet_forecasting_model.pkl')
        
        # Load loan classifier
        models['classifier'] = joblib.load('../models/loan_classifier_model.pkl')
        
        # Load scaler
        models['scaler'] = joblib.load('../models/scaler.pkl')
        
        # Load feature names
        with open('../models/feature_names.json', 'r') as f:
            models['features'] = json.load(f)
            
        return models
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model files not found. Please run the notebook first to train and save models. Error: {str(e)}")
    except Exception as e:
        raise Exception(f"Error loading models: {str(e)}")


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate dataset overview statistics.
    
    Args:
        df: Transaction DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    # Ensure isSalary is properly encoded
    if df['isSalary'].dtype == 'object':
        df['isSalary'] = df['isSalary'].map({'Yes': 1, 'No': 0})
    
    summary = {
        'total_records': len(df),
        'date_range': {
            'start': df['Date'].min().strftime('%Y-%m-%d'),
            'end': df['Date'].max().strftime('%Y-%m-%d'),
            'days': (df['Date'].max() - df['Date'].min()).days
        },
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'total_transactions': len(df),
        'credit_count': len(df[df['Debit/Credit'].str.lower() == 'credit']),
        'debit_count': len(df[df['Debit/Credit'].str.lower() == 'debit']),
        'salary_transactions': df['isSalary'].sum() if 'isSalary' in df.columns else 0,
        'non_salary_transactions': len(df) - (df['isSalary'].sum() if 'isSalary' in df.columns else 0),
        'total_amount': df['Amount'].sum(),
        'avg_transaction': df['Amount'].mean(),
        'median_transaction': df['Amount'].median()
    }
    
    return summary


def create_monthly_trend_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create monthly spending trends visualization.
    
    Args:
        df: Transaction DataFrame
        
    Returns:
        Plotly figure object
    """
    # Aggregate by month
    df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
    
    monthly_data = df.groupby('YearMonth').agg({
        'Income': 'sum',
        'Spending': 'sum',
        'Amount': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=monthly_data['YearMonth'],
        y=monthly_data['Income'],
        mode='lines+markers',
        name='Income',
        line=dict(color='#2ecc71', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=monthly_data['YearMonth'],
        y=monthly_data['Spending'],
        mode='lines+markers',
        name='Spending',
        line=dict(color='#e74c3c', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Monthly Income vs Spending Trends',
        xaxis_title='Month',
        yaxis_title='Amount',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def create_category_analysis(df: pd.DataFrame) -> go.Figure:
    """
    Create category-wise expense analysis.
    
    Args:
        df: Transaction DataFrame
        
    Returns:
        Plotly figure object
    """
    # Category analysis by Debit/Credit
    category_data = df.groupby('Debit/Credit')['Amount'].sum().reset_index()
    
    fig = px.pie(
        category_data,
        values='Amount',
        names='Debit/Credit',
        title='Credit vs Debit Distribution',
        color_discrete_map={'Credit': '#2ecc71', 'Debit': '#e74c3c'},
        hole=0.4
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig


def create_credit_debit_timeline(df: pd.DataFrame) -> go.Figure:
    """
    Create timeline showing credit vs debit over time.
    
    Args:
        df: Transaction DataFrame
        
    Returns:
        Plotly figure object
    """
    df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
    
    timeline_data = df.groupby(['YearMonth', 'Debit/Credit'])['Amount'].sum().reset_index()
    
    fig = px.bar(
        timeline_data,
        x='YearMonth',
        y='Amount',
        color='Debit/Credit',
        title='Credit vs Debit Over Time',
        color_discrete_map={'Credit': '#2ecc71', 'Debit': '#e74c3c'},
        barmode='group'
    )
    
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Amount',
        height=400,
        template='plotly_white'
    )
    
    return fig


def detect_anomalies(df: pd.DataFrame, threshold: float = 2.5) -> pd.DataFrame:
    """
    Detect anomalies/spikes in transaction amounts.
    
    Args:
        df: Transaction DataFrame
        threshold: Standard deviation threshold for anomaly detection
        
    Returns:
        DataFrame with anomaly flags
    """
    df_copy = df.copy()
    
    # Calculate z-score for amounts
    mean_amount = df_copy['Amount'].mean()
    std_amount = df_copy['Amount'].std()
    
    df_copy['z_score'] = (df_copy['Amount'] - mean_amount) / std_amount
    df_copy['is_anomaly'] = abs(df_copy['z_score']) > threshold
    
    return df_copy[df_copy['is_anomaly']].sort_values('Amount', ascending=False)


def prepare_forecast_data(df: pd.DataFrame, models: Dict[str, Any]) -> pd.DataFrame:
    """
    Prepare data for Prophet forecasting.
    
    Args:
        df: Transaction DataFrame
        models: Dictionary containing models and scaler
        
    Returns:
        Prepared DataFrame for forecasting
    """
    # Binary encode salary flag
    df_copy = df.copy()
    if df_copy['isSalary'].dtype == 'object':
        df_copy['isSalary'] = df_copy['isSalary'].map({'Yes': 1, 'No': 0})
    
    # Aggregate daily totals
    daily = (
        df_copy.groupby('Date')
        .Amount.sum()
        .reset_index()
        .rename(columns={'Date': 'ds', 'Amount': 'y'})
    )
    
    regs = (
        df_copy.groupby('Date')
        .agg(
            isSalary=('isSalary', 'max'),
            Income=('Income', 'sum'),
            Spending=('Spending', 'sum'),
            Loan_Debit_Count=('Loan_Debit_Count', 'sum')
        )
        .reset_index()
        .rename(columns={'Date': 'ds'})
    )
    
    # Scale features
    features = models['features']
    scaler = models['scaler']
    X_scaled = scaler.transform(regs[features])
    
    # Merge scaled features with daily data
    forecast_df = daily.merge(
        pd.DataFrame(X_scaled, columns=features, index=regs.index).assign(ds=regs['ds']),
        on='ds',
        how='left'
    ).ffill()
    
    return forecast_df


def make_forecast(prophet_model, historical_df: pd.DataFrame, periods: int = 30) -> pd.DataFrame:
    """
    Generate forecast using Prophet model.
    
    Args:
        prophet_model: Trained Prophet model
        historical_df: Historical data with features
        periods: Number of periods to forecast
        
    Returns:
        DataFrame with forecast
    """
    # Create future dataframe
    last_date = historical_df['ds'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    
    # Get last values for regressors (simple forward fill)
    last_values = historical_df.iloc[-1][['isSalary', 'Income', 'Spending', 'Loan_Debit_Count']]
    
    future_df = pd.DataFrame({
        'ds': future_dates,
        'isSalary': last_values['isSalary'],
        'Income': last_values['Income'],
        'Spending': last_values['Spending'],
        'Loan_Debit_Count': last_values['Loan_Debit_Count']
    })
    
    # Make prediction
    forecast = prophet_model.predict(future_df)
    
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


def create_forecast_chart(historical_df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    """
    Create forecast visualization.
    
    Args:
        historical_df: Historical data
        forecast_df: Forecasted data
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Historical data (last 90 days)
    recent_historical = historical_df.tail(90)
    
    fig.add_trace(go.Scatter(
        x=recent_historical['ds'],
        y=recent_historical['y'],
        mode='lines',
        name='Historical',
        line=dict(color='#3498db', width=2)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='#e74c3c', width=2, dash='dash')
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_upper'],
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_lower'],
        mode='lines',
        name='Confidence Interval',
        line=dict(width=0),
        fillcolor='rgba(231, 76, 60, 0.2)',
        fill='tonexty'
    ))
    
    fig.update_layout(
        title='Transaction Amount Forecast',
        xaxis_title='Date',
        yaxis_title='Amount',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig


def predict_loan_risk(classifier, scaler, features: list, 
                      income: float, spending: float, 
                      balance: float, is_salary: int = 1,
                      loan_debit_count: int = 0) -> Tuple[str, float, Dict[str, float]]:
    """
    Predict loan repayment risk category.
    
    Args:
        classifier: Trained classifier model
        scaler: Feature scaler
        features: List of feature names
        income: Average monthly income
        spending: Average monthly spending
        balance: Average balance
        is_salary: Salary flag (0 or 1)
        loan_debit_count: Number of loan debits
        
    Returns:
        Tuple of (risk_category, probability, feature_importance)
    """
    # Prepare input features
    input_data = pd.DataFrame({
        'isSalary': [is_salary],
        'Income': [income],
        'Spending': [spending],
        'Loan_Debit_Count': [loan_debit_count]
    })
    
    # Scale features
    input_scaled = scaler.transform(input_data[features])
    
    # Predict
    prediction = classifier.predict(input_scaled)[0]
    probability = classifier.predict_proba(input_scaled)[0]
    
    # Determine risk category
    if prediction == 1:
        risk_category = "Low Risk"
        risk_prob = probability[1]
    else:
        risk_category = "High Risk"
        risk_prob = probability[0]
    
    # Get feature importance
    feature_importance = dict(zip(features, classifier.feature_importances_))
    
    # Calculate additional metrics
    expense_ratio = spending / income if income > 0 else 1.0
    savings_rate = (income - spending) / income if income > 0 else 0.0
    
    return risk_category, risk_prob, feature_importance, {
        'expense_ratio': expense_ratio,
        'savings_rate': savings_rate,
        'eligibility': prediction
    }


def explain_prediction(risk_category: str, metrics: Dict[str, float]) -> str:
    """
    Generate human-readable explanation of risk prediction.
    
    Args:
        risk_category: Predicted risk category
        metrics: Dictionary of financial metrics
        
    Returns:
        Explanation string
    """
    expense_ratio = metrics['expense_ratio']
    savings_rate = metrics['savings_rate']
    
    explanation = f"**Risk Assessment: {risk_category}**\n\n"
    
    if risk_category == "Low Risk":
        explanation += "✅ **Eligible for Loan Repayment**\n\n"
        explanation += "**Analysis:**\n"
        explanation += f"- Your expense ratio is {expense_ratio:.2%}, indicating good financial management\n"
        explanation += f"- You maintain a savings rate of {savings_rate:.2%}\n"
        explanation += "- Your income exceeds spending, showing financial stability\n"
        explanation += "- You have sufficient balance to handle loan obligations\n"
    else:
        explanation += "⚠️ **High Risk - Not Recommended for Loan**\n\n"
        explanation += "**Analysis:**\n"
        explanation += f"- Your expense ratio is {expense_ratio:.2%}, which may indicate financial stress\n"
        explanation += f"- Your savings rate is {savings_rate:.2%}\n"
        
        if expense_ratio >= 1.0:
            explanation += "- Your spending exceeds income, which is concerning\n"
        if savings_rate <= 0:
            explanation += "- You are not able to save, indicating tight finances\n"
        
        explanation += "\n**Recommendations:**\n"
        explanation += "- Reduce discretionary spending\n"
        explanation += "- Increase income sources if possible\n"
        explanation += "- Build emergency savings before taking loans\n"
    
    return explanation
