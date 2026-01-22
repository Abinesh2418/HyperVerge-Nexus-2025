"""
FinSight - Financial Data Analytics Dashboard
A Streamlit application for financial transaction analysis, forecasting, and loan risk assessment
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import utils

# Page configuration
st.set_page_config(
    page_title="FinSight - Financial Analytics",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# App title
st.markdown('<div class="main-header">💰 FinSight – Financial Data Analytics</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar navigation
st.sidebar.title("📊 Navigation")
st.sidebar.markdown("Select a section to explore:")

section = st.sidebar.radio(
    "",
    ["🏠 Home", "📈 Dataset Overview", "🔍 EDA & Trends", "🔮 Forecasting", "⚠️ Loan Risk Assessment"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "**FinSight** provides comprehensive financial analytics including:\n"
    "- Transaction insights\n"
    "- Spending patterns\n"
    "- Future forecasting\n"
    "- Loan risk prediction"
)

# Load data and models
@st.cache_data
def load_app_data():
    """Load transaction data"""
    try:
        return utils.load_data()
    except FileNotFoundError:
        st.error("⚠️ Data file not found. Please run the Jupyter notebook first to generate the preprocessed data.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

@st.cache_resource
def load_app_models():
    """Load trained models"""
    try:
        return utils.load_models()
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please run the Jupyter notebook first to train and save the models.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

# Load data
df = load_app_data()
models = load_app_models()


# ============================================================================
# HOME SECTION
# ============================================================================
if section == "🏠 Home":
    st.markdown('<div class="section-header">Welcome to FinSight</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Data Analytics")
        st.write("Explore your financial transaction data with comprehensive visualizations and statistics.")
    
    with col2:
        st.markdown("### 🔮 AI Forecasting")
        st.write("Predict future transaction amounts using advanced time-series forecasting with Prophet.")
    
    with col3:
        st.markdown("### ⚠️ Risk Assessment")
        st.write("Evaluate loan repayment capability using machine learning classification models.")
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("### 📈 Quick Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", f"{len(df):,}")
    
    with col2:
        date_range = (df['Date'].max() - df['Date'].min()).days
        st.metric("Data Period (Days)", f"{date_range:,}")
    
    with col3:
        total_amount = df['Amount'].sum()
        st.metric("Total Transaction Amount", f"₹{total_amount:,.2f}")
    
    with col4:
        avg_amount = df['Amount'].mean()
        st.metric("Average Transaction", f"₹{avg_amount:,.2f}")
    
    st.markdown("---")
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    **Getting Started:**
    1. Navigate using the sidebar menu
    2. Start with **Dataset Overview** to understand your data
    3. Explore **EDA & Trends** for insights and patterns
    4. Use **Forecasting** to predict future transactions
    5. Try **Loan Risk Assessment** to evaluate repayment capability
    """)
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# SECTION 1: DATASET OVERVIEW
# ============================================================================
elif section == "📈 Dataset Overview":
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
    
    # Get summary statistics
    summary = utils.get_dataset_summary(df)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{summary['total_records']:,}")
    
    with col2:
        st.metric("Date Range", f"{summary['date_range']['days']} days")
    
    with col3:
        st.metric("Total Amount", f"₹{summary['total_amount']:,.2f}")
    
    with col4:
        st.metric("Avg Transaction", f"₹{summary['avg_transaction']:,.2f}")
    
    st.markdown("---")
    
    # Date range details
    st.markdown("### 📅 Date Range")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Start Date:** {summary['date_range']['start']}")
    with col2:
        st.info(f"**End Date:** {summary['date_range']['end']}")
    
    # Transaction breakdown
    st.markdown("### 💳 Transaction Breakdown")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Credit Transactions", f"{summary['credit_count']:,}")
        credit_pct = (summary['credit_count'] / summary['total_transactions']) * 100
        st.caption(f"{credit_pct:.1f}% of total")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Debit Transactions", f"{summary['debit_count']:,}")
        debit_pct = (summary['debit_count'] / summary['total_transactions']) * 100
        st.caption(f"{debit_pct:.1f}% of total")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Salary Transactions", f"{summary['salary_transactions']:,}")
        salary_pct = (summary['salary_transactions'] / summary['total_transactions']) * 100
        st.caption(f"{salary_pct:.1f}% of total")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Columns info
    st.markdown("### 📋 Dataset Columns")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write(f"**Total Columns:** {len(summary['columns'])}")
        for col in summary['columns']:
            st.text(f"• {col}")
    
    with col2:
        st.markdown("**Missing Values Summary:**")
        missing_df = pd.DataFrame.from_dict(summary['missing_values'], orient='index', columns=['Missing Count'])
        missing_df['Missing %'] = (missing_df['Missing Count'] / summary['total_records'] * 100).round(2)
        st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
        
        if missing_df['Missing Count'].sum() == 0:
            st.success("✅ No missing values in the dataset!")
    
    # Sample data
    st.markdown("### 🔍 Sample Transactions")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Data quality indicators
    st.markdown("### ✅ Data Quality Indicators")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Data Completeness", f"{completeness:.1f}%")
    
    with col2:
        st.metric("Unique Dates", f"{df['Date'].nunique():,}")
    
    with col3:
        st.metric("Median Transaction", f"₹{summary['median_transaction']:,.2f}")


# ============================================================================
# SECTION 2: EDA & TREND ANALYSIS
# ============================================================================
elif section == "🔍 EDA & Trends":
    st.markdown('<div class="section-header">Exploratory Data Analysis & Trends</div>', unsafe_allow_html=True)
    
    # Monthly trends
    st.markdown("### 📊 Monthly Income vs Spending Trends")
    monthly_chart = utils.create_monthly_trend_chart(df)
    st.plotly_chart(monthly_chart, use_container_width=True)
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("**Insight:** This chart shows your monthly income and spending patterns over time. "
                "Look for trends, seasonal variations, or unusual spikes.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💰 Credit vs Debit Distribution")
        category_chart = utils.create_category_analysis(df)
        st.plotly_chart(category_chart, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Credit vs Debit Timeline")
        timeline_chart = utils.create_credit_debit_timeline(df)
        st.plotly_chart(timeline_chart, use_container_width=True)
    
    # Transaction statistics
    st.markdown("### 📊 Transaction Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_income = df['Income'].sum()
        st.metric("Total Income", f"₹{total_income:,.2f}")
    
    with col2:
        total_spending = df['Spending'].sum()
        st.metric("Total Spending", f"₹{total_spending:,.2f}")
    
    with col3:
        net_savings = total_income - total_spending
        st.metric("Net Savings", f"₹{net_savings:,.2f}", delta=f"{(net_savings/total_income)*100:.1f}%" if total_income > 0 else "N/A")
    
    with col4:
        avg_balance = df['Balance'].mean()
        st.metric("Average Balance", f"₹{avg_balance:,.2f}")
    
    # Anomaly detection
    st.markdown("### 🔔 Anomaly Detection - Unusual Transactions")
    
    threshold = st.slider("Set anomaly detection sensitivity (standard deviations)", 
                         min_value=1.5, max_value=4.0, value=2.5, step=0.5,
                         help="Higher values = fewer anomalies detected")
    
    anomalies = utils.detect_anomalies(df, threshold=threshold)
    
    if len(anomalies) > 0:
        st.warning(f"⚠️ Found {len(anomalies)} unusual transactions")
        
        # Display top anomalies
        st.dataframe(
            anomalies[['Date', 'Amount', 'Debit/Credit', 'Entities', 'z_score']]
            .head(10)
            .style.format({'Amount': '₹{:,.2f}', 'z_score': '{:.2f}'}),
            use_container_width=True
        )
        
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("**Note:** These transactions significantly deviate from your typical spending pattern. "
                    "Review them to ensure they are legitimate.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.success("✅ No significant anomalies detected in your transactions.")
    
    # Distribution analysis
    st.markdown("### 📊 Transaction Amount Distribution")
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df['Amount'],
        nbinsx=50,
        name='Transaction Amount',
        marker_color='#3498db'
    ))
    
    fig.update_layout(
        title='Distribution of Transaction Amounts',
        xaxis_title='Amount',
        yaxis_title='Frequency',
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# SECTION 3: TIME-SERIES FORECASTING
# ============================================================================
elif section == "🔮 Forecasting":
    st.markdown('<div class="section-header">Time-Series Forecasting</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("**Forecast Future Transactions** using Facebook Prophet - an advanced time-series forecasting model. "
                "The model considers historical patterns, seasonality, and financial indicators.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # User controls
    st.markdown("### ⚙️ Forecast Configuration")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        forecast_period = st.slider(
            "Select forecast horizon (days)",
            min_value=7,
            max_value=90,
            value=30,
            step=7,
            help="Number of days to forecast into the future"
        )
    
    with col2:
        st.metric("Forecast Period", f"{forecast_period} days")
        st.caption(f"Approximately {forecast_period//30} month(s)")
    
    # Prepare data
    with st.spinner("Preparing forecast data..."):
        forecast_df = utils.prepare_forecast_data(df, models)
    
    # Generate forecast
    if st.button("🔮 Generate Forecast", type="primary"):
        with st.spinner("Generating forecast... This may take a moment."):
            try:
                prophet_model = models['prophet']
                forecast = utils.make_forecast(prophet_model, forecast_df, periods=forecast_period)
                
                st.success("✅ Forecast generated successfully!")
                
                # Display forecast chart
                st.markdown("### 📈 Forecast Visualization")
                forecast_chart = utils.create_forecast_chart(forecast_df, forecast)
                st.plotly_chart(forecast_chart, use_container_width=True)
                
                # Forecast summary
                st.markdown("### 📊 Forecast Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_forecast = forecast['yhat'].mean()
                    st.metric("Avg Forecasted Amount", f"₹{avg_forecast:,.2f}")
                
                with col2:
                    total_forecast = forecast['yhat'].sum()
                    st.metric("Total Forecasted", f"₹{total_forecast:,.2f}")
                
                with col3:
                    max_forecast = forecast['yhat'].max()
                    st.metric("Maximum Expected", f"₹{max_forecast:,.2f}")
                
                with col4:
                    min_forecast = forecast['yhat'].min()
                    st.metric("Minimum Expected", f"₹{min_forecast:,.2f}")
                
                # Forecast details table
                st.markdown("### 📋 Detailed Forecast Data")
                
                forecast_display = forecast.copy()
                forecast_display['ds'] = pd.to_datetime(forecast_display['ds']).dt.strftime('%Y-%m-%d')
                forecast_display.columns = ['Date', 'Forecasted Amount', 'Lower Bound', 'Upper Bound']
                
                st.dataframe(
                    forecast_display.style.format({
                        'Forecasted Amount': '₹{:,.2f}',
                        'Lower Bound': '₹{:,.2f}',
                        'Upper Bound': '₹{:,.2f}'
                    }),
                    use_container_width=True
                )
                
                # Interpretation
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("""
                **How to Interpret:**
                - **Forecasted Amount (yhat):** The model's prediction for transaction amounts
                - **Lower/Upper Bounds:** 95% confidence interval - actual values are likely to fall within this range
                - **Trend:** Observe if spending is expected to increase, decrease, or remain stable
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")
    else:
        st.info("👆 Click the button above to generate your forecast")
        
        # Show recent historical data
        st.markdown("### 📊 Recent Transaction History")
        recent_data = forecast_df.tail(30)[['ds', 'y']].copy()
        recent_data.columns = ['Date', 'Amount']
        recent_data['Date'] = pd.to_datetime(recent_data['Date']).dt.strftime('%Y-%m-%d')
        
        st.dataframe(
            recent_data.style.format({'Amount': '₹{:,.2f}'}),
            use_container_width=True
        )


# ============================================================================
# SECTION 4: LOAN REPAYMENT CLASSIFICATION
# ============================================================================
elif section == "⚠️ Loan Risk Assessment":
    st.markdown('<div class="section-header">Loan Repayment Risk Assessment</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("**Evaluate Loan Repayment Capability** using machine learning. "
                "Input your financial metrics to assess the risk of loan default.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input form
    st.markdown("### 📝 Enter Financial Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        avg_income = st.number_input(
            "Average Monthly Income (₹)",
            min_value=0.0,
            value=50000.0,
            step=1000.0,
            help="Your typical monthly income"
        )
        
        avg_spending = st.number_input(
            "Average Monthly Spending (₹)",
            min_value=0.0,
            value=30000.0,
            step=1000.0,
            help="Your typical monthly expenses"
        )
    
    with col2:
        avg_balance = st.number_input(
            "Average Account Balance (₹)",
            min_value=0.0,
            value=15000.0,
            step=1000.0,
            help="Your typical account balance"
        )
        
        is_salary = st.selectbox(
            "Regular Salary Income?",
            options=[("Yes", 1), ("No", 0)],
            format_func=lambda x: x[0],
            help="Do you receive regular salary income?"
        )[1]
    
    loan_debit_count = st.slider(
        "Number of Existing Loan Debits",
        min_value=0,
        max_value=10,
        value=0,
        help="Number of existing loan-related debit transactions"
    )
    
    # Calculate metrics preview
    st.markdown("### 📊 Financial Health Indicators")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        expense_ratio = (avg_spending / avg_income * 100) if avg_income > 0 else 0
        st.metric("Expense Ratio", f"{expense_ratio:.1f}%")
        if expense_ratio < 60:
            st.success("✅ Good")
        elif expense_ratio < 80:
            st.warning("⚠️ Moderate")
        else:
            st.error("❌ High")
    
    with col2:
        savings_rate = ((avg_income - avg_spending) / avg_income * 100) if avg_income > 0 else 0
        st.metric("Savings Rate", f"{savings_rate:.1f}%")
        if savings_rate > 20:
            st.success("✅ Excellent")
        elif savings_rate > 10:
            st.warning("⚠️ Fair")
        else:
            st.error("❌ Poor")
    
    with col3:
        net_monthly = avg_income - avg_spending
        st.metric("Net Monthly Savings", f"₹{net_monthly:,.2f}")
        if net_monthly > 0:
            st.success("✅ Positive")
        else:
            st.error("❌ Negative")
    
    # Predict button
    st.markdown("---")
    
    if st.button("🎯 Assess Loan Risk", type="primary"):
        with st.spinner("Analyzing your financial profile..."):
            try:
                # Make prediction
                risk_category, risk_prob, feature_importance, metrics = utils.predict_loan_risk(
                    classifier=models['classifier'],
                    scaler=models['scaler'],
                    features=models['features'],
                    income=avg_income,
                    spending=avg_spending,
                    balance=avg_balance,
                    is_salary=is_salary,
                    loan_debit_count=loan_debit_count
                )
                
                st.markdown("---")
                st.markdown("### 🎯 Risk Assessment Results")
                
                # Display risk category
                if risk_category == "Low Risk":
                    st.success(f"## ✅ {risk_category}")
                    st.markdown(f"**Confidence:** {risk_prob*100:.1f}%")
                else:
                    st.error(f"## ⚠️ {risk_category}")
                    st.markdown(f"**Confidence:** {risk_prob*100:.1f}%")
                
                # Progress bar for confidence
                st.progress(risk_prob)
                
                # Explanation
                st.markdown("### 📋 Detailed Analysis")
                explanation = utils.explain_prediction(risk_category, metrics)
                st.markdown(explanation)
                
                # Feature importance
                st.markdown("### 🔍 Key Factors Affecting Decision")
                
                importance_df = pd.DataFrame.from_dict(
                    feature_importance,
                    orient='index',
                    columns=['Importance']
                ).sort_values('Importance', ascending=False)
                
                fig = go.Figure(go.Bar(
                    x=importance_df['Importance'],
                    y=importance_df.index,
                    orientation='h',
                    marker_color='#3498db'
                ))
                
                fig.update_layout(
                    title='Feature Importance',
                    xaxis_title='Importance Score',
                    yaxis_title='Feature',
                    height=300,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("""
                **Understanding the Factors:**
                - **Income:** Higher income improves repayment capability
                - **Spending:** Lower spending relative to income is favorable
                - **isSalary:** Regular salary income indicates stability
                - **Loan_Debit_Count:** Fewer existing loans reduce risk
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Additional recommendations
                if risk_category == "High Risk":
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.markdown("""
                    ### 💡 Recommendations to Improve Your Financial Health:
                    1. **Reduce Expenses:** Cut discretionary spending by 10-20%
                    2. **Increase Income:** Consider additional income sources
                    3. **Build Emergency Fund:** Save at least 3-6 months of expenses
                    4. **Pay Off Existing Debts:** Reduce loan obligations before taking new ones
                    5. **Track Spending:** Use budgeting tools to monitor expenses
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    else:
        st.info("👆 Enter your financial information above and click the button to assess loan risk")

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #7f8c8d; padding: 2rem;">'
    '💰 FinSight - Financial Data Analytics | Built with Streamlit & Prophet | '
    'HyperVerge Nexus Project'
    '</div>',
    unsafe_allow_html=True
)
