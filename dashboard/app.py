import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG & STYLING
# ============================================================
st.set_page_config(
    page_title="Credit Risk Loan Pricing Engine",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6B7280;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
        color: #1E3A5F;
    }
    .stButton>button {
        background-color: #1E3A5F;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover { background-color: #2C5282; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL ARTIFACTS (lightweight JSON — no scikit-learn needed)
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_artifacts():
    with open(BASE_DIR / 'model_coefficients.json', 'r') as f:
        model = json.load(f)
    with open(BASE_DIR / 'pricing_artifacts.json', 'r') as f:
        pricing = json.load(f)
    return model, pricing

model, pricing = load_artifacts()

FEATURE_NAMES = model['feature_names']
COEFS = np.array(model['coefficients'])
INTERCEPT = model['intercept']
SCALER_MEAN = np.array(model['scaler_mean'])
SCALER_SCALE = np.array(model['scaler_scale'])

# ============================================================
# CORE PRICING LOGIC
# ============================================================
def predict_pd(feature_values: dict) -> float:
    """Manually reproduce StandardScaler + Logistic Regression prediction
    from saved coefficients — no scikit-learn dependency needed."""
    x = np.array([feature_values[f] for f in FEATURE_NAMES])
    x_scaled = (x - SCALER_MEAN) / SCALER_SCALE
    linear_pred = INTERCEPT + np.dot(COEFS, x_scaled)
    return 1 / (1 + np.exp(-linear_pred))

def compute_pricing(revolving_util, age, debt_ratio, monthly_income,
                     open_credit_lines, real_estate_loans, dependents, total_past_due):
    feature_values = {
        'RevolvingUtilizationOfUnsecuredLines': revolving_util,
        'age': age,
        'DebtRatio': debt_ratio,
        'MonthlyIncome': monthly_income,
        'NumberOfOpenCreditLinesAndLoans': open_credit_lines,
        'NumberRealEstateLoansOrLines': real_estate_loans,
        'NumberOfDependents': dependents,
        'TotalTimesPastDue': total_past_due,
    }
    pd_est = predict_pd(feature_values)

    lgd = pricing['LGD_SECURED'] if real_estate_loans > 0 else pricing['LGD_UNSECURED']
    ead = min(max(monthly_income * pricing['EAD_INCOME_MULTIPLE'], pricing['EAD_FLOOR']), pricing['EAD_CAP'])

    expected_loss = pd_est * lgd * ead
    expected_loss_rate = expected_loss / ead
    interest_rate = (
        pricing['COST_OF_FUNDS'] + expected_loss_rate +
        pricing['OPERATING_COST_MARGIN'] + pricing['PROFIT_MARGIN']
    )
    declined = pd_est > 0.50

    return {
        'pd': pd_est, 'lgd': lgd, 'ead': ead,
        'expected_loss': expected_loss, 'expected_loss_rate': expected_loss_rate,
        'interest_rate': interest_rate, 'declined': declined
    }

def risk_gauge(pd_value, title="Probability of Default"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pd_value * 100,
        number={'suffix': '%'},
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 60]},
            'bar': {'color': "#1E3A5F"},
            'steps': [
                {'range': [0, 10], 'color': "#D1FAE5"},
                {'range': [10, 25], 'color': "#FEF3C7"},
                {'range': [25, 60], 'color': "#FEE2E2"},
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(t=40, b=10, l=20, r=20))
    return fig

# ============================================================
# HEADER
# ============================================================
st.markdown('<p class="main-header">💳 Credit Risk Loan Pricing Engine</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Basel-aligned pricing model — Probability of Default (Logistic Regression) '
    '× LGD × EAD, trained on 150,000 borrower records (Give Me Some Credit dataset)</p>',
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs(["💳 Loan Quote", "📊 Risk Factor Insights", "⚖️ Compare Borrowers"])

# ============================================================
# TAB 1 — LOAN QUOTE
# ============================================================
with tab1:
    st.sidebar.header("📋 Borrower Profile")
    revolving_util = st.sidebar.slider("Credit Utilization Ratio", 0.0, 2.0, 0.3, 0.01, key="q_util",
                                        help="Balance on credit lines relative to credit limits")
    age = st.sidebar.slider("Age", 18, 100, 40, key="q_age")
    debt_ratio = st.sidebar.slider("Debt Ratio", 0.0, 5.0, 0.35, 0.01, key="q_debt")
    monthly_income = st.sidebar.slider("Monthly Income (€)", 0, 20000, 5000, 100, key="q_income")
    open_credit_lines = st.sidebar.slider("Open Credit Lines / Loans", 0, 30, 8, key="q_lines")
    real_estate_loans = st.sidebar.slider("Real Estate Loans/Lines", 0, 10, 1, key="q_re")
    dependents = st.sidebar.slider("Number of Dependents", 0, 10, 0, key="q_dep")
    total_past_due = st.sidebar.slider("Total Times Past Due", 0, 20, 0, key="q_pastdue",
                                        help="Combined count of 30-59, 60-89, and 90+ day late payments")

    r = compute_pricing(revolving_util, age, debt_ratio, monthly_income,
                         open_credit_lines, real_estate_loans, dependents, total_past_due)

    if r['declined']:
        st.error(f"⛔ **Application would be DECLINED** — estimated PD ({r['pd']*100:.1f}%) exceeds the "
                 f"50% acceptance threshold. Pricing alone cannot compensate for this level of risk.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Probability of Default", f"{r['pd']*100:.2f}%")
    col2.metric("Loss Given Default", f"{r['lgd']*100:.0f}%")
    col3.metric("Exposure at Default", f"€{r['ead']:,.0f}")
    col4.metric("💰 Interest Rate", f"{r['interest_rate']*100:.2f}%")

    st.divider()
    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.subheader("Expected Loss Breakdown")
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "absolute", "absolute", "total"],
            x=["Cost of Funds", "Expected Loss Rate", "Operating + Profit Margin", "Interest Rate"],
            y=[pricing['COST_OF_FUNDS']*100, r['expected_loss_rate']*100,
               (pricing['OPERATING_COST_MARGIN']+pricing['PROFIT_MARGIN'])*100, 0],
            connector={"line": {"color": "#CBD5E1"}},
            increasing={"marker": {"color": "#3B82F6"}},
            totals={"marker": {"color": "#1E3A5F"}}
        ))
        fig.update_layout(height=380, margin=dict(t=20, b=20), showlegend=False,
                           yaxis_title="Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Risk Gauge")
        st.plotly_chart(risk_gauge(r['pd']), use_container_width=True)
        st.write(f"**Expected Loss (€):** €{r['expected_loss']:,.2f}")
        st.write(f"**Collateral status:** {'Secured (real estate)' if real_estate_loans > 0 else 'Unsecured'}")

    st.download_button(
        "📄 Download Loan Quote",
        data=(
            f"CREDIT RISK LOAN QUOTE\n{'='*30}\n"
            f"Age: {age}  |  Monthly Income: €{monthly_income:,}\n"
            f"Credit Utilization: {revolving_util:.2f}  |  Debt Ratio: {debt_ratio:.2f}\n"
            f"Past Due History: {total_past_due}  |  Real Estate Loans: {real_estate_loans}\n\n"
            f"Probability of Default: {r['pd']*100:.2f}%\n"
            f"Loss Given Default: {r['lgd']*100:.0f}%\n"
            f"Exposure at Default: €{r['ead']:,.2f}\n"
            f"Expected Loss: €{r['expected_loss']:,.2f}\n"
            f"Recommended Interest Rate: {r['interest_rate']*100:.2f}%\n"
            f"Decision: {'DECLINED' if r['declined'] else 'APPROVED'}\n"
        ),
        file_name="loan_quote.txt"
    )

# ============================================================
# TAB 2 — RISK FACTOR INSIGHTS (Interpretability)
# ============================================================
with tab2:
    st.subheader("How Each Risk Factor Affects Default Odds")
    st.caption(
        "Effect of a one-standard-deviation increase in each feature on the odds of default, "
        "derived directly from the Logistic Regression coefficients — full transparency, "
        "as required for regulated lending decisions."
    )

    odds_ratios = np.exp(COEFS)
    factor_df = pd.DataFrame({
        'Feature': FEATURE_NAMES,
        'OddsRatio': odds_ratios
    }).sort_values('OddsRatio', ascending=True)
    factor_df['Effect'] = factor_df['OddsRatio'].apply(lambda x: f"{(x-1)*100:+.1f}%")

    fig2 = go.Figure(go.Bar(
        x=factor_df['OddsRatio'] - 1,
        y=factor_df['Feature'],
        orientation='h',
        marker_color=np.where(factor_df['OddsRatio'] >= 1, '#EF4444', '#22C55E'),
        text=factor_df['Effect'], textposition='outside'
    ))
    fig2.update_layout(
        title="Change in Default Odds per +1 Std. Deviation",
        xaxis_title="Change in Odds of Default", height=420,
        xaxis_tickformat='+.0%', margin=dict(l=10, r=60, t=60, b=20)
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.info(
        "📌 **Reading this chart**: a bar at +80% means a one-standard-deviation increase in that "
        "factor multiplies the odds of default by 1.8x. Negative bars (green) represent "
        "protective factors that lower default risk."
    )

    st.subheader("Model Performance Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC", "0.850")
    c2.metric("Calibration", "Predicted 6.72% vs Actual 6.68%")
    c3.metric("Dominant Driver", "Payment History")

# ============================================================
# TAB 3 — COMPARE BORROWERS
# ============================================================
with tab3:
    st.subheader("Compare Two Borrower Profiles Side by Side")
    colA, colB = st.columns(2)

    with colA:
        st.markdown("**Borrower A**")
        a_util = st.slider("Credit Utilization (A)", 0.0, 2.0, 0.8, 0.01, key="a_util")
        a_age = st.slider("Age (A)", 18, 100, 28, key="a_age")
        a_debt = st.slider("Debt Ratio (A)", 0.0, 5.0, 0.6, 0.01, key="a_debt")
        a_income = st.slider("Monthly Income (A, €)", 0, 20000, 2500, 100, key="a_income")
        a_lines = st.slider("Open Credit Lines (A)", 0, 30, 4, key="a_lines")
        a_re = st.slider("Real Estate Loans (A)", 0, 10, 0, key="a_re")
        a_dep = st.slider("Dependents (A)", 0, 10, 2, key="a_dep")
        a_pastdue = st.slider("Past Due Count (A)", 0, 20, 3, key="a_pastdue")

    with colB:
        st.markdown("**Borrower B**")
        b_util = st.slider("Credit Utilization (B)", 0.0, 2.0, 0.15, 0.01, key="b_util")
        b_age = st.slider("Age (B)", 18, 100, 50, key="b_age")
        b_debt = st.slider("Debt Ratio (B)", 0.0, 5.0, 0.25, 0.01, key="b_debt")
        b_income = st.slider("Monthly Income (B, €)", 0, 20000, 7500, 100, key="b_income")
        b_lines = st.slider("Open Credit Lines (B)", 0, 30, 10, key="b_lines")
        b_re = st.slider("Real Estate Loans (B)", 0, 10, 1, key="b_re")
        b_dep = st.slider("Dependents (B)", 0, 10, 1, key="b_dep")
        b_pastdue = st.slider("Past Due Count (B)", 0, 20, 0, key="b_pastdue")

    ra = compute_pricing(a_util, a_age, a_debt, a_income, a_lines, a_re, a_dep, a_pastdue)
    rb = compute_pricing(b_util, b_age, b_debt, b_income, b_lines, b_re, b_dep, b_pastdue)

    st.divider()
    comp_df = pd.DataFrame({
        'Metric': ['Probability of Default', 'Loss Given Default', 'Exposure at Default (€)',
                   'Expected Loss (€)', 'Interest Rate', 'Decision'],
        'Borrower A': [f"{ra['pd']*100:.2f}%", f"{ra['lgd']*100:.0f}%", f"{ra['ead']:,.0f}",
                       f"{ra['expected_loss']:,.2f}", f"{ra['interest_rate']*100:.2f}%",
                       "DECLINED" if ra['declined'] else "APPROVED"],
        'Borrower B': [f"{rb['pd']*100:.2f}%", f"{rb['lgd']*100:.0f}%", f"{rb['ead']:,.0f}",
                       f"{rb['expected_loss']:,.2f}", f"{rb['interest_rate']*100:.2f}%",
                       "DECLINED" if rb['declined'] else "APPROVED"],
    })
    st.table(comp_df.set_index('Metric'))

    if not ra['declined'] and not rb['declined']:
        diff_pct = (rb['interest_rate'] / ra['interest_rate'] - 1) * 100
        if diff_pct > 0:
            st.success(f"Borrower B pays a **{diff_pct:.1f}% higher** interest rate than Borrower A.")
        else:
            st.success(f"Borrower B pays a **{abs(diff_pct):.1f}% lower** interest rate than Borrower A.")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "Built with Streamlit | Model: Logistic Regression (PD) + Documented LGD/EAD Assumptions "
    "| Benchmark: XGBoost | Data: Give Me Some Credit (Kaggle)"
)

