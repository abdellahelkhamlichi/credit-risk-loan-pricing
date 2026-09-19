# Credit Risk-Based Loan Pricing Engine

## Overview
This project builds a professional credit risk pricing model for consumer loans, using the **Basel-aligned Expected Loss framework** (PD × LGD × EAD), benchmarked against a modern machine learning approach (XGBoost). It demonstrates the full pipeline from raw credit application data to a risk-adjusted loan interest rate, including an interactive pricing dashboard.

## Business Problem
Lenders need to price loans fairly: charging a high enough interest rate to cover expected losses from default, while remaining competitive enough to attract good borrowers. This project answers: **"What interest rate should we charge a given borrower, based on their credit risk profile?"**

## Methodology

1. **Probability of Default (PD)** — Logistic Regression predicting the likelihood a borrower defaults within two years, based on features like income, debt ratio, credit utilization, and payment history.
2. **PD Benchmark** — XGBoost model compared against Logistic Regression, discussing the interpretability vs. accuracy trade-off relevant to regulated credit decisions.
3. **Loss Given Default (LGD)** — The proportion of exposure lost if a borrower defaults, based on documented industry-standard assumptions (Basel framework) for unsecured consumer loans.
4. **Exposure at Default (EAD)** — The amount a lender is exposed to at the point of default.
5. **Expected Loss** — `Expected Loss = PD × LGD × EAD`, the core Basel II/III risk equation used by banks worldwide.
6. **Risk-Based Loan Pricing** — Expected Loss converted into a fair interest rate, adding cost of capital and profit margin, using Net Present Value (NPV) logic.
7. **Interactive Dashboard** — A Streamlit app where a user inputs a borrower profile and receives an instant risk-based interest rate quote.

## Dataset
[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit/data) — a well-known Kaggle credit scoring dataset containing 150,000 borrower records with financial and credit history features, and a binary default indicator (`SeriousDlqin2yrs`).

## Project Structure
```
credit-risk-loan-pricing/
├── data/
│   ├── raw/                  # Original cs-training.csv
│   └── processed/            # Cleaned dataset
├── notebook/
│   └── credit_risk_pricing_project.ipynb   # Full analysis pipeline
├── dashboard/
│   └── app.py                # Streamlit pricing dashboard
├── reports/
│   └── final_report.pdf      # Summary report for non-technical stakeholders
├── requirements.txt
└── README.md
```

## Key Results
*(To be filled in after model training — e.g. PD model performance, expected loss estimates, pricing examples)*

## Tech Stack
- Python (pandas, numpy, scikit-learn, statsmodels)
- XGBoost
- Streamlit (dashboard)
- Matplotlib / Seaborn / Plotly (visualization)

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebook/credit_risk_pricing_project.ipynb
streamlit run dashboard/app.py
```

## Author
Abdellah El Khamlichi — Master's in Finance, Actuarial Science & Data Science

## License
MIT