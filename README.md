# 🏀 NBA Player Injury Risk Dashboard

**Live app:** https://nba-injury-risk-sdgziesinfyzjpy5ruboik.streamlit.app

## Problem
NBA teams lose millions when star players get injured. This tool predicts 
which players are at elevated soft tissue injury risk in the next 14 days — 
before the injury happens.

## Dataset
- NBA game logs pulled via `nba_api` (3 seasons, 20 players, 3,500+ records)
- NBA injury history 1951-2023 (Kaggle)

## Approach
1. Pulled real-time player workload data from the NBA API
2. Engineered sports-science features including the Acute:Chronic Workload 
   Ratio (ACWR) — the same metric used by professional sports scientists
3. Trained XGBoost classifier to predict soft tissue injury in next 14 days
4. Added SHAP explainability to explain individual player risk scores
5. Deployed as a 3-page Streamlit dashboard

## Key Features
- **Acute:Chronic Ratio** — flags players whose recent workload spikes above 
  their long-term average (danger zone: above 1.5)
- **Back-to-back tracking** — counts consecutive game days with no rest
- **SHAP waterfall plots** — explains exactly why each player is flagged

## Results
- ROC-AUC: 0.727 (comparable to professional sports analytics tools)
- Top injury predictor: back-to-back game accumulation

## Tech Stack
Python · XGBoost · SHAP · scikit-learn · NBA API · Streamlit · pandas

## How to Run Locally
pip install -r requirements.txt
streamlit run app.py