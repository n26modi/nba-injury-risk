import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title='NBA Injury Risk',
    page_icon='🏀',
    layout='wide'
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {
        background-color: #0e1117;
    }
    [data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2d3748;
    }
    .risk-card-high {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #ef4444;
    }
    .risk-card-medium {
        background: linear-gradient(135deg, #78350f, #92400e);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #f59e0b;
    }
    .risk-card-low {
        background: linear-gradient(135deg, #14532d, #166534);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #22c55e;
    }
    .risk-number {
        font-size: 48px;
        font-weight: 700;
        margin: 8px 0;
        color: white;
    }
    .risk-label {
        font-size: 14px;
        color: #cbd5e1;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .score-high {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        border: 2px solid #ef4444;
    }
    .score-medium {
        background: linear-gradient(135deg, #78350f, #92400e);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        border: 2px solid #f59e0b;
    }
    .score-low {
        background: linear-gradient(135deg, #14532d, #166534);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        border: 2px solid #22c55e;
    }
    .score-number {
        font-size: 64px;
        font-weight: 800;
        color: white;
    }
    .score-text {
        font-size: 18px;
        color: #cbd5e1;
        font-weight: 600;
    }
    .metric-card {
        background-color: #1a1d27;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2d3748;
        text-align: center;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: white;
    }
    .metric-label {
        font-size: 13px;
        color: #94a3b8;
        margin-top: 4px;
    }
    .metric-help {
        font-size: 11px;
        color: #64748b;
        margin-top: 6px;
    }
    .section-header {
        font-size: 22px;
        font-weight: 700;
        color: #f1f5f9;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #2d3748;
    }
    .info-box {
        background-color: #1e3a5f;
        border-left: 4px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 16px 0;
        color: #bfdbfe;
        font-size: 14px;
    }
    .warning-box {
        background-color: #451a03;
        border-left: 4px solid #f59e0b;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 16px 0;
        color: #fde68a;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open('injury_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv('player_gamelogs.csv', parse_dates=['GAME_DATE'])

pipeline = load_model()
df = load_data()

features = [
    'MIN', 'minutes_last_7_days', 'minutes_last_28_days',
    'acute_chronic_ratio', 'days_rest', 'is_back_to_back',
    'back_to_back_count_month', 'PTS', 'REB', 'AST'
]

latest = df.sort_values('GAME_DATE').groupby('player_name').last().reset_index()
X_latest = latest[features].fillna(0)
latest['injury_risk'] = pipeline.predict_proba(X_latest)[:, 1]
latest['risk_pct'] = (latest['injury_risk'] * 100).clip(0, 100).round(1)

def risk_label(score):
    if score >= 60:
        return '🔴 High'
    elif score >= 35:
        return '🟡 Medium'
    else:
        return '🟢 Low'

latest['risk_level'] = latest['risk_pct'].apply(risk_label)

with st.sidebar:
    st.markdown('## 🏀 NBA Injury Risk')
    st.markdown('---')
    st.markdown('''
    **What is this?**  
    Predicts which NBA players are at risk of a soft tissue injury in the next 14 days — based on workload data.
    ''')
    st.markdown('---')
    st.markdown('### Navigate')

    if 'page' not in st.session_state:
        st.session_state.page = 'Home'

    if st.button('🏠  Home', use_container_width=True):
        st.session_state.page = 'Home'
    if st.button('📊  League Dashboard', use_container_width=True):
        st.session_state.page = 'League Dashboard'
    if st.button('🔍  Player Deep Dive', use_container_width=True):
        st.session_state.page = 'Player Deep Dive'
    if st.button('👥  Team View', use_container_width=True):
        st.session_state.page = 'Team View'

    st.markdown('---')
    st.markdown(f'*Currently viewing: **{st.session_state.page}***')

page = st.session_state.page

# ── HOME ──────────────────────────────────────────────────────
if page == 'Home':
    st.markdown('# 🏀 NBA Player Injury Risk Dashboard')
    st.markdown('### Predicting soft tissue injuries before they happen')
    st.markdown('---')
    st.markdown('''
    NBA teams lose millions when star players get injured. This tool uses 
    **machine learning trained on 3 years of NBA workload data** to flag 
    players at elevated injury risk — before the injury happens.
    ''')

    high_risk = len(latest[latest['risk_pct'] >= 60])
    medium_risk = len(latest[(latest['risk_pct'] >= 35) & (latest['risk_pct'] < 60)])
    low_risk = len(latest[latest['risk_pct'] < 35])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'''
        <div class="risk-card-high">
            <div class="risk-label">🔴 High Risk Players</div>
            <div class="risk-number">{high_risk}</div>
        </div>''', unsafe_allow_html=True)
    with col2:
        st.markdown(f'''
        <div class="risk-card-medium">
            <div class="risk-label">🟡 Medium Risk Players</div>
            <div class="risk-number">{medium_risk}</div>
        </div>''', unsafe_allow_html=True)
    with col3:
        st.markdown(f'''
        <div class="risk-card-low">
            <div class="risk-label">🟢 Low Risk Players</div>
            <div class="risk-number">{low_risk}</div>
        </div>''', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('---')

    st.markdown('<div class="section-header">⚠️ Highest Risk Players Right Now</div>', unsafe_allow_html=True)
    top5 = latest.nlargest(5, 'risk_pct')[['player_name', 'risk_pct', 'risk_level']]
    top5.columns = ['Player', 'Injury Risk %', 'Risk Level']
    st.dataframe(top5, use_container_width=True, hide_index=True)

    st.markdown('---')
    st.markdown('<div class="section-header">How does the model work?</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('''
        The model analyzes each player's recent workload and flags players 
        whose bodies are under unusual stress. It was trained on **3,500+ 
        player-game records** and achieves **0.727 AUC** — comparable to 
        professional sports analytics tools.
        ''')
    with col2:
        st.markdown('''
        | Signal | What it means |
        |---|---|
        | **Acute:Chronic Ratio** | Recent vs long-term workload. Above 1.5 = danger |
        | **Back-to-Back Games** | Consecutive games with no rest |
        | **Minutes Spike** | Sudden jump above normal playing time |
        | **Days Rest** | Less rest = more fatigue accumulation |
        ''')

    st.markdown('''
    <div class="info-box">
    👈 Use the sidebar buttons to navigate to the League Dashboard, 
    Player Deep Dive, or Team View
    </div>
    ''', unsafe_allow_html=True)

# ── LEAGUE DASHBOARD ──────────────────────────────────────────
elif page == 'League Dashboard':
    st.markdown('# 📊 League Injury Risk Dashboard')
    st.markdown('All players ranked by current injury risk based on most recent game workload.')
    st.markdown('---')

    filter_risk = st.radio(
        'Filter by risk level:',
        ['All Players', '🔴 High Risk Only', '🟡 Medium Risk Only', '🟢 Low Risk Only'],
        horizontal=True
    )

    display = latest[['player_name', 'risk_pct', 'risk_level',
                      'acute_chronic_ratio', 'back_to_back_count_month']]\
                .sort_values('risk_pct', ascending=False)\
                .reset_index(drop=True)
    display.columns = ['Player', 'Injury Risk %', 'Risk Level',
                       'Acute:Chronic Ratio', 'Back-to-Backs (Month)']

    if 'High' in filter_risk:
        display = display[display['Risk Level'] == '🔴 High']
    elif 'Medium' in filter_risk:
        display = display[display['Risk Level'] == '🟡 Medium']
    elif 'Low' in filter_risk:
        display = display[display['Risk Level'] == '🟢 Low']

    st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown('---')
    st.markdown('<div class="section-header">Risk Score Comparison</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1a1d27')
    colors = ['#ef4444' if x >= 60 else '#f59e0b' if x >= 35 else '#22c55e'
              for x in display['Injury Risk %']]
    ax.barh(display['Player'], display['Injury Risk %'], color=colors)
    ax.axvline(x=60, color='#ef4444', linestyle='--', alpha=0.5, label='High risk (60%)')
    ax.axvline(x=35, color='#f59e0b', linestyle='--', alpha=0.5, label='Medium risk (35%)')
    ax.set_xlabel('Injury Risk %', color='#94a3b8')
    ax.set_title('Player Injury Risk Scores', color='#f1f5f9', pad=15)
    ax.tick_params(colors='#94a3b8')
    ax.spines['bottom'].set_color('#2d3748')
    ax.spines['left'].set_color('#2d3748')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(facecolor='#1a1d27', labelcolor='#94a3b8')
    ax.invert_yaxis()
    st.pyplot(fig)

# ── PLAYER DEEP DIVE ──────────────────────────────────────────
elif page == 'Player Deep Dive':
    st.markdown('# 🔍 Player Deep Dive')
    st.markdown('Select any player to see their injury risk score and a full explanation.')
    st.markdown('---')

    selected_player = st.selectbox('Select a player:', sorted(df['player_name'].unique()))
    player_data = latest[latest['player_name'] == selected_player].iloc[0]
    risk_pct = float(round(player_data['risk_pct'], 1))

    col1, col2 = st.columns([1, 2])

    with col1:
        if risk_pct >= 60:
            st.markdown(f'''
            <div class="score-high">
                <div class="score-text">🔴 HIGH RISK</div>
                <div class="score-number">{risk_pct:.1f}%</div>
                <div class="score-text">injury probability</div>
            </div>''', unsafe_allow_html=True)
        elif risk_pct >= 35:
            st.markdown(f'''
            <div class="score-medium">
                <div class="score-text">🟡 MEDIUM RISK</div>
                <div class="score-number">{risk_pct:.1f}%</div>
                <div class="score-text">injury probability</div>
            </div>''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="score-low">
                <div class="score-text">🟢 LOW RISK</div>
                <div class="score-number">{risk_pct:.1f}%</div>
                <div class="score-text">injury probability</div>
            </div>''', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Key Workload Signals</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)

        acwr = float(round(player_data['acute_chronic_ratio'], 2))
        acwr_delta = '⚠️ Above danger zone' if acwr > 1.5 else '✅ Normal range'

        m1.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{acwr:.2f}</div>
            <div class="metric-label">Acute:Chronic Ratio</div>
            <div class="metric-help">{acwr_delta}</div>
        </div>''', unsafe_allow_html=True)

        b2b = int(player_data['back_to_back_count_month'])
        m2.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{b2b}</div>
            <div class="metric-label">Back-to-Backs (Month)</div>
            <div class="metric-help">Consecutive game days</div>
        </div>''', unsafe_allow_html=True)

        mins = int(round(player_data['minutes_last_7_days'], 0))
        m3.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{mins}</div>
            <div class="metric-label">Minutes Last 7 Days</div>
            <div class="metric-help">Total playing time</div>
        </div>''', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('---')

    st.markdown('<div class="section-header">Why is this player flagged?</div>', unsafe_allow_html=True)
    st.markdown('Each bar shows how much that feature pushed the risk score up (red) or down (blue).')

    xgb_model = pipeline.named_steps['model']
    X_player = player_data[features].values.reshape(1, -1)
    X_transformed = pipeline.named_steps['scaler'].transform(
        pipeline.named_steps['imputer'].transform(X_player)
    )

    explainer = shap.TreeExplainer(xgb_model)
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#0e1117')
    shap.waterfall_plot(explainer(X_transformed)[0], show=False)
    st.pyplot(fig)

    st.markdown('---')
    st.markdown('<div class="section-header">Minutes Played — Last 20 Games</div>', unsafe_allow_html=True)
    st.markdown('Spikes in minutes can indicate elevated injury risk.')
    player_history = df[df['player_name'] == selected_player]\
                        .sort_values('GAME_DATE').tail(20)
    st.line_chart(player_history.set_index('GAME_DATE')['MIN'])

# ── TEAM VIEW ─────────────────────────────────────────────────
elif page == 'Team View':
    st.markdown('# 👥 Team View')
    st.markdown('Compare injury risk across any group of players.')
    st.markdown('---')

    selected_players = st.multiselect(
        'Select players to compare:',
        sorted(df['player_name'].unique()),
        default=sorted(df['player_name'].unique())[:5]
    )

    if selected_players:
        team_view = latest[latest['player_name'].isin(selected_players)]\
                        [['player_name', 'risk_pct', 'risk_level',
                          'acute_chronic_ratio', 'back_to_back_count_month']]\
                        .sort_values('risk_pct', ascending=False)\
                        .reset_index(drop=True)
        team_view.columns = ['Player', 'Risk %', 'Risk Level',
                             'Acute:Chronic Ratio', 'Back-to-Backs (Month)']

        st.dataframe(team_view, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#1a1d27')
        colors = ['#ef4444' if x >= 60 else '#f59e0b' if x >= 35 else '#22c55e'
                  for x in team_view['Risk %']]
        ax.barh(team_view['Player'], team_view['Risk %'], color=colors)
        ax.axvline(x=60, color='#ef4444', linestyle='--', alpha=0.3)
        ax.axvline(x=35, color='#f59e0b', linestyle='--', alpha=0.3)
        ax.set_xlabel('Injury Risk %', color='#94a3b8')
        ax.set_title('Injury Risk Comparison', color='#f1f5f9')
        ax.tick_params(colors='#94a3b8')
        ax.spines['bottom'].set_color('#2d3748')
        ax.spines['left'].set_color('#2d3748')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_yaxis()
        st.pyplot(fig)

        highest = team_view.iloc[0]
        if highest['Risk %'] >= 35:
            st.markdown(f'''
            <div class="warning-box">
            ⚠️ <strong>{highest['Player']}</strong> is your highest risk player 
            at {highest['Risk %']:.1f}% — consider managing their minutes carefully.
            </div>''', unsafe_allow_html=True)
        else:
            st.success('✅ All selected players are currently at low injury risk.')