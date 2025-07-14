# robo_advisor_app_v3.py
# Stable version with PyPortfolioOpt removed to ensure Python 3.13 compatibility.
# Reverts to original cvxpy optimization, while keeping UI and metric enhancements.

import json
import datetime as dt
from pathlib import Path
from typing import Dict, List, Any, Tuple

from arch import arch_model
import cvxpy as cp
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from hmmlearn import hmm

# ======================================================================================
# CONFIGURATION
# ======================================================================================

st.set_page_config(page_title="WealthGenius | AI Advisor", page_icon="🧠", layout="wide")
PORTFOLIO_FILE = Path("user_portfolios.json")

# <<< NOTE: RISK_AVERSION_FACTORS are now used by the cvxpy optimizer >>>
RISK_AVERSION_FACTORS = {"Conservative": 4.0, "Balanced": 2.5, "Aggressive": 1.0}

# Asset list includes factors for analysis, but optimization is Mean-Variance.
MASTER_ASSET_LIST = [
    "VTI",   # Core: U.S. Total Stock Market
    "VXUS",  # Core: Total International Stock Market (ex-US)
    "BND",   # Core: U.S. Total Bond Market
    "QUAL",  # Factor: Quality
    "AVUV",  # Factor: Small-Cap Value
    "MTUM",  # Factor: Momentum
    "USMV",  # Factor: Minimum Volatility
]

# <<< NOTE: Questionnaire simplified by removing the incompatible model choice >>>
QUESTIONNAIRE = {
    "Financial Goal": ["Capital Preservation", "Generate Income", "Long-Term Growth"],
    "Investment Horizon": ["Short-term (< 3 years)", "Medium-term (3-7 years)", "Long-term (> 7 years)"],
    "Risk Tolerance": [
        "Sell all to prevent further loss if my portfolio drops 20%.",
        "Hold on and wait for it to recover.",
        "Buy more while prices are low.",
    ],
}

CRASH_SCENARIOS = {
    "2008 Financial Crisis": ("2007-10-09", "2009-03-09"),
    "COVID-19 Crash": ("2020-02-19", "2020-03-23"),
    "Dot-Com Bubble Burst": ("2000-03-10", "2002-10-09"),
}

# ======================================================================================
# DATA & PERSISTENCE
# ======================================================================================

@st.cache_data(ttl=dt.timedelta(hours=12))
def get_price_data(tickers: List[str], start_date: str, end_date: str = None) -> pd.DataFrame:
    end_date = end_date or dt.date.today().isoformat()
    try:
        prices = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)["Close"]
        return prices.ffill().dropna(axis=1, how="all") if not prices.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=dt.timedelta(days=7))
def get_cpi_data(start_date="2010-01-01"):
    try:
        return web.DataReader("CPIAUCSL", "fred", start=start_date).pct_change(12) * 100
    except Exception: return None

@st.cache_data(ttl=dt.timedelta(hours=4))
def get_yield_curve_data():
    tickers = {"3M": "^IRX", "5Y": "^FVX", "10Y": "^TNX", "30Y": "^TYX"}
    try:
        yields_raw = yf.Tickers(list(tickers.values())).history(period="5d")['Close'].iloc[-1]
        yield_curve = pd.Series({name: yields_raw[ticker] for name, ticker in tickers.items()})
        return yield_curve.reindex(["3M", "5Y", "10Y", "30Y"])
    except Exception: return None

def load_portfolios():
    if PORTFOLIO_FILE.exists():
        try: return json.loads(PORTFOLIO_FILE.read_text())
        except json.JSONDecodeError: return {}
    return {}

def save_portfolios(portfolios):
    try: PORTFOLIO_FILE.write_text(json.dumps(portfolios, indent=2))
    except Exception as e: st.error(f"Failed to save portfolios: {e}")

# ======================================================================================
# CORE FINANCE & ML LOGIC
# ======================================================================================

def optimize_portfolio(returns: pd.DataFrame, risk_profile: str) -> pd.Series:
    """Performs Mean-Variance Optimization using cvxpy."""
    mu = returns.mean().to_numpy() * 252
    Sigma = returns.cov().to_numpy() * 252
    gamma = cp.Parameter(nonneg=True, value=RISK_AVERSION_FACTORS.get(risk_profile, 2.5))
    w = cp.Variable(len(mu))
    
    objective = cp.Maximize(mu @ w - 0.5 * gamma * cp.quad_form(w, Sigma))
    constraints = [cp.sum(w) == 1, w >= 0, w <= 0.35]
    
    try:
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)
        if prob.status != cp.OPTIMAL:
            raise ValueError("Solver failed to find optimal weights.")
        
        weights = pd.Series(w.value, index=returns.columns)
        weights[weights < 1e-4] = 0
        weights /= weights.sum()
        return weights
    except Exception as e:
        st.error(f"Optimization failed: {e}. A basic equal-weight portfolio will be used.")
        num_assets = len(returns.columns)
        return pd.Series(1/num_assets, index=returns.columns)

def analyze_portfolio(weights: pd.Series, returns: pd.DataFrame) -> Dict[str, float]:
    portfolio_returns_ts = returns.dot(weights)
    expected_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = portfolio_returns_ts.std() * np.sqrt(252)
    sharpe_ratio = expected_return / portfolio_volatility if portfolio_volatility != 0 else 0
    var_95 = portfolio_returns_ts.quantile(0.05)
    cvar_95 = portfolio_returns_ts[portfolio_returns_ts <= var_95].mean()

    return {
        "expected_return": expected_return,
        "expected_volatility": portfolio_volatility,
        "sharpe_ratio": sharpe_ratio,
        "value_at_risk_95": var_95,
        "conditional_value_at_risk_95": cvar_95,
    }

@st.cache_data(ttl=dt.timedelta(hours=12))
def detect_market_regimes(start_date="2010-01-01"):
    try:
        spy_prices = yf.download("SPY", start=start_date, progress=False, auto_adjust=True)['Close']
        returns = np.log(spy_prices).diff().dropna()
        model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
        model.fit(returns.to_numpy().reshape(-1, 1))
        hidden_states = model.predict(returns.to_numpy().reshape(-1, 1))
        vols = [np.sqrt(model.covars_[i][0][0]) for i in range(model.n_components)]
        high_vol_state = np.argmax(vols)
        regime_df = pd.DataFrame({'regime_label': ['High Volatility' if s == high_vol_state else 'Low Volatility' for s in hidden_states]}, index=returns.index)
        final_df = pd.DataFrame(spy_prices).join(regime_df)
        final_df['regime_label'].ffill(inplace=True)
        return final_df
    except Exception: return None
    
# ======================================================================================
# UI & MAIN LOGIC
# ======================================================================================

def display_dashboard(username: str, portfolio: Dict[str, Any]):
    st.subheader(f"Welcome Back, {username.title()}!")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📈 Future Projection", "🔍 Performance Analysis", "🧠 Portfolio Intelligence"])
    weights = pd.Series(portfolio["weights"])

    with tab1:
        last_rebalanced_date = dt.date.fromisoformat(portfolio.get("last_rebalanced_date", "2000-01-01"))
        if (dt.date.today() - last_rebalanced_date).days > 180:
            st.warning("Your portfolio is over 6 months old and may need rebalancing.")

        profile_cols = st.columns(2)
        profile_cols[0].metric("Risk Profile", portfolio['risk_profile'])
        profile_cols[1].metric("Financial Goal", portfolio.get('profile_answers', {}).get('Financial Goal', 'N/A'))
        
        st.markdown("---")
        metric_cols = st.columns(5)
        metric_cols[0].metric("Expected Annual Return", f"{portfolio['metrics']['expected_return']:.2%}")
        metric_cols[1].metric("Annual Volatility", f"{portfolio['metrics']['expected_volatility']:.2%}")
        metric_cols[2].metric("Sharpe Ratio", f"{portfolio['metrics']['sharpe_ratio']:.2f}")
        metric_cols[3].metric("Daily VaR (95%)", f"{portfolio['metrics']['value_at_risk_95']:.2%}")
        metric_cols[4].metric("Daily CVaR (95%)", f"{portfolio['metrics']['conditional_value_at_risk_95']:.2%}", help="The expected loss on days within the worst 5% of scenarios.")
        st.markdown("---")
        
        fig_pie = go.Figure(go.Pie(labels=weights.index, values=weights.values, hole=0.4, marker_colors=px.colors.sequential.GnBu_r, textinfo="label+percent"))
        fig_pie.update_layout(showlegend=False, title_text="Current Portfolio Allocation", title_x=0.5)
        st.plotly_chart(fig_pie, use_container_width=True)

        with st.expander("⚙️ Settings & Rebalancing"):
            if st.button("Manually Rebalance Portfolio Now"):
                st.session_state.rebalance_now = True
                st.rerun()

    with tab4:
        st.header("Portfolio Intelligence & Market Insights")
        st.subheader("Factor Exposure Analysis")
        factor_map = {"QUAL": "Quality", "AVUV": "Value", "MTUM": "Momentum", "USMV": "Min Volatility"}
        core_assets = {"VTI": "US Stocks", "VXUS": "Int'l Stocks", "BND": "Bonds"}
        exposure_weights = {factor_map.get(k, core_assets.get(k, k)): v for k, v in portfolio["weights"].items()}
        if exposure_weights:
            exposure_df = pd.DataFrame.from_dict(exposure_weights, orient='index', columns=['Weight'])
            fig_factor = px.bar(exposure_df, y='Weight', title="Portfolio Exposure to Core Assets and Style Factors", text_auto='.2%')
            st.plotly_chart(fig_factor, use_container_width=True)

def display_questionnaire() -> Tuple[str, Dict]:
    st.subheader("Please Complete Your Investor Profile")
    answers = {key: st.radio(key.replace("_", " "), options) for key, options in QUESTIONNAIRE.items()}
    score = sum(QUESTIONNAIRE[key].index(answers[key]) for key in ["Risk Tolerance", "Investment Horizon"])
    risk_profile = "Conservative" if score <= 1 else "Balanced" if score <= 3 else "Aggressive"
    
    if st.button("📈 Build My Portfolio", type="primary"):
        return risk_profile, answers
    return "", {}

def run_portfolio_creation(risk_profile: str, profile_answers: Dict) -> Dict | None:
    with st.spinner(f"Building your '{risk_profile}' portfolio..."):
        prices = get_price_data(MASTER_ASSET_LIST, "2018-01-01")
        if prices.empty: return None
        
        returns = prices.pct_change().dropna()
        weights = optimize_portfolio(returns, risk_profile)
        
        if weights is not None:
            metrics = analyze_portfolio(weights, returns)
            return {
                "risk_profile": risk_profile,
                "weights": {k: v for k, v in weights.items() if v > 0},
                "metrics": metrics,
                "last_rebalanced_date": dt.date.today().isoformat(),
                "profile_answers": profile_answers
            }
    return None

def main():
    st.title("WealthGenius 🧠 AI-Powered Investment Advisor")
    st.markdown("Welcome! This tool uses **Modern Portfolio Theory (MPT)** to build and analyze a diversified investment portfolio tailored to your unique investor profile.")
    st.markdown("---")
    
    all_portfolios = load_portfolios()
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Create or Load Portfolio")
        with st.form("user_form"):
            username = st.text_input("Enter your name:")
            submitted = st.form_submit_button("Begin")

    if not submitted or not username:
        st.info("Please enter your name to begin.")
        st.stop()
    
    if username not in all_portfolios:
        with col2:
            risk_profile, answers = display_questionnaire()
            if risk_profile:
                new_portfolio = run_portfolio_creation(risk_profile, answers)
                if new_portfolio:
                    all_portfolios[username] = new_portfolio
                    save_portfolios(all_portfolios)
                    st.success("Your portfolio has been created!")
                    st.rerun()
    else:
        st.session_state.username = username
        display_dashboard(username, all_portfolios[username])

    if st.session_state.get('rebalance_now') and 'username' in st.session_state:
        username = st.session_state.username
        portfolio = all_portfolios[username]
        new_portfolio = run_portfolio_creation(portfolio['risk_profile'], portfolio['profile_answers'])
        if new_portfolio:
            all_portfolios[username] = new_portfolio
            save_portfolios(all_portfolios)
            st.success("Portfolio has been rebalanced!")
        st.session_state.rebalance_now = False
        st.rerun()

if __name__ == "__main__":
    main()
