# robo_advisor_app_v11.py
# Final version adding the Black-Litterman model with a UI for user views.

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
RISK_AVERSION_FACTORS = {"Conservative": 4.0, "Balanced": 2.5, "Aggressive": 1.0}
MASTER_ASSET_LIST = ["VTI", "VXUS", "BND", "QUAL", "AVUV", "MTUM", "USMV"]

# ... (Questionnaire and Scenarios are unchanged) ...
QUESTIONNAIRE = {
    "Financial Goal": {
        "question": "What is your primary financial goal?",
        "options": ["Capital Preservation", "Generate Income", "Long-Term Growth"],
        "help": "This helps determine the overall strategy. Preservation focuses on low-risk assets, while Growth targets higher-return assets."
    },
    "Investment Horizon": {
        "question": "How long do you plan to invest for?",
        "options": ["Short-term (< 3 years)", "Medium-term (3-7 years)", "Long-term (> 7 years)"],
        "help": "A longer horizon means your portfolio has more time to recover from market downturns, allowing for potentially higher-risk, higher-reward investments."
    },
    "Risk Tolerance": {
        "question": "How would you react if your portfolio suddenly dropped 20%?",
        "options": [
            "Sell all to prevent further loss.",
            "Hold on and wait for it to recover.",
            "Buy more while prices are low.",
        ],
        "help": "This question helps gauge your emotional response to risk. Panic-selling during a downturn is one of the biggest risks to long-term success."
    },
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
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if data.empty: return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex): prices = data['Close']
        else:
            prices = data[['Close']]
            if len(tickers) == 1: prices = prices.rename(columns={'Close': tickers[0]})
        return prices.ffill().dropna(axis=1, how="all")
    except Exception: return pd.DataFrame()

# <<< NEW FEATURE: Function to get market caps for Black-Litterman >>>
@st.cache_data(ttl=dt.timedelta(days=1))
def get_market_caps(tickers: List[str]) -> Dict[str, float]:
    caps = {}
    for t in tickers:
        try:
            caps[t] = yf.Ticker(t).info.get('marketCap', 0)
        except Exception:
            caps[t] = 0 # Default to 0 if API fails
    return caps

# ... (Other data functions are unchanged) ...
@st.cache_data(ttl=dt.timedelta(days=7))
def get_cpi_data(start_date="2010-01-01"):
    try: return web.DataReader("CPIAUCSL", "fred", start_date).pct_change(12) * 100
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

# <<< NEW FEATURE: Black-Litterman Model Implementation >>>
def optimize_black_litterman(returns: pd.DataFrame, risk_profile: str, views: Dict[str, float]) -> pd.Series:
    st.write("---")
    st.info("Running Black-Litterman Optimization...")

    # 1. Market-Implied Equilibrium Returns (pi)
    S = returns.cov() * 252
    market_caps = get_market_caps(list(returns.columns))
    market_weights = pd.Series(market_caps) / sum(market_caps.values())
    risk_aversion = RISK_AVERSION_FACTORS.get(risk_profile, 2.5)
    pi = risk_aversion * S.dot(market_weights)

    # 2. User Views (P, Q, Omega)
    # This is a simplified example where we create views based on the sliders
    view_confidences = []
    q_list = []
    p_rows = []

    # Example View 1: Quality vs Market (VTI)
    if 'quality_view' in views and views['quality_view'] != 0 and 'QUAL' in returns.columns and 'VTI' in returns.columns:
        q_list.append(views['quality_view'] / 100) # Convert percentage to decimal
        p_row = pd.Series(0.0, index=returns.columns)
        p_row['QUAL'] = 1.0
        p_row['VTI'] = -1.0
        p_rows.append(p_row)
        view_confidences.append(0.3) # Example confidence

    # Example View 2: Small-Cap Value vs Market (VTI)
    if 'small_cap_view' in views and views['small_cap_view'] != 0 and 'AVUV' in returns.columns and 'VTI' in returns.columns:
        q_list.append(views['small_cap_view'] / 100)
        p_row = pd.Series(0.0, index=returns.columns)
        p_row['AVUV'] = 1.0
        p_row['VTI'] = -1.0
        p_rows.append(p_row)
        view_confidences.append(0.3) # Example confidence

    # If there are views, build the matrices
    if q_list:
        Q = np.array(q_list)
        P = np.array(p_rows)
        tau = 0.05 # Scaler for uncertainty
        Omega = np.diag(np.diag(P @ (tau * S) @ P.T)) # Simplified uncertainty matrix

        # 3. Posterior Returns (Combined)
        pi_series = pd.Series(pi, index=returns.columns)
        mu_bl = pi_series + (tau * S @ P.T) @ np.linalg.inv(tau * P @ S @ P.T + Omega) @ (Q - P @ pi_series)
        mu = mu_bl.to_numpy()
    else:
        st.warning("No views provided. Black-Litterman is defaulting to market equilibrium returns.")
        mu = pi # If no views, use market equilibrium returns

    # 4. Optimization
    Sigma = S.to_numpy()
    gamma = cp.Parameter(nonneg=True, value=risk_aversion)
    w = cp.Variable(len(mu))
    objective = cp.Maximize(mu @ w - 0.5 * gamma * cp.quad_form(w, Sigma))
    constraints = [cp.sum(w) == 1, w >= 0, w <= 0.35]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)
    
    weights = pd.Series(w.value, index=returns.columns)
    weights[weights < 1e-4] = 0; weights /= weights.sum()
    return weights


# ... (GARCH and standard MVO functions are unchanged) ...
@st.cache_data(ttl=dt.timedelta(hours=12))
def forecast_garch_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    # ... (code unchanged)
    scaled_returns = returns * 100
    forecasted_variances = {}
    for col in scaled_returns.columns:
        model = arch_model(scaled_returns[col].dropna(), p=1, q=1)
        res = model.fit(disp='off', show_warning=False)
        forecasted_variances[col] = res.forecast(horizon=1).variance.iloc[-1, 0]
    corr_matrix = returns.corr()
    std_devs = pd.Series({k: np.sqrt(v / 10000) for k,v in forecasted_variances.items()})
    return corr_matrix.mul(std_devs, axis=0).mul(std_devs, axis=1) * 252

def optimize_mvo(returns: pd.DataFrame, risk_profile: str, use_garch: bool = False) -> pd.Series:
    mu = returns.mean().to_numpy() * 252
    if use_garch: Sigma = forecast_garch_covariance(returns).to_numpy()
    else: Sigma = returns.cov().to_numpy() * 252
    gamma = cp.Parameter(nonneg=True, value=RISK_AVERSION_FACTORS.get(risk_profile, 2.5))
    w = cp.Variable(len(mu))
    objective = cp.Maximize(mu @ w - 0.5 * gamma * cp.quad_form(w, Sigma))
    constraints = [cp.sum(w) == 1, w >= 0, w <= 0.35]
    try:
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)
        if prob.status != cp.OPTIMAL: raise ValueError("Solver failed.")
        weights = pd.Series(w.value, index=returns.columns)
        weights[weights < 1e-4] = 0; weights /= weights.sum()
        return weights
    except Exception as e:
        st.error(f"Optimization failed: {e}. A basic equal-weight portfolio will be used.")
        return pd.Series(1/len(returns.columns), index=returns.columns)
        
# ... (Other helpers are unchanged) ...
def analyze_portfolio(weights, returns): # ...
    portfolio_returns_ts = returns.dot(weights)
    expected_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = portfolio_returns_ts.std() * np.sqrt(252)
    sharpe_ratio = expected_return / portfolio_volatility if portfolio_volatility != 0 else 0
    var_95 = portfolio_returns_ts.quantile(0.05)
    cvar_95 = portfolio_returns_ts[portfolio_returns_ts <= var_95].mean()
    return {"expected_return": expected_return, "expected_volatility": portfolio_volatility, "sharpe_ratio": sharpe_ratio, "value_at_risk_95": var_95, "conditional_value_at_risk_95": cvar_95}

@st.cache_data(ttl=dt.timedelta(hours=12))
def detect_market_regimes(start_date="2010-01-01"): # ...
    try:
        spy_prices_df = get_price_data(["SPY"], start_date=start_date)
        if spy_prices_df.empty or "SPY" not in spy_prices_df.columns: return None
        spy_prices = spy_prices_df["SPY"]
        returns = np.log(spy_prices).diff().dropna()
        if returns.empty: return None
        model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
        model.fit(returns.to_numpy().reshape(-1, 1))
        hidden_states = model.predict(returns.to_numpy().reshape(-1, 1))
        vols = [np.sqrt(model.covars_[i][0][0]) for i in range(model.n_components)]
        high_vol_state = np.argmax(vols)
        regime_df = pd.DataFrame({'regime_label': ['High Volatility' if s == high_vol_state else 'Low Volatility' for s in hidden_states]}, index=returns.index)
        final_df = pd.DataFrame(spy_prices).join(regime_df)
        final_df['regime_label'] = final_df['regime_label'].ffill()
        final_df = final_df.rename(columns={'SPY': 'Close'})
        return final_df
    except Exception: return None

# ... (UI components and main logic will be updated) ...
def display_dashboard(username, portfolio): #...
    # Unchanged
    pass
def run_monte_carlo(initial_value, er, vol, years, simulations): #...
    # Unchanged
    pass
def calculate_efficient_frontier(returns, num_portfolios=2000): #...
    # Unchanged
    pass
def calculate_drawdown(performance_series): #...
    # Unchanged
    pass

# ======================================================================================
# UI COMPONENTS
# ======================================================================================

def display_dashboard(username: str, portfolio: Dict[str, Any]):
    st.subheader(f"Welcome Back, {username.title()}!")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📈 Future Projection", "🔍 Performance Analysis", "🧠 Portfolio Intelligence"])
    weights = pd.Series(portfolio["weights"])

    with tab1:
        # <<< MODIFIED: Added Optimization Model to status display >>>
        profile_cols = st.columns(4)
        profile_cols[0].metric("Risk Profile", portfolio['risk_profile'])
        profile_cols[1].metric("Financial Goal", portfolio.get('profile_answers', {}).get('Financial Goal', 'N/A'))
        profile_cols[2].metric("Optimization Model", portfolio.get('model_choice', 'Mean-Variance'))
        garch_status = "Active (GARCH)" if portfolio.get('used_garch', False) else "Inactive"
        profile_cols[3].metric("🧠 ML Volatility", garch_status)
        
        st.markdown("---")
        # ... (Rest of dashboard unchanged) ...
        metric_cols = st.columns(5)
        metric_cols[0].metric("Expected Annual Return", f"{portfolio['metrics']['expected_return']:.2%}")
        metric_cols[1].metric("Annual Volatility", f"{portfolio['metrics']['expected_volatility']:.2%}")
        # ...

    # ... (Other tabs are the same, so they are omitted for brevity) ...


def display_questionnaire() -> Tuple[str, bool, str, dict, Dict]:
    st.subheader("Complete Your Investor Profile")
    st.write("Your answers to these questions will help us tailor a portfolio that matches your financial situation and comfort with risk.")
    
    # Basic Profile
    answers = {}
    for key, value in QUESTIONNAIRE.items():
        answers[key] = st.radio(f"**{value['question']}**", value['options'])
        st.caption(f"_{value['help']}_")
        st.markdown("---")

    score = sum(QUESTIONNAIRE[key]['options'].index(answers[key]) for key in ["Risk Tolerance", "Investment Horizon"])
    risk_profile = "Conservative" if score <= 1 else "Balanced" if score <= 3 else "Aggressive"
    
    st.markdown("##### Portfolio Construction Method")

    # <<< MODIFIED: Add Black-Litterman to model choices >>>
    model_choice = st.selectbox(
        "Choose your portfolio optimization model:",
        ["Mean-Variance (Standard)", "Black-Litterman (With Your Views)"],
        help="Mean-Variance uses historical data. Black-Litterman blends historical data with your specific views on asset performance."
    )
    
    # <<< MODIFIED: Dynamic UI for Black-Litterman views >>>
    views = {}
    if model_choice == "Black-Litterman (With Your Views)":
        with st.container(border=True):
            st.markdown("##### Express Your Investment Views")
            st.write("Indicate how much you expect certain factors to outperform or underperform the broad US market (VTI).")
            views['quality_view'] = st.slider(
                "Quality (QUAL) vs. Market (VTI) Outperformance (%)",
                -5.0, 5.0, 0.0, 0.5, help="View on high-quality companies."
            )
            views['small_cap_view'] = st.slider(
                "Small-Cap Value (AVUV) vs. Market (VTI) Outperformance (%)",
                -5.0, 5.0, 0.0, 0.5, help="View on small, undervalued companies."
            )

    use_garch = st.toggle("Use ML-Enhanced Volatility Forecast (GARCH)")
    st.caption("_This uses a GARCH model to create a more dynamic forecast of market risk._")
    
    if st.button("📈 Build My Portfolio", type="primary"):
        return risk_profile, use_garch, model_choice, views, answers
    return "", False, "", {}, {}

def run_portfolio_creation(risk_profile: str, use_garch: bool, model_choice: str, views: Dict, profile_answers: Dict) -> Dict | None:
    spinner_msg = f"Building your '{risk_profile}' portfolio using {model_choice}..."
    with st.spinner(spinner_msg):
        prices = get_price_data(MASTER_ASSET_LIST, "2018-01-01")
        if prices.empty: st.error("Could not download market data."); return None
        
        returns = prices.pct_change().dropna()
        if returns.empty: st.error("Could not calculate returns."); return None

        # <<< MODIFIED: Route to the correct optimizer >>>
        if model_choice == "Black-Litterman (With Your Views)":
            weights = optimize_black_litterman(returns, risk_profile, views)
        else: # Default to Mean-Variance
            weights = optimize_mvo(returns, risk_profile, use_garch)
        
        if weights is not None:
            metrics = analyze_portfolio(weights, returns)
            return {
                "risk_profile": risk_profile,
                "weights": {k: v for k, v in weights.items() if v > 0},
                "metrics": metrics,
                "last_rebalanced_date": dt.date.today().isoformat(),
                "profile_answers": profile_answers,
                "used_garch": use_garch if model_choice == "Mean-Variance (Standard)" else False,
                "model_choice": model_choice,
                "views": views
            }
    return None

# ======================================================================================
# MAIN APP FLOW
# ======================================================================================

def main():
    st.title("WealthGenius 🧠 AI-Powered Investment Advisor")
    st.markdown("Welcome! This tool uses advanced financial models to build a portfolio tailored to your insights.")
    st.markdown("---")
    
    if "username" not in st.session_state: st.session_state.username = None
    if "rebalance_request" not in st.session_state: st.session_state.rebalance_request = None

    all_portfolios = load_portfolios()

    if st.session_state.username is None:
        st.subheader("Create or Load Portfolio")
        with st.form("login_form"):
            username_input = st.text_input("Enter your name to begin:")
            submitted = st.form_submit_button("Begin")
            if submitted and username_input: st.session_state.username = username_input; st.rerun()
        st.stop()
    
    username = st.session_state.username
    
    # ... (Rebalancing logic would need to be updated to handle views) ...

    if username not in all_portfolios:
        risk_profile, use_garch, model_choice, views, answers = display_questionnaire()
        if risk_profile:
            new_portfolio = run_portfolio_creation(risk_profile, use_garch, model_choice, views, answers)
            if new_portfolio:
                all_portfolios[username] = new_portfolio
                save_portfolios(all_portfolios)
                st.success("Your portfolio has been created!")
                st.balloons()
                st.rerun()
    else:
        # For simplicity, rebalancing is not fully implemented for Black-Litterman in this example
        # The focus is on the initial creation.
        display_dashboard(username, all_portfolios[username])

if __name__ == "__main__":
    main()
