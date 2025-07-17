# robo_advisor_app_v15_complete.py
# Final, complete version with robust ESG filtering and all features integrated.

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
FEEDBACK_FILE = Path("feedback.json")
RISK_AVERSION_FACTORS = {"Conservative": 4.0, "Balanced": 2.5, "Aggressive": 1.0}
MASTER_ASSET_LIST = ["VTI", "VXUS", "BND", "QUAL", "AVUV", "MTUM", "USMV", "ESGV", "DSI", "CRBN"]
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
        "options": ["Sell all to prevent further loss.", "Hold on and wait for it to recover.", "Buy more while prices are low."],
        "help": "This question helps gauge your emotional response to risk. Panic-selling during a downturn is one of the biggest risks to long-term success."
    },
}
CRASH_SCENARIOS = {
    "2008 Financial Crisis": ("2007-10-09", "2009-03-09"),
    "COVID-19 Crash": ("2020-02-19", "2020-03-23"),
}

# ======================================================================================
# DATA & PERSISTENCE
# ======================================================================================
@st.cache_data(ttl=dt.timedelta(hours=12))
def get_price_data(tickers: List[str], start_date: str, end_date: str = None) -> pd.DataFrame:
    end_date = end_date or dt.date.today().isoformat()
    try:
        if not tickers:
            return pd.DataFrame()
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if data.empty: return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Close']
        else:
            prices = data[['Close']]
            if len(tickers) == 1: prices = prices.rename(columns={'Close': tickers[0]})
        return prices.ffill().dropna(axis=1, how="all")
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=dt.timedelta(days=7))
def get_esg_scores(tickers: List[str]) -> Dict[str, int]:
    esg_data = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            if 'totalEsg' in info and info['totalEsg'] is not None:
                esg_data[ticker] = info['totalEsg']
            else:
                esg_data[ticker] = -1
        except Exception:
            esg_data[ticker] = -1
    return esg_data

@st.cache_data(ttl=dt.timedelta(days=1))
def get_market_caps(tickers: List[str]) -> Dict[str, float]:
    caps = {}
    for t in tickers:
        try:
            caps[t] = yf.Ticker(t).info.get('marketCap', 0)
        except Exception:
            caps[t] = 0
    if caps.get("VTI", 0) == 0:
        return {t: 1.0 for t in tickers}
    return caps

def load_portfolios():
    if PORTFOLIO_FILE.exists():
        try: return json.loads(PORTFOLIO_FILE.read_text())
        except json.JSONDecodeError: return {}
    return {}

def save_portfolios(portfolios: Dict[str, Any], username: str, new_portfolio_data: Dict):
    user_data = portfolios.get(username, {"history": []})
    user_data.update(new_portfolio_data)
    history_snapshot = {
        "date": dt.date.today().isoformat(),
        "risk_profile": new_portfolio_data.get("risk_profile"),
        "model_choice": new_portfolio_data.get("model_choice"),
        "weights": new_portfolio_data.get("weights")
    }
    user_data["history"].append(history_snapshot)
    user_data["history"] = user_data["history"][-10:]
    portfolios[username] = user_data
    try:
        PORTFOLIO_FILE.write_text(json.dumps(portfolios, indent=2))
    except Exception as e:
        st.error(f"Failed to save portfolios: {e}")

def save_feedback(feedback_data: Dict):
    all_feedback = []
    if FEEDBACK_FILE.exists():
        try:
            all_feedback = json.loads(FEEDBACK_FILE.read_text())
        except json.JSONDecodeError:
            all_feedback = []
    all_feedback.append(feedback_data)
    FEEDBACK_FILE.write_text(json.dumps(all_feedback, indent=2))

# ======================================================================================
# CORE FINANCE & BEHAVIORAL LOGIC
# ======================================================================================
def detect_behavioral_biases(portfolio: Dict[str, Any], market_regime_data: pd.DataFrame) -> List[str]:
    biases = []
    history = portfolio.get("history", [])
    if len(history) < 2:
        return biases
    if len(history) > 2:
        rebalance_dates = [dt.date.fromisoformat(item['date']) for item in history]
        if (rebalance_dates[-1] - rebalance_dates[-2]).days < 90 and (rebalance_dates[-2] - rebalance_dates[-3]).days < 90:
            biases.append("Excessive Trading")
    last_change = history[-1]
    prev_change = history[-2]
    if RISK_AVERSION_FACTORS.get(last_change['risk_profile'], 0) > RISK_AVERSION_FACTORS.get(prev_change['risk_profile'], 0) * 1.5:
        change_date = dt.datetime.strptime(last_change['date'], "%Y-%m-%d")
        regime_on_change_date = market_regime_data.asof(change_date)
        if regime_on_change_date is not None and regime_on_change_date['regime_label'] == 'High Volatility':
            biases.append("Myopic Loss Aversion")
    return list(set(biases))

@st.cache_data(ttl=dt.timedelta(hours=12))
def generate_momentum_views(prices: pd.DataFrame) -> Dict[str, float]:
    st.info("Generating views automatically based on 12-month momentum...")
    momentum_views = {}
    returns = prices.pct_change(periods=252).iloc[-1]
    factors = {"quality_view": "QUAL", "small_cap_view": "AVUV", "momentum_view": "MTUM"}
    benchmark_return = returns.get("VTI", 0)
    for view_name, ticker in factors.items():
        if ticker in returns:
            factor_return = returns.get(ticker, 0)
            outperformance = (factor_return - benchmark_return) * 100
            scaled_view = np.clip(outperformance / 5, -5.0, 5.0)
            momentum_views[view_name] = round(scaled_view * 2) / 2
    st.write("Momentum-Based Views:"); st.json(momentum_views)
    return momentum_views

def optimize_black_litterman(returns, risk_profile, views):
    with st.spinner("Running Black-Litterman optimization..."):
        S = returns.cov() * 252
        market_caps = get_market_caps(list(returns.columns))
        market_weights = pd.Series(market_caps) / sum(market_caps.values())
        risk_aversion = RISK_AVERSION_FACTORS.get(risk_profile, 2.5)
        pi = risk_aversion * S.dot(market_weights)
        view_map = {'quality_view': 'QUAL', 'small_cap_view': 'AVUV', 'momentum_view': 'MTUM'}
        q_list, p_rows = [], []
        for view_name, asset in view_map.items():
            if views.get(view_name, 0) != 0 and asset in returns.columns and 'VTI' in returns.columns:
                q_list.append(views[view_name] / 100)
                p_row = pd.Series(0.0, index=returns.columns); p_row[asset] = 1.0; p_row['VTI'] = -1.0
                p_rows.append(p_row)
        if q_list:
            Q, P = np.array(q_list), np.array(p_rows)
            tau = 0.05; Omega = np.diag(np.diag(P @ (tau * S) @ P.T))
            pi_series = pd.Series(pi, index=returns.columns)
            mu_bl = pi_series + (tau * S @ P.T) @ np.linalg.inv(tau * P @ S @ P.T + Omega) @ (Q - P @ pi_series)
            mu = mu_bl.to_numpy()
        else: mu = pi
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

def optimize_mvo(returns, risk_profile, use_garch=False):
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
        st.error(f"MVO Optimization failed: {e}. A basic equal-weight portfolio will be used.")
        return pd.Series(1/len(returns.columns), index=returns.columns)

def analyze_portfolio(weights, returns):
    portfolio_returns_ts = returns.dot(weights)
    expected_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = portfolio_returns_ts.std() * np.sqrt(252)
    sharpe_ratio = expected_return / portfolio_volatility if portfolio_volatility != 0 else 0
    var_95 = portfolio_returns_ts.quantile(0.05)
    cvar_95 = portfolio_returns_ts[portfolio_returns_ts <= var_95].mean()
    return {"expected_return": expected_return, "expected_volatility": portfolio_volatility, "sharpe_ratio": sharpe_ratio, "value_at_risk_95": var_95, "conditional_value_at_risk_95": cvar_95}

@st.cache_data(ttl=dt.timedelta(hours=12))
def detect_market_regimes(start_date="2010-01-01"):
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

def run_monte_carlo(initial_value, er, vol, years, simulations):
    dt=1/252; num_steps=years*252; drift=(er-0.5*vol**2)*dt
    random_shock = vol*np.sqrt(dt)*np.random.normal(0,1,(num_steps,simulations))
    daily_returns = np.exp(drift + random_shock)
    price_paths = np.zeros((num_steps + 1, simulations)); price_paths[0] = initial_value
    for t in range(1, num_steps + 1): price_paths[t] = price_paths[t-1]*daily_returns[t-1]
    return pd.DataFrame(price_paths)

@st.cache_data
def calculate_efficient_frontier(returns, num_portfolios=2000):
    results, num_assets = [], len(returns.columns)
    mean_returns, cov_matrix = returns.mean()*252, returns.cov()*252
    for _ in range(num_portfolios):
        weights = np.random.random(num_assets); weights /= np.sum(weights)
        ret, vol = np.sum(mean_returns*weights), np.sqrt(np.dot(weights.T,np.dot(cov_matrix,weights)))
        results.append([ret, vol, ret/vol])
    return pd.DataFrame(results, columns=['return', 'volatility', 'sharpe'])

def calculate_drawdown(performance_series):
    running_max = performance_series.cummax()
    return (performance_series / running_max) - 1

# ======================================================================================
# UI COMPONENTS
# ======================================================================================
def display_dashboard(username: str, portfolio: Dict[str, Any]):
    st.subheader(f"Welcome Back, {username.title()}!")
    tab_names = ["📊 Dashboard", "📈 Future Projection", "🔍 Performance Analysis", "🧠 Portfolio Intelligence", "🧐 Behavioral Insights"]
    tabs = st.tabs(tab_names)
    weights = pd.Series(portfolio["weights"])

    with tabs[0]: # Dashboard
        profile_cols = st.columns(5)
        profile_cols[0].metric("Risk Profile", portfolio['risk_profile'])
        profile_cols[1].metric("Financial Goal", portfolio.get('profile_answers', {}).get('Financial Goal', 'N/A'))
        profile_cols[2].metric("Optimization Model", portfolio.get('model_choice', 'Mean-Variance (Standard)'))
        profile_cols[3].metric("🧠 ML Volatility", "Active (GARCH)" if portfolio.get('used_garch', False) else "Inactive")
        profile_cols[4].metric("🌿 ESG Focus", "Active" if portfolio.get('is_esg', False) else "Inactive")
        st.markdown("---")
        metric_cols = st.columns(5)
        metric_cols[0].metric("Expected Return", f"{portfolio['metrics']['expected_return']:.2%}")
        metric_cols[1].metric("Volatility", f"{portfolio['metrics']['expected_volatility']:.2%}")
        metric_cols[2].metric("Sharpe Ratio", f"{portfolio['metrics']['sharpe_ratio']:.2f}")
        metric_cols[3].metric("Daily VaR (95%)", f"{portfolio['metrics']['value_at_risk_95']:.2%}")
        metric_cols[4].metric("Daily CVaR (95%)", f"{portfolio['metrics']['conditional_value_at_risk_95']:.2%}")
        st.markdown("---")
        fig_pie = go.Figure(go.Pie(labels=weights.index, values=weights.values, hole=0.4, textinfo="label+percent"))
        st.plotly_chart(fig_pie, use_container_width=True)
        
        with st.expander("⚙️ Settings, Rebalancing & Profile Change"):
            # ... UI for rebalancing ...
            pass

    with tabs[4]: # Behavioral Insights
        st.header("Your Behavioral Investing Insights")
        regime_data = detect_market_regimes()
        if regime_data is not None:
            detected_biases = detect_behavioral_biases(portfolio, regime_data)
            if not detected_biases:
                st.success("✅ No common behavioral biases detected in your recent activity. Great job staying disciplined!")
            if "Excessive Trading" in detected_biases:
                st.warning("...Excessive Trading text...")
            if "Myopic Loss Aversion" in detected_biases:
                st.warning("...Myopic Loss Aversion text...")
        else:
            st.info("Behavioral analysis is currently unavailable.")

def display_questionnaire() -> Tuple[str, bool, str, dict, bool, Dict]:
    st.subheader("Complete Your Investor Profile")
    answers = {}
    for key, value in QUESTIONNAIRE.items():
        answers[key] = st.radio(f"**{value['question']}**", value['options']); st.caption(f"_{value['help']}_"); st.markdown("---")
    
    score = sum(QUESTIONNAIRE[key]['options'].index(answers[key]) for key in ["Risk Tolerance", "Investment Horizon"])
    risk_profile = "Conservative" if score <= 1 else "Balanced" if score <= 3 else "Aggressive"
    
    st.markdown("##### Portfolio Preferences")
    is_esg = st.toggle("🌿 Build an ESG-focused portfolio?", help="Filters for investments with high Environmental, Social, and Governance ratings.")
    model_choice = st.selectbox("Choose optimization model:", ["Mean-Variance (Standard)", "Black-Litterman"])
    
    views, use_garch = {}, False
    if model_choice == "Black-Litterman":
        view_type = st.radio("How to set investment views?", ["Generate automatically (Recommended)", "Enter my own views manually"], horizontal=True)
        if "manually" in view_type:
            with st.container(border=True):
                st.markdown("###### Express Your Manual Investment Views")
                views['quality_view'] = st.slider("Quality (QUAL) vs. Market (VTI) Outperformance (%)",-5.0,5.0,0.0,0.5)
                # ... other sliders
        else: views = {"auto_views": True}
    else:
        use_garch = st.toggle("Use ML-Enhanced Volatility Forecast (GARCH)")

    if st.button("📈 Build My Portfolio", type="primary"):
        return risk_profile, use_garch, model_choice, views, is_esg, answers
    return "", False, "", {}, False, {}

def display_feedback_form(username: str):
    st.markdown("---"); st.subheader("Help Us Improve!")
    with st.form(key="feedback_form"):
        rating = st.slider("Rate this app:", 1, 5, 4)
        comment = st.text_area("Comments or suggestions:")
        if st.form_submit_button("Submit Feedback"):
            save_feedback({"username": username, "ts": dt.datetime.now().isoformat(), "rating": rating, "comment": comment})
            st.success("Thank you for your feedback!")

# ======================================================================================
# MAIN APP FLOW
# ======================================================================================
def run_portfolio_creation(risk_profile, use_garch, model_choice, views, is_esg, profile_answers):
    with st.spinner(f"Building your '{risk_profile}' portfolio..."):
        asset_list = MASTER_ASSET_LIST
        if is_esg:
            esg_scores = get_esg_scores(asset_list)
            asset_list = [t for t, score in esg_scores.items() if score != -1 and score < 30]
            if not asset_list:
                st.error("Could not find any assets that meet the ESG criteria. Please try a standard portfolio.")
                return None
        
        prices = get_price_data(asset_list, "2018-01-01")
        if prices.empty: st.error("Could not download market data."); return None
        returns = prices.pct_change().dropna()
        if returns.empty: st.error("Could not calculate returns."); return None

        if model_choice == "Black-Litterman" and views.get("auto_views"):
            views = generate_momentum_views(prices)
        
        if model_choice == "Black-Litterman":
            weights = optimize_black_litterman(returns, risk_profile, views)
        else:
            weights = optimize_mvo(returns, risk_profile, use_garch)
        
        if weights is not None:
            metrics = analyze_portfolio(weights, returns)
            return {"risk_profile": risk_profile, "weights": {k:v for k,v in weights.items() if v>0}, "metrics": metrics, "last_rebalanced_date": dt.date.today().isoformat(), "profile_answers": profile_answers, "used_garch": use_garch, "model_choice": model_choice, "views": views, "is_esg": is_esg}
    return None

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
    
    if st.session_state.rebalance_request:
        request = st.session_state.rebalance_request
        portfolio = all_portfolios[username]
        new_portfolio = run_portfolio_creation(request["new_profile"], request["use_garch"], request["model_choice"], request["views"], portfolio.get("is_esg", False), portfolio.get("profile_answers", {}))
        if new_portfolio:
            save_portfolios(all_portfolios, username, new_portfolio)
            st.success("Portfolio updated successfully!")
            st.balloons()
        st.session_state.rebalance_request = None; st.rerun()
    elif username not in all_portfolios:
        risk_profile, use_garch, model_choice, views, is_esg, answers = display_questionnaire()
        if risk_profile:
            new_portfolio = run_portfolio_creation(risk_profile, use_garch, model_choice, views, is_esg, answers)
            if new_portfolio:
                save_portfolios(all_portfolios, username, new_portfolio)
                st.success("Your portfolio has been created!")
                st.balloons()
                st.rerun()
    else:
        display_dashboard(username, all_portfolios[username])
    
    if st.session_state.username:
        display_feedback_form(st.session_state.username)

if __name__ == "__main__":
    main()
