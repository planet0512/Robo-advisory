# robo_advisor_app_v19_complete.py
# Final version restoring the HMM chart and placing the feedback form in an expander.

import json
import datetime as dt
from pathlib import Path
from typing import Dict, List, Any, Tuple
import requests

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
ESG_ASSET_UNIVERSE = ["ESGV", "DSI", "CRBN", "BND"]

QUESTIONNAIRE = {
    "Financial Goal": {
        "question": "What is your primary financial goal?", "options": ["Capital Preservation", "Generate Income", "Long-Term Growth"],
        "help": "This helps determine the overall strategy..."
    },
    "Investment Horizon": {
        "question": "How long do you plan to invest for?", "options": ["Short-term (< 3 years)", "Medium-term (3-7 years)", "Long-term (> 7 years)"],
        "help": "A longer horizon means your portfolio has more time to recover..."
    },
    "Risk Tolerance": {
        "question": "How would you react if your portfolio suddenly dropped 20%?", "options": ["Sell all to prevent further loss.", "Hold on and wait for it to recover.", "Buy more while prices are low."],
        "help": "This question helps gauge your emotional response to risk..."
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
        if not tickers: return pd.DataFrame()
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if data.empty: return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex): prices = data['Close']
        else:
            prices = data[['Close']]
            if len(tickers) == 1: prices = prices.rename(columns={'Close': tickers[0]})
        return prices.ffill().dropna(axis=1, how="all")
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=dt.timedelta(minutes=30))
def get_fear_greed_index() -> Dict[str, Any]:
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1")
        r.raise_for_status(); data = r.json()['data'][0]
        return {"value": int(data['value']), "classification": data['value_classification']}
    except Exception: return None

@st.cache_data(ttl=dt.timedelta(minutes=30))
def get_vix_data() -> pd.DataFrame:
    try:
        end_date = dt.date.today()
        start_date = end_date - dt.timedelta(days=90)
        vix_data = yf.download("^VIX", start=start_date.isoformat(), end=end_date.isoformat(), progress=False, auto_adjust=True)
        return vix_data
    except Exception: return None

@st.cache_data(ttl=dt.timedelta(days=7))
def get_esg_scores(tickers: List[str]) -> Dict[str, int]:
    esg_data = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            if 'totalEsg' in info and info['totalEsg'] is not None: esg_data[ticker] = info['totalEsg']
            else: esg_data[ticker] = -1
        except Exception: esg_data[ticker] = -1
    return esg_data

@st.cache_data(ttl=dt.timedelta(days=1))
def get_market_caps(tickers: List[str]) -> Dict[str, float]:
    caps = {}
    for t in tickers:
        try: caps[t] = yf.Ticker(t).info.get('marketCap', 0)
        except Exception: caps[t] = 0
    if caps.get("VTI", 0) == 0 and len(tickers) > 0: return {t: 1.0 for t in tickers}
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
        "date": dt.date.today().isoformat(), "risk_profile": new_portfolio_data.get("risk_profile"),
        "model_choice": new_portfolio_data.get("model_choice"), "weights": new_portfolio_data.get("weights")
    }
    user_data["history"].append(history_snapshot)
    user_data["history"] = user_data["history"][-10:]
    portfolios[username] = user_data
    try: PORTFOLIO_FILE.write_text(json.dumps(portfolios, indent=2))
    except Exception as e: st.error(f"Failed to save portfolios: {e}")

def save_feedback(feedback_data: Dict):
    all_feedback = []
    if FEEDBACK_FILE.exists():
        try: all_feedback = json.loads(FEEDBACK_FILE.read_text())
        except json.JSONDecodeError: all_feedback = []
    all_feedback.append(feedback_data)
    FEEDBACK_FILE.write_text(json.dumps(all_feedback, indent=2))

# ======================================================================================
# CORE FINANCE & BEHAVIORAL LOGIC
# ======================================================================================
def detect_behavioral_biases(portfolio: Dict[str, Any], market_regime_data: pd.DataFrame) -> List[str]:
    biases, history = [], portfolio.get("history", [])
    if len(history) < 2: return biases
    if len(history) > 2:
        rebalance_dates = [dt.date.fromisoformat(item['date']) for item in history]
        if (rebalance_dates[-1] - rebalance_dates[-2]).days < 90 and (rebalance_dates[-2] - rebalance_dates[-3]).days < 90:
            biases.append("Excessive Trading")
    last_change, prev_change = history[-1], history[-2]
    if RISK_AVERSION_FACTORS.get(last_change['risk_profile'],0) > RISK_AVERSION_FACTORS.get(prev_change['risk_profile'],0) * 1.5:
        change_date = dt.datetime.strptime(last_change['date'], "%Y-%m-%d")
        regime_on_change_date = market_regime_data.asof(change_date)
        if regime_on_change_date is not None and regime_on_change_date['regime_label'] == 'High Volatility':
            biases.append("Myopic Loss Aversion")
    return list(set(biases))

@st.cache_data(ttl=dt.timedelta(hours=12))
def generate_momentum_views(prices: pd.DataFrame) -> Dict[str, float]:
    st.info("Generating views automatically based on 12-month momentum...")
    momentum_views, returns = {}, prices.pct_change(periods=252).iloc[-1]
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

@st.cache_data(ttl=dt.timedelta(hours=12))
def forecast_garch_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    scaled_returns = returns * 100
    forecasted_variances = {}
    for col in scaled_returns.columns:
        model = arch_model(scaled_returns[col].dropna(), p=1, q=1, vol='Garch', dist='Normal')
        res = model.fit(disp='off', show_warning=False)
        forecast = res.forecast(horizon=1)
        forecasted_variances[col] = forecast.variance.iloc[-1, 0]
    corr_matrix = returns.corr()
    std_devs = pd.Series({k: np.sqrt(v / 10000) for k, v in forecasted_variances.items()})
    garch_cov = corr_matrix.mul(std_devs, axis=0).mul(std_devs, axis=1) * 252
    return garch_cov

def optimize_black_litterman(returns, risk_profile, views):
    with st.spinner("Running Black-Litterman optimization..."):
        S = returns.cov() * 252; market_caps = get_market_caps(list(returns.columns))
        market_weights = pd.Series(market_caps) / sum(market_caps.values())
        risk_aversion = RISK_AVERSION_FACTORS.get(risk_profile, 2.5); pi = risk_aversion * S.dot(market_weights)
        view_map = {'quality_view': 'QUAL', 'small_cap_view': 'AVUV', 'momentum_view': 'MTUM'}
        q_list, p_rows = [], []
        for view_name, asset in view_map.items():
            if views.get(view_name, 0) != 0 and asset in returns.columns and 'VTI' in returns.columns:
                q_list.append(views[view_name] / 100)
                p_row = pd.Series(0.0, index=returns.columns); p_row[asset] = 1.0; p_row['VTI'] = -1.0
                p_rows.append(p_row)
        if q_list:
            Q, P = np.array(q_list), np.array(p_rows); tau = 0.05
            Omega = np.diag(np.diag(P @ (tau * S) @ P.T))
            pi_series = pd.Series(pi, index=returns.columns)
            mu_bl = pi_series + (tau * S @ P.T) @ np.linalg.inv(tau * P @ S @ P.T + Omega) @ (Q - P @ pi_series)
            mu = mu_bl.to_numpy()
        else: mu = pi
        Sigma = S.to_numpy(); gamma = cp.Parameter(nonneg=True, value=risk_aversion)
        w = cp.Variable(len(mu))
        objective = cp.Maximize(mu @ w - 0.5 * gamma * cp.quad_form(w, Sigma))
        constraints = [cp.sum(w) == 1, w >= 0, w <= 0.35]
        prob = cp.Problem(objective, constraints); prob.solve(solver=cp.SCS)
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
        prob = cp.Problem(objective, constraints); prob.solve(solver=cp.SCS)
        if prob.status != cp.OPTIMAL: raise ValueError("Solver failed.")
        weights = pd.Series(w.value, index=returns.columns)
        weights[weights < 1e-4] = 0; weights /= weights.sum()
        return weights
    except Exception as e:
        st.error(f"MVO Optimization failed: {e}. A basic equal-weight portfolio will be used.")
        return pd.Series(1/len(returns.columns), index=returns.columns)

def analyze_portfolio(weights, returns):
    portfolio_returns_ts = returns.dot(weights)
    expected_return = np.sum(returns.mean()*weights)*252
    portfolio_volatility = portfolio_returns_ts.std()*np.sqrt(252)
    sharpe_ratio = expected_return / portfolio_volatility if portfolio_volatility != 0 else 0
    var_95 = portfolio_returns_ts.quantile(0.05)
    cvar_95 = portfolio_returns_ts[portfolio_returns_ts <= var_95].mean()
    return {"expected_return":expected_return, "expected_volatility":portfolio_volatility, "sharpe_ratio":sharpe_ratio, "value_at_risk_95":var_95, "conditional_value_at_risk_95": cvar_95}

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
        results.append([ret, vol, ret/vol if vol != 0 else 0])
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
            st.write("Update your portfolio settings and rebalance to the latest market data.")
            current_profile_index = list(RISK_AVERSION_FACTORS.keys()).index(portfolio['risk_profile'])
            new_profile = st.selectbox("Change risk profile:", list(RISK_AVERSION_FACTORS.keys()), index=current_profile_index, key="rebal_profile")
            model_options = ["Mean-Variance (Standard)", "Black-Litterman"]
            current_model_index = model_options.index(portfolio.get('model_choice', 'Mean-Variance (Standard)'))
            model_choice_rebal = st.selectbox("Change optimization model:", model_options, index=current_model_index, key="rebal_model")
            views_rebal = portfolio.get('views', {})
            use_garch_rebalance = portfolio.get('used_garch', False)
            if model_choice_rebal == "Black-Litterman":
                view_type_rebal = st.radio("Update your investment views:", ["Use automatically generated momentum views", "Update my manual views"], horizontal=True, key="rebal_view_type")
                if "manual" in view_type_rebal:
                    with st.container(border=True):
                        views_rebal['quality_view'] = st.slider("Quality (QUAL) vs. Market (VTI) Outperformance (%)", -5.0, 5.0, views_rebal.get('quality_view', 0.0), 0.5, key="rebal_qual")
                        views_rebal['small_cap_view'] = st.slider("Small-Cap Value (AVUV) vs. Market (VTI) Outperformance (%)", -5.0, 5.0, views_rebal.get('small_cap_view', 0.0), 0.5, key="rebal_scv")
                        views_rebal['momentum_view'] = st.slider("Momentum (MTUM) vs. Market (VTI) Outperformance (%)", -5.0, 5.0, views_rebal.get('momentum_view', 0.0), 0.5, key="rebal_mom")
                else: views_rebal = {"auto_views": True}
            else:
                use_garch_rebalance = st.toggle("Use ML-Enhanced Volatility Forecast (GARCH)", value=use_garch_rebalance, key="rebal_garch")
            if st.button("Update and Rebalance Portfolio", type="primary"):
                st.session_state.rebalance_request = {"new_profile": new_profile, "use_garch": use_garch_rebalance, "model_choice": model_choice_rebal, "views": views_rebal}
                st.rerun()

    with tabs[1]:
        st.header("Future Growth Simulation (Monte Carlo)")
        sim_cols = st.columns([1, 3])
        with sim_cols[0]:
            initial_investment = st.number_input("Initial Investment ($)", 1000, 1000000, 10000, 1000, key="mc_invest")
            simulation_years = st.slider("Investment Horizon (Years)", 1, 40, 10, key="mc_years")
        sim_results = run_monte_carlo(initial_investment, portfolio['metrics']['expected_return'], portfolio['metrics']['expected_volatility'], simulation_years, 500)
        with sim_cols[1]:
            fig_sim = go.Figure()
            fig_sim.add_traces([go.Scatter(x=sim_results.index/252, y=sim_results[col], line_color='lightgrey', showlegend=False) for col in sim_results.columns[:100]])
            quantiles = sim_results.quantile([0.1, 0.5, 0.9], axis=1).T
            for q_val, q_name in zip([0.1, 0.5, 0.9], ["10th Percentile", "Median", "90th Percentile"]):
                 fig_sim.add_trace(go.Scatter(x=sim_results.index/252, y=quantiles[q_val], line=dict(width=3), name=q_name))
            st.plotly_chart(fig_sim, use_container_width=True)
        st.markdown("---")
        final_values = sim_results.iloc[-1]
        pessimistic, median, optimistic = final_values.quantile(0.1), final_values.quantile(0.5), final_values.quantile(0.9)
        st.subheader(f"Projected Outcomes after {simulation_years} Years")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Pessimistic Outcome (10%)", f"${pessimistic:,.2f}")
        metric_cols[1].metric("Median Outcome (50%)", f"${median:,.2f}")
        metric_cols[2].metric("Optimistic Outcome (90%)", f"${optimistic:,.2f}")

    with tabs[2]:
        st.header("Performance & Risk Analysis")
        all_prices = get_price_data(list(weights.index) + ["SPY"], "2018-01-01")
        if not all_prices.empty and not all_prices.isnull().all().all():
            valid_assets = [col for col in weights.index if col in all_prices.columns and not all_prices[col].isnull().all()]
            returns = all_prices[valid_assets].pct_change().dropna()
            if not returns.empty:
                st.subheader("Historical Performance Backtest")
                aligned_weights = weights[valid_assets].copy(); aligned_weights /= aligned_weights.sum()
                portfolio_performance = (1 + returns.dot(aligned_weights)).cumprod()
                spy_performance = (1 + all_prices["SPY"].pct_change().dropna()).cumprod()
                fig_backtest = go.Figure()
                fig_backtest.add_trace(go.Scatter(x=portfolio_performance.index, y=portfolio_performance, name='Your Portfolio'))
                fig_backtest.add_trace(go.Scatter(x=spy_performance.index, y=spy_performance, name='S&P 500 (SPY)', line=dict(dash='dash')))
                st.plotly_chart(fig_backtest, use_container_width=True)
                st.subheader("Sharpe Ratio Comparison")
                asset_returns = returns.mean() * 252
                asset_std_dev = returns.std() * np.sqrt(252)
                individual_sharpes = (asset_returns / asset_std_dev).dropna()
                sharpe_ratios_df = pd.DataFrame(individual_sharpes, columns=['Sharpe Ratio'])
                sharpe_ratios_df.loc['Your Portfolio'] = portfolio['metrics']['sharpe_ratio']
                st.bar_chart(sharpe_ratios_df)
                st.subheader("Efficient Frontier")
                frontier_df = calculate_efficient_frontier(returns)
                fig_frontier = px.scatter(frontier_df, x='volatility', y='return', color='sharpe', title='Efficient Frontier & Your Portfolio')
                fig_frontier.add_trace(go.Scatter(x=[portfolio['metrics']['expected_volatility']], y=[portfolio['metrics']['expected_return']], mode='markers', marker=dict(color='red', size=15, symbol='star'), name='Your Portfolio'))
                st.plotly_chart(fig_frontier, use_container_width=True)
        else: st.warning("Could not retrieve sufficient historical data for the Performance Analysis tab.")
    
    with tabs[3]:
        # <<< NEW FEATURE: Six-Month Review Trigger >>>
        st.subheader("💡 Six-Month Review Trigger")
        last_rebalanced_date = dt.date.fromisoformat(portfolio.get("last_rebalanced_date", "2000-01-01"))
        days_since_last_review = (dt.date.today() - last_rebalanced_date).days

        if days_since_last_review > 180:
            st.warning(
                f"**Time for Your Six-Month Review!**\n\n"
                f"It's been **{days_since_last_review} days** since you last updated your portfolio. Financial advisors recommend reviewing your plan at least every six months or after major life events.\n\n"
                "**Have any of the following occurred recently?**\n"
                "- New Job or Change in Income\n"
                "- Marriage or Change in Family Status\n"
                "- Birth of a Child\n"
                "- Purchasing a Home\n\n"
                "If so, your financial goals may have changed. Consider updating your profile in the 'Settings' section on the **📊 Dashboard** tab."
            )
        else:
            next_review_in = 180 - days_since_last_review
            st.success(
                f"✅ **Your financial plan is on track.**\n\n"
                f"Your last review was {days_since_last_review} days ago. Your next scheduled check-in is in approximately **{next_review_in // 30} months**."
            )
        st.markdown("---")
        st.header("Portfolio Intelligence & Market Insights")
        st.subheader("Market Sentiment Indicators")
        sentiment_cols = st.columns(2)
        with sentiment_cols[0]:
            st.markdown("##### Fear & Greed Index")
            fng_data = get_fear_greed_index()
            if fng_data:
                st.metric(label=fng_data['classification'], value=fng_data['value'])
                st.progress(fng_data['value'] / 100)
                st.caption("Source: alternative.me")
            else: st.warning("Could not retrieve Fear & Greed data.")
        with sentiment_cols[1]:
            st.markdown("##### VIX (Volatility Index)")
            vix_data = get_vix_data()
            if vix_data is not None and not vix_data.empty:
                current_vix = vix_data['Close'].iloc[-1]
                delta_vix = current_vix - vix_data['Close'].iloc[-2]
                st.metric(label="Current VIX Level", value=f"{float(current_vix):.2f}", delta=f"{float(delta_vix):.2f}")
                st.line_chart(vix_data['Close'], height=120)
                st.caption("Measures the market's expectation of 30-day volatility.")
            else: st.warning("Could not retrieve VIX data.")
        st.markdown("---")
        st.subheader("Historical Stress Testing")
        for name, (start, end) in CRASH_SCENARIOS.items():
            st.markdown(f"#### {name} (`{start}` to `{end}`)")
            all_assets_for_period = list(weights.index) + ["SPY"]
            crisis_prices = get_price_data(all_assets_for_period, start, end)
            if crisis_prices.empty or "SPY" not in crisis_prices.columns or crisis_prices['SPY'].isnull().all():
                st.warning(f"Could not retrieve valid market data for the {name} period."); st.markdown("---"); continue
            available_portfolio_assets = [t for t in weights.index if t in crisis_prices.columns and not crisis_prices[t].isnull().all()]
            if not available_portfolio_assets:
                st.warning(f"None of your portfolio's assets existed during the {name} period."); st.markdown("---"); continue
            aligned_weights = weights[available_portfolio_assets].copy(); aligned_weights /= aligned_weights.sum()
            portfolio_returns = crisis_prices[available_portfolio_assets].pct_change().dot(aligned_weights)
            portfolio_cumulative = (1 + portfolio_returns).cumprod()
            spy_returns = crisis_prices['SPY'].pct_change()
            spy_cumulative = (1 + spy_returns).cumprod()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Your Portfolio Total Return", f"{portfolio_cumulative.iloc[-1] - 1:.2%}")
                st.metric("Your Portfolio Max Drawdown", f"{calculate_drawdown(portfolio_cumulative).min():.2%}")
            with col2:
                st.metric("S&P 500 Total Return", f"{spy_cumulative.iloc[-1] - 1:.2%}")
                st.metric("S&P 500 Max Drawdown", f"{calculate_drawdown(spy_cumulative).min():.2%}")
            st.markdown("---")
        st.subheader("Machine Learning: Market Regime Detection")
        regime_data = detect_market_regimes()
        if regime_data is not None:
            current_regime = regime_data['regime_label'].iloc[-1]
            st.info(f"The HMM model indicates the market is currently in a **{current_regime}** state.")
            fig_regime = go.Figure()
            fig_regime.add_trace(go.Scatter(x=regime_data.index, y=regime_data['Close'], mode='lines', name='SPY Price', line_color='black', showlegend=False))
            colors = {'Low Volatility': 'rgba(0, 176, 246, 0.2)', 'High Volatility': 'rgba(255, 82, 82, 0.2)'}
            for state in colors:
                fig_regime.add_trace(go.Bar(name=f'{state} Period', x=[None], y=[None], marker_color=colors[state]))
                for _, g in regime_data[regime_data['regime_label'] == state].groupby((regime_data['regime_label'] != regime_data['regime_label'].shift()).cumsum()):
                    fig_regime.add_vrect(x0=g.index.min(), x1=g.index.max(), fillcolor=colors[state], line_width=0, annotation_text=None)
            st.plotly_chart(fig_regime, use_container_width=True)
        else: st.warning("Market regime analysis is currently unavailable.")

    with tabs[4]:
        st.header("Your Behavioral Investing Insights")
        regime_data = detect_market_regimes()
        if regime_data is not None:
            detected_biases = detect_behavioral_biases(portfolio, regime_data)
            if not detected_biases:
                st.success("✅ No common behavioral biases detected in your recent activity. Great job staying disciplined!")
            if "Excessive Trading" in detected_biases:
                st.warning("Potential Bias Detected: Overconfidence / Excessive Trading")
                st.info("Observation: You have rebalanced your portfolio multiple times in a short period...")
            if "Myopic Loss Aversion" in detected_biases:
                st.warning("Potential Bias Detected: Myopic Loss Aversion")
                st.info("Observation: You significantly lowered your risk profile during a recent period of high market volatility...")
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
                views['small_cap_view'] = st.slider("Small-Cap Value (AVUV) vs. Market (VTI) Outperformance (%)", -5.0, 5.0, 0.0, 0.5)
                views['momentum_view'] = st.slider("Momentum (MTUM) vs. Market (VTI) Outperformance (%)", -5.0, 5.0, 0.0, 0.5)
        else: views = {"auto_views": True}
    else:
        use_garch = st.toggle("Use ML-Enhanced Volatility Forecast (GARCH)")

    if st.button("📈 Build My Portfolio", type="primary"):
        return risk_profile, use_garch, model_choice, views, is_esg, answers
    return "", False, "", {}, False, {}

def display_feedback_form(username: str):
    st.markdown("---")
    with st.expander("✍️ Help Us Improve!"):
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
        asset_list = ESG_ASSET_UNIVERSE if is_esg else MASTER_ASSET_LIST
        if is_esg: st.info(f"Constructing portfolio from ESG universe: {asset_list}")

        prices = get_price_data(asset_list, "2018-01-01")
        if prices.empty: 
            st.error("Could not download market data for the selected assets.")
            return None
        
        returns = prices.pct_change().dropna()
        if returns.empty: st.error("Could not calculate returns from market data."); return None

        if model_choice == "Black-Litterman":
            if views.get("auto_views"):
                views = generate_momentum_views(prices)
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
        request, portfolio = st.session_state.rebalance_request, all_portfolios[username]
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
