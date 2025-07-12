# robo_advisor_app.py
# Final showcase version with enhanced user profiling, VaR, economic data, and UI polish.

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

# ======================================================================================
# CONFIGURATION
# ======================================================================================

st.set_page_config(
    page_title="WealthFlow | AI Advisor",
    page_icon="🤖",
    layout="wide",
)

PORTFOLIO_FILE = Path("user_portfolios.json")

RISK_AVERSION_FACTORS = {"Conservative": 4.0, "Balanced": 2.5, "Aggressive": 1.0}

MASTER_ASSET_LIST = ["VTI", "VEA", "VWO", "BND", "VNQ", "TIP"]

# --- <<< FEATURE: ENHANCED USER PROFILING >>> ---
QUESTIONNAIRE = {
    "Financial Goal": ["Capital Preservation", "Generate Income", "Long-Term Growth"],
    "Investment Horizon": ["Short-term (< 3 years)", "Medium-term (3-7 years)", "Long-term (> 7 years)"],
    "Risk Tolerance": [
        "Sell all to prevent further loss if my portfolio drops 20%.",
        "Hold on and wait for it to recover.",
        "Buy more while prices are low.",
    ],
    "Annual Income": ["<$50,000", "$50,000 - $150,000", ">$150,000"],
}

# ======================================================================================
# DATA & PERSISTENCE
# ======================================================================================

@st.cache_data(ttl=dt.timedelta(hours=12))
def get_price_data(tickers: List[str], start_date: str = "2018-01-01") -> pd.DataFrame:
    try:
        prices = yf.download(tickers, start=start_date, progress=False)["Close"].dropna(axis=1, how="all")
        return prices.ffill().dropna() if not prices.empty else pd.DataFrame()
    except Exception: return pd.DataFrame()

# --- <<< FEATURE: MARKET ANALYSIS (CPI DATA) >>> ---
@st.cache_data(ttl=dt.timedelta(days=7))
def get_cpi_data(start_date="2010-01-01"):
    try:
        cpi = web.DataReader("CPIAUCSL", "fred", start=start_date)
        return cpi.pct_change(12) * 100 # Year-over-year inflation rate
    except Exception: return None

def load_portfolios() -> Dict[str, Any]:
    if PORTFOLIO_FILE.exists():
        try: return json.loads(PORTFOLIO_FILE.read_text())
        except json.JSONDecodeError: return {}
    return {}

def save_portfolios(portfolios: Dict[str, Any]):
    try: PORTFOLIO_FILE.write_text(json.dumps(portfolios, indent=2))
    except Exception as e: st.error(f"Failed to save portfolios: {e}")

# ======================================================================================
# CORE FINANCE & ML LOGIC
# ======================================================================================

def forecast_covariance_garch(returns: pd.DataFrame) -> pd.DataFrame:
    # ... (function is unchanged)
    std_returns = returns * 100
    residuals, variances = [], []
    for asset in std_returns.columns:
        model = arch_model(std_returns[asset], p=1, q=1, vol='Garch', dist='Normal')
        res = model.fit(disp="off")
        residuals.append(res.std_resid)
        forecast = res.forecast(horizon=1)
        variances.append(forecast.variance.iloc[-1, 0])
    corr_matrix = pd.concat(residuals, axis=1).corr()
    variances = np.array(variances) / (100**2)
    diag_vol = np.diag(np.sqrt(variances))
    cov_matrix = diag_vol @ corr_matrix @ diag_vol
    return pd.DataFrame(cov_matrix * 252, index=returns.columns, columns=returns.columns)

def optimize_portfolio(returns: pd.DataFrame, risk_profile: str, use_garch: bool = False) -> pd.Series:
    # ... (function is unchanged)
    mu = returns.mean().to_numpy() * 252
    if use_garch:
        st.toast("Using GARCH model for risk forecast...", icon="🧠")
        Sigma = forecast_covariance_garch(returns).to_numpy()
    else:
        st.toast("Using historical model for risk...", icon="📜")
        Sigma = returns.cov().to_numpy() * 252
    mu, Sigma = np.nan_to_num(mu), np.nan_to_num(Sigma)
    try:
        gamma = cp.Parameter(nonneg=True)
        gamma.value = RISK_AVERSION_FACTORS[risk_profile]
        Sigma = 0.5 * (Sigma + Sigma.T)
        P = cp.psd_wrap(Sigma)
        w = cp.Variable(len(mu))
        risk = cp.quad_form(w, P)
        constraints = [cp.sum(w) == 1, w >= 0, w <= 0.35]
        prob = cp.Problem(cp.Maximize(mu @ w - 0.5 * gamma * risk), constraints)
        prob.solve(solver=cp.SCS)
        if prob.status != cp.OPTIMAL: raise ValueError("Solver could not find an optimal solution.")
        weights = pd.Series(w.value, index=returns.columns)
        weights[weights < 1e-4] = 0
        weights /= weights.sum()
        return weights
    except Exception as e:
        st.error(f"Optimization failed: {e}")
        return None

def analyze_portfolio(weights: pd.Series, returns: pd.DataFrame) -> Dict[str, float]:
    # ... (function is unchanged)
    portfolio_return = np.sum(returns.mean() * 252 * weights)
    cov_matrix = returns.cov() * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility
    # --- <<< FEATURE: RISK ASSESSMENT (VALUE AT RISK) >>> ---
    portfolio_daily_returns = returns.dot(weights)
    var_95 = portfolio_daily_returns.quantile(0.05)
    return {"expected_return": portfolio_return, "expected_volatility": portfolio_volatility, "sharpe_ratio": sharpe_ratio, "value_at_risk_95": var_95}

def run_monte_carlo(initial_value: float, er: float, vol: float, years: int, simulations: int) -> pd.DataFrame:
    # ... (function is unchanged)
    dt = 1 / 252
    num_steps = years * 252
    drift = (er - 0.5 * vol**2) * dt
    random_shock = vol * np.sqrt(dt) * np.random.normal(0, 1, (num_steps, simulations))
    daily_returns = np.exp(drift + random_shock)
    price_paths = np.zeros((num_steps + 1, simulations))
    price_paths[0] = initial_value
    for t in range(1, num_steps + 1): price_paths[t] = price_paths[t - 1] * daily_returns[t - 1]
    return pd.DataFrame(price_paths)

@st.cache_data
def calculate_efficient_frontier(returns: pd.DataFrame, num_portfolios: int = 2000):
    # ... (function is unchanged)
    results = []
    num_assets = len(returns.columns)
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    for _ in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        ret, vol = np.sum(mean_returns * weights), np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        results.append([ret, vol, ret / vol])
    return pd.DataFrame(results, columns=['return', 'volatility', 'sharpe'])

# ======================================================================================
# UI COMPONENTS
# ======================================================================================

def display_dashboard(username: str, portfolio: Dict[str, Any]):
    st.subheader(f"Welcome Back, {username.title()}!")

    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Future Projection", "🔍 Performance Analysis"])

    with tab1:
        # Rebalance Check and Profile Display
        last_rebalanced_date = dt.date.fromisoformat(portfolio.get("last_rebalanced_date", "2000-01-01"))
        if (dt.date.today() - last_rebalanced_date).days > 180:
            st.warning("**Time to Rebalance!** Your portfolio is over 6 months old.")
        
        # --- <<< UI UPDATE: Display full user profile >>> ---
        profile_cols = st.columns(3)
        profile_cols[0].metric("Risk Profile", portfolio['risk_profile'])
        profile_cols[1].metric("Financial Goal", portfolio.get('profile_answers', {}).get('Financial Goal', 'N/A'))
        profile_cols[2].metric("Investment Horizon", portfolio.get('profile_answers', {}).get('Investment Horizon', 'N/A'))

        # Core Metrics including VaR
        metric_cols = st.columns(4)
        metric_cols[0].metric("Expected Annual Return", f"{portfolio['metrics']['expected_return']:.2%}")
        metric_cols[1].metric("Expected Annual Volatility", f"{portfolio['metrics']['expected_volatility']:.2%}")
        metric_cols[2].metric("Sharpe Ratio", f"{portfolio['metrics']['sharpe_ratio']:.2f}")
        metric_cols[3].metric("Daily Value at Risk (95%)", f"{portfolio['metrics']['value_at_risk_95']:.2%}")

        weights = pd.Series(portfolio["weights"])
        fig_pie = go.Figure(go.Pie(labels=weights.index, values=weights.values, hole=0.4, marker_colors=px.colors.sequential.GnBu_r, textinfo="label+percent"))
        fig_pie.update_layout(showlegend=False, title_text="Current Portfolio Allocation", title_x=0.5)
        st.plotly_chart(fig_pie, use_container_width=True)

        with st.expander("⚙️ Settings, Rebalancing & Profile Change"):
            # (UI for rebalancing and profile change is unchanged)
            st.write("Here you can manually rebalance your portfolio or change your risk profile entirely.")
            if st.button("Manually Rebalance Portfolio", key="rebalance_manual"): st.session_state.rebalance_now = True; st.rerun()
            st.markdown("---")
            st.write("**Change Your Risk Profile**")
            new_profile_options = list(RISK_AVERSION_FACTORS.keys())
            new_profile = st.selectbox("Select new profile:", options=new_profile_options, index=new_profile_options.index(portfolio['risk_profile']))
            if st.button("Update Profile & Rebalance", type="primary"): st.session_state.profile_change_request, st.session_state.new_profile = True, new_profile; st.rerun()

    with tab2:
        # (Monte Carlo UI is unchanged)
        st.header("Future Growth Simulation")
        sim_cols = st.columns([1, 3])
        with sim_cols[0]:
            initial_investment = st.number_input("Initial Investment ($)", min_value=1000, value=10000, step=1000, format="%d")
            simulation_years = st.slider("Investment Horizon (Years)", min_value=1, max_value=30, value=10)
        sim_results = run_monte_carlo(initial_investment, portfolio['metrics']['expected_return'], portfolio['metrics']['expected_volatility'], simulation_years, 500)
        with sim_cols[1]:
            fig_sim = go.Figure()
            fig_sim.add_traces([go.Scatter(x=sim_results.index/252, y=sim_results[col], line_color='lightgrey', showlegend=False) for col in sim_results.columns[:100]])
            fig_sim.add_traces([go.Scatter(x=sim_results.index/252, y=sim_results.quantile(q, axis=1), line=dict(width=3), name=f'{q*100:.0f}th Percentile') for q in [0.1, 0.5, 0.9]])
            fig_sim.update_layout(title_text=f"Projected Growth of ${initial_investment:,.0f}", yaxis_tickformat="$,.0f")
            st.plotly_chart(fig_sim, use_container_width=True)
    
    with tab3:
        st.header("Performance & Risk Analysis")
        # Backtest vs SPY/QQQ
        st.subheader("Historical Performance Backtest")
        all_prices = get_price_data(list(weights.index) + ["SPY", "QQQ"])
        mpt_performance = (1 + all_prices[weights.index].pct_change().dropna().dot(weights)).cumprod()
        spy_performance, qqq_performance = (all_prices["SPY"] / all_prices["SPY"].iloc[0]), (all_prices["QQQ"] / all_prices["QQQ"].iloc[0])
        fig_backtest = go.Figure()
        fig_backtest.add_trace(go.Scatter(x=mpt_performance.index, y=mpt_performance, name='Your MPT Portfolio'))
        fig_backtest.add_trace(go.Scatter(x=spy_performance.index, y=spy_performance, name='S&P 500 (SPY)', line=dict(dash='dash')))
        fig_backtest.add_trace(go.Scatter(x=qqq_performance.index, y=qqq_performance, name='NASDAQ 100 (QQQ)', line=dict(dash='dot')))
        fig_backtest.update_layout(title="Performance vs. Market Benchmarks", yaxis_title="Growth of $1")
        st.plotly_chart(fig_backtest, use_container_width=True)

        # Efficient Frontier
        st.subheader("Efficient Frontier Analysis")
        returns = all_prices[weights.index].pct_change().dropna()
        frontier_df = calculate_efficient_frontier(returns)
        fig_frontier = px.scatter(frontier_df, x='volatility', y='return', color='sharpe', title='Efficient Frontier Analysis')
        fig_frontier.add_trace(go.Scatter(x=[portfolio['metrics']['expected_volatility']], y=[portfolio['metrics']['expected_return']], mode='markers', marker=dict(color='red', size=15, symbol='star'), name='Your Portfolio'))
        st.plotly_chart(fig_frontier, use_container_width=True)

        # --- <<< UI UPDATE: CPI CHART >>> ---
        st.subheader("Economic Context: Inflation")
        cpi_data = get_cpi_data()
        if cpi_data is not None:
            fig_cpi = go.Figure()
            fig_cpi.add_trace(go.Scatter(x=cpi_data.index, y=cpi_data['CPIAUCSL'], name='US Inflation Rate (YoY)'))
            fig_cpi.update_layout(title="US Inflation Rate (CPI)", yaxis_title="Annual Rate (%)")
            st.plotly_chart(fig_cpi, use_container_width=True)

def display_questionnaire() -> Tuple[str, bool, Dict]:
    st.subheader("Please Complete Your Investor Profile")
    answers = {}
    for key, options in QUESTIONNAIRE.items():
        answers[key] = st.radio(key.replace("_", " "), options)
    
    # Determine risk profile from score
    score = sum(QUESTIONNAIRE[key].index(answers[key]) for key in ["Risk Tolerance", "Investment Horizon"])
    if score <= 1: risk_profile = "Conservative"
    elif score <= 3: risk_profile = "Balanced"
    else: risk_profile = "Aggressive"

    use_ml_model = st.toggle("🧠 Use ML-Enhanced Volatility Forecast (GARCH)", value=False)
    if st.button("📈 Build My Portfolio", type="primary"):
        return risk_profile, use_ml_model, answers
    return "", False, {}

# ======================================================================================
# MAIN APPLICATION LOGIC
# ======================================================================================

def run_portfolio_creation(risk_profile: str, use_garch: bool, profile_answers: Dict) -> Dict | None:
    with st.spinner(f"Building your '{risk_profile}' portfolio..."):
        assets = MASTER_ASSET_LIST
        prices = get_price_data(assets)
        if prices.empty: return None
        returns = prices.pct_change().dropna()
        weights = optimize_portfolio(returns, risk_profile, use_garch=use_garch)
        if weights is not None:
            metrics = analyze_portfolio(weights, returns)
            return {"risk_profile": risk_profile, "weights": weights.to_dict(), "metrics": metrics, "last_rebalanced_date": dt.date.today().isoformat(), "used_garch": use_garch, "profile_answers": profile_answers}
    return None

def main():
    st.title("WealthFlow 🤖 AI-Powered Investment Advisor")
    st.markdown("This tool uses **Modern Portfolio Theory (MPT)** and optional **Machine Learning (GARCH) models** to build a diversified investment portfolio based on your unique investor profile.")
    
    all_portfolios = load_portfolios()
    username = st.text_input("Please enter your name to begin:", key="username_input")
    if not username: st.stop()

    user_exists = username in all_portfolios
    rebalance_triggered = st.session_state.get("rebalance_now", False)
    profile_change_triggered = st.session_state.get("profile_change_request", False)

    if (user_exists and rebalance_triggered) or profile_change_triggered:
        risk_profile = st.session_state.new_profile if profile_change_triggered else all_portfolios[username]['risk_profile']
        # Preserve original profile answers and GARCH choice on rebalance
        use_garch = all_portfolios[username].get('used_garch', False)
        profile_answers = all_portfolios[username].get('profile_answers', {})
        new_portfolio = run_portfolio_creation(risk_profile, use_garch, profile_answers)
        if new_portfolio:
            all_portfolios[username] = new_portfolio
            save_portfolios(all_portfolios)
            st.success(f"Your portfolio has been successfully updated to a '{risk_profile}' allocation!")
            st.balloons()
        st.session_state.rebalance_now, st.session_state.profile_change_request = False, False
        st.rerun()

    elif not user_exists:
        risk_profile, use_garch, answers = display_questionnaire()
        if risk_profile:
            new_portfolio = run_portfolio_creation(risk_profile, use_garch, answers)
            if new_portfolio:
                all_portfolios[username] = new_portfolio
                save_portfolios(all_portfolios)
                st.success("Your portfolio has been created!")
                st.balloons()
                st.rerun()
    
    else:
        display_dashboard(username, all_portfolios[username])

    st.markdown("---")
    st.caption("Ethical Considerations & Disclaimer: This application is a technology demonstration and does not constitute financial advice. All portfolio recommendations are generated based on mathematical models and historical data, which are not indicative of future results. User data is stored in a local `user_portfolios.json` file in the application's directory and is not transmitted elsewhere. Always consult with a qualified financial professional before making investment decisions.")

if __name__ == "__main__":
    main()
