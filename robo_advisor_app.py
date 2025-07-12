# robo_advisor_app.py
# Final, polished version with advanced benchmarking, user profile updates, and enhanced UI.

import json
import datetime as dt
from pathlib import Path
from typing import Dict, List, Any, Tuple

from arch import arch_model
import cvxpy as cp
import numpy as np
import pandas as pd
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

RISK_AVERSION_FACTORS = {
    "Conservative": 4.0,
    "Balanced": 2.5,
    "Aggressive": 1.0,
}

MASTER_ASSET_LIST = [
    "VTI",  # U.S. Total Stock Market
    "VEA",  # Developed Markets (ex-US)
    "VWO",  # Emerging Markets
    "BND",  # U.S. Total Bond Market
    "VNQ",  # U.S. Real Estate
    "TIP",  # Inflation-Protected Bonds
]

QUESTIONNAIRE = {
    "When you think about investing, what's your primary goal?": [
        "Preserving my capital is my #1 priority.",
        "I need a balance of safety and growth.",
        "Growth is my main objective, and I can tolerate volatility.",
    ],
    "If your portfolio lost 20% of its value in one year, what would you do?": [
        "Sell all of it to prevent further loss.",
        "Sell some, but not all.",
        "Hold on and wait for it to recover.",
        "Buy more at the dip.",
    ],
    "What is your investment time horizon?": [
        "Short-term (less than 3 years)",
        "Medium-term (3 to 7 years)",
        "Long-term (more than 7 years)",
    ],
}

# ======================================================================================
# DATA & PERSISTENCE
# ======================================================================================

@st.cache_data(ttl=dt.timedelta(hours=12))
def get_price_data(tickers: List[str], start_date: str = "2018-01-01") -> pd.DataFrame:
    try:
        prices = yf.download(tickers, start=start_date, progress=False)["Close"]
        if prices.empty: return pd.DataFrame()
        prices = prices.dropna(axis=1, how="all")
        return prices.ffill().dropna()
    except Exception as e:
        st.error(f"An error occurred while fetching data: {e}")
        return pd.DataFrame()

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
    mu = returns.mean().to_numpy() * 252
    if use_garch:
        st.toast("Using GARCH model for risk forecast...", icon="🧠")
        Sigma = forecast_covariance_garch(returns).to_numpy()
    else:
        st.toast("Using historical model for risk...", icon="📜")
        Sigma = returns.cov().to_numpy() * 252
    
    mu = np.nan_to_num(mu)
    Sigma = np.nan_to_num(Sigma)
    
    try:
        gamma = cp.Parameter(nonneg=True)
        gamma.value = RISK_AVERSION_FACTORS[risk_profile]
        Sigma = 0.5 * (Sigma + Sigma.T)
        P = cp.psd_wrap(Sigma)
        w = cp.Variable(len(mu))
        risk = cp.quad_form(w, P)
        max_weight = 0.35 
        constraints = [cp.sum(w) == 1, w >= 0, w <= max_weight]
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
    portfolio_return = np.sum(returns.mean() * 252 * weights)
    cov_matrix = returns.cov() * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility
    return {"expected_return": portfolio_return, "expected_volatility": portfolio_volatility, "sharpe_ratio": sharpe_ratio}

def run_monte_carlo(initial_value: float, er: float, vol: float, years: int, simulations: int) -> pd.DataFrame:
    dt = 1 / 252
    num_steps = years * 252
    drift = (er - 0.5 * vol**2) * dt
    random_shock = vol * np.sqrt(dt) * np.random.normal(0, 1, (num_steps, simulations))
    daily_returns = np.exp(drift + random_shock)
    price_paths = np.zeros((num_steps + 1, simulations))
    price_paths[0] = initial_value
    for t in range(1, num_steps + 1):
        price_paths[t] = price_paths[t - 1] * daily_returns[t - 1]
    return pd.DataFrame(price_paths)

@st.cache_data
def calculate_efficient_frontier(returns: pd.DataFrame, num_portfolios: int = 2000):
    results = []
    num_assets = len(returns.columns)
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    for _ in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        ret = np.sum(mean_returns * weights)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = ret / vol
        results.append([ret, vol, sharpe])
    return pd.DataFrame(results, columns=['return', 'volatility', 'sharpe'])

# ======================================================================================
# UI COMPONENTS
# ======================================================================================

def display_dashboard(username: str, portfolio: Dict[str, Any]):
    st.subheader(f"Welcome Back, {username.title()}!")

    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Future Projection", "🔍 Performance Analysis"])

    with tab1:
        last_rebalanced_date = dt.date.fromisoformat(portfolio.get("last_rebalanced_date", "2000-01-01"))
        if (dt.date.today() - last_rebalanced_date).days > 180:
            st.warning("**Time to Rebalance!** Your portfolio is over 6 months old and may have drifted from its target.")
        
        st.metric("Current Risk Profile", portfolio['risk_profile'])
        cols = st.columns(3)
        cols[0].metric("Expected Annual Return", f"{portfolio['metrics']['expected_return']:.2%}")
        cols[1].metric("Expected Annual Volatility", f"{portfolio['metrics']['expected_volatility']:.2%}")
        cols[2].metric("Sharpe Ratio", f"{portfolio['metrics']['sharpe_ratio']:.2f}")

        weights = pd.Series(portfolio["weights"])
        fig_pie = go.Figure(go.Pie(labels=weights.index, values=weights.values, hole=0.4, marker_colors=px.colors.sequential.GnBu_r, textinfo="label+percent"))
        fig_pie.update_layout(showlegend=False, title_text="Current Portfolio Allocation", title_x=0.5)
        st.plotly_chart(fig_pie, use_container_width=True)

        with st.expander("⚙️ Settings & Rebalancing"):
            st.write("Here you can manually rebalance your portfolio or change your risk profile entirely.")
            
            # Manual Rebalance
            if st.button("Manually Rebalance Portfolio", key="rebalance_manual"):
                st.session_state.rebalance_now = True
                st.rerun()

            # --- FEATURE: Change Risk Profile ---
            st.markdown("---")
            st.write("**Change Your Risk Profile**")
            new_profile = st.selectbox(
                "Select your new risk profile:",
                options=["Conservative", "Balanced", "Aggressive"],
                index=["Conservative", "Balanced", "Aggressive"].index(portfolio['risk_profile'])
            )
            if st.button("Update Profile & Rebalance", type="primary"):
                st.session_state.profile_change_request = True
                st.session_state.new_profile = new_profile
                st.rerun()

    with tab2:
        # Monte Carlo Simulation UI
        st.header("Future Growth Simulation")
        # ... (code is unchanged)
        sim_cols = st.columns([1, 3])
        with sim_cols[0]:
            initial_investment = st.number_input("Initial Investment ($)", min_value=1000, value=10000, step=1000, format="%d")
            simulation_years = st.slider("Investment Horizon (Years)", min_value=1, max_value=30, value=10)
        sim_results = run_monte_carlo(initial_investment, portfolio['metrics']['expected_return'], portfolio['metrics']['expected_volatility'], simulation_years, 500)
        final_values = sim_results.iloc[-1]
        with sim_cols[1]:
            fig_sim = go.Figure()
            fig_sim.add_traces([go.Scatter(x=sim_results.index / 252, y=sim_results[col], mode='lines', line_color='lightgrey', showlegend=False) for col in sim_results.columns[:100]])
            fig_sim.add_traces([go.Scatter(x=sim_results.index / 252, y=sim_results.quantile(q, axis=1), mode='lines', line=dict(width=3), name=f'{q*100:.0f}th Percentile') for q in [0.1, 0.5, 0.9]])
            fig_sim.update_layout(title_text=f"Projected Growth of ${initial_investment:,.0f}", xaxis_title="Years", yaxis_title="Portfolio Value ($)", yaxis_tickformat="$,.0f")
            st.plotly_chart(fig_sim, use_container_width=True)
        st.info(f"After **{simulation_years} years**, your portfolio has a projected median value of **${final_values.median():,.0f}**.")
    
    with tab3:
        st.header("Performance & Risk Analysis")
        
        # --- FEATURE: Backtest against SPY and QQQ ---
        st.subheader("Historical Performance Backtest")
        st.write("This chart shows how your MPT-optimized portfolio would have performed historically against the S&P 500 (SPY) and NASDAQ 100 (QQQ) market benchmarks.")
        
        @st.cache_data
        def get_benchmark_data(tickers, start_date):
            return yf.download(tickers, start=start_date, progress=False)["Close"]

        # Get data for portfolio and benchmarks
        all_tickers = list(weights.index) + ["SPY", "QQQ"]
        all_prices = get_benchmark_data(all_tickers, "2018-01-01")
        
        # Calculate performance
        mpt_performance = (1 + all_prices[weights.index].pct_change().dropna().dot(weights)).cumprod()
        spy_performance = (all_prices["SPY"] / all_prices["SPY"].iloc[0])
        qqq_performance = (all_prices["QQQ"] / all_prices["QQQ"].iloc[0])
        
        fig_backtest = go.Figure()
        fig_backtest.add_trace(go.Scatter(x=mpt_performance.index, y=mpt_performance, mode='lines', name='Your MPT Portfolio'))
        fig_backtest.add_trace(go.Scatter(x=spy_performance.index, y=spy_performance, mode='lines', name='S&P 500 (SPY)', line=dict(dash='dash')))
        fig_backtest.add_trace(go.Scatter(x=qqq_performance.index, y=qqq_performance, mode='lines', name='NASDAQ 100 (QQQ)', line=dict(dash='dot')))
        fig_backtest.update_layout(title="Historical Performance vs. Market Benchmarks", yaxis_title="Growth of $1", yaxis_tickformat=".2f")
        st.plotly_chart(fig_backtest, use_container_width=True)

        # Efficient Frontier
        st.subheader("Efficient Frontier Analysis")
        returns = all_prices[weights.index].pct_change().dropna()
        frontier_df = calculate_efficient_frontier(returns)
        fig_frontier = px.scatter(frontier_df, x='volatility', y='return', color='sharpe', title='Efficient Frontier Analysis')
        fig_frontier.add_trace(go.Scatter(x=[portfolio['metrics']['expected_volatility']], y=[portfolio['metrics']['expected_return']], mode='markers', marker=dict(color='red', size=15, symbol='star'), name='Your Portfolio'))
        st.plotly_chart(fig_frontier, use_container_width=True)

def display_questionnaire() -> Tuple[str, bool]:
    st.subheader("Answer a Few Questions to Build Your Portfolio")
    total_score = 0
    for i, (question, options) in enumerate(QUESTIONNAIRE.items()):
        response_index = options.index(st.radio(question, options, key=f"q_{i}"))
        total_score += response_index
    use_ml_model = st.toggle("🧠 Use ML-Enhanced Volatility Forecast (GARCH)", value=False)
    if st.button("📈 Build My Portfolio", type="primary"):
        if total_score <= 2: return "Conservative", use_ml_model
        if total_score <= 5: return "Balanced", use_ml_model
        return "Aggressive", use_ml_model
    return "", False

# ======================================================================================
# MAIN APPLICATION LOGIC
# ======================================================================================

def run_portfolio_creation(risk_profile: str, use_garch: bool) -> Dict | None:
    with st.spinner(f"Building your '{risk_profile}' portfolio..."):
        assets = MASTER_ASSET_LIST
        prices = get_price_data(assets)
        if prices.empty: return None
        returns = prices.pct_change().dropna()
        weights = optimize_portfolio(returns, risk_profile, use_garch=use_garch)
        if weights is not None:
            metrics = analyze_portfolio(weights, returns)
            return {"risk_profile": risk_profile, "weights": weights.to_dict(), "metrics": metrics, "last_rebalanced_date": dt.date.today().isoformat(), "used_garch": use_garch}
    return None

def main():
    st.title("WealthFlow 🤖 AI-Powered Investment Advisor")
    st.markdown("Welcome to your personal robo-advisor. This tool uses **Modern Portfolio Theory (MPT)** and optional **Machine Learning (GARCH) models** to build a diversified investment portfolio tailored to your risk preferences.")
    
    all_portfolios = load_portfolios()
    
    username = st.text_input("Please enter your name to begin:", key="username_input")
    if not username:
        st.info("Enter a name to load or create your investment portfolio.")
        st.stop()

    user_exists = username in all_portfolios
    rebalance_triggered = st.session_state.get("rebalance_now", False)
    profile_change_triggered = st.session_state.get("profile_change_request", False)

    if (user_exists and rebalance_triggered) or profile_change_triggered:
        risk_profile = st.session_state.new_profile if profile_change_triggered else all_portfolios[username]['risk_profile']
        use_garch = all_portfolios[username].get('used_garch', False)
        new_portfolio = run_portfolio_creation(risk_profile, use_garch)
        if new_portfolio:
            all_portfolios[username] = new_portfolio
            save_portfolios(all_portfolios)
            st.success(f"Your portfolio has been successfully updated to a '{risk_profile}' allocation!")
            st.balloons()
        st.session_state.rebalance_now = False
        st.session_state.profile_change_request = False
        st.rerun()

    elif not user_exists:
        risk_profile, use_garch = display_questionnaire()
        if risk_profile:
            new_portfolio = run_portfolio_creation(risk_profile, use_garch)
            if new_portfolio:
                all_portfolios[username] = new_portfolio
                save_portfolios(all_portfolios)
                st.success("Your portfolio has been created!")
                st.balloons()
                st.rerun()
    
    else:
        display_dashboard(username, all_portfolios[username])

    st.markdown("---")
    st.caption("Disclaimer: This is a technology demonstration and not financial advice. All investment decisions should be made with the consultation of a qualified financial professional.")

if __name__ == "__main__":
    main()
