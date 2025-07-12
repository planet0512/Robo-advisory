Of course. You've made two excellent observations: the **UI can be cleaner**, and more importantly, the **optimizer is not diversifying correctly**, leading to 100% allocation in a single ETF. This also explains why rebalancing shows no change.

Let's fix both issues. The optimizer problem is critical, so we'll address that first.

-----

### \#\# 1. Fixing the Portfolio Optimizer

The reason the optimizer is putting 100% into one asset is that our current formula doesn't properly penalize risk. It's simply picking the single asset with the best historical risk-adjusted return.

To fix this, we'll introduce a **Risk Aversion** parameter. This tells the optimizer how much to penalize risk, forcing it to diversify based on the user's profile:

  * **Conservative users** will have a high risk aversion, leading to a more diversified, lower-volatility portfolio.
  * **Aggressive users** will have a low risk aversion, allowing for a more concentrated, higher-return portfolio.

This will fix both the diversification and the rebalancing problems.

### \#\# 2. Optimizing the User Interface (UI)

To make the app cleaner and more professional, I will reorganize the entire dashboard into a series of **tabs**. This eliminates the long scroll and groups related information logically. The new layout will be:

  * **Dashboard**: The main overview with key metrics, the allocation pie chart, and the rebalance prompt.
  * **Future Projection**: The interactive Monte Carlo simulation.
  * **Performance Analysis**: The Historical Backtest and Efficient Frontier charts.

-----

### \#\# The Updated Code

Here is the complete, updated script with both the critical optimizer fix and the new tab-based UI.

```python
# robo_advisor_app.py
# Final version with optimizer fix and tabbed UI.

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
    page_title="WealthFlow | Robo-Advisor",
    page_icon="🤖",
    layout="wide",
)

PORTFOLIO_FILE = Path("user_portfolios.json")

# <<< FIX: ADD RISK AVERSION FACTORS FOR DIVERSIFICATION >>>
# Higher value = more risk aversion = more diversification
RISK_AVERSION_FACTORS = {
    "Conservative": 4.0,
    "Balanced": 2.5,
    "Aggressive": 1.0,
}

ASSET_POOLS = {
    "Conservative": ["BND", "TIP", "LQD", "IEF", "AGG"],
    "Balanced": ["SPY", "VEA", "VWO", "BND", "VNQ"],
    "Aggressive": ["QQQ", "SPYG", "VGT", "ARKK", "IWM"],
}

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

# <<< FIX: PASS RISK_PROFILE TO OPTIMIZER TO USE RISK AVERSION FACTOR >>>
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
        gamma = cp.Parameter(nonneg=True) # <<< FIX: Define gamma as a parameter
        gamma.value = RISK_AVERSION_FACTORS[risk_profile] # <<< FIX: Set gamma based on risk profile

        Sigma = 0.5 * (Sigma + Sigma.T)
        P = cp.psd_wrap(Sigma)
        w = cp.Variable(len(mu))
        risk = cp.quad_form(w, P)
        
        # <<< FIX: Use gamma in the objective function to penalize risk >>>
        prob = cp.Problem(cp.Maximize(mu @ w - 0.5 * gamma * risk), [cp.sum(w) == 1, w >= 0])
        
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
    # ... (This function remains the same)
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
    # ... (This function remains the same)
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

@st.cache_data
def backtest_portfolio(prices: pd.DataFrame, weights: pd.Series) -> pd.Series:
    # ... (This function remains the same)
    returns = prices.pct_change().dropna()
    portfolio_returns = returns.dot(weights)
    return (1 + portfolio_returns).cumprod()


# ======================================================================================
# UI COMPONENTS
# ======================================================================================

# <<< UI UPDATE: The entire dashboard is now in tabs for a cleaner look >>>
def display_dashboard(username: str, portfolio: Dict[str, Any]):
    st.subheader(f"Welcome Back, {username.title()}!")

    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Future Projection", "🔍 Performance Analysis"])

    # --- TAB 1: Main Dashboard ---
    with tab1:
        # Rebalance Check
        last_rebalanced_date = dt.date.fromisoformat(portfolio.get("last_rebalanced_date", "2000-01-01"))
        days_since_rebalance = (dt.date.today() - last_rebalanced_date).days
        if days_since_rebalance > 180:
            st.warning(f"**Time to Rebalance!** Your portfolio is over 6 months old.")
            if st.button("🔄 Rebalance Now", type="primary", key="rebalance_main"):
                st.session_state.rebalance_now = True
                st.rerun()

        # Core Metrics & Allocation
        st.metric("Risk Profile", portfolio['risk_profile'])
        cols = st.columns(3)
        cols[0].metric("Expected Annual Return", f"{portfolio['metrics']['expected_return']:.2%}")
        cols[1].metric("Expected Annual Volatility", f"{portfolio['metrics']['expected_volatility']:.2%}")
        cols[2].metric("Sharpe Ratio", f"{portfolio['metrics']['sharpe_ratio']:.2f}")

        weights = pd.Series(portfolio["weights"])
        fig_pie = go.Figure(go.Pie(labels=weights.index, values=weights.values, hole=0.4, marker_colors=px.colors.sequential.GnBu_r, textinfo="label+percent"))
        fig_pie.update_layout(showlegend=False, title_text="Current Portfolio Allocation", title_x=0.5)
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- TAB 2: Monte Carlo Simulation ---
    with tab2:
        st.header("Future Growth Simulation")
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
    
    # --- TAB 3: Performance Analysis ---
    with tab3:
        st.header("Performance & Risk Analysis")
        prices = get_price_data(list(weights.index))
        returns = prices.pct_change().dropna()

        # Backtesting
        st.subheader("Historical Performance Backtest")
        st.write("This chart shows how your MPT-optimized portfolio would have performed historically against a simple, equal-weight portfolio.")
        equal_weights = pd.Series([1/len(weights)] * len(weights), index=weights.index)
        mpt_performance = backtest_portfolio(prices, weights)
        benchmark_performance = backtest_portfolio(prices, equal_weights)
        fig_backtest = go.Figure()
        fig_backtest.add_trace(go.Scatter(x=mpt_performance.index, y=mpt_performance, mode='lines', name='Your MPT Portfolio'))
        fig_backtest.add_trace(go.Scatter(x=benchmark_performance.index, y=benchmark_performance, mode='lines', name='Equal-Weight Benchmark', line=dict(dash='dash')))
        fig_backtest.update_layout(title="Historical Performance: MPT vs. Benchmark", yaxis_title="Growth of $1", yaxis_tickformat=".2f")
        st.plotly_chart(fig_backtest, use_container_width=True)

        # Efficient Frontier
        st.subheader("Efficient Frontier Analysis")
        st.write("The efficient frontier shows thousands of possible portfolios. Your portfolio (red star) is optimized to provide the best return for its level of risk.")
        frontier_df = calculate_efficient_frontier(returns)
        fig_frontier = px.scatter(frontier_df, x='volatility', y='return', color='sharpe', title='Efficient Frontier Analysis')
        fig_frontier.add_trace(go.Scatter(x=[portfolio['metrics']['expected_volatility']], y=[portfolio['metrics']['expected_return']], mode='markers', marker=dict(color='red', size=15, symbol='star'), name='Your Portfolio'))
        st.plotly_chart(fig_frontier, use_container_width=True)


def display_questionnaire() -> Tuple[str, bool]:
    # ... (This function remains the same)
    st.subheader("Answer a Few Questions to Build Your Portfolio")
    total_score = 0
    for i, (question, options) in enumerate(QUESTIONNAIRE.items()):
        response_index = options.index(st.radio(question, options, key=f"q_{i}"))
        total_score += response_index
    use_ml_model = st.toggle("🧠 Use ML-Enhanced Volatility Forecast (GARCH)", value=False, help="Uses a machine learning model to forecast risk instead of relying only on historical data.")
    if st.button("📈 Build My Portfolio", type="primary"):
        if total_score <= 2: return "Conservative", use_ml_model
        if total_score <= 5: return "Balanced", use_ml_model
        return "Aggressive", use_ml_model
    return "", False

# ======================================================================================
# MAIN APPLICATION LOGIC
# ======================================================================================

# <<< FIX: PASS RISK_PROFILE TO HELPER FUNCTION >>>
def run_portfolio_creation(risk_profile: str, use_garch: bool) -> Dict | None:
    with st.spinner(f"Building your '{risk_profile}' portfolio..."):
        assets = ASSET_POOLS[risk_profile]
        prices = get_price_data(assets)
        if prices.empty: return None
        
        returns = prices.pct_change().dropna()
        # <<< FIX: PASS RISK_PROFILE TO OPTIMIZER >>>
        weights = optimize_portfolio(returns, risk_profile, use_garch=use_garch)
        
        if weights is not None:
            metrics = analyze_portfolio(weights, returns)
            return {
                "risk_profile": risk_profile,
                "weights": weights.to_dict(),
                "metrics": metrics,
                "last_rebalanced_date": dt.date.today().isoformat(),
                "used_garch": use_garch
            }
    return None

def main():
    st.title("WealthFlow 🤖")
    st.caption("Your Personal AI-Powered Investment Advisor")
    all_portfolios = load_portfolios()
    
    username = st.text_input("Please enter your name to begin:", key="username_input")
    if not username:
        st.info("Enter a name to load or create your investment portfolio.")
        st.stop()

    user_exists = username in all_portfolios
    rebalance_triggered = st.session_state.get("rebalance_now", False)

    if user_exists and rebalance_triggered:
        st.session_state.rebalance_now = False
        risk_profile = all_portfolios[username]['risk_profile']
        use_garch = all_portfolios[username].get('used_garch', False)
        
        new_portfolio = run_portfolio_creation(risk_profile, use_garch)
        if new_portfolio:
            all_portfolios[username] = new_portfolio
            save_portfolios(all_portfolios)
            st.success("Your portfolio has been successfully rebalanced!")
            st.balloons()
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

if __name__ == "__main__":
    main()
```
