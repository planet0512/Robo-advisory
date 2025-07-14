# robo_advisor_app.py
# Showcase version with advanced analytics: stress testing, yield curve, and HMM market regimes.

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
from hmmlearn import hmm # For Market Regime Detection

# ======================================================================================
# CONFIGURATION
# ======================================================================================

st.set_page_config(page_title="WealthFlow | AI Advisor", page_icon="🤖", layout="wide")
PORTFOLIO_FILE = Path("user_portfolios.json")
RISK_AVERSION_FACTORS = {"Conservative": 4.0, "Balanced": 2.5, "Aggressive": 1.0}
# <<< FINAL UPDATE: A professional-grade Core-Satellite / Factor-based asset list >>>
MASTER_ASSET_LIST = [
    "VTI",  # Core: U.S. Total Stock Market
    "VXUS", # Core: Total International Stock Market (ex-US)
    "BND",  # Core: U.S. Total Bond Market
    "QUAL", # Satellite/Factor: Quality (companies with stable earnings, strong balance sheets)
    "AVUV", # Satellite/Factor: Small-Cap Value (small, inexpensive companies)
]

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
# <<< FEATURE: HISTORICAL STRESS TEST SCENARIOS >>>
CRASH_SCENARIOS = {
    "2008 Financial Crisis": ("2007-10-09", "2009-03-09"),
    "COVID-19 Crash": ("2020-02-19", "2020-03-23"),
    "Dot-Com Bubble Burst": ("2000-03-10", "2002-10-09"),
}

# ======================================================================================
# DATA & PERSISTENCE
# ======================================================================================

@st.cache_data(ttl=dt.timedelta(hours=12))
@st.cache_data(ttl=dt.timedelta(hours=12))
def get_price_data(tickers: List[str], start_date: str, end_date: str = None) -> pd.DataFrame:
    end_date = end_date or dt.date.today().isoformat()
    try:
        # <<< FIX: Enforce auto_adjust=True for consistent, clean data >>>
        prices = yf.download(
            tickers, 
            start=start_date, 
            end=end_date, 
            progress=False, 
            auto_adjust=True
        )["Close"]
        return prices.ffill().dropna(axis=1, how="all") if not prices.empty else pd.DataFrame()
    except Exception: 
        return pd.DataFrame()

@st.cache_data(ttl=dt.timedelta(days=7))
def get_cpi_data(start_date="2010-01-01"):
    try:
        cpi = web.DataReader("CPIAUCSL", "fred", start=start_date)
        return cpi.pct_change(12) * 100
    except Exception: return None

# <<< FEATURE: REAL-TIME YIELD CURVE DATA >>>
@st.cache_data(ttl=dt.timedelta(hours=4))
@st.cache_data(ttl=dt.timedelta(hours=4))
def get_yield_curve_data():
    """
    Fetches and sorts the current US Treasury yield curve data.
    """
    tickers = {"3M": "^IRX", "5Y": "^FVX", "10Y": "^TNX", "30Y": "^TYX"}
    try:
        yields_raw = yf.Tickers(list(tickers.values())).history(period="5d")['Close'].iloc[-1]
        yield_curve = pd.Series({name: yields_raw[ticker] for name, ticker in tickers.items()})

        # <<< FIX: Enforce a logical sort order for the maturities >>>
        maturity_order = ["3M", "5Y", "10Y", "30Y"]
        yield_curve_sorted = yield_curve.reindex(maturity_order)
        
        return yield_curve_sorted
    except Exception: 
        return None

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

def optimize_portfolio(returns: pd.DataFrame, risk_profile: str, use_garch: bool = False) -> pd.Series:
    # ... (function is unchanged from previous version)
    mu = returns.mean().to_numpy() * 252
    Sigma = returns.cov().to_numpy() * 252 if not use_garch else forecast_covariance_garch(returns).to_numpy()
    mu, Sigma = np.nan_to_num(mu), np.nan_to_num(Sigma)
    try:
        gamma = cp.Parameter(nonneg=True, value=RISK_AVERSION_FACTORS[risk_profile])
        w = cp.Variable(len(mu))
        prob = cp.Problem(cp.Maximize(mu @ w - 0.5 * gamma * cp.quad_form(w, cp.psd_wrap(0.5 * (Sigma + Sigma.T)))), [cp.sum(w) == 1, w >= 0, w <= 0.35])
        prob.solve(solver=cp.SCS)
        if prob.status != cp.OPTIMAL: raise ValueError("Solver failed.")
        weights = pd.Series(w.value, index=returns.columns); weights[weights < 1e-4] = 0; weights /= weights.sum()
        return weights
    except Exception as e:
        st.error(f"Optimization failed: {e}"); return None

# <<< FEATURE: ML MARKET REGIME DETECTION (HMM) >>>
@st.cache_data(ttl=dt.timedelta(hours=12))
@st.cache_data(ttl=dt.timedelta(hours=12))
@st.cache_data(ttl=dt.timedelta(hours=12))
@st.cache_data(ttl=dt.timedelta(hours=12))
@st.cache_data(ttl=dt.timedelta(hours=12))
@st.cache_data(ttl=dt.timedelta(hours=12))
@st.cache_data(ttl=dt.timedelta(hours=12))
def detect_market_regimes(start_date="2010-01-01"):
    """
    Detects market regimes using a Hidden Markov Model (HMM) on SPY returns.
    This final version explicitly handles the multi-level column index from yfinance.
    """
    # Download raw data - we know it has a multi-level column index
    spy_raw_data = yf.download("SPY", start=start_date, progress=False, auto_adjust=False)

    # --- FINAL, ROBUST FIX ---
    # Explicitly handle the multi-level column index: ('Close', 'SPY')
    # First, select the 'Close' level, which results in a DataFrame with ticker names as columns.
    spy_close_df = spy_raw_data['Close']
    
    # Then, select the column for the SPY ticker, which is now a simple Series of prices.
    spy_close_prices = spy_close_df['SPY']

    # Calculate returns from the 'Close' price Series
    returns = np.log(spy_close_prices).diff().dropna()
    
    # Fit the HMM model
    model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(returns.to_numpy().reshape(-1, 1))
    hidden_states = model.predict(returns.to_numpy().reshape(-1, 1))
    
    # Identify regimes based on volatility
    vols = [np.sqrt(model.covars_[i][0][0]) for i in range(model.n_components)]
    high_vol_state = np.argmax(vols)
    
    # Create the regime DataFrame with the correct index
    regime_df = pd.DataFrame({
        'regime_label': ['High Volatility' if s == high_vol_state else 'Low Volatility' for s in hidden_states]
    }, index=returns.index)
    
    # Build the final DataFrame for the chart, starting with the price Series
    final_df = pd.DataFrame(spy_close_prices)
    final_df = final_df.join(regime_df['regime_label'])
    
    # Rename the price column from 'SPY' to 'Close' for consistency with the rest of the app
    final_df.rename(columns={'SPY': 'Close'}, inplace=True)
    
    # Forward-fill the first row's regime label which is NaN
    final_df['regime_label'].ffill(inplace=True)
    
    # Return the required columns with the correct names
    return final_df[['Close', 'regime_label']]

# Other financial functions (analyze_portfolio, run_monte_carlo, etc.) are here
def analyze_portfolio(weights: pd.Series, returns: pd.DataFrame) -> Dict[str, float]:
    portfolio_return = np.sum(returns.mean() * 252 * weights)
    cov_matrix = returns.cov() * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    var_95 = returns.dot(weights).quantile(0.05)
    return {"expected_return": portfolio_return, "expected_volatility": portfolio_volatility, "sharpe_ratio": portfolio_return / portfolio_volatility, "value_at_risk_95": var_95}

# All other helper functions from previous versions are assumed to be here and unchanged...
# (run_monte_carlo, calculate_efficient_frontier)
def run_monte_carlo(initial_value: float, er: float, vol: float, years: int, simulations: int) -> pd.DataFrame:
    """
    Runs a Monte Carlo simulation using Geometric Brownian Motion.
    """
    dt = 1 / 252
    num_steps = years * 252

    # Calculate drift and random component
    drift = (er - 0.5 * vol**2) * dt
    random_shock = vol * np.sqrt(dt) * np.random.normal(0, 1, (num_steps, simulations))

    # Calculate daily returns
    daily_returns = np.exp(drift + random_shock)

    # Simulate price paths
    price_paths = np.zeros((num_steps + 1, simulations))
    # <<< FIX: Use the correct argument name 'initial_value' >>>
    price_paths[0] = initial_value
    for t in range(1, num_steps + 1):
        price_paths[t] = price_paths[t - 1] * daily_returns[t - 1]

    return pd.DataFrame(price_paths)
@st.cache_data
def calculate_efficient_frontier(returns: pd.DataFrame, num_portfolios: int = 2000):
    """Generates random portfolios to visualize the efficient frontier."""
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
    """Calculates the historical cumulative performance of a portfolio."""
    returns = prices.pct_change().dropna()
    portfolio_returns = returns.dot(weights)
    return (1 + portfolio_returns).cumprod()

def calculate_drawdown(performance_series: pd.Series) -> pd.Series:
    """Calculates the drawdown of a performance series."""
    running_max = performance_series.cummax()
    drawdown = (performance_series / running_max) - 1
    return drawdown

# ======================================================================================
# UI COMPONENTS
# ======================================================================================

def display_dashboard(username: str, portfolio: Dict[str, Any]):
    st.subheader(f"Welcome Back, {username.title()}!")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📈 Future Projection", "🔍 Performance Analysis", "🧠 Advanced Analytics"])

    # --- TAB 1: Main Dashboard ---
    with tab1:
        # Rebalance Check
        last_rebalanced_date = dt.date.fromisoformat(portfolio.get("last_rebalanced_date", "2000-01-01"))
        if (dt.date.today() - last_rebalanced_date).days > 180:
            st.warning("**Time to Rebalance!** Your portfolio is over 6 months old.")

        # Display User Profile and Core Metrics
        profile_cols = st.columns(3)
        profile_cols[0].metric("Risk Profile", portfolio['risk_profile'])
        profile_cols[1].metric("Financial Goal", portfolio.get('profile_answers', {}).get('Financial Goal', 'N/A'))
        profile_cols[2].metric("Investment Horizon", portfolio.get('profile_answers', {}).get('Investment Horizon', 'N/A'))

        metric_cols = st.columns(4)
        metric_cols[0].metric("Expected Annual Return", f"{portfolio['metrics']['expected_return']:.2%}")
        metric_cols[1].metric("Expected Annual Volatility", f"{portfolio['metrics']['expected_volatility']:.2%}")
        metric_cols[2].metric("Sharpe Ratio", f"{portfolio['metrics']['sharpe_ratio']:.2f}")
        metric_cols[3].metric("Daily Value at Risk (95%)", f"{portfolio['metrics']['value_at_risk_95']:.2%}")

        # Allocation Pie Chart
        weights = pd.Series(portfolio["weights"])
        fig_pie = go.Figure(go.Pie(labels=weights.index, values=weights.values, hole=0.4, marker_colors=px.colors.sequential.GnBu_r, textinfo="label+percent"))
        fig_pie.update_layout(showlegend=False, title_text="Current Portfolio Allocation", title_x=0.5)
        st.plotly_chart(fig_pie, use_container_width=True)

        with st.expander("⚙️ Settings, Rebalancing & Profile Change"):
            if st.button("Manually Rebalance Portfolio", key="rebalance_manual"):
                st.session_state.rebalance_now = True
                st.rerun()
            st.markdown("---")
            new_profile = st.selectbox("Change risk profile:", options=list(RISK_AVERSION_FACTORS.keys()), index=list(RISK_AVERSION_FACTORS.keys()).index(portfolio['risk_profile']))
            if st.button("Update Profile & Rebalance", type="primary"):
                st.session_state.profile_change_request, st.session_state.new_profile = True, new_profile
                st.rerun()

    # --- TAB 2: Future Projection ---
    with tab2:
        st.header("Future Growth Simulation")
        sim_cols = st.columns([1, 3])
        with sim_cols[0]:
            initial_investment = st.number_input("Initial Investment ($)", min_value=1000, value=10000, step=1000, format="%d", key="mc_investment")
            simulation_years = st.slider("Investment Horizon (Years)", min_value=1, max_value=30, value=10, key="mc_years")
        sim_results = run_monte_carlo(initial_investment, portfolio['metrics']['expected_return'], portfolio['metrics']['expected_volatility'], simulation_years, 500)
        final_values = sim_results.iloc[-1]
        with sim_cols[1]:
            fig_sim = go.Figure()
            fig_sim.add_traces([go.Scatter(x=sim_results.index/252, y=sim_results[col], line_color='lightgrey', showlegend=False) for col in sim_results.columns[:100]])
            fig_sim.add_traces([go.Scatter(x=sim_results.index/252, y=sim_results.quantile(q, axis=1), line=dict(width=3), name=f'{q*100:.0f}th Percentile') for q in [0.1, 0.5, 0.9]])
            fig_sim.update_layout(title_text=f"Projected Growth of ${initial_investment:,.0f}", yaxis_tickformat="$,.0f", xaxis_title="Years", yaxis_title="Portfolio Value ($)")
            st.plotly_chart(fig_sim, use_container_width=True)

    # --- TAB 3: Performance Analysis ---
    with tab3:
        st.header("Performance & Risk Analysis")
        weights = pd.Series(portfolio["weights"])
        all_prices = get_price_data(list(weights.index) + ["SPY", "QQQ"], "2018-01-01")
        returns = all_prices[weights.index].pct_change().dropna()

        st.subheader("Historical Performance Backtest")
        mpt_performance = (1 + returns.dot(weights)).cumprod()
        spy_performance = (all_prices["SPY"] / all_prices["SPY"].iloc[0])
        fig_backtest = go.Figure()
        fig_backtest.add_trace(go.Scatter(x=mpt_performance.index, y=mpt_performance, name='Your MPT Portfolio'))
        fig_backtest.add_trace(go.Scatter(x=spy_performance.index, y=spy_performance, name='S&P 500 (SPY)', line=dict(dash='dash')))
        fig_backtest.update_layout(title="Performance vs. S&P 500 Benchmark", yaxis_title="Growth of $1")
        st.plotly_chart(fig_backtest, use_container_width=True)

        st.subheader("Sharpe Ratio Comparison")
        asset_returns = returns.mean() * 252
        asset_std_dev = returns.std() * np.sqrt(252)
        individual_sharpes = np.divide(asset_returns, asset_std_dev, out=np.zeros_like(asset_returns), where=asset_std_dev!=0)
        individual_sharpes = pd.Series(individual_sharpes, index=returns.columns)
        portfolio_sharpe = portfolio['metrics']['sharpe_ratio']
        sharpe_ratios_df = pd.DataFrame(individual_sharpes, columns=['Sharpe Ratio'])
        sharpe_ratios_df.loc['Your Portfolio'] = portfolio_sharpe
        st.bar_chart(sharpe_ratios_df)

        st.subheader("Efficient Frontier Analysis")
        frontier_df = calculate_efficient_frontier(returns)
        fig_frontier = px.scatter(frontier_df, x='volatility', y='return', color='sharpe', title='Efficient Frontier Analysis')
        fig_frontier.add_trace(go.Scatter(x=[portfolio['metrics']['expected_volatility']], y=[portfolio['metrics']['expected_return']], mode='markers', marker=dict(color='red', size=15, symbol='star'), name='Your Portfolio'))
        st.plotly_chart(fig_frontier, use_container_width=True)

    # --- TAB 4: Advanced Analytics ---
    with tab4:
        st.header("Advanced Analytics & Market Insights")
        # --- Historical Stress Testing ---
        st.subheader("Historical Stress Testing")
        st.write("This analysis shows how your current portfolio allocation would have performed during historical market crises compared to the S&P 500 (SPY) benchmark.")

        # Loop through each crisis scenario
        for name, (start, end) in CRASH_SCENARIOS.items():
            st.markdown(f"#### {name} (`{start}` to `{end}`)")
            
            # Get data for the specific crisis period for both portfolio and benchmark
            portfolio_assets = list(weights.index)
            spy_ticker = ["SPY"]
            
            crisis_prices = get_price_data(portfolio_assets, start, end)
            spy_prices = get_price_data(spy_ticker, start, end)

            if crisis_prices.empty or spy_prices.empty:
                st.warning("Could not retrieve sufficient data for this period.")
                continue

            # Calculate metrics for user's portfolio
            portfolio_returns = crisis_prices.pct_change().dot(weights).dropna()
            portfolio_cumulative = (1 + portfolio_returns).cumprod()
            portfolio_return = portfolio_cumulative.iloc[-1] - 1
            portfolio_drawdown = calculate_drawdown(portfolio_cumulative).min()

            # Calculate metrics for SPY benchmark
            spy_returns = spy_prices['SPY'].pct_change().dropna()
            spy_cumulative = (1 + spy_returns).cumprod()
            spy_return = spy_cumulative.iloc[-1] - 1
            spy_drawdown = calculate_drawdown(spy_cumulative).min()

            # Display results in a clean side-by-side layout
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Your Portfolio**")
                st.metric(label="Total Return", value=f"{portfolio_return:.2%}", delta_color="off")
                st.metric(label="Max Drawdown", value=f"{portfolio_drawdown:.2%}", delta_color="off")
            
            with col2:
                st.write("**S&P 500 (SPY)**")
                st.metric(label="Total Return", value=f"{spy_return:.2%}", delta_color="off")
                st.metric(label="Max Drawdown", value=f"{spy_drawdown:.2%}", delta_color="off")
            
            st.markdown("---")

        st.subheader("Live Market Indicators")
        indicator_cols = st.columns(2)
        with indicator_cols[0]:
            st.write("**US Treasury Yield Curve**"); yield_curve = get_yield_curve_data()
            if yield_curve is not None: st.line_chart(yield_curve)
        with indicator_cols[1]:
            st.write("**US Inflation Rate (YoY)**"); cpi_data = get_cpi_data()
            if cpi_data is not None: st.line_chart(cpi_data)
                
        
        # --- Machine Learning: Market Regime Detection ---
        st.subheader("Machine Learning: Market Regime Detection")
        st.write("This chart uses a Hidden Markov Model (HMM) to identify underlying market states. The colored backgrounds indicate periods of high or low volatility.")
        
        regime_data = detect_market_regimes()
        if regime_data is not None and not regime_data.empty:
            current_regime = regime_data['regime_label'].iloc[-1]
            st.info(f"The ML model indicates the market is currently in a **{current_regime}** state.")
            
            # --- FINAL UI FIX: Use background shading for a clearer visualization ---
            fig_regime = go.Figure()

            # Add the main SPY price line as a single, continuous line
            fig_regime.add_trace(go.Scatter(
                x=regime_data.index, y=regime_data['Close'], 
                mode='lines', name='SPY Price', line_color='black', showlegend=False
            ))

            # Define colors for the shaded regions and the legend
            colors = {'Low Volatility': 'rgba(0, 176, 246, 0.2)', 'High Volatility': 'rgba(255, 82, 82, 0.2)'}
            
            # Add invisible dummy traces to create a clean legend
            fig_regime.add_trace(go.Bar(name='Low Volatility Period', x=[None], y=[None], marker_color=colors['Low Volatility']))
            fig_regime.add_trace(go.Bar(name='High Volatility Period', x=[None], y=[None], marker_color=colors['High Volatility']))

            # Loop through the regimes to draw the shaded background rectangles
            for state in ['Low Volatility', 'High Volatility']:
                # Create groups of consecutive identical states
                for _, g in regime_data[regime_data['regime_label'] == state].groupby((regime_data['regime_label'] != regime_data['regime_label'].shift()).cumsum()):
                    fig_regime.add_vrect(
                        x0=g.index.min(), 
                        x1=g.index.max(), 
                        fillcolor=colors[state], 
                        line_width=0
                    )
            
            fig_regime.update_layout(
                title="Market Regimes Identified by HMM on SPY",
                yaxis_title="SPY Price",
                xaxis_title="Date",
                legend_title_text='Market Regime'
            )
            st.plotly_chart(fig_regime, use_container_width=True)

        else:
            st.warning("Could not generate the market regime analysis.")
            
def display_questionnaire() -> Tuple[str, bool, Dict]:
    # (function is unchanged)
    st.subheader("Please Complete Your Investor Profile")
    answers = {key: st.radio(key.replace("_", " "), options) for key, options in QUESTIONNAIRE.items()}
    score = sum(QUESTIONNAIRE[key].index(answers[key]) for key in ["Risk Tolerance", "Investment Horizon"])
    risk_profile = "Conservative" if score <= 1 else "Balanced" if score <= 3 else "Aggressive"
    use_ml_model = st.toggle("🧠 Use ML-Enhanced Volatility Forecast (GARCH)")
    if st.button("📈 Build My Portfolio", type="primary"): return risk_profile, use_ml_model, answers
    return "", False, {}

# ======================================================================================
# MAIN APPLICATION LOGIC
# ======================================================================================

def run_portfolio_creation(risk_profile: str, use_garch: bool, profile_answers: Dict) -> Dict | None:
    # (function is unchanged)
    with st.spinner(f"Building portfolio..."):
        prices = get_price_data(MASTER_ASSET_LIST, "2018-01-01")
        if prices.empty: return None
        returns = prices.pct_change().dropna()
        weights = optimize_portfolio(returns, risk_profile, use_garch)
        if weights is not None:
            metrics = analyze_portfolio(weights, returns)
            return {"risk_profile": risk_profile, "weights": weights.to_dict(), "metrics": metrics, "last_rebalanced_date": dt.date.today().isoformat(), "used_garch": use_garch, "profile_answers": profile_answers}
    return None

def main():
    # (function is largely unchanged, just the intro text)
    st.title("WealthFlow 🤖 AI-Powered Investment Advisor")
    st.markdown("Welcome! This tool uses **Modern Portfolio Theory (MPT)** and advanced **Machine Learning models** to build and analyze a diversified investment portfolio tailored to your unique investor profile.")
    
    all_portfolios = load_portfolios()
    username = st.text_input("Please enter your name to begin:", key="username_input")
    if not username: st.stop()

    user_exists = username in all_portfolios
    rebalance_triggered = st.session_state.get("rebalance_now", False)
    profile_change_triggered = st.session_state.get("profile_change_request", False)

    if (user_exists and rebalance_triggered) or profile_change_triggered:
        risk_profile = st.session_state.new_profile if profile_change_triggered else all_portfolios[username]['risk_profile']
        use_garch = all_portfolios[username].get('used_garch', False)
        profile_answers = all_portfolios[username].get('profile_answers', {})
        new_portfolio = run_portfolio_creation(risk_profile, use_garch, profile_answers)
        if new_portfolio:
            all_portfolios[username] = new_portfolio
            save_portfolios(all_portfolios)
            st.success(f"Portfolio updated to '{risk_profile}' allocation!")
            st.balloons()
        st.session_state.rebalance_now = st.session_state.profile_change_request = False
        st.rerun()

    elif not user_exists:
        risk_profile, use_garch, answers = display_questionnaire()
        if risk_profile:
            new_portfolio = run_portfolio_creation(risk_profile, use_garch, answers)
            if new_portfolio:
                all_portfolios[username] = new_portfolio
                save_portfolios(all_portfolios)
                st.success("Your portfolio has been created!"); st.balloons(); st.rerun()
    
    else:
        display_dashboard(username, all_portfolios[username])

    st.markdown("---")
    st.caption("Ethical Considerations & Disclaimer: This application is a technology demonstration and not financial advice. All recommendations are based on mathematical models and historical data, which are not indicative of future results. User data is stored locally in your browser's session and in a `user_portfolios.json` file in the application's directory; it is not transmitted elsewhere. Always consult a qualified financial professional.")

if __name__ == "__main__":
    main()
