# robo_advisor_app_v9.py
# Final version with UI enhancements for clarity on Monte Carlo results
# and GARCH model usage.

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
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Close']
        else:
            prices = data[['Close']]
            if len(tickers) == 1:
                prices = prices.rename(columns={'Close': tickers[0]})
        return prices.ffill().dropna(axis=1, how="all")
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

def optimize_portfolio(returns: pd.DataFrame, risk_profile: str, use_garch: bool = False) -> pd.Series:
    mu = returns.mean().to_numpy() * 252
    if use_garch:
        Sigma = forecast_garch_covariance(returns).to_numpy()
    else:
        Sigma = returns.cov().to_numpy() * 252

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

def analyze_portfolio(weights: pd.Series, returns: pd.DataFrame) -> Dict[str, float]:
    portfolio_returns_ts = returns.dot(weights)
    expected_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = portfolio_returns_ts.std() * np.sqrt(252)
    sharpe_ratio = expected_return / portfolio_volatility if portfolio_volatility != 0 else 0
    var_95 = portfolio_returns_ts.quantile(0.05)
    cvar_95 = portfolio_returns_ts[portfolio_returns_ts <= var_95].mean()
    return {
        "expected_return": expected_return, "expected_volatility": portfolio_volatility,
        "sharpe_ratio": sharpe_ratio, "value_at_risk_95": var_95,
        "conditional_value_at_risk_95": cvar_95,
    }

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
    except Exception:
        return None

def run_monte_carlo(initial_value, er, vol, years, simulations):
    dt = 1/252
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
def calculate_efficient_frontier(returns, num_portfolios=2000):
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

def calculate_drawdown(performance_series):
    running_max = performance_series.cummax()
    return (performance_series / running_max) - 1

# ======================================================================================
# UI COMPONENTS
# ======================================================================================

def display_dashboard(username: str, portfolio: Dict[str, Any]):
    st.subheader(f"Welcome Back, {username.title()}!")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📈 Future Projection", "🔍 Performance Analysis", "🧠 Portfolio Intelligence"])
    weights = pd.Series(portfolio["weights"])

    with tab1:
        # <<< ENHANCEMENT: Added ML Model Status Indicator >>>
        profile_cols = st.columns(3)
        profile_cols[0].metric("Risk Profile", portfolio['risk_profile'])
        profile_cols[1].metric("Financial Goal", portfolio.get('profile_answers', {}).get('Financial Goal', 'N/A'))
        garch_status = "Active (GARCH)" if portfolio.get('used_garch', False) else "Inactive"
        profile_cols[2].metric("🧠 ML Volatility Model", garch_status)
        
        st.markdown("---")
        metric_cols = st.columns(5)
        metric_cols[0].metric("Expected Annual Return", f"{portfolio['metrics']['expected_return']:.2%}")
        metric_cols[1].metric("Annual Volatility", f"{portfolio['metrics']['expected_volatility']:.2%}")
        metric_cols[2].metric("Sharpe Ratio", f"{portfolio['metrics']['sharpe_ratio']:.2f}")
        metric_cols[3].metric("Daily VaR (95%)", f"{portfolio['metrics']['value_at_risk_95']:.2%}")
        metric_cols[4].metric("Daily CVaR (95%)", f"{portfolio['metrics']['conditional_value_at_risk_95']:.2%}", help="The expected loss on days within the worst 5% of scenarios.")
        st.markdown("---")

        fig_pie = go.Figure(go.Pie(labels=weights.index, values=weights.values, hole=0.4, textinfo="label+percent"))
        fig_pie.update_layout(showlegend=False, title_text="Current Portfolio Allocation", title_x=0.5)
        st.plotly_chart(fig_pie, use_container_width=True)

        with st.expander("⚙️ Settings, Rebalancing & Profile Change"):
            st.write("Update your risk profile or rebalance to the latest market data.")
            current_profile_index = list(RISK_AVERSION_FACTORS.keys()).index(portfolio['risk_profile'])
            was_garch_used = portfolio.get('used_garch', False)
            new_profile = st.selectbox("Change risk profile:", options=list(RISK_AVERSION_FACTORS.keys()), index=current_profile_index)
            use_garch_rebalance = st.toggle("🧠 Use ML-Enhanced Volatility Forecast (GARCH)", value=was_garch_used, key="rebalance_garch")
            if st.button("Update and Rebalance Portfolio", type="primary"):
                st.session_state.rebalance_request = {"new_profile": new_profile, "use_garch": use_garch_rebalance}
                st.rerun()

    with tab2:
        st.header("Future Growth Simulation (Monte Carlo)")
        sim_cols = st.columns([1, 3])
        with sim_cols[0]:
            initial_investment = st.number_input("Initial Investment ($)", 1000, 1000000, 10000, 1000)
            simulation_years = st.slider("Investment Horizon (Years)", 1, 40, 10)
        
        sim_results = run_monte_carlo(initial_investment, portfolio['metrics']['expected_return'], portfolio['metrics']['expected_volatility'], simulation_years, 500)
        
        with sim_cols[1]:
            fig_sim = go.Figure()
            fig_sim.add_traces([go.Scatter(x=sim_results.index/252, y=sim_results[col], line_color='lightgrey', showlegend=False) for col in sim_results.columns[:100]])
            quantiles = sim_results.quantile([0.1, 0.5, 0.9], axis=1).T
            for q_val, q_name in zip([0.1, 0.5, 0.9], ["10th Percentile", "Median", "90th Percentile"]):
                 fig_sim.add_trace(go.Scatter(x=sim_results.index/252, y=quantiles[q_val], line=dict(width=3), name=q_name))
            fig_sim.update_layout(title_text=f"Projected Growth of ${initial_investment:,.0f}", yaxis_tickformat="$,.0f", xaxis_title="Years", yaxis_title="Portfolio Value ($)")
            st.plotly_chart(fig_sim, use_container_width=True)
        
        # <<< ENHANCEMENT: Display specific projection values >>>
        st.markdown("---")
        final_values = sim_results.iloc[-1]
        pessimistic = final_values.quantile(0.1)
        median = final_values.quantile(0.5)
        optimistic = final_values.quantile(0.9)
        
        st.subheader(f"Projected Outcomes after {simulation_years} Years")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Pessimistic Outcome (10%)", f"${pessimistic:,.2f}")
        metric_cols[1].metric("Median Outcome (50%)", f"${median:,.2f}")
        metric_cols[2].metric("Optimistic Outcome (90%)", f"${optimistic:,.2f}")


    with tab3:
        # ... (This tab's code remains the same) ...
        st.header("Performance & Risk Analysis")
        all_prices = get_price_data(list(weights.index) + ["SPY"], "2018-01-01")
        if not all_prices.empty and not all_prices.isnull().all().all():
            valid_assets = [col for col in weights.index if col in all_prices.columns and not all_prices[col].isnull().all()]
            returns = all_prices[valid_assets].pct_change().dropna()
            if not returns.empty:
                st.subheader("Historical Performance Backtest")
                aligned_weights = weights[valid_assets].copy()
                aligned_weights /= aligned_weights.sum()
                portfolio_performance = (1 + returns.dot(aligned_weights)).cumprod()
                spy_performance = (1 + all_prices["SPY"].pct_change().dropna()).cumprod()
                fig_backtest = go.Figure()
                fig_backtest.add_trace(go.Scatter(x=portfolio_performance.index, y=portfolio_performance, name='Your Portfolio'))
                fig_backtest.add_trace(go.Scatter(x=spy_performance.index, y=spy_performance, name='S&P 500 (SPY)', line=dict(dash='dash')))
                fig_backtest.update_layout(title="Performance vs. S&P 500 Benchmark (Growth of $1)", yaxis_title="Cumulative Growth")
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
        else:
            st.warning("Could not retrieve sufficient historical data for the Performance Analysis tab.")
    
    with tab4:
        # ... (This tab's code remains the same) ...
        st.header("Portfolio Intelligence")
        st.subheader("Historical Stress Testing")
        for name, (start, end) in CRASH_SCENARIOS.items():
            st.markdown(f"#### {name} (`{start}` to `{end}`)")
            all_assets_for_period = list(weights.index) + ["SPY"]
            crisis_prices = get_price_data(all_assets_for_period, start, end)
            if crisis_prices.empty or "SPY" not in crisis_prices.columns or crisis_prices['SPY'].isnull().all():
                st.warning(f"Could not retrieve valid market data for the {name} period.")
                st.markdown("---")
                continue
            available_portfolio_assets = [t for t in weights.index if t in crisis_prices.columns and not crisis_prices[t].isnull().all()]
            if not available_portfolio_assets:
                st.warning(f"None of your portfolio's assets existed during the {name} period.")
                st.markdown("---")
                continue
            aligned_weights = weights[available_portfolio_assets].copy()
            aligned_weights /= aligned_weights.sum()
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
        else:
            st.warning("Market regime analysis is currently unavailable due to a data issue.")


def display_questionnaire() -> Tuple[str, bool, Dict]:
    st.subheader("Please Complete Your Investor Profile")
    answers = {key: st.radio(f"**{key.replace('_', ' ')}**", options) for key, options in QUESTIONNAIRE.items()}
    score = sum(QUESTIONNAIRE[key].index(answers[key]) for key in ["Risk Tolerance", "Investment Horizon"])
    risk_profile = "Conservative" if score <= 1 else "Balanced" if score <= 3 else "Aggressive"
    use_garch = st.toggle("🧠 Use ML-Enhanced Volatility Forecast (GARCH)", key="new_garch")
    if st.button("📈 Build My Portfolio", type="primary"):
        return risk_profile, use_garch, answers
    return "", False, {}

def run_portfolio_creation(risk_profile: str, use_garch: bool, profile_answers: Dict) -> Dict | None:
    spinner_msg = f"Building your '{risk_profile}' portfolio"
    spinner_msg += " with GARCH volatility..." if use_garch else "..."
    with st.spinner(spinner_msg):
        prices = get_price_data(MASTER_ASSET_LIST, "2018-01-01")
        if prices.empty: 
            st.error("Could not download market data to create the portfolio.")
            return None
        returns = prices.pct_change().dropna()
        if returns.empty:
            st.error("Could not calculate returns from market data.")
            return None
        weights = optimize_portfolio(returns, risk_profile, use_garch)
        if weights is not None:
            metrics = analyze_portfolio(weights, returns)
            return {
                "risk_profile": risk_profile, "weights": {k: v for k, v in weights.items() if v > 0},
                "metrics": metrics, "last_rebalanced_date": dt.date.today().isoformat(),
                "profile_answers": profile_answers, "used_garch": use_garch
            }
    return None

# ======================================================================================
# MAIN APP FLOW
# ======================================================================================

def main():
    st.title("WealthGenius 🧠 AI-Powered Investment Advisor")
    st.markdown("Welcome! This tool uses Modern Portfolio Theory (MPT) and Machine Learning models to build and analyze a diversified investment portfolio.")
    st.markdown("---")
    
    if "username" not in st.session_state: st.session_state.username = None
    if "rebalance_request" not in st.session_state: st.session_state.rebalance_request = None

    all_portfolios = load_portfolios()

    if st.session_state.username is None:
        st.subheader("Create or Load Portfolio")
        with st.form("login_form"):
            username_input = st.text_input("Enter your name to begin:")
            submitted = st.form_submit_button("Begin")
            if submitted and username_input:
                st.session_state.username = username_input
                st.rerun()
        st.stop()
    
    username = st.session_state.username
    
    if st.session_state.rebalance_request:
        request = st.session_state.rebalance_request
        profile_answers = all_portfolios[username].get("profile_answers", {})
        new_portfolio = run_portfolio_creation(request["new_profile"], request["use_garch"], profile_answers)
        if new_portfolio:
            all_portfolios[username] = new_portfolio
            save_portfolios(all_portfolios)
            st.success(f"Portfolio updated to '{request['new_profile']}' profile!")
            st.balloons()
        st.session_state.rebalance_request = None
        st.rerun()

    elif username not in all_portfolios:
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

if __name__ == "__main__":
    main()
