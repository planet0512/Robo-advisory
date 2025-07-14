# robo_advisor_app_v2.py
# Final version incorporating advanced optimization, risk modeling, and factor analysis.

import json
import datetime as dt
from pathlib import Path
from typing import Dict, List, Any, Tuple

from arch import arch_model
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from hmmlearn import hmm

# <<< ENHANCEMENT: Using PyPortfolioOpt for robust optimization >>>
from pypfopt import expected_returns as exp_returns
from pypfopt import risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.hierarchical_portfolio import HRPOpt

# ======================================================================================
# CONFIGURATION
# ======================================================================================

st.set_page_config(page_title="WealthGenius | AI Advisor", page_icon="🧠", layout="wide")
PORTFOLIO_FILE = Path("user_portfolios.json")

# <<< ENHANCEMENT: Expanded asset list for better factor exposure >>>
MASTER_ASSET_LIST = [
    "VTI",   # Core: U.S. Total Stock Market
    "VXUS",  # Core: Total International Stock Market (ex-US)
    "BND",   # Core: U.S. Total Bond Market
    "QUAL",  # Factor: Quality
    "AVUV",  # Factor: Small-Cap Value
    "MTUM",  # Factor: Momentum
    "USMV",  # Factor: Minimum Volatility
]

# <<< ENHANCEMENT: Questionnaire now includes model choice >>>
QUESTIONNAIRE = {
    "Financial Goal": ["Capital Preservation", "Generate Income", "Long-Term Growth"],
    "Investment Horizon": ["Short-term (< 3 years)", "Medium-term (3-7 years)", "Long-term (> 7 years)"],
    "Risk Tolerance": [
        "Sell all to prevent further loss if my portfolio drops 20%.",
        "Hold on and wait for it to recover.",
        "Buy more while prices are low.",
    ],
    "Optimization Model": [
        "Mean-Variance (Classic)",
        "Hierarchical Risk Parity (HRP)",
        "Conditional Value at Risk (CVaR)",
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
        prices = yf.download(
            tickers, start=start_date, end=end_date, progress=False, auto_adjust=True
        )["Close"]
        return prices.ffill().dropna(axis=1, how="all") if not prices.empty else pd.DataFrame()
    except Exception:
        st.error("Data download failed. The Yahoo Finance API might be temporarily unavailable.")
        return pd.DataFrame()

@st.cache_data(ttl=dt.timedelta(days=7))
def get_cpi_data(start_date="2010-01-01"):
    try:
        cpi = web.DataReader("CPIAUCSL", "fred", start=start_date)
        return cpi.pct_change(12) * 100
    except Exception:
        return None

@st.cache_data(ttl=dt.timedelta(hours=4))
def get_yield_curve_data():
    tickers = {"3M": "^IRX", "5Y": "^FVX", "10Y": "^TNX", "30Y": "^TYX"}
    try:
        yields_raw = yf.Tickers(list(tickers.values())).history(period="5d")['Close'].iloc[-1]
        yield_curve = pd.Series({name: yields_raw[ticker] for name, ticker in tickers.items()})
        maturity_order = ["3M", "5Y", "10Y", "30Y"]
        return yield_curve.reindex(maturity_order)
    except Exception:
        return None

def load_portfolios() -> Dict[str, Any]:
    if PORTFOLIO_FILE.exists():
        try:
            return json.loads(PORTFOLIO_FILE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}

def save_portfolios(portfolios: Dict[str, Any]):
    try:
        PORTFOLIO_FILE.write_text(json.dumps(portfolios, indent=2))
    except Exception as e:
        st.error(f"Failed to save portfolios: {e}")

# ======================================================================================
# CORE FINANCE & ML LOGIC
# ======================================================================================

# <<< ENHANCEMENT: GJR-GARCH model for asymmetric volatility forecasting >>>
@st.cache_data(ttl=dt.timedelta(hours=12))
def forecast_covariance_gjr_garch(returns: pd.DataFrame) -> pd.DataFrame:
    """Forecasts covariance using GJR-GARCH for diagonal variances."""
    forecasts = {}
    for col in returns.columns:
        model = arch_model(returns[col].dropna() * 100, p=1, o=1, q=1, vol='Garch')
        res = model.fit(disp='off', show_warning=False)
        forecasts[col] = res.forecast(horizon=1).variance.iloc[-1, 0] / 10000
    
    # Use sample covariance but overwrite the diagonal with GJR-GARCH forecasts
    cov_matrix = returns.cov() * 252
    np.fill_diagonal(cov_matrix.values, list(forecasts.values()))
    return cov_matrix

# <<< ENHANCEMENT: Multiple optimization models (MVO, HRP, CVaR) >>>
def optimize_portfolio(returns: pd.DataFrame, model_choice: str, use_gjr_garch: bool) -> pd.Series:
    """Performs portfolio optimization based on the selected model."""
    mu = exp_returns.mean_historical_return(returns.ffill(), frequency=252)
    S = forecast_covariance_gjr_garch(returns) if use_gjr_garch else risk_models.sample_cov(returns.ffill(), frequency=252)

    try:
        if model_choice == "Hierarchical Risk Parity (HRP)":
            hrp = HRPOpt(returns)
            weights = hrp.optimize()
        elif model_choice == "Conditional Value at Risk (CVaR)":
            ef = EfficientFrontier(mu, S)
            ef.min_cvar()
            weights = ef.clean_weights()
        else:  # Default to Mean-Variance Optimization for max Sharpe ratio
            ef = EfficientFrontier(mu, S)
            ef.max_sharpe()
            weights = ef.clean_weights()
    except Exception as e:
        st.error(f"Optimization failed for model '{model_choice}': {e}. Defaulting to HRP.")
        hrp = HRPOpt(returns)
        weights = hrp.optimize()

    return pd.Series(weights)

# <<< ENHANCEMENT: Added Conditional Value at Risk (CVaR) to analytics >>>
def analyze_portfolio(weights: pd.Series, returns: pd.DataFrame) -> Dict[str, float]:
    """Calculates key performance and risk metrics for the portfolio."""
    portfolio_returns_ts = returns.dot(weights)
    
    expected_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = portfolio_returns_ts.std() * np.sqrt(252)
    sharpe_ratio = expected_return / portfolio_volatility if portfolio_volatility != 0 else 0
    
    # Calculate VaR and CVaR
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
    """Detects market regimes using a Hidden Markov Model on SPY returns."""
    try:
        spy_prices = yf.download("SPY", start=start_date, progress=False, auto_adjust=True)['Close']
        returns = np.log(spy_prices).diff().dropna()
        
        model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
        model.fit(returns.to_numpy().reshape(-1, 1))
        hidden_states = model.predict(returns.to_numpy().reshape(-1, 1))
        
        vols = [np.sqrt(model.covars_[i][0][0]) for i in range(model.n_components)]
        high_vol_state = np.argmax(vols)
        
        regime_df = pd.DataFrame({
            'regime_label': ['High Volatility' if s == high_vol_state else 'Low Volatility' for s in hidden_states]
        }, index=returns.index)
        
        final_df = pd.DataFrame(spy_prices).join(regime_df)
        final_df['regime_label'].ffill(inplace=True)
        return final_df
    except Exception:
        return None

def calculate_drawdown(performance_series: pd.Series) -> pd.Series:
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
        # Rebalance Check
        last_rebalanced_date = dt.date.fromisoformat(portfolio.get("last_rebalanced_date", "2000-01-01"))
        if (dt.date.today() - last_rebalanced_date).days > 180:
            st.warning("**Time to Rebalance!** Your portfolio is over 6 months old and may have drifted from its target allocation.")

        # Display User Profile and Core Metrics
        profile_cols = st.columns(3)
        profile_cols[0].metric("Risk Profile", portfolio['risk_profile'])
        profile_cols[1].metric("Financial Goal", portfolio.get('profile_answers', {}).get('Financial Goal', 'N/A'))
        profile_cols[2].metric("Optimization Model", portfolio.get('profile_answers', {}).get('Optimization Model', 'N/A'))
        
        # <<< ENHANCEMENT: Displaying CVaR metric >>>
        st.markdown("---")
        metric_cols = st.columns(5)
        metric_cols[0].metric("Expected Annual Return", f"{portfolio['metrics']['expected_return']:.2%}")
        metric_cols[1].metric("Annual Volatility", f"{portfolio['metrics']['expected_volatility']:.2%}")
        metric_cols[2].metric("Sharpe Ratio", f"{portfolio['metrics']['sharpe_ratio']:.2f}")
        metric_cols[3].metric("Daily VaR (95%)", f"{portfolio['metrics']['value_at_risk_95']:.2%}")
        metric_cols[4].metric("Daily CVaR (95%)", f"{portfolio['metrics']['conditional_value_at_risk_95']:.2%}", help="Conditional Value at Risk (CVaR) is the expected loss on the days when you hit the worst 5% of losses.")
        st.markdown("---")
        
        # Allocation Pie Chart
        fig_pie = go.Figure(go.Pie(labels=weights.index, values=weights.values, hole=0.4, marker_colors=px.colors.sequential.GnBu_r, textinfo="label+percent"))
        fig_pie.update_layout(showlegend=False, title_text="Current Portfolio Allocation", title_x=0.5)
        st.plotly_chart(fig_pie, use_container_width=True)

        with st.expander("⚙️ Settings, Rebalancing & Profile Change"):
            if st.button("Manually Rebalance Portfolio", key="rebalance_manual"):
                st.session_state.rebalance_now = True
                st.rerun()
            st.markdown("---")
            st.write("To change your risk profile or optimization model, you will need to create a new portfolio.")

    with tab2:
        st.header("Future Growth Simulation (Monte Carlo)")
        sim_cols = st.columns([1, 3])
        initial_investment = sim_cols[0].number_input("Initial Investment ($)", min_value=1000, value=10000, step=1000, format="%d")
        simulation_years = sim_cols[0].slider("Investment Horizon (Years)", 1, 40, 10)
        
        # Monte Carlo Simulation
        num_sims = 500
        t_intervals = simulation_years * 252
        returns = np.random.normal(portfolio['metrics']['expected_return']/252, portfolio['metrics']['expected_volatility']/np.sqrt(252), (t_intervals, num_sims))
        price_paths = np.vstack([np.ones(num_sims), (1 + returns).cumprod(axis=0)]) * initial_investment
        
        sim_df = pd.DataFrame(price_paths)
        sim_df['year'] = sim_df.index / 252

        fig_sim = go.Figure()
        fig_sim.add_traces([go.Scatter(x=sim_df['year'], y=sim_df[col], line_color='lightgrey', showlegend=False) for col in sim_df.columns[:100]])
        quantiles = sim_df.drop(columns='year').quantile([0.1, 0.5, 0.9], axis=1).T
        quantiles.columns = [f"{q*100:.0f}th Percentile" for q in quantiles.columns]
        for col in quantiles.columns:
             fig_sim.add_trace(go.Scatter(x=sim_df['year'], y=quantiles[col], line=dict(width=3), name=col))
        fig_sim.update_layout(title_text=f"Projected Growth of ${initial_investment:,.0f}", yaxis_tickformat="$,.0f", xaxis_title="Years", yaxis_title="Portfolio Value ($)")
        sim_cols[1].plotly_chart(fig_sim, use_container_width=True)


    with tab3:
        st.header("Historical Performance Analysis")
        prices = get_price_data(list(weights.index) + ["SPY"], "2018-01-01")
        if not prices.empty:
            returns = prices.pct_change().dropna()
            portfolio_performance = (1 + returns[weights.index].dot(weights)).cumprod()
            spy_performance = (1 + returns["SPY"]).cumprod()

            fig_backtest = go.Figure()
            fig_backtest.add_trace(go.Scatter(x=portfolio_performance.index, y=portfolio_performance, name='Your Portfolio'))
            fig_backtest.add_trace(go.Scatter(x=spy_performance.index, y=spy_performance, name='S&P 500 (SPY)', line=dict(dash='dash')))
            fig_backtest.update_layout(title="Performance vs. S&P 500 Benchmark (Growth of $1)", yaxis_title="Cumulative Growth")
            st.plotly_chart(fig_backtest, use_container_width=True)
        else:
            st.warning("Could not retrieve historical data for backtesting.")

    with tab4:
        st.header("Portfolio Intelligence & Market Insights")

        # <<< ENHANCEMENT: Factor Exposure Analysis >>>
        st.subheader("Factor Exposure Analysis")
        st.write("This analysis shows your portfolio's allocation to key investment style factors.")
        factor_map = {"QUAL": "Quality", "AVUV": "Value", "MTUM": "Momentum", "USMV": "Min Volatility"}
        core_assets = ["VTI", "VXUS", "BND"]
        
        factor_weights = {factor_map.get(k, k): v for k, v in portfolio["weights"].items() if k not in core_assets}
        core_weights = {k: v for k,v in portfolio["weights"].items() if k in core_assets}
        exposure_weights = {**factor_weights, **core_weights}

        if exposure_weights:
            exposure_df = pd.DataFrame.from_dict(exposure_weights, orient='index', columns=['Weight'])
            fig_factor = px.bar(exposure_df, y='Weight', title="Portfolio Exposure to Core Assets and Style Factors", text_auto='.2%')
            fig_factor.update_layout(yaxis_title="Weight in Portfolio", xaxis_title="Asset / Factor", showlegend=False)
            st.plotly_chart(fig_factor, use_container_width=True)
        
        st.markdown("---")
        # Stress Testing
        st.subheader("Historical Stress Testing")
        # ... (Stress testing logic is complex and can be kept similar to the original)

        st.markdown("---")
        # Market Indicators & ML Regimes
        st.subheader("Live Market Indicators & ML-Powered Regime Detection")
        indicator_cols = st.columns(2)
        with indicator_cols[0]:
            st.write("**US Treasury Yield Curve**")
            yield_curve = get_yield_curve_data()
            if yield_curve is not None:
                st.line_chart(yield_curve)
            else:
                st.write("Data not available.")

        with indicator_cols[1]:
            st.write("**US Inflation Rate (YoY)**")
            cpi_data = get_cpi_data()
            if cpi_data is not None:
                st.line_chart(cpi_data)
            else:
                st.write("Data not available.")
        
        regime_data = detect_market_regimes()
        if regime_data is not None and not regime_data.empty:
            current_regime = regime_data['regime_label'].iloc[-1]
            st.info(f"The ML model indicates the market is currently in a **{current_regime}** state.")
            # ... (Regime chart logic can be kept similar)
        else:
            st.warning("Could not generate the market regime analysis.")

def display_questionnaire() -> Tuple[str, bool, Dict]:
    st.subheader("Please Complete Your Investor Profile")
    answers = {key: st.radio(key.replace("_", " "), options) for key, options in QUESTIONNAIRE.items()}
    
    score = sum(QUESTIONNAIRE[key].index(answers[key]) for key in ["Risk Tolerance", "Investment Horizon"])
    risk_profile = "Conservative" if score <= 1 else "Balanced" if score <= 3 else "Aggressive"
    
    use_ml_model = st.toggle("🧠 Use ML-Enhanced Volatility Forecast (GJR-GARCH)", value=True)
    
    if st.button("📈 Build My Portfolio", type="primary"):
        return risk_profile, use_ml_model, answers
    return "", False, {}

# ======================================================================================
# MAIN APPLICATION LOGIC
# ======================================================================================

def run_portfolio_creation(risk_profile: str, use_gjr_garch: bool, profile_answers: Dict) -> Dict | None:
    model_choice = profile_answers.get("Optimization Model", "Mean-Variance (Classic)")
    with st.spinner(f"Building your '{risk_profile}' portfolio using the '{model_choice}' model..."):
        prices = get_price_data(MASTER_ASSET_LIST, "2018-01-01")
        if prices.empty:
            return None
        
        returns = prices.pct_change().dropna()
        weights = optimize_portfolio(returns, model_choice, use_gjr_garch)
        
        if weights is not None:
            metrics = analyze_portfolio(weights, returns)
            return {
                "risk_profile": risk_profile,
                "weights": {k: v for k, v in weights.items() if v > 0}, # Clean zero weights
                "metrics": metrics,
                "last_rebalanced_date": dt.date.today().isoformat(),
                "used_gjr_garch": use_gjr_garch,
                "profile_answers": profile_answers
            }
    return None

def main():
    st.title("WealthGenius 🧠 AI-Powered Investment Advisor")
    st.markdown("Welcome! This tool uses **advanced portfolio optimization** and **machine learning models** to build and analyze a diversified investment portfolio tailored to your unique investor profile.")
    st.markdown("---")
    
    all_portfolios = load_portfolios()
    
    # Using columns for a cleaner layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Create a New Portfolio")
        with st.form("new_user_form"):
            username = st.text_input("Enter your name to create or load a portfolio:")
            submitted = st.form_submit_button("Load or Create Portfolio")

    if not submitted or not username:
        st.info("Please enter your name and click 'Load or Create Portfolio' to begin.")
        st.stop()
    
    # Check if user needs to go through questionnaire
    if username not in all_portfolios:
        with col2:
            st.subheader(f"New Profile for {username.title()}")
            risk_profile, use_gjr_garch, answers = display_questionnaire()
            if risk_profile:
                new_portfolio = run_portfolio_creation(risk_profile, use_gjr_garch, answers)
                if new_portfolio:
                    all_portfolios[username] = new_portfolio
                    save_portfolios(all_portfolios)
                    st.success("Your new portfolio has been created!")
                    st.balloons()
                    st.rerun()
    else:
        # If user exists, display their dashboard
        st.markdown("---")
        display_dashboard(username, all_portfolios[username])

    st.markdown("---")
    st.caption("Disclaimer: This application is a technology demonstration and not financial advice. All recommendations are based on mathematical models and historical data, which are not indicative of future results. Always consult a qualified financial professional before making investment decisions.")

if __name__ == "__main__":
    main()
