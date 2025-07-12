# robo_advisor_app.py
# An enhanced robo-advisor with MPT optimization, Monte Carlo, and GARCH ML model.

import json
import datetime as dt
from pathlib import Path
from typing import Dict, List, Any, Tuple

# <<< ML FEATURE: Add arch for GARCH models >>>
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
    page_title="Robo-Advisor | MPT+ML",
    page_icon="🤖",
    layout="wide",
)

PORTFOLIO_FILE = Path("user_portfolios.json")

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
        "Buy more while prices are low.",
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

@st.cache_data(ttl=dt.timedelta(hours=24))
def get_price_data(tickers: List[str], start_date: str = "2018-01-01") -> pd.DataFrame:
    try:
        prices = yf.download(tickers, start=start_date, progress=False)["Close"]
        if prices.empty:
            st.error("Data download failed. No data returned from the API.")
            return pd.DataFrame()
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
# <<< ML FEATURE: GARCH VOLATILITY FORECASTING >>>
# ======================================================================================

def forecast_covariance_garch(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Forecasts the covariance matrix using a GARCH(1,1) model for each asset's
    volatility and the historical correlation of their residuals.
    """
    # Standardize returns by multiplying by 100, a common practice for GARCH
    std_returns = returns * 100
    
    # Fit GARCH model to each asset and get standardized residuals
    residuals = []
    variances = []
    for asset in std_returns.columns:
        model = arch_model(std_returns[asset], p=1, q=1, vol='Garch', dist='Normal')
        res = model.fit(disp="off")
        residuals.append(res.std_resid)
        # Forecast 1-day ahead variance and take the value
        forecast = res.forecast(horizon=1)
        variances.append(forecast.variance.iloc[-1, 0])
    
    # Calculate correlation of residuals
    residual_df = pd.concat(residuals, axis=1)
    corr_matrix = residual_df.corr()
    
    # Construct the forecasted covariance matrix
    # Undo standardization by dividing variances by 100^2
    variances = np.array(variances) / (100**2)
    diag_vol = np.diag(np.sqrt(variances))
    cov_matrix = diag_vol @ corr_matrix @ diag_vol
    
    # Annualize the daily covariance matrix
    return pd.DataFrame(cov_matrix * 252, index=returns.columns, columns=returns.columns)


# ======================================================================================
# PORTFOLIO OPTIMIZATION & ANALYSIS
# ======================================================================================

def optimize_portfolio(returns: pd.DataFrame, use_garch: bool = False) -> pd.Series:
    """
    Calculates optimal portfolio weights. Can use a simple historical covariance
    or an advanced GARCH-forecasted covariance.
    """
    mu = returns.mean().to_numpy() * 252

    if use_garch:
        st.toast("Using GARCH model for risk forecast...", icon="🧠")
        Sigma = forecast_covariance_garch(returns).to_numpy()
    else:
        st.toast("Using historical model for risk...", icon="📜")
        Sigma = returns.cov().to_numpy() * 252

    # <<< FIX: Sanitize numpy arrays to remove any NaN/inf values >>>
    mu = np.nan_to_num(mu)
    Sigma = np.nan_to_num(Sigma)

    try:
        # Enforce and wrap the covariance matrix for the solver
        Sigma = 0.5 * (Sigma + Sigma.T)
        P = cp.psd_wrap(Sigma)

        # Define optimization variables and problem
        w = cp.Variable(len(mu))
        risk = cp.quad_form(w, P)
        prob = cp.Problem(
            cp.Maximize(mu @ w - 0.5 * risk),
            [cp.sum(w) == 1, w >= 0]
        )

        # Solve the problem
        prob.solve(solver=cp.SCS)
        if prob.status != cp.OPTIMAL:
            raise ValueError("Solver could not find an optimal solution.")

        # Process and return the weights
        weights = pd.Series(w.value, index=returns.columns)
        weights[weights < 1e-4] = 0
        weights /= weights.sum()
        return weights

    except Exception as e:
        # This will now catch the original, informative error
        st.error(f"Optimization failed: {e}")
        return None
        
def analyze_portfolio(weights: pd.Series, returns: pd.DataFrame) -> Dict[str, float]:
    portfolio_return = np.sum(returns.mean() * 252 * weights)
    cov_matrix = returns.cov() * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility
    return {
        "expected_return": portfolio_return,
        "expected_volatility": portfolio_volatility,
        "sharpe_ratio": sharpe_ratio,
    }

# ======================================================================================
# MONTE CARLO SIMULATION
# ======================================================================================

def run_monte_carlo(
    initial_value: float, er: float, vol: float, years: int, simulations: int
) -> pd.DataFrame:
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

# ======================================================================================
# UI COMPONENTS
# ======================================================================================

def display_dashboard(username: str, portfolio: Dict[str, Any]):
    st.subheader(f"Welcome Back, {username.title()}!")
    
    st.write(f"Your recommended portfolio is **{portfolio['risk_profile']}**.")
    cols = st.columns(3)
    cols[0].metric("Expected Annual Return", f"{portfolio['metrics']['expected_return']:.2%}")
    cols[1].metric("Expected Annual Volatility", f"{portfolio['metrics']['expected_volatility']:.2%}")
    cols[2].metric("Sharpe Ratio", f"{portfolio['metrics']['sharpe_ratio']:.2f}")

    weights = pd.Series(portfolio["weights"])
    fig = go.Figure(go.Pie(
        labels=weights.index, values=weights.values, hole=0.4,
        marker_colors=px.colors.sequential.GnBu_r, textinfo="label+percent"
    ))
    fig.update_layout(showlegend=False, title_text="Portfolio Allocation", title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("📈 Future Growth Simulation")
    sim_cols = st.columns([1, 3])
    with sim_cols[0]:
        initial_investment = st.number_input("Initial Investment ($)", min_value=1000, value=10000, step=1000, format="%d")
        simulation_years = st.slider("Investment Horizon (Years)", min_value=1, max_value=30, value=10)
    
    sim_results = run_monte_carlo(initial_investment, portfolio['metrics']['expected_return'], portfolio['metrics']['expected_volatility'], simulation_years, 500)
    
    final_values = sim_results.iloc[-1]
    with sim_cols[1]:
        fig_sim = go.Figure()
        fig_sim.add_traces([go.Scatter(x=sim_results.index / 252, y=sim_results[col], mode='lines', line_color='lightgrey', showlegend=False) for col in sim_results.columns[:100]])
        fig_sim.add_traces([
            go.Scatter(x=sim_results.index / 252, y=sim_results.quantile(q, axis=1), mode='lines', line=dict(width=3), name=f'{q*100:.0f}th Percentile') for q in [0.1, 0.5, 0.9]
        ])
        fig_sim.update_layout(title_text=f"Projected Growth of ${initial_investment:,.0f} over {simulation_years} Years", xaxis_title="Years", yaxis_title="Portfolio Value ($)", yaxis_tickformat="$,.0f")
        st.plotly_chart(fig_sim, use_container_width=True)

    st.info(f"After **{simulation_years} years**, your portfolio has a projected median value of **${final_values.median():,.0f}**.")
    st.caption(f"Plausible range: **${final_values.quantile(0.1):,.0f}** to **${final_values.quantile(0.9):,.0f}**.")

def display_questionnaire() -> Tuple[str, bool]:
    st.subheader("Answer a Few Questions to Build Your Portfolio")
    total_score = 0
    for i, (question, options) in enumerate(QUESTIONNAIRE.items()):
        response_index = options.index(st.radio(question, options, key=f"q_{i}"))
        total_score += response_index

    # <<< ML FEATURE: Add toggle to questionnaire >>>
    use_ml_model = st.toggle("🧠 Use ML-Enhanced Volatility Forecast (GARCH)", value=False, help="Uses a machine learning model to forecast risk instead of relying only on historical data. This may result in a different portfolio allocation.")

    if st.button("📈 Build My Portfolio"):
        if total_score <= 2: return "Conservative", use_ml_model
        if total_score <= 5: return "Balanced", use_ml_model
        return "Aggressive", use_ml_model
    return "", False

# ======================================================================================
# MAIN APPLICATION LOGIC
# ======================================================================================

def main():
    st.title("🤖 Automated Portfolio Advisor")
    all_portfolios = load_portfolios()
    
    username = st.text_input("Please enter your name to begin:", key="username_input")
    if not username:
        st.info("Enter a name to load or create your investment portfolio.")
        st.stop()

    if username not in all_portfolios:
        risk_profile, use_garch = display_questionnaire()
        if risk_profile:
            with st.spinner(f"Building your '{risk_profile}' portfolio..."):
                assets = ASSET_POOLS[risk_profile]
                prices = get_price_data(assets)
                if prices.empty: st.stop()
                returns = prices.pct_change().dropna()
                
                weights = optimize_portfolio(returns, use_garch=use_garch)
                
                if weights is not None:
                    metrics = analyze_portfolio(weights, returns)
                    all_portfolios[username] = {
                        "risk_profile": risk_profile,
                        "weights": weights.to_dict(),
                        "metrics": metrics,
                        "created_date": dt.date.today().isoformat(),
                        "used_garch": use_garch # Save model choice
                    }
                    save_portfolios(all_portfolios)
                    st.success("Your portfolio has been created!")
                    st.balloons()
                    st.rerun()
    else:
        display_dashboard(username, all_portfolios[username])

if __name__ == "__main__":
    main()
