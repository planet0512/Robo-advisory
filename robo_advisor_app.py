# robo_advisor_app.py
# A robust prototype for a robo-advisor based on Modern Portfolio Theory.

import json
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cvxpy as cp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import plotly.express as px


# ======================================================================================
# CONFIGURATION
# Define asset pools, the questionnaire, and the file for storing portfolios.
# ======================================================================================

# --- App Config ---
st.set_page_config(
    page_title="Robo-Advisor | MPT",
    page_icon="🤖",
    layout="wide",
)

# --- Persistence ---
PORTFOLIO_FILE = Path("user_portfolios.json")

# --- Asset Pools ---
# Tickers for different risk profiles.
ASSET_POOLS = {
    "Conservative": ["BND", "TIP", "LQD", "IEF", "AGG"],
    "Balanced": ["SPY", "VEA", "VWO", "BND", "VNQ"],
    "Aggressive": ["QQQ", "SPYG", "VGT", "ARKK", "IWM"],
}

# --- Risk Questionnaire ---
# Questions and their corresponding answers.
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
# Functions to fetch market data and save/load user portfolios.
# ======================================================================================

@st.cache_data(ttl=dt.timedelta(hours=24))
def get_price_data(tickers: List[str], start_date: str = "2018-01-01") -> pd.DataFrame:
    """
    Fetches daily closing prices from yfinance.
    Handles errors for failed tickers and empty results.
    """
    try:
        prices = yf.download(tickers, start=start_date, progress=False)["Close"]
        if prices.empty:
            st.error("Data download failed. No data returned from the API.")
            return pd.DataFrame()

        # Drop tickers that failed to download (all NaN columns)
        prices = prices.dropna(axis=1, how="all")
        # Forward-fill any remaining missing values
        return prices.ffill().dropna()

    except Exception as e:
        st.error(f"An error occurred while fetching data: {e}")
        return pd.DataFrame()


def load_portfolios() -> Dict[str, Any]:
    """Loads all user portfolios from the JSON file."""
    if PORTFOLIO_FILE.exists():
        try:
            return json.loads(PORTFOLIO_FILE.read_text())
        except json.JSONDecodeError:
            st.warning("Could not read portfolio file. Starting fresh.")
            return {}
    return {}


def save_portfolios(portfolios: Dict[str, Any]):
    """Saves all user portfolios to the JSON file."""
    try:
        PORTFOLIO_FILE.write_text(json.dumps(portfolios, indent=2))
    except Exception as e:
        st.error(f"Failed to save portfolios: {e}")


# ======================================================================================
# PORTFOLIO OPTIMIZATION & ANALYSIS (Mean-Variance Optimization)
# ======================================================================================

def optimize_portfolio(returns: pd.DataFrame) -> pd.Series:
    """
    Calculates optimal portfolio weights to maximize the Sharpe Ratio.
    This is the core of Modern Portfolio Theory (MPT).
    """
    mu = returns.mean().to_numpy() * 252  # Annualized expected return
    Sigma = returns.cov().to_numpy() * 252  # Annualized covariance matrix
    
    # Ensure covariance matrix is positive semidefinite for the solver
    Sigma = 0.5 * (Sigma + Sigma.T)

    w = cp.Variable(len(mu))
    risk = cp.quad_form(w, Sigma)
    
    # MVO Problem: Maximize return - 0.5 * gamma * risk
    # We use a common formulation equivalent to maximizing Sharpe Ratio.
    prob = cp.Problem(
        cp.Maximize(mu @ w - 0.5 * risk),
        [cp.sum(w) == 1, w >= 0],  # Constraints: weights sum to 1, no shorting
    )
    
    try:
        prob.solve()
        if prob.status != cp.OPTIMAL:
            raise ValueError("Solver could not find an optimal solution.")
        # Round to 4 decimal places and clean up tiny values
        weights = pd.Series(w.value, index=returns.columns)
        weights[weights < 1e-4] = 0
        weights /= weights.sum()
        return weights

    except Exception as e:
        st.error(f"Optimization failed: {e}")
        return None


def analyze_portfolio(weights: pd.Series, returns: pd.DataFrame) -> Dict[str, float]:
    """Calculates key performance metrics for a portfolio."""
    annualized_returns = returns.mean() * 252
    portfolio_return = np.sum(annualized_returns * weights)
    
    cov_matrix = returns.cov() * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Assume a risk-free rate of 0 for Sharpe Ratio calculation
    sharpe_ratio = portfolio_return / portfolio_volatility
    
    return {
        "expected_return": portfolio_return,
        "expected_volatility": portfolio_volatility,
        "sharpe_ratio": sharpe_ratio,
    }


# ======================================================================================
# UI COMPONENTS
# Functions to render different parts of the Streamlit interface.
# ======================================================================================

def display_dashboard(username: str, portfolio: Dict[str, Any]):
    """Displays the main dashboard with portfolio details and charts."""
    st.subheader(f"Welcome Back, {username.title()}!")
    
    # --- Metrics ---
    st.write(f"Your recommended portfolio is **{portfolio['risk_profile']}**.")
    cols = st.columns(3)
    cols[0].metric("Expected Annual Return", f"{portfolio['metrics']['expected_return']:.2%}")
    cols[1].metric("Expected Annual Volatility", f"{portfolio['metrics']['expected_volatility']:.2%}")
    cols[2].metric("Sharpe Ratio", f"{portfolio['metrics']['sharpe_ratio']:.2f}")

    # --- Charts ---
    weights = pd.Series(portfolio["weights"])
    fig = go.Figure(
        go.Pie(
            labels=weights.index,
            values=weights.values,
            hole=0.4,
            marker_colors=px.colors.sequential.GnBu_r,
            textinfo="label+percent"
        )
    )
    fig.update_layout(showlegend=False, title_text="Portfolio Allocation", title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)

    # --- Rebalance Button ---
    if st.button("🔄 Rebalance Portfolio"):
        # This will trigger the portfolio creation logic again on the next run
        # by temporarily removing the user from the loaded portfolios dict.
        st.session_state.rebalance_request = True
        st.rerun()


def display_questionnaire() -> str:
    """Displays the risk questionnaire and returns a risk profile on submission."""
    st.subheader("Answer a Few Questions to Build Your Portfolio")
    total_score = 0
    for i, (question, options) in enumerate(QUESTIONNAIRE.items()):
        # Use a unique key for each radio button widget
        response_index = options.index(st.radio(question, options, key=f"q_{i}"))
        total_score += response_index

    if st.button("📈 Build My Portfolio"):
        if total_score <= 2:
            return "Conservative"
        if total_score <= 5:
            return "Balanced"
        return "Aggressive"
    return ""


# ======================================================================================
# MAIN APPLICATION LOGIC
# ======================================================================================

def main():
    """Controls the main application flow."""
    st.title("🤖 Automated Portfolio Advisor")

    all_portfolios = load_portfolios()
    
    username = st.text_input("Please enter your name to begin:", key="username_input")
    if not username:
        st.info("Enter a name to load or create your investment portfolio.")
        st.stop()

    # --- Rebalance Logic ---
    # If a rebalance was requested, act as if the user is new to trigger recreation.
    if st.session_state.get("rebalance_request"):
        st.session_state.rebalance_request = False # Reset the flag
        user_portfolio = all_portfolios.pop(username, None)
        # Fall through to the 'new user' logic below

    # --- Main Flow ---
    if username not in all_portfolios:
        risk_profile = display_questionnaire()
        
        if risk_profile:
            with st.spinner(f"Building your '{risk_profile}' portfolio..."):
                assets = ASSET_POOLS[risk_profile]
                prices = get_price_data(assets)
                
                if prices.empty:
                    st.error("Could not build portfolio due to a data fetching issue.")
                    st.stop()
                
                returns = prices.pct_change().dropna()
                weights = optimize_portfolio(returns)
                
                if weights is not None:
                    metrics = analyze_portfolio(weights, returns)
                    
                    # Save the new portfolio
                    all_portfolios[username] = {
                        "risk_profile": risk_profile,
                        "weights": weights.to_dict(),
                        "metrics": metrics,
                        "created_date": dt.date.today().isoformat(),
                    }
                    save_portfolios(all_portfolios)
                    st.success("Your portfolio has been created!")
                    # Short delay and rerun to show the dashboard
                    st.balloons()
                    st.rerun()

    else:
        # Existing user: show the dashboard
        display_dashboard(username, all_portfolios[username])


if __name__ == "__main__":
    main()
