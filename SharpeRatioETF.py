"""
Sharpe Ratio Calculator for ETFs and ETF Portfolios
=====================================================
Calcola lo Sharpe Ratio di singoli ETF e di portafogli composti da ETF,
usando dati storici scaricati da Yahoo Finance.

Dipendenze: pip install yfinance numpy pandas matplotlib
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Optional


# ─────────────────────────────────────────────
#  Core: download & compute
# ─────────────────────────────────────────────

def download_prices(
    tickers: list[str],
    start: str = "2020-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Scarica i prezzi adjusted close da Yahoo Finance."""
    end = end or datetime.today().strftime("%Y-%m-%d")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    return data.dropna()


def compute_returns(prices: pd.DataFrame, freq: str = "daily") -> pd.DataFrame:
    """Calcola i rendimenti logaritmici (daily o monthly)."""
    if freq == "monthly":
        prices = prices.resample("ME").last()
    return np.log(prices / prices.shift(1)).dropna()


def sharpe_ratio(
    returns: pd.Series | pd.DataFrame,
    risk_free_annual: float = 0.0,
    periods_per_year: int = 252,
) -> float | pd.Series:
    """
    Sharpe Ratio annualizzato.

    SR = (mean_return - rf_per_period) / std_return * sqrt(N)

    Parameters
    ----------
    returns : rendimenti periodici (daily/monthly)
    risk_free_annual : tasso risk-free annuo (es. 0.05 = 5%)
    periods_per_year : 252 per daily, 12 per monthly
    """
    rf_per_period = risk_free_annual / periods_per_year
    excess = returns - rf_per_period
    sr = excess.mean() / excess.std() * np.sqrt(periods_per_year)
    return sr


def portfolio_returns(
    returns: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """
    Rendimento di un portafoglio dato un dict {ticker: peso}.
    I pesi vengono normalizzati automaticamente a somma 1.
    """
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()  # normalizza
    return returns[tickers].dot(w)


# ─────────────────────────────────────────────
#  Analisi singoli ETF
# ─────────────────────────────────────────────

def analyze_etfs(
    tickers: list[str],
    start: str = "2020-01-01",
    end: Optional[str] = None,
    risk_free_annual: float = 0.0,
    freq: str = "daily",
) -> pd.DataFrame:
    """
    Analisi completa di una lista di ETF: rendimento, volatilità, Sharpe.
    """
    prices = download_prices(tickers, start, end)
    rets = compute_returns(prices, freq)

    periods = 252 if freq == "daily" else 12

    results = []
    for t in tickers:
        r = rets[t]
        ann_ret = r.mean() * periods
        ann_vol = r.std() * np.sqrt(periods)
        sr = sharpe_ratio(r, risk_free_annual, periods)
        max_dd = _max_drawdown(prices[t])
        results.append({
            "Ticker": t,
            "Ann. Return (%)": round(ann_ret * 100, 2),
            "Ann. Volatility (%)": round(ann_vol * 100, 2),
            "Sharpe Ratio": round(sr, 3),
            "Max Drawdown (%)": round(max_dd * 100, 2),
        })

    return pd.DataFrame(results).set_index("Ticker")


def _max_drawdown(prices: pd.Series) -> float:
    """Maximum drawdown da una serie di prezzi."""
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    return drawdown.min()


# ─────────────────────────────────────────────
#  Analisi portafogli
# ─────────────────────────────────────────────

def analyze_portfolio(
    weights: dict[str, float],
    start: str = "2020-01-01",
    end: Optional[str] = None,
    risk_free_annual: float = 0.0,
    freq: str = "daily",
    label: str = "Portfolio",
) -> dict:
    """
    Analisi completa di un portafoglio di ETF.
    weights: {ticker: peso} — verranno normalizzati a 1.
    """
    tickers = list(weights.keys())
    prices = download_prices(tickers, start, end)
    rets = compute_returns(prices, freq)
    periods = 252 if freq == "daily" else 12

    port_rets = portfolio_returns(rets, weights)
    ann_ret = port_rets.mean() * periods
    ann_vol = port_rets.std() * np.sqrt(periods)
    sr = sharpe_ratio(port_rets, risk_free_annual, periods)

    # equity curve per max drawdown
    equity = (1 + port_rets).cumprod()
    max_dd = _max_drawdown(equity)

    # pesi normalizzati
    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()
    norm_weights = {t: round(wi, 4) for t, wi in zip(tickers, w)}

    return {
        "label": label,
        "weights": norm_weights,
        "ann_return": round(ann_ret * 100, 2),
        "ann_volatility": round(ann_vol * 100, 2),
        "sharpe_ratio": round(sr, 3),
        "max_drawdown": round(max_dd * 100, 2),
        "returns_series": port_rets,
        "equity_curve": equity,
    }


def compare_portfolios(
    portfolios: list[dict[str, float]],
    labels: list[str],
    start: str = "2020-01-01",
    end: Optional[str] = None,
    risk_free_annual: float = 0.0,
    freq: str = "daily",
) -> pd.DataFrame:
    """Confronta più portafogli side-by-side."""
    results = []
    for w, label in zip(portfolios, labels):
        info = analyze_portfolio(w, start, end, risk_free_annual, freq, label)
        results.append({
            "Portfolio": info["label"],
            "Weights": info["weights"],
            "Ann. Return (%)": info["ann_return"],
            "Ann. Volatility (%)": info["ann_volatility"],
            "Sharpe Ratio": info["sharpe_ratio"],
            "Max Drawdown (%)": info["max_drawdown"],
        })
    return pd.DataFrame(results).set_index("Portfolio")


# ─────────────────────────────────────────────
#  Visualizzazione
# ─────────────────────────────────────────────

def plot_equity_curves(
    portfolios_info: list[dict],
    title: str = "Equity Curves",
    save_path: Optional[str] = None,
):
    """Plotta le equity curve di più portafogli."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    for p in portfolios_info:
        ax.plot(p["equity_curve"].index, p["equity_curve"].values,
                label=f'{p["label"]} (SR={p["sharpe_ratio"]:.2f})')

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Crescita di 1€")
    ax.set_xlabel("")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Grafico salvato in: {save_path}")
    plt.show()


def plot_rolling_sharpe(
    portfolios_info: list[dict],
    window: int = 126,
    risk_free_annual: float = 0.0,
    save_path: Optional[str] = None,
):
    """Rolling Sharpe Ratio (default: finestra 6 mesi)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    for p in portfolios_info:
        rets = p["returns_series"]
        rf_daily = risk_free_annual / 252
        excess = rets - rf_daily
        rolling_sr = (
            excess.rolling(window).mean()
            / excess.rolling(window).std()
            * np.sqrt(252)
        )
        ax.plot(rolling_sr.index, rolling_sr.values, label=p["label"])

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(f"Rolling Sharpe Ratio ({window}d window)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────
#  Esempio di utilizzo
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # --- Config ---
    START = "2020-01-01"
    RISK_FREE = 0.045  # ~4.5% annuo (proxy T-bill attuale)

    # === 1. Analisi singoli ETF ===
    etfs = ["SPY", "QQQ", "IWM", "EFA", "AGG", "GLD", "VNQ"]

    print("=" * 60)
    print("  SHARPE RATIO — SINGOLI ETF")
    print("=" * 60)
    df = analyze_etfs(etfs, start=START, risk_free_annual=RISK_FREE)
    print(df.to_string())
    print()

    # === 2. Confronto portafogli ===
    portfolios = [
        # Classico 60/40
        {"SPY": 0.60, "AGG": 0.40},
        # Growth-tilted
        {"QQQ": 0.40, "SPY": 0.30, "IWM": 0.15, "EFA": 0.15},
        # All-weather style
        {"SPY": 0.30, "AGG": 0.40, "GLD": 0.15, "VNQ": 0.15},
    ]
    labels = ["60/40 Classic", "Growth Tilt", "All-Weather"]

    print("=" * 60)
    print("  SHARPE RATIO — PORTAFOGLI")
    print("=" * 60)
    comp = compare_portfolios(portfolios, labels, start=START, risk_free_annual=RISK_FREE)
    print(comp.to_string())
    print()

    # === 3. Visualizzazione ===
    infos = [
        analyze_portfolio(w, START, risk_free_annual=RISK_FREE, label=l)
        for w, l in zip(portfolios, labels)
    ]
    plot_equity_curves(infos, title="Confronto Portafogli ETF", save_path="equity_curves.png")
    plot_rolling_sharpe(infos, risk_free_annual=RISK_FREE, save_path="rolling_sharpe.png")