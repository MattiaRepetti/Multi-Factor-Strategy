"""
Dynamic Portfolio Framework — Multi-Strategy Backtesting
==========================================================
Implementa e confronta strategie di allocazione dinamica su ETF:

1. Buy & Hold (benchmark)
2. Equal Weight Rebalanced
3. Momentum Rotazionale (top-N per 12m momentum)
4. Dual Momentum (Antonacci-style: relative + absolute momentum)
5. Trend Following + Momentum (SMA filter + rotazione)
6. Risk Parity Dinamico
7. Ensemble (media segnali delle strategie attive)

Backtesting con:
- Walk-forward (no look-ahead bias)
- Transaction costs
- Rolling Sharpe analysis
- Monte Carlo bootstrap per confidence intervals
- Drawdown analysis

Dipendenze:
    pip install yfinance numpy pandas matplotlib scipy
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════
#  1. DATA LAYER
# ═══════════════════════════════════════════════════════════════

# Universo ETF diversificato per asset class
DEFAULT_UNIVERSE = {
    # Equity
    "SPY":  "S&P 500",
    "QQQ":  "Nasdaq 100",
    "EFA":  "Int'l Developed",
    "EEM":  "Emerging Markets",
    "IWM":  "US Small Cap",
    # Fixed Income
    "AGG":  "US Aggregate Bond",
    "TLT":  "20+ Year Treasury",
    "TIP":  "TIPS (Inflation)",
    # Alternatives
    "GLD":  "Gold",
    "VNQ":  "REITs",
    "DBC":  "Commodities",
}

SAFE_ASSET = "SHY"  # 1-3 Year Treasury — proxy risk-off


def download_data(
    tickers: list[str],
    start: str = "2007-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Scarica adjusted close prices."""
    end = end or datetime.today().strftime("%Y-%m-%d")
    all_tickers = list(set(tickers + [SAFE_ASSET]))
    data = yf.download(all_tickers, start=start, end=end, auto_adjust=True)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data.dropna()


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Rendimenti logaritmici giornalieri."""
    return np.log(prices / prices.shift(1)).dropna()


def to_monthly(prices: pd.DataFrame) -> pd.DataFrame:
    """Resample a fine mese."""
    return prices.resample("ME").last().dropna()


# ═══════════════════════════════════════════════════════════════
#  2. STRATEGIE
# ═══════════════════════════════════════════════════════════════

class Strategy:
    """Classe base per le strategie."""

    def __init__(self, name: str):
        self.name = name

    def get_weights(self, prices: pd.DataFrame, date: pd.Timestamp,
                    lookback_prices: pd.DataFrame) -> dict:
        """Ritorna {ticker: weight} per la data corrente."""
        raise NotImplementedError


class BuyAndHold(Strategy):
    """Benchmark: 60% SPY / 40% AGG."""

    def __init__(self):
        super().__init__("Buy & Hold 60/40")

    def get_weights(self, prices, date, lookback_prices):
        return {"SPY": 0.60, "AGG": 0.40}


class EqualWeight(Strategy):
    """Equal weight mensile su tutto l'universo."""

    def __init__(self, tickers: list[str]):
        super().__init__("Equal Weight")
        self.tickers = tickers

    def get_weights(self, prices, date, lookback_prices):
        n = len(self.tickers)
        return {t: 1.0 / n for t in self.tickers}


class MomentumRotation(Strategy):
    """
    Rotazione momentum: seleziona i top-N ETF per rendimento
    negli ultimi `lookback` mesi. Ribilanciamento mensile.
    """

    def __init__(self, tickers: list[str], top_n: int = 3,
                 lookback_months: int = 12):
        super().__init__(f"Momentum Top-{top_n} ({lookback_months}m)")
        self.tickers = tickers
        self.top_n = top_n
        self.lookback = lookback_months

    def get_weights(self, prices, date, lookback_prices):
        if len(lookback_prices) < self.lookback:
            # Non abbastanza dati, equal weight
            n = len(self.tickers)
            return {t: 1.0 / n for t in self.tickers}

        # Rendimento totale sui lookback mesi
        available = [t for t in self.tickers if t in lookback_prices.columns]
        returns = {}
        for t in available:
            series = lookback_prices[t].dropna()
            if len(series) >= 2:
                returns[t] = series.iloc[-1] / series.iloc[-self.lookback] - 1

        # Seleziona top N
        sorted_tickers = sorted(returns, key=returns.get, reverse=True)[:self.top_n]
        w = 1.0 / self.top_n
        return {t: w for t in sorted_tickers}


class DualMomentum(Strategy):
    """
    Dual Momentum (ispirato a Antonacci):
    - Relative momentum: confronta US equity vs Int'l equity (12m)
    - Absolute momentum: se il winner ha rendimento < cash (SHY), vai risk-off
    - Risk-off = AGG (bonds)
    """

    def __init__(self, us_equity: str = "SPY", intl_equity: str = "EFA",
                 bond: str = "AGG", lookback_months: int = 12):
        super().__init__("Dual Momentum (Antonacci)")
        self.us = us_equity
        self.intl = intl_equity
        self.bond = bond
        self.lookback = lookback_months

    def get_weights(self, prices, date, lookback_prices):
        if len(lookback_prices) < self.lookback:
            return {self.bond: 1.0}

        def ret(ticker):
            s = lookback_prices[ticker].dropna()
            if len(s) < 2:
                return 0
            return s.iloc[-1] / s.iloc[-min(self.lookback, len(s))] - 1

        ret_us = ret(self.us)
        ret_intl = ret(self.intl)
        ret_cash = ret(SAFE_ASSET)

        # Step 1: relative momentum — chi è meglio tra US e Int'l?
        winner = self.us if ret_us >= ret_intl else self.intl
        winner_ret = max(ret_us, ret_intl)

        # Step 2: absolute momentum — il winner batte cash?
        if winner_ret > ret_cash:
            return {winner: 1.0}
        else:
            return {self.bond: 1.0}


class TrendMomentum(Strategy):
    """
    Trend Following + Momentum:
    - Filtra: solo ETF sopra la SMA a 200 giorni (trend positivo)
    - Tra quelli in trend, seleziona top-N per momentum 6 mesi
    - Se nessuno è in trend positivo → 100% safe asset
    """

    def __init__(self, tickers: list[str], top_n: int = 3,
                 sma_days: int = 200, mom_months: int = 6):
        super().__init__(f"Trend+Momentum Top-{top_n}")
        self.tickers = tickers
        self.top_n = top_n
        self.sma_days = sma_days
        self.mom_months = mom_months

    def get_weights(self, prices, date, lookback_prices):
        # Filtra per trend (SMA su daily prices fino a `date`)
        in_trend = []
        for t in self.tickers:
            if t not in prices.columns:
                continue
            daily = prices[t].loc[:date].dropna()
            if len(daily) < self.sma_days:
                continue
            sma = daily.iloc[-self.sma_days:].mean()
            if daily.iloc[-1] > sma:
                in_trend.append(t)

        if not in_trend:
            return {SAFE_ASSET: 1.0}

        # Momentum tra quelli in trend
        mom = {}
        for t in in_trend:
            s = lookback_prices[t].dropna()
            if len(s) >= self.mom_months:
                mom[t] = s.iloc[-1] / s.iloc[-self.mom_months] - 1

        if not mom:
            return {SAFE_ASSET: 1.0}

        top = sorted(mom, key=mom.get, reverse=True)[:self.top_n]
        w = 1.0 / len(top)
        return {t: w for t in top}


class RiskParity(Strategy):
    """
    Risk Parity Dinamico:
    Pesi inversamente proporzionali alla volatilità rolling (60 giorni).
    """

    def __init__(self, tickers: list[str], vol_window: int = 60):
        super().__init__("Risk Parity")
        self.tickers = tickers
        self.vol_window = vol_window

    def get_weights(self, prices, date, lookback_prices):
        rets = compute_returns(prices.loc[:date])
        if len(rets) < self.vol_window:
            n = len(self.tickers)
            return {t: 1.0 / n for t in self.tickers}

        recent = rets[self.tickers].iloc[-self.vol_window:]
        vols = recent.std()
        vols = vols.replace(0, np.nan).dropna()

        if vols.empty:
            n = len(self.tickers)
            return {t: 1.0 / n for t in self.tickers}

        inv_vol = 1.0 / vols
        weights = inv_vol / inv_vol.sum()
        return weights.to_dict()


class EnsembleStrategy(Strategy):
    """
    Ensemble: combina i pesi di più strategie con media semplice.
    Diversificazione dei segnali per ridurre overfitting.
    """

    def __init__(self, strategies: list[Strategy]):
        names = " + ".join([s.name.split("(")[0].strip() for s in strategies])
        super().__init__(f"Ensemble ({len(strategies)} strat.)")
        self.strategies = strategies

    def get_weights(self, prices, date, lookback_prices):
        all_weights = []
        for s in self.strategies:
            w = s.get_weights(prices, date, lookback_prices)
            all_weights.append(w)

        # Media dei pesi
        combined = {}
        for w_dict in all_weights:
            for t, w in w_dict.items():
                combined[t] = combined.get(t, 0) + w / len(all_weights)

        return combined


# ═══════════════════════════════════════════════════════════════
#  3. BACKTESTING ENGINE
# ═══════════════════════════════════════════════════════════════

def backtest(
    strategy: Strategy,
    daily_prices: pd.DataFrame,
    monthly_prices: pd.DataFrame,
    start_date: str = "2008-01-01",
    initial_capital: float = 100_000,
    transaction_cost_bps: float = 10,  # 10 bps = 0.10%
) -> dict:
    """
    Backtest walk-forward con ribilanciamento mensile.

    - Nessun look-ahead bias: ogni decisione usa solo dati passati
    - Transaction costs applicati ad ogni cambio di peso
    - Traccia equity curve, pesi, turnover
    """
    cost_rate = transaction_cost_bps / 10_000

    # Date di ribilanciamento (fine mese)
    rebal_dates = monthly_prices.loc[start_date:].index

    portfolio_value = initial_capital
    equity_curve = []
    weights_history = []
    turnover_history = []
    current_weights = {}
    prev_date = None

    for i, date in enumerate(rebal_dates):
        # Lookback: prezzi mensili fino a questa data
        lookback = monthly_prices.loc[:date]

        # Ottieni nuovi pesi dalla strategia
        new_weights = strategy.get_weights(daily_prices, date, lookback)

        # Calcola turnover
        all_tickers = set(list(current_weights.keys()) + list(new_weights.keys()))
        turnover = sum(
            abs(new_weights.get(t, 0) - current_weights.get(t, 0))
            for t in all_tickers
        ) / 2  # one-way turnover

        # Applica costi di transazione
        cost = portfolio_value * turnover * cost_rate

        # Rendimento del mese successivo (se esiste)
        if i + 1 < len(rebal_dates):
            next_date = rebal_dates[i + 1]

            # Rendimenti mensili per ticker
            monthly_ret = 0
            for t, w in new_weights.items():
                if t in monthly_prices.columns:
                    p_now = monthly_prices[t].loc[date]
                    p_next = monthly_prices[t].loc[next_date]
                    if pd.notna(p_now) and pd.notna(p_next) and p_now > 0:
                        monthly_ret += w * (p_next / p_now - 1)

            portfolio_value = (portfolio_value - cost) * (1 + monthly_ret)

        equity_curve.append({"date": date, "value": portfolio_value})
        weights_history.append({"date": date, **new_weights})
        turnover_history.append({"date": date, "turnover": turnover})
        current_weights = new_weights

    equity_df = pd.DataFrame(equity_curve).set_index("date")
    weights_df = pd.DataFrame(weights_history).set_index("date").fillna(0)
    turnover_df = pd.DataFrame(turnover_history).set_index("date")

    return {
        "strategy": strategy.name,
        "equity": equity_df,
        "weights": weights_df,
        "turnover": turnover_df,
    }


# ═══════════════════════════════════════════════════════════════
#  4. METRICHE DI PERFORMANCE
# ═══════════════════════════════════════════════════════════════

def compute_metrics(
    equity: pd.DataFrame,
    risk_free_annual: float = 0.04,
) -> dict:
    """Calcola metriche complete di performance."""
    values = equity["value"]
    returns = values.pct_change().dropna()

    # Annualized return (CAGR)
    n_years = len(returns) / 12
    if n_years <= 0 or values.iloc[0] <= 0:
        return {}
    total_return = values.iloc[-1] / values.iloc[0]
    cagr = total_return ** (1 / n_years) - 1

    # Volatility
    ann_vol = returns.std() * np.sqrt(12)

    # Sharpe
    rf_monthly = risk_free_annual / 12
    excess = returns - rf_monthly
    sharpe = excess.mean() / excess.std() * np.sqrt(12) if excess.std() > 0 else 0

    # Sortino
    downside = excess[excess < 0]
    downside_std = downside.std() * np.sqrt(12)
    sortino = (cagr - risk_free_annual) / downside_std if downside_std > 0 else 0

    # Max Drawdown
    cummax = values.cummax()
    drawdowns = (values - cummax) / cummax
    max_dd = drawdowns.min()

    # Calmar Ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate (monthly)
    win_rate = (returns > 0).sum() / len(returns)

    # Best / Worst month
    best_month = returns.max()
    worst_month = returns.min()

    return {
        "CAGR (%)": round(cagr * 100, 2),
        "Ann. Volatility (%)": round(ann_vol * 100, 2),
        "Sharpe Ratio": round(sharpe, 3),
        "Sortino Ratio": round(sortino, 3),
        "Max Drawdown (%)": round(max_dd * 100, 2),
        "Calmar Ratio": round(calmar, 3),
        "Win Rate (%)": round(win_rate * 100, 1),
        "Best Month (%)": round(best_month * 100, 2),
        "Worst Month (%)": round(worst_month * 100, 2),
        "Total Return (%)": round((total_return - 1) * 100, 2),
    }


# ═══════════════════════════════════════════════════════════════
#  5. MONTE CARLO BOOTSTRAP
# ═══════════════════════════════════════════════════════════════

def monte_carlo_sharpe(
    equity: pd.DataFrame,
    n_simulations: int = 5000,
    risk_free_annual: float = 0.04,
    confidence: float = 0.95,
) -> dict:
    """
    Bootstrap Monte Carlo per stimare confidence interval dello Sharpe Ratio.
    Ricampiona con rimpiazzamento i rendimenti mensili.
    """
    returns = equity["value"].pct_change().dropna().values
    rf_monthly = risk_free_annual / 12
    n = len(returns)

    sharpes = []
    for _ in range(n_simulations):
        sample = np.random.choice(returns, size=n, replace=True)
        excess = sample - rf_monthly
        if excess.std() > 0:
            sr = excess.mean() / excess.std() * np.sqrt(12)
            sharpes.append(sr)

    sharpes = np.array(sharpes)
    alpha = (1 - confidence) / 2

    return {
        "mean_sharpe": round(np.mean(sharpes), 3),
        "median_sharpe": round(np.median(sharpes), 3),
        "ci_lower": round(np.percentile(sharpes, alpha * 100), 3),
        "ci_upper": round(np.percentile(sharpes, (1 - alpha) * 100), 3),
        "std_sharpe": round(np.std(sharpes), 3),
        "prob_positive": round((sharpes > 0).mean() * 100, 1),
        "prob_above_1": round((sharpes > 1.0).mean() * 100, 1),
    }


# ═══════════════════════════════════════════════════════════════
#  6. VISUALIZZAZIONE
# ═══════════════════════════════════════════════════════════════

def plot_results(results: list[dict], save_prefix: str = "backtest"):
    """Dashboard completa dei risultati."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

    # --- 1. Equity Curves ---
    ax = axes[0, 0]
    for r, c in zip(results, colors):
        eq = r["equity"]["value"]
        ax.plot(eq.index, eq.values / eq.iloc[0], label=r["strategy"], color=c, linewidth=1.5)
    ax.set_title("Equity Curves (normalizzate a 1)", fontweight="bold")
    ax.set_ylabel("Crescita di 1€")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # --- 2. Drawdowns ---
    ax = axes[0, 1]
    for r, c in zip(results, colors):
        eq = r["equity"]["value"]
        dd = (eq - eq.cummax()) / eq.cummax() * 100
        ax.fill_between(dd.index, dd.values, 0, alpha=0.3, color=c, label=r["strategy"])
    ax.set_title("Drawdowns (%)", fontweight="bold")
    ax.set_ylabel("Drawdown %")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)

    # --- 3. Rolling Sharpe (12 mesi) ---
    ax = axes[1, 0]
    for r, c in zip(results, colors):
        eq = r["equity"]["value"]
        rets = eq.pct_change().dropna()
        rolling_sr = rets.rolling(12).mean() / rets.rolling(12).std() * np.sqrt(12)
        ax.plot(rolling_sr.index, rolling_sr.values, color=c, label=r["strategy"], linewidth=1.2)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=1.0, color="green", linestyle=":", alpha=0.4, label="SR = 1.0")
    ax.axhline(y=1.5, color="red", linestyle=":", alpha=0.4, label="SR = 1.5")
    ax.set_title("Rolling Sharpe Ratio (12 mesi)", fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    # --- 4. Turnover mensile ---
    ax = axes[1, 1]
    for r, c in zip(results, colors):
        if "turnover" in r:
            t = r["turnover"]["turnover"]
            ax.plot(t.index, t.values * 100, color=c, label=r["strategy"],
                    linewidth=0.8, alpha=0.7)
    ax.set_title("Turnover Mensile (%)", fontweight="bold")
    ax.set_ylabel("Turnover %")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_dashboard.png", dpi=150, bbox_inches="tight")
    print(f"Dashboard salvata: {save_prefix}_dashboard.png")
    plt.show()


def plot_monte_carlo_distribution(mc_results: dict[str, dict],
                                  save_path: str = "mc_sharpe.png"):
    """Plotta le distribuzioni bootstrap dello Sharpe."""
    # Questa funzione è più per reference — le stats sono nel dict
    print("\n" + "=" * 60)
    print("  MONTE CARLO BOOTSTRAP — Sharpe Ratio Confidence Intervals")
    print("=" * 60)
    for name, mc in mc_results.items():
        print(f"\n  {name}:")
        print(f"    Mean SR:        {mc['mean_sharpe']:.3f}")
        print(f"    Median SR:      {mc['median_sharpe']:.3f}")
        print(f"    95% CI:         [{mc['ci_lower']:.3f}, {mc['ci_upper']:.3f}]")
        print(f"    P(SR > 0):      {mc['prob_positive']:.1f}%")
        print(f"    P(SR > 1.0):    {mc['prob_above_1']:.1f}%")


# ═══════════════════════════════════════════════════════════════
#  7. MAIN — ESECUZIONE COMPLETA
# ═══════════════════════════════════════════════════════════════

def run_full_analysis(
    start: str = "2007-01-01",
    backtest_start: str = "2008-01-01",
    risk_free: float = 0.04,
    transaction_cost_bps: float = 10,
):
    """Esegue l'analisi completa: download, backtest, metriche, MC, plot."""

    tickers = list(DEFAULT_UNIVERSE.keys())

    print("Downloading data...")
    daily_prices = download_data(tickers, start=start)
    monthly_prices = to_monthly(daily_prices)
    print(f"  Periodo: {daily_prices.index[0].date()} → {daily_prices.index[-1].date()}")
    print(f"  Ticker:  {len(daily_prices.columns)} asset")

    # Definisci strategie
    strategies = [
        BuyAndHold(),
        EqualWeight(tickers),
        MomentumRotation(tickers, top_n=3, lookback_months=12),
        DualMomentum(),
        TrendMomentum(tickers, top_n=3, sma_days=200, mom_months=6),
        RiskParity(tickers, vol_window=60),
    ]

    # Aggiungi Ensemble (combina Momentum, Dual, Trend, RiskParity)
    ensemble = EnsembleStrategy([
        MomentumRotation(tickers, top_n=3, lookback_months=12),
        DualMomentum(),
        TrendMomentum(tickers, top_n=3, sma_days=200, mom_months=6),
        RiskParity(tickers, vol_window=60),
    ])
    strategies.append(ensemble)

    # Backtest
    print("\nRunning backtests...")
    results = []
    for s in strategies:
        print(f"  → {s.name}...")
        res = backtest(s, daily_prices, monthly_prices,
                       start_date=backtest_start,
                       transaction_cost_bps=transaction_cost_bps)
        results.append(res)

    # Metriche
    print("\n" + "=" * 70)
    print("  PERFORMANCE METRICS")
    print("=" * 70)

    metrics_list = []
    for r in results:
        m = compute_metrics(r["equity"], risk_free)
        m["Strategy"] = r["strategy"]
        metrics_list.append(m)

    metrics_df = pd.DataFrame(metrics_list).set_index("Strategy")
    print(metrics_df.to_string())

    # Average annual turnover
    print("\n  Average Annual Turnover:")
    for r in results:
        avg_monthly = r["turnover"]["turnover"].mean()
        print(f"    {r['strategy']:40s}  {avg_monthly*12*100:.1f}%")

    # Monte Carlo
    print("\nRunning Monte Carlo bootstrap (5000 simulations)...")
    mc_results = {}
    for r in results:
        mc = monte_carlo_sharpe(r["equity"], risk_free_annual=risk_free)
        mc_results[r["strategy"]] = mc

    plot_monte_carlo_distribution(mc_results)

    # Plots
    print("\nGenerating plots...")
    plot_results(results)

    # Riepilogo finale
    print("\n" + "=" * 70)
    print("  RIEPILOGO FINALE")
    print("=" * 70)
    best = max(metrics_list, key=lambda x: x.get("Sharpe Ratio", 0))
    print(f"\n  Miglior Sharpe Ratio: {best['Strategy']}")
    print(f"    SR = {best['Sharpe Ratio']:.3f}")
    print(f"    CAGR = {best['CAGR (%)']}%")
    print(f"    Max DD = {best['Max Drawdown (%)']}%")

    mc_best = mc_results[best["Strategy"]]
    print(f"    MC 95% CI: [{mc_best['ci_lower']:.3f}, {mc_best['ci_upper']:.3f}]")
    print(f"    P(SR > 1.0) = {mc_best['prob_above_1']}%")
    print(f"    P(SR > 1.5) = {round((np.array([mc_results[best['Strategy']]['mean_sharpe']]) > 1.5).mean() * 100, 1)}% (point est.)")

    print("\n  ⚠️  DISCLAIMER:")
    print("  I risultati passati non garantiscono performance future.")
    print("  Uno SR > 1.5 sostenuto è estremamente raro fuori dal backtest.")
    print("  Il backtest è soggetto a survivorship bias (ETF che esistono oggi).")
    print("  Questo è un framework educativo, non un consiglio finanziario.")

    return results, metrics_df, mc_results


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    results, metrics, mc = run_full_analysis(
        start="2007-01-01",
        backtest_start="2008-01-01",
        risk_free=0.04,
        transaction_cost_bps=10,
    )