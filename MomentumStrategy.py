"""
S&P 500 Momentum Scoring System
=================================
Sistema multi-indicatore per selezionare titoli in forte momentum dall'S&P 500.

LOGICA:
  Per ogni titolo, calcola un "momentum score" (0-5) basato su 5 indicatori:
    1. Price > SMA(200)          — trend di lungo periodo
    2. SMA(50) > SMA(200)        — golden cross / trend alignment
    3. RSI in zona momentum      — forza senza ipercomprato estremo
    4. ROC(N) > 0                — rendimento positivo su N giorni
    5. Prezzo vicino al max 52w  — forza relativa alta

  ENTRY: score >= entry_threshold (default 4/5)
  EXIT:  score <  exit_threshold  (default 3/5)
  Hysteresis tra entry/exit per ridurre whipsaw.

  Rebalancing mensile. Top-K titoli per score, equal weight.
  Quando nessun titolo qualifica → 100% cash (SHY proxy).

GRID SEARCH:
  Ottimizza parametri su training set, valida su test set.
  Parametri: soglie entry/exit, top_k, RSI bounds, ROC period, etc.

Dipendenze:
    pip install yfinance numpy pandas matplotlib tqdm
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product
from datetime import datetime
from typing import Optional
import warnings
import os
import json

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════
#  1. DATA LAYER
# ═══════════════════════════════════════════════════════════════

def get_sp500_tickers() -> list[str]:
    """
    Lista dei componenti attuali dell'S&P 500.
    Nota: survivorship bias — usiamo i componenti attuali.
    Per un backtest più rigoroso servirebbe la composizione storica.
    """
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
        return sorted(tickers)
    except Exception:
        # Fallback: top ~100 titoli noti dell'S&P 500
        return [
            "AAPL", "ABBV", "ABT", "ACN", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AFL",
            "AIG", "AMAT", "AMD", "AMGN", "AMZN", "ANET", "APD", "APH", "AVGO", "AXP",
            "BA", "BAC", "BDX", "BK", "BKNG", "BLK", "BMY", "BRK-B", "BSX", "C",
            "CAT", "CCI", "CDNS", "CI", "CL", "CMCSA", "CME", "COF", "COP", "COST",
            "CRM", "CSCO", "CTAS", "CVS", "CVX", "D", "DD", "DE", "DHR", "DIS",
            "DOW", "DUK", "ECL", "EL", "EMR", "EOG", "EXC", "F", "FDX", "FISV",
            "GD", "GE", "GILD", "GM", "GOOG", "GOOGL", "GPN", "GS", "HD", "HON",
            "IBM", "ICE", "INTC", "INTU", "ISRG", "ITW", "JNJ", "JPM", "KO", "LIN",
            "LLY", "LMT", "LOW", "MA", "MCD", "MCHP", "MCK", "MDLZ", "MDT", "MET",
            "META", "MMC", "MMM", "MO", "MRK", "MS", "MSFT", "MU", "NEE", "NFLX",
            "NKE", "NOC", "NOW", "NSC", "NVDA", "ORCL", "OXY", "PEP", "PFE", "PG",
            "PGR", "PLD", "PM", "PNC", "PSA", "PYPL", "QCOM", "ROP", "RTX", "SBUX",
            "SCHW", "SHW", "SLB", "SNPS", "SO", "SPG", "SPGI", "SYK", "T", "TGT",
            "TMO", "TMUS", "TRV", "TSLA", "TXN", "UNH", "UNP", "UPS", "USB", "V",
            "VLO", "VRTX", "VZ", "WBA", "WFC", "WM", "WMT", "XOM", "ZTS",
        ]


def download_stock_data(
    tickers: list[str],
    start: str = "1999-01-01",
    end: Optional[str] = None,
    batch_size: int = 50,
    cache_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Scarica dati storici per una lista di ticker.
    Scarica a batch per evitare timeout.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading cached data from {cache_path}")
        df = pd.read_parquet(cache_path)
        return df

    end = end or datetime.today().strftime("%Y-%m-%d")
    all_data = {}
    failed = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"  Downloading batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1} "
              f"({len(batch)} tickers)...")
        try:
            data = yf.download(batch, start=start, end=end, auto_adjust=True,
                               threads=True, progress=False)["Close"]
            if isinstance(data, pd.Series):
                data = data.to_frame(name=batch[0])
            for col in data.columns:
                if data[col].notna().sum() > 252:  # almeno 1 anno di dati
                    all_data[col] = data[col]
        except Exception as e:
            print(f"    Batch failed: {e}")
            failed.extend(batch)

    if not all_data:
        raise ValueError("No data downloaded!")

    df = pd.DataFrame(all_data)
    print(f"  Downloaded {len(df.columns)} tickers, {len(df)} trading days")
    print(f"  Failed: {len(failed)} tickers")

    if cache_path:
        df.to_parquet(cache_path)
        print(f"  Cached to {cache_path}")

    return df


# Benchmark
def download_benchmark(start="1999-01-01", end=None):
    """Scarica SPY come benchmark."""
    end = end or datetime.today().strftime("%Y-%m-%d")
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)["Close"]
    return spy


# ═══════════════════════════════════════════════════════════════
#  2. INDICATORI TECNICI
# ═══════════════════════════════════════════════════════════════

def compute_sma(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window, min_periods=window).mean()


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_roc(prices: pd.Series, period: int) -> pd.Series:
    return prices.pct_change(period)


def compute_pct_from_high(prices: pd.Series, window: int = 252) -> pd.Series:
    """% dal massimo a 52 settimane (252 trading days)."""
    rolling_max = prices.rolling(window, min_periods=window).max()
    return prices / rolling_max


# ═══════════════════════════════════════════════════════════════
#  3. MOMENTUM SCORING ENGINE
# ═══════════════════════════════════════════════════════════════

class MomentumScorer:
    """
    Calcola il momentum score (0-5) per ogni titolo ad ogni data.

    Indicatori:
      1. Price > SMA(sma_long)
      2. SMA(sma_short) > SMA(sma_long)
      3. RSI(rsi_period) in [rsi_low, rsi_high]
      4. ROC(roc_period) > 0
      5. Price >= pct_from_high_thresh * 52w_high
    """

    def __init__(
        self,
        sma_short: int = 50,
        sma_long: int = 200,
        rsi_period: int = 14,
        rsi_low: float = 40,
        rsi_high: float = 80,
        roc_period: int = 126,  # ~6 mesi
        pct_from_high_thresh: float = 0.90,  # entro 10% dal max 52w
        high_window: int = 252,
    ):
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.rsi_period = rsi_period
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high
        self.roc_period = roc_period
        self.pct_from_high_thresh = pct_from_high_thresh
        self.high_window = high_window

    def score_universe(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calcola lo score per ogni titolo ad ogni data.
        Ritorna DataFrame con stesse dimensioni di prices, valori 0-5.
        """
        scores = pd.DataFrame(0, index=prices.index, columns=prices.columns, dtype=float)

        for ticker in prices.columns:
            p = prices[ticker].dropna()
            if len(p) < self.sma_long + 50:
                continue

            sma_s = compute_sma(p, self.sma_short)
            sma_l = compute_sma(p, self.sma_long)
            rsi = compute_rsi(p, self.rsi_period)
            roc = compute_roc(p, self.roc_period)
            pct_high = compute_pct_from_high(p, self.high_window)

            # Indicatore 1: Price > SMA long
            i1 = (p > sma_l).astype(float)

            # Indicatore 2: SMA short > SMA long (trend alignment)
            i2 = (sma_s > sma_l).astype(float)

            # Indicatore 3: RSI in zona momentum (non ipervenduto, non ipercomprato estremo)
            i3 = ((rsi >= self.rsi_low) & (rsi <= self.rsi_high)).astype(float)

            # Indicatore 4: ROC positivo
            i4 = (roc > 0).astype(float)

            # Indicatore 5: Vicino al max 52 settimane
            i5 = (pct_high >= self.pct_from_high_thresh).astype(float)

            total = i1 + i2 + i3 + i4 + i5
            scores[ticker] = total.reindex(prices.index)

        return scores.fillna(0)

    def params_dict(self) -> dict:
        return {
            "sma_short": self.sma_short, "sma_long": self.sma_long,
            "rsi_period": self.rsi_period, "rsi_low": self.rsi_low,
            "rsi_high": self.rsi_high, "roc_period": self.roc_period,
            "pct_from_high_thresh": self.pct_from_high_thresh,
        }


# ═══════════════════════════════════════════════════════════════
#  4. PORTFOLIO CONSTRUCTION & BACKTEST
# ═══════════════════════════════════════════════════════════════

def run_backtest(
    prices: pd.DataFrame,
    scores: pd.DataFrame,
    benchmark: pd.Series,
    entry_threshold: int = 4,
    exit_threshold: int = 3,
    top_k: int = 20,
    rebalance_freq: str = "ME",  # month-end
    transaction_cost_bps: float = 10,
    start_date: str = "2000-06-01",
    initial_capital: float = 100_000,
) -> dict:
    """
    Backtest con hysteresis entry/exit.

    - Entry: score >= entry_threshold → candidato
    - Exit: score < exit_threshold → rimuovi
    - Tra i candidati, prendi top_k per score (tie-break: ROC più alto)
    - Equal weight
    - Se nessuno qualifica → cash
    """
    cost_rate = transaction_cost_bps / 10_000

    # Resample a fine mese per ribilanciamento
    monthly_dates = prices.resample(rebalance_freq).last().loc[start_date:].index
    monthly_prices = prices.resample(rebalance_freq).last()
    monthly_scores = scores.resample(rebalance_freq).last()

    # Align benchmark
    monthly_bench = benchmark.resample(rebalance_freq).last().reindex(monthly_dates)

    portfolio_value = initial_capital
    bench_value = initial_capital

    equity_curve = []
    bench_curve = []
    holdings_history = []
    n_holdings = []
    current_holdings = set()

    for i in range(len(monthly_dates) - 1):
        date = monthly_dates[i]
        next_date = monthly_dates[i + 1]

        if date not in monthly_scores.index:
            continue

        score_row = monthly_scores.loc[date]

        # ─── Hysteresis logic ───
        # Mantieni chi è ancora sopra exit_threshold
        surviving = {t for t in current_holdings
                     if t in score_row.index and score_row[t] >= exit_threshold}

        # Nuovi candidati: score >= entry_threshold e non già dentro
        new_candidates = {t for t in score_row.index
                         if score_row[t] >= entry_threshold and t not in surviving}

        # Pool totale
        pool = surviving | new_candidates

        # Seleziona top_k per score (poi per rendimento come tiebreak)
        if pool:
            pool_scores = {t: score_row[t] for t in pool
                          if t in monthly_prices.columns and pd.notna(monthly_prices[t].get(date))}
            sorted_pool = sorted(pool_scores, key=lambda t: pool_scores[t], reverse=True)[:top_k]
        else:
            sorted_pool = []

        current_holdings = set(sorted_pool)
        n_hold = len(sorted_pool)
        n_holdings.append(n_hold)

        # ─── Calcola rendimenti ───
        if n_hold > 0:
            weight = 1.0 / n_hold
            port_ret = 0
            for t in sorted_pool:
                if t in monthly_prices.columns:
                    p_now = monthly_prices[t].get(date, np.nan)
                    p_next = monthly_prices[t].get(next_date, np.nan)
                    if pd.notna(p_now) and pd.notna(p_next) and p_now > 0:
                        port_ret += weight * (p_next / p_now - 1)
        else:
            # Cash: ~0.2% mensile (approssimazione)
            port_ret = 0.002

        # Turnover (semplificato: new holdings / total)
        turnover_estimate = 0.3  # semplificazione conservativa
        cost = portfolio_value * turnover_estimate * cost_rate

        portfolio_value = (portfolio_value - cost) * (1 + port_ret)

        # Benchmark
        b_now = monthly_bench.get(date, np.nan)
        b_next = monthly_bench.get(next_date, np.nan)
        if pd.notna(b_now) and pd.notna(b_next) and b_now > 0:
            bench_value *= (1 + (b_next / b_now - 1))

        equity_curve.append({"date": next_date, "value": portfolio_value})
        bench_curve.append({"date": next_date, "value": bench_value})
        holdings_history.append({"date": date, "n_holdings": n_hold,
                                  "holdings": list(sorted_pool)[:10]})  # top 10 for log

    eq_df = pd.DataFrame(equity_curve).set_index("date")
    bench_df = pd.DataFrame(bench_curve).set_index("date")

    return {
        "equity": eq_df,
        "benchmark": bench_df,
        "avg_holdings": np.mean(n_holdings) if n_holdings else 0,
        "holdings_history": holdings_history,
        "params": {
            "entry_threshold": entry_threshold,
            "exit_threshold": exit_threshold,
            "top_k": top_k,
        }
    }


# ═══════════════════════════════════════════════════════════════
#  5. METRICHE
# ═══════════════════════════════════════════════════════════════

def compute_metrics(equity: pd.DataFrame, risk_free_annual: float = 0.04) -> dict:
    v = equity["value"]
    r = v.pct_change().dropna()
    n_years = len(r) / 12

    if n_years <= 0 or v.iloc[0] <= 0:
        return {"Sharpe": -99}

    total_ret = v.iloc[-1] / v.iloc[0]
    cagr = total_ret ** (1 / n_years) - 1
    vol = r.std() * np.sqrt(12)

    rf_m = risk_free_annual / 12
    excess = r - rf_m
    sharpe = excess.mean() / excess.std() * np.sqrt(12) if excess.std() > 0 else 0

    down = excess[excess < 0]
    down_std = down.std() * np.sqrt(12) if len(down) > 0 else 1
    sortino = (cagr - risk_free_annual) / down_std if down_std > 0 else 0

    dd = (v - v.cummax()) / v.cummax()
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    win_rate = (r > 0).sum() / len(r) if len(r) > 0 else 0

    return {
        "CAGR (%)": round(cagr * 100, 2),
        "Volatility (%)": round(vol * 100, 2),
        "Sharpe": round(sharpe, 3),
        "Sortino": round(sortino, 3),
        "Max DD (%)": round(max_dd * 100, 2),
        "Calmar": round(calmar, 3),
        "Win Rate (%)": round(win_rate * 100, 1),
        "Total Return (%)": round((total_ret - 1) * 100, 2),
    }


def monte_carlo_sharpe(equity, n_sim=5000, rf=0.04):
    r = equity["value"].pct_change().dropna().values
    rf_m = rf / 12
    n = len(r)
    srs = []
    for _ in range(n_sim):
        s = np.random.choice(r, size=n, replace=True)
        ex = s - rf_m
        if ex.std() > 0:
            srs.append(ex.mean() / ex.std() * np.sqrt(12))
    srs = np.array(srs)
    return {
        "mean": round(np.mean(srs), 3),
        "ci_lo": round(np.percentile(srs, 2.5), 3),
        "ci_hi": round(np.percentile(srs, 97.5), 3),
        "p_gt_1": round((srs > 1.0).mean() * 100, 1),
        "p_gt_1.5": round((srs > 1.5).mean() * 100, 1),
    }


# ═══════════════════════════════════════════════════════════════
#  6. GRID SEARCH OPTIMIZER
# ═══════════════════════════════════════════════════════════════

def grid_search(
    prices: pd.DataFrame,
    benchmark: pd.Series,
    train_end: str = "2016-12-31",
    test_start: str = "2017-01-01",
    start_date: str = "2000-06-01",
) -> dict:
    """
    Grid search su parametri con walk-forward split.
    Train: start_date → train_end
    Test:  test_start → fine dati

    Ottimizza per Sharpe Ratio sul training set.
    """

    # ─── Griglia parametri ───
    param_grid = {
        # Scorer params
        "rsi_low":     [30, 40, 50],
        "rsi_high":    [75, 80, 85],
        "roc_period":  [63, 126, 189],    # 3m, 6m, 9m
        "pct_high":    [0.85, 0.90, 0.95],
        # Portfolio params
        "entry_thresh": [4, 5],
        "exit_thresh":  [2, 3],
        "top_k":        [10, 20, 30],
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(product(*values))
    total = len(combos)

    print(f"\n  Grid Search: {total} combinazioni")
    print(f"  Train: {start_date} → {train_end}")
    print(f"  Test:  {test_start} → end\n")

    # Pre-filter: rimuovi entry < exit
    valid_combos = [c for c in combos if c[keys.index("entry_thresh")] > c[keys.index("exit_thresh")]]
    print(f"  Combinazioni valide (entry > exit): {len(valid_combos)}")

    best_sharpe = -999
    best_params = None
    best_result = None
    all_results = []

    for idx, combo in enumerate(valid_combos):
        params = dict(zip(keys, combo))

        if idx % 20 == 0:
            print(f"  [{idx+1}/{len(valid_combos)}] Testing... best SR so far: {best_sharpe:.3f}")

        # Build scorer
        scorer = MomentumScorer(
            sma_short=50, sma_long=200,
            rsi_period=14,
            rsi_low=params["rsi_low"],
            rsi_high=params["rsi_high"],
            roc_period=params["roc_period"],
            pct_from_high_thresh=params["pct_high"],
        )

        # Score (solo training period per velocità)
        train_prices = prices.loc[:train_end]
        scores = scorer.score_universe(train_prices)

        # Backtest su training
        result = run_backtest(
            train_prices, scores, benchmark,
            entry_threshold=params["entry_thresh"],
            exit_threshold=params["exit_thresh"],
            top_k=params["top_k"],
            start_date=start_date,
        )

        metrics = compute_metrics(result["equity"])
        sr = metrics.get("Sharpe", -99)

        all_results.append({**params, **metrics})

        if sr > best_sharpe:
            best_sharpe = sr
            best_params = params
            best_result = result

    print(f"\n  ✓ Best Training Sharpe: {best_sharpe:.3f}")
    print(f"    Params: {best_params}")

    # ─── Validation su test set ───
    print(f"\n  Running best params on TEST set ({test_start} → end)...")

    scorer = MomentumScorer(
        sma_short=50, sma_long=200, rsi_period=14,
        rsi_low=best_params["rsi_low"],
        rsi_high=best_params["rsi_high"],
        roc_period=best_params["roc_period"],
        pct_from_high_thresh=best_params["pct_high"],
    )

    # Full scores (need history before test_start for indicators)
    full_scores = scorer.score_universe(prices)

    test_result = run_backtest(
        prices, full_scores, benchmark,
        entry_threshold=best_params["entry_thresh"],
        exit_threshold=best_params["exit_thresh"],
        top_k=best_params["top_k"],
        start_date=test_start,
    )

    test_metrics = compute_metrics(test_result["equity"])
    bench_test = compute_metrics(test_result["benchmark"])

    # Full period backtest
    full_result = run_backtest(
        prices, full_scores, benchmark,
        entry_threshold=best_params["entry_thresh"],
        exit_threshold=best_params["exit_thresh"],
        top_k=best_params["top_k"],
        start_date=start_date,
    )
    full_metrics = compute_metrics(full_result["equity"])

    return {
        "best_params": best_params,
        "train_metrics": compute_metrics(best_result["equity"]),
        "test_metrics": test_metrics,
        "bench_test_metrics": bench_test,
        "full_metrics": full_metrics,
        "full_result": full_result,
        "test_result": test_result,
        "all_results": pd.DataFrame(all_results),
        "scorer": scorer,
    }


# ═══════════════════════════════════════════════════════════════
#  7. VISUALIZZAZIONE
# ═══════════════════════════════════════════════════════════════

def plot_backtest_results(
    full_result: dict,
    test_result: dict,
    full_metrics: dict,
    test_metrics: dict,
    bench_metrics: dict,
    best_params: dict,
    save_dir: str = ".",
):
    """Dashboard completa dei risultati."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("S&P 500 Momentum Scoring System — Backtest Results",
                 fontsize=14, fontweight="bold", y=1.01)

    # ─── 1. Full Period Equity Curve ───
    ax = axes[0, 0]
    eq = full_result["equity"]["value"]
    bm = full_result["benchmark"]["value"]
    ax.plot(eq.index, eq / eq.iloc[0], label=f'Momentum (SR={full_metrics["Sharpe"]:.2f})',
            color="#2ECC71", lw=1.8)
    ax.plot(bm.index, bm / bm.iloc[0], label=f'S&P 500 (SR={compute_metrics(full_result["benchmark"])["Sharpe"]:.2f})',
            color="#E74C3C", lw=1.5, alpha=0.7)
    ax.set_title("Full Period — Equity Curves", fontweight="bold")
    ax.set_ylabel("Crescita di 1€")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ─── 2. Test Period (Out-of-Sample) ───
    ax = axes[0, 1]
    eq_t = test_result["equity"]["value"]
    bm_t = test_result["benchmark"]["value"]
    ax.plot(eq_t.index, eq_t / eq_t.iloc[0],
            label=f'Momentum OOS (SR={test_metrics["Sharpe"]:.2f})', color="#3498DB", lw=1.8)
    ax.plot(bm_t.index, bm_t / bm_t.iloc[0],
            label=f'S&P 500 OOS (SR={bench_metrics["Sharpe"]:.2f})', color="#E74C3C", lw=1.5, alpha=0.7)
    ax.set_title("Out-of-Sample (Test) Period", fontweight="bold")
    ax.set_ylabel("Crescita di 1€")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ─── 3. Drawdowns ───
    ax = axes[1, 0]
    eq_full = full_result["equity"]["value"]
    dd_strat = (eq_full - eq_full.cummax()) / eq_full.cummax() * 100
    bm_full = full_result["benchmark"]["value"]
    dd_bench = (bm_full - bm_full.cummax()) / bm_full.cummax() * 100
    ax.fill_between(dd_strat.index, dd_strat.values, 0, alpha=0.4, color="#2ECC71", label="Momentum")
    ax.fill_between(dd_bench.index, dd_bench.values, 0, alpha=0.3, color="#E74C3C", label="S&P 500")
    ax.set_title("Drawdowns", fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ─── 4. Rolling Sharpe ───
    ax = axes[1, 1]
    r_strat = eq_full.pct_change().dropna()
    r_bench = bm_full.pct_change().dropna()
    rs_strat = r_strat.rolling(12).mean() / r_strat.rolling(12).std() * np.sqrt(12)
    rs_bench = r_bench.rolling(12).mean() / r_bench.rolling(12).std() * np.sqrt(12)
    ax.plot(rs_strat.index, rs_strat.values, color="#2ECC71", label="Momentum", lw=1.2)
    ax.plot(rs_bench.index, rs_bench.values, color="#E74C3C", label="S&P 500", lw=1.2, alpha=0.7)
    ax.axhline(y=0, color="gray", ls="--", alpha=0.5)
    ax.axhline(y=1.0, color="blue", ls=":", alpha=0.4, label="SR=1.0")
    ax.set_title("Rolling Sharpe (12 mesi)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "momentum_system_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Dashboard saved: {path}")
    plt.close()

    # ─── Heatmap parametri ───
    return path


def plot_param_sensitivity(all_results: pd.DataFrame, save_dir: str = "."):
    """Heatmap della sensibilità dei parametri allo Sharpe."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Entry threshold vs Top K
    pivot1 = all_results.pivot_table(values="Sharpe", index="entry_thresh",
                                      columns="top_k", aggfunc="mean")
    im1 = axes[0].imshow(pivot1.values, cmap="RdYlGn", aspect="auto")
    axes[0].set_xticks(range(len(pivot1.columns)))
    axes[0].set_xticklabels(pivot1.columns)
    axes[0].set_yticks(range(len(pivot1.index)))
    axes[0].set_yticklabels(pivot1.index)
    axes[0].set_xlabel("Top K")
    axes[0].set_ylabel("Entry Threshold")
    axes[0].set_title("Sharpe: Entry vs Top K")
    for i in range(len(pivot1.index)):
        for j in range(len(pivot1.columns)):
            axes[0].text(j, i, f"{pivot1.values[i,j]:.2f}", ha="center", va="center", fontsize=9)

    # 2. RSI bounds vs ROC period
    pivot2 = all_results.pivot_table(values="Sharpe", index="rsi_low",
                                      columns="roc_period", aggfunc="mean")
    axes[1].imshow(pivot2.values, cmap="RdYlGn", aspect="auto")
    axes[1].set_xticks(range(len(pivot2.columns)))
    axes[1].set_xticklabels(pivot2.columns)
    axes[1].set_yticks(range(len(pivot2.index)))
    axes[1].set_yticklabels(pivot2.index)
    axes[1].set_xlabel("ROC Period")
    axes[1].set_ylabel("RSI Low")
    axes[1].set_title("Sharpe: RSI Low vs ROC Period")
    for i in range(len(pivot2.index)):
        for j in range(len(pivot2.columns)):
            axes[1].text(j, i, f"{pivot2.values[i,j]:.2f}", ha="center", va="center", fontsize=9)

    # 3. % from high vs exit threshold
    pivot3 = all_results.pivot_table(values="Sharpe", index="pct_high",
                                      columns="exit_thresh", aggfunc="mean")
    axes[2].imshow(pivot3.values, cmap="RdYlGn", aspect="auto")
    axes[2].set_xticks(range(len(pivot3.columns)))
    axes[2].set_xticklabels(pivot3.columns)
    axes[2].set_yticks(range(len(pivot3.index)))
    axes[2].set_yticklabels(pivot3.index)
    axes[2].set_xlabel("Exit Threshold")
    axes[2].set_ylabel("% from 52w High")
    axes[2].set_title("Sharpe: %High vs Exit Thresh")
    for i in range(len(pivot3.index)):
        for j in range(len(pivot3.columns)):
            axes[2].text(j, i, f"{pivot3.values[i,j]:.2f}", ha="center", va="center", fontsize=9)

    plt.suptitle("Parameter Sensitivity Analysis (Training Set)", fontweight="bold", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, "param_sensitivity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Sensitivity plot saved: {path}")
    plt.close()
    return path


# ═══════════════════════════════════════════════════════════════
#  8. MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    OUTPUT_DIR = "/mnt/user-data/outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    START = "1999-01-01"       # Extra per warmup indicatori
    BACKTEST_START = "2000-06-01"
    TRAIN_END = "2016-12-31"
    TEST_START = "2017-01-01"

    # ─── Download ───
    print("=" * 70)
    print("  S&P 500 MOMENTUM SCORING SYSTEM")
    print("=" * 70)

    print("\n1. Getting S&P 500 tickers...")
    tickers = get_sp500_tickers()
    print(f"   {len(tickers)} tickers")

    print("\n2. Downloading price data...")
    prices = download_stock_data(
        tickers, start=START,
        cache_path=os.path.join(OUTPUT_DIR, "sp500_prices.parquet")
    )

    print("\n3. Downloading benchmark (SPY)...")
    benchmark = download_benchmark(start=START)

    # ─── Grid Search ───
    print("\n4. Running Grid Search Optimization...")
    gs = grid_search(
        prices, benchmark,
        train_end=TRAIN_END,
        test_start=TEST_START,
        start_date=BACKTEST_START,
    )

    # ─── Results ───
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    print("\n  Best Parameters:")
    for k, v in gs["best_params"].items():
        print(f"    {k:20s}: {v}")

    print(f"\n  {'Metric':<22s} {'Train':>10s} {'Test (OOS)':>12s} {'Bench (OOS)':>12s} {'Full':>10s}")
    print("  " + "-" * 68)
    for key in gs["train_metrics"]:
        t = gs["train_metrics"].get(key, "-")
        o = gs["test_metrics"].get(key, "-")
        b = gs["bench_test_metrics"].get(key, "-")
        f = gs["full_metrics"].get(key, "-")
        print(f"  {key:<22s} {str(t):>10s} {str(o):>12s} {str(b):>12s} {str(f):>10s}")

    # Monte Carlo
    print("\n  Monte Carlo Bootstrap (full period)...")
    mc = monte_carlo_sharpe(gs["full_result"]["equity"])
    print(f"    Mean SR:     {mc['mean']:.3f}")
    print(f"    95% CI:      [{mc['ci_lo']:.3f}, {mc['ci_hi']:.3f}]")
    print(f"    P(SR > 1.0): {mc['p_gt_1']}%")
    print(f"    P(SR > 1.5): {mc['p_gt_1.5']}%")

    # ─── Plots ───
    print("\n5. Generating plots...")
    plot_backtest_results(
        gs["full_result"], gs["test_result"],
        gs["full_metrics"], gs["test_metrics"],
        gs["bench_test_metrics"], gs["best_params"],
        save_dir=OUTPUT_DIR,
    )
    plot_param_sensitivity(gs["all_results"], save_dir=OUTPUT_DIR)

    # ─── Save results ───
    summary = {
        "best_params": gs["best_params"],
        "train_metrics": gs["train_metrics"],
        "test_metrics": gs["test_metrics"],
        "benchmark_test_metrics": gs["bench_test_metrics"],
        "full_metrics": gs["full_metrics"],
        "monte_carlo": mc,
    }
    with open(os.path.join(OUTPUT_DIR, "optimization_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {OUTPUT_DIR}/optimization_results.json")

    # ─── Holdings snapshot ───
    print("\n  Recent Holdings (last 3 months):")
    for h in gs["full_result"]["holdings_history"][-3:]:
        print(f"    {h['date'].strftime('%Y-%m')}: {h['n_holdings']} stocks → {h['holdings']}")

    print("\n  ⚠️  DISCLAIMER:")
    print("  - Survivorship bias: usiamo componenti ATTUALI dell'S&P 500")
    print("  - Grid search su training set → rischio overfitting residuo")
    print("  - Il test OOS è il vero indicatore di robustezza")
    print("  - Questo è un framework educativo, non un consiglio finanziario")
    print()

    return gs


if __name__ == "__main__":
    gs = main()