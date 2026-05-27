"""
S&P 500 Momentum Scoring System — V2
======================================
Miglioramenti rispetto a V1:
  ✓ FIX: Benchmark tracking corretto (bug .get() su DatetimeIndex)
  ✓ NEW: Filtro regime di mercato (SPY > SMA200 → risk-on, altrimenti cash)
  ✓ NEW: Position sizing proporzionale allo score (5/5 → peso maggiore)
  ✓ NEW: Turnover realistico (tracciato effettivamente, non stimato)
  ✓ NEW: Griglia estesa con regime filter come parametro ottimizzabile
  ✓ NEW: Confronto diretto V1 vs V2 vs Benchmark

Dipendenze:
    pip install yfinance numpy pandas matplotlib
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
import warnings, os, json

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════
#  1. DATA LAYER (identico a V1)
# ═══════════════════════════════════════════════════════════════

def get_sp500_tickers() -> list[str]:
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        return sorted(table["Symbol"].str.replace(".", "-", regex=False).tolist())
    except Exception:
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


def download_stock_data(tickers, start="1999-01-01", end=None, batch_size=50, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)
    end = end or datetime.today().strftime("%Y-%m-%d")
    all_data = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        print(f"  Downloading batch {i//batch_size+1}/{(len(tickers)-1)//batch_size+1} ({len(batch)} tickers)...")
        try:
            data = yf.download(batch, start=start, end=end, auto_adjust=True, threads=True, progress=False)["Close"]
            if isinstance(data, pd.Series): data = data.to_frame(name=batch[0])
            for col in data.columns:
                if data[col].notna().sum() > 252:
                    all_data[col] = data[col]
        except Exception as e:
            print(f"    Batch failed: {e}")
    df = pd.DataFrame(all_data)
    print(f"  Downloaded {len(df.columns)} tickers, {len(df)} trading days")
    if cache_path:
        df.to_parquet(cache_path)
    return df


def download_benchmark(start="1999-01-01", end=None):
    end = end or datetime.today().strftime("%Y-%m-%d")
    return yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)["Close"]


# ═══════════════════════════════════════════════════════════════
#  2. INDICATORI
# ═══════════════════════════════════════════════════════════════

def compute_sma(p, w): return p.rolling(w, min_periods=w).mean()

def compute_rsi(p, n=14):
    d = p.diff()
    g, l = d.where(d > 0, 0.0), -d.where(d < 0, 0.0)
    return 100 - 100 / (1 + g.ewm(com=n-1, min_periods=n).mean() / l.ewm(com=n-1, min_periods=n).mean())

def compute_roc(p, n): return p.pct_change(n)

def compute_pct_from_high(p, w=252): return p / p.rolling(w, min_periods=w).max()


# ═══════════════════════════════════════════════════════════════
#  3. MOMENTUM SCORER
# ═══════════════════════════════════════════════════════════════

class MomentumScorer:
    def __init__(self, sma_short=50, sma_long=200, rsi_period=14,
                 rsi_low=40, rsi_high=80, roc_period=126,
                 pct_from_high_thresh=0.90, high_window=252):
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.rsi_period = rsi_period
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high
        self.roc_period = roc_period
        self.pct_from_high_thresh = pct_from_high_thresh
        self.high_window = high_window

    def score_universe(self, prices):
        scores = pd.DataFrame(0, index=prices.index, columns=prices.columns, dtype=float)
        for ticker in prices.columns:
            p = prices[ticker].dropna()
            if len(p) < self.sma_long + 50:
                continue
            sma_s = compute_sma(p, self.sma_short)
            sma_l = compute_sma(p, self.sma_long)
            rsi_val = compute_rsi(p, self.rsi_period)
            roc_val = compute_roc(p, self.roc_period)
            pct_h = compute_pct_from_high(p, self.high_window)

            total = ((p > sma_l).astype(float) +
                     (sma_s > sma_l).astype(float) +
                     ((rsi_val >= self.rsi_low) & (rsi_val <= self.rsi_high)).astype(float) +
                     (roc_val > 0).astype(float) +
                     (pct_h >= self.pct_from_high_thresh).astype(float))
            scores[ticker] = total.reindex(prices.index)
        return scores.fillna(0)


# ═══════════════════════════════════════════════════════════════
#  4. BACKTEST ENGINE V2
# ═══════════════════════════════════════════════════════════════

def run_backtest_v2(
    prices: pd.DataFrame,
    scores: pd.DataFrame,
    benchmark_daily: pd.Series,
    entry_threshold: int = 4,
    exit_threshold: int = 3,
    top_k: int = 20,
    use_regime_filter: bool = True,
    regime_sma: int = 200,
    score_weighted: bool = True,
    rebalance_freq: str = "ME",
    transaction_cost_bps: float = 10,
    start_date: str = "2000-06-01",
    initial_capital: float = 100_000,
) -> dict:
    """
    V2 Backtest con:
    - FIX: benchmark usa .loc[] correttamente
    - NEW: regime filter (SPY > SMA → risk-on)
    - NEW: score-weighted position sizing
    - NEW: turnover reale tracciato
    """
    cost_rate = transaction_cost_bps / 10_000

    monthly_prices = prices.resample(rebalance_freq).last()
    monthly_scores = scores.resample(rebalance_freq).last()
    monthly_dates = monthly_prices.loc[start_date:].index

    # ─── Benchmark: resample e allinea ───
    monthly_bench = benchmark_daily.resample(rebalance_freq).last()

    # ─── Regime: SPY daily SMA ───
    if use_regime_filter:
        spy_sma = compute_sma(benchmark_daily, regime_sma)
    else:
        spy_sma = None

    portfolio_value = initial_capital
    bench_value = initial_capital

    equity_curve, bench_curve = [], []
    n_holdings_list = []
    prev_weights = {}  # per calcolo turnover reale
    total_turnover = 0
    n_rebal = 0
    regime_history = []

    for i in range(len(monthly_dates) - 1):
        date = monthly_dates[i]
        next_date = monthly_dates[i + 1]

        # ─── Regime check ───
        risk_on = True
        if use_regime_filter and spy_sma is not None:
            # Trova l'ultimo giorno di trading <= date
            spy_val = benchmark_daily.loc[:date]
            sma_val = spy_sma.loc[:date]
            if len(spy_val) > 0 and len(sma_val) > 0:
                if pd.notna(spy_val.iloc[-1]) and pd.notna(sma_val.iloc[-1]):
                    risk_on = spy_val.iloc[-1] > sma_val.iloc[-1]
        regime_history.append({"date": date, "risk_on": risk_on})

        # ─── Portfolio selection ───
        if not risk_on:
            # Risk-off: 100% cash
            new_weights = {}
        elif date in monthly_scores.index:
            score_row = monthly_scores.loc[date]

            # Hysteresis
            surviving = {t for t in prev_weights
                         if t in score_row.index and score_row[t] >= exit_threshold}
            new_cand = {t for t in score_row.index
                        if score_row[t] >= entry_threshold and t not in surviving}
            pool = surviving | new_cand

            if pool:
                pool_scores = {t: score_row[t] for t in pool
                              if t in monthly_prices.columns
                              and date in monthly_prices.index
                              and pd.notna(monthly_prices.loc[date, t] if t in monthly_prices.columns else np.nan)}
                sorted_pool = sorted(pool_scores, key=pool_scores.get, reverse=True)[:top_k]

                if score_weighted and sorted_pool:
                    # Peso proporzionale allo score
                    raw_w = {t: pool_scores[t] for t in sorted_pool}
                    total_w = sum(raw_w.values())
                    new_weights = {t: w / total_w for t, w in raw_w.items()} if total_w > 0 else {}
                elif sorted_pool:
                    w = 1.0 / len(sorted_pool)
                    new_weights = {t: w for t in sorted_pool}
                else:
                    new_weights = {}
            else:
                new_weights = {}
        else:
            new_weights = {}

        n_holdings_list.append(len(new_weights))

        # ─── Turnover reale ───
        all_tickers = set(list(prev_weights.keys()) + list(new_weights.keys()))
        turnover = sum(abs(new_weights.get(t, 0) - prev_weights.get(t, 0)) for t in all_tickers) / 2
        total_turnover += turnover
        n_rebal += 1

        # ─── Costi ───
        cost = portfolio_value * turnover * cost_rate

        # ─── Rendimento portafoglio ───
        if new_weights:
            port_ret = 0
            for t, w in new_weights.items():
                if t in monthly_prices.columns:
                    try:
                        p_now = monthly_prices.loc[date, t]
                        p_next = monthly_prices.loc[next_date, t]
                        if pd.notna(p_now) and pd.notna(p_next) and p_now > 0:
                            port_ret += w * (p_next / p_now - 1)
                    except KeyError:
                        pass
        else:
            port_ret = 0.003  # ~3.6% annuo cash proxy

        portfolio_value = (portfolio_value - cost) * (1 + port_ret)

        # ─── Benchmark (FIX: usa .loc[]) ───
        try:
            b_now = monthly_bench.loc[date]
            b_next = monthly_bench.loc[next_date]
            if pd.notna(b_now) and pd.notna(b_next) and b_now > 0:
                bench_value *= (b_next / b_now)
        except KeyError:
            pass

        equity_curve.append({"date": next_date, "value": portfolio_value})
        bench_curve.append({"date": next_date, "value": bench_value})
        prev_weights = new_weights

    eq_df = pd.DataFrame(equity_curve).set_index("date")
    bench_df = pd.DataFrame(bench_curve).set_index("date")
    regime_df = pd.DataFrame(regime_history).set_index("date")

    avg_turnover_annual = (total_turnover / n_rebal * 12) if n_rebal > 0 else 0

    return {
        "equity": eq_df,
        "benchmark": bench_df,
        "avg_holdings": np.mean(n_holdings_list) if n_holdings_list else 0,
        "avg_annual_turnover": round(avg_turnover_annual * 100, 1),
        "regime": regime_df,
    }


# ═══════════════════════════════════════════════════════════════
#  5. METRICHE
# ═══════════════════════════════════════════════════════════════

def compute_metrics(equity, rf=0.04):
    v = equity["value"]
    r = v.pct_change().dropna()
    ny = len(r) / 12
    if ny <= 0 or v.iloc[0] <= 0:
        return {"Sharpe": -99}
    tr = v.iloc[-1] / v.iloc[0]
    cagr = tr ** (1/ny) - 1
    vol = r.std() * np.sqrt(12)
    rfm = rf / 12
    ex = r - rfm
    sr = ex.mean() / ex.std() * np.sqrt(12) if ex.std() > 0 else 0
    down = ex[ex < 0]
    ds = down.std() * np.sqrt(12) if len(down) > 0 else 1
    sortino = (cagr - rf) / ds if ds > 0 else 0
    dd = (v - v.cummax()) / v.cummax()
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    return {
        "CAGR%": round(cagr*100, 2),
        "Vol%": round(vol*100, 2),
        "Sharpe": round(sr, 3),
        "Sortino": round(sortino, 3),
        "MaxDD%": round(max_dd*100, 2),
        "Calmar": round(calmar, 3),
        "WinRate%": round((r > 0).mean()*100, 1),
        "TotalRet%": round((tr-1)*100, 2),
    }


def monte_carlo_sharpe(equity, n_sim=5000, rf=0.04):
    r = equity["value"].pct_change().dropna().values
    rfm = rf / 12
    srs = []
    for _ in range(n_sim):
        s = np.random.choice(r, size=len(r), replace=True)
        ex = s - rfm
        if ex.std() > 0: srs.append(ex.mean()/ex.std()*np.sqrt(12))
    srs = np.array(srs)
    return {
        "mean": round(np.mean(srs), 3),
        "ci_lo": round(np.percentile(srs, 2.5), 3),
        "ci_hi": round(np.percentile(srs, 97.5), 3),
        "p_gt_1": round((srs > 1.0).mean()*100, 1),
        "p_gt_1.5": round((srs > 1.5).mean()*100, 1),
    }


# ═══════════════════════════════════════════════════════════════
#  6. GRID SEARCH V2
# ═══════════════════════════════════════════════════════════════

def grid_search_v2(prices, benchmark_daily, train_end="2016-12-31",
                   test_start="2017-01-01", start_date="2000-06-01"):

    param_grid = {
        "rsi_low":        [30, 40, 50],
        "rsi_high":       [75, 80, 85],
        "roc_period":     [63, 126, 189],
        "pct_high":       [0.85, 0.90, 0.95],
        "entry_thresh":   [4, 5],
        "exit_thresh":    [2, 3],
        "top_k":          [10, 20, 30],
        "regime_filter":  [True, False],
        "score_weighted": [True, False],
    }

    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    valid = [c for c in combos
             if c[keys.index("entry_thresh")] > c[keys.index("exit_thresh")]]

    print(f"\n  Grid Search V2: {len(valid)} combinazioni")
    print(f"  Train: {start_date} → {train_end}")
    print(f"  Test:  {test_start} → end\n")

    best_sr, best_p = -999, None
    all_res = []

    for idx, combo in enumerate(valid):
        params = dict(zip(keys, combo))
        if idx % 50 == 0:
            print(f"  [{idx+1}/{len(valid)}] best SR: {best_sr:.3f}")

        scorer = MomentumScorer(
            rsi_low=params["rsi_low"], rsi_high=params["rsi_high"],
            roc_period=params["roc_period"], pct_from_high_thresh=params["pct_high"],
        )
        train_prices = prices.loc[:train_end]
        scores = scorer.score_universe(train_prices)

        result = run_backtest_v2(
            train_prices, scores, benchmark_daily,
            entry_threshold=params["entry_thresh"],
            exit_threshold=params["exit_thresh"],
            top_k=params["top_k"],
            use_regime_filter=params["regime_filter"],
            score_weighted=params["score_weighted"],
            start_date=start_date,
        )
        m = compute_metrics(result["equity"])
        all_res.append({**params, **m})
        if m["Sharpe"] > best_sr:
            best_sr, best_p = m["Sharpe"], params

    print(f"\n  ✓ Best TRAIN Sharpe: {best_sr:.3f}")
    print(f"    Params: {best_p}")

    # ─── Validation ───
    print(f"\n  Running on TEST ({test_start} → end) and FULL period...")
    scorer = MomentumScorer(
        rsi_low=best_p["rsi_low"], rsi_high=best_p["rsi_high"],
        roc_period=best_p["roc_period"], pct_from_high_thresh=best_p["pct_high"],
    )
    full_scores = scorer.score_universe(prices)

    test_r = run_backtest_v2(
        prices, full_scores, benchmark_daily,
        entry_threshold=best_p["entry_thresh"], exit_threshold=best_p["exit_thresh"],
        top_k=best_p["top_k"], use_regime_filter=best_p["regime_filter"],
        score_weighted=best_p["score_weighted"], start_date=test_start,
    )
    full_r = run_backtest_v2(
        prices, full_scores, benchmark_daily,
        entry_threshold=best_p["entry_thresh"], exit_threshold=best_p["exit_thresh"],
        top_k=best_p["top_k"], use_regime_filter=best_p["regime_filter"],
        score_weighted=best_p["score_weighted"], start_date=start_date,
    )

    # ─── V1 baseline (no regime, no score-weight) con stessi scorer params ───
    v1_r = run_backtest_v2(
        prices, full_scores, benchmark_daily,
        entry_threshold=best_p["entry_thresh"], exit_threshold=best_p["exit_thresh"],
        top_k=best_p["top_k"], use_regime_filter=False,
        score_weighted=False, start_date=start_date,
    )

    return {
        "best_params": best_p,
        "train_metrics": compute_metrics(run_backtest_v2(
            prices.loc[:train_end], scorer.score_universe(prices.loc[:train_end]),
            benchmark_daily, entry_threshold=best_p["entry_thresh"],
            exit_threshold=best_p["exit_thresh"], top_k=best_p["top_k"],
            use_regime_filter=best_p["regime_filter"],
            score_weighted=best_p["score_weighted"], start_date=start_date,
        )["equity"]),
        "test_metrics": compute_metrics(test_r["equity"]),
        "bench_test": compute_metrics(test_r["benchmark"]),
        "full_metrics": compute_metrics(full_r["equity"]),
        "bench_full": compute_metrics(full_r["benchmark"]),
        "v1_metrics": compute_metrics(v1_r["equity"]),
        "full_result": full_r,
        "test_result": test_r,
        "v1_result": v1_r,
        "all_results": pd.DataFrame(all_res),
        "avg_turnover": full_r["avg_annual_turnover"],
    }


# ═══════════════════════════════════════════════════════════════
#  7. PLOTS V2
# ═══════════════════════════════════════════════════════════════

def plot_v2(gs, save_dir="."):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("S&P 500 Momentum System V2 — Regime Filter + Score Weighting",
                 fontsize=13, fontweight="bold")

    # 1. Full equity: V2 vs V1 vs Benchmark
    ax = axes[0, 0]
    for key, label, color, lw in [
        ("full_result", f'V2 (SR={gs["full_metrics"]["Sharpe"]:.2f})', "#2ECC71", 2),
        ("v1_result", f'V1 no-regime (SR={gs["v1_metrics"]["Sharpe"]:.2f})', "#F39C12", 1.5),
    ]:
        eq = gs[key]["equity"]["value"]
        ax.plot(eq.index, eq/eq.iloc[0], label=label, color=color, lw=lw)
    bm = gs["full_result"]["benchmark"]["value"]
    ax.plot(bm.index, bm/bm.iloc[0], label=f'S&P 500 (SR={gs["bench_full"]["Sharpe"]:.2f})',
            color="#E74C3C", lw=1.3, alpha=0.7)
    ax.set_title("Full Period: V2 vs V1 vs Benchmark", fontweight="bold")
    ax.set_ylabel("Growth of 1€"); ax.set_yscale("log")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Shade risk-off periods
    regime = gs["full_result"].get("regime")
    if regime is not None and not regime.empty:
        risk_off = regime[~regime["risk_on"]]
        for d in risk_off.index:
            ax.axvline(x=d, color="red", alpha=0.05, lw=3)

    # 2. OOS
    ax = axes[0, 1]
    eq_t = gs["test_result"]["equity"]["value"]
    bm_t = gs["test_result"]["benchmark"]["value"]
    ax.plot(eq_t.index, eq_t/eq_t.iloc[0],
            label=f'V2 OOS (SR={gs["test_metrics"]["Sharpe"]:.2f})', color="#3498DB", lw=2)
    ax.plot(bm_t.index, bm_t/bm_t.iloc[0],
            label=f'S&P 500 OOS (SR={gs["bench_test"]["Sharpe"]:.2f})', color="#E74C3C", lw=1.3, alpha=0.7)
    ax.set_title("Out-of-Sample", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # 3. Drawdowns V2 vs Benchmark
    ax = axes[1, 0]
    eq_f = gs["full_result"]["equity"]["value"]
    dd_s = (eq_f - eq_f.cummax())/eq_f.cummax()*100
    dd_b = (bm - bm.cummax())/bm.cummax()*100
    ax.fill_between(dd_s.index, dd_s.values, 0, alpha=0.4, color="#2ECC71", label="V2 Momentum")
    ax.fill_between(dd_b.index, dd_b.values, 0, alpha=0.3, color="#E74C3C", label="S&P 500")
    ax.set_title("Drawdowns", fontweight="bold")
    ax.set_ylabel("DD (%)"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # 4. Rolling Sharpe
    ax = axes[1, 1]
    for key, label, color in [
        ("full_result", "V2", "#2ECC71"), ("v1_result", "V1", "#F39C12")
    ]:
        eq = gs[key]["equity"]["value"]
        r = eq.pct_change().dropna()
        rs = r.rolling(12).mean()/r.rolling(12).std()*np.sqrt(12)
        ax.plot(rs.index, rs.values, color=color, label=label, lw=1.2)
    r_b = bm.pct_change().dropna()
    rs_b = r_b.rolling(12).mean()/r_b.rolling(12).std()*np.sqrt(12)
    ax.plot(rs_b.index, rs_b.values, color="#E74C3C", label="S&P 500", lw=1, alpha=0.6)
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.axhline(1.0, color="blue", ls=":", alpha=0.3)
    ax.set_title("Rolling Sharpe (12m)", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(save_dir, "momentum_v2_results.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Dashboard: {p}")

    # Param sensitivity
    ar = gs["all_results"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (rows, col) in zip(axes, [
        ("entry_thresh", "top_k"), ("rsi_low", "roc_period"), ("regime_filter", "score_weighted")
    ]):
        pv = ar.pivot_table(values="Sharpe", index=rows, columns=col, aggfunc="mean")
        ax.imshow(pv.values, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels(pv.columns)
        ax.set_yticks(range(len(pv.index))); ax.set_yticklabels(pv.index)
        ax.set_xlabel(col); ax.set_ylabel(rows)
        ax.set_title(f"Avg Sharpe: {rows} vs {col}")
        for i in range(len(pv.index)):
            for j in range(len(pv.columns)):
                ax.text(j, i, f"{pv.values[i,j]:.2f}", ha="center", va="center", fontsize=10, fontweight="bold")

    plt.suptitle("V2 Parameter Sensitivity (Training Set)", fontweight="bold")
    plt.tight_layout()
    p2 = os.path.join(save_dir, "momentum_v2_sensitivity.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Sensitivity: {p2}")

    return p, p2


# ═══════════════════════════════════════════════════════════════
#  8. MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  S&P 500 MOMENTUM SCORING SYSTEM — V2")
    print("  + Regime Filter + Score-Weighted Sizing")
    print("=" * 70)

    print("\n1. Getting S&P 500 tickers...")
    tickers = get_sp500_tickers()
    print(f"   {len(tickers)} tickers")

    print("\n2. Downloading price data...")
    prices = download_stock_data(
        tickers, start="1999-01-01",
        cache_path=os.path.join(OUTPUT_DIR, "sp500_prices.parquet")
    )

    print("\n3. Downloading benchmark (SPY)...")
    benchmark = download_benchmark(start="1999-01-01")

    print("\n4. Grid Search V2...")
    gs = grid_search_v2(prices, benchmark)

    # Results
    print("\n" + "=" * 70)
    print("  RESULTS V2")
    print("=" * 70)

    print("\n  Best Parameters:")
    for k, v in gs["best_params"].items():
        print(f"    {k:18s}: {v}")

    print(f"\n  {'Metric':<12s} {'Train':>8s} {'Test':>8s} {'Bench_T':>8s} {'Full':>8s} {'Bench_F':>8s} {'V1(no-reg)':>10s}")
    print("  " + "-" * 70)
    for key in gs["train_metrics"]:
        vals = [gs[p].get(key, "-") for p in ["train_metrics", "test_metrics", "bench_test", "full_metrics", "bench_full", "v1_metrics"]]
        print(f"  {key:<12s}" + "".join(f"{str(v):>10s}" for v in vals))

    print(f"\n  Avg Annual Turnover: {gs['avg_turnover']}%")

    # MC
    print("\n  Monte Carlo (full period, 5000 sim)...")
    mc = monte_carlo_sharpe(gs["full_result"]["equity"])
    print(f"    Mean SR:     {mc['mean']:.3f}   95%CI: [{mc['ci_lo']:.3f}, {mc['ci_hi']:.3f}]")
    print(f"    P(SR > 1.0): {mc['p_gt_1']}%   P(SR > 1.5): {mc['p_gt_1.5']}%")

    # Plots
    print("\n5. Plots...")
    plot_v2(gs, save_dir=OUTPUT_DIR)

    # Save
    summary = {
        "best_params": gs["best_params"],
        "train": gs["train_metrics"], "test": gs["test_metrics"],
        "bench_test": gs["bench_test"], "full": gs["full_metrics"],
        "bench_full": gs["bench_full"], "v1_baseline": gs["v1_metrics"],
        "monte_carlo": mc, "avg_annual_turnover_pct": gs["avg_turnover"],
    }
    with open(os.path.join(OUTPUT_DIR, "v2_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n  ⚠️  DISCLAIMER:")
    print("  - Survivorship bias presente (componenti attuali S&P 500)")
    print("  - Grid search → rischio overfitting residuo, OOS è il vero test")
    print("  - Framework educativo, non consiglio finanziario\n")
    return gs


if __name__ == "__main__":
    gs = main()