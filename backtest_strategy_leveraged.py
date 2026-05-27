"""
Backtest v3 — Multi-Asset, Multi-Leva
=======================================
Modulo 6 del percorso "Certificati a Leva Fissa"

Testa la strategia v2 ottimizzata su:
  - Sottostanti: FTSE MIB, DAX, EuroStoxx 50, S&P 500, CAC 40
  - Leve: 3x, 5x, 7x

Produce:
  1. Tabella comparativa di tutte le combinazioni
  2. Heatmap profit factor per (sottostante, leva)
  3. Equity curves sovrapposte per leva sullo stesso sottostante
  4. Walk-forward OOS per ogni combinazione
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# CONFIGURAZIONE
# ============================================================
TICKERS = {
    'FTSE MIB':    'FTSEMIB.MI',
    #'DAX':         '^GDAXI',
    #'EuroStoxx50': '^STOXX50E',
    #'S&P 500':     '^GSPC',
    #'CAC 40':      '^FCHI',
    'Intesa Sanpaolo':   'ISP.MI',
    'UniCredit':         'UCG.MI',
    'Enel':              'ENEL.MI',
    'Eni':               'ENI.MI',
    'Ferrari':           'RACE.MI',
    'STMicroelectronics':'STMMI.MI',
    'Generali':          'G.MI',
    'Stellantis':        'STLAM.MI',
    'Tenaris':           'TEN.MI',
    'Prysmian':          'PRY.MI',
    'Leonardo':          'LDO.MI',
    'Banco BPM':         'BAMI.MI',
    'Mediobanca':        'MB.MI',
    'Moncler':           'MONC.MI',
    'Campari':           'CPR.MI',
    'Pirelli':           'PIRC.MI',
    'A2A':               'A2A.MI',
    'Saipem':            'SPM.MI',
    'Italgas':           'IG.MI',
    'BPER Banca':        'BPE.MI',
}

#LEVERAGE_LEVELS = [3, 5, 7]
LEVERAGE_LEVELS = [3,5]

RISK_PER_TRADE = 0.05
INITIAL_CAPITAL = 10_000
MAX_HOLDING_DAYS = 5
TARGET_MULTIPLIER = 1.5
STOP_LOSS_ATR_MULT = 1.5

# Scanner
ADX_THRESHOLD_LONG = 25
ADX_THRESHOLD_SHORT = 30
ADX_WEAK = 20
ROC_THRESHOLD_LONG = 0.5
ROC_THRESHOLD_SHORT = 0.8
ATR_VOL_LIMIT = 1.3

# Partial exit
PARTIAL_EXIT_THRESHOLD = 0.07
PARTIAL_EXIT_FRACTION = 0.5

# Costi (scalano con la leva)
BASE_BID_ASK = 0.008           # Spread base per leva 7x
ANNUAL_FUNDING_RATE = 0.035    # 3.5% annuo

# Walk-forward
WF_TRAIN_DAYS = 150
WF_TEST_DAYS = 50
WF_STEP_DAYS = 50

print("=" * 75)
print("BACKTEST v3 — MULTI-ASSET, MULTI-LEVA")
print("=" * 75)

# ============================================================
# FUNZIONI CORE (identiche a v2)
# ============================================================
def compute_indicators(df):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift(1)).abs(),
                     (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    atr_pct = (atr14 / c) * 100
    atr_pct_ma = atr_pct.rolling(20).mean()

    plus_dm = h.diff()
    minus_dm = -l.diff()
    plus_dm = pd.Series(np.where((plus_dm > minus_dm) & (plus_dm > 0),
                                  plus_dm, 0), index=df.index)
    minus_dm = pd.Series(np.where((minus_dm > plus_dm) & (minus_dm > 0),
                                   minus_dm, 0), index=df.index)
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).abs()
    adx = dx.rolling(14).mean()
    roc5 = ((c - c.shift(5)) / c.shift(5)) * 100
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_hist = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
    return adx, plus_di, minus_di, atr_pct, atr_pct_ma, roc5, macd_hist, atr14


def make_signal_fn(adx, atr_pct, atr_pct_ma, roc5, macd_hist, plus_di, minus_di):
    def get_signal(i):
        if i < 1:
            return None
        vals = [adx.iloc[i], atr_pct.iloc[i], atr_pct_ma.iloc[i],
                roc5.iloc[i], macd_hist.iloc[i], macd_hist.iloc[i-1],
                plus_di.iloc[i], minus_di.iloc[i]]
        if any(pd.isna(v) for v in vals):
            return None
        a, atr_v, atr_ma, roc, mh, mh_prev, pdi, mdi = vals
        if a < ADX_WEAK:
            return None
        if atr_v > atr_ma * ATR_VOL_LIMIT:
            return None
        vol_ok = atr_v <= atr_ma
        if pdi > mdi:
            direction = 'LONG'
            adx_ok = a >= ADX_THRESHOLD_LONG
            macd_exp = mh > mh_prev and mh > 0
            roc_ok = roc > ROC_THRESHOLD_LONG
        else:
            direction = 'SHORT'
            adx_ok = a >= ADX_THRESHOLD_SHORT
            macd_exp = mh < mh_prev and mh < 0
            roc_ok = roc < -ROC_THRESHOLD_SHORT
        if adx_ok and vol_ok and macd_exp and roc_ok:
            return direction
        return None
    return get_signal


class Trade:
    def __init__(self, entry_date, direction, entry_price, stop_price,
                 target_price, position_size, entry_idx):
        self.entry_date = entry_date
        self.direction = direction
        self.entry_price = entry_price
        self.stop_price = stop_price
        self.target_price = target_price
        self.position_size = position_size
        self.remaining_size = position_size
        self.entry_idx = entry_idx
        self.exit_date = None
        self.exit_price = None
        self.exit_reason = None
        self.pnl_pct = 0
        self.pnl_eur = 0
        self.holding_days = 0
        self.cert_return = 0
        self.partial_exit_done = False
        self.partial_pnl_eur = 0
        self.total_costs = 0


def run_backtest(df, start_idx, end_idx, capital, signal_fn, leverage,
                 bid_ask, daily_funding):
    trades = []
    equity = [capital]
    equity_dates = [df.index[start_idx]]
    current_capital = capital
    in_trade = None
    indicators = compute_indicators(df)
    atr14_local = indicators[7]

    for i in range(start_idx, min(end_idx, len(df) - 1)):
        date = df.index[i]
        close_today = df['Close'].iloc[i]

        if in_trade is not None:
            in_trade.holding_days += 1
            day_return = (df['Close'].iloc[i] - df['Close'].iloc[i-1]) / \
                         df['Close'].iloc[i-1]
            if in_trade.direction == 'LONG':
                lev_return = leverage * day_return
            else:
                lev_return = -leverage * day_return

            lev_return -= daily_funding
            in_trade.total_costs += in_trade.remaining_size * daily_funding

            in_trade.cert_return = (1 + in_trade.cert_return) * \
                                   (1 + lev_return) - 1

            # Partial exit
            if (not in_trade.partial_exit_done and
                    in_trade.cert_return >= PARTIAL_EXIT_THRESHOLD):
                partial_size = in_trade.remaining_size * PARTIAL_EXIT_FRACTION
                partial_pnl = partial_size * in_trade.cert_return
                partial_cost = partial_size * bid_ask / 2
                partial_pnl -= partial_cost
                in_trade.total_costs += partial_cost
                in_trade.partial_pnl_eur += partial_pnl
                current_capital += partial_pnl
                in_trade.remaining_size -= partial_size
                in_trade.partial_exit_done = True
                in_trade.stop_price = in_trade.entry_price

            exit_now = False
            reason = None

            if in_trade.direction == 'LONG' and close_today <= in_trade.stop_price:
                exit_now = True
                reason = 'STOP_BE' if in_trade.partial_exit_done else 'STOP_LOSS'
            elif in_trade.direction == 'SHORT' and close_today >= in_trade.stop_price:
                exit_now = True
                reason = 'STOP_BE' if in_trade.partial_exit_done else 'STOP_LOSS'

            if in_trade.direction == 'LONG' and close_today >= in_trade.target_price:
                exit_now, reason = True, 'TARGET'
            elif in_trade.direction == 'SHORT' and close_today <= in_trade.target_price:
                exit_now, reason = True, 'TARGET'

            if in_trade.holding_days >= MAX_HOLDING_DAYS:
                exit_now, reason = True, 'TIME_STOP'

            signal = signal_fn(i)
            if in_trade.direction == 'LONG' and signal != 'LONG':
                if in_trade.holding_days >= 2:
                    exit_now, reason = True, 'REGIME_EXIT'
            elif in_trade.direction == 'SHORT' and signal != 'SHORT':
                if in_trade.holding_days >= 2:
                    exit_now, reason = True, 'REGIME_EXIT'

            if exit_now:
                in_trade.exit_date = date
                in_trade.exit_price = close_today
                remaining_pnl = in_trade.remaining_size * in_trade.cert_return
                exit_cost = in_trade.remaining_size * bid_ask / 2
                remaining_pnl -= exit_cost
                in_trade.total_costs += exit_cost
                in_trade.exit_reason = reason
                in_trade.pnl_eur = in_trade.partial_pnl_eur + remaining_pnl
                in_trade.pnl_pct = (in_trade.pnl_eur / in_trade.position_size) * 100
                current_capital += remaining_pnl
                trades.append(in_trade)
                in_trade = None

        elif in_trade is None:
            signal = signal_fn(i)
            if signal is not None:
                atr_val = atr14_local.iloc[i]
                if pd.isna(atr_val) or atr_val <= 0:
                    continue
                stop_dist_pct = (STOP_LOSS_ATR_MULT * atr_val) / close_today
                if stop_dist_pct < 0.003:
                    stop_dist_pct = 0.003
                if stop_dist_pct > 0.03:
                    continue

                max_loss = current_capital * RISK_PER_TRADE
                pos_size = max_loss / (leverage * stop_dist_pct)
                pos_size = min(pos_size, current_capital * 0.25)

                if signal == 'LONG':
                    stop_p = close_today * (1 - stop_dist_pct)
                    target_p = close_today * (1 + stop_dist_pct * TARGET_MULTIPLIER)
                else:
                    stop_p = close_today * (1 + stop_dist_pct)
                    target_p = close_today * (1 - stop_dist_pct * TARGET_MULTIPLIER)

                trade = Trade(
                    entry_date=date, direction=signal,
                    entry_price=close_today, stop_price=stop_p,
                    target_price=target_p, position_size=pos_size,
                    entry_idx=i)
                trade.cert_return = 0
                entry_cost = pos_size * bid_ask / 2
                current_capital -= entry_cost
                trade.total_costs = entry_cost
                in_trade = trade

        equity.append(current_capital)
        equity_dates.append(date)

    if in_trade is not None:
        in_trade.exit_date = df.index[min(end_idx, len(df)-1)]
        in_trade.exit_price = df['Close'].iloc[min(end_idx, len(df)-1)]
        in_trade.exit_reason = 'END_OF_PERIOD'
        remaining_pnl = in_trade.remaining_size * in_trade.cert_return
        exit_cost = in_trade.remaining_size * bid_ask / 2
        remaining_pnl -= exit_cost
        in_trade.total_costs += exit_cost
        in_trade.pnl_eur = in_trade.partial_pnl_eur + remaining_pnl
        in_trade.pnl_pct = (in_trade.pnl_eur / in_trade.position_size) * 100
        current_capital += remaining_pnl
        trades.append(in_trade)
        equity.append(current_capital)
        equity_dates.append(in_trade.exit_date)

    return trades, equity, equity_dates, current_capital


def calc_metrics(trades, initial_cap, final_cap):
    if not trades:
        return {'n_trades': 0, 'win_rate': 0, 'expectancy': 0,
                'profit_factor': 0, 'total_return': 0, 'max_dd': 0,
                'sharpe': 0, 'sortino': 0, 'holding': 0, 'costs': 0,
                'n_long': 0, 'n_short': 0, 'n_partial': 0}

    pnls = [t.pnl_eur for t in trades]
    pnl_pcts = [t.pnl_pct for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    holding = [t.holding_days for t in trades]

    n = len(trades)
    wr = len(wins) / n * 100
    pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 99
    exp = np.mean(pnls)
    ret = (final_cap / initial_cap - 1) * 100

    eq_arr = np.array([initial_cap] + [initial_cap + sum(pnls[:i+1])
                       for i in range(len(pnls))])
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak * 100
    max_dd = dd.min()

    tpy = 252 / np.mean(holding) if np.mean(holding) > 0 else 50
    sharpe = (np.mean(pnl_pcts) / np.std(pnl_pcts)) * np.sqrt(tpy) \
        if len(pnl_pcts) > 1 and np.std(pnl_pcts) > 0 else 0
    down = [p for p in pnl_pcts if p < 0]
    sortino = (np.mean(pnl_pcts) / np.std(down)) * np.sqrt(tpy) \
        if down and np.std(down) > 0 else 0

    return {
        'n_trades': n, 'win_rate': wr, 'expectancy': exp,
        'profit_factor': pf, 'total_return': ret, 'max_dd': max_dd,
        'sharpe': sharpe, 'sortino': sortino,
        'holding': np.mean(holding),
        'costs': sum(t.total_costs for t in trades),
        'n_long': sum(1 for t in trades if t.direction == 'LONG'),
        'n_short': sum(1 for t in trades if t.direction == 'SHORT'),
        'n_partial': sum(1 for t in trades if t.partial_exit_done),
    }


# ============================================================
# SCARICA DATI
# ============================================================
print(f"\nScaricando dati per {len(TICKERS)} sottostanti...")
end = datetime.now()
start = end - timedelta(days=1500)
datasets = {}

for name, ticker in TICKERS.items():
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if hasattr(df.columns, 'levels'):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        if len(df) > 100:
            datasets[name] = df
            print(f"  {name:15s} ({ticker:12s}): {len(df)} righe")
        else:
            print(f"  {name:15s} ({ticker:12s}): SKIP (solo {len(df)} righe)")
    except Exception as e:
        print(f"  {name:15s} ({ticker:12s}): ERRORE — {e}")

# ============================================================
# RUN BACKTESTS
# ============================================================
print(f"\n{'═' * 75}")
print("ESECUZIONE BACKTEST: "
      f"{len(datasets)} asset x {len(LEVERAGE_LEVELS)} leve "
      f"= {len(datasets) * len(LEVERAGE_LEVELS)} combinazioni")
print(f"{'═' * 75}")

all_results = {}
all_equities = {}

for asset_name, df in datasets.items():
    # Calcola indicatori una volta per asset
    adx, plus_di, minus_di, atr_pct, atr_pct_ma, roc5, macd_hist, atr14 = \
        compute_indicators(df)
    signal_fn = make_signal_fn(adx, atr_pct, atr_pct_ma, roc5,
                                macd_hist, plus_di, minus_di)
    start_bt = 60

    for lev in LEVERAGE_LEVELS:
        # Costi scalano con la leva
        bid_ask = BASE_BID_ASK * (lev / 7)  # Spread proporzionale alla leva
        daily_funding = ANNUAL_FUNDING_RATE / 252

        key = (asset_name, lev)
        print(f"\n  {asset_name} — Leva {lev}x "
              f"(spread={bid_ask*100:.2f}%, funding={daily_funding*100:.4f}%/gg)")

        # Full backtest
        trades, equity, eq_dates, final_cap = run_backtest(
            df, start_bt, len(df), INITIAL_CAPITAL, signal_fn,
            lev, bid_ask, daily_funding)
        m = calc_metrics(trades, INITIAL_CAPITAL, final_cap)
        all_results[key] = m
        all_equities[key] = (eq_dates, equity)

        print(f"    Trades: {m['n_trades']} (L:{m['n_long']} S:{m['n_short']}) | "
              f"WR: {m['win_rate']:.0f}% | PF: {m['profit_factor']:.2f} | "
              f"Return: {m['total_return']:+.2f}% | "
              f"MaxDD: {m['max_dd']:.2f}% | Sharpe: {m['sharpe']:.2f}")

        # Walk-forward
        wf_trades = []
        wf_start = start_bt
        while wf_start + WF_TRAIN_DAYS + WF_TEST_DAYS <= len(df):
            train_end = wf_start + WF_TRAIN_DAYS
            test_end = train_end + WF_TEST_DAYS
            t_trades, _, _, _ = run_backtest(
                df, train_end, test_end, INITIAL_CAPITAL, signal_fn,
                lev, bid_ask, daily_funding)
            wf_trades.extend(t_trades)
            wf_start += WF_STEP_DAYS

        if wf_trades:
            wf_final = INITIAL_CAPITAL + sum(t.pnl_eur for t in wf_trades)
            wf_m = calc_metrics(wf_trades, INITIAL_CAPITAL, wf_final)
            all_results[(asset_name, lev, 'OOS')] = wf_m
            print(f"    WF OOS: Trades: {wf_m['n_trades']} | "
                  f"WR: {wf_m['win_rate']:.0f}% | PF: {wf_m['profit_factor']:.2f} | "
                  f"Return: {wf_m['total_return']:+.2f}% | "
                  f"Sharpe: {wf_m['sharpe']:.2f}")

# ============================================================
# TABELLA COMPARATIVA
# ============================================================
print(f"\n{'═' * 75}")
print("TABELLA COMPARATIVA — FULL BACKTEST (in-sample)")
print(f"{'═' * 75}")
print(f"\n{'Asset':<15} {'Leva':>5} {'Trades':>7} {'WR%':>6} {'PF':>6} "
      f"{'Return%':>9} {'MaxDD%':>8} {'Sharpe':>7} {'Sortino':>8} "
      f"{'Costi':>8}")
print("─" * 85)

for asset_name in datasets.keys():
    for lev in LEVERAGE_LEVELS:
        key = (asset_name, lev)
        m = all_results.get(key, {})
        if not m or m['n_trades'] == 0:
            continue
        print(f"{asset_name:<15} {lev:>4}x {m['n_trades']:>7} "
              f"{m['win_rate']:>5.0f}% {m['profit_factor']:>6.2f} "
              f"{m['total_return']:>+8.2f}% {m['max_dd']:>7.2f}% "
              f"{m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
              f"{m['costs']:>7.0f}E")
    print()

print(f"\n{'═' * 75}")
print("TABELLA COMPARATIVA — WALK-FORWARD OOS")
print(f"{'═' * 75}")
print(f"\n{'Asset':<15} {'Leva':>5} {'Trades':>7} {'WR%':>6} {'PF':>6} "
      f"{'Return%':>9} {'MaxDD%':>8} {'Sharpe':>7}")
print("─" * 65)

for asset_name in datasets.keys():
    for lev in LEVERAGE_LEVELS:
        key = (asset_name, lev, 'OOS')
        m = all_results.get(key, {})
        if not m or m['n_trades'] == 0:
            continue
        print(f"{asset_name:<15} {lev:>4}x {m['n_trades']:>7} "
              f"{m['win_rate']:>5.0f}% {m['profit_factor']:>6.2f} "
              f"{m['total_return']:>+8.2f}% {m['max_dd']:>7.2f}% "
              f"{m['sharpe']:>7.2f}")
    print()

# ============================================================
# IDENTIFICAZIONE MIGLIORI COMBINAZIONI
# ============================================================
print(f"\n{'═' * 75}")
print("RANKING — Migliori combinazioni per Sharpe OOS")
print(f"{'═' * 75}")

oos_ranked = []
for asset_name in datasets.keys():
    for lev in LEVERAGE_LEVELS:
        key = (asset_name, lev, 'OOS')
        m = all_results.get(key, {})
        if m and m['n_trades'] >= 5:
            oos_ranked.append((asset_name, lev, m))

oos_ranked.sort(key=lambda x: x[2]['sharpe'], reverse=True)

print(f"\n{'#':>3} {'Asset':<15} {'Leva':>5} {'Sharpe':>8} {'PF':>6} "
      f"{'Return%':>9} {'MaxDD%':>8} {'Trades':>7}")
print("─" * 65)
for rank, (name, lev, m) in enumerate(oos_ranked[:10], 1):
    print(f"{rank:>3} {name:<15} {lev:>4}x {m['sharpe']:>8.2f} "
          f"{m['profit_factor']:>6.2f} {m['total_return']:>+8.2f}% "
          f"{m['max_dd']:>7.2f}% {m['n_trades']:>7}")

# ============================================================
# GRAFICI
# ============================================================
n_assets = len(datasets)
fig = plt.figure(figsize=(18, 6 * n_assets + 8))
fig.patch.set_facecolor('white')
gs = GridSpec(n_assets + 2, 2, figure=fig, hspace=0.4, wspace=0.3)

# Equity curves per asset (un pannello per asset, 3 leve sovrapposte)
lev_colors = {3: '#378ADD', 5: '#EF9F27', 7: '#D85A30'}

for idx, asset_name in enumerate(datasets.keys()):
    ax = fig.add_subplot(gs[idx, :])
    for lev in LEVERAGE_LEVELS:
        key = (asset_name, lev)
        if key in all_equities:
            eq_dates, equity = all_equities[key]
            m = all_results.get(key, {})
            ret_str = f"{m.get('total_return', 0):+.1f}%"
            ax.plot(eq_dates, equity, color=lev_colors[lev],
                    linewidth=1.5 if lev == 7 else 1.2,
                    label=f'Leva {lev}x ({ret_str})')
    ax.axhline(INITIAL_CAPITAL, color='#888780', linewidth=0.5,
               linestyle='--')
    ax.set_title(f'{asset_name}', fontsize=14, fontweight='500')
    ax.set_ylabel('EUR', fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.tick_params(labelsize=10)

# Heatmap: Sharpe OOS per (asset, leva)
ax_heat = fig.add_subplot(gs[n_assets, 0])
asset_names = list(datasets.keys())
heatmap_data = np.zeros((len(asset_names), len(LEVERAGE_LEVELS)))

for i, name in enumerate(asset_names):
    for j, lev in enumerate(LEVERAGE_LEVELS):
        key = (name, lev, 'OOS')
        m = all_results.get(key, {})
        heatmap_data[i, j] = m.get('sharpe', 0)

im = ax_heat.imshow(heatmap_data, aspect='auto', cmap='RdYlGn',
                     vmin=-2, vmax=4)
ax_heat.set_xticks(range(len(LEVERAGE_LEVELS)))
ax_heat.set_xticklabels([f'{l}x' for l in LEVERAGE_LEVELS], fontsize=12)
ax_heat.set_yticks(range(len(asset_names)))
ax_heat.set_yticklabels(asset_names, fontsize=11)
for i in range(len(asset_names)):
    for j in range(len(LEVERAGE_LEVELS)):
        val = heatmap_data[i, j]
        color = 'white' if abs(val) > 1.5 else 'black'
        ax_heat.text(j, i, f'{val:.2f}', ha='center', va='center',
                     fontsize=13, fontweight='500', color=color)
ax_heat.set_title('Sharpe OOS per asset e leva', fontsize=14, fontweight='500')
plt.colorbar(im, ax=ax_heat, shrink=0.8)

# Heatmap: Profit Factor OOS
ax_pf = fig.add_subplot(gs[n_assets, 1])
pf_data = np.zeros((len(asset_names), len(LEVERAGE_LEVELS)))

for i, name in enumerate(asset_names):
    for j, lev in enumerate(LEVERAGE_LEVELS):
        key = (name, lev, 'OOS')
        m = all_results.get(key, {})
        pf_data[i, j] = min(m.get('profit_factor', 0), 5)

im2 = ax_pf.imshow(pf_data, aspect='auto', cmap='RdYlGn',
                     vmin=0, vmax=3)
ax_pf.set_xticks(range(len(LEVERAGE_LEVELS)))
ax_pf.set_xticklabels([f'{l}x' for l in LEVERAGE_LEVELS], fontsize=12)
ax_pf.set_yticks(range(len(asset_names)))
ax_pf.set_yticklabels(asset_names, fontsize=11)
for i in range(len(asset_names)):
    for j in range(len(LEVERAGE_LEVELS)):
        val = pf_data[i, j]
        color = 'white' if val > 2 else 'black'
        ax_pf.text(j, i, f'{val:.2f}', ha='center', va='center',
                   fontsize=13, fontweight='500', color=color)
ax_pf.set_title('Profit factor OOS per asset e leva',
                fontsize=14, fontweight='500')
plt.colorbar(im2, ax=ax_pf, shrink=0.8)

# Scatter: Sharpe vs MaxDD (ogni punto = una combinazione)
ax_sc = fig.add_subplot(gs[n_assets + 1, :])
for asset_name in datasets.keys():
    for lev in LEVERAGE_LEVELS:
        key = (asset_name, lev, 'OOS')
        m = all_results.get(key, {})
        if m and m['n_trades'] >= 3:
            ax_sc.scatter(abs(m['max_dd']), m['sharpe'],
                         s=lev * 30, color=lev_colors[lev], alpha=0.7,
                         edgecolors='#2C2C2A', linewidth=0.5)
            ax_sc.annotate(f"{asset_name}\n{lev}x",
                          (abs(m['max_dd']), m['sharpe']),
                          fontsize=8, ha='center', va='bottom')

ax_sc.axhline(0, color='#888780', linewidth=0.5, linestyle='--')
ax_sc.axhline(1, color='#639922', linewidth=0.5, linestyle=':', alpha=0.5)
ax_sc.set_xlabel('Max drawdown OOS (%)', fontsize=12)
ax_sc.set_ylabel('Sharpe OOS', fontsize=12)
ax_sc.set_title('Sharpe vs max drawdown OOS (dimensione = leva)',
                fontsize=14, fontweight='500')
ax_sc.tick_params(labelsize=10)

fig.suptitle('Backtest v3 — Multi-asset, multi-leva\n'
             f'{len(datasets)} sottostanti x {len(LEVERAGE_LEVELS)} leve '
             f'| Walk-forward OOS | Costi reali inclusi',
             fontsize=15, fontweight='500', y=0.995)

plt.savefig('backtest_results.png', dpi=150,
            bbox_inches='tight', facecolor='white')
print(f"\n[Grafico salvato: backtest_results.png]")

print(f"\n{'═' * 75}")
print("BACKTEST v3 COMPLETATO")
print(f"{'═' * 75}")