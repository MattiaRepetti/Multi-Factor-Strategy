"""
Stock Screener per Trading con Certificati a Leva
===================================================
Modulo 6 (estensione) — Screening single stocks

Valuta le azioni di un universo (es. FTSE MIB, DAX) su 4 criteri:
  1. Volatilita relativa (ATR% giornaliero)
  2. Trend quality (autocorrelazione rendimenti + ADX medio)
  3. Liquidita (volume medio giornaliero)
  4. Rischio gap (frequenza di gap > 2%)

Produce un ranking con punteggio composito e identifica i candidati
piu adatti per operativita con leva.

NOTA: richiede connessione internet per scaricare dati da Yahoo Finance.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# UNIVERSO DI AZIONI
# ============================================================
# FTSE MIB — principali componenti
FTSE_MIB_STOCKS = {
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

# DAX — principali componenti
DAX_STOCKS = {
    'SAP':               'SAP.DE',
    'Siemens':           'SIE.DE',
    'Allianz':           'ALV.DE',
    'Deutsche Telekom':  'DTE.DE',
    'BMW':               'BMW.DE',
    'BASF':              'BAS.DE',
    'Adidas':            'ADS.DE',
    'Bayer':             'BAYN.DE',
    'Munich Re':         'MUV2.DE',
    'Infineon':          'IFX.DE',
    'Deutsche Bank':     'DBK.DE',
    'Mercedes-Benz':     'MBG.DE',
    'Rheinmetall':       'RHM.DE',
    'Vonovia':           'VNA.DE',
    'Heidelberg Mat.':   'HEI.DE',
}

# Scegli quale universo analizzare
UNIVERSE = {**FTSE_MIB_STOCKS, **DAX_STOCKS}
LOOKBACK_DAYS = 500  # ~2 anni di dati

print("=" * 75)
print("STOCK SCREENER — Candidati per trading a leva")
print("=" * 75)

# ============================================================
# SCARICA DATI
# ============================================================
print(f"\nScaricando dati per {len(UNIVERSE)} azioni...")
end = datetime.now()
start = end - timedelta(days=LOOKBACK_DAYS + 100)

stock_data = {}
for name, ticker in UNIVERSE.items():
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if hasattr(df.columns, 'levels'):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        if len(df) > 200:
            stock_data[name] = df
    except Exception:
        pass

print(f"  Scaricati con successo: {len(stock_data)}/{len(UNIVERSE)} titoli\n")

# ============================================================
# CALCOLO METRICHE DI SCREENING
# ============================================================
results = []

for name, df in stock_data.items():
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    daily_returns = close.pct_change().dropna()

    # --- 1. VOLATILITA RELATIVA (ATR%) ---
    tr = pd.concat([high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    atr_pct = (atr14 / close * 100).dropna()
    avg_atr_pct = atr_pct.iloc[-60:].mean()  # Media ultimi 3 mesi

    # --- 2. TREND QUALITY ---
    # 2a. Autocorrelazione lag-1 dei rendimenti (positiva = trend persistente)
    autocorr_1 = daily_returns.iloc[-252:].autocorr(lag=1)

    # 2b. Autocorrelazione lag-5 (trend multi-day)
    autocorr_5 = daily_returns.iloc[-252:].autocorr(lag=5)

    # 2c. ADX medio ultimi 3 mesi
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = pd.Series(np.where((plus_dm > minus_dm) & (plus_dm > 0),
                                  plus_dm, 0), index=df.index)
    minus_dm = pd.Series(np.where((minus_dm > plus_dm) & (minus_dm > 0),
                                   minus_dm, 0), index=df.index)
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).abs()
    adx = dx.rolling(14).mean()
    avg_adx = adx.iloc[-60:].mean()

    # 2d. Percentuale di giorni con ADX > 25 (negli ultimi 6 mesi)
    adx_recent = adx.iloc[-126:]
    pct_trending = (adx_recent > 25).sum() / len(adx_recent) * 100

    # --- 3. LIQUIDITA ---
    avg_volume = volume.iloc[-60:].mean()
    avg_turnover = (volume * close).iloc[-60:].mean()  # Volume in EUR

    # --- 4. RISCHIO GAP ---
    # Gap = differenza tra apertura e chiusura precedente
    gaps = ((df['Open'] - close.shift(1)) / close.shift(1) * 100).dropna()
    gaps_recent = gaps.iloc[-252:]
    large_gaps = (gaps_recent.abs() > 2).sum()  # Gap > 2%
    very_large_gaps = (gaps_recent.abs() > 4).sum()  # Gap > 4%
    avg_gap = gaps_recent.abs().mean()
    max_gap = gaps_recent.abs().max()

    # --- 5. CORRELAZIONE CON L'INDICE ---
    # (Calcolata dopo per chi ha un indice di riferimento)

    # --- 6. SHARPE GIORNALIERO ---
    sharpe_daily = daily_returns.iloc[-252:].mean() / daily_returns.iloc[-252:].std() \
        if daily_returns.iloc[-252:].std() > 0 else 0

    # --- 7. RENDIMENTO ANNUO ---
    annual_return = (close.iloc[-1] / close.iloc[-252] - 1) * 100 \
        if len(close) > 252 else 0

    results.append({
        'name': name,
        'ticker': UNIVERSE[name],
        'avg_atr_pct': avg_atr_pct,
        'autocorr_1': autocorr_1,
        'autocorr_5': autocorr_5,
        'avg_adx': avg_adx,
        'pct_trending': pct_trending,
        'avg_volume': avg_volume,
        'avg_turnover': avg_turnover,
        'large_gaps': large_gaps,
        'very_large_gaps': very_large_gaps,
        'avg_gap': avg_gap,
        'max_gap': max_gap,
        'sharpe_daily': sharpe_daily,
        'annual_return': annual_return,
    })

df_results = pd.DataFrame(results)

# ============================================================
# SCORING COMPOSITO
# ============================================================
# Normalizza ogni metrica tra 0 e 1 (0 = peggiore, 1 = migliore)
def normalize(series, higher_is_better=True):
    if series.max() == series.min():
        return pd.Series(0.5, index=series.index)
    norm = (series - series.min()) / (series.max() - series.min())
    return norm if higher_is_better else (1 - norm)

# Pesi dei criteri
W_VOLATILITY = 0.25     # Bassa volatilita = meglio
W_TREND = 0.30           # Trend quality alto = meglio
W_LIQUIDITY = 0.15       # Alta liquidita = meglio
W_GAP_RISK = 0.20        # Basso rischio gap = meglio
W_MOMENTUM = 0.10        # Sharpe positivo = meglio

df_results['score_vol'] = normalize(df_results['avg_atr_pct'],
                                     higher_is_better=False)
df_results['score_trend'] = (
    normalize(df_results['avg_adx']) * 0.4 +
    normalize(df_results['pct_trending']) * 0.3 +
    normalize(df_results['autocorr_1']) * 0.3
)
df_results['score_liq'] = normalize(df_results['avg_turnover'])
df_results['score_gap'] = normalize(df_results['large_gaps'],
                                     higher_is_better=False)
df_results['score_momentum'] = normalize(df_results['sharpe_daily'])

df_results['total_score'] = (
    W_VOLATILITY * df_results['score_vol'] +
    W_TREND * df_results['score_trend'] +
    W_LIQUIDITY * df_results['score_liq'] +
    W_GAP_RISK * df_results['score_gap'] +
    W_MOMENTUM * df_results['score_momentum']
)

df_results = df_results.sort_values('total_score', ascending=False)

# ============================================================
# OUTPUT
# ============================================================
print(f"{'═' * 75}")
print("RANKING COMPLETO")
print(f"{'═' * 75}")
print(f"\n{'#':>3} {'Nome':<22} {'Ticker':<12} {'ATR%':>6} {'ADX':>5} "
      f"{'Trend%':>7} {'Gaps>2%':>8} {'MaxGap':>7} {'Sharpe':>7} "
      f"{'Score':>7}")
print("─" * 95)

for rank, (_, row) in enumerate(df_results.iterrows(), 1):
    flag = " ***" if rank <= 5 else ""
    print(f"{rank:>3} {row['name']:<22} {row['ticker']:<12} "
          f"{row['avg_atr_pct']:>5.2f}% {row['avg_adx']:>5.1f} "
          f"{row['pct_trending']:>6.1f}% {row['large_gaps']:>7.0f} "
          f"{row['max_gap']:>6.1f}% {row['sharpe_daily']:>7.3f} "
          f"{row['total_score']:>7.3f}{flag}")

# Top 5 dettaglio
print(f"\n{'═' * 75}")
print("TOP 5 — DETTAGLIO CANDIDATI")
print(f"{'═' * 75}")

for rank, (_, row) in enumerate(df_results.head(5).iterrows(), 1):
    print(f"\n  #{rank} — {row['name']} ({row['ticker']})")
    print(f"  {'─' * 50}")
    print(f"    Volatilita:   ATR% medio = {row['avg_atr_pct']:.2f}%"
          f"  (score: {row['score_vol']:.2f})")
    print(f"    Trend:        ADX medio = {row['avg_adx']:.1f}, "
          f"trending {row['pct_trending']:.0f}% del tempo"
          f"  (score: {row['score_trend']:.2f})")
    print(f"    Liquidita:    Turnover medio = "
          f"{row['avg_turnover']/1e6:.1f}M EUR/gg"
          f"  (score: {row['score_liq']:.2f})")
    print(f"    Gap risk:     {row['large_gaps']:.0f} gap > 2% in un anno, "
          f"max gap = {row['max_gap']:.1f}%"
          f"  (score: {row['score_gap']:.2f})")
    print(f"    Momentum:     Sharpe daily = {row['sharpe_daily']:.3f}"
          f"  (score: {row['score_momentum']:.2f})")
    print(f"    Autocorr(1):  {row['autocorr_1']:.3f} "
          f"({'trend persistente' if row['autocorr_1'] > 0 else 'mean-reverting'})")
    print(f"    Autocorr(5):  {row['autocorr_5']:.3f}")
    print(f"    Rendimento 1Y:{row['annual_return']:+.1f}%")
    print(f"    SCORE TOTALE: {row['total_score']:.3f}")

# Analisi per mercato
print(f"\n{'═' * 75}")
print("MEDIA PER MERCATO")
print(f"{'═' * 75}")

for market, tickers in [('FTSE MIB', FTSE_MIB_STOCKS),
                          ('DAX', DAX_STOCKS)]:
    names_in_market = [n for n in tickers.keys() if n in df_results['name'].values]
    market_df = df_results[df_results['name'].isin(names_in_market)]
    if len(market_df) > 0:
        print(f"\n  {market} ({len(market_df)} titoli):")
        print(f"    ATR% medio:     {market_df['avg_atr_pct'].mean():.2f}%")
        print(f"    ADX medio:      {market_df['avg_adx'].mean():.1f}")
        print(f"    Gap >2% medio:  {market_df['large_gaps'].mean():.1f}/anno")
        print(f"    Score medio:    {market_df['total_score'].mean():.3f}")
        print(f"    Migliore:       {market_df.iloc[0]['name']} "
              f"(score {market_df.iloc[0]['total_score']:.3f})")

# ============================================================
# GRAFICI
# ============================================================
fig = plt.figure(figsize=(16, 18))
fig.patch.set_facecolor('white')
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

# 1. Score ranking
ax1 = fig.add_subplot(gs[0, :])
top_n = min(20, len(df_results))
top_df = df_results.head(top_n).iloc[::-1]

colors_bar = []
for _, row in top_df.iterrows():
    if row['ticker'].endswith('.MI'):
        colors_bar.append('#378ADD')
    else:
        colors_bar.append('#D85A30')

ax1.barh(range(top_n), top_df['total_score'], color=colors_bar, alpha=0.7)
ax1.set_yticks(range(top_n))
ax1.set_yticklabels([f"{r['name']} ({r['ticker']})"
                     for _, r in top_df.iterrows()], fontsize=9)
ax1.set_xlabel('Score composito', fontsize=12)
ax1.set_title('Ranking azioni — Score composito (blu=IT, arancio=DE)',
              fontsize=14, fontweight='500')
ax1.tick_params(labelsize=10)

# 2. ATR% vs ADX scatter
ax2 = fig.add_subplot(gs[1, 0])
for _, row in df_results.iterrows():
    c = '#378ADD' if row['ticker'].endswith('.MI') else '#D85A30'
    ax2.scatter(row['avg_atr_pct'], row['avg_adx'],
               s=row['total_score'] * 200, color=c, alpha=0.6,
               edgecolors='#2C2C2A', linewidth=0.3)
    if row['total_score'] >= df_results['total_score'].quantile(0.8):
        ax2.annotate(row['name'], (row['avg_atr_pct'], row['avg_adx']),
                    fontsize=7, ha='center', va='bottom')

ax2.set_xlabel('ATR% giornaliero (volatilita)', fontsize=11)
ax2.set_ylabel('ADX medio (forza trend)', fontsize=11)
ax2.set_title('Volatilita vs trend quality\n(angolo alto-sx = ideale)',
              fontsize=13, fontweight='500')
ax2.axhline(25, color='#639922', linewidth=0.5, linestyle=':', alpha=0.5)

# 3. Gap risk vs Score
ax3 = fig.add_subplot(gs[1, 1])
for _, row in df_results.iterrows():
    c = '#378ADD' if row['ticker'].endswith('.MI') else '#D85A30'
    ax3.scatter(row['large_gaps'], row['total_score'],
               s=80, color=c, alpha=0.6,
               edgecolors='#2C2C2A', linewidth=0.3)
    if row['total_score'] >= df_results['total_score'].quantile(0.8):
        ax3.annotate(row['name'], (row['large_gaps'], row['total_score']),
                    fontsize=7, ha='center', va='bottom')

ax3.set_xlabel('Numero gap > 2% (ultimo anno)', fontsize=11)
ax3.set_ylabel('Score composito', fontsize=11)
ax3.set_title('Rischio gap vs score\n(angolo alto-sx = ideale)',
              fontsize=13, fontweight='500')

# 4. Score breakdown top 10
ax4 = fig.add_subplot(gs[2, :])
top10 = df_results.head(10)
x = np.arange(10)
w = 0.15

ax4.bar(x - 2*w, top10['score_vol'], w, label='Bassa vol',
        color='#378ADD', alpha=0.7)
ax4.bar(x - w, top10['score_trend'], w, label='Trend quality',
        color='#639922', alpha=0.7)
ax4.bar(x, top10['score_liq'], w, label='Liquidita',
        color='#EF9F27', alpha=0.7)
ax4.bar(x + w, top10['score_gap'], w, label='Basso gap risk',
        color='#D85A30', alpha=0.7)
ax4.bar(x + 2*w, top10['score_momentum'], w, label='Momentum',
        color='#534AB7', alpha=0.7)

ax4.set_xticks(x)
ax4.set_xticklabels(top10['name'], rotation=45, ha='right', fontsize=9)
ax4.set_ylabel('Score (0-1)', fontsize=11)
ax4.set_title('Decomposizione score — Top 10', fontsize=14, fontweight='500')
ax4.legend(fontsize=9, ncol=5, loc='upper right')

fig.suptitle('Stock Screener per trading a leva\n'
             f'{len(stock_data)} azioni analizzate (FTSE MIB + DAX)',
             fontsize=15, fontweight='500', y=0.995)

plt.savefig('screener_results.png', dpi=150,
            bbox_inches='tight', facecolor='white')
print(f"\n[Grafico salvato: screener_results.png]")

# ============================================================
# AVVERTENZE
# ============================================================
print(f"\n{'═' * 75}")
print("AVVERTENZE IMPORTANTI PER SINGOLE AZIONI")
print(f"{'═' * 75}")
print("""
  1. Verifica che esistano certificati a leva fissa su queste azioni
     sul sito di SocGen (prodotti.societegenerale.it) o altri emittenti.
     La maggior parte dei certificati a leva fissa e su indici, non azioni.

  2. Se non esistono certificati, le alternative sono:
     - Turbo certificates (SocGen, BNP, UniCredit)
     - Mini-futures
     - CFD (attenzione: non quotati su mercati regolamentati)

  3. Il rischio gap su singole azioni e MOLTO piu alto che su indici.
     Mai tenere posizioni overnight su azioni durante earnings season
     o prima di eventi corporate (ex-dividendo, M&A, ecc.)

  4. Lo screening deve essere rieseguito periodicamente (mensile)
     perche le caratteristiche di volatilita e trend cambiano.
""")

print(f"{'═' * 75}")
print("SCREENER COMPLETATO")
print(f"{'═' * 75}")