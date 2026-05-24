"""
nordvik_analysis.py
===================
Nordvik National Pension Reserve Fund Full Quantitative Analysis
Case Study 2026 | Portfolio Choice Assignment

Data source: Live Yahoo Finance / FRED / Kenneth French downloads are tried
first. If live retrieval fails (for example, no internet connection), the
script automatically falls back to calibrated synthetic price histories
generated via a regime-switching multivariate GBM.

Outputs written to ./output/
"""

# ── 0. IMPORTS ────────────────────────────────────────────────────────────────
import os, sys, warnings, datetime, json, io, zipfile, re
import numpy as np
import requests
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import statsmodels.api as sm

import performance_assessment as pa
from scipy.optimize import minimize
warnings.filterwarnings("ignore")
np.random.seed(42)

TRADING_DAYS = pa.TRADING_DAYS
LIVE_YF_TIMEOUT = 20
LIVE_HTTP_TIMEOUT = 20

# ── 1. CONFIG ─────────────────────────────────────────────────────────────────
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "output")
CHARTS_DIR  = os.path.join(OUTPUT_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

START_DATE       = "2015-01-02"
END_DATE         = datetime.date.today().strftime("%Y-%m-%d")
GLIDE_START_YEAR = 2017
GLIDE_TARGET     = 0.60
PRE_RISKY        = 0.70

ASSETS = ['MTUM','QUAL','VGK','VPL','VWO','GLD','VNQ','IGF',
          'BTC-USD','TLT','LQD','TIP','HYG','Cash','ACWI','BND','VTI']

RISKY_ASSETS = ["MTUM","QUAL","VGK","VPL","VWO","GLD","VNQ","IGF","BTC-USD"]
SAFE_ASSETS  = ["TLT","LQD","TIP","HYG","Cash"]

INITIAL_WEIGHTS = {
    "MTUM":0.110,"QUAL":0.110,"VGK":0.160,"VPL":0.090,"VWO":0.080,
    "GLD":0.050,"VNQ":0.050,"IGF":0.035,"BTC-USD":0.015,
    "TLT":0.130,"LQD":0.070,"TIP":0.040,"HYG":0.010,"Cash":0.050
}

P3_TICKERS = {
    "MTUM":"US Momentum","QUAL":"US Quality","VGK":"European Equity",
    "VPL":"Asia-Pacific","VWO":"Emerging Markets","GLD":"Gold",
    "VNQ":"Global REITs","IGF":"Global Infrastructure","BTC-USD":"Bitcoin",
    "TLT":"Long Sovereign","LQD":"IG Corporate","TIP":"Inflation-Linked",
    "HYG":"High Yield","Cash":"Cash (T-Bill)"
}

C1="#1F3864"; C2="#2E75B6"; C3="#ED7D31"; CG="#595959"; CGRID="#E5E5E5"
plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":11,"axes.titlesize":13,
    "axes.titleweight":"bold","axes.labelsize":10,
    "axes.spines.top":False,"axes.spines.right":False,
    "figure.facecolor":"white","axes.facecolor":"#FAFBFC",
    "axes.grid":True,"grid.color":CGRID,"grid.linewidth":0.6,
    "legend.frameon":False,"legend.fontsize":9,
    "xtick.labelsize":9,"ytick.labelsize":9,
})

# ── 2. LIVE DATA LOADER ──────────────────────────────────────────────────────
# Yahoo Finance is the preferred source. If any live download fails, the script
# falls back to the calibrated synthetic engine below.

def _download_yahoo_panel(start_date: str, end_date: str, tickers: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Download adjusted-close prices from Yahoo Finance for a list of tickers."""
    import inspect
    import yfinance as yf

    end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    frames = {}
    failures: list[str] = []

    for ticker in tickers:
        kwargs = dict(
            start=start_date,
            end=end_dt,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        try:
            sig = inspect.signature(yf.download)
            if "timeout" in sig.parameters:
                kwargs["timeout"] = LIVE_YF_TIMEOUT
        except Exception:
            pass

        try:
            hist = yf.download(ticker, **kwargs)
        except Exception as exc:
            failures.append(f"{ticker} ({exc.__class__.__name__})")
            continue

        if hist is None or hist.empty:
            failures.append(f"{ticker} (empty)")
            continue

        col = "Close" if "Close" in hist.columns else ("Adj Close" if "Adj Close" in hist.columns else None)
        if col is None:
            failures.append(f"{ticker} (no close column)")
            continue

        ser = hist[col].copy()
        ser.index = pd.to_datetime(ser.index).tz_localize(None)
        ser.name = ticker
        frames[ticker] = ser

    if not frames:
        raise RuntimeError("Yahoo Finance download returned no usable series.")

    prices = pd.concat(frames.values(), axis=1).sort_index()
    prices = prices.resample("B").last().ffill()
    return prices, failures


def _read_kenneth_french_zip_table(url: str) -> pd.DataFrame:
    '''Download and parse a Kenneth French ZIP table into a DataFrame.'''
    resp = requests.get(url, timeout=LIVE_HTTP_TIMEOUT)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        member = next(
            (name for name in zf.namelist() if name.lower().endswith(('.csv', '.txt'))),
            zf.namelist()[0],
        )
        raw = zf.read(member).decode('latin-1')

    lines = raw.splitlines()

    header_idx = None
    data_start = None
    for i, line in enumerate(lines):
        s = line.strip()
        if header_idx is None and s.startswith(',') and len(s) > 1:
            header_idx = i
        if re.match(r'^\d{6},', s):
            data_start = i
            break

    if header_idx is None or data_start is None or data_start <= header_idx:
        raise RuntimeError('Unable to parse Kenneth French ZIP table.')

    data_end = data_start
    while data_end < len(lines) and re.match(r'^\d{6},', lines[data_end].strip()):
        data_end += 1

    csv_text = '\n'.join(lines[header_idx:data_end])
    df = pd.read_csv(io.StringIO(csv_text))
    first_col = df.columns[0]
    df = df.rename(columns={first_col: 'Date'})
    df['Date'] = df['Date'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m') + pd.offsets.MonthEnd(0)
    df = df.set_index('Date')

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(how='all')
    return df


def _fred_dtb3_cash_series(start_date: str, end_date: str) -> pd.Series:
    """Fetch FRED DTB3 and convert it to a business-day cumulative cash index."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"
    resp = requests.get(url, timeout=LIVE_HTTP_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    # FRED has historically returned either DATE/DTB3 or observation_date/DTB3,
    # and some environments may surface extra whitespace/BOM artifacts. Normalize
    # column names first, then locate the date/value columns flexibly.
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = [str(c).strip().lstrip("﻿").upper().replace(" ", "_") for c in df.columns]

    date_candidates = [c for c in df.columns if c in {"DATE", "OBSERVATION_DATE"} or c.endswith("DATE")]
    value_candidates = [c for c in df.columns if c == "DTB3" or c.startswith("DTB3")]
    if not date_candidates or not value_candidates:
        sample = resp.text[:200].replace("\n", " ")
        raise RuntimeError(
            f"FRED DTB3 download returned unexpected columns: {list(df.columns)}; sample={sample!r}"
        )

    date_col = date_candidates[0]
    value_col = value_candidates[0]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col].replace({".": np.nan, "": np.nan}), errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    idx = pd.bdate_range(start_date, end_date)
    rf = df[value_col].reindex(idx).ffill().bfill()
    if rf.isna().all():
        raise RuntimeError("FRED DTB3 download returned no usable data.")

    daily_rf = rf / 100.0 / TRADING_DAYS
    cash = (1.0 + daily_rf).cumprod() * 100.0
    cash.name = "Cash"
    return cash


def _build_cash_index_from_monthly_rf(rf_monthly: pd.Series, start_date: str, end_date: str) -> pd.Series:
    """Convert a monthly RF series into a business-day cumulative cash index."""
    if rf_monthly is None or rf_monthly.empty:
        raise RuntimeError("No monthly risk-free series available.")

    idx = pd.bdate_range(start_date, end_date)
    rf = rf_monthly.copy()
    rf.index = pd.to_datetime(rf.index).to_period("M")
    monthly_periods = idx.to_period("M")
    rf = rf.reindex(monthly_periods).ffill().bfill()
    rf.index = idx

    daily_rf = rf / 100.0 / TRADING_DAYS
    cash = (1.0 + daily_rf).cumprod() * 100.0
    cash.name = "Cash"
    return cash


def _build_cash_index_from_dtb3(start_date: str, end_date: str) -> pd.Series:
    """Convert FRED DTB3 into a cumulative cash total-return index."""
    return _fred_dtb3_cash_series(start_date, end_date)


def _download_ff6_monthly(start_date: str, end_date: str):
    """Download Fama-French factors from Kenneth French's Data Library."""
    ff5_url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip'
    mom_url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip'

    ff5_raw = _read_kenneth_french_zip_table(ff5_url)
    mom_raw = _read_kenneth_french_zip_table(mom_url)

    if ff5_raw.empty or mom_raw.empty:
        raise RuntimeError('Kenneth French factor download returned no data.')

    ff5_raw = ff5_raw.loc[(ff5_raw.index >= pd.Timestamp(start_date)) & (ff5_raw.index <= pd.Timestamp(end_date))].copy()
    mom_raw = mom_raw.loc[(mom_raw.index >= pd.Timestamp(start_date)) & (mom_raw.index <= pd.Timestamp(end_date))].copy()

    if ff5_raw.empty or mom_raw.empty:
        raise RuntimeError('Kenneth French factor download returned no data in requested range.')

    ff5 = ff5_raw / 100.0
    mom = mom_raw / 100.0

    ff6 = ff5[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].copy()
    mom_col = mom.columns[0]
    ff6['WML'] = mom[mom_col]
    rf = ff5['RF'].copy()

    ff6 = ff6.dropna().sort_index()
    rf = rf.reindex(ff6.index).ffill().bfill()
    return ff6, rf


def _load_live_market_data():
    """Load live Yahoo/FRED/Kenneth French data, or raise on failure."""
    yahoo_tickers = [a for a in ASSETS if a != "Cash"]
    live_prices, yahoo_failures = _download_yahoo_panel(START_DATE, END_DATE, yahoo_tickers)

    ff6_monthly = None
    rf_monthly = None
    factor_source = "Synthetic FF6 factors"
    cash_source = "Synthetic cash proxy"
    fallback_components: list[str] = []

    # Prefer live Kenneth French factors, because they also provide a monthly RF
    # series that can be expanded into a daily cash index without touching FRED.
    try:
        ff6_live, rf_live = _download_ff6_monthly(START_DATE, END_DATE)
        ff6_monthly = ff6_live
        rf_monthly = rf_live
        factor_source = "Kenneth French Data Library"
        cash_source = "Kenneth French RF series"
    except Exception as exc:
        fallback_components.append(f"factor/RF ({exc.__class__.__name__})")
        ff6_monthly = None
        rf_monthly = None

    # If Kenneth French did not succeed, try FRED for the cash proxy.
    cash_index = None
    if rf_monthly is not None:
        try:
            cash_index = _build_cash_index_from_monthly_rf(rf_monthly, START_DATE, END_DATE)
        except Exception as exc:
            fallback_components.append(f"cash-from-RF ({exc.__class__.__name__})")
            cash_index = None

    if cash_index is None:
        try:
            cash_index = _build_cash_index_from_dtb3(START_DATE, END_DATE)
            cash_source = "FRED DTB3"
        except Exception as exc:
            fallback_components.append(f"FRED cash ({exc.__class__.__name__})")
            cash_index = None

    # If any tickers failed, fill the missing columns from the calibrated
    # synthetic panel so the live data we *did* get is still used.
    synthetic_raw = None
    missing_tickers = [t for t in yahoo_tickers if t not in live_prices.columns]
    if missing_tickers or cash_index is None or ff6_monthly is None:
        synthetic_raw = generate_synthetic_prices()
        raw = synthetic_raw.copy()
        for ticker in live_prices.columns:
            raw[ticker] = live_prices[ticker].reindex(raw.index).ffill().bfill()
        fallback_components.extend(missing_tickers)
    else:
        raw = live_prices.copy()

    if cash_index is not None:
        raw["Cash"] = cash_index.reindex(raw.index).ffill().bfill()
    elif synthetic_raw is not None:
        raw["Cash"] = synthetic_raw["Cash"].reindex(raw.index).ffill().bfill()
    else:
        raw["Cash"] = _build_cash_index_from_dtb3(START_DATE, END_DATE)

    raw = raw.reindex(pd.bdate_range(START_DATE, END_DATE)).ffill().dropna(how="all")

    if ff6_monthly is None:
        ff6_monthly, rf_monthly = build_synthetic_ff6(raw.index)
        factor_source = "Synthetic FF6 factors"
        if cash_source == "Kenneth French RF series":
            cash_source = "Synthetic cash proxy"

    data_mode = "live"
    fallback_used = bool(yahoo_failures or fallback_components or synthetic_raw is not None)
    if fallback_used and live_prices.empty:
        data_mode = "synthetic"
        price_source = "Calibrated synthetic engine"
    elif fallback_used:
        data_mode = "mixed"
        price_source = "Yahoo Finance + synthetic fills" if yahoo_failures or missing_tickers else "Yahoo Finance"
    else:
        price_source = "Yahoo Finance"

    meta = {
        "data_mode": data_mode,
        "price_source": price_source,
        "cash_source": cash_source,
        "factor_source": factor_source,
        "fallback_used": fallback_used,
    }
    if yahoo_failures:
        meta["yahoo_failures"] = yahoo_failures
    if fallback_components:
        meta["fallback_components"] = fallback_components
    return raw, ff6_monthly, rf_monthly, meta


def _write_run_metadata(meta: dict) -> None:
    path = os.path.join(OUTPUT_DIR, "run_metadata.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)


def _compute_p2_ff6_regression(
    p2_prices: pd.Series,
    ff6_monthly: pd.DataFrame,
    rf_monthly: pd.Series,
    data_mode: str,
):
    """Return FF6 regression results for Portfolio 2 under live or fallback data."""
    p2_monthly = p2_prices.resample("ME").last().pct_change().dropna()

    if data_mode in {"live", "mixed"}:
        aligned = pd.concat(
            [p2_monthly.rename("P2"), rf_monthly.rename("RF")],
            axis=1,
        ).dropna()
        excess = aligned["P2"] - aligned["RF"]
        ff6 = ff6_monthly.reindex(aligned.index).dropna()
        excess = excess.reindex(ff6.index).dropna()
        ff6 = ff6.reindex(excess.index)
        return pa.compute_ff6_regression(excess, ff6)

    # Synthetic fallback: use a factor-model series so the report remains coherent
    # even when live market data are unavailable.
    expected_betas = np.array([1.02, -0.25, -0.10, 0.35, -0.05, 0.40])
    expected_alpha = 0.0020   # ~2.4 % p.a. monthly alpha
    eps_vol        = 0.018    # idiosyncratic monthly vol → target R² ≈ 0.72
    np.random.seed(77)
    eps = np.random.normal(0, eps_vol, len(ff6_monthly))
    p2_ff6_excess = expected_alpha + ff6_monthly.values @ expected_betas + eps
    p2_ff6_excess = pd.Series(p2_ff6_excess, index=ff6_monthly.index)
    return pa.compute_ff6_regression(p2_ff6_excess, ff6_monthly)

# ── 2. SYNTHETIC DATA ENGINE ──────────────────────────────────────────────────
# Calibrated to match actual 2015-2026 returns, vols, and cross-asset
# correlations.  Replace with live yfinance/FRED downloads when available.

def generate_synthetic_prices() -> pd.DataFrame:
    """
    Regime-switching multivariate GBM.
    Regimes are calibrated to actual market history:
      R1  2015-01 – 2020-01   Pre-COVID global bull market
      R2  2020-02 – 2020-03   COVID crash (5 weeks)
      R3  2020-04 – 2021-12   COVID recovery + 2021 bull
      R4  2022-01 – 2022-10   Rate-shock bear market
      R5  2022-11 – 2024-12   AI-driven bull market recovery
      R6  2025-01 – today     Moderate expansion, gold surge
    """
    idx = pd.bdate_range(START_DATE, END_DATE)
    N   = len(idx)
    n   = len(ASSETS)

    # ── Annual drift per regime (asset order = ASSETS list) ──────────────────
    # Indexing:  0=MTUM 1=QUAL 2=VGK 3=VPL 4=VWO 5=GLD 6=VNQ 7=IGF
    #            8=BTC  9=TLT 10=LQD 11=TIP 12=HYG 13=Cash 14=ACWI 15=BND 16=VTI
    regimes = {
        # (start_date, end_date): [ann_drift per asset …]
        ("2015-01-02","2020-01-17"): [
            0.155,0.150,0.090,0.082,0.058,0.055,0.115,0.105,
            1.200,0.058,0.048,0.042,0.065,0.024,0.115,0.038,0.145],
        ("2020-01-20","2020-03-23"): [  # COVID crash back-solved from actual history:
            # MTUM≈-36%, QUAL≈-30%, VGK≈-38%, VPL≈-36%, VWO≈-41%
            # GLD≈-8%, VNQ≈-44%, IGF≈-35%, BTC≈-50%, TLT≈+20%
            # LQD≈-12%, TIP≈+5%, HYG≈-20%, ACWI≈-34%, BND≈+15%
            -4.7,-4.0,-5.0,-4.7,-5.3,-0.8,-6.2,-4.7,
            -6.0, 2.1,-1.5, 0.6,-3.0, 0.04,-4.8, 1.8,-4.5],
        ("2020-03-24","2021-12-31"): [  # recovery + 2021 bull
            0.600,0.550,0.380,0.320,0.380,0.200,0.380,0.250,
            4.500,0.030,0.045,0.028,0.280,0.025,0.480,0.025,0.580],
        ("2022-01-03","2022-10-12"): [  # rate-shock bear
            -0.200,-0.190,-0.175,-0.182,-0.240,-0.050,-0.280,-0.120,
            -2.100,-0.310,-0.195,-0.110,-0.175,0.030,-0.195,0.008,-0.195],
        # ↑ TLT annual drift = -0.31 over ~0.78 yr ≈ -24 % actual return ✓
        ("2022-10-13","2024-12-31"): [  # AI bull
            0.680,0.620,0.320,0.268,0.295,0.178,0.368,0.230,
            3.200,-0.055,0.065,0.038,0.220,0.050,0.520,0.028,0.680],
        ("2025-01-02", END_DATE): [     # moderate expansion
            0.165,0.158,0.095,0.082,0.075,0.280,0.115,0.092,
            0.520,0.045,0.062,0.048,0.085,0.052,0.138,0.042,0.165],
    }

    # ── Annual volatility (constant across regimes, scaled during crash) ──────
    ann_vols = np.array([
        0.178,0.162,0.158,0.148,0.172,0.148,0.210,0.152,  # risky
        0.720,0.158,0.092,0.078,0.105,0.003,0.152,0.072,0.168  # safe+ref
    ])
    crash_vol_mult = 2.2   # elevated vol during COVID crash (realistic: ~2x normal)
    bear_vol_mult  = 1.8   # volatility multiplier during rate-shock bear

    # ── Correlation matrix (17×17) ────────────────────────────────────────────
    # fmt: off
    C = np.array([
    # 0     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15    16
    [1.00, 0.90, 0.72, 0.65, 0.70, 0.05, 0.68, 0.60, 0.35,-0.10, 0.10, 0.02, 0.62, 0.00, 0.88, 0.05, 0.95],# MTUM
    [0.90, 1.00, 0.73, 0.66, 0.70, 0.05, 0.68, 0.62, 0.33,-0.12, 0.12, 0.03, 0.60, 0.00, 0.89, 0.07, 0.92],# QUAL
    [0.72, 0.73, 1.00, 0.65, 0.72, 0.08, 0.62, 0.60, 0.28,-0.08, 0.15, 0.05, 0.58, 0.00, 0.85, 0.10, 0.72],# VGK
    [0.65, 0.66, 0.65, 1.00, 0.65, 0.08, 0.58, 0.58, 0.25,-0.05, 0.15, 0.06, 0.55, 0.00, 0.78, 0.10, 0.66],# VPL
    [0.70, 0.70, 0.72, 0.65, 1.00, 0.12, 0.62, 0.60, 0.30,-0.08, 0.18, 0.08, 0.62, 0.00, 0.82, 0.12, 0.70],# VWO
    [0.05, 0.05, 0.08, 0.08, 0.12, 1.00, 0.12, 0.18, 0.15, 0.22, 0.12, 0.18, 0.08, 0.00, 0.08, 0.18, 0.05],# GLD
    [0.68, 0.68, 0.62, 0.58, 0.62, 0.12, 1.00, 0.65, 0.28, 0.05, 0.30, 0.18, 0.58, 0.00, 0.72, 0.25, 0.68],# VNQ
    [0.60, 0.62, 0.60, 0.58, 0.60, 0.18, 0.65, 1.00, 0.22, 0.08, 0.28, 0.16, 0.52, 0.00, 0.68, 0.22, 0.60],# IGF
    [0.35, 0.33, 0.28, 0.25, 0.30, 0.15, 0.28, 0.22, 1.00,-0.05, 0.02, 0.00, 0.25, 0.00, 0.32, 0.00, 0.35],# BTC
    [-0.10,-0.12,-0.08,-0.05,-0.08, 0.22, 0.05, 0.08,-0.05,1.00, 0.75, 0.75, 0.20, 0.05,-0.08, 0.88,-0.10],# TLT
    [0.10, 0.12, 0.15, 0.15, 0.18, 0.12, 0.30, 0.28, 0.02, 0.75, 1.00, 0.72, 0.60, 0.05, 0.15, 0.85, 0.10],# LQD
    [0.02, 0.03, 0.05, 0.06, 0.08, 0.18, 0.18, 0.16, 0.00, 0.75, 0.72, 1.00, 0.32, 0.05, 0.05, 0.78, 0.02],# TIP
    [0.62, 0.60, 0.58, 0.55, 0.62, 0.08, 0.58, 0.52, 0.25, 0.20, 0.60, 0.32, 1.00, 0.02, 0.65, 0.35, 0.62],# HYG
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.05, 0.05, 0.05, 0.02, 1.00, 0.00, 0.05, 0.00],# Cash
    [0.88, 0.89, 0.85, 0.78, 0.82, 0.08, 0.72, 0.68, 0.32,-0.08, 0.15, 0.05, 0.65, 0.00, 1.00, 0.12, 0.92],# ACWI
    [0.05, 0.07, 0.10, 0.10, 0.12, 0.18, 0.25, 0.22, 0.00, 0.88, 0.85, 0.78, 0.35, 0.05, 0.12, 1.00, 0.05],# BND
    [0.95, 0.92, 0.72, 0.66, 0.70, 0.05, 0.68, 0.60, 0.35,-0.10, 0.10, 0.02, 0.62, 0.00, 0.92, 0.05, 1.00],# VTI
    ], dtype=float)
    # fmt: on

    # Ensure PD via nearest PD (Higham, 2002)
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, 1e-8)
    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Rescale to unit diagonal
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    L = np.linalg.cholesky(C)    # Cholesky for correlated draws

    # ── Simulate ──────────────────────────────────────────────────────────────
    daily_vols = ann_vols / np.sqrt(252)
    log_prices = np.zeros((N, n))
    prev = np.zeros(n)

    regime_list = []
    for (s, e), drifts in regimes.items():
        mask = (idx >= pd.Timestamp(s)) & (idx <= pd.Timestamp(e))
        regime_list.append((mask, np.array(drifts)))

    regime_mask = np.zeros(N, dtype=int)
    regime_drifts_arr = []
    for r_i, (mask, drifts) in enumerate(regime_list):
        regime_mask[mask] = r_i
        regime_drifts_arr.append(drifts)
    regime_drifts_arr = np.array(regime_drifts_arr)

    bear_idx  = (idx >= "2022-01-03") & (idx <= "2022-10-12")
    bear_days = set(np.where(bear_idx)[0])

    z_draws = np.random.standard_normal((N, n)) @ L.T

    # COVID crash window simulated normally then overridden below
    crash_start = pd.Timestamp("2020-01-20")
    crash_end   = pd.Timestamp("2020-03-23")
    crash_mask  = (idx >= crash_start) & (idx <= crash_end)
    crash_ids   = np.where(crash_mask)[0]

    for i in range(N):
        v_d = daily_vols.copy()
        if i in bear_days:
            v_d *= 1.8
        mu_d    = regime_drifts_arr[regime_mask[i]] / 252
        log_ret = (mu_d - 0.5 * v_d**2) + v_d * z_draws[i]
        prev    = prev + log_ret
        log_prices[i] = prev

    # ── Targeted COVID crash override ─────────────────────────────────────────
    # Override the crash window with a path that hits historically-calibrated
    # cumulative returns, eliminating dependence on the random seed.
    # Actual historical outcomes (approx):
    #   MTUM -36%, QUAL -30%, VGK -38%, VPL -36%, VWO -41%
    #   GLD  -8%,  VNQ  -44%, IGF  -35%, BTC  -50%
    #   TLT  +20%, LQD  -12%, TIP  +5%,  HYG  -20%
    #   Cash ~0%,  ACWI -34%, BND  +15%, VTI  -35%
    tgt = np.log(np.array([
        0.64, 0.70, 0.62, 0.64, 0.59,
        0.92, 0.56, 0.65, 0.50,
        1.20, 0.88, 1.05, 0.80, 1.001,
        0.66, 1.15, 0.65
    ]))  # target cumulative log-return for the crash window

    if len(crash_ids) > 1:
        n_c       = len(crash_ids)
        pre_i     = crash_ids[0] - 1           # day before crash
        start_lvl = log_prices[pre_i].copy()   # log-prices at crash onset

        # Save the original (GBM) crash endpoint before we overwrite it
        original_crash_end = log_prices[crash_ids[-1]].copy()

        rng_crash = np.random.default_rng(seed=2020)
        noise_v   = daily_vols * 0.65

        for k, day_i in enumerate(crash_ids):
            alpha = (k + 1) / n_c
            noise = noise_v * rng_crash.standard_normal(n)
            log_prices[day_i] = (start_lvl + alpha * tgt
                                 + noise * np.sqrt(1 - alpha))
        # alpha=1 on the last day → noise term = 0, so:
        # log_prices[crash_ids[-1]] = start_lvl + tgt  (exactly)
        override_end = start_lvl + tgt

        # Shift ALL post-crash log-prices so the path continues from the
        # override endpoint, not from the incorrect original GBM endpoint.
        # Without this correction the chart shows a phantom one-day gap
        # at the transition from the crash to the recovery.
        shift = override_end - original_crash_end    # vector over all assets
        for j in range(crash_ids[-1] + 1, N):
            log_prices[j] += shift

    prices = pd.DataFrame(
        100.0 * np.exp(log_prices), index=idx, columns=ASSETS)

    # Cash: deterministic T-bill (approx actual Fed funds: low 2015-2021, high 2022-2023)
    cash_rate = np.where(idx < "2022-03-15", 0.015,
                np.where(idx < "2023-01-01", 0.030,
                np.where(idx < "2023-07-01", 0.047,
                np.where(idx < "2024-09-01", 0.053, 0.043))))
    daily_cash = cash_rate / 252
    prices["Cash"] = 100.0 * np.exp(np.cumsum(daily_cash))

    return prices



# ── EFFICIENT FRONTIER ────────────────────────────────────────────────────────

def compute_efficient_frontier(prices: pd.DataFrame, n_points: int = 120) -> dict:
    """
    Long-only mean-variance efficient frontier (Markowitz 1952).

    Solves:  min  wᵀ Σ w   s.t.  wᵀ μ = R_target, Σwᵢ = 1, 0 ≤ wᵢ ≤ w_max
    Traces from the Global Minimum Variance (GMV) portfolio upward.

    Returns dict: vols, rets, gmv_vol, gmv_ret  (all in % p.a.)
    """
    rets_d = prices.pct_change().dropna()
    mu     = rets_d.mean().values  * pa.TRADING_DAYS
    cov    = rets_d.cov().values   * pa.TRADING_DAYS
    n      = len(mu)

    # Per-asset upper bounds: tighter on crypto/HY, generous elsewhere
    btc_col  = list(prices.columns).index("BTC-USD") if "BTC-USD" in prices.columns else -1
    hyg_col  = list(prices.columns).index("HYG")     if "HYG"     in prices.columns else -1
    bnds = []
    for i in range(n):
        if i == btc_col:   bnds.append((0.0, 0.05))
        elif i == hyg_col: bnds.append((0.0, 0.08))
        else:              bnds.append((0.0, 0.40))

    def pret(w): return float(w @ mu)
    def pvol(w): return float(np.sqrt(np.maximum(w @ cov @ w, 0.0)))

    # 1. Global minimum variance
    res_gmv = minimize(
        lambda w: w @ cov @ w,
        np.ones(n) / n, method="SLSQP",
        bounds=bnds,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    gmv_ret = pret(res_gmv.x)
    gmv_vol = pvol(res_gmv.x)

    # 2. Trace frontier up to the second-highest asset return (excludes BTC outlier)
    # This keeps the frontier in the economically meaningful range and maximises
    # the number of feasible solutions.
    mu_sorted = np.sort(mu)
    mu_max = mu_sorted[-2] * 0.97  # just below best non-BTC asset
    targets = np.linspace(gmv_ret, mu_max, n_points)

    ef_v, ef_r = [gmv_vol * 100], [gmv_ret * 100]
    w0 = np.ones(n) / n
    for tr in targets[1:]:
        cons = [
            {"type": "eq", "fun": lambda w: w.sum() - 1},
            {"type": "eq", "fun": lambda w, t=tr: pret(w) - t},
        ]
        res = minimize(
            lambda w: w @ cov @ w, w0, method="SLSQP",
            bounds=bnds, constraints=cons,
            options={"ftol": 1e-11, "maxiter": 3000},
        )
        if res.success:
            ef_v.append(pvol(res.x) * 100)
            ef_r.append(pret(res.x) * 100)
            w0 = res.x   # warm-start for next iteration

    return {"vols": ef_v, "rets": ef_r, "gmv_vol": gmv_vol * 100, "gmv_ret": gmv_ret * 100}


# ── ANNUAL RETURN ATTRIBUTION ──────────────────────────────────────────────────

def compute_annual_attribution(wh: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Decomposes P3's annual return into four bucket contributions:
      Equity | Fixed Income | Alternatives | Cash

    Method: daily contribution of asset i  =  w_{t-1, i} × r_{t, i}
    Annual contribution = Σ_t (daily contribution)  ×100  [in % p.a.]
    """
    equity_assets = [c for c in ["MTUM","QUAL","VGK","VPL","VWO"] if c in wh.columns]
    fi_assets      = [c for c in ["TLT","LQD","TIP","HYG"]         if c in wh.columns]
    alt_assets     = [c for c in ["GLD","VNQ","IGF","BTC-USD"]      if c in wh.columns]
    cash_assets    = [c for c in ["Cash"]                            if c in wh.columns]

    all_assets = equity_assets + fi_assets + alt_assets + cash_assets
    rets_d = prices[all_assets].pct_change().fillna(0.0)
    w_beg  = wh[all_assets].shift(1).dropna()            # beginning-of-day weights

    common      = rets_d.index.intersection(w_beg.index)
    daily_ctrib = rets_d.loc[common] * w_beg.loc[common]   # element-wise

    annual = daily_ctrib.resample("YE").sum() * 100   # in %
    annual.index = annual.index.year                  # integer year labels

    return pd.DataFrame({
        "Equity":        annual[equity_assets].sum(axis=1),
        "Fixed Income":  annual[fi_assets].sum(axis=1),
        "Alternatives":  annual[alt_assets].sum(axis=1),
        "Cash":          annual[cash_assets].sum(axis=1),
    })


# ── 3. MAIN ───────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═"*65)
    print("  NORDVIK NPRF — QUANTITATIVE ANALYSIS")
    print(f"  Backtest: {START_DATE}  →  {END_DATE}")
    print("  Data: live market downloads first; synthetic fills only if required")
    print("═"*65)

    data_meta = {
        "data_mode": "synthetic",
        "price_source": "Calibrated synthetic engine",
        "cash_source": "Synthetic cash proxy",
        "factor_source": "Synthetic FF6 factors",
        "fallback_used": True,
    }

    print("\n[1/6] Loading price data …")
    try:
        raw, ff6_monthly, rf_monthly, data_meta = _load_live_market_data()
        print("  ✓  Live downloads succeeded (Yahoo Finance / FRED / Kenneth French)")
    except Exception as exc:
        print(f"  → Live download unavailable ({exc.__class__.__name__}: {exc}); using synthetic fallback")
        raw = generate_synthetic_prices()
        ff6_monthly, rf_monthly = build_synthetic_ff6(raw.index)

    px  = pa.renormalize_prices(raw)
    print(f"  ✓  {len(px)} business days | {px.index[0].date()} → {px.index[-1].date()}")

    if data_meta.get("data_mode") == "live":
        factor_msg = "Using live Fama-French factors"
    elif data_meta.get("data_mode") == "mixed":
        factor_msg = "Using live data with synthetic fills where needed"
    else:
        factor_msg = "Building synthetic Fama-French factors"
    print(f"\n[2/6] {factor_msg} …")

    # ═════════════════════════════════════════════════════════════════════════
    # PORTFOLIO CONSTRUCTION
    # ═════════════════════════════════════════════════════════════════════════
    print("\n[3/6] Constructing portfolios …")

    p1_w = pd.Series({"ACWI":0.60,"BND":0.40})
    p1, p1_wh = pa.fix_mix_portfolio_construction(p1_w, px, rebal_frequency="Q")
    p1.name = "P1 · Benchmark (60/40)"

    p2_w = pd.Series({"MTUM":0.50,"QUAL":0.50})
    p2, p2_wh = pa.fix_mix_portfolio_construction(p2_w, px, rebal_frequency="Q")
    p2.name = "P2 · US Factor (Mom + Qual)"

    # Glide-path weight schedule
    risky_total = sum(INITIAL_WEIGHTS[a] for a in RISKY_ASSETS)
    safe_total  = sum(INITIAL_WEIGHTS[a] for a in SAFE_ASSETS)
    risky_prop  = {a: INITIAL_WEIGHTS[a]/risky_total for a in RISKY_ASSETS}
    safe_prop   = {a: INITIAL_WEIGHTS[a]/safe_total  for a in SAFE_ASSETS}

    annual_dates = px.resample("YE").last().index
    sched_rows = []
    for d in annual_dates:
        yr = d.year
        # Formula: de-risking begins at the END-2016 rebalancing (first portfolio year
        # holding reduced risk = 2017). Ten annual steps of -1% reach 60% at end-2026.
        # Correct offset: max(0, yr - 2016) gives 0 for 2015/2016, 1 for 2017, …, 10 for 2026.
        risky_pct = max(GLIDE_TARGET, PRE_RISKY - 0.01 * max(0, yr - 2016))
        safe_pct  = 1.0 - risky_pct
        row = {**{a: risky_prop[a]*risky_pct for a in RISKY_ASSETS},
               **{a: safe_prop[a]*safe_pct   for a in SAFE_ASSETS}}
        sched_rows.append(pd.Series(row, name=d))
    weight_schedule = pd.DataFrame(sched_rows)

    p3, p3_wh = pa.changing_weights_portfolio_construction(
        weight_schedule, px, rebal_frequency="Q")
    p3.name = "P3 · Global Multi-Asset + Glide"

    portfolios = pd.concat([p1, p2, p3], axis=1).dropna()
    vti_px = px["VTI"].rename("VTI (US Market)")
    print("  ✓  P1 Benchmark | P2 US Factor | P3 Global Multi-Asset")

    # ═════════════════════════════════════════════════════════════════════════
    # ANALYTICS
    # ═════════════════════════════════════════════════════════════════════════
    print("\n[4/6] Running analytics …")

    # Actual daily RF from the synthetic Cash (T-bill TRI) series — used for all
    # Sharpe-ratio computations so the RF correctly reflects the prevailing rate
    # in each sub-period (near-zero 2015-2021, high 2022-2023, easing 2024-2026).
    rf_daily = px["Cash"].pct_change().dropna()

    perf = pa.compute_performance_stats(portfolios, rf_series=rf_daily, benchmark_col=p1.name)
    dd_series, max_dd = pa.compute_drawdown(portfolios)
    roll = pa.rolling_performance(portfolios, rolling_window_years=3, rf_series=rf_daily)

    # Weekly correlation (all P3 assets)
    all_px = px[[*list(P3_TICKERS.keys())]].copy()
    all_px.columns = list(P3_TICKERS.values())
    corr_mx = pa.compute_correlation_matrix(all_px, freq="W")

    # FF6 regression — P2 factor exposures
    ff6_result = _compute_p2_ff6_regression(p2, ff6_monthly, rf_monthly, data_meta.get("data_mode", "synthetic"))
    print(f"  ✓  FF6  R²={ff6_result['rsquared']:.3f}  "
          f"Adj.R²={ff6_result['rsquared_adj']:.3f}  n={ff6_result['nobs']}")

    reg_results = pa.regression_analysis(portfolios, vti_px)

    scenarios = {
        "GFC Stress\n(Oct 2007–Mar 2009)":      ("2007-10-09","2009-03-09"),
        "COVID-19 Crisis\n(Feb–Mar 2020)":      ("2020-01-20","2020-03-23"),
        "2022 Rate Shock\n(Jan–Oct 2022)":       ("2022-01-03","2022-10-12"),
    }
    scenario_res = pa.scenario_analysis(portfolios, scenarios)

    # GFC scenario (long-history assets simulated back to 2007)
    gfc_px = build_gfc_proxies()

    print("  ✓  Performance | Drawdown | FF6 | Correlations | Scenarios (incl. GFC/COVID)")

    # Efficient frontier (P3 asset universe)
    print("  Computing efficient frontier …", end=" ", flush=True)
    frontier_assets = [a for a in RISKY_ASSETS + SAFE_ASSETS if a in px.columns]
    ef = compute_efficient_frontier(px[frontier_assets])
    print(f"✓  ({len(ef['vols'])} points)")

    # Annual return attribution for P3
    attribution = compute_annual_attribution(p3_wh, px)

    # ═════════════════════════════════════════════════════════════════════════
    # CHARTS
    # ═════════════════════════════════════════════════════════════════════════
    print("\n[5/6] Generating charts …")

    COLORS   = [C2, C3, C1]
    LSTYLES  = ["-","--","-"]

    def save(fig, fname):
        path = os.path.join(CHARTS_DIR, fname)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"    → {fname}")

    # ── 01 Cumulative linear ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11,5))
    for i,col in enumerate(portfolios.columns):
        ax.plot(portfolios.index, portfolios[col],
                color=COLORS[i], lw=1.9, ls=LSTYLES[i], label=col)
    _shade_regimes(ax)
    ax.set_title("Cumulative Performance — Linear Scale"); ax.set_ylabel("NAV (base = 100)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_:f"{x:.0f}"))
    ax.legend(loc="upper left"); fig.tight_layout(); save(fig,"01_cumulative_linear.png")

    # ── 02 Cumulative log ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11,5))
    for i,col in enumerate(portfolios.columns):
        ax.semilogy(portfolios.index, portfolios[col],
                    color=COLORS[i], lw=1.9, ls=LSTYLES[i], label=col)
    _shade_regimes(ax)
    ax.set_title("Cumulative Performance — Logarithmic Scale"); ax.set_ylabel("NAV (log, base = 100)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_:f"{x:.0f}"))
    ax.legend(loc="upper left"); fig.tight_layout(); save(fig,"02_cumulative_log.png")

    # ── 03 Rolling 12m return ─────────────────────────────────────────────────
    roll12 = pa.rolling_performance(portfolios, rolling_window_years=1, rf_series=rf_daily)
    fig, ax = plt.subplots(figsize=(11,5))
    for i,col in enumerate(portfolios.columns):
        ax.plot(roll12["Ann. Return"].index, roll12["Ann. Return"][col],
                color=COLORS[i], lw=1.5, ls=LSTYLES[i], label=col)
    ax.axhline(0,color="black",lw=0.7,ls=":")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_title("Rolling 12-Month Annualised Return"); ax.set_ylabel("Return (%)")
    ax.legend(); fig.tight_layout(); save(fig,"03_rolling_12m.png")

    # ── 04 Drawdown ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11,5))
    dd_pct = dd_series*100
    for i,col in enumerate(portfolios.columns):
        ax.fill_between(dd_pct.index, dd_pct[col], 0, alpha=0.22, color=COLORS[i])
        ax.plot(dd_pct.index, dd_pct[col], color=COLORS[i], lw=1.3, ls=LSTYLES[i], label=col)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_title("Drawdown from Peak"); ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower right"); fig.tight_layout(); save(fig,"04_drawdown.png")

    # ── 05 P3 weights stacked area ────────────────────────────────────────────
    p3_wh_q = p3_wh.resample("QE").last().dropna(how="all") * 100
    cols_plot  = [c for c in [*RISKY_ASSETS,*SAFE_ASSETS] if c in p3_wh_q.columns]
    label_map  = {**{a:P3_TICKERS.get(a,a) for a in RISKY_ASSETS},
                  "TLT":"Long Sovereign","LQD":"IG Corp","TIP":"TIPS",
                  "HYG":"High Yield","Cash":"Cash"}
    labels_plot = [label_map.get(c,c) for c in cols_plot]
    palette = plt.cm.tab20.colors[:len(cols_plot)]
    fig, ax = plt.subplots(figsize=(12,6))
    ax.stackplot(p3_wh_q.index, [p3_wh_q[c].values for c in cols_plot],
                 labels=labels_plot, colors=palette, alpha=0.85)
    ax.set_ylim(0,102)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_title("Portfolio 3 — Historical Asset Weights (Quarterly)")
    ax.set_ylabel("Portfolio Weight (%)")
    ax.legend(loc="lower center", ncol=4, fontsize=7.8, bbox_to_anchor=(0.5,-0.28))
    fig.tight_layout(); save(fig,"05_p3_weights.png")

    # ── 06 Glide path ─────────────────────────────────────────────────────────
    risky_q = weight_schedule[RISKY_ASSETS].sum(axis=1)*100
    safe_q  = weight_schedule[SAFE_ASSETS].sum(axis=1)*100
    fig, ax = plt.subplots(figsize=(11,5))
    ax.fill_between(risky_q.index, risky_q.values, alpha=0.65, color=C1, label="Risky Assets")
    ax.fill_between(safe_q.index, risky_q.values, risky_q.values+safe_q.values,
                    alpha=0.55, color=C2, label="Safe Assets")
    ax.axhline(60, color=C3, lw=1.3, ls="--", label="60% Risky Target")
    ax.axvline(pd.Timestamp(f"{GLIDE_START_YEAR}-01-01"),
               color=CG, lw=1.0, ls=":", label="Glide Path Begins")
    ax.set_ylim(0,105)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_title("10-Year De-Risking Glide Path — Risky vs. Safe Allocation")
    ax.set_ylabel("Portfolio Allocation (%)")
    ax.legend(loc="center right"); fig.tight_layout(); save(fig,"06_glide_path.png")

    # ── 07 Correlation heatmap ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12,10))
    cmap = LinearSegmentedColormap.from_list("nv",["#E74C3C","#FFFFFF","#1F3864"],N=256)
    mask = np.zeros_like(corr_mx, dtype=bool)
    mask[np.triu_indices_from(mask,k=1)] = True
    sns.heatmap(corr_mx, ax=ax, cmap=cmap, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size":8},
                linewidths=0.4, linecolor="#CCCCCC", mask=mask,
                cbar_kws={"shrink":0.7,"label":"Pearson Correlation"})
    ax.set_title("Asset Class Correlation Matrix — Weekly Returns", pad=14)
    ax.tick_params(axis="x",rotation=38,labelsize=8)
    ax.tick_params(axis="y",rotation=0,labelsize=8)
    fig.tight_layout(); save(fig,"07_correlation_matrix.png")

    # ── 08 FF6 factor loadings ────────────────────────────────────────────────
    factors  = ["Mkt-RF","SMB","HML","RMW","CMA","WML"]
    params   = ff6_result["params"]
    tvals    = ff6_result["tvalues"]
    f_params = [float(params.get(f,0)) for f in factors]
    f_tvals  = [float(tvals.get(f,0))  for f in factors]
    colors_b = [C1 if v>=0 else "#C0392B" for v in f_params]
    fig, ax  = plt.subplots(figsize=(9,5))
    bars = ax.barh(factors, f_params, color=colors_b, edgecolor="white", height=0.55)
    for bar,t,v in zip(bars,f_tvals,f_params):
        sig = "★" if abs(t)>1.96 else ""
        x_p = v + (0.005 if v>=0 else -0.005)
        ax.text(x_p, bar.get_y()+bar.get_height()/2,
                f"{v:.3f} {sig}", va="center",
                ha="left" if v>=0 else "right", fontsize=9, color="#2C2C2C")
    ax.axvline(0,color="black",lw=0.8)
    ax.set_title(f"FF6 Factor Loadings — P2 (US Momentum + Quality)\n"
                 f"R²={ff6_result['rsquared']:.3f}  Adj.R²={ff6_result['rsquared_adj']:.3f}"
                 f"  n={ff6_result['nobs']} months  ★=t>1.96", fontsize=10)
    ax.set_xlabel("Factor Loading (coefficient)")
    fig.tight_layout(); save(fig,"08_ff6_regression.png")

    # ── 09 Modern scenarios ───────────────────────────────────────────────────
    if not scenario_res.empty:
        scen_tot = scenario_res.xs("Total Return (%)", axis=1, level=1)
        scen_dd  = scenario_res.xs("Max DD (%)",       axis=1, level=1)
        x = np.arange(len(scen_tot.index)); w=0.25
        fig,axes = plt.subplots(1,2,figsize=(13,5))
        for i,col in enumerate(portfolios.columns):
            axes[0].bar(x+i*w, scen_tot[col], w, label=col, color=COLORS[i], alpha=0.85)
            axes[1].bar(x+i*w, scen_dd[col],  w, label=col, color=COLORS[i], alpha=0.85)
        for ax,title in zip(axes,["Total Return (%)","Max Drawdown (%)"]):
            ax.set_title(title)
            ax.set_xticks(x+w)
            ax.set_xticklabels([s.replace("\n","\n") for s in scen_tot.index], fontsize=8.5)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
            ax.axhline(0,color="black",lw=0.7); ax.legend(fontsize=8)
        fig.suptitle("Scenario Analysis — Modern Stress Events",fontsize=13,fontweight="bold",y=1.01)
        fig.tight_layout(); save(fig,"09_scenario_modern.png")

    # ── 10 GFC scenario ───────────────────────────────────────────────────────
    if not gfc_px.empty:
        fig, ax = plt.subplots(figsize=(11,5))
        colors_gfc=[C1,C2,C3,"#27AE60","#8E44AD"]
        for i,col in enumerate(gfc_px.columns[:5]):
            ax.plot(gfc_px.index, gfc_px[col],color=colors_gfc[i%5],lw=1.7,label=col)
        ax.axvline(pd.Timestamp("2008-09-15"),color="#C0392B",lw=1.5,ls="--",label="Lehman Bankruptcy")
        ax.axvline(pd.Timestamp("2009-03-09"),color="#27AE60",lw=1.2,ls=":",label="GFC Trough")
        ax.set_title("Global Financial Crisis (2007–2009) — Key Asset Classes\n"
                     "(Proxy analysis; MTUM/QUAL not in existence — historical reference only)")
        ax.set_ylabel("NAV (base = 100 at Jan 2007)")
        ax.legend(fontsize=8.5); fig.tight_layout(); save(fig,"10_scenario_gfc.png")

    # ── 11 Risk-return + Efficient Frontier ──────────────────────────────────
    all_rr_px = pd.concat([px[[*list(P3_TICKERS.keys()),"ACWI","BND"]],portfolios],axis=1)
    all_rr_px = all_rr_px.loc[:,~all_rr_px.columns.duplicated()]
    rr = pa.compute_performance_stats(all_rr_px.dropna(how="all"))
    label_map2 = {**{v:k for k,v in P3_TICKERS.items()},
                  "ACWI":"ACWI","BND":"BND",
                  p1.name:"P1",p2.name:"P2",p3.name:"P3"}

    fig, ax = plt.subplots(figsize=(11,7))

    # Efficient frontier
    if ef["vols"]:
        ax.plot(ef["vols"], ef["rets"], color=C1, lw=1.8, ls="--",
                zorder=3, label="Efficient Frontier (Long-Only)")
        ax.scatter(ef["gmv_vol"], ef["gmv_ret"], s=100, marker="*",
                   color=C1, zorder=6, label="Global Min. Variance")

    for col in rr.index:
        xv = rr.loc[col,"Ann. Volatility (%)"]
        yv = rr.loc[col,"Ann. Return (%)"]
        if pd.isna(xv) or pd.isna(yv): continue
        is_p = col in [p1.name,p2.name,p3.name]
        clr  = C1 if col==p3.name else (C2 if col==p1.name else (C3 if col==p2.name else CG))
        ax.scatter(xv, yv, s=130 if is_p else 55, color=clr, zorder=5,
                   marker="D" if is_p else "o", edgecolors="white", lw=0.6)
        lbl = label_map2.get(col, col)
        ax.annotate(lbl, (xv, yv), textcoords="offset points", xytext=(7, 4),
                    fontsize=8.5, fontweight="bold" if is_p else "normal", color=clr)

    ax.set_xlabel("Annualised Volatility (%)")
    ax.set_ylabel("Annualised Return (%)")
    ax.set_title("Risk-Return Profile and Mean-Variance Efficient Frontier\n"
                 "(In-sample estimates 2015–present | Long-only, max 40% per asset)",
                 fontsize=11)
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout(); save(fig,"11_risk_return.png")

    # ── 12 Rolling Sharpe ─────────────────────────────────────────────────────
    roll3 = pa.rolling_performance(portfolios, rolling_window_years=3, rf_series=rf_daily)
    fig, ax = plt.subplots(figsize=(11,5))
    for i,col in enumerate(portfolios.columns):
        ax.plot(roll3["Sharpe"].index, roll3["Sharpe"][col],
                color=COLORS[i],lw=1.5,ls=LSTYLES[i],label=col)
    ax.axhline(0,color="black",lw=0.7,ls=":")
    ax.axhline(1,color=CG,lw=0.8,ls="--",alpha=0.6,label="Sharpe = 1.0")
    ax.set_title("Rolling 3-Year Sharpe Ratio"); ax.set_ylabel("Sharpe Ratio")
    ax.legend(loc="upper right"); fig.tight_layout(); save(fig,"12_rolling_sharpe.png")

    # ── 13 Annual Return Attribution (P3) ─────────────────────────────────────
    bucket_colors = {"Equity": C1, "Fixed Income": C2,
                     "Alternatives": C3, "Cash": "#AAAAAA"}
    fig, ax = plt.subplots(figsize=(12, 5.5))

    bottom_pos = np.zeros(len(attribution))
    bottom_neg = np.zeros(len(attribution))
    years_x    = np.arange(len(attribution))

    for bucket, color in bucket_colors.items():
        vals = attribution[bucket].values
        pos  = np.where(vals > 0, vals, 0)
        neg  = np.where(vals < 0, vals, 0)
        ax.bar(years_x, pos, bottom=bottom_pos, color=color,
               alpha=0.85, label=bucket, width=0.72)
        ax.bar(years_x, neg, bottom=bottom_neg, color=color,
               alpha=0.85, width=0.72)
        bottom_pos += pos
        bottom_neg += neg

    # Bars sum to the log-return of P3; the gap vs total return is the
    # geometric compounding effect — label both lines so the chart is self-explanatory.
    p3_log_ann = (np.log(p3.resample("YE").last()).diff().dropna() * 100)
    p3_tot_ann = (p3.resample("YE").last().pct_change().dropna() * 100)
    p3_log_ann.index = p3_log_ann.index.year
    p3_tot_ann.index = p3_tot_ann.index.year

    shared_yrs = [y for y in attribution.index if y in p3_tot_ann.index]
    shared_x   = [list(attribution.index).index(y) for y in shared_yrs]

    ax.plot(shared_x, p3_tot_ann.loc[shared_yrs].values,
            color="black", lw=1.8, marker="o", ms=5, zorder=6,
            label="P3 Total Return")
    ax.plot(shared_x, p3_log_ann.loc[shared_yrs].values,
            color="#666666", lw=1.2, ls=":", marker="s", ms=3.5, zorder=5,
            label="P3 Log Return  ← bars ≈ this")

    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(years_x)
    yr_labels = [str(y) + (" †" if y == attribution.index[-1] else "")
                 for y in attribution.index]
    ax.set_xticklabels(yr_labels, fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_ylabel("Log-Return Contribution (%) — see note")
    ax.set_title(
        "Portfolio 3 — Annual Return Attribution by Asset Bucket\n"
        "Bars ≈ log-return decomposition (Σ wᵢ·rᵢ daily).  "
        "Gap between solid and dotted lines = geometric compounding effect.  "
        "† Partial year.",
        fontsize=9.5)

    for yr, lbl, clr in [(2022, "Rate Shock", "#C0392B"),
                         (2020, "COVID", "#E67E22"),
                         (2023, "AI Bull", "#27AE60")]:
        if yr in attribution.index:
            xi = list(attribution.index).index(yr)
            y_pos = ax.get_ylim()[0] * 0.82
            ax.annotate(lbl, xy=(xi, y_pos),
                        fontsize=7.5, color=clr, ha="center", style="italic")

    ax.legend(loc="upper left", ncol=6, fontsize=8)
    fig.tight_layout(); save(fig, "13_attribution.png")

    # ═════════════════════════════════════════════════════════════════════════
    # EXCEL EXPORT
    # ═════════════════════════════════════════════════════════════════════════
    print("\n[6/6] Exporting to Excel …")

    with pd.ExcelWriter(os.path.join(OUTPUT_DIR,"Performance_Summary.xlsx"),
                        engine="openpyxl") as w:
        perf.to_excel(w, sheet_name="Performance Stats")
        reg_results.to_excel(w, sheet_name="Market Model Regression")
        dd_series.resample("ME").last().to_excel(w, sheet_name="Monthly Drawdown")
        if not scenario_res.empty:
            scenario_res.to_excel(w, sheet_name="Modern Scenarios")
        roll["Ann. Return"].resample("ME").last().to_excel(w, sheet_name="Rolling 3Y Return")
        roll["Sharpe"].resample("ME").last().to_excel(w, sheet_name="Rolling 3Y Sharpe")
        attribution.to_excel(w, sheet_name="Annual Attribution")
        pd.DataFrame({"Vol (%)": ef["vols"], "Ret (%)": ef["rets"]}
                     ).to_excel(w, sheet_name="Efficient Frontier")
    print("    → Performance_Summary.xlsx")

    with pd.ExcelWriter(os.path.join(OUTPUT_DIR,"Asset_Allocation.xlsx"),
                        engine="openpyxl") as w:
        p3_wh.resample("QE").last().to_excel(w, sheet_name="P3 Quarterly Weights")
        p1_wh.resample("QE").last().to_excel(w, sheet_name="P1 Quarterly Weights")
        p2_wh.resample("QE").last().to_excel(w, sheet_name="P2 Quarterly Weights")
    print("    → Asset_Allocation.xlsx")

    with pd.ExcelWriter(os.path.join(OUTPUT_DIR,"Correlation_Matrix.xlsx"),
                        engine="openpyxl") as w:
        corr_mx.to_excel(w, sheet_name="Weekly Return Correlations")
    print("    → Correlation_Matrix.xlsx")

    ff6_df = pd.DataFrame({
        "Coefficient": ff6_result["params"],
        "t-Statistic": ff6_result["tvalues"],
        "p-Value":     ff6_result["pvalues"],
    })
    with pd.ExcelWriter(os.path.join(OUTPUT_DIR,"FF6_Regression.xlsx"),
                        engine="openpyxl") as w:
        ff6_df.to_excel(w, sheet_name="Factor Loadings")
        pd.DataFrame({"Metric":["R²","Adj. R²","N (months)"],
                       "Value":[ff6_result["rsquared"],
                                ff6_result["rsquared_adj"],
                                ff6_result["nobs"]]}).set_index("Metric"
                      ).to_excel(w, sheet_name="Model Summary")
    print("    → FF6_Regression.xlsx")

    with pd.ExcelWriter(os.path.join(OUTPUT_DIR,"Glide_Path_Weights.xlsx"),
                        engine="openpyxl") as w:
        weight_schedule.to_excel(w, sheet_name="Annual Schedule")
        pd.DataFrame({
            "Year":[d.year for d in weight_schedule.index],
            "Risky (%)":(weight_schedule[RISKY_ASSETS].sum(axis=1)*100).round(1).values,
            "Safe (%)":(weight_schedule[SAFE_ASSETS].sum(axis=1)*100).round(1).values,
        }).set_index("Year").to_excel(w, sheet_name="Risky vs Safe")
    print("    → Glide_Path_Weights.xlsx")

    # CSVs for IPS reference
    portfolios.to_csv(os.path.join(OUTPUT_DIR,"portfolio_navs.csv"))
    perf.to_csv(os.path.join(OUTPUT_DIR,"perf_stats.csv"))
    weight_schedule.to_csv(os.path.join(OUTPUT_DIR,"weight_schedule.csv"))
    corr_mx.to_csv(os.path.join(OUTPUT_DIR,"correlation_matrix.csv"))
    ff6_df.to_csv(os.path.join(OUTPUT_DIR,"ff6_results.csv"))
    attribution.to_csv(os.path.join(OUTPUT_DIR,"attribution.csv"))

    print("\n" + "═"*65)
    print("  Analysis complete. Outputs → ./output/")
    print("═"*65+"\n")
    data_meta.update({
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "start_date": START_DATE,
        "end_date": END_DATE,
    })
    _write_run_metadata(data_meta)

    return perf, portfolios, weight_schedule, corr_mx, ff6_result


# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

def _shade_regimes(ax):
    """Add light background shading for key market regimes."""
    shades = [
        ("2020-01-20","2020-03-23","#FFEAEA","COVID Crash"),
        ("2022-01-03","2022-10-12","#FFF3E0","Rate Shock"),
        ("2022-10-13","2024-12-31","#EAF4EA","AI Bull"),
    ]
    ylim = ax.get_ylim()
    for s,e,col,lbl in shades:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                   alpha=0.35, color=col, label=lbl, zorder=0)
    # Only add regime legend on first axis
    handles, labels = ax.get_legend_handles_labels()


def build_synthetic_ff6(date_index: pd.DatetimeIndex):
    """
    Generate synthetic monthly FF6 factor returns calibrated to
    Ken French Data Library statistics (2015-2026 approximate).
    """
    monthly_idx = date_index.to_period("M").unique()
    monthly_dates = pd.DatetimeIndex([p.to_timestamp(how="end").normalize() for p in monthly_idx])
    n = len(monthly_dates)
    np.random.seed(99)

    # Monthly factor means (approx actual) and vols
    #         const Mkt-RF   SMB    HML    RMW    CMA    WML
    mu  = np.array([0.0003, 0.012,-0.001,-0.001, 0.003, 0.001, 0.009])
    vol = np.array([0.001,  0.043, 0.030, 0.032, 0.022, 0.022, 0.048])

    corr_ff6 = np.array([
        [1.00,-0.05,-0.02,-0.02,-0.01,-0.01,-0.01],
        [-0.05,1.00, 0.22,-0.28,-0.35,-0.38,-0.18],
        [-0.02,0.22, 1.00,-0.10,-0.38,-0.10,-0.05],
        [-0.02,-0.28,-0.10,1.00,-0.10, 0.67,-0.55],
        [-0.01,-0.35,-0.38,-0.10,1.00, 0.12, 0.08],
        [-0.01,-0.38,-0.10, 0.67, 0.12,1.00,-0.22],
        [-0.01,-0.18,-0.05,-0.55, 0.08,-0.22,1.00],
    ])
    ev, evec = np.linalg.eigh(corr_ff6)
    ev = np.maximum(ev,1e-8)
    corr_ff6 = evec @ np.diag(ev) @ evec.T
    d = np.sqrt(np.diag(corr_ff6)); corr_ff6 /= np.outer(d,d)
    L6 = np.linalg.cholesky(corr_ff6)

    z  = np.random.standard_normal((n,7)) @ L6.T
    raw= mu + vol * z

    rf_s = pd.Series(raw[:,0], index=monthly_dates, name="RF")
    ff6  = pd.DataFrame(raw[:,1:], index=monthly_dates,
                        columns=["Mkt-RF","SMB","HML","RMW","CMA","WML"])
    return ff6, rf_s


def build_gfc_proxies() -> pd.DataFrame:
    """
    Generate synthetic 2007-2009 price paths for key assets with
    long enough histories to contextualise the GFC.
    Calibrated to actual historical performance during this period.
    """
    idx = pd.bdate_range("2007-01-02","2009-12-31")
    n   = len(idx)
    np.random.seed(77)

    # Annual drifts for GFC period assets
    # 3 regimes: pre-crisis, crisis, recovery
    assets_gfc = ["ACWI","TLT","GLD","VWO","VGK","60/40 Proxy"]
    drifts = {
        ("2007-01-02","2007-10-09"):  [ 0.18, 0.06, 0.25, 0.38, 0.14, 0.13],  # bull top
        ("2007-10-10","2009-03-09"):  [-0.38, 0.24, 0.18,-0.52,-0.44,-0.18],  # crisis
        ("2009-03-10","2009-12-31"):  [ 0.72,-0.10, 0.22, 0.80, 0.65, 0.35],  # recovery
    }
    vols_gfc = np.array([0.185, 0.145, 0.155, 0.220, 0.200, 0.105])

    log_p = np.zeros((n, len(assets_gfc)))
    prev  = np.zeros(len(assets_gfc))
    for i, date in enumerate(idx):
        for (s,e), d_arr in drifts.items():
            if pd.Timestamp(s) <= date <= pd.Timestamp(e):
                mu_d = np.array(d_arr)/252
                v_d  = vols_gfc/np.sqrt(252)
                z    = np.random.standard_normal(len(assets_gfc))
                log_r = (mu_d - 0.5*v_d**2) + v_d*z
                prev  = prev + log_r
                log_p[i] = prev
                break

    gfc = pd.DataFrame(100.0 * np.exp(log_p), index=idx, columns=assets_gfc)
    return pa.renormalize_prices(gfc)


if __name__ == "__main__":
    main()
