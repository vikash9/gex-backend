from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
import requests
from concurrent.futures import ThreadPoolExecutor
import hmac
import io
import os
import threading
import time
from typing import Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def calculate_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def is_monthly_opex(date_obj):
    if date_obj.weekday() != 4:
        return False
    return 15 <= date_obj.day <= 21

def is_quarterly_opex(date_obj):
    if date_obj.month not in [3, 6, 9, 12]:
        return False
    next_month = date_obj.replace(day=28) + pd.Timedelta(days=4)
    last_day_of_month = next_month - pd.Timedelta(days=next_month.day)
    if last_day_of_month.weekday() == 5:
        last_bus_day = last_day_of_month - pd.Timedelta(days=1)
    elif last_day_of_month.weekday() == 6:
        last_bus_day = last_day_of_month - pd.Timedelta(days=2)
    else:
        last_bus_day = last_day_of_month
    return date_obj.date() == last_bus_day.date()

@app.get("/api/gex")
def get_gex(ticker: str = "SPY", expiration: str = None):
    ticker_symbol = ticker.upper()
    try:
        ticker_obj = yf.Ticker(ticker_symbol)
        hist_weekly = ticker_obj.history(period="5y", interval="1wk")
        if hist_weekly.empty:
            return {"error": f"Ticker '{ticker_symbol}' spot data unavailable."}
            
        spot_price = float(hist_weekly['Close'].iloc[-1])
        hist_weekly['200_WMA'] = hist_weekly['Close'].rolling(window=200).mean()
        wma_200 = float(hist_weekly['200_WMA'].iloc[-1]) if not np.isnan(hist_weekly['200_WMA'].iloc[-1]) else 0.0
    except Exception as e:
        return {"error": f"Failed to fetch stock history: {str(e)}"}

    cboe_url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/_{ticker_symbol}.json"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    try:
        res = requests.get(cboe_url, headers=headers, timeout=10)
        if res.status_code != 200:
            cboe_url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker_symbol}.json"
            res = requests.get(cboe_url, headers=headers, timeout=10)
            
        data = res.json()
        options_data = data.get("data", {}).get("options", [])
        if not options_data:
            return {"error": f"No options data found for {ticker_symbol}."}
    except Exception as e:
        return {"error": f"Failed to reach options server: {str(e)}"}

    exp_set = set()
    for item in options_data:
        sym = item.get("option", "")
        if sym and len(sym) >= 15:
            date_str = sym[len(ticker_symbol):len(ticker_symbol)+6]
            if len(date_str) == 6 and date_str.isdigit():
                formatted_date = f"20{date_str[0:2]}-{date_str[2:4]}-{date_str[4:6]}"
                exp_set.add(formatted_date)

    available_expirations = sorted(list(exp_set))
    selected_exp = expiration if expiration in available_expirations else (available_expirations[0] if available_expirations else None)

    r = 0.045
    if selected_exp:
        exp_date = pd.to_datetime(selected_exp)
        today = pd.to_datetime('today')
        dte = max((exp_date - today).days, 1) / 365.0
    else:
        dte = 7 / 365.0

    strikes_dict = {}
    for item in options_data:
        sym = item.get("option", "")
        oi = item.get("open_interest", 0) or 0
        iv = item.get("iv", 0.2) or 0.2

        if not sym or len(sym) < 15:
            continue

        if selected_exp:
            exp_code = sym[len(ticker_symbol):len(ticker_symbol)+6]
            expected_code = selected_exp.replace("-", "")[2:]
            if exp_code != expected_code:
                continue

        contract_type = "call" if "C" in sym[len(ticker_symbol):] else "put"
        try:
            K = float(sym[-8:]) / 1000.0
        except ValueError:
            continue

        if K not in strikes_dict:
            strikes_dict[K] = {"call_OI": 0, "put_OI": 0, "call_IV": 0.2, "put_IV": 0.2}

        if contract_type == "call":
            strikes_dict[K]["call_OI"] += oi
            strikes_dict[K]["call_IV"] = iv if iv > 0 else 0.2
        else:
            strikes_dict[K]["put_OI"] += oi
            strikes_dict[K]["put_IV"] = iv if iv > 0 else 0.2

    strikes_data = []
    for K, vals in strikes_dict.items():
        c_gamma = calculate_gamma(spot_price, K, dte, r, vals["call_IV"])
        p_gamma = calculate_gamma(spot_price, K, dte, r, vals["put_IV"])

        c_gex = float(c_gamma * vals["call_OI"] * 100 * (spot_price**2) * 0.01 / 1e6)
        p_gex = float(-(p_gamma * vals["put_OI"] * 100 * (spot_price**2) * 0.01 / 1e6))
        net_gex = c_gex + p_gex

        strikes_data.append({
            "strike": K,
            "call_gex": round(c_gex, 2),
            "put_gex": round(p_gex, 2),
            "net_gex": round(net_gex, 2)
        })

    strikes_filtered = [s for s in strikes_data if spot_price * 0.85 <= s["strike"] <= spot_price * 1.15]
    strikes_filtered.sort(key=lambda x: x["strike"])

    call_wall = max(strikes_filtered, key=lambda x: x["call_gex"])["strike"] if strikes_filtered else 0
    put_wall = min(strikes_filtered, key=lambda x: x["put_gex"])["strike"] if strikes_filtered else 0
    total_net_gex = sum(s["net_gex"] for s in strikes_filtered)

    gamma_flip = spot_price
    cum_gex = 0
    for s in strikes_filtered:
        prev_cum = cum_gex
        cum_gex += s["net_gex"]
        if (prev_cum < 0 and cum_gex >= 0) or (prev_cum > 0 and cum_gex <= 0):
            gamma_flip = s["strike"]
            break

    return {
        "ticker": ticker_symbol,
        "selected_expiration": selected_exp,
        "expirations": available_expirations,
        "spot_price": round(spot_price, 2),
        "wma_200": round(wma_200, 2),
        "total_net_gex": round(total_net_gex, 2),
        "call_wall": call_wall,
        "put_wall": put_wall,
        "gamma_flip": round(gamma_flip, 2),
        "strikes": strikes_filtered
    }

@app.get("/api/gex-heatmap")
def get_gex_heatmap(ticker: str = "SPY"):
    ticker_symbol = ticker.upper()
    try:
        ticker_obj = yf.Ticker(ticker_symbol)
        hist = ticker_obj.history(period="1d")
        if hist.empty:
            return {"error": f"Ticker '{ticker_symbol}' not found."}
        spot_price = float(hist['Close'].iloc[-1])
    except Exception as e:
        return {"error": f"Spot price error: {str(e)}"}

    cboe_url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/_{ticker_symbol}.json"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    try:
        res = requests.get(cboe_url, headers=headers, timeout=10)
        if res.status_code != 200:
            cboe_url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker_symbol}.json"
            res = requests.get(cboe_url, headers=headers, timeout=10)
            
        data = res.json()
        options_data = data.get("data", {}).get("options", [])
    except Exception as e:
        return {"error": f"Failed to fetch options: {str(e)}"}

    r = 0.045
    today = pd.to_datetime('today')
    heatmap_matrix = {}
    expirations_set = set()

    for item in options_data:
        sym = item.get("option", "")
        oi = item.get("open_interest", 0) or 0
        iv = item.get("iv", 0.2) or 0.2

        if not sym or len(sym) < 15:
            continue

        date_str = sym[len(ticker_symbol):len(ticker_symbol)+6]
        if not (len(date_str) == 6 and date_str.isdigit()):
            continue
            
        formatted_exp = f"20{date_str[0:2]}-{date_str[2:4]}-{date_str[4:6]}"
        exp_date = pd.to_datetime(formatted_exp)

        if not (is_monthly_opex(exp_date) or is_quarterly_opex(exp_date)):
            continue

        expirations_set.add(formatted_exp)
        dte = max((exp_date - today).days, 1) / 365.0

        contract_type = "call" if "C" in sym[len(ticker_symbol):] else "put"
        try:
            K = float(sym[-8:]) / 1000.0
        except ValueError:
            continue

        if not (spot_price * 0.80 <= K <= spot_price * 1.20):
            continue

        if K not in heatmap_matrix:
            heatmap_matrix[K] = {}

        if formatted_exp not in heatmap_matrix[K]:
            heatmap_matrix[K][formatted_exp] = {"call_OI": 0, "put_OI": 0, "call_IV": 0.2, "put_IV": 0.2}

        if contract_type == "call":
            heatmap_matrix[K][formatted_exp]["call_OI"] += oi
            heatmap_matrix[K][formatted_exp]["call_IV"] = iv if iv > 0 else 0.2
        else:
            heatmap_matrix[K][formatted_exp]["put_OI"] += oi
            heatmap_matrix[K][formatted_exp]["put_IV"] = iv if iv > 0 else 0.2

    sorted_expirations = sorted(list(expirations_set))[:10]
    grid_data = []
    sorted_strikes = sorted(heatmap_matrix.keys(), reverse=True)

    for K in sorted_strikes:
        row = {"strike": K, "is_spot": abs(K - spot_price) < (spot_price * 0.005)}
        for exp in sorted_expirations:
            vals = heatmap_matrix[K].get(exp, {"call_OI": 0, "put_OI": 0, "call_IV": 0.2, "put_IV": 0.2})
            exp_date = pd.to_datetime(exp)
            dte = max((exp_date - today).days, 1) / 365.0
            
            c_gamma = calculate_gamma(spot_price, K, dte, r, vals["call_IV"])
            p_gamma = calculate_gamma(spot_price, K, dte, r, vals["put_IV"])

            c_gex = float(c_gamma * vals["call_OI"] * 100 * (spot_price**2) * 0.01 / 1e6)
            p_gex = float(-(p_gamma * vals["put_OI"] * 100 * (spot_price**2) * 0.01 / 1e6))
            row[exp] = round(c_gex + p_gex, 1)
        grid_data.append(row)

    return {
        "ticker": ticker_symbol,
        "spot_price": round(spot_price, 2),
        "expirations": sorted_expirations,
        "matrix": grid_data
    }


# =====================================================================
# SMC BUY ZONE SCANNER (v2)
# ---------------------------------------------------------------------
# What changed vs v1:
#   * Order block = lowest-low candle between the broken swing high and
#     the BOS candle (LuxAlgo-style), not "last red candle before BOS"
#   * Mitigated zones are dropped (close below zone low, or wick mode)
#   * ALL bullish OBs are evaluated; the nearest unmitigated one is used
#   * Price inside the zone is now included ("IN ZONE")
#   * Swing length is a parameter (default 50; use 5 for internal-style)
#   * Distance reported in % and in ATR(14) units
#   * Unadjusted-for-dividends prices (split-adjusted only), like TradingView
#   * Today's unfinished candle is dropped while the market is open
#   * One batched download instead of ~1000 individual Yahoo calls
#   * Failures are counted and reported instead of silently swallowed
#   * Results are cached; the endpoint no longer re-scans on every request
# =====================================================================

SCAN_PERIOD = "2y"                      # history pulled per ticker
MIN_PRICE = 5.0                         # skip penny stocks
MIN_AVG_DOLLAR_VOLUME = 20_000_000      # 50-day avg $ volume floor
DOWNLOAD_CHUNK = 100                    # tickers per Yahoo batch request
CACHE_TTL_SECONDS = 6 * 60 * 60         # re-scan at most every 6 hours
DEFAULT_SWING_LENGTH = 50
REFRESH_TOKEN = os.environ.get("SMC_REFRESH_TOKEN")  # set on Render to allow ?refresh=true

_FALLBACK_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "BRK-B", "UNH",
    "JNJ", "JPM", "V", "XOM", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "COST", "PEP",
    "KO", "BAC", "WMT", "MCD", "CSCO", "CRM", "ACN", "ABT", "LIN", "ORCL", "INTC",
    "DIS", "TXN", "PM", "CAT", "QCOM", "GE", "AMAT", "IBM", "BA", "AMGN", "NOW",
    "SPGI", "HON", "GS", "COP", "BLK", "RTX", "TMO", "PLTR", "PANW", "MU", "NFLX",
]

_last_good_universe = {"symbols": None}
_scan_cache = {}
_scan_lock = threading.Lock()


# ---------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------
def fetch_us_market_universe():
    """Returns (symbols, source_label). Source label makes fallbacks visible."""
    symbols, source = None, None
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df_sp = pd.read_csv(io.StringIO(resp.text))
        if not df_sp.empty and "Symbol" in df_sp.columns:
            symbols = [str(s).strip().replace(".", "-") for s in df_sp["Symbol"].dropna().tolist()]
            source = "S&P 500 (GitHub constituents.csv)"
            _last_good_universe["symbols"] = symbols
    except Exception:
        pass

    if symbols is None and _last_good_universe["symbols"]:
        symbols = list(_last_good_universe["symbols"])
        source = "S&P 500 (last good copy; live fetch failed)"

    if symbols is None:
        symbols = list(_FALLBACK_UNIVERSE)
        source = "FALLBACK SHORT LIST (S&P 500 fetch failed)"

    # Optional: add your own names via env var, e.g. SMC_EXTRA_TICKERS="SOFI,HOOD,RKLB"
    extra_raw = os.environ.get("SMC_EXTRA_TICKERS", "")
    extras = [t.strip().upper().replace(".", "-") for t in extra_raw.split(",") if t.strip()]
    if extras:
        symbols = list(dict.fromkeys(symbols + extras))
        source += f" + {len(extras)} extra"

    return symbols, source


# ---------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------
def _download_chunk(tickers):
    for _ in range(2):
        try:
            data = yf.download(
                tickers,
                period=SCAN_PERIOD,
                interval="1d",
                auto_adjust=False,   # split-adjusted only (matches TradingView default)
                group_by="ticker",
                threads=True,
                progress=False,
            )
            if data is not None and not data.empty:
                return data
        except Exception:
            pass
        time.sleep(2)
    return None


def _extract_ticker_frame(data, ticker):
    if data is None:
        return None
    cols = data.columns
    if isinstance(cols, pd.MultiIndex):
        if ticker not in cols.get_level_values(0):
            return None
        d = data[ticker]
    else:
        d = data
    need = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in d.columns for c in need):
        return None
    d = d[need].dropna()
    return d if len(d) else None


def _drop_partial_bar(df):
    """Drop today's still-forming daily candle while the US market is open."""
    if df.empty:
        return df
    now_et = pd.Timestamp.now(tz="America/New_York")
    last_date = pd.Timestamp(df.index[-1]).date()
    before_settle = (now_et.hour, now_et.minute) < (16, 10)
    if last_date == now_et.date() and now_et.weekday() < 5 and before_settle:
        return df.iloc[:-1]
    return df


def _atr(high, low, close, period):
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()


def _fmt_date(ts):
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------
# Core SMC logic
# ---------------------------------------------------------------------
def find_bullish_order_blocks(df, length, mitigation="close"):
    """
    Bullish OB detection (approximation of LuxAlgo's logic):
      1. Find pivot highs (highest high within +/- `length` bars).
      2. BOS = first CLOSE above that pivot high.
      3. OB candle = lowest-low candle between the pivot and the BOS bar
         (bars with range >= 2x ATR(200) are de-prioritised, like LuxAlgo's
         high-volatility filter).
      4. Mitigated if a later close (or low, in 'wick' mode) is below the OB low.
    Returns a list of dicts (mitigated ones included, flagged).
    """
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    n = len(c)

    atr200 = _atr(h, l, c, 200)
    high_vol = (h - l) >= 2.0 * atr200
    parsed_low = np.where(high_vol, h, l)

    roll_max = pd.Series(h).rolling(2 * length + 1, center=True).max().to_numpy()
    pivots = np.where(h == roll_max)[0]   # NaN edges compare False automatically

    found = {}
    for i in pivots:
        level = h[i]
        broke = np.nonzero(c[i + 1:] > level)[0]
        if broke.size == 0:
            continue
        j = i + 1 + int(broke[0])                              # BOS bar
        k = (i + 1) + int(np.argmin(parsed_low[i + 1:j + 1]))  # OB candle
        if k in found:
            continue

        zone_low, zone_high = float(l[k]), float(h[k])

        later = l[k + 1:] if mitigation == "wick" else c[k + 1:]
        mitigated = bool(later.size and (later < zone_low).any())

        touched = l[j + 1:] <= zone_high
        if touched.size:
            starts = touched & ~np.concatenate(([False], touched[:-1]))
            touches = int(starts.sum())
        else:
            touches = 0

        found[k] = {
            "ob_idx": int(k),
            "bos_idx": int(j),
            "swing_idx": int(i),
            "swing_level": float(level),
            "zone_low": zone_low,
            "zone_high": zone_high,
            "mitigated": mitigated,
            "touches": touches,
        }
    return list(found.values())


def analyze_ticker(ticker, df, length, mitigation):
    """Returns a result row (nearest unmitigated bullish OB at/below price) or None."""
    df = _drop_partial_bar(df)
    if len(df) < 2 * length + 30:
        return None

    c = df["Close"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    spot = float(c[-1])

    if spot < MIN_PRICE:
        return None
    avg_dollar_vol = float((df["Close"] * df["Volume"]).tail(50).mean())
    if avg_dollar_vol < MIN_AVG_DOLLAR_VOLUME:
        return None

    obs = find_bullish_order_blocks(df, length, mitigation)
    cands = [z for z in obs if not z["mitigated"] and spot >= z["zone_low"]]
    if not cands:
        return None

    for z in cands:
        z["dist_pct"] = max(0.0, (spot - z["zone_high"]) / z["zone_high"] * 100.0)
    best = min(cands, key=lambda z: (z["dist_pct"], -z["ob_idx"]))  # nearest, then most recent

    atr14 = float(_atr(h, l, c, 14)[-1])
    in_zone = spot <= best["zone_high"]
    dist_pct = round(best["dist_pct"], 2)
    dist_atr = 0.0 if in_zone or atr14 <= 0 else round((spot - best["zone_high"]) / atr14, 2)
    tv_symbol = ticker.replace("-", ".")

    return {
        "ticker": ticker,
        "spot_price": round(spot, 2),
        "zone_low": round(best["zone_low"], 2),
        "zone_high": round(best["zone_high"], 2),
        "distance_pct": dist_pct,
        "distance_atr": dist_atr,
        "in_zone": in_zone,
        "status": "IN ZONE" if in_zone else f"+{dist_pct}% ABOVE ZONE",
        "ob_date": _fmt_date(df.index[best["ob_idx"]]),
        "bos_date": _fmt_date(df.index[best["bos_idx"]]),
        "swing_high_date": _fmt_date(df.index[best["swing_idx"]]),
        "broken_swing_level": round(best["swing_level"], 2),
        "zone_age_bars": int(len(df) - 1 - best["ob_idx"]),
        "touches": best["touches"],
        "active_zones": len(cands),
        "atr14": round(atr14, 2),
        "avg_dollar_volume_m": round(avg_dollar_vol / 1e6, 1),
        "tv_symbol": tv_symbol,
        "tv_link": f"https://www.tradingview.com/chart/?symbol={tv_symbol}",
        "_last_bar": _fmt_date(df.index[-1]),
    }


# ---------------------------------------------------------------------
# Full scan + cache
# ---------------------------------------------------------------------
def run_full_scan(length, mitigation):
    started = time.time()
    universe, source = fetch_us_market_universe()

    rows, failed = [], []
    scanned = 0
    last_bar = None

    for start in range(0, len(universe), DOWNLOAD_CHUNK):
        chunk = universe[start:start + DOWNLOAD_CHUNK]
        data = _download_chunk(chunk)
        for t in chunk:
            df = _extract_ticker_frame(data, t)
            if df is None:
                failed.append(t)
                continue
            try:
                row = analyze_ticker(t, df, length, mitigation)
            except Exception:
                failed.append(t)
                continue
            scanned += 1
            if row:
                last_bar = max(last_bar or row["_last_bar"], row["_last_bar"])
                rows.append(row)
        time.sleep(0.5)

    for r in rows:
        r.pop("_last_bar", None)

    return {
        "rows": rows,
        "universe_source": source,
        "universe_count": len(universe),
        "scanned_count": scanned,
        "failed_count": len(failed),
        "failed_tickers": failed[:50],
        "last_bar_date": last_bar,
        "as_of_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S"),
        "scan_seconds": round(time.time() - started, 1),
    }


def get_scan_result(length, mitigation, force=False):
    key = (length, mitigation)
    requested_at = time.time()

    entry = _scan_cache.get(key)
    if entry and not force and requested_at - entry["_ts"] < CACHE_TTL_SECONDS:
        return entry

    with _scan_lock:   # only one scan at a time; waiting requests reuse the result
        entry = _scan_cache.get(key)
        if entry:
            fresh = time.time() - entry["_ts"] < CACHE_TTL_SECONDS
            if (not force and fresh) or (force and entry["_ts"] >= requested_at):
                return entry

        new_entry = run_full_scan(length, mitigation)
        if new_entry["scanned_count"] == 0:
            # Total failure (e.g. Yahoo blocking us): keep serving the old cache if we have one
            if entry:
                return dict(entry, stale=True)
            return new_entry
        new_entry["_ts"] = time.time()
        _scan_cache[key] = new_entry
        return new_entry


@app.on_event("startup")
def _warm_smc_cache():
    def _job():
        try:
            get_scan_result(DEFAULT_SWING_LENGTH, "close")
        except Exception:
            pass
    threading.Thread(target=_job, daemon=True).start()


@app.get("/api/smc-scan")
def get_smc_scan(
    length: int = DEFAULT_SWING_LENGTH,
    max_distance_pct: float = 5.0,
    max_atr: Optional[float] = None,
    mitigation: str = "close",
    refresh: bool = False,
    token: Optional[str] = None,
):
    """
    length            swing length (50 = LuxAlgo-style swing structure, 5 = internal-style)
    max_distance_pct  max % above the zone's top edge (in-zone tickers always pass)
    max_atr           optional extra filter: max distance in ATR(14) units
    mitigation        'close' (zone dies on a close below its low) or 'wick'
    refresh           force a re-scan; only honoured if SMC_REFRESH_TOKEN is set and `token` matches
    """
    length = min(max(int(length), 3), 100)
    mitigation = mitigation if mitigation in ("close", "wick") else "close"

    force = bool(
        refresh and REFRESH_TOKEN and token and hmac.compare_digest(str(token), str(REFRESH_TOKEN))
    )
    entry = get_scan_result(length, mitigation, force=force)

    results = [
        r for r in entry["rows"]
        if r["distance_pct"] <= max_distance_pct
        and (max_atr is None or r["distance_atr"] <= max_atr)
    ]
    results.sort(key=lambda r: (r["distance_pct"], r["distance_atr"]))

    return {
        "timeframe": "1d (Daily)",
        "min_market_cap": "None (S&P 500 universe; price >= $5, 50d avg $ volume >= $20M)",
        "parameters": {
            "swing_length": length,
            "mitigation": mitigation,
            "max_distance_pct": max_distance_pct,
            "max_atr": max_atr,
        },
        "universe_source": entry["universe_source"],
        "scanned_universe_count": entry["universe_count"],
        "scanned_ok_count": entry["scanned_count"],
        "failed_count": entry["failed_count"],
        "failed_tickers": entry["failed_tickers"],
        "last_bar_date": entry["last_bar_date"],
        "as_of_utc": entry["as_of_utc"],
        "scan_seconds": entry["scan_seconds"],
        "stale": entry.get("stale", False),
        "total_matches": len(results),
        "watchlist_csv": ",".join(r["tv_symbol"] for r in results),
        "results": results,
    }
