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
import re
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


# =====================================================================
# GEX ENGINE (v2)
# ---------------------------------------------------------------------
# What changed vs v1:
#   * Option symbols are parsed with a regex from the right, so SPXW / NDXP
#     style roots (where 0DTE and weeklies live) are no longer dropped
#   * Spot comes from the CBOE quote feed (same timestamp as the options),
#     with Yahoo only as a fallback
#   * Default view aggregates ALL unexpired expirations (use expiration=
#     YYYY-MM-DD, next, or max_dte=N to narrow it)
#   * Time to expiry is measured to 4:00pm ET on expiration day, in fractional
#     years, with a 15-minute floor (v1 used whole days, minimum 1 day)
#   * Contracts with open interest but no usable IV are EXCLUDED and reported
#     (v1 silently substituted 20% IV)
#   * Gamma flip is a true zero-gamma level: total GEX is recomputed on a grid
#     of hypothetical spot prices and the zero crossing is interpolated
#     (v1 summed strike by strike, which is not the same thing)
#   * Totals use every strike; only the chart list is limited to +/-15%
#   * Heatmap marks only the nearest strike as spot, can include weeklies,
#     and handles holiday-shifted monthly expirations
#   * Option chain is cached for 90s so the tabs respond quickly
#   * Diagnostics block shows what was used / dropped, plus a model-vs-CBOE
#     gamma sanity check
# Units: gamma exposure in $ millions per 1% move in the underlying.
# Assumptions: dealers long calls / short puts, open interest as of the prior
# close, IV held fixed when re-pricing at other spot levels.
# =====================================================================

GEX_RISK_FREE = 0.045
GEX_CHAIN_TTL = 90                      # seconds to cache a CBOE chain
WMA_TTL = 6 * 3600
SECONDS_PER_YEAR = 365.0 * 24 * 3600
MIN_T_SECONDS = 15 * 60                 # floor on time to expiry
IV_MIN, IV_MAX = 0.01, 5.0              # usable IV range (decimal)
PROFILE_SPAN = 0.20                     # +/-20% spot grid for the gamma profile
PROFILE_POINTS = 161

INDEX_SYMBOLS = {"SPX", "NDX", "RUT", "VIX", "XSP", "DJX", "OEX", "XND"}
YF_SYMBOL_MAP = {"SPX": "^GSPC", "NDX": "^NDX", "RUT": "^RUT", "VIX": "^VIX"}

_OCC_RE = re.compile(
    r"^(?P<root>[A-Z0-9.]+?)(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})(?P<cp>[CP])(?P<k>\d{8})$"
)
_chain_cache = {}
_wma_cache = {}


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
def _now_et():
    return pd.Timestamp.now(tz="America/New_York")


def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _opt_num(x):
    """float or None (for fields that may be absent)."""
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _clean_symbol(ticker):
    s = (ticker or "SPY").strip().upper().lstrip("^")
    if s == "GSPC":
        s = "SPX"
    return s


def bs_gamma_vec(S, K, T, r, q, sigma):
    """Vectorised Black-Scholes(-Merton) gamma. Invalid inputs give 0."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        sqrt_t = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_t)
        pdf = np.exp(-0.5 * d1 ** 2) / np.sqrt(2.0 * np.pi)
        gamma = np.exp(-q * T) * pdf / (S * sigma * sqrt_t)
    return np.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------
# Data fetching (cached)
# ---------------------------------------------------------------------
def fetch_cboe_chain(symbol):
    cached = _chain_cache.get(symbol)
    if cached and time.time() - cached[0] < GEX_CHAIN_TTL:
        return cached[1]

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    base = "https://cdn.cboe.com/api/global/delayed_quotes/options/"
    urls = [f"{base}_{symbol}.json", f"{base}{symbol}.json"]
    if symbol not in INDEX_SYMBOLS:
        urls.reverse()

    last_err = None
    for url in urls:
        try:
            res = requests.get(url, headers=headers, timeout=15)
            if res.status_code != 200:
                last_err = f"HTTP {res.status_code}"
                continue
            payload = res.json()
            data = payload.get("data") or {}
            if data.get("options"):
                result = {
                    "options": data["options"],
                    "current_price": data.get("current_price"),
                    "timestamp": payload.get("timestamp") or data.get("last_trade_time"),
                    "price_change": _opt_num(data.get("price_change")),
                }
                _chain_cache[symbol] = (time.time(), result)
                return result
            last_err = "empty options list"
        except Exception as e:
            last_err = str(e)
    raise ValueError(f"no options data ({last_err})")


def _resolve_spot(symbol, chain):
    px = _num(chain.get("current_price"))
    if px > 0:
        return px, "CBOE"
    yf_symbol = YF_SYMBOL_MAP.get(symbol, symbol)
    try:
        px = _num(getattr(yf.Ticker(yf_symbol).fast_info, "last_price", 0))
        if px > 0:
            return px, "Yahoo"
    except Exception:
        pass
    hist = yf.Ticker(yf_symbol).history(period="5d")
    if not hist.empty:
        return float(hist["Close"].iloc[-1]), "Yahoo (last close)"
    raise ValueError("spot price unavailable")


def _get_wma200(symbol):
    cached = _wma_cache.get(symbol)
    if cached and time.time() - cached[0] < WMA_TTL:
        return cached[1]
    value = None
    try:
        hist = yf.Ticker(YF_SYMBOL_MAP.get(symbol, symbol)).history(period="5y", interval="1wk")
        if not hist.empty:
            w = hist["Close"].rolling(window=200).mean().iloc[-1]
            if not np.isnan(w):
                value = float(w)
    except Exception:
        pass
    # cache failures only briefly so a transient Yahoo error doesn't stick for hours
    stamp = time.time() if value is not None else time.time() - WMA_TTL + 300
    _wma_cache[symbol] = (stamp, value)
    return value


# ---------------------------------------------------------------------
# Chain parsing and selection
# ---------------------------------------------------------------------
def parse_chain(options, symbol, now_et):
    """Returns (DataFrame of unexpired contracts | None, sorted expiration list, diagnostics)."""
    diag = {"contracts_total": len(options), "unparsed": 0, "other_root": 0, "expired": 0}
    exp_secs = {}
    rows = []

    for opt in options:
        m = _OCC_RE.match(str(opt.get("option") or ""))
        if not m:
            diag["unparsed"] += 1
            continue
        root = m.group("root")
        suffix = root[len(symbol):]
        # accept ROOT, ROOT+W (weeklies) and ROOT+P (PM-settled); reject adjusted roots like SPY1
        if not root.startswith(symbol) or (suffix and not (suffix.isalpha() and len(suffix) <= 2)):
            diag["other_root"] += 1
            continue

        exp = f"20{m.group('yy')}-{m.group('mm')}-{m.group('dd')}"
        if exp not in exp_secs:
            try:
                close = pd.Timestamp(f"{exp} 16:00", tz="America/New_York")
                exp_secs[exp] = (close - now_et).total_seconds()
            except Exception:
                exp_secs[exp] = None
        secs = exp_secs[exp]
        if secs is None:
            diag["unparsed"] += 1
            continue
        if secs <= 0:
            diag["expired"] += 1
            continue

        rows.append((
            int(m.group("k")) / 1000.0,
            m.group("cp") == "C",
            _num(opt.get("open_interest")),
            _num(opt.get("iv")),
            exp,
            _num(opt.get("gamma")),
        ))

    if not rows:
        return None, [], diag

    d = pd.DataFrame(rows, columns=["K", "is_call", "oi", "iv", "exp", "gamma_cboe"])
    d["secs"] = d["exp"].map(exp_secs)
    d["T"] = np.maximum(d["secs"], MIN_T_SECONDS) / SECONDS_PER_YEAR
    d["dte"] = d["secs"] / 86400.0
    return d, sorted(d["exp"].unique().tolist()), diag


def _select_expirations(d, exps, expiration, max_dte):
    sel, label, note = d, "ALL", None
    raw = (expiration or "all").strip()
    key = raw.lower()
    if key == "next":
        label = exps[0]
        sel = d[d["exp"] == label]
    elif key not in ("all", ""):
        if raw in exps:
            label = raw
            sel = d[d["exp"] == raw]
        else:
            note = f"Expiration {raw} is not in the chain; showing ALL expirations."
    if max_dte is not None:
        sel = sel[sel["dte"] <= max_dte]
        label = f"{label} (<= {max_dte:g} DTE)"
    return sel, label, note


def _valid_mask(d):
    return (d["oi"] > 0) & (d["iv"] >= IV_MIN) & (d["iv"] <= IV_MAX)


# ---------------------------------------------------------------------
# GEX math
# ---------------------------------------------------------------------
def compute_gex(sel, spot, r, q):
    """Per-contract and per-strike gamma exposure at the current spot."""
    valid = _valid_mask(sel)
    oi_total = float(sel.loc[sel["oi"] > 0, "oi"].sum())
    u = sel[valid].copy()
    if u.empty:
        return None

    g = bs_gamma_vec(spot, u["K"].to_numpy(), u["T"].to_numpy(), r, q, u["iv"].to_numpy())
    gex = g * u["oi"].to_numpy() * 100.0 * spot ** 2 * 0.01 / 1e6
    is_call = u["is_call"].to_numpy()
    u["gex"] = np.where(is_call, gex, -gex)                 # calls +, puts -
    u["call_gex"] = np.where(is_call, gex, 0.0)
    u["put_gex"] = np.where(is_call, 0.0, -gex)
    u["call_oi"] = np.where(is_call, u["oi"].to_numpy(), 0.0)
    u["put_oi"] = np.where(is_call, 0.0, u["oi"].to_numpy())

    by = u.groupby("K")[["call_gex", "put_gex", "gex", "call_oi", "put_oi"]].sum()
    return {
        "u": u,
        "by": by,
        "total": float(by["gex"].sum()),
        "oi_total": oi_total,
        "oi_used": float(u["oi"].sum()),
    }


def gamma_profile(u, spot, r, q):
    """Total net GEX recomputed at hypothetical spot prices (IV held fixed)."""
    u = u[(u["K"] >= 0.5 * spot) & (u["K"] <= 1.5 * spot)]
    K, T, iv, oi = u["K"].to_numpy(), u["T"].to_numpy(), u["iv"].to_numpy(), u["oi"].to_numpy()
    sign = np.where(u["is_call"].to_numpy(), 1.0, -1.0)
    grid = np.linspace(spot * (1 - PROFILE_SPAN), spot * (1 + PROFILE_SPAN), PROFILE_POINTS)
    vals = np.empty(len(grid))
    for i, s in enumerate(grid):
        gm = bs_gamma_vec(s, K, T, r, q, iv)
        vals[i] = float(np.sum(sign * gm * oi) * 100.0 * s ** 2 * 0.01 / 1e6)
    return grid, vals


def find_flips(grid, vals, spot):
    flips = []
    for i in range(len(grid) - 1):
        a, b = vals[i], vals[i + 1]
        if a == 0:
            flips.append(float(grid[i]))
        elif a * b < 0:
            flips.append(float(grid[i] - a * (grid[i + 1] - grid[i]) / (b - a)))
    flips.sort(key=lambda x: abs(x - spot))
    return flips


def _is_opex_exp(exp_str, exp_set):
    """Monthly (3rd Friday), quarter-end, or a Thursday standing in for a holiday Friday."""
    d = pd.Timestamp(exp_str)
    if is_monthly_opex(d) or is_quarterly_opex(d):
        return True
    if d.weekday() == 3 and 14 <= d.day <= 20:
        return (d + pd.Timedelta(days=1)).strftime("%Y-%m-%d") not in exp_set
    return False


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------
@app.get("/api/gex")
def get_gex(
    ticker: str = "SPY",
    expiration: str = "all",
    max_dte: Optional[float] = None,
    r: float = GEX_RISK_FREE,
    q: float = 0.0,
):
    """
    expiration  'all' (default), 'next', or a date like 2026-11-20
    max_dte     optional cap on days to expiration (e.g. 0 or 1 = 0DTE/1DTE, 7 = this week)
    r, q        risk-free rate and dividend yield used in the gamma calculation
    """
    symbol = _clean_symbol(ticker)
    try:
        chain = fetch_cboe_chain(symbol)
    except Exception as e:
        return {"error": f"Failed to fetch options data for {symbol}: {e}"}
    try:
        spot, spot_source = _resolve_spot(symbol, chain)
    except Exception as e:
        return {"error": f"Spot price unavailable for {symbol}: {e}"}

    d, exps, diag = parse_chain(chain["options"], symbol, _now_et())
    if d is None:
        return {"error": f"No unexpired option contracts found for {symbol}."}

    sel, label, note = _select_expirations(d, exps, expiration, max_dte)
    if sel.empty:
        return {"error": "No contracts match the selected expiration / max_dte filter."}

    res = compute_gex(sel, spot, r, q)
    if res is None:
        return {"error": "No contracts with both open interest and a usable IV for this selection."}
    u, by = res["u"], res["by"]

    # strike list for charts: +/-15% of spot
    table = by.reset_index().rename(columns={"gex": "net_gex"})
    window = table[(table["K"] >= 0.85 * spot) & (table["K"] <= 1.15 * spot)].sort_values("K")
    strikes = [
        {
            "strike": float(row.K),
            "call_gex": round(float(row.call_gex), 2),
            "put_gex": round(float(row.put_gex), 2),
            "net_gex": round(float(row.net_gex), 2),
            "call_oi": int(row.call_oi),
            "put_oi": int(row.put_oi),
        }
        for row in window.itertuples()
    ]

    call_wall = put_wall = None
    if not window.empty:
        if window["call_gex"].max() > 0:
            call_wall = float(window.loc[window["call_gex"].idxmax(), "K"])
        if window["put_gex"].min() < 0:
            put_wall = float(window.loc[window["put_gex"].idxmin(), "K"])
    top = window.reindex(window["net_gex"].abs().sort_values(ascending=False).index).head(5)
    top_strikes = [{"strike": float(x.K), "net_gex": round(float(x.net_gex), 2)} for x in top.itertuples()]

    grid, vals = gamma_profile(u, spot, r, q)
    flips = find_flips(grid, vals, spot)
    gamma_flip = round(flips[0], 2) if flips else None
    profile = [{"spot": round(float(s), 2), "net_gex": round(float(v), 2)} for s, v in zip(grid[::2], vals[::2])]

    # sanity check: our model gamma vs CBOE's own gamma near the money
    cboe_check = None
    chk = u[(u["gamma_cboe"] > 0) & (u["K"] / spot).between(0.9, 1.1)]
    if len(chk) >= 5:
        mg = bs_gamma_vec(spot, chk["K"].to_numpy(), chk["T"].to_numpy(), r, q, chk["iv"].to_numpy())
        cboe_check = round(float(np.median(mg / chk["gamma_cboe"].to_numpy())), 3)

    total = res["total"]
    wma_200 = _get_wma200(symbol)

    def dist(level):
        return None if level is None else round((level - spot) / spot * 100.0, 2)

    return {
        "ticker": symbol,
        "selected_expiration": label,
        "expiration_note": note,
        "expirations": exps,
        "spot_price": round(spot, 2),
        "spot_source": spot_source,
        "wma_200": round(wma_200, 2) if wma_200 is not None else None,
        "total_net_gex": round(total, 2),
        "regime": "positive" if total > 0 else "negative",
        "regime_note": (
            "Positive net gamma: dealer hedging tends to dampen moves."
            if total > 0 else
            "Negative net gamma: dealer hedging tends to amplify moves."
        ),
        "call_wall": call_wall,
        "put_wall": put_wall,
        "call_wall_dist_pct": dist(call_wall),
        "put_wall_dist_pct": dist(put_wall),
        "gamma_flip": gamma_flip,
        "gamma_flip_dist_pct": dist(gamma_flip),
        "gamma_flip_candidates": [round(f, 2) for f in flips[:5]],
        "flip_note": None if flips else "No zero-gamma crossing within +/-20% of spot.",
        "top_strikes": top_strikes,
        "strikes": strikes,
        "gex_profile": profile,
        "data_timestamp": chain.get("timestamp"),
        "units": "$ millions of gamma exposure per 1% move in the underlying",
        "assumptions": [
            "Dealers assumed long calls and short puts (an assumption, not observed positioning)",
            "Open interest is as of the prior close; quotes are delayed about 15 minutes",
            "IV is held fixed when re-pricing at other spot levels (gamma profile and flip)",
            "Expiry measured to 4:00pm ET on expiration day",
        ],
        "diagnostics": {
            **diag,
            "contracts_in_selection": int(len(sel)),
            "contracts_used": int(len(u)),
            "oi_used_pct": round(res["oi_used"] / res["oi_total"] * 100.0, 1) if res["oi_total"] else None,
            "model_vs_cboe_gamma_ratio": cboe_check,
            "risk_free": r,
            "dividend_yield": q,
        },
    }


@app.get("/api/gex-heatmap")
def get_gex_heatmap(
    ticker: str = "SPY",
    mode: str = "opex",
    n: int = 10,
    range_pct: float = 0.20,
    units: str = "musd",
    near_days: float = 45.0,
    r: float = GEX_RISK_FREE,
    q: float = 0.0,
):
    """
    mode       'opex'  = monthly / quarterly expirations (default)
               'next'  = the next n expirations including weeklies
               'mixed' = every expiration within near_days, then monthlies/quarterlies beyond, up to n
    n          number of expiration columns (1-24)
    range_pct  strike range around spot, as a fraction (0.20 = +/-20%)
    units      'musd' = $ millions per 1% move (default, unchanged) or 'usd' = whole dollars per 1% move
    near_days  window for the weekly-heavy part of mode=mixed
    """
    symbol = _clean_symbol(ticker)
    n = min(max(int(n), 1), 24)
    range_pct = min(max(float(range_pct), 0.02), 0.5)
    use_usd = (units or "").lower() == "usd"
    scale, nd = (1e6, 2) if use_usd else (1.0, 1)

    try:
        chain = fetch_cboe_chain(symbol)
    except Exception as e:
        return {"error": f"Failed to fetch options data for {symbol}: {e}"}
    try:
        spot, spot_source = _resolve_spot(symbol, chain)
    except Exception as e:
        return {"error": f"Spot price unavailable for {symbol}: {e}"}

    d, exps, diag = parse_chain(chain["options"], symbol, _now_et())
    if d is None:
        return {"error": f"No unexpired option contracts found for {symbol}."}

    dte_map = d.drop_duplicates("exp").set_index("exp")["dte"]
    exp_set = set(exps)
    if mode == "next":
        chosen = exps[:n]
    elif mode == "mixed":
        near = [e for e in exps if dte_map[e] <= near_days]
        far = [e for e in exps if dte_map[e] > near_days and _is_opex_exp(e, exp_set)]
        chosen = (near + far)[:n]
    else:
        mode = "opex"
        chosen = [e for e in exps if _is_opex_exp(e, exp_set)][:n]
    if not chosen:
        return {"error": f"No {mode} expirations found for {symbol}."}

    sel = d[
        d["exp"].isin(chosen)
        & (d["K"] >= spot * (1 - range_pct))
        & (d["K"] <= spot * (1 + range_pct))
    ]
    res = compute_gex(sel, spot, r, q) if not sel.empty else None
    if res is None:
        return {"error": "No contracts with open interest and a usable IV for these expirations."}

    piv = res["u"].groupby(["K", "exp"])["gex"].sum().unstack("exp").reindex(columns=chosen).fillna(0.0)
    strikes_desc = sorted(piv.index.tolist(), reverse=True)
    nearest = min(strikes_desc, key=lambda k: abs(k - spot))
    row_totals = piv.sum(axis=1)

    grid_data = []
    for K in strikes_desc:
        row = {"strike": float(K), "is_spot": bool(K == nearest)}
        for e in chosen:
            row[e] = round(float(piv.loc[K, e]) * scale, nd)
        row["total"] = round(float(row_totals.loc[K]) * scale, nd)
        grid_data.append(row)

    # summary strip: totals / flip use ALL expirations (same basis as /api/gex); the
    # max +/- strikes and largest cell describe what is displayed in the grid
    full = compute_gex(d, spot, r, q)
    total_all = gamma_flip = gamma_slope = None
    if full is not None:
        total_all = full["total"] * scale
        grid, vals = gamma_profile(full["u"], spot, r, q)
        flips = find_flips(grid, vals, spot)
        gamma_flip = round(flips[0], 2) if flips else None
        gamma_slope = float(np.interp(spot, grid, np.gradient(vals, grid))) * scale   # change in GEX per $1 of spot

    pos_k = row_totals.idxmax() if row_totals.max() > 0 else None
    neg_k = row_totals.idxmin() if row_totals.min() < 0 else None
    cell_k, cell_e = piv.abs().stack().idxmax()
    price_change = chain.get("price_change")
    prev = spot - price_change if price_change is not None else None

    return {
        "ticker": symbol,
        "spot_price": round(spot, 2),
        "spot_source": spot_source,
        "price_change": round(price_change, 2) if price_change is not None else None,
        "price_change_pct": round(price_change / prev * 100.0, 2) if prev else None,
        "mode": mode,
        "units": "USD per 1% move" if use_usd else "$ millions per 1% move",
        "expirations": chosen,
        "dte": {e: round(float(dte_map[e]), 1) for e in chosen},
        "column_totals": {e: round(float(piv[e].sum()) * scale, nd) for e in chosen},
        "summary": {
            "total_net_gex": None if total_all is None else round(total_all, nd),
            "displayed_net_gex": round(float(piv.to_numpy().sum()) * scale, nd),
            "gamma_flip": gamma_flip,
            "gamma_slope": None if gamma_slope is None else round(gamma_slope, nd),
            "max_pos_gex": None if pos_k is None else {"strike": float(pos_k), "value": round(float(row_totals.loc[pos_k]) * scale, nd)},
            "max_neg_gex": None if neg_k is None else {"strike": float(neg_k), "value": round(float(row_totals.loc[neg_k]) * scale, nd)},
            "largest_cell": {"strike": float(cell_k), "expiration": cell_e, "value": round(float(piv.loc[cell_k, cell_e]) * scale, nd)},
        },
        "matrix": grid_data,
        "data_timestamp": chain.get("timestamp"),
        "diagnostics": {
            **diag,
            "contracts_used": int(len(res["u"])),
            "oi_used_pct": round(res["oi_used"] / res["oi_total"] * 100.0, 1) if res["oi_total"] else None,
        },
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
