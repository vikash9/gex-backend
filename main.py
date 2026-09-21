from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
import requests

# 1. INITIALIZE FASTAPI APP (MUST BE BEFORE ANY DECORATORS)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. HELPER FUNCTIONS
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

# 3. ROUTE: EXHAUSTION ENGINE
@app.get("/api/exhaustion")
def get_exhaustion(ticker: str = "GOOGL", timeframe: str = "1d"):
    ticker_symbol = ticker.upper()
    
    tf_mapping = {
        "5m": ("5m", "7d"),
        "15m": ("15m", "14d"),
        "30m": ("30m", "30d"),
        "1h": ("60m", "60d"),
        "4h": ("60m", "120d"),
        "daily": ("1d", "2y"),
        "1d": ("1d", "2y"),
        "weekly": ("1wk", "5y"),
        "1wk": ("1wk", "5y"),
        "monthly": ("1mo", "10y"),
        "1mo": ("1mo", "10y")
    }

    interval, period = tf_mapping.get(timeframe.lower(), ("1d", "2y"))

    try:
        df = yf.download(
            ticker_symbol, 
            period=period, 
            interval=interval, 
            auto_adjust=True, 
            multi_level_index=False, 
            progress=False
        )
        
        if df.empty:
            return {"error": f"No data returned for {ticker_symbol} with interval {interval}"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if timeframe.lower() == "4h":
            df = df.resample('4h').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()

        df = df.dropna()

        # 1. Volume Climax
        vol_threshold = 1.5 if timeframe in ["5m", "15m", "30m"] else 1.8
        df['Vol_MA'] = df['Volume'].rolling(20).mean()
        df['Is_Vol_Climax'] = df['Volume'] > (df['Vol_MA'] * vol_threshold)

        # 2. Bollinger Bands
        df['BB_Mid'] = df['Close'].rolling(20).mean()
        df['BB_Std'] = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2.0)
        df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2.0)

        # 3. MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Hist'] = df['MACD'] - df['Signal']

        # 4. Candlestick Wicks
        body = (df['Close'] - df['Open']).abs()
        candle_range = df['High'] - df['Low']
        upper_wick = df['High'] - df[['Open', 'Close']].max(axis=1)
        lower_wick = df[['Open', 'Close']].min(axis=1) - df['Low']

        is_shooting_star = (upper_wick > body * 1.5) & (lower_wick < candle_range * 0.3)
        is_hammer = (lower_wick > body * 1.5) & (upper_wick < candle_range * 0.3)

        # Confluence Scoring
        buyer_score = (
            df['Is_Vol_Climax'].astype(int) +
            ((df['Close'] >= df['Close'].rolling(5).max()) & (df['Hist'] < df['Hist'].shift(1))).astype(int) +
            is_shooting_star.astype(int) +
            (df['High'] >= df['BB_Upper']).astype(int)
        )

        seller_score = (
            df['Is_Vol_Climax'].astype(int) +
            ((df['Close'] <= df['Close'].rolling(5).min()) & (df['Hist'] > df['Hist'].shift(1))).astype(int) +
            is_hammer.astype(int) +
            (df['Low'] <= df['BB_Lower']).astype(int)
        )

        candles = []
        signals = []

        min_score = 2 if timeframe in ["5m", "15m", "30m"] else 3

        for idx, row in df.iterrows():
            time_str = idx.strftime("%Y-%m-%d %H:%M") if hasattr(idx, 'strftime') else str(idx)
            
            candles.append({
                "time": time_str,
                "open": round(float(row['Open']), 2),
                "high": round(float(row['High']), 2),
                "low": round(float(row['Low']), 2),
                "close": round(float(row['Close']), 2),
                "volume": int(row['Volume']),
                "bb_upper": round(float(row['BB_Upper']), 2) if not np.isnan(row['BB_Upper']) else None,
                "bb_lower": round(float(row['BB_Lower']), 2) if not np.isnan(row['BB_Lower']) else None,
            })

            b_score = int(buyer_score[idx])
            s_score = int(seller_score[idx])

            if b_score >= min_score:
                signals.append({
                    "time": time_str,
                    "type": "BUYER_EXHAUSTION",
                    "price": round(float(row['High']), 2),
                    "score": b_score,
                    "label": f"BUYER EXHAUST ({b_score}/4)"
                })
            elif s_score >= min_score:
                signals.append({
                    "time": time_str,
                    "type": "SELLER_EXHAUSTION",
                    "price": round(float(row['Low']), 2),
                    "score": s_score,
                    "label": f"SELLER EXHAUST ({s_score}/4)"
                })

        return {
            "ticker": ticker_symbol,
            "spot_price": round(float(df['Close'].iloc[-1]), 2),
            "timeframe": timeframe,
            "candles": candles,
            "signals": signals
        }

    except Exception as e:
        return {"error": f"Exhaustion calculation error: {str(e)}"}

# 4. ROUTE: GEX
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

# 5. ROUTE: GEX HEATMAP
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
