from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
import requests

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
    """Check if date is 3rd Friday of the month."""
    if date_obj.weekday() != 4:  # 4 = Friday
        return False
    return 15 <= date_obj.day <= 21

def is_quarterly_opex(date_obj):
    """Check if date is the last business day of March, June, September, or December."""
    if date_obj.month not in [3, 6, 9, 12]:
        return False
    # Get last day of the month
    next_month = date_obj.replace(day=28) + pd.Timedelta(days=4)
    last_day_of_month = next_month - pd.Timedelta(days=next_month.day)
    
    # Adjust for weekend (if month ends on Saturday/Sunday, last business day is Friday)
    if last_day_of_month.weekday() == 5: # Saturday
        last_bus_day = last_day_of_month - pd.Timedelta(days=1)
    elif last_day_of_month.weekday() == 6: # Sunday
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

        # INCLUDE BOTH MONTHLY OPEX AND QUARTERLY OPEX
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

    sorted_expirations = sorted(list(expirations_set))[:10]  # First 10 valid Monthly + Quarterly expirations
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
