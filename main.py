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

# Massive.com / Polygon.io API key
POLYGON_API_KEY = "z5vCInfqDmbgC7vGT4xFSl0XPUTTHbPV"

def calculate_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

@app.get("/api/gex")
def get_gex(ticker: str = "SPY"):
    ticker_symbol = ticker.upper()
    
    # 1. Fetch Spot Price & 200 WMA via yfinance
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

    # 2. Fetch Options Data via Massive.com / Polygon.io API
    url = f"https://api.polygon.io/v3/snapshot/options/{ticker_symbol}?apiKey={POLYGON_API_KEY}&limit=250"
    
    try:
        res = requests.get(url, timeout=10).json()
        results = res.get("results", [])
        
        if not results:
            return {"error": f"No options available for {ticker_symbol} from data provider."}
    except Exception as e:
        return {"error": f"Failed to reach options API provider: {str(e)}"}

    strikes_dict = {}
    r = 0.045
    dte = 7 / 365.0  # Approx 1 week expiration baseline

    for item in results:
        details = item.get("details", {})
        greeks = item.get("greeks", {})
        
        K = details.get("strike_price")
        contract_type = details.get("contract_type") # 'call' or 'put'
        oi = item.get("open_interest", 0) or 0
        iv = greeks.get("implied_volatility", 0.2) or 0.2

        if not K:
            continue

        if K not in strikes_dict:
            strikes_dict[K] = {"call_OI": 0, "put_OI": 0, "call_IV": 0.2, "put_IV": 0.2}

        if contract_type == "call":
            strikes_dict[K]["call_OI"] += oi
            strikes_dict[K]["call_IV"] = iv if iv > 0 else 0.2
        elif contract_type == "put":
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

    # Filter strikes +/- 15% around current spot price
    strikes_filtered = [s for s in strikes_data if spot_price * 0.85 <= s["strike"] <= spot_price * 1.15]
    strikes_filtered.sort(key=lambda x: x["strike"])

    if not strikes_filtered:
        strikes_filtered = strikes_data[:30] # Fallback display if bounds miss

    call_wall = max(strikes_filtered, key=lambda x: x["call_gex"])["strike"] if strikes_filtered else 0
    put_wall = min(strikes_filtered, key=lambda x: x["put_gex"])["strike"] if strikes_filtered else 0
    total_net_gex = sum(s["net_gex"] for s in strikes_filtered)

    return {
        "ticker": ticker_symbol,
        "spot_price": round(spot_price, 2),
        "wma_200": round(wma_200, 2),
        "total_net_gex": round(total_net_gex, 2),
        "call_wall": call_wall,
        "put_wall": put_wall,
        "gamma_flip": spot_price,
        "strikes": strikes_filtered
    }
