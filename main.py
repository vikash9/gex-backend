from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
import requests

app = FastAPI()

# Enable CORS so your Lovable frontend can connect smoothly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom session to mimic a regular Chrome browser and bypass cloud server blocks
def get_custom_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    return session

def calculate_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

@app.get("/api/gex")
def get_gex(ticker: str = "SPY"):
    session = get_custom_session()
    ticker_obj = yf.Ticker(ticker.upper(), session=session)
    
    # 1. Fetch Spot Price & 200 WMA
    hist_weekly = ticker_obj.history(period="5y", interval="1wk")
    if hist_weekly.empty:
        return {"error": f"Ticker '{ticker}' not found or price data blocked."}
        
    spot_price = float(hist_weekly['Close'].iloc[-1])
    hist_weekly['200_WMA'] = hist_weekly['Close'].rolling(window=200).mean()
    wma_200 = float(hist_weekly['200_WMA'].iloc[-1]) if not np.isnan(hist_weekly['200_WMA'].iloc[-1]) else 0.0

    # 2. Options Data Processing
    try:
        expirations = ticker_obj.options
    except Exception as e:
        return {"error": f"Failed to retrieve expirations: {str(e)}"}

    if not expirations:
        return {"error": f"No options available for {ticker}. Yahoo Finance may be rate-limiting cloud IP requests."}
        
    selected_exp = expirations[0] # Near-term expiration
    opt = ticker_obj.option_chain(selected_exp)
    
    calls = opt.calls[['strike', 'openInterest', 'impliedVolatility']].rename(
        columns={'openInterest': 'call_OI', 'impliedVolatility': 'call_IV'}
    )
    puts = opt.puts[['strike', 'openInterest', 'impliedVolatility']].rename(
        columns={'openInterest': 'put_OI', 'impliedVolatility': 'put_IV'}
    )
    df = pd.merge(calls, puts, on='strike', how='outer').fillna(0)

    # 3. GEX Calculations
    exp_date = pd.to_datetime(selected_exp)
    today = pd.to_datetime('today')
    dte = max((exp_date - today).days, 1) / 365.0
    r = 0.045

    strikes_data = []
    for _, row in df.iterrows():
        K = float(row['strike'])
        c_iv = row['call_IV'] if row['call_IV'] > 0 else 0.2
        p_iv = row['put_IV'] if row['put_IV'] > 0 else 0.2

        c_gamma = calculate_gamma(spot_price, K, dte, r, c_iv)
        p_gamma = calculate_gamma(spot_price, K, dte, r, p_iv)

        c_gex = float(c_gamma * row['call_OI'] * 100 * (spot_price**2) * 0.01 / 1e6)
        p_gex = float(-(p_gamma * row['put_OI'] * 100 * (spot_price**2) * 0.01 / 1e6))
        net_gex = c_gex + p_gex

        strikes_data.append({
            "strike": K,
            "call_gex": round(c_gex, 2),
            "put_gex": round(p_gex, 2),
            "net_gex": round(net_gex, 2)
        })

    # Filter strikes +/- 15% around spot
    strikes_filtered = [s for s in strikes_data if spot_price * 0.85 <= s["strike"] <= spot_price * 1.15]
    
    # Key levels
    call_wall = max(strikes_filtered, key=lambda x: x["call_gex"])["strike"] if strikes_filtered else 0
    put_wall = min(strikes_filtered, key=lambda x: x["put_gex"])["strike"] if strikes_filtered else 0
    total_net_gex = sum(s["net_gex"] for s in strikes_filtered)

    return {
        "ticker": ticker.upper(),
        "spot_price": round(spot_price, 2),
        "wma_200": round(wma_200, 2),
        "total_net_gex": round(total_net_gex, 2),
        "call_wall": call_wall,
        "put_wall": put_wall,
        "gamma_flip": spot_price,
        "strikes": strikes_filtered
    }
