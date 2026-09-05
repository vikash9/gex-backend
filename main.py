from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from curl_cffi import requests as curl_requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom curl-cffi session impersonating a real Chrome browser on Cloud hosts
def get_impersonated_session():
    session = curl_requests.Session(impersonate="chrome120")
    return session

def calculate_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

@app.get("/api/gex")
def get_gex(ticker: str = "SPY"):
    ticker_symbol = ticker.upper()
    
    try:
        session = get_impersonated_session()
        ticker_obj = yf.Ticker(ticker_symbol, session=session)
        
        # 1. Fetch Spot Price & 200 WMA
        hist_weekly = ticker_obj.history(period="5y", interval="1wk")
        if hist_weekly.empty:
            return {"error": f"Ticker '{ticker_symbol}' not found or blocked."}
            
        spot_price = float(hist_weekly['Close'].iloc[-1])
        hist_weekly['200_WMA'] = hist_weekly['Close'].rolling(window=200).mean()
        wma_200 = float(hist_weekly['200_WMA'].iloc[-1]) if not np.isnan(hist_weekly['200_WMA'].iloc[-1]) else 0.0

        # 2. Fetch Options Expirations
        expirations = ticker_obj.options
        if not expirations:
            # Fallback retry without custom session if initial impersonation yields empty tuple
            ticker_obj = yf.Ticker(ticker_symbol)
            expirations = ticker_obj.options
            
        if not expirations:
            return {"error": f"No options chain found for {ticker_symbol}."}
            
        selected_exp = expirations[0]
        opt = ticker_obj.option_chain(selected_exp)
        
        calls = opt.calls[['strike', 'openInterest', 'impliedVolatility']].rename(
            columns={'openInterest': 'call_OI', 'impliedVolatility': 'call_IV'}
        )
        puts = opt.puts[['strike', 'openInterest', 'impliedVolatility']].rename(
            columns={'openInterest': 'put_OI', 'impliedVolatility': 'put_IV'}
        )
        df = pd.merge(calls, puts, on='strike', how='outer').fillna(0)

        # 3. GEX Math
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

        strikes_filtered = [s for s in strikes_data if spot_price * 0.85 <= s["strike"] <= spot_price * 1.15]
        
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
    except Exception as e:
        return {"error": f"Server processing error: {str(e)}"}
