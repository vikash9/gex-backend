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
        # 1. FIX MULTIINDEX BUG BY PASSING multi_level_index=False
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

        # Double check column flattening
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 2. FIX 4H RESAMPLING
        if timeframe.lower() == "4h":
            df = df.resample('4h').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()

        # Fill any missing volume/price gaps
        df = df.dropna()

        # 1. Volume Climax (Dynamic threshold: lower for intraday)
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

        # Flexible wick ratios for lower timeframes
        is_shooting_star = (upper_wick > body * 1.5) & (lower_wick < candle_range * 0.3)
        is_hammer = (lower_wick > body * 1.5) & (upper_wick < candle_range * 0.3)

        # Scoring Logic
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

        # Min score required: 2 for intraday (5m/15m), 3 for daily/weekly
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
