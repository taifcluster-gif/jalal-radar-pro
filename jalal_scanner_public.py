# -*- coding: utf-8 -*-
"""
jalal_scanner_public.py
════════════════════════════════════════════════════════════
نسخة الماسح العامة — المحدثة بالقائمة المنسدلة وترتيب الخشاش في النهاية.

الاعتماديات (requirements.txt): flask, yfinance, pandas, numpy
"""
import os, json, time, threading, csv, io
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, request, Response

app = Flask(__name__)
USD_TO_SAR = 3.75

# ════════════════════════════════════════════════════════════
# قوائم الأسهم والعملات
# ════════════════════════════════════════════════════════════
SA_STOCKS = {
    "2222":"أرامكو السعودية","1180":"الأهلي السعودي","1120":"الراجحي",
    "2010":"سابك","7010":"stc","4200":"الدريس",
    "1211":"معادن","1030":"الإستثمار","2380":"بترو رابغ",
    "2350":"كيان السعودية","3010":"أسمنت العربية","1010":"الرياض البنك",
    "2290":"ينساب","4001":"أسواق العثيم","2250":"المجموعة السعودية",
    "6010":"نادك","3020":"أسمنت اليمامة","4030":"البحري",
    "8010":"التعاونية","2020":"سابك للمغذيات الزراعية","7201":"بحر العرب",
    "4140":"صادرات","1832":"صدر","2130":"صدق","4061":"أنعام القابضة"
}
US_STOCKS = {
    "AAPL":"Apple","MSFT":"Microsoft","GOOGL":"Alphabet","META":"Meta",
    "NVDA":"NVIDIA","AMD":"AMD","TSLA":"Tesla","AMZN":"Amazon",
    "INTC":"Intel","QCOM":"Qualcomm","AVGO":"Broadcom","TXN":"Texas Instruments",
    "MU":"Micron","AMAT":"Applied Materials","CRM":"Salesforce","SNOW":"Snowflake",
    "PLTR":"Palantir","SOFI":"SoFi","MARA":"Marathon Digital","RIOT":"Riot Platforms",
    "CLSK":"CleanSpark","COIN":"Coinbase","HOOD":"Robinhood","NIO":"NIO",
    "XPEV":"XPeng","LCID":"Lucid","BABA":"Alibaba","FSLR":"First Solar"
}
CRYPTO = {
    "BTC-USD":"Bitcoin","ETH-USD":"Ethereum","SOL-USD":"Solana",
    "AVAX-USD":"Avalanche","LINK-USD":"Chainlink","DOT-USD":"Polkadot",
    "LTC-USD":"Litecoin","UNI-USD":"Uniswap","AAVE-USD":"Aave",
    "BCH-USD":"Bitcoin Cash","XRP-USD":"Ripple","DOGE-USD":"Dogecoin",
    "SHIB-USD":"Shiba Inu","PEPE-USD":"Pepe","SUI-USD":"Sui",
    "NEAR-USD":"Near Protocol","APT-USD":"Aptos","FET-USD":"Fetch.ai"
}

# ════════════════════════════════════════════════════════════
# المؤشرات الفنية
# ════════════════════════════════════════════════════════════
def _ema(s, p): return s.ewm(span=p, adjust=False).mean()
def _sma(s, p): return s.rolling(p).mean()

def _rsi(s, p=14):
    d = s.diff(); g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))

def _macd(s):
    ml = _ema(s,12) - _ema(s,26); sg = _ema(ml,9); return ml, sg, ml-sg

def _adx(h,l,c,p=14):
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    at = tr.rolling(p).mean()
    up,dn = h.diff(),-l.diff()
    dp = pd.Series(np.where((up>dn)&(up>0),up,0.),index=h.index).rolling(p).mean()
    dm = pd.Series(np.where((dn>up)&(dn>0),dn,0.),index=h.index).rolling(p).mean()
    dip,dim = 100*dp/at,100*dm/at
    return (100*(dip-dim).abs()/(dip+dim).replace(0,np.nan)).rolling(p).mean()

def _atr(h,l,c,p=14):
    return pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1).rolling(p).mean()

def _stoch(h,l,c,k=14,d=3):
    kv = 100*(c-l.rolling(k).min())/(h.rolling(k).max()-l.rolling(k).min()).replace(0,np.nan)
    return kv, kv.rolling(d).mean()

def sanitize_nan(obj):
    if isinstance(obj, dict): return {k: sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list): return [sanitize_nan(v) for v in obj]
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))): return None
    return obj

# ════════════════════════════════════════════════════════════
# جلب البيانات
# ════════════════════════════════════════════════════════════
def get_history(ticker, period="1y", interval="1d", retries=2):
    for attempt in range(retries+1):
        try:
            df = yf.download(ticker, period=period, interval=interval,
                              progress=False, auto_adjust=True, timeout=15)
            if df.empty:
                if attempt < retries: time.sleep(1); continue
                return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        except Exception:
            if attempt < retries: time.sleep(1); continue
            return pd.DataFrame()
    return pd.DataFrame()

SAHMK_BASE = "https://api.sahmk.sa"
SAHMK_KEY = os.environ.get("SAHMK_KEY_PUBLIC", "")

def sahmk_get(path):
    if not SAHMK_KEY: return {"error": "no key"}
    try:
        import urllib.request
        req = urllib.request.Request(SAHMK_BASE+path, headers={"Authorization":f"Bearer {SAHMK_KEY}","Accept":"application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e: return {"error": str(e)}

def sahmk_price(symbol):
    r = sahmk_get(f"/quote/{symbol}/")
    try: return float(r.get("price") or r.get("last_price") or r.get("close") or 0)
    except: return None

def sahmk_history(symbol):
    r = sahmk_get(f"/historical/{symbol}/")
    try:
        if isinstance(r, list) and len(r)>=50:
            df = pd.DataFrame(r); df.columns = [c.capitalize() for c in df.columns]
            if "Date" in df.columns: df["Date"] = pd.to_datetime(df["Date"])
            return df.sort_values("Date").reset_index(drop=True)
    except: pass
    return None

# ════════════════════════════════════════════════════════════
# 1. فلتر التذبذب العالي والزخم (High-Volatility & Momentum / الخشاش السريع)
# ════════════════════════════════════════════════════════════
def analyze_high_volatility(code, name, is_crypto=False):
    try:
        df = get_history(code, "60d", "1d")
        if df.empty or len(df) < 15: return None
        
        price = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2])
        chg = round((price - prev_close) / prev_close * 100, 2)

        if not is_crypto and price > 15.0:
            return None

        c = df.copy()
        e5, e13 = _ema(c["Close"], 5), _ema(c["Close"], 13)
        rv = _rsi(c["Close"], 14)
        va = _sma(c["Volume"], 20)
        av = _atr(c["High"], c["Low"], c["Close"], 14)

        last = -1
        vol_now = float(c["Volume"].iloc[last])
        vol_avg = float(va.iloc[last]) if float(va.iloc[last]) > 0 else 1.0
        vr_now = vol_now / vol_avg

        if vr_now < 1.2:
            return None

        rsi_v = float(rv.iloc[last])
        
        score = 5
        if vr_now >= 2.0: score += 4
        if 40 <= rsi_v <= 85: score += 4
        if chg > 1.0: score += 4

        atr_v = float(av.iloc[last]) if not pd.isna(float(av.iloc[-1])) else price * 0.02
        t1 = round(price + atr_v * 1.5, 2)
        t2 = round(price + atr_v * 3.0, 2)
        sl = round(price - atr_v * 1.0, 2)
        rr = round((t1 - price) / max(price - sl, 0.01), 2)
        ptp = round((t1 - price) / price * 100, 2)
        psl = round((price - sl) / price * 100, 2)

        verdict = "HIGH_MOMENTUM" if score >= 12 else "VOLATILE_WATCH"

        return {
            "code": code, "name": name, "is_crypto": is_crypto,
            "price": round(price, 2), "price_sar": round(price * USD_TO_SAR, 2),
            "score": score, "verdict": verdict, "trend": "خشاش/زخم سريع",
            "confidence": min(99, score * 6),
            "lb": round(price * 0.99, 2), "t1": t1, "t2": t2, "sl": sl,
            "rr": rr, "ptp": ptp, "psl": psl,
            "rsi": round(rsi_v, 1), "adx": 0, "vr": round(vr_now, 1), "chg": chg,
            "eta": "1-2 يوم", "price_tier": "penny" if price < 5 else "low",
            "is_sa": False, "tag": "⚡ High Volatility"
        }
    except Exception:
        return None

# ════════════════════════════════════════════════════════════
# تحليل JRF القياسي — أمريكي / كريبتو
# ════════════════════════════════════════════════════════════
def analyze_symbol(code, name, is_crypto=False):
    try:
        df = get_history(code, "2y", "1d")
        if df.empty or len(df) < 50: return None
        price = float(df["Close"].iloc[-1])

        price_tier = "normal"
        if not is_crypto:
            if price >= 5: price_tier = "normal"
            elif price >= 1: price_tier = "low"
            elif price > 0: price_tier = "penny"
            else: return None

        c = df.copy()
        e20, e50, e200 = _ema(c["Close"], 20), _ema(c["Close"], 50), _ema(c["Close"], 200)
        rv = _rsi(c["Close"])
        ml, sg, mh = _macd(c["Close"])
        adv = _adx(c["High"], c["Low"], c["Close"])
        sk, sd = _stoch(c["High"], c["Low"], c["Close"])
        va = _sma(c["Volume"], 20)
        av = _atr(c["High"], c["Low"], c["Close"])

        avg_vol = float(va.iloc[-1]) if not pd.isna(va.iloc[-1]) else 0
        last = -1
        vr_now = float(c["Volume"].iloc[last]) / float(va.iloc[last]) if float(va.iloc[last]) > 0 else 1.0

        c1 = price > float(e20.iloc[last])
        c2 = float(e20.iloc[last]) > float(e50.iloc[last])
        c3 = price > float(e200.iloc[last])
        c4 = float(ml.iloc[last]) > float(sg.iloc[last])
        c5 = float(mh.iloc[last]) > float(mh.iloc[last-1])
        c6 = 35 <= float(rv.iloc[last]) <= 75
        c7 = float(rv.iloc[last]) > float(rv.iloc[last-1])
        c8 = float(adv.iloc[last]) > 15
        c9 = float(sk.iloc[last]) > 15 and float(sk.iloc[last]) > float(sd.iloc[last])
        c10 = vr_now > 1.0
        c11 = float(adv.iloc[last]) > float(adv.iloc[last-1])
        score = (2*c1+2*c2+2*c3+2*c4+1*c5+2*c6+1*c7+2*c8+2*c9+2*c10+2*c11)

        atr_v = float(av.iloc[-1]) if not pd.isna(float(av.iloc[-1])) else price * 0.02
        lb = round(price * 0.998, 2)
        t1 = round(price + atr_v*2.0, 2)
        t2 = round(price + atr_v*4.0, 2)
        sl = round(price - atr_v*1.0, 2)
        rr = round((t1-price)/max(price-sl,0.01), 2)
        ptp = round((t1-price)/price*100, 2)
        psl = round((price-sl)/price*100, 2)

        rsi_v = round(float(rv.iloc[last]),1)
        adx_v = round(float(adv.iloc[last]),1)
        if pd.isna(adx_v) or
