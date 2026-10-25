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

        # توسيع النطاق السعري قليلاً لضمان عدم إرجاع 0 نتائج (تحت 15$)
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

        # تيسير شرط حجم التداول قليلاً لتجنب صفر نتائج
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
        e20,e50,e200 = _ema(c["Close"],20),_ema(c["Close"],50),_ema(c["Close"],200)
        rv = _rsi(c["Close"]); ml,sg,mh = _macd(c["Close"])
        adv = _adx(c["High"],c["Low"],c["Close"])
        sk,sd = _stoch(c["High"],c["Low"],c["Close"])
        va = _sma(c["Volume"],20); av = _atr(c["High"],c["Low"],c["Close"])

        avg_vol = float(va.iloc[-1]) if not pd.isna(va.iloc[-1]) else 0
        if not is_crypto and price_tier == "low" and avg_vol < 300000: pass

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
        if pd.isna(adx_v) or adx_v != adx_v: adx_v = 0
        vr = round(vr_now, 1)

        e100 = float(_ema(c["Close"],100).iloc[last])
        ad = price > float(e20.iloc[last])
        aw = price > e100
        if score>=14 and ad and aw: trend="استثمار"
        elif score>=11 and ad: trend="سوينج"
        elif ad: trend="مضاربة"
        else: trend="ضعيف"

        if score>=14 and rr>=1.2: verdict="BUY"
        elif score>=10 and rr>=0.9 and ad: verdict="BUY_COND"
        elif score>=7: verdict="WATCH"
        else: verdict="AVOID"

        confidence = min(100, round(score/20*80 + min(rr,3)/3*20))
        prev = float(df["Close"].iloc[last-1])
        chg = round((price-prev)/prev*100, 2)

        daily_move = atr_v if atr_v > 0 else price*0.015
        high20 = float(c["High"].iloc[-20:].max())
        dist = (high20-price) if high20>price else abs(t1-price)
        raw_days = max(1, round(dist/daily_move))
        est_days = max(1, round(raw_days))
        eta = f"{est_days}-{est_days+2} أيام"

        return {
            "code":code, "name":name, "is_crypto":is_crypto,
            "price":round(price,2), "price_sar":round(price*USD_TO_SAR,2),
            "live":False, "score":score, "verdict":verdict,
            "trend":trend, "confidence":confidence,
            "lb":lb, "t1":t1, "t2":t2, "sl":sl, "rr":rr, "ptp":ptp, "psl":psl,
            "rsi":rsi_v, "adx":adx_v, "vr":vr, "chg":chg,
            "eta":eta, "price_tier":price_tier, "is_sa": False,
        }
    except Exception:
        return None

def analyze_symbol_sa(df, code, name, price):
    try:
        c = df.copy()
        e20=_ema(c["Close"],20); e50=_ema(c["Close"],50); e200=_ema(c["Close"],200)
        rv=_rsi(c["Close"]); ml,sg,_=_macd(c["Close"])
        rsi=float(rv.iloc[-1]); macd_val=float(ml.iloc[-1]); macd_sig=float(sg.iloc[-1])
        ema20=float(e20.iloc[-1]); ema50=float(e50.iloc[-1]); ema200=float(e200.iloc[-1])
        vol_now=float(c["Volume"].iloc[-1]); vol_avg=float(c["Volume"].rolling(20).mean().iloc[-1]) or 1
        vr=round(vol_now/vol_avg,2)
        score=5
        if price>ema20: score+=3
        if price>ema50: score+=3
        if 35<rsi<75: score+=2
        if macd_val>macd_sig: score+=3
        if vr>1.0: score+=2
        
        if score>=13: verdict="BUY"
        elif score>=9: verdict="BUY_COND"
        elif score>=6: verdict="WATCH"
        else: verdict="AVOID"
        
        confidence=min(99,int(score/20*100))
        atr=float(c["High"].sub(c["Low"]).rolling(14).mean().iloc[-1])
        if pd.isna(atr): atr = price * 0.02
        t1=round(price+atr*1.5,2); t2=round(price+atr*3.0,2); sl=round(price-atr*1.0,2)
        rr=round((t1-price)/(price-sl),2) if price>sl else 0
        ptp=round((t1-price)/price*100,1) if price>0 else 0
        psl=round((price-sl)/price*100,1) if price>0 else 0
        trend="صاعد" if price>ema20 else "محايد"
        eta="2-4 أيام"
        return {
            "code":code,"name":name,"price":round(price,2),"currency":"SAR",
            "score":score,"confidence":confidence,"verdict":verdict,"rr":rr,
            "t1":t1,"t2":t2,"sl":sl,"lb":round(price*0.99,2),
            "ptp":ptp,"psl":psl,"rsi":round(rsi,1),"adx":20,
            "trend":trend,"eta":eta,"vr":vr,
            "is_crypto":False,"is_sa":True,"live":False,
        }
    except: return None

# ════════════════════════════════════════════════════════════
# حالة المسح والمفضلة
# ════════════════════════════════════════════════════════════
scan_state = {
    "us":       {"data": [], "last": None, "status": "idle", "progress": 0, "scanned": 0},
    "sa":       {"data": [], "last": None, "status": "idle", "progress": 0, "scanned": 0},
    "crypto":   {"data": [], "last": None, "status": "idle", "progress": 0, "scanned": 0},
    "volatile": {"data": [], "last": None, "status": "idle", "progress": 0, "scanned": 0},
}
FAV_FILE = os.path.join(os.getcwd(), "public_favorites.json")

def load_favs():
    if os.path.exists(FAV_FILE):
        try:
            with open(FAV_FILE) as f: return json.load(f)
        except: return {"us": [], "sa": [], "crypto": [], "volatile": []}
    return {"us": [], "sa": [], "crypto": [], "volatile": []}

def save_favs(d):
    with open(FAV_FILE, "w") as f: json.dump(d, f, ensure_ascii=False, indent=2)

SCAN_HISTORY_FILE = os.path.join(os.getcwd(), "public_scan_history.json")
history_lock = threading.Lock()

def load_scan_history():
    with history_lock:
        if os.path.exists(SCAN_HISTORY_FILE):
            try:
                with open(SCAN_HISTORY_FILE) as f: return json.load(f)
            except: return []
        return []

def save_scan_history(d):
    with history_lock:
        with open(SCAN_HISTORY_FILE, "w") as f: json.dump(d, f, ensure_ascii=False, indent=2)

def record_scan_history(market, picks):
    if not picks: return
    try:
        hist = load_scan_history()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        for s in picks:
            if not s.get("code"): continue
            hist.append({
                "ts": ts, "market": market, "code": s.get("code"), "name": s.get("name"),
                "verdict": s.get("verdict"), "score": s.get("score"),
                "entry": s.get("price"), "t1": s.get("t1"), "t2": s.get("t2"), "sl": s.get("sl"),
            })
        hist = hist[-1200:]
        save_scan_history(hist)
    except Exception:
        pass

def _current_price_for(market, code):
    try:
        if market == "sa":
            p = sahmk_price(code)
            if p: return float(p)
            try:
                df = yf.Ticker(code + ".SR").history(period="5d")
                if not df.empty: return float(df["Close"].iloc[-1])
            except Exception: pass
            return None
        else:
            df = get_history(code, "5d", "1d")
            if not df.empty: return float(df["Close"].iloc[-1])
            return None
    except Exception: return None

def compute_scan_history_status(market=None, days=10):
    hist = load_scan_history()
    cutoff = datetime.now() - timedelta(days=days)
    rows = []
    price_cache = {}
    for rec in hist:
        if market and rec.get("market") != market: continue
        try: rec_dt = datetime.strptime(rec["ts"], "%Y-%m-%d %H:%M")
        except Exception: continue
        if rec_dt < cutoff: continue
        key = (rec.get("market"), rec.get("code"))
        if key not in price_cache:
            price_cache[key] = _current_price_for(rec.get("market"), rec.get("code"))
        cur = price_cache[key]
        row = dict(rec)
        row["current"] = cur
        row["days_elapsed"] = (datetime.now() - rec_dt).days
        if cur is None:
            row["status"] = "unknown"; row["pnl_pct"] = None
        else:
            entry = rec.get("entry") or 0
            row["pnl_pct"] = round((cur - entry) / entry * 100, 2) if entry else 0
            t1, t2, sl = rec.get("t1"), rec.get("t2"), rec.get("sl")
            if t2 and cur >= t2: row["status"] = "hit_t2"
            elif t1 and cur >= t1: row["status"] = "hit_t1"
            elif sl and cur <= sl: row["status"] = "hit_sl"
            else: row["status"] = "open"
        rows.append(row)
    rows.sort(key=lambda r: r["ts"], reverse=True)
    return rows

# ════════════════════════════════════════════════════════════
# تشغيل عمليات المسح
# ════════════════════════════════════════════════════════════
def run_scan(market="us"):
    scan_state[market]["status"] = "scanning"; scan_state[market]["progress"] = 0
    stocks = US_STOCKS if market=="us" else CRYPTO
    is_crypto = (market=="crypto")
    items = list(stocks.items())
    if market=="us":
        import random
        random.seed(int(time.time())//1800)
        random.shuffle(items)
        items = items[:40]
    total = len(items); results = []; lock = threading.Lock(); done = [0]

    def scan_one(code, name):
        r = analyze_symbol(code, name, is_crypto)
        with lock:
            done[0] += 1; scan_state[market]["progress"] = round(done[0]/total*100)
            if r: results.append(r)

    with ThreadPoolExecutor(max_workers=2) as ex:
