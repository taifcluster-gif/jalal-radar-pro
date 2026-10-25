# -*- coding: utf-8 -*-
"""
jalal_scanner_public.py
════════════════════════════════════════════════════════════
نسخة الماسح العامة — النسخة المباشرة والفورية (Direct Synchronous Scan).

الاعتماديات (requirements.txt): flask, yfinance, pandas, numpy
"""
import os, json, time, csv, io
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
# تحليل الخشاش / الزخم العالي
# ════════════════════════════════════════════════════════════
def analyze_high_volatility(code, name, is_crypto=False):
    try:
        df = get_history(code, "60d", "1d")
        if df.empty or len(df) < 15: return None
        
        price = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2])
        chg = round((price - prev_close) / prev_close * 100, 2)

        if not is_crypto and price > 15.0: return None

        c = df.copy()
        rv = _rsi(c["Close"], 14)
        va = _sma(c["Volume"], 20)
        av = _atr(c["High"], c["Low"], c["Close"], 14)

        last = -1
        vol_now = float(c["Volume"].iloc[last])
        vol_avg = float(va.iloc[last]) if float(va.iloc[last]) > 0 else 1.0
        vr_now = vol_now / vol_avg

        if vr_now < 1.1: return None
        rsi_v = float(rv.iloc[last])
        
        score = 5
        if vr_now >= 1.5: score += 4
        if 30 <= rsi_v <= 90: score += 4
        if chg > 0.5: score += 4

        atr_v = float(av.iloc[last]) if not pd.isna(float(av.iloc[-1])) else price * 0.02
        t1 = round(price + atr_v * 1.5, 2)
        t2 = round(price + atr_v * 3.0, 2)
        sl = round(price - atr_v * 1.0, 2)
        rr = round((t1 - price) / max(price - sl, 0.01), 2)
        ptp = round((t1 - price) / price * 100, 2)
        psl = round((price - sl) / price * 100, 2)

        verdict = "HIGH_MOMENTUM" if score >= 11 else "VOLATILE_WATCH"

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
# تحليل JRF القياسي
# ════════════════════════════════════════════════════════════
def analyze_symbol(code, name, is_crypto=False):
    try:
        df = get_history(code, "2y", "1d")
        if df.empty or len(df) < 50: return None
        price = float(df["Close"].iloc[-1])

        c = df.copy()
        e20, e50, e200 = _ema(c["Close"], 20), _ema(c["Close"], 50), _ema(c["Close"], 200)
        rv = _rsi(c["Close"])
        ml, sg, mh = _macd(c["Close"])
        adv = _adx(c["High"], c["Low"], c["Close"])
        sk, sd = _stoch(c["High"], c["Low"], c["Close"])
        va = _sma(c["Volume"], 20)
        av = _atr(c["High"], c["Low"], c["Close"])

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

        rsi_v = round(float(rv.iloc[last]), 1)
        adx_v = round(float(adv.iloc[last]), 1)
        if pd.isna(adx_v) or adx_v != adx_v: adx_v = 0
        vr = round(vr_now, 1)

        e100 = float(_ema(c["Close"],100).iloc[last])
        ad = price > float(e20.iloc[last])
        aw = price > e100
        if score>=14 and ad and aw: trend="استثمار"
        elif score>=11 and ad: trend="سوينج"
        elif ad: trend="مضاربة"
        else: trend="ضعيف"

        if score>=13 and rr>=1.1: verdict="BUY"
        elif score>=9 and rr>=0.8: verdict="BUY_COND"
        elif score>=6: verdict="WATCH"
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
            "eta":eta, "price_tier":"normal", "is_sa": False,
        }
    except Exception:
        return None

def analyze_symbol_sa(df, code, name, price):
    try:
        c = df.copy()
        e20=_ema(c["Close"],20); e50=_ema(c["Close"],50)
        rv=_rsi(c["Close"]); ml,sg,_=_macd(c["Close"])
        rsi=float(rv.iloc[-1]); macd_val=float(ml.iloc[-1]); macd_sig=float(sg.iloc[-1])
        ema20=float(e20.iloc[-1]); ema50=float(e50.iloc[-1])
        vol_now=float(c["Volume"].iloc[-1]); vol_avg=float(c["Volume"].rolling(20).mean().iloc[-1]) or 1
        vr=round(vol_now/vol_avg,2)
        score=5
        if price>ema20: score+=3
        if price>ema50: score+=3
        if 30<rsi<80: score+=2
        if macd_val>macd_sig: score+=3
        if vr>0.9: score+=2
        
        if score>=12: verdict="BUY"
        elif score>=8: verdict="BUY_COND"
        elif score>=5: verdict="WATCH"
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

FAV_FILE = os.path.join(os.getcwd(), "public_favorites.json")
def load_favs():
    if os.path.exists(FAV_FILE):
        try:
            with open(FAV_FILE) as f: return json.load(f)
        except: return {"us": [], "sa": [], "crypto": []}
    return {"us": [], "sa": [], "crypto": []}

def save_favs(d):
    with open(FAV_FILE, "w") as f: json.dump(d, f, ensure_ascii=False, indent=2)

SCAN_HISTORY_FILE = os.path.join(os.getcwd(), "public_scan_history.json")
history_lock = threading.Lock() if 'threading' in globals() else None

def load_scan_history():
    if os.path.exists(SCAN_HISTORY_FILE):
        try:
            with open(SCAN_HISTORY_FILE) as f: return json.load(f)
        except: return []
    return []

def save_scan_history(d):
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
        save_scan_history(hist[-1200:])
    except Exception: pass

def _current_price_for(market, code):
    try:
        if market == "sa":
            p = sahmk_price(code)
            if p: return float(p)
            df = yf.Ticker(code + ".SR").history(period="5d")
            if not df.empty: return float(df["Close"].iloc[-1])
        else:
            df = get_history(code, "5d", "1d")
            if not df.empty: return float(df["Close"].iloc[-1])
    except Exception: pass
    return None

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

# مسح مباشر وفوري (Synchronous Scan)
def execute_sync_scan(market="us", volatile_only=False):
    stocks = US_STOCKS if market=="us" else (SA_STOCKS if market=="sa" else CRYPTO)
    is_crypto = (market=="crypto")
    items = list(stocks.items())
    if market=="us" and not volatile_only:
        import random
        random.seed(int(time.time())//1800)
        random.shuffle(items)
        items = items[:25] # عدد مناسب جداً لسرعة الاستجابة المباشرة

    results = []
    def scan_one(code, name):
        try:
            if volatile_only and market != "sa":
                return analyze_high_volatility(code, name, is_crypto)
            elif market == "sa":
                live_price = sahmk_price(code); df = sahmk_history(code)
                if df is None or len(df) < 50:
                    try:
                        ticker = yf.Ticker(code + ".SR")
                        df = ticker.history(period="2y", interval="1d")
                        df.reset_index(inplace=True)
                    except: pass
                if df is not None and not df.empty and len(df) >= 50:
                    price = live_price if live_price and live_price > 0 else float(df["Close"].iloc[-1])
                    if volatile_only:
                        if price <= 25.0:
                            res = analyze_symbol_sa(df, code, name, price)
                            if res and res["vr"] >= 1.1: return res
                    else:
                        return analyze_symbol_sa(df, code, name, price)
            else:
                return analyze_symbol(code, name, is_crypto)
        except: pass
        return None

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(scan_one, c, n) for c, n in items]
        for f in as_completed(futs):
            res = f.result()
            if res: results.append(res)

    results.sort(key=lambda x:(-x["score"],-x["confidence"],-x["rr"]))
    top = results[:8]
    medals = ["🥇","🥈","🥉","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣"]
    for i,s in enumerate(top): s["medal"] = medals[i] if i<len(medals) else str(i+1)
    fav_set = set(load_favs().get(market, []))
    for s in top: s["is_fav"] = s["code"] in fav_set

    scan_data = {
        "data": top,
        "last": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "scanned": len(results),
        "status": "done"
    }
    record_scan_history(market + ("_vol" if volatile_only else ""), top)
    return scan_data

# مسارات Flask
@app.route("/scan", methods=["GET", "POST"])
def scan_route():
    market = request.args.get("market", "us")
    volatile_only = request.args.get("volatile", "false").lower() == "true"
    if market not in ["us", "sa", "crypto"]: market = "us"
    
    # تنفيذ المسح مباشرة وإرجاع النتائج فورا بدون تعليق
    data = execute_sync_scan(market, volatile_only)
    app.config[f"scan_res_{market}"] = data
    return jsonify({"ok": True, **data})

@app.route("/scan_status", methods=["GET"])
def scan_status():
    return jsonify({"status": "done", "progress": 100, "scanned": 25})

@app.route("/results", methods=["GET"])
def results_route():
    market = request.args.get("market", "us")
    if market not in ["us", "sa", "crypto"]: market = "us"
    data = app.config.get(f"scan_res_{market}", {"data": [], "last": None, "scanned": 0, "status": "done"})
    return jsonify(sanitize_nan(data))

@app.route("/analyze_one", methods=["POST"])
def analyze_one():
    d = request.get_json() or {}
    code = d.get("code","").strip().upper()
    is_crypto = d.get("is_crypto", False)
    market = d.get("market","us")
    if not code: return jsonify({"ok":False,"msg":"أدخل رمز"})
    if market=="sa" or code in SA_STOCKS:
        name = SA_STOCKS.get(code, code)
        live_price=sahmk_price(code); df=sahmk_history(code)
        if df is None or len(df)<50:
            try:
                ticker=yf.Ticker(code+".SR")
                df=ticker.history(period="2y",interval="1d"); df.reset_index(inplace=True)
            except: df=None
        if df is None or df.empty or len(df)<50:
            return jsonify({"ok":False,"msg":f"تعذّر جلب بيانات {code}"})
        price=live_price if live_price and live_price>0 else float(df["Close"].iloc[-1])
        r = analyze_symbol_sa(df, code, name, price)
    else:
        name = US_STOCKS.get(code) or CRYPTO.get(code) or code
        r = analyze_symbol(code, name, is_crypto)
    if not r: return jsonify({"ok":False,"msg":f"تعذّر تحليل {code}"})
    return jsonify(sanitize_nan({"ok":True,"result":r}))

@app.route("/api/favorites", methods=["GET"])
def get_favorites(): return jsonify(load_favs())

@app.route("/api/favorites/toggle", methods=["POST"])
def toggle_favorite():
    d = request.get_json() or {}
    code = d.get("code","").strip().upper()
    market = d.get("market","us")
    favs = load_favs(); lst = favs.get(market, [])
    if code in lst: lst.remove(code); added=False
    else: lst.append(code); added=True
    favs[market] = lst; save_favs(favs)
    return jsonify({"ok":True,"added":added})

@app.route("/api/favorites/analyze", methods=["GET"])
def analyze_favorites():
    market = request.args.get("market","us")
    favs = load_favs().get(market, [])
    if not favs: return jsonify({"data":[]})
    out = []
    if market == "sa":
        for code in favs:
            name = SA_STOCKS.get(code, code)
            try:
                live_price=sahmk_price(code); df=sahmk_history(code)
                if df is None or len(df)<50:
                    ticker=yf.Ticker(code+".SR")
                    df=ticker.history(period="2y",interval="1d"); df.reset_index(inplace=True)
                    if df.empty or len(df)<50: continue
                price=live_price if live_price and live_price>0 else float(df["Close"].iloc[-1])
                r = analyze_symbol_sa(df, code, name, price)
                if r: r["is_fav"]=True; out.append(r)
            except: continue
    else:
        is_crypto = (market=="crypto")
        stocks = US_STOCKS if market=="us" else CRYPTO
        for code in favs:
            name = stocks.get(code, code)
            r = analyze_symbol(code, name, is_crypto)
            if r: r["is_fav"]=True; out.append(r)
    out.sort(key=lambda x:(-x["score"],-x["confidence"]))
    return jsonify(sanitize_nan({"data":out}))

@app.route("/api/scan_history", methods=["GET"])
def api_scan_history():
    market = request.args.get("market", "").strip()
    try: days = int(request.args.get("days", 10))
    except Exception: days = 10
    days = max(1, min(days, 90))
    rows = compute_scan_history_status(market or None, days)
    buy_rows = [r for r in rows if r.get("verdict") in ("BUY", "BUY_COND", "HIGH_MOMENTUM")]
    total = len(buy_rows)
    hit = len([r for r in buy_rows if r.get("status") in ("hit_t1", "hit_t2")])
    stopped = len([r for r in buy_rows if r.get("status") == "hit_sl"])
    summary = {
        "total": total, "hit_target": hit, "hit_stop": stopped,
        "still_open": max(0, total - hit - stopped),
        "win_rate": round(hit / total * 100, 1) if total else 0,
    }
    return jsonify(sanitize_nan({"rows": rows, "summary": summary}))

@app.route("/api/export_scan_history", methods=["GET"])
def export_scan_history():
    market = request.args.get("market", "").strip()
    rows = compute_scan_history_status(market or None, days=365)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["التاريخ","السوق","الرمز","الاسم","الحكم","السكور","سعر الدخول","هدف 1","هدف 2","وقف الخسارة","السعر الحالي","التغير %","الحالة","أيام منذ الترشيح"])
    status_ar = {"hit_t1":"✅ لمس هدف 1","hit_t2":"✅ لمس هدف 2","hit_sl":"❌ لمس الوقف","open":"⏳ لسه مفتوح","unknown":"؟ تعذر الجلب"}
    mkt_ar = {"us":"أمريكي","crypto":"كريبتو","sa":"تاسي"}
    for r in rows:
        writer.writerow([
            r.get("ts",""), mkt_ar.get(r.get("market"), r.get("market","")),
            r.get("code",""), r.get("name",""), r.get("verdict",""), r.get("score",""),
            r.get("entry",""), r.get("t1",""), r.get("t2",""), r.get("sl",""),
            r.get("current",""), r.get("pnl_pct",""),
            status_ar.get(r.get("status"), r.get("status","")), r.get("days_elapsed",""),
        ])
    csv_bytes = output.getvalue().encode("utf-8-sig")
    return Response(
        csv_bytes,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=jalal_scanner_history.csv"}
    )

# الواجهة المباشرة والمضمونة
PAGE = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ماسح جلال رادار — عام</title>
<link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap" rel="stylesheet">
<style>
:root {
  --bg:#0e1015; --surface:#16181d; --surface-2:#1c1f26;
  --ink:#e8e8e6; --ink-2:#a8a8a4; --ink-3:#6b6f76; --line:#2a2d34;
  --brand:#ef9f27; --green:#1d9e75; --green-soft:#10251e;
  --red:#e2504a; --amber:#ef9f27; --amber-soft:#2a2010;
  --r:14px; --r-lg:20px;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:'Tajawal',system-ui,sans-serif; background:var(--bg); color:var(--ink); direction:rtl; line-height:1.6; }
.shell { max-width:760px; margin:0 auto; padding:16px 16px 64px; }
.hdr { text-align:center; padding:18px 0 10px; }
.hdr h1 { font-size:19px; font-weight:800; }
.hdr p { color:var(--ink-3); font-size:12px; margin-top:4px; }
.dropdown-box { margin-bottom:10px; }
.dropdown-box select { width:100%; padding:12px 14px; border-radius:var(--r); border:1px solid var(--line); background:var(--surface); color:var(--ink); font-family:inherit; font-size:14px; font-weight:700; cursor:pointer; }
.filter-box { display:flex; align-items:center; gap:8px; margin-bottom:14px; background:var(--surface); padding:10px 14px; border-radius:var(--r); border:1px solid var(--line); }
.filter-box input[type="checkbox"] { width:18px; height:18px; accent-color:var(--brand); cursor:pointer; }
.filter-box label { font-size:13px; font-weight:700; cursor:pointer; color:var(--ink); }
.q-box { display:flex; gap:8px; margin-bottom:14px; }
.q-box input { flex:1; padding:10px 12px; border-radius:var(--r); border:1px solid var(--line); background:var(--surface); color:var(--ink); font-family:inherit; }
.q-box button { padding:10px 16px; border-radius:var(--r); border:0; background:var(--surface-2); color:var(--ink); font-family:inherit; cursor:pointer; }
.scan-btn { width:100%; padding:13px; border-radius:var(--r); border:0; background:var(--brand); color:#2a1a00; font-size:14px; font-weight:800; cursor:pointer; margin-bottom:8px; box-shadow:0 4px 12px rgba(239,159,39,0.2); transition:0.2s; }
.scan-btn:active { transform:scale(0.98); }
.scan-meta { text-align:center; color:var(--ink-3); font-size:11px; margin-bottom:12px; }
.tally { display:flex; gap:8px; margin-bottom:12px; }
.tally-cell { flex:1; background:var(--surface); border-radius:var(--r); padding:10px; text-align:center; }
.tally-num { font-size:19px; font-weight:800; } .tally-lbl { font-size:11px; color:var(--ink-3); }
.empty { text-align:center; color:var(--ink-3); padding:40px 0; font-size:13px; }
.empty-ico { font-size:32px; margin-bottom:8px; }
.sig { background:var(--surface); border:1px solid var(--line); border-radius:var(--r-lg); margin-top:12px; overflow:hidden; }
.sig-verdict { padding:11px 16px; display:flex; align-items:center; justify-content:space-between; }
.sig-verdict.buy { background:var(--green-soft); border-bottom:1px solid #1a4a3a; }
.sig-verdict.cond { background:var(--amber-soft); border-bottom:1px solid #4a3818; }
.sig-verdict.watch { background:var(--surface-2); border-bottom:1px solid var(--line); }
.verdict-main { display:flex; align-items:center; gap:9px; }
.verdict-label { font-weight:800; font-size:15px; }
.verdict-sub { font-size:11px; color:var(--ink-3); margin-top:1px; }
.verdict-score { text-align:left; } .verdict-score-num { font-size:19px; font-weight:800; }
.verdict-score-lbl { font-size:10px; color:var(--ink-3); }
.sig-body { padding:14px 16px; }
.sig-top { display:flex; align-items:flex-start; gap:11px; }
.sig-medal { font-size:22px; }
.sig-id { flex:1; min-width:0; }
.sig-name { font-weight:700; font-size:14px; display:flex; align-items:center; gap:6px; }
.sig-code { color:var(--ink-3); font-size:12px; font-weight:400; }
.fav-star { cursor:pointer; color:var(--ink-3); font-size:16px; }
.fav-star.on { color:var(--brand); }
.sig-meta { display:flex; gap:8px; margin-top:4px; }
.mb { font-size:11px; color:var(--ink-3); background:var(--surface-2); padding:2px 6px; border-radius:6px; }
.sig-px { text-align:left; }
.sig-px-main { font-size:17px; font-weight:800; }
.sig-px-sar { font-size:11px; color:var(--ink-3); }
.sig-bar { height:3px; background:var(--surface-2); margin:10px 0; border-radius:3px; overflow:hidden; }
.sig-bar-fill { height:100%; background:var(--brand); }
.sig-tags { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin-bottom:10px; }
.tag { font-size:11px; background:var(--surface-2); padding:5px 9px; border-radius:8px; color:var(--ink-2); }
.sig-trade-btn { margin-right:auto; background:var(--green); color:#fff; border:0; border-radius:9px; padding:6px 14px; font-family:inherit; font-size:12px; font-weight:700; text-decoration:none; }
.sig-levels { display:grid; grid-template-columns:repeat(4,1fr); gap:6px; background:var(--surface-2); border-radius:var(--r); padding:8px; }
.lvl { text-align:center; } .lvl-lbl { font-size:10px; color:var(--ink-3); }
.lvl-val { font-size:12px; font-weight:700; margin-top:2px; }
.lvl-pct { font-size:10px; color:var(--ink-3); }
.fav-header { color:var(--brand); font-size:12px; font-weight:700; margin:16px 0 4px; }
.loader { display:none; text-align:center; padding:16px; color:var(--brand); font-weight:700; font-size:13px; background:var(--surface); border-radius:var(--r); margin-bottom:12px; border:1px solid var(--line); }
.loader.show { display:block; }
.footer-note { text-align:center; color:var(--ink-3); font-size:11px; margin-top:24px; padding-top:14px; border-top:1px solid var(--line); }
.hidden-view { display:none; }
.mkt-tabs { display:flex; gap:6px; margin-bottom:16px; }
.mkt-tab { flex:1; padding:10px; border-radius:var(--r); border:1px solid var(--line); background:var(--surface); color:var(--ink-2); text-align:center; font-size:12px; cursor:pointer; }
.mkt-tab.on { background:var(--brand); color:#2a1a00; border-color:var(--brand); font-weight:700; }
</style>
</head>
<body>
<div class="shell">
  <div class="hdr">
    <h1>⚡ ماسح جلال رادار</h1>
    <p>تحليل فني آلي (JRF & Volatility Filter) — عرض فقط، بدون تنفيذ صفقات</p>
  </div>
  
  <div class="dropdown-box">
    <select id="market-select" onchange="onMarketChange()">
      <option value="us">🇺🇸 السوق الأمريكي</option>
      <option value="sa">🇸🇦 السوق السعودي (تاسي)</option>
      <option value="crypto">💰 الكريبتو</option>
    </select>
  </div>

  <div class="filter-box">
    <input type="checkbox" id="volatile-chk">
    <label for="volatile-chk">⚡ تفعيل مسح الخشاش والزخم العالي لهذا السوق</label>
  </div>

  <div class="q-box">
    <input id="q-code" placeholder="الرمز (AAPL أو BTC-USD)">
    <button onclick="analyzeOne()">تحليل فوري</button>
  </div>
  <div id="instant"></div>
  
  <button class="scan-btn" id="scan-btn" onclick="startScan()">🔍 مسح السوق الآن</button>
  <div class="scan-meta" id="scan-meta"></div>
  <div class="loader" id="loader"><div id="loader-sub">⚡ جارٍ فحص السوق وجلب الأسعار الحية، يرجى الانتظار قليلاً...</div></div>

  <div class="mkt-tabs" style="margin-top:4px;">
    <button class="mkt-tab on" id="vt-scan" onclick="setView('scan')">🔍 نتائج المسح</button>
    <button class="mkt-tab" id="vt-perf" onclick="setView('perf')">📜 أداء الترشيحات</button>
  </div>

  <div id="view-scan">
    <div id="tally"></div>
    <div id="fav-section"></div>
    <div id="signals"><div class="empty"><div class="empty-ico">📡</div>اضغط مسح السوق الآن لبدء التحليل</div></div>
  </div>

  <div id="view-perf" class="hidden-view">
    <div style="color:var(--ink-3);font-size:12px;margin-bottom:10px;line-height:1.7;">
      كل مسح تسويه يُحفظ تلقائياً هنا — ارجع بعد كم يوم تشوف هل الترشيح وصل هدفه أو لمس وقفه.
    </div>
    <div class="q-box">
      <select id="ph-days" class="input" style="flex:1;padding:10px 12px;border-radius:var(--r);border:1px solid var(--line);background:var(--surface);color:var(--ink);font-family:inherit;" onchange="loadPerformance()">
        <option value="3">آخر 3 أيام</option>
        <option value="5">آخر 5 أيام</option>
        <option value="10" selected>آخر 10 أيام</option>
        <option value="30">آخر 30 يوم</option>
        <option value="90">آخر 90 يوم</option>
      </select>
      <button onclick="loadPerformance()">🔄 تحديث</button>
    </div>
    <a id="ph-export" href="/api/export_scan_history" style="display:inline-block;margin-bottom:12px;padding:8px 14px;border-radius:10px;background:var(--surface-2);color:var(--ink);font-size:12px;text-decoration:none;">📥 تصدير Excel (CSV)</a>
    <div id="ph-summary"></div>
    <div id="ph-list"><div class="empty"><div class="empty-ico">📜</div>اضغط تحديث لعرض سجل الأداء</div></div>
  </div>

  <div class="footer-note">أداة تحليل فني تعليمية — ليست توصية استثمارية. راجع TradingView وتحقق بنفسك قبل أي قرار.</div>
</div>
<div id="toast" style="position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:var(--surface-2);color:var(--ink);padding:10px 18px;border-radius:10px;font-size:13px;opacity:0;pointer-events:none;transition:.2s;z-index:99;"></div>

<script>
var market = 'us';

function onMarketChange() {
  market = document.getElementById('market-select').value;
  var lbls={'us':'🔍 مسح السوق الأمريكي','sa':'🔍 مسح السوق السعودي (تاسي)','crypto':'🔍 مسح الكريبتو 💰'};
  document.getElementById('scan-btn').innerHTML = lbls[market]||'🔍 مسح';
  loadResults(); loadFavorites();
  if (document.getElementById('vt-perf').classList.contains('on')) loadPerformance();
}

function startScan() {
  var isVolatile = document.getElementById('volatile-chk').checked;
  var btn = document.getElementById('scan-btn');
  btn.disabled = true;
  document.getElementById('loader').classList.add('show');
  document.getElementById('signals').innerHTML = '<div class="empty"><div class="empty-ico">⏳</div>جارٍ فحص الأسهم وحساب المؤشرات الفنية، قد يستغرق ذلك بضع ثوانٍ...</div>';

  fetch('/scan?market='+encodeURIComponent(market)+'&volatile='+(isVolatile?'true':'false'))
    .then(function(r){ return r.json(); })
    .then(function(d){
      btn.disabled = false;
      document.getElementById('loader').classList.remove('show');
      renderResultsData(d);
      loadFavorites();
      toast('تم مسح السوق وعرض النتائج بنجاح! 🎉');
    })
    .catch(function(err){
      btn.disabled = false;
      document.getElementById('loader').classList.remove('show');
      toast('حدث خطأ أثناء المسح، حاول مجدداً');
    });
}

function loadResults() {
  fetch('/results?market='+encodeURIComponent(market))
    .then(function(r){ return r.json(); })
    .then(function(d){
      renderResultsData(d);
    });
}

function renderResultsData(d) {
  var data = d.data||[];
  document.getElementById('scan-meta').textContent = d.last ? ('آخر مسح: '+d.last+' · رُشّح من '+d.scanned) : '';
  var buys = data.filter(function(s){return s.verdict==='BUY' || s.verdict==='HIGH_MOMENTUM';}).length;
  var conds = data.filter(function(s){return s.verdict==='BUY_COND' || s.verdict==='VOLATILE_WATCH';}).length;
  document.getElementById('tally').innerHTML = data.length ?
    '<div class="tally"><div class="tally-cell"><div class="tally-num">'+buys+'</div><div class="tally-lbl">زخم/شراء</div></div>'+
    '<div class="tally-cell"><div class="tally-num">'+conds+'</div><div class="tally-lbl">مراقبة</div></div>'+
    '<div class="tally-cell"><div class="tally-num">'+data.length+'</div><div class="tally-lbl">مرشّح</div></div></div>' : '';
  document.getElementById('signals').innerHTML = data.length ?
    data.map(renderSig).join('') :
    '<div class="empty"><div class="empty-ico">📡</div>لا توجد نتائج مطابقة، اضغط مسح السوق الآن</div>';
}

function renderSig(s) {
  var isMom = s.verdict==='HIGH_MOMENTUM';
  var vClass = (s.verdict==='BUY' || isMom)?'buy':(s.verdict==='BUY_COND'?'cond':'watch');
  var vIco = (s.verdict==='BUY' || isMom)?'🟢':(s.verdict==='BUY_COND'?'🟡':'⏳');
  var vLabel = isMom?'زخم عالي جداً':(s.verdict==='BUY'?'شراء':(s.verdict==='BUY_COND'?'شراء مشروط':'مراقبة'));
  var isSA = s.is_sa === true;
  var cur = isSA ? ' ر.س' : '$'; var curPre = isSA ? '' : '$';
  var ptp = s.ptp||0, psl = s.psl||0, trend = s.trend||'', eta = s.eta||'—';
  var priceDisp = isSA
    ? '<div class="sig-px-main">'+s.price+' ر.س</div>'
    : '<div class="sig-px-main">$'+s.price+'</div>'+(s.price_sar?'<div class="sig-px-sar">'+s.price_sar+' ر.س</div>':'');
  var tvSym = isSA ? ('TADAWUL-'+s.code) : s.code;
  var tvBtn = '<a class="sig-trade-btn" href="https://www.tradingview.com/symbols/'+tvSym+'/" target="_blank" rel="noopener">📈 TradingView</a>';
  return '<div class="sig">'+
    '<div class="sig-verdict '+vClass+'"><div class="verdict-main"><span>'+vIco+'</span>'+
    '<div><div class="verdict-label">'+vLabel+'</div><div class="verdict-sub">ثقة '+s.confidence+'% · '+trend+'</div></div></div>'+
    '<div class="verdict-score"><div class="verdict-score-num">'+s.score+'</div><div class="verdict-score-lbl">JRF/Vol</div></div></div>'+
    '<div class="sig-body"><div class="sig-top"><span class="sig-medal">'+(s.medal||'')+'</span>'+
    '<div class="sig-id"><div class="sig-name">'+s.name+'<span class="sig-code">'+s.code+'</span>'+
    '<span class="fav-star'+(s.is_fav?' on':'')+'" onclick="event.stopPropagation();toggleFav(\''+s.code+'\')">'+(s.is_fav?'★':'☆')+'</span></div>'+
    '<div class="sig-meta"><span class="mb">RSI '+s.rsi+'</span><span class="mb">Volume Surge x'+s.vr+'</span><span class="mb">⏱ '+eta+'</span></div></div>'+
    '<div class="sig-px">'+priceDisp+'</div></div>'+
    '<div class="sig-bar"><div class="sig-bar-fill" style="width:'+s.confidence+'%"></div></div>'+
    '<div class="sig-tags"><span class="tag">R:R <b>'+s.rr+'</b></span>'+
    '<span class="tag">🎯 <b>'+curPre+s.t1+cur+'</b> +'+ptp+'%</span>'+
    '<span class="tag">🛡 <b>'+curPre+s.sl+cur+'</b> -'+psl+'%</span>'+tvBtn+'</div>'+
    '<div class="sig-levels"><div class="lvl"><div class="lvl-lbl">دخول</div><div class="lvl-val">'+curPre+s.lb+cur+'</div></div>'+
    '<div class="lvl"><div class="lvl-lbl">هدف 1</div><div class="lvl-val">'+curPre+s.t1+cur+'</div><div class="lvl-pct">+'+ptp+'%</div></div>'+
    '<div class="lvl"><div class="lvl-lbl">هدف 2</div><div class="lvl-val">'+curPre+s.t2+cur+'</div></div>'+
    '<div class="lvl"><div class="lvl-lbl">وقف</div><div class="lvl-val">'+curPre+s.sl+cur+'</div><div class="lvl-pct">-'+psl+'%</div></div></div></div></div>';
}

function toggleFav(code) {
  fetch('/api/favorites/toggle',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code:code,market:market})})
    .then(function(r){return r.json();}).then(function(d){
      if(d.ok){ toast(d.added?'⭐ أُضيف للمفضلة':'تمت الإزالة'); loadResults(); loadFavorites(); }
    });
}

function loadFavorites() {
  fetch('/api/favorites/analyze?market='+encodeURIComponent(market)).then(function(r){return r.json();}).then(function(d){
    var box = document.getElementById('fav-section');
    if(!d.data || !d.data.length){ box.innerHTML=''; return; }
    box.innerHTML = '<div class="fav-header">⭐ مفضلتك ('+d.data.length+')</div>' + d.data.map(renderSig).join('');
  });
}

function analyzeOne() {
  var code = document.getElementById('q-code').value.trim().toUpperCase();
  if(!code){ toast('أدخل الرمز'); return; }
  var box = document.getElementById('instant');
  box.innerHTML = '<div style="text-align:center;color:var(--ink-3);padding:14px;">جاري تحليل '+code+'…</div>';
  fetch('/analyze_one',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code:code,is_crypto:market==='crypto',market:market})})
    .then(function(r){return r.json();}).then(function(d){
      if(!d.ok){ box.innerHTML='<div style="color:var(--red);padding:10px;">❌ '+(d.msg||'تعذّر')+'</div>'; return; }
      box.innerHTML = renderSig(d.result);
    });
}

function toast(msg) {
  var t = document.getElementById('toast');
  t.textContent = msg; t.style.opacity = '1';
  setTimeout(function(){ t.style.opacity = '0'; }, 2400);
}

function setView(v) {
  document.getElementById('vt-scan').classList.toggle('on', v==='scan');
  document.getElementById('vt-perf').classList.toggle('on', v==='perf');
  document.getElementById('view-scan').classList.toggle('hidden-view', v!=='scan');
  document.getElementById('view-perf').classList.toggle('hidden-view', v!=='perf');
  if (v==='perf') loadPerformance();
}

function loadPerformance() {
  var days = document.getElementById('ph-days').value;
  document.getElementById('ph-export').href = '/api/export_scan_history?market='+encodeURIComponent(market)+'&days='+days;
  document.getElementById('ph-list').innerHTML = '<div class="empty"><div class="empty-ico">⏳</div>جاري الجلب…</div>';
  fetch('/api/scan_history?market='+encodeURIComponent(market)+'&days='+days).then(function(r){return r.json();}).then(function(d){
    var rows = d.rows||[]; var sum = d.summary||{};
    document.getElementById('ph-summary').innerHTML = sum.total ?
      '<div class="tally"><div class="tally-cell"><div class="tally-num">'+sum.hit_target+'</div><div class="tally-lbl">حقق هدف</div></div>'+
      '<div class="tally-cell"><div class="tally-num">'+sum.hit_stop+'</div><div class="tally-lbl">لمس وقف</div></div>'+
      '<div class="tally-cell"><div class="tally-num">'+sum.still_open+'</div><div class="tally-lbl">لسه مفتوح</div></div>'+
      '<div class="tally-cell"><div class="tally-num">'+sum.win_rate+'%</div><div class="tally-lbl">نسبة النجاح</div></div></div>' : '';
    document.getElementById('ph-list').innerHTML = rows.length ?
      rows.map(renderHistCard).join('') :
      '<div class="empty"><div class="empty-ico">📜</div>ما فيه ترشيحات محفوظة بهالفترة بعد لسوق '+market+'</div>';
  });
}

function renderHistCard(r) {
  var isSA = r.market==='sa';
  var cur = isSA ? ' ر.س' : '$'; var curPre = isSA ? '' : '$';
  var statusMap = {
    hit_t1:{ico:'✅',lbl:'لمس هدف 1',cls:'buy'}, hit_t2:{ico:'✅',lbl:'لمس هدف 2',cls:'buy'},
    hit_sl:{ico:'❌',lbl:'لمس الوقف',cls:'watch'}, open:{ico:'⏳',lbl:'لسه مفتوح',cls:'cond'},
    unknown:{ico:'❓',lbl:'تعذر جلب السعر',cls:'cond'},
  };
  var st = statusMap[r.status] || statusMap.unknown;
  var mktLbl = {us:'🇺🇸 أمريكي', sa:'🇸🇦 تاسي', crypto:'💰 كريبتو'}[r.market] || r.market;
  var pnl = (r.pnl_pct===null || r.pnl_pct===undefined) ? '—' : (r.pnl_pct>=0?'+':'')+r.pnl_pct+'%';
  return '<div class="sig">'+
    '<div class="sig-verdict '+st.cls+'"><div class="verdict-main"><span>'+st.ico+'</span>'+
    '<div><div class="verdict-label">'+st.lbl+'</div><div class="verdict-sub">'+mktLbl+' · '+r.ts+' · قبل '+r.days_elapsed+' يوم</div></div></div>'+
    '<div class="verdict-score"><div class="verdict-score-num">'+pnl+'</div><div class="verdict-score-lbl">التغيّر</div></div></div>'+
    '<div class="sig-body"><div class="sig-top"><div class="sig-id"><div class="sig-name">'+(r.name||r.code)+'<span class="sig-code">'+r.code+'</span></div>'+
    '<div class="sig-meta"><span class="mb">'+(r.verdict||'')+'</span><span class="mb">سكور '+(r.score!=null?r.score:'—')+'</span></div></div></div>'+
    '<div class="sig-levels"><div class="lvl"><div class="lvl-lbl">دخول</div><div class="lvl-val">'+curPre+r.entry+cur+'</div></div>'+
    '<div class="lvl"><div class="lvl-lbl">هدف 1</div><div class="lvl-val">'+curPre+r.t1+cur+'</div></div>'+
    '<div class="lvl"><div class="lvl-lbl">هدف 2</div><div class="lvl-val">'+curPre+r.t2+cur+'</div></div>'+
    '<div class="lvl"><div class="lvl-lbl">وقف</div><div class="lvl-val">'+curPre+r.sl+cur+'</div></div></div>'+
    '<div class="sig-tags" style="margin-top:8px;"><span class="tag">السعر الحالي <b>'+(r.current!=null?(curPre+r.current+cur):'—')+'</b></span></div>'+
    '</div></div>';
}

loadResults(); loadFavorites();
</script>
</body>
</html>"""

@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
