#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════
# جلال رادار PRO — Alpaca-Native Trading System
# مؤسس بالكامل على Alpaca للتنفيذ والأسعار اللحظية
# yfinance للتحليل التاريخي فقط
# ════════════════════════════════════════════════════════════
from flask import Flask, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings, threading, os, json, time, csv, io
import urllib.request, urllib.parse
warnings.filterwarnings("ignore")

app = Flask(__name__)

USD_TO_SAR = 3.75

# ════════════════════════════════════════════════════════════
# طبقة Alpaca — مصدر الأسعار اللحظية + التنفيذ
# ════════════════════════════════════════════════════════════
ALPACA_CONFIG_FILE = os.path.join(os.getcwd(), "pro_alpaca.json")

def load_cfg():
    env_key = os.environ.get("ALPACA_KEY", "")
    env_secret = os.environ.get("ALPACA_SECRET", "")
    defaults = {
        "key": env_key, "secret": env_secret,
        "trade_url": "https://paper-api.alpaca.markets",
        "data_url": "https://data.alpaca.markets",
        "enabled": bool(env_key),
        "max_position_usd": 500,   # احتياطي فقط — يُستخدم لو تعذّر جلب رأس المال من Alpaca
        "tier1_pct": 2.0,           # % من رأس المال — مطابقة كاملة
        "tier2_pct": 1.2,           # % من رأس المال — مطابقة جزئية + إشارة قوية
        "tier3_pct": 0.5,           # % من رأس المال — فرصة استثنائية غير مطابقة
        "allow_low_price": True,    # السماح بفئة 1$-5$
        "allow_penny_price": False, # السماح بفئة أقل من 1$ (افتراضي مغلق لحساسيتها)
        "max_daily_trades": 5,
        "daily_loss_pct": 3.0,
        "min_confidence": 70,
        "auto_buy": True,
        "auto_sell": True,
        "markets": ["us"],          # us, crypto
        "buy_order_type": "market", # market = تنفيذ مضمون · limit = سعر محدد
        "tp1_pct": 2.0,             # بيع 40% عند هذا الربح
        "tp2_pct": 4.0,             # بيع 30% عند هذا الربح
        "trail_pct": 1.5,           # وقف متحرك للباقي
    }
    if os.path.exists(ALPACA_CONFIG_FILE):
        try:
            with open(ALPACA_CONFIG_FILE) as f:
                saved = json.load(f)
            if not saved.get("key") and env_key: saved["key"] = env_key
            if not saved.get("secret") and env_secret: saved["secret"] = env_secret
            if env_key and not saved.get("enabled"): saved["enabled"] = True
            return {**defaults, **saved}
        except: pass
    return defaults

def save_cfg(cfg):
    with open(ALPACA_CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
    if cfg.get("key"): os.environ["ALPACA_KEY"] = cfg["key"]
    if cfg.get("secret"): os.environ["ALPACA_SECRET"] = cfg["secret"]

def _req(base, path, method="GET", data=None):
    cfg = load_cfg()
    if not cfg.get("key"): return {"error": "no_keys"}
    url = base.rstrip("/") + path
    headers = {
        "APCA-API-KEY-ID": cfg["key"],
        "APCA-API-SECRET-KEY": cfg["secret"],
        "Content-Type": "application/json",
    }
    try:
        body = json.dumps(data).encode() if data else None
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=6) as r:
            txt = r.read()
            return json.loads(txt) if txt else {}
    except urllib.error.HTTPError as e:
        try: return {"error": json.loads(e.read()).get("message", str(e))}
        except: return {"error": f"HTTP {e.code}"}
    except Exception as e:
        return {"error": str(e)}

# ── حساب ──
def api_account():
    cfg = load_cfg()
    return _req(cfg["trade_url"] + "/v2", "/account")

# ── ساعة السوق (هل مفتوح فعلاً؟) ──
def api_clock():
    cfg = load_cfg()
    return _req(cfg["trade_url"] + "/v2", "/clock")

def market_is_open():
    clock = api_clock()
    if "error" in clock: return False
    return clock.get("is_open", False)

# ── السعر اللحظي من Alpaca (مو yfinance) ──
def api_last_price(symbol, is_crypto=False):
    cfg = load_cfg()
    if is_crypto:
        sym = symbol.replace("-USD", "/USD") if "-USD" in symbol else symbol
        r = _req(cfg["data_url"], f"/v1beta3/crypto/us/latest/trades?symbols={urllib.parse.quote(sym)}")
        try:
            return float(r["trades"][sym]["p"])
        except: return None
    else:
        r = _req(cfg["data_url"], f"/v2/stocks/{symbol}/trades/latest")
        try:
            return float(r["trade"]["p"])
        except: return None

# ── أوامر ──
def api_place_order(symbol, qty, side, otype="market", limit=None, tif="day", is_crypto=False):
    cfg = load_cfg()
    sym = symbol
    if is_crypto and "-USD" in symbol:
        sym = symbol.replace("-USD", "/USD")
    data = {"symbol": sym, "qty": str(qty), "side": side, "type": otype, "time_in_force": tif}
    if limit: data["limit_price"] = str(round(float(limit), 2))
    return _req(cfg["trade_url"] + "/v2", "/orders", "POST", data)

def api_get_order(order_id):
    cfg = load_cfg()
    return _req(cfg["trade_url"] + "/v2", f"/orders/{order_id}")

def api_positions():
    cfg = load_cfg()
    r = _req(cfg["trade_url"] + "/v2", "/positions")
    return r if isinstance(r, list) else []

def api_close_position(symbol):
    cfg = load_cfg()
    sym = symbol.replace("-USD", "/USD") if "-USD" in symbol else symbol
    return _req(cfg["trade_url"] + "/v2", f"/positions/{urllib.parse.quote(sym)}", "DELETE")

def api_open_orders():
    cfg = load_cfg()
    r = _req(cfg["trade_url"] + "/v2", "/orders?status=open&limit=100")
    return r if isinstance(r, list) else []

def api_cancel_order(order_id):
    cfg = load_cfg()
    return _req(cfg["trade_url"] + "/v2", f"/orders/{order_id}", "DELETE")

print("✅ Part 1: Alpaca core جاهز")


# ════════════════════════════════════════════════════════════
# Part 2: طبقة التحليل — yfinance للتاريخي + Alpaca للسعر اللحظي
# ════════════════════════════════════════════════════════════

# ── قوائم الأسهم الحلال ──
US_STOCKS = {
    "AAPL":"Apple","MSFT":"Microsoft","GOOGL":"Alphabet","META":"Meta",
    "NVDA":"NVIDIA","AMD":"AMD","TSLA":"Tesla","AMZN":"Amazon",
    "INTC":"Intel","QCOM":"Qualcomm","AVGO":"Broadcom","TXN":"Texas Instruments",
    "MU":"Micron","AMAT":"Applied Materials","LRCX":"Lam Research","KLAC":"KLA",
    "CRM":"Salesforce","NOW":"ServiceNow","SNOW":"Snowflake","DDOG":"Datadog",
    "ADSK":"Autodesk","ORCL":"Oracle","INTU":"Intuit","CDNS":"Cadence",
    "SNPS":"Synopsys","ZS":"Zscaler","CRWD":"CrowdStrike","PANW":"Palo Alto",
    "FTNT":"Fortinet","NET":"Cloudflare","JNJ":"Johnson & Johnson","UNH":"UnitedHealth",
    "ABBV":"AbbVie","TMO":"Thermo Fisher","ABT":"Abbott","DHR":"Danaher",
    "ISRG":"Intuitive Surgical","SYK":"Stryker","VRTX":"Vertex","REGN":"Regeneron",
    "COST":"Costco","HD":"Home Depot","WMT":"Walmart","NKE":"Nike",
    "MCD":"McDonald's","SBUX":"Starbucks","LULU":"Lululemon","TJX":"TJX",
    "XOM":"ExxonMobil","CVX":"Chevron","COP":"ConocoPhillips","EOG":"EOG",
    "LIN":"Linde","APD":"Air Products","SHW":"Sherwin-Williams","HON":"Honeywell",
    "CAT":"Caterpillar","DE":"John Deere","ETN":"Eaton","GE":"GE Aerospace",
    "UBER":"Uber","ABNB":"Airbnb","BKNG":"Booking","NFLX":"Netflix",
    "ADI":"Analog Devices","MRVL":"Marvell","NUE":"Nucor","FSLR":"First Solar",
    "PANW":"Palo Alto","DUOL":"Duolingo","TTWO":"Take-Two","ODFL":"Old Dominion",
}

CRYPTO = {
    "BTC-USD":"Bitcoin","ETH-USD":"Ethereum","SOL-USD":"Solana",
    "AVAX-USD":"Avalanche","LINK-USD":"Chainlink","DOT-USD":"Polkadot",
    "LTC-USD":"Litecoin","UNI-USD":"Uniswap","AAVE-USD":"Aave",
    "BCH-USD":"Bitcoin Cash","XRP-USD":"Ripple","DOGE-USD":"Dogecoin",
}

# ── مؤشرات فنية ──
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

def get_history(ticker, period="2y", interval="1d", retries=2):
    """بيانات تاريخية من yfinance للتحليل فقط"""
    for attempt in range(retries+1):
        try:
            df = yf.download(ticker, period=period, interval=interval,
                           progress=False, auto_adjust=True, timeout=15)
            if df.empty:
                if attempt < retries: time.sleep(2); continue
                return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        except Exception as e:
            if "rate" in str(e).lower() or "too many" in str(e).lower():
                if attempt < retries: time.sleep(3+attempt*2); continue
            elif attempt < retries: time.sleep(2); continue
            return pd.DataFrame()
    return pd.DataFrame()

def analyze_symbol(code, name, is_crypto=False, live_price=None):
    """
    تحليل JRF كامل.
    التحليل من yfinance (تاريخي)، السعر اللحظي من Alpaca (live_price).
    """
    try:
        df = get_history(code, "2y", "1d")
        if df.empty or len(df) < 50: return None

        # السعر: نفضّل Alpaca اللحظي، fallback لـ yfinance
        yf_price = float(df["Close"].iloc[-1])
        price = live_price if live_price else yf_price

        # ── فلتر السعر: 3 فئات سعرية، كل فئة لها حد سيولة مختلف ──
        # عادي: 5$-150$ | منخفض: 1$-5$ | بايني: أقل من 1$
        price_tier = "normal"
        if not is_crypto:
            if price > 150:
                return None
            elif price >= 5:
                price_tier = "normal"
            elif price >= 1:
                price_tier = "low"
            elif price > 0:
                price_tier = "penny"
            else:
                return None

        c = df.copy()
        e20,e50,e200 = _ema(c["Close"],20),_ema(c["Close"],50),_ema(c["Close"],200)
        rv = _rsi(c["Close"]); ml,sg,mh = _macd(c["Close"])
        adv = _adx(c["High"],c["Low"],c["Close"])
        sk,sd = _stoch(c["High"],c["Low"],c["Close"])
        va = _sma(c["Volume"],20); av = _atr(c["High"],c["Low"],c["Close"])

        # ── حماية إضافية للفئات منخفضة السعر: حد أدنى سيولة يومية ──
        avg_vol = float(va.iloc[-1]) if not pd.isna(va.iloc[-1]) else 0
        if not is_crypto and price_tier == "low" and avg_vol < 500000:
            return None
        if not is_crypto and price_tier == "penny" and avg_vol < 2000000:
            return None

        last = -1
        # نظام JRF — 20 نقطة
        c1 = price > float(e20.iloc[last])
        c2 = float(e20.iloc[last]) > float(e50.iloc[last])
        c3 = price > float(e200.iloc[last])
        c4 = float(ml.iloc[last]) > float(sg.iloc[last])
        c5 = float(mh.iloc[last]) > float(mh.iloc[last-1])
        c6 = 40 <= float(rv.iloc[last]) <= 70
        c7 = float(rv.iloc[last]) > float(rv.iloc[last-1])
        c8 = float(adv.iloc[last]) > 20
        c9 = float(sk.iloc[last]) > 20 and float(sk.iloc[last]) > float(sd.iloc[last])
        c10 = float(c["Volume"].iloc[last]) > float(va.iloc[last]) * 1.2
        c11 = float(adv.iloc[last]) > float(adv.iloc[last-1])

        score = (2*c1+2*c2+2*c3+2*c4+1*c5+2*c6+1*c7+2*c8+2*c9+2*c10+2*c11)

        atr_v = float(av.iloc[last])
        lb = round(price * 0.998, 2)
        t1 = round(price + atr_v*2.0, 2)
        t2 = round(price + atr_v*4.0, 2)
        sl = round(price - atr_v*1.5, 2)
        rr = round((t1-price)/max(price-sl,0.01), 2)
        ptp = round((t1-price)/price*100, 2)
        psl = round((price-sl)/price*100, 2)

        rsi_v = round(float(rv.iloc[last]),1)
        adx_v = round(float(adv.iloc[last]),1)
        vr = round(float(c["Volume"].iloc[last])/float(va.iloc[last]),1) if float(va.iloc[last])>0 else 1

        # اتجاه
        e20w = float(_ema(c["Close"],100).iloc[last])  # تقريب أسبوعي
        ad = price > float(e20.iloc[last])
        aw = price > e20w
        if score>=15 and ad and aw: trend="استثمار"; stars=3
        elif score>=12 and ad: trend="سوينج"; stars=2
        elif ad: trend="مضاربة"; stars=1
        else: trend="ضعيف"; stars=0

        # الحكم
        if score>=15 and rr>=1.3: verdict="BUY"
        elif score>=12 and rr>=1.0 and ad: verdict="BUY_COND"
        elif score>=9: verdict="WATCH"
        else: verdict="AVOID"

        # ── نظام المستويات الثلاثة (Tier) لتحديد حجم المركز ──
        # المستوى 1: مطابقة كاملة لشروط JRF
        # المستوى 2: مطابقة جزئية (BUY_COND) + إشارة قوة إضافية (حجم تداول مرتفع)
        # المستوى 3: غير مطابق (WATCH) لكن بإشارة استثنائية قوية جداً
        vr_now = round(float(c["Volume"].iloc[last])/float(va.iloc[last]),1) if float(va.iloc[last])>0 else 1
        entry_tier = None
        if verdict == "BUY":
            entry_tier = 1
        elif verdict == "BUY_COND" and vr_now >= 1.2:
            entry_tier = 2
        elif verdict == "WATCH" and score >= 9 and vr_now >= 2.0 and float(adv.iloc[last]) > float(adv.iloc[last-1]):
            entry_tier = 3

        confidence = min(100, round(score/20*60 + min(rr,3)/3*20 + stars/3*20))
        prev = float(df["Close"].iloc[last-1])
        chg = round((yf_price-prev)/prev*100, 2)

        # الوقت المتوقع — ديناميكي: المسافة لأعلى قمة 20 يوم (مقاومة فعلية)
        daily_move = atr_v if atr_v > 0 else price*0.015
        high20 = float(c["High"].iloc[-20:].max())
        # لو السعر تحت القمة، المسافة لها؛ لو فوقها، نستخدم الهدف t1
        if high20 > price:
            dist = high20 - price
        else:
            dist = abs(t1 - price)
        raw_days = max(1, round(dist / daily_move))
        # قوة الاتجاه تسرّع الوصول
        if adx_v >= 30: factor = 0.8
        elif adx_v >= 20: factor = 1.0
        else: factor = 1.5
        est_days = max(1, round(raw_days * factor))
        if est_days <= 3: eta = f"{est_days}-{est_days+2} أيام"
        elif est_days <= 10: eta = f"{est_days}-{est_days+4} أيام"
        else: eta = f"{est_days}-{est_days+7} يوم"

        return {
            "code":code, "name":name, "is_crypto":is_crypto,
            "price":round(price,2), "price_sar":round(price*USD_TO_SAR,2),
            "live":bool(live_price), "score":score, "verdict":verdict,
            "trend":trend, "stars":stars, "confidence":confidence,
            "lb":lb, "t1":t1, "t2":t2, "sl":sl, "rr":rr, "ptp":ptp, "psl":psl,
            "rsi":rsi_v, "adx":adx_v, "vr":vr, "chg":chg,
            "above_daily":ad, "above_weekly":aw, "eta":eta,
            "price_tier":price_tier, "entry_tier":entry_tier,
        }
    except Exception as e:
        return None



# ════════════════════════════════════════════════════════════
# Part 3: محرّك التداول الآمن — كل الحمايات والتحقّقات
# ════════════════════════════════════════════════════════════

file_lock = threading.Lock()
TRADES_FILE = os.path.join(os.getcwd(), "pro_trades.json")
POS_STATE_FILE = os.path.join(os.getcwd(), "pro_posstate.json")
SCAN_FILE = os.path.join(os.getcwd(), "pro_lastscan.json")
FAV_FILE = os.path.join(os.getcwd(), "pro_favorites.json")

def load_favs():
    if os.path.exists(FAV_FILE):
        try:
            with open(FAV_FILE) as f: return json.load(f)
        except: return {"us": [], "crypto": []}
    return {"us": [], "crypto": []}

def save_favs(d):
    with open(FAV_FILE, "w") as f: json.dump(d, f, ensure_ascii=False, indent=2)

def load_trades(limit=500):
    with file_lock:
        if os.path.exists(TRADES_FILE):
            try:
                with open(TRADES_FILE) as f:
                    data = json.load(f)
                    return data[-limit:] if limit else data
            except: return []
        return []

def add_trade(t):
    with file_lock:
        trades = []
        if os.path.exists(TRADES_FILE):
            try:
                with open(TRADES_FILE) as f: trades = json.load(f)
            except: pass
        trades.append(t)
        with open(TRADES_FILE,"w") as f: json.dump(trades, f, ensure_ascii=False, indent=2)

def load_posstate():
    with file_lock:
        if os.path.exists(POS_STATE_FILE):
            try:
                with open(POS_STATE_FILE) as f: return json.load(f)
            except: return {}
        return {}

def save_posstate(d):
    with file_lock:
        with open(POS_STATE_FILE,"w") as f: json.dump(d, f, ensure_ascii=False, indent=2)

# حالة المسح (للعرض)
scan_state = {
    "us": {"data": [], "last": None, "status": "idle", "progress": 0, "scanned": 0},
    "crypto": {"data": [], "last": None, "status": "idle", "progress": 0, "scanned": 0},
}

# ════════════════════════════════════════════════════════════
# المسح — يجيب السعر اللحظي من Alpaca لكل سهم
# ════════════════════════════════════════════════════════════
def run_scan(market="us"):
    scan_state[market]["status"] = "scanning"
    scan_state[market]["progress"] = 0

    stocks = US_STOCKS if market=="us" else CRYPTO
    is_crypto = (market=="crypto")
    items = list(stocks.items())

    # نضمّن المفضلة دائماً (تتحلل كل مسح حتى لو ما طلعت في العيّنة)
    favs = load_favs().get(market, [])
    fav_items = [(c, stocks.get(c, c)) for c in favs]

    # للأمريكي: عيّنة عشوائية 40 لتجنب حظر yfinance
    if market=="us":
        import random
        random.seed(int(time.time())//1800)
        random.shuffle(items)
        items = items[:40]

    # ندمج المفضلة مع العيّنة (بدون تكرار)
    seen = set(c for c,_ in items)
    for c, n in fav_items:
        if c not in seen:
            items.append((c, n)); seen.add(c)

    total = len(items)
    results = []
    lock = threading.Lock()
    done = [0]

    def scan_one(code, name):
        # السعر اللحظي من Alpaca أولاً
        live = api_last_price(code, is_crypto) if load_cfg().get("key") else None
        r = analyze_symbol(code, name, is_crypto, live)
        with lock:
            done[0] += 1
            scan_state[market]["progress"] = round(done[0]/total*100)
            if r: results.append(r)

    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = []
        for code, name in items:
            futs.append(ex.submit(scan_one, code, name))
            time.sleep(0.25)
        for f in as_completed(futs): pass

    # ترتيب أفضل 5
    buys = [s for s in results if s["verdict"] in ("BUY","BUY_COND")]
    buys.sort(key=lambda x:(-x["score"],-x["confidence"],-x["rr"]))
    top = buys[:5]
    if len(top) < 5:
        watch = [s for s in results if s["verdict"]=="WATCH"]
        watch.sort(key=lambda x:(-x["score"],-x["confidence"]))
        top += watch[:5-len(top)]

    medals = ["🥇","🥈","🥉","4️⃣","5️⃣"]
    for i,s in enumerate(top):
        s["medal"] = medals[i] if i<len(medals) else str(i+1)
    # نعلّم المفضلة في كل النتائج
    fav_set = set(load_favs().get(market, []))
    for s in top:
        s["is_fav"] = s["code"] in fav_set

    scan_state[market]["data"] = top
    scan_state[market]["last"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    scan_state[market]["scanned"] = len(results)
    scan_state[market]["status"] = "done"
    scan_state[market]["progress"] = 100
    return top

# ════════════════════════════════════════════════════════════
# الشراء الآمن — مع كل التحقّقات
# ════════════════════════════════════════════════════════════
def safe_buy(signal, source="auto"):
    """
    شراء آمن مع تحقّقات:
    1. السوق مفتوح؟ (من Alpaca)
    2. ما اشترينا نفس السهم اليوم؟
    3. ما تجاوزنا حد الصفقات اليومي؟
    4. ما تجاوزنا حد الخسارة اليومي؟
    5. نتحقق من تنفيذ الأمر فعلاً
    """
    cfg = load_cfg()
    if not cfg.get("enabled") or not cfg.get("key"):
        return {"ok": False, "msg": "التداول غير مفعّل"}

    code = signal["code"]
    is_crypto = signal.get("is_crypto", False)

    # 1. السوق مفتوح؟ (الكريبتو 24/7)
    if not is_crypto and not market_is_open():
        return {"ok": False, "msg": "السوق مغلق"}

    today = datetime.now().strftime("%Y-%m-%d")
    trades = load_trades()
    today_buys = [t for t in trades if t.get("time","").startswith(today)
                  and t.get("side")=="buy" and t.get("source")=="auto"]

    # 2. اشترينا نفس السهم اليوم؟
    if any(t.get("symbol")==code for t in today_buys):
        return {"ok": False, "msg": "اشترينا هذا السهم اليوم"}

    # 3. حد الصفقات اليومي
    if source=="auto" and len(today_buys) >= int(cfg.get("max_daily_trades",5)):
        return {"ok": False, "msg": "وصلنا حد الصفقات اليومي"}

    # 4. حد الخسارة اليومي
    if check_daily_loss():
        return {"ok": False, "msg": "تجاوزنا حد الخسارة اليومي — التداول متوقف"}

    # 5. حساب الكمية بالسعر اللحظي
    live = api_last_price(code, is_crypto)
    price = live if live else signal["price"]
    if price <= 0: return {"ok": False, "msg": "سعر غير صالح"}

    # ── فلتر الفئة السعرية: التحقق من السماح بها قبل الشراء ──
    p_tier = signal.get("price_tier", "normal")
    if not is_crypto:
        if p_tier == "low" and not cfg.get("allow_low_price", True):
            return {"ok": False, "msg": "فئة 1$-5$ غير مفعّلة بالإعدادات"}
        if p_tier == "penny" and not cfg.get("allow_penny_price", False):
            return {"ok": False, "msg": "فئة أقل من 1$ غير مفعّلة بالإعدادات"}

    # ── حجم المركز الذكي: % من رأس المال الفعلي حسب المستوى (Tier) ──
    e_tier = signal.get("entry_tier") or 1
    tier_pct = {1: cfg.get("tier1_pct", 2.0), 2: cfg.get("tier2_pct", 1.2), 3: cfg.get("tier3_pct", 0.5)}.get(e_tier, 2.0)

    acct = api_account()
    equity = None
    if "error" not in acct:
        try: equity = float(acct.get("equity", 0))
        except: equity = None

    if equity and equity > 0:
        max_usd = equity * (tier_pct / 100.0)
    else:
        # احتياطي لو فشل الاتصال بالحساب
        max_usd = float(cfg.get("max_position_usd", 500))

    # كسور للاثنين (Alpaca يدعم fractional shares) — يحترم حجم الصفقة بدقة
    if is_crypto:
        qty = round(max_usd / price, 6)
    else:
        qty = round(max_usd / price, 3)  # كسور للأسهم أيضاً

    # نوع الأمر من الإعدادات (market = تنفيذ مضمون)
    otype = cfg.get("buy_order_type", "market")
    tif = "gtc" if is_crypto else "day"
    limit_px = signal["lb"] if otype == "limit" else None
    result = api_place_order(code, qty, "buy", otype, limit_px, tif, is_crypto)
    # fallback: لو رُفض الأمر الكسري للأسهم، نعيد المحاولة برقم صحيح
    if "error" in result and not is_crypto and qty != int(qty):
        int_qty = max(1, int(qty))
        result = api_place_order(code, int_qty, "buy", otype, limit_px, tif, is_crypto)
        if "error" not in result:
            qty = int_qty
    if "error" in result:
        return {"ok": False, "msg": result["error"]}

    order_id = result.get("id","")

    # نسجّل الصفقة
    add_trade({
        "symbol":code, "name":signal.get("name",""),
        "side":"buy", "qty":qty, "order_type":otype,
        "limit_price":limit_px, "tp1":signal["t1"], "tp2":signal["t2"],
        "sl":signal["sl"], "score":signal["score"], "confidence":signal["confidence"],
        "is_crypto":is_crypto, "order_id":order_id, "status":result.get("status",""),
        "time":datetime.now().strftime("%Y-%m-%d %H:%M"), "source":source,
        "entry_tier":e_tier, "price_tier":p_tier,
    })
    return {"ok": True, "qty": qty, "order_id": order_id, "price": price}

# ════════════════════════════════════════════════════════════
# حد الخسارة اليومي
# ════════════════════════════════════════════════════════════
def check_daily_loss():
    cfg = load_cfg()
    acc = api_account()
    if "error" in acc: return False
    try:
        equity = float(acc.get("equity", 0))
        last_equity = float(acc.get("last_equity", equity))
        if last_equity <= 0: return False
        day_change_pct = (equity - last_equity) / last_equity * 100
        return day_change_pct <= -float(cfg.get("daily_loss_pct", 3.0))
    except:
        return False



# ════════════════════════════════════════════════════════════
# Part 4: محرّك البيع الذكي + تحقّق التنفيذ + إلغاء المعلّقة
# ════════════════════════════════════════════════════════════

def verify_pending_orders():
    """
    يتحقق من الأوامر المعلّقة:
    - لو أمر شراء معلّق من يوم سابق → يلغيه (ما عاد السعر مناسب)
    """
    cfg = load_cfg()
    if not cfg.get("enabled"): return
    open_orders = api_open_orders()
    today = datetime.now().strftime("%Y-%m-%d")
    for o in open_orders:
        created = o.get("created_at","")[:10]
        # لو الأمر من يوم سابق وما زال معلّق → ألغِه
        if created and created < today:
            oid = o.get("id","")
            if oid:
                api_cancel_order(oid)
                add_trade({
                    "symbol":o.get("symbol",""), "side":"cancel",
                    "qty":o.get("qty","0"), "reason":"أمر معلّق من يوم سابق — أُلغي",
                    "time":datetime.now().strftime("%Y-%m-%d %H:%M"), "source":"auto",
                })

def monitor_positions():
    """
    البيع الذكي:
    - +tp1% → بيع 40%
    - +tp2% → بيع 30%
    - الباقي → trailing stop
    - وقف خسارة أصلي لو نزل قبل الربح
    - حد خسارة يومي → إغلاق الكل
    """
    cfg = load_cfg()
    if not cfg.get("enabled") or not cfg.get("auto_sell"): return

    # حد الخسارة اليومي
    if check_daily_loss():
        for pos in api_positions():
            api_close_position(pos.get("symbol",""))
        return

    positions = api_positions()
    if not positions: return

    trades = load_trades()
    pstate = load_posstate()
    tp1_pct = float(cfg.get("tp1_pct",2.0))
    tp2_pct = float(cfg.get("tp2_pct",4.0))
    trail_pct = float(cfg.get("trail_pct",1.5))

    for pos in positions:
        symbol = pos.get("symbol","")
        # نطبّع رمز الكريبتو
        norm = symbol.replace("/USD","-USD") if "/USD" in symbol else symbol
        try:
            current = float(pos.get("current_price",0))
            entry = float(pos.get("avg_entry_price",0))
            qty = float(pos.get("qty",0))
        except:
            continue
        if current<=0 or entry<=0 or qty<=0: continue

        is_crypto = "/USD" in symbol or "-USD" in norm
        pnl_pct = (current-entry)/entry*100

        st = pstate.get(norm, {"sold_tp1":False,"sold_tp2":False,"highest":current,"orig_qty":qty})
        if current > st.get("highest",0): st["highest"] = current

        # وقف الخسارة: من الصفقة المسجّلة، أو افتراضي للمراكز اليدوية
        matching = [t for t in trades if t.get("symbol")==norm and t.get("side")=="buy"]
        pos_tier = matching[-1].get("entry_tier", 1) if matching else 1
        if matching and matching[-1].get("sl"):
            orig_sl = float(matching[-1].get("sl") or 0)
        else:
            orig_sl = round(entry * 0.97, 2)

        # ── المستوى 3 (فرص استثنائية غير مطابقة): ستوب أضيق وخروج أسرع ──
        pos_trail_pct = trail_pct
        if pos_tier == 3:
            orig_sl = max(orig_sl, round(entry * 0.985, 2))  # ستوب أضيق (1.5% بدل ~3%)
            pos_trail_pct = min(trail_pct, 1.0)              # تريلينج أضيق

        # نقرأ الوقف المعدّل من الحالة لو وُجد (يحفظ Break-Even عبر الدقائق)
        current_sl = st.get("adj_sl", orig_sl)
        # Break-Even: لو وصل +1.5% قبل TP1، نرفع الوقف لسعر الدخول ونحفظه
        if not st["sold_tp1"] and pnl_pct >= 1.5:
            current_sl = max(current_sl, entry)
            st["adj_sl"] = current_sl  # حفظ في الحالة عشان ما ينساه

        sold = False

        # ── المرحلة 1: +tp1% بيع 40% ──
        if not st["sold_tp1"] and pnl_pct >= tp1_pct:
            q = round(st["orig_qty"]*0.40, 6 if is_crypto else 3)
            if q <= 0: q = qty  # لو الكمية صغيرة جداً، بِع الكل
            r = api_place_order(norm, q, "sell", "market",
                                tif=("gtc" if is_crypto else "day"), is_crypto=is_crypto)
            if "error" not in r:
                st["sold_tp1"]=True; sold=True
                add_trade({"symbol":norm,"side":"sell","qty":q,"reason":f"هدف 1 (+{tp1_pct}%) بيع 40%",
                          "exit_price":current,"pnl":round((current-entry)*q,2),
                          "time":datetime.now().strftime("%Y-%m-%d %H:%M"),"source":"auto"})

        # ── المرحلة 2: +tp2% بيع 30% ──
        elif st["sold_tp1"] and not st["sold_tp2"] and pnl_pct >= tp2_pct:
            q = round(st["orig_qty"]*0.30, 6 if is_crypto else 3)
            q = min(q, qty)
            if q <= 0: q = qty
            r = api_place_order(norm, q, "sell", "market",
                                tif=("gtc" if is_crypto else "day"), is_crypto=is_crypto)
            if "error" not in r:
                st["sold_tp2"]=True; sold=True
                add_trade({"symbol":norm,"side":"sell","qty":q,"reason":f"هدف 2 (+{tp2_pct}%) بيع 30%",
                          "exit_price":current,"pnl":round((current-entry)*q,2),
                          "time":datetime.now().strftime("%Y-%m-%d %H:%M"),"source":"auto"})

        # ── Trailing Stop للباقي (بعد بيع جزء) ──
        if st["sold_tp1"] and not sold:
            trail = st["highest"]*(1-pos_trail_pct/100)
            if current <= trail:
                r = api_close_position(norm)
                if "error" not in r:
                    add_trade({"symbol":norm,"side":"sell","qty":qty,
                              "reason":f"وقف متحرك (قمة ${round(st['highest'],2)})",
                              "exit_price":current,"pnl":pos.get("unrealized_pl","0"),
                              "time":datetime.now().strftime("%Y-%m-%d %H:%M"),"source":"auto"})
                    pstate.pop(norm,None); save_posstate(pstate); continue

        # ── وقف الخسارة (الأصلي أو المعدّل بـ Break-Even) ──
        if not st["sold_tp1"] and current_sl>0 and current<=current_sl:
            r = api_close_position(norm)
            if "error" not in r:
                add_trade({"symbol":norm,"side":"sell","qty":qty,"reason":"وقف الخسارة 🛡",
                          "exit_price":current,"pnl":pos.get("unrealized_pl","0"),
                          "time":datetime.now().strftime("%Y-%m-%d %H:%M"),"source":"auto"})
                pstate.pop(norm,None); save_posstate(pstate)
                # مسح جديد بعد إغلاق المركز
                threading.Thread(target=_rescan_after_sell, daemon=True).start()
                continue

        pstate[norm] = st

    save_posstate(pstate)

def _rescan_after_sell():
    """مسح جديد بعد البيع لإيجاد فرصة بديلة"""
    time.sleep(30)
    cfg = load_cfg()
    if not cfg.get("enabled") or not cfg.get("auto_buy"): return
    if not market_is_open(): return
    print("🔄 مسح جديد بعد البيع...")
    auto_scan_and_trade("us")



# ════════════════════════════════════════════════════════════
# Part 5: المجدول الذكي — مؤسس على ساعة Alpaca الحقيقية
# ════════════════════════════════════════════════════════════

def auto_scan_and_trade(market="us"):
    """مسح + تداول تلقائي — المستويات الثلاثة (Tier 1/2/3)"""
    cfg = load_cfg()
    if not cfg.get("enabled"): return
    top = run_scan(market)
    if not cfg.get("auto_buy"): return
    bought = 0
    for s in top:
        e_tier = s.get("entry_tier")
        # المستوى المسموح للشراء التلقائي: 1، 2، أو 3 — لازم يكون مصنّف (verdict AVOID أو WATCH ضعيف يُستبعد تلقائياً)
        if e_tier not in (1, 2, 3):
            continue
        # المستوى 1 يستخدم حد الثقة العادي، المستويات 2/3 تتطلب ثقة أعلى تعويضاً عن ضعف المطابقة
        min_conf = int(cfg.get("min_confidence", 70))
        if e_tier == 2: min_conf = max(min_conf, 75)
        if e_tier == 3: min_conf = max(min_conf, 80)
        if s["confidence"] < min_conf: continue
        # ── Volume spike: حجم التداول لازم يكون أعلى من المعدل بـ 20% على الأقل (المستوى 1 فقط، لأن 2و3 محسوبين أصلاً بشرط حجم أعلى) ──
        if e_tier == 1 and not s.get("is_crypto") and s.get("vr", 0) < 1.2:
            print(f"⏭️ {s['code']} تجاهل — حجم منخفض ({s.get('vr',0)}x)")
            continue
        res = safe_buy(s, "auto")
        if res["ok"]: bought += 1
    return bought

def scheduler_loop():
    """
    حلقة المجدول — كل دقيقة: مراقبة + إلغاء معلّق + مسح عند الفتح.
    استدعاء api_clock مُخزّن 5 دقائق لتوفير rate limit.
    """
    last_us_scan_day = None
    last_crypto_scan_hour = None
    clock_cache = {"data": None, "ts": 0}

    def cached_clock():
        # نخزّن نتيجة الساعة 5 دقائق
        if time.time() - clock_cache["ts"] > 300 or clock_cache["data"] is None:
            clock_cache["data"] = api_clock()
            clock_cache["ts"] = time.time()
        return clock_cache["data"]

    while True:
        try:
            cfg = load_cfg()
            if cfg.get("enabled") and cfg.get("key"):
                # 1. مراقبة المراكز دائماً
                monitor_positions()

                # 2. إلغاء الأوامر المعلّقة القديمة
                verify_pending_orders()

                today = datetime.now().strftime("%Y-%m-%d")

                # 3. السوق الأمريكي — ساعة Alpaca (مخزّنة 5 دقائق)
                if "us" in cfg.get("markets",["us"]):
                    clock = cached_clock()
                    is_open = clock.get("is_open", False) if "error" not in clock else False

                    # Pre-market: مسح قبل 30 دقيقة من الفتح
                    next_open_str = clock.get("next_open", "") if "error" not in clock else ""
                    if not is_open and next_open_str and last_us_scan_day != today + "_pre":
                        try:
                            from datetime import timezone
                            next_open_dt = datetime.fromisoformat(next_open_str.replace("Z","+00:00"))
                            mins_to_open = (next_open_dt - datetime.now(timezone.utc)).total_seconds() / 60
                            if 0 < mins_to_open <= 30:
                                print(f"⏰ Pre-market مسح — {round(mins_to_open)} دقيقة قبل الفتح")
                                run_scan("us")  # مسح بدون شراء (السوق مغلق)
                                last_us_scan_day = today + "_pre"
                        except Exception as _e:
                            print(f"pre-market error: {_e}")

                    if is_open:
                        # السوق مفتوح فعلاً (Alpaca أكّد)
                        cur_15 = datetime.now().strftime("%Y-%m-%d-%H") + "-" + str(datetime.now().minute // 15)
                        if last_us_scan_day != cur_15:
                            res = auto_scan_and_trade("us")
                            if scan_state["us"].get("data"):
                                last_us_scan_day = cur_15
                                print(f"✅ مسح أمريكي تلقائي {cur_15} — {res} صفقة")

                # 4. الكريبتو 24/7 — مسح كل ساعة
                if "crypto" in cfg.get("markets",[]):
                    cur_hour = datetime.utcnow().strftime("%Y-%m-%d-%H")
                    if last_crypto_scan_hour != cur_hour:
                        last_crypto_scan_hour = cur_hour
                        auto_scan_and_trade("crypto")
                        print(f"✅ مسح كريبتو {cur_hour}")
        except Exception as e:
            print(f"⚠️ خطأ في المجدول: {e}")

        time.sleep(60)

def start_scheduler():
    # قفل ملف يضمن مجدول واحد فقط حتى مع gunicorn multi-workers
    lock_file = os.path.join(os.getcwd(), "scheduler.lock")
    try:
        import fcntl
        global _lock_fd
        _lock_fd = open(lock_file, "w")
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # نجحنا بأخذ القفل → هذا الـ worker الوحيد اللي يشغّل المجدول
    except (ImportError, IOError, OSError):
        # worker ثاني (القفل مأخوذ) أو نظام ما يدعم fcntl → ما نشغّل المجدول
        print("⏭️ المجدول يعمل في worker آخر — تخطّينا")
        return
    t = threading.Thread(target=scheduler_loop, daemon=True)
    t.start()
    print("🤖 المجدول الذكي شغّال (worker واحد)")



# ════════════════════════════════════════════════════════════
# Part 7a: HTML + CSS — ثيم فاتح عصري نظيف
# ════════════════════════════════════════════════════════════

PAGE_CSS = r"""
:root {
  --bg: #f7f9fc;
  --surface: #ffffff;
  --surface-2: #f1f5f9;
  --ink: #0f1729;
  --ink-2: #475569;
  --ink-3: #94a3b8;
  --line: #e5eaf1;
  --brand: #2563eb;
  --brand-dark: #1d4ed8;
  --brand-soft: #eff6ff;
  --green: #059669;
  --green-soft: #ecfdf5;
  --red: #e11d48;
  --red-soft: #fff1f3;
  --amber: #d97706;
  --amber-soft: #fffbeb;
  --shadow-sm: 0 1px 2px rgba(15,23,41,.06);
  --shadow: 0 2px 8px rgba(15,23,41,.06), 0 1px 3px rgba(15,23,41,.04);
  --shadow-lg: 0 12px 32px rgba(15,23,41,.10);
  --r: 14px;
  --r-lg: 20px;
}
* { margin:0; padding:0; box-sizing:border-box; }
body {
  font-family:'Tajawal',system-ui,sans-serif;
  background:var(--bg); color:var(--ink);
  direction:rtl; line-height:1.6; -webkit-font-smoothing:antialiased;
}
.shell { max-width:760px; margin:0 auto; padding:0 16px 64px; }

/* الهيدر */
.top {
  background:linear-gradient(160deg,#1e3a8a 0%,#2563eb 55%,#3b82f6 100%);
  margin:0 -16px; padding:28px 16px 64px;
  border-radius:0 0 28px 28px; color:#fff; position:relative;
}
.top::after {
  content:''; position:absolute; inset:0; border-radius:0 0 28px 28px;
  background:radial-gradient(circle at 80% 0%, rgba(255,255,255,.12), transparent 50%);
  pointer-events:none;
}
.brand { display:flex; align-items:center; justify-content:center; gap:10px; }
.brand-mark {
  width:42px; height:42px; border-radius:12px;
  background:rgba(255,255,255,.16); display:grid; place-items:center;
  font-size:22px; backdrop-filter:blur(8px);
  border:1px solid rgba(255,255,255,.2);
}
.brand-name { font-size:26px; font-weight:800; letter-spacing:-.5px; }
.brand-sub {
  text-align:center; font-size:11px; letter-spacing:3px;
  opacity:.7; margin-top:4px; font-weight:500;
}
.clock-chip {
  display:flex; align-items:center; gap:7px; width:fit-content; margin:14px auto 0;
  background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.2);
  padding:6px 16px; border-radius:30px; font-size:13px; font-weight:600;
  backdrop-filter:blur(8px);
}
.clock-dot { width:7px; height:7px; border-radius:50%; background:#fbbf24; }
.clock-dot.open { background:#34d399; box-shadow:0 0 8px #34d399; }

/* تبويبات السوق */
.mkt-tabs {
  display:flex; gap:6px; background:var(--surface); padding:6px;
  border-radius:var(--r); box-shadow:var(--shadow); margin-top:-32px;
  position:relative; z-index:2;
}
.mkt-tab {
  flex:1; padding:11px; border:0; border-radius:10px; background:transparent;
  font-family:inherit; font-size:14px; font-weight:600; color:var(--ink-2);
  cursor:pointer; transition:.18s; display:flex; align-items:center;
  justify-content:center; gap:6px;
}
.mkt-tab.on { background:var(--brand); color:#fff; box-shadow:0 2px 8px rgba(37,99,235,.3); }

/* بطاقة التداول */
.panel {
  background:var(--surface); border-radius:var(--r-lg); box-shadow:var(--shadow);
  border:1px solid var(--line); margin-top:16px; overflow:hidden;
}
.panel-head {
  display:flex; align-items:center; justify-content:space-between;
  padding:16px 18px; border-bottom:1px solid var(--line);
}
.panel-title { font-size:15px; font-weight:800; display:flex; align-items:center; gap:8px; }
.conn { display:flex; align-items:center; gap:6px; font-size:12px; font-weight:600; color:var(--ink-3); }
.conn-dot { width:8px; height:8px; border-radius:50%; background:var(--ink-3); }
.conn.live { color:var(--green); }
.conn.live .conn-dot { background:var(--green); box-shadow:0 0 6px var(--green); }

/* تبويبات فرعية */
.sub-tabs { display:flex; gap:4px; padding:12px 14px 0; flex-wrap:wrap; }
.sub-tab {
  padding:8px 14px; border:1px solid var(--line); border-radius:9px;
  background:var(--surface); font-family:inherit; font-size:13px; font-weight:600;
  color:var(--ink-2); cursor:pointer; transition:.15s;
}
.sub-tab.on { background:var(--brand); color:#fff; border-color:var(--brand); }
.sub-body { padding:16px 18px; }

/* بطاقات الحساب */
.acct-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; }
.acct-cell {
  background:var(--surface-2); border-radius:12px; padding:13px 10px; text-align:center;
}
.acct-val { font-size:17px; font-weight:800; letter-spacing:-.5px; }
.acct-lbl { font-size:11px; color:var(--ink-3); margin-top:3px; }
.acct-val.pos { color:var(--green); } .acct-val.neg { color:var(--red); }

.info-strip {
  margin-top:13px; background:var(--brand-soft); border:1px solid #dbeafe;
  border-radius:11px; padding:11px 14px; font-size:13px; color:var(--brand-dark);
  font-weight:600; display:flex; align-items:center; gap:8px;
}
.stat-strip {
  margin-top:10px; background:var(--surface-2); border-radius:11px;
  padding:11px 14px; font-size:12.5px; color:var(--ink-2);
  display:flex; flex-wrap:wrap; gap:6px 14px; align-items:center;
}
.stat-strip b { color:var(--ink); }

.btn {
  font-family:inherit; font-weight:700; cursor:pointer; border:0;
  border-radius:10px; transition:.15s; font-size:13px;
}
.btn-primary { background:var(--brand); color:#fff; padding:9px 18px; }
.btn-primary:hover { background:var(--brand-dark); }
.btn-ghost { background:var(--surface-2); color:var(--ink-2); padding:9px 18px; border:1px solid var(--line); }
.btn-block { width:100%; }
.btn-lg { padding:13px; font-size:15px; }

/* حقول */
.field { display:flex; flex-direction:column; gap:5px; margin-bottom:11px; }
.field label { font-size:12px; color:var(--ink-3); font-weight:600; }
.input {
  background:var(--surface); border:1.5px solid var(--line); border-radius:10px;
  padding:10px 13px; font-family:inherit; font-size:14px; color:var(--ink);
  transition:.15s; width:100%;
}
.input:focus { outline:0; border-color:var(--brand); box-shadow:0 0 0 3px var(--brand-soft); }
.grid-2 { display:grid; grid-template-columns:1fr 1fr; gap:11px; }

/* مفاتيح التبديل */
.switch-row {
  display:flex; align-items:center; justify-content:space-between;
  padding:11px 0; border-bottom:1px solid var(--line);
}
.switch-row:last-of-type { border-bottom:0; }
.switch-txt { font-size:13.5px; color:var(--ink-2); font-weight:600; }
.switch {
  position:relative; width:44px; height:25px; flex-shrink:0;
}
.switch input { opacity:0; width:0; height:0; }
.switch-slider {
  position:absolute; inset:0; background:var(--line); border-radius:30px;
  cursor:pointer; transition:.2s;
}
.switch-slider::before {
  content:''; position:absolute; width:19px; height:19px; right:3px; top:3px;
  background:#fff; border-radius:50%; transition:.2s; box-shadow:var(--shadow-sm);
}
.switch input:checked + .switch-slider { background:var(--brand); }
.switch input:checked + .switch-slider::before { transform:translateX(-19px); }

/* صندوق الاستراتيجية */
.strat-box {
  margin-top:14px; background:var(--brand-soft); border:1px solid #dbeafe;
  border-radius:12px; padding:14px; font-size:12.5px; color:var(--brand-dark); line-height:1.9;
}
.strat-box .strat-h { font-weight:800; margin-bottom:7px; font-size:13px; }

/* المسح */
.scan-zone { text-align:center; margin-top:18px; }
.scan-btn {
  background:var(--brand); color:#fff; border:0; border-radius:30px;
  padding:14px 36px; font-family:inherit; font-size:15px; font-weight:700;
  cursor:pointer; box-shadow:0 6px 18px rgba(37,99,235,.32); transition:.18s;
  display:inline-flex; align-items:center; gap:9px;
}
.scan-btn:hover { transform:translateY(-2px); box-shadow:0 10px 24px rgba(37,99,235,.4); }
.scan-meta { font-size:12px; color:var(--ink-3); margin-top:9px; }

/* ملخص */
.tally { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-top:16px; }
.tally-cell { background:var(--surface); border:1px solid var(--line); border-radius:13px; padding:14px; text-align:center; box-shadow:var(--shadow-sm); }
.tally-num { font-size:23px; font-weight:800; }
.tally-lbl { font-size:11px; color:var(--ink-3); margin-top:2px; }
.tally-cell.g .tally-num { color:var(--green); }
.tally-cell.a .tally-num { color:var(--amber); }
.tally-cell.b .tally-num { color:var(--brand); }

/* بطاقة سهم */
.sig { background:var(--surface); border:1px solid var(--line); border-radius:var(--r-lg); box-shadow:var(--shadow-sm); margin-top:12px; overflow:hidden; transition:.18s; }
.sig:hover { box-shadow:var(--shadow); }
.sig-verdict { padding:11px 16px; display:flex; align-items:center; justify-content:space-between; }
.sig-verdict.buy { background:var(--green-soft); border-bottom:1px solid #d1fae5; }
.sig-verdict.cond { background:var(--amber-soft); border-bottom:1px solid #fde68a; }
.sig-verdict.watch { background:var(--surface-2); border-bottom:1px solid var(--line); }
.verdict-main { display:flex; align-items:center; gap:9px; }
.verdict-ico { font-size:21px; }
.verdict-label { font-weight:800; font-size:15px; }
.verdict-sub { font-size:11px; color:var(--ink-3); margin-top:1px; }
.verdict-score { text-align:left; }
.verdict-score-num { font-size:19px; font-weight:800; }
.verdict-score-lbl { font-size:10px; color:var(--ink-3); }

.sig-body { padding:14px 16px; }
.sig-top { display:flex; align-items:flex-start; gap:11px; }
.sig-medal { font-size:22px; }
.sig-id { flex:1; min-width:0; }
.sig-name { font-weight:700; font-size:15px; display:flex; align-items:center; flex-wrap:wrap; gap:4px; }
.sig-code { color:var(--ink-3); font-size:12px; font-weight:500; margin-right:6px; }
.sig-meta { font-size:12px; color:var(--ink-2); margin-top:4px; display:flex; flex-wrap:wrap; gap:5px 10px; }
.sig-meta .mb { unicode-bidi:isolate; direction:rtl; white-space:nowrap; }
.sig-px { text-align:left; flex-shrink:0; }
.sig-px-main { font-size:16px; font-weight:800; letter-spacing:-.5px; }
.sig-px-sar { font-size:11px; color:var(--ink-3); }
.sig-px-live { display:inline-flex; align-items:center; gap:3px; font-size:9px; color:var(--green); font-weight:700; margin-top:1px; }
.sig-px-live::before { content:''; width:5px; height:5px; border-radius:50%; background:var(--green); box-shadow:0 0 5px var(--green); }

.sig-bar { height:4px; background:var(--surface-2); border-radius:4px; overflow:hidden; margin:12px 0; }
.sig-bar-fill { height:100%; background:linear-gradient(90deg,var(--green),#34d399); border-radius:4px; }

.sig-tags { display:flex; gap:7px; flex-wrap:wrap; align-items:center; }
.tag { background:var(--surface-2); border:1px solid var(--line); border-radius:8px; padding:4px 9px; font-size:11.5px; color:var(--ink-2); unicode-bidi:isolate; white-space:nowrap; }
.tag b { color:var(--ink); }
.sig-trade-btn { margin-right:auto; background:var(--green); color:#fff; border:0; border-radius:9px; padding:6px 14px; font-family:inherit; font-size:12px; font-weight:700; cursor:pointer; }
.sig-trade-btn:hover { background:#047857; }

.sig-levels { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin-top:13px; }
.lvl { background:var(--surface-2); border-radius:10px; padding:10px 8px; text-align:center; }
.lvl-lbl { font-size:10px; color:var(--ink-3); }
.lvl-val { font-size:14px; font-weight:700; margin-top:2px; unicode-bidi:isolate; }
.lvl-pct { font-size:10px; margin-top:1px; }
.lvl.entry .lvl-val { color:var(--ink); }
.lvl.tp .lvl-val { color:var(--green); } .lvl.tp .lvl-pct { color:var(--green); }
.lvl.sl .lvl-val { color:var(--red); } .lvl.sl .lvl-pct { color:var(--red); }

/* مركز */
.pos { display:flex; align-items:center; justify-content:space-between; padding:13px; background:var(--surface-2); border-radius:12px; margin-bottom:8px; }
.pos-sym { font-weight:700; font-size:14px; }
.pos-info { font-size:11px; color:var(--ink-2); margin-top:2px; }
.pos-pnl { text-align:left; }
.pos-pnl-pct { font-weight:800; font-size:15px; }
.pos-close { background:transparent; border:1px solid var(--red-soft); color:var(--red); padding:5px 11px; border-radius:8px; font-family:inherit; font-size:11px; cursor:pointer; margin-top:4px; }

/* سجل */
.log-item { display:flex; justify-content:space-between; align-items:center; padding:9px 12px; border-radius:9px; margin-bottom:5px; font-size:12px; }
.log-item.buy { background:var(--green-soft); }
.log-item.sell { background:var(--red-soft); }
.log-item.cancel { background:var(--surface-2); }

/* إضافة سهم */
.add-zone { background:var(--surface); border:1px solid var(--line); border-radius:var(--r-lg); box-shadow:var(--shadow-sm); padding:16px 18px; margin-top:16px; }
.add-title { font-size:14px; font-weight:800; margin-bottom:11px; display:flex; align-items:center; gap:7px; }
.add-row { display:flex; gap:8px; flex-wrap:wrap; }
.add-row .input { flex:1; min-width:110px; }

/* فارغ */
.empty { text-align:center; padding:50px 20px; color:var(--ink-3); }
.empty-ico { font-size:40px; margin-bottom:12px; opacity:.5; }

/* توست */
.toast { position:fixed; bottom:24px; left:50%; transform:translateX(-50%) translateY(20px); background:var(--ink); color:#fff; padding:11px 22px; border-radius:30px; font-size:13px; font-weight:600; opacity:0; transition:.25s; z-index:100; pointer-events:none; }
.toast.show { opacity:1; transform:translateX(-50%) translateY(0); }

/* تحميل */
.loader { position:fixed; inset:0; background:rgba(247,249,252,.94); display:none; flex-direction:column; align-items:center; justify-content:center; z-index:90; backdrop-filter:blur(4px); }
.loader.show { display:flex; }
.loader-ring { width:46px; height:46px; border:3px solid var(--line); border-top-color:var(--brand); border-radius:50%; animation:spin .8s linear infinite; }
@keyframes spin { to { transform:rotate(360deg); } }
.loader-txt { margin-top:16px; font-weight:700; }
.loader-sub { font-size:12px; color:var(--ink-3); margin-top:4px; }

.foot { text-align:center; font-size:11px; color:var(--ink-3); padding:24px 0 0; }

.fav-btn { cursor:pointer; font-size:11px; font-weight:700; font-family:inherit; color:var(--ink-3); background:var(--surface-2); border:1px solid var(--line); border-radius:7px; padding:3px 9px; margin-right:8px; transition:.15s; }
.fav-btn.on { color:var(--amber); background:var(--amber-soft); border-color:#fde68a; }
.fav-btn:hover { border-color:var(--amber); }
.fav-header { font-size:13px; font-weight:800; color:var(--amber); margin:18px 0 4px; padding:8px 14px; background:var(--amber-soft); border-radius:11px; border:1px solid #fde68a; }
.hidden { display:none !important; }
@media(max-width:560px){ .acct-grid{grid-template-columns:repeat(2,1fr);} .sig-levels{grid-template-columns:repeat(2,1fr);} }
"""


# ════════════════════════════════════════════════════════════
# Part 7b: HTML Body
# ════════════════════════════════════════════════════════════

PAGE_BODY = r"""
<div class="loader" id="loader">
  <div class="loader-ring"></div>
  <div class="loader-txt">جاري المسح</div>
  <div class="loader-sub" id="loader-sub">يجلب الأسعار من Alpaca</div>
</div>
<div class="toast" id="toast"></div>

<div class="top">
  <div class="brand">
    <div class="brand-mark">⚡</div>
    <div class="brand-name">جلال رادار برو</div>
  </div>
  <div class="brand-sub">ALPACA-NATIVE · PRO</div>
  <div class="clock-chip" id="clock-chip">
    <span class="clock-dot" id="clock-dot"></span>
    <span id="clock-txt">يتحقق من السوق…</span>
  </div>
</div>

<div class="shell">
  <div class="mkt-tabs">
    <button class="mkt-tab on" id="mt-us" onclick="setMarket('us')">🇺🇸 أمريكي</button>
    <button class="mkt-tab" id="mt-crypto" onclick="setMarket('crypto')">💰 كريبتو</button>
  </div>

  <!-- لوحة التداول -->
  <div class="panel">
    <div class="panel-head">
      <div class="panel-title">🤖 التداول الآلي</div>
      <div class="conn" id="conn"><span class="conn-dot"></span><span id="conn-txt">غير متصل</span></div>
    </div>
    <div class="sub-tabs">
      <button class="sub-tab on" onclick="setSub('account',this)">💰 الحساب</button>
      <button class="sub-tab" onclick="setSub('trade',this)">📊 تداول</button>
      <button class="sub-tab" onclick="setSub('positions',this)">📁 مراكزي</button>
      <button class="sub-tab" onclick="setSub('settings',this)">⚙️ الإعدادات</button>
      <button class="sub-tab" onclick="setSub('log',this)">📋 السجل</button>
    </div>
    <div class="sub-body">
      <!-- الحساب -->
      <div id="sub-account">
        <div class="acct-grid">
          <div class="acct-cell"><div class="acct-val" id="a-equity">—</div><div class="acct-lbl">القيمة الكلية $</div></div>
          <div class="acct-cell"><div class="acct-val" id="a-cash">—</div><div class="acct-lbl">الكاش $</div></div>
          <div class="acct-cell"><div class="acct-val" id="a-bp">—</div><div class="acct-lbl">القوة الشرائية $</div></div>
          <div class="acct-cell"><div class="acct-val" id="a-pnl">—</div><div class="acct-lbl">ربح/خسارة $</div></div>
        </div>
        <div class="info-strip" id="market-info">🕐 يتحقق من حالة السوق…</div>
        <div class="stat-strip" id="stat-strip">
          📊 إجمالي <b id="s-total">0</b> · ✅ ربح <b id="s-wins">0</b> · ❌ خسارة <b id="s-losses">0</b> ·
          🎯 نجاح <b id="s-wr">0%</b> · 💰 <b id="s-pnl">$0</b> · 📅 اليوم <b id="s-today">0</b>/<b id="s-max">5</b>
        </div>
        <div style="margin-top:13px;"><button class="btn btn-ghost" onclick="loadAccount()">🔄 تحديث</button></div>
      </div>

      <!-- تداول يدوي -->
      <div id="sub-trade" class="hidden">
        <div class="grid-2">
          <div class="field"><label>الرمز</label><input class="input" id="t-sym" placeholder="AAPL"></div>
          <div class="field"><label>الكمية</label><input class="input" id="t-qty" type="number" value="1"></div>
        </div>
        <div class="field"><label>نوع الأمر</label>
          <select class="input" id="t-type"><option value="market">سوق (فوري)</option><option value="limit">محدد (Limit)</option></select>
        </div>
        <div class="field" id="t-limit-wrap" style="display:none;"><label>سعر Limit</label><input class="input" id="t-limit" type="number" step="0.01"></div>
        <div style="display:flex; gap:9px;">
          <button class="btn btn-primary btn-block" style="background:var(--green);" onclick="manualTrade('buy')">🟢 شراء</button>
          <button class="btn btn-primary btn-block" style="background:var(--red);" onclick="manualTrade('sell')">🔴 بيع</button>
        </div>
      </div>

      <!-- المراكز -->
      <div id="sub-positions" class="hidden">
        <button class="btn btn-ghost" onclick="loadPositions()" style="margin-bottom:12px;">🔄 تحديث</button>
        <div id="pos-list"><div class="empty"><div class="empty-ico">📭</div>اضغط تحديث لعرض مراكزك</div></div>
      </div>

      <!-- الإعدادات -->
      <div id="sub-settings" class="hidden">
        <div class="grid-2">
          <div class="field"><label>API Key</label><input class="input" id="c-key" type="password" placeholder="PK…"></div>
          <div class="field"><label>Secret Key</label><input class="input" id="c-secret" type="password" placeholder="••••"></div>
          <div class="field"><label>حجم المركز الأقصى ($)</label><input class="input" id="c-max" type="number" value="500"></div>
          <div class="field"><label>حد الخسارة اليومي (%)</label><input class="input" id="c-loss" type="number" value="3" step="0.5"></div>
          <div class="field"><label>أقصى صفقات يومياً</label><input class="input" id="c-maxtrades" type="number" value="5"></div>
          <div class="field"><label>أقل ثقة للشراء (%)</label><input class="input" id="c-minconf" type="number" value="70"></div>
        </div>
        <div class="switch-row"><span class="switch-txt">تفعيل التداول الآلي</span>
          <label class="switch"><input type="checkbox" id="c-enabled"><span class="switch-slider"></span></label></div>
        <div class="switch-row"><span class="switch-txt">شراء تلقائي عند إشارة BUY</span>
          <label class="switch"><input type="checkbox" id="c-autobuy"><span class="switch-slider"></span></label></div>
        <div class="field" style="margin-top:11px;"><label>نوع أمر الشراء</label>
          <select class="input" id="c-ordertype">
            <option value="market">سوق (تنفيذ مضمون فوراً)</option>
            <option value="limit">محدد (سعر الدخول المقترح)</option>
          </select></div>
        <div class="switch-row"><span class="switch-txt">بيع ذكي تلقائي (جزئي + متحرك)</span>
          <label class="switch"><input type="checkbox" id="c-autosell" checked><span class="switch-slider"></span></label></div>
        <button class="btn btn-primary btn-block btn-lg" style="margin-top:14px;" onclick="saveConfig()">💾 حفظ الإعدادات</button>
        <button class="btn btn-ghost btn-block" style="margin-top:8px;" onclick="testConn()">🔌 اختبار الاتصال</button>
        <div class="strat-box">
          <div class="strat-h">📋 استراتيجية البيع الذكية</div>
          • ربح <b>+2%</b> ← بيع <b>40%</b> (ربح مبكر مضمون)<br>
          • ربح <b>+4%</b> ← بيع <b>30%</b> إضافية<br>
          • الباقي <b>30%</b> ← وقف متحرك <b>1.5%</b> تحت القمة<br>
          • نزل قبل الربح ← وقف خسارة 🛡<br>
          • خسارة يومية تجاوزت الحد ← إغلاق الكل وتوقف
        </div>
      </div>

      <!-- السجل -->
      <div id="sub-log" class="hidden">
        <!-- إحصائيات سريعة -->
        <div id="log-summary" style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:14px;"></div>
        <!-- أزرار -->
        <div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap;">
          <button class="btn btn-primary" onclick="loadLog()" style="flex:1;">🔄 تحديث</button>
          <a href="/api/export_excel" class="btn btn-ghost" style="flex:1;text-align:center;text-decoration:none;">📥 تصدير Excel</a>
        </div>
        <!-- جدول الصفقات -->
        <div id="log-list"><div class="empty"><div class="empty-ico">📜</div>لا توجد صفقات بعد</div></div>
      </div>
    </div>
  </div>

  <!-- إضافة سهم -->
  <div class="add-zone">
    <div class="add-title">🔍 تحليل سهم محدد</div>
    <div class="add-row">
      <input class="input" id="q-code" placeholder="الرمز (AAPL أو BTC-USD)">
      <button class="btn btn-primary" onclick="analyzeOne()">تحليل فوري</button>
    </div>
    <div id="instant" style="margin-top:14px;"></div>
  </div>

  <!-- المسح -->
  <div class="scan-zone">
    <button class="scan-btn" id="scan-btn" onclick="startScan()">🔍 مسح السوق الأمريكي</button>
    <div class="scan-meta" id="scan-meta"></div>
  </div>

  <div id="fav-section"></div>
  <div id="tally"></div>
  <div id="signals"></div>

  <div class="foot">جلال رادار برو · مؤسس على Alpaca · للأغراض التعليمية فقط · ليس توصية</div>
</div>
"""



# ════════════════════════════════════════════════════════════
# دالة بناء الصفحة
# ════════════════════════════════════════════════════════════
APP_JS = r"""var market = 'us';

function setMarket(m) {
  market = m;
  document.getElementById('mt-us').classList.toggle('on', m==='us');
  document.getElementById('mt-crypto').classList.toggle('on', m==='crypto');
  document.getElementById('scan-btn').innerHTML = m==='us' ? '🔍 مسح السوق الأمريكي' : '🔍 مسح الكريبتو';
  loadResults();
  loadFavorites();
}

function setSub(name, btn) {
  ['account','trade','positions','settings','log'].forEach(function(s){
    document.getElementById('sub-'+s).classList.add('hidden');
  });
  document.getElementById('sub-'+name).classList.remove('hidden');
  document.querySelectorAll('.sub-tab').forEach(function(t){ t.classList.remove('on'); });
  btn.classList.add('on');
  if (name==='account') loadAccount();
  if (name==='positions') loadPositions();
  if (name==='settings') loadConfig();
  if (name==='log') loadLog();
}

function toast(msg) {
  var t = document.getElementById('toast');
  t.textContent = msg; t.classList.add('show');
  setTimeout(function(){ t.classList.remove('show'); }, 2400);
}

// ── الساعة وحالة السوق ──
function loadClock() {
  fetch('/api/clock').then(function(r){return r.json();}).then(function(d){
    var dot = document.getElementById('clock-dot');
    var txt = document.getElementById('clock-txt');
    var info = document.getElementById('market-info');
    if (d.ok && d.open) {
      dot.classList.add('open');
      txt.textContent = 'السوق الأمريكي مفتوح';
      if (info) info.innerHTML = '🟢 السوق مفتوح الآن — التداول الآلي نشط';
    } else if (d.ok) {
      dot.classList.remove('open');
      txt.textContent = 'السوق مغلق';
      var no = d.next_open ? (' · يفتح: ' + d.next_open.slice(5,16).replace('T',' ')) : '';
      if (info) info.innerHTML = '🔴 السوق مغلق' + no;
    } else {
      txt.textContent = 'أدخل مفاتيح Alpaca';
      if (info) info.innerHTML = '⚙️ أدخل مفاتيح Alpaca في الإعدادات لمعرفة حالة السوق';
    }
  }).catch(function(){});
}

// ── الحساب ──
function loadAccount() {
  loadClock();
  loadStats();
  fetch('/api/account').then(function(r){return r.json();}).then(function(d){
    var conn = document.getElementById('conn');
    if (d.ok) {
      document.getElementById('a-equity').textContent = '$'+(+d.equity).toFixed(0);
      document.getElementById('a-cash').textContent = '$'+(+d.cash).toFixed(0);
      document.getElementById('a-bp').textContent = '$'+(+d.buying_power).toFixed(0);
      var pnl = +d.pnl;
      var pe = document.getElementById('a-pnl');
      pe.textContent = (pnl>=0?'+$':'-$')+Math.abs(pnl).toFixed(2);
      pe.className = 'acct-val ' + (pnl>=0?'pos':'neg');
      conn.classList.add('live');
      document.getElementById('conn-txt').textContent = 'متصل';
    } else {
      conn.classList.remove('live');
      document.getElementById('conn-txt').textContent = 'غير متصل';
    }
  }).catch(function(){});
}

function loadStats() {
  fetch('/api/stats').then(function(r){return r.json();}).then(function(d){
    document.getElementById('s-total').textContent = d.total||0;
    document.getElementById('s-wins').textContent = d.wins||0;
    document.getElementById('s-losses').textContent = d.losses||0;
    document.getElementById('s-wr').textContent = (d.win_rate||0)+'%';
    var pnl = d.total_pnl||0;
    document.getElementById('s-pnl').textContent = (pnl>=0?'+$':'-$')+Math.abs(pnl).toFixed(2);
    document.getElementById('s-today').textContent = d.today||0;
    document.getElementById('s-max').textContent = d.max_daily||5;
  }).catch(function(){});
}

// ── الإعدادات ──
function loadConfig() {
  fetch('/api/config').then(function(r){return r.json();}).then(function(d){
    document.getElementById('c-key').value = d.key||'';
    document.getElementById('c-secret').value = d.secret||'';
    document.getElementById('c-max').value = d.max_position_usd||500;
    document.getElementById('c-loss').value = d.daily_loss_pct||3;
    document.getElementById('c-maxtrades').value = d.max_daily_trades||5;
    document.getElementById('c-minconf').value = d.min_confidence||70;
    document.getElementById('c-enabled').checked = d.enabled||false;
    document.getElementById('c-autobuy').checked = d.auto_buy||false;
    document.getElementById('c-autosell').checked = d.auto_sell!==false;
    if(document.getElementById('c-ordertype')) document.getElementById('c-ordertype').value = d.buy_order_type||'market';
  }).catch(function(){});
}

function saveConfig() {
  var cfg = {
    key: document.getElementById('c-key').value.trim(),
    secret: document.getElementById('c-secret').value.trim(),
    max_position_usd: +document.getElementById('c-max').value||500,
    daily_loss_pct: +document.getElementById('c-loss').value||3,
    max_daily_trades: +document.getElementById('c-maxtrades').value||5,
    min_confidence: +document.getElementById('c-minconf').value||70,
    enabled: document.getElementById('c-enabled').checked,
    auto_buy: document.getElementById('c-autobuy').checked,
    auto_sell: document.getElementById('c-autosell').checked,
    buy_order_type: document.getElementById('c-ordertype') ? document.getElementById('c-ordertype').value : 'market'
  };
  fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(cfg)})
    .then(function(r){return r.json();}).then(function(d){
      if(d.ok){ toast('✅ تم حفظ الإعدادات'); loadAccount(); }
    });
}

function testConn() {
  toast('جاري الاختبار…');
  saveConfig();
  setTimeout(loadAccount, 1000);
}

// ── تداول يدوي ──
document.addEventListener('change', function(e){
  if (e.target && e.target.id==='t-type') {
    document.getElementById('t-limit-wrap').style.display = e.target.value==='limit'?'block':'none';
  }
});

function manualTrade(side) {
  var sym = document.getElementById('t-sym').value.trim().toUpperCase();
  var qty = +document.getElementById('t-qty').value||1;
  var type = document.getElementById('t-type').value;
  if(!sym){ toast('أدخل الرمز'); return; }
  var body = {symbol:sym, qty:qty, side:side, order_type:type, is_crypto:market==='crypto'};
  if(type==='limit') body.limit_price = +document.getElementById('t-limit').value;
  toast('جاري إرسال الأمر…');
  fetch('/api/trade',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
    .then(function(r){return r.json();}).then(function(d){
      if(d.ok) toast((side==='buy'?'✅ شراء ':'✅ بيع ')+sym+' ×'+qty);
      else toast('❌ '+(d.msg||'فشل'));
    });
}

// ── المراكز ──
function loadPositions() {
  fetch('/api/positions').then(function(r){return r.json();}).then(function(d){
    var el = document.getElementById('pos-list');
    if(!d.ok || !d.positions.length){
      el.innerHTML = '<div class="empty"><div class="empty-ico">📭</div>لا توجد مراكز مفتوحة</div>';
      return;
    }
    el.innerHTML = d.positions.map(function(p){
      var cls = p.pnl_pct>=0?'pos':'neg';
      var sign = p.pnl_pct>=0?'+':'';
      var color = p.pnl_pct>=0?'var(--green)':'var(--red)';
      return '<div class="pos"><div><div class="pos-sym">'+p.symbol+'</div>'+
        '<div class="pos-info">'+p.qty+' · دخول $'+p.entry+' · الآن $'+p.current+'</div></div>'+
        '<div class="pos-pnl"><div class="pos-pnl-pct" style="color:'+color+';">'+sign+p.pnl_pct+'%</div>'+
        '<div style="font-size:11px;color:var(--ink-3);">$'+p.pnl+'</div>'+
        '<button class="pos-close" onclick="closePos(this.dataset.s)" data-s="'+p.symbol+'">إغلاق</button></div></div>';
    }).join('');
  });
}

function closePos(sym) {
  if(!confirm('إغلاق مركز '+sym+'؟')) return;
  fetch('/api/close',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({symbol:sym})})
    .then(function(r){return r.json();}).then(function(d){
      if(d.ok){ toast('✅ تم إغلاق '+sym); loadPositions(); }
      else toast('❌ '+(d.msg||'فشل'));
    });
}

// ── السجل التفصيلي ──
function loadLog() {
  fetch('/api/trades_detailed').then(function(r){return r.json();}).then(function(trades){
    var el = document.getElementById('log-list');
    var sumEl = document.getElementById('log-summary');

    if(!trades.length){
      el.innerHTML='<div class="empty"><div class="empty-ico">📜</div>لا توجد صفقات بعد</div>';
      sumEl.innerHTML='';
      return;
    }

    // ── إحصائيات ──
    var closed = trades.filter(function(t){ return t.pnl !== null; });
    var wins   = closed.filter(function(t){ return t.pnl > 0; });
    var losses = closed.filter(function(t){ return t.pnl < 0; });
    var totalPnl = closed.reduce(function(a,t){ return a + (t.pnl||0); }, 0);
    var wr = closed.length ? Math.round(wins.length/closed.length*100) : 0;
    var avgDur = closed.length ? Math.round(closed.reduce(function(a,t){return a+(t.duration_min||0);},0)/closed.length) : 0;

    sumEl.innerHTML = [
      ['📊 إجمالي', trades.length + ' صفقة'],
      ['✅ ربح', wins.length + ' (' + wr + '%)'],
      ['❌ خسارة', losses.length],
      ['💰 صافي', (totalPnl>=0?'+':'')+totalPnl.toFixed(2)+'$'],
    ].map(function(s){
      var col = s[0].includes('💰') ? (totalPnl>=0?'var(--green)':'var(--red)') : 'var(--ink)';
      return '<div style="background:var(--surface-2);border-radius:10px;padding:10px;text-align:center;">'
        +'<div style="font-size:11px;color:var(--ink-2);margin-bottom:3px;">'+s[0]+'</div>'
        +'<div style="font-weight:700;color:'+col+';">'+s[1]+'</div></div>';
    }).join('');

    // ── جدول الصفقات ──
    var rows = trades.map(function(t){
      var isWin  = t.pnl > 0;
      var isOpen = t.result === '🔵 مفتوح';
      var bg = isOpen ? 'var(--surface-2)' : (isWin ? 'var(--green-soft)' : 'var(--red-soft)');
      var pnlStr = t.pnl !== null
        ? '<span style="color:'+(isWin?'var(--green)':'var(--red)');font-weight:700;">'
          +(isWin?'+':'')+t.pnl.toFixed(2)+'$ ('+(isWin?'+':'')+t.pnl_pct+'%)</span>'
        : '<span style="color:var(--ink-3);">—</span>';
      var dur = t.duration_min !== null
        ? (t.duration_min >= 60
            ? Math.floor(t.duration_min/60)+'س '+( t.duration_min%60)+'د'
            : t.duration_min+'د')
        : '—';
      return '<div style="background:'+bg+';border-radius:10px;padding:12px 14px;margin-bottom:8px;">'
        +'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
        +'<span style="font-weight:700;font-size:15px;">'+t.result+' '+t.symbol+'</span>'
        +'<span style="font-size:12px;color:var(--ink-2);">'+(t.source==='auto'?'🤖':'👤')+' '+t.entry_time+'</span>'
        +'</div>'
        +'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;font-size:12px;color:var(--ink-2);">'
        +'<div>📥 دخول<br><b style="color:var(--ink);">'+(t.entry_price||'—')+'$</b></div>'
        +'<div>📤 خروج<br><b style="color:var(--ink);">'+(t.exit_price||'—')+'$</b></div>'
        +'<div>⏱ مدة<br><b style="color:var(--ink);">'+dur+'</b></div>'
        +'</div>'
        +'<div style="display:flex;justify-content:space-between;align-items:center;margin-top:8px;font-size:12px;">'
        +'<span style="color:var(--ink-3);">'+( t.reason||'')+'</span>'
        +pnlStr
        +'</div>'
        +'</div>';
    }).join('');

    el.innerHTML = rows;
  });
}

// ── المسح ──
var scanPoll = null;
function startScan() {
  document.getElementById('loader').classList.add('show');
  fetch('/scan?market='+market).then(function(r){return r.json();}).then(function(){
    scanPoll = setInterval(pollScan, 1200);
  });
}
function pollScan() {
  fetch('/scan_status?market='+market).then(function(r){return r.json();}).then(function(d){
    document.getElementById('loader-sub').textContent = 'تم '+(d.progress||0)+'%';
    if(d.status==='done'){
      clearInterval(scanPoll);
      document.getElementById('loader').classList.remove('show');
      loadResults();
    }
  });
}

function loadResults() {
  fetch('/results?market='+market).then(function(r){return r.json();}).then(function(d){
    var data = d.data||[];
    document.getElementById('scan-meta').textContent = d.last ? ('آخر مسح: '+d.last+' · رُشّح من '+d.scanned) : '';
    var buys = data.filter(function(s){return s.verdict==='BUY';}).length;
    var conds = data.filter(function(s){return s.verdict==='BUY_COND';}).length;
    if(data.length){
      document.getElementById('tally').innerHTML =
        '<div class="tally"><div class="tally-cell g"><div class="tally-num">'+buys+'</div><div class="tally-lbl">شراء</div></div>'+
        '<div class="tally-cell a"><div class="tally-num">'+conds+'</div><div class="tally-lbl">مشروط</div></div>'+
        '<div class="tally-cell b"><div class="tally-num">'+data.length+'</div><div class="tally-lbl">مرشّح</div></div></div>';
    } else { document.getElementById('tally').innerHTML=''; }
    document.getElementById('signals').innerHTML = data.length ?
      data.map(renderSig).join('') :
      '<div class="empty"><div class="empty-ico">📡</div>اضغط مسح لتحليل السوق</div>';
  });
}

function renderSig(s) {
  var vClass = s.verdict==='BUY'?'buy':(s.verdict==='BUY_COND'?'cond':'watch');
  var vIco = s.verdict==='BUY'?'🟢':(s.verdict==='BUY_COND'?'🟡':'⏳');
  var vLabel = s.verdict==='BUY'?'شراء':(s.verdict==='BUY_COND'?'شراء مشروط':'مراقبة');
  var live = s.live ? '<div class="sig-px-live">مباشر Alpaca</div>' : '';
  var tradeBtn = (s.verdict==='BUY') ?
    '<button class="sig-trade-btn" onclick=\'buyFromCard('+JSON.stringify(s).replace(/'/g,"&#39;")+')\'>⚡ تداول</button>' : '';
  return '<div class="sig">'+
    '<div class="sig-verdict '+vClass+'"><div class="verdict-main"><span class="verdict-ico">'+vIco+'</span>'+
    '<div><div class="verdict-label">'+vLabel+'</div><div class="verdict-sub">ثقة '+s.confidence+'% · '+s.trend+'</div></div></div>'+
    '<div class="verdict-score"><div class="verdict-score-num">'+s.score+'/20</div><div class="verdict-score-lbl">JRF</div></div></div>'+
    '<div class="sig-body"><div class="sig-top"><span class="sig-medal">'+(s.medal||'')+'</span>'+
    '<div class="sig-id"><div class="sig-name">'+s.name+'<span class="sig-code">'+s.code+'</span>'+'<span class="fav-star'+(s.is_fav?' on':'')+'" onclick="event.stopPropagation();toggleFav(\''+s.code+'\')">'+(s.is_fav?'★':'☆')+'</span></div>'+
    '<div class="sig-meta"><span class="mb">RSI '+s.rsi+'</span><span class="mb">ADX '+s.adx+'</span><span class="mb">⏱ '+(s.eta||'—')+'</span></div></div>'+
    '<div class="sig-px"><div class="sig-px-main">$'+s.price+'</div><div class="sig-px-sar">'+s.price_sar+' ر.س</div>'+live+'</div></div>'+
    '<div class="sig-bar"><div class="sig-bar-fill" style="width:'+s.confidence+'%"></div></div>'+
    '<div class="sig-tags"><span class="tag">R:R <b>'+s.rr+'</b></span>'+
    '<span class="tag">🎯 <b>$'+s.t1+'</b> +'+s.ptp+'%</span>'+
    '<span class="tag">🛡 <b>$'+s.sl+'</b> -'+s.psl+'%</span>'+tradeBtn+'</div>'+
    '<div class="sig-levels"><div class="lvl entry"><div class="lvl-lbl">دخول</div><div class="lvl-val">$'+s.lb+'</div></div>'+
    '<div class="lvl tp"><div class="lvl-lbl">هدف 1</div><div class="lvl-val">$'+s.t1+'</div><div class="lvl-pct">+'+s.ptp+'%</div></div>'+
    '<div class="lvl tp"><div class="lvl-lbl">هدف 2</div><div class="lvl-val">$'+s.t2+'</div></div>'+
    '<div class="lvl sl"><div class="lvl-lbl">وقف</div><div class="lvl-val">$'+s.sl+'</div><div class="lvl-pct">-'+s.psl+'%</div></div></div></div></div>';
}

function buyFromCard(s) {
  var qty = Math.max(1, Math.floor(500/s.price));
  if(!confirm('⚡ شراء '+s.name+' ('+s.code+')\nالسعر: $'+s.price+'\nالكمية: ~'+qty+'\nتأكيد؟')) return;
  fetch('/api/buy_signal',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(s)})
    .then(function(r){return r.json();}).then(function(d){
      if(d.ok){ toast('✅ تم الشراء: '+s.code+' ×'+d.qty); loadStats(); }
      else toast('❌ '+(d.msg||'فشل'));
    });
}

// ── تحليل فوري ──
function toggleFav(code) {
  fetch('/api/favorites/toggle',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code:code,market:market})})
    .then(function(r){return r.json();}).then(function(d){
      if(d.ok){ toast(d.added?'⭐ أُضيف للمفضلة':'تمت الإزالة'); loadResults(); loadFavorites(); }
    });
}

function loadFavorites() {
  fetch('/api/favorites/analyze?market='+market).then(function(r){return r.json();}).then(function(d){
    var box = document.getElementById('fav-section');
    if(!box) return;
    if(!d.data || !d.data.length){ box.innerHTML=''; return; }
    box.innerHTML = '<div class="fav-header">⭐ مفضلتك ('+d.data.length+') — تتحلل كل مسح</div>' +
      d.data.map(renderSig).join('');
  });
}

function analyzeOne() {
  var code = document.getElementById('q-code').value.trim().toUpperCase();
  if(!code){ toast('أدخل الرمز'); return; }
  var box = document.getElementById('instant');
  box.innerHTML = '<div style="text-align:center;color:var(--ink-3);padding:14px;">جاري تحليل '+code+'…</div>';
  fetch('/analyze_one',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code:code,is_crypto:market==='crypto'})})
    .then(function(r){return r.json();}).then(function(d){
      if(!d.ok){ box.innerHTML='<div style="color:var(--red);padding:10px;">❌ '+(d.msg||'تعذّر')+'</div>'; return; }
      box.innerHTML = renderSig(d.result);
    });
}

// ── تشغيل ──
document.addEventListener('DOMContentLoaded', function(){
  loadConfig();
  setTimeout(loadAccount, 400);
  loadResults();
  loadFavorites();
  setInterval(loadClock, 60000);
});
"""

def render_page():
    return f"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>جلال رادار برو</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;800&display=swap" rel="stylesheet">
<style>{PAGE_CSS}</style>
</head>
<body>
{PAGE_BODY}
<script>{APP_JS}</script>
</body>
</html>"""


# ════════════════════════════════════════════════════════════
# Part 6: Flask Routes
# ════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_page()

@app.route("/scan")
def scan():
    market = request.args.get("market","us")
    if scan_state[market]["status"]=="scanning":
        return jsonify({"status":"already_running"})
    t = threading.Thread(target=run_scan, args=(market,), daemon=True)
    t.start()
    return jsonify({"status":"started"})

@app.route("/scan_status")
def scan_status():
    market = request.args.get("market","us")
    return jsonify({
        "status": scan_state[market]["status"],
        "progress": scan_state[market]["progress"],
        "scanned": scan_state[market]["scanned"],
    })

@app.route("/results")
def results():
    market = request.args.get("market","us")
    return jsonify({
        "data": scan_state[market]["data"],
        "last": scan_state[market]["last"],
        "scanned": scan_state[market]["scanned"],
    })

@app.route("/analyze_one", methods=["POST"])
def analyze_one():
    d = request.get_json()
    code = d.get("code","").strip().upper()
    name = d.get("name","").strip() or code
    is_crypto = d.get("is_crypto", False)
    if not code: return jsonify({"ok":False,"msg":"أدخل الرمز"})
    live = api_last_price(code, is_crypto) if load_cfg().get("key") else None
    r = analyze_symbol(code, name, is_crypto, live)
    if not r: return jsonify({"ok":False,"msg":f"تعذّر تحليل {code}"})
    return jsonify({"ok":True,"result":r})

# ── Alpaca ──
@app.route("/api/config", methods=["GET","POST"])
def api_config():
    if request.method=="POST":
        d = request.get_json()
        cfg = load_cfg()
        for k in d:
            if k in cfg: cfg[k] = d[k]
        save_cfg(cfg)
        return jsonify({"ok":True})
    cfg = load_cfg()
    safe = {k:v for k,v in cfg.items() if k not in ("key","secret")}
    safe["has_key"] = bool(cfg.get("key"))
    safe["key"] = cfg.get("key","")
    safe["secret"] = cfg.get("secret","")
    return jsonify(safe)

@app.route("/api/account")
def account():
    acc = api_account()
    if "error" in acc:
        return jsonify({"ok":False,"msg":acc["error"]})
    return jsonify({
        "ok":True,
        "equity":acc.get("equity","0"),
        "cash":acc.get("cash","0"),
        "buying_power":acc.get("buying_power","0"),
        "pnl":acc.get("unrealized_pl","0") if acc.get("unrealized_pl") else
              str(float(acc.get("equity",0))-float(acc.get("last_equity",0))),
    })

@app.route("/api/clock")
def clock():
    c = api_clock()
    if "error" in c: return jsonify({"ok":False,"open":False})
    return jsonify({"ok":True,"open":c.get("is_open",False),
                    "next_open":c.get("next_open",""),"next_close":c.get("next_close","")})

@app.route("/api/positions")
def positions():
    pos = api_positions()
    out = []
    for p in pos:
        try:
            out.append({
                "symbol":p.get("symbol","").replace("/USD","-USD"),
                "qty":p.get("qty","0"),
                "entry":round(float(p.get("avg_entry_price",0)),2),
                "current":round(float(p.get("current_price",0)),2),
                "pnl":round(float(p.get("unrealized_pl",0)),2),
                "pnl_pct":round(float(p.get("unrealized_plpc",0))*100,2),
            })
        except: pass
    return jsonify({"ok":True,"positions":out})

@app.route("/api/trade", methods=["POST"])
def trade():
    d = request.get_json()
    symbol = d.get("symbol","").upper()
    side = d.get("side","buy")
    qty = d.get("qty",1)
    is_crypto = d.get("is_crypto",False)
    if not symbol: return jsonify({"ok":False,"msg":"أدخل الرمز"})
    cfg = load_cfg()
    if not cfg.get("key"): return jsonify({"ok":False,"msg":"أدخل مفاتيح Alpaca"})
    otype = d.get("order_type","market")
    limit = d.get("limit_price")
    tif = "gtc" if is_crypto else "day"
    r = api_place_order(symbol, qty, side, otype, limit, tif, is_crypto)
    if "error" in r: return jsonify({"ok":False,"msg":r["error"]})
    add_trade({"symbol":symbol,"side":side,"qty":qty,"order_type":otype,
               "order_id":r.get("id",""),"status":r.get("status",""),
               "time":datetime.now().strftime("%Y-%m-%d %H:%M"),"source":"manual"})
    return jsonify({"ok":True,"order_id":r.get("id",""),"status":r.get("status","")})

@app.route("/api/close", methods=["POST"])
def close():
    symbol = request.get_json().get("symbol","").upper()
    r = api_close_position(symbol)
    if "error" in r: return jsonify({"ok":False,"msg":r["error"]})
    return jsonify({"ok":True})

@app.route("/api/buy_signal", methods=["POST"])
def buy_signal():
    """شراء من كارد الرادar"""
    sig = request.get_json()
    res = safe_buy(sig, "manual")
    return jsonify(res)

@app.route("/api/favorites", methods=["GET"])
def get_favorites():
    return jsonify(load_favs())

@app.route("/api/favorites/toggle", methods=["POST"])
def toggle_favorite():
    d = request.get_json()
    code = d.get("code","").strip().upper()
    market = d.get("market","us")
    favs = load_favs()
    lst = favs.get(market, [])
    if code in lst:
        lst.remove(code); added = False
    else:
        lst.append(code); added = True
    favs[market] = lst
    save_favs(favs)
    return jsonify({"ok":True,"added":added})

@app.route("/api/favorites/analyze")
def analyze_favorites():
    """يحلّل كل المفضلة ويرجّع نتائجها"""
    market = request.args.get("market","us")
    favs = load_favs().get(market, [])
    if not favs: return jsonify({"data":[]})
    is_crypto = (market=="crypto")
    stocks = US_STOCKS if market=="us" else CRYPTO
    out = []
    for code in favs:
        name = stocks.get(code, code)
        live = api_last_price(code, is_crypto) if load_cfg().get("key") else None
        r = analyze_symbol(code, name, is_crypto, live)
        if r:
            r["is_fav"] = True
            out.append(r)
    out.sort(key=lambda x:(-x["score"],-x["confidence"]))
    return jsonify({"data":out})

@app.route("/api/trades")
def trades_log():
    return jsonify(load_trades())

@app.route("/api/stats")
def stats():
    trades = load_trades()
    today = datetime.now().strftime("%Y-%m-%d")
    buys = [t for t in trades if t.get("side")=="buy"]
    sells = [t for t in trades if t.get("side")=="sell"]
    wins = [t for t in sells if float(t.get("pnl",0))>0]
    losses = [t for t in sells if float(t.get("pnl",0))<=0]
    total_pnl = sum(float(t.get("pnl",0)) for t in sells)
    wr = round(len(wins)/len(sells)*100,1) if sells else 0
    today_buys = [t for t in buys if t.get("time","").startswith(today) and t.get("source")=="auto"]
    cfg = load_cfg()
    return jsonify({
        "total":len(buys),"closed":len(sells),"wins":len(wins),"losses":len(losses),
        "win_rate":wr,"total_pnl":round(total_pnl,2),
        "today":len(today_buys),"max_daily":int(cfg.get("max_daily_trades",5)),
    })

@app.route("/api/trades_detailed")
def trades_detailed():
    """صفقات مقترنة: كل شراء مع بيعه ونتيجته"""
    trades = load_trades()
    buys = [t for t in trades if t.get("side")=="buy"]
    sells = [t for t in trades if t.get("side")=="sell"]
    paired = []
    for b in buys:
        sym = b.get("symbol","")
        buy_time = b.get("time","")
        # نبحث عن أقرب بيع لنفس السهم بعد وقت الشراء
        matching_sells = [s for s in sells if s.get("symbol")==sym and s.get("time","") >= buy_time]
        matching_sells.sort(key=lambda x: x.get("time",""))
        entry_price = float(b.get("entry_price") or b.get("price") or 0)
        if matching_sells:
            s = matching_sells[0]
            exit_price = float(s.get("exit_price") or 0)
            pnl = float(s.get("pnl") or 0)
            pnl_pct = round((exit_price - entry_price) / entry_price * 100, 2) if entry_price > 0 and exit_price > 0 else 0
            # حساب المدة
            try:
                t_in  = datetime.strptime(buy_time, "%Y-%m-%d %H:%M")
                t_out = datetime.strptime(s.get("time",""), "%Y-%m-%d %H:%M")
                duration_min = int((t_out - t_in).total_seconds() / 60)
            except: duration_min = 0
            paired.append({
                "symbol": sym, "name": b.get("name",""),
                "qty": b.get("qty", 0),
                "entry_price": entry_price, "exit_price": exit_price,
                "entry_time": buy_time, "exit_time": s.get("time",""),
                "duration_min": duration_min,
                "pnl": pnl, "pnl_pct": pnl_pct,
                "reason": s.get("reason",""),
                "score": b.get("score",0), "confidence": b.get("confidence",0),
                "source": b.get("source",""),
                "result": "✅ ربح" if pnl > 0 else ("❌ خسارة" if pnl < 0 else "➖ تعادل"),
            })
        else:
            # مركز مفتوح بعد
            paired.append({
                "symbol": sym, "name": b.get("name",""),
                "qty": b.get("qty", 0),
                "entry_price": entry_price, "exit_price": None,
                "entry_time": buy_time, "exit_time": None,
                "duration_min": None,
                "pnl": None, "pnl_pct": None,
                "reason": "مفتوح",
                "score": b.get("score",0), "confidence": b.get("confidence",0),
                "source": b.get("source",""),
                "result": "🔵 مفتوح",
            })
    paired.sort(key=lambda x: x["entry_time"], reverse=True)
    return jsonify(paired)

@app.route("/api/export_excel")
def export_excel():
    """تصدير سجل الصفقات بصيغة CSV متوافقة مع Excel"""
    trades = load_trades()
    buys = [t for t in trades if t.get("side")=="buy"]
    sells = [t for t in trades if t.get("side")=="sell"]
    rows = []
    for b in buys:
        sym = b.get("symbol","")
        buy_time = b.get("time","")
        matching_sells = [s for s in sells if s.get("symbol")==sym and s.get("time","") >= buy_time]
        matching_sells.sort(key=lambda x: x.get("time",""))
        entry_price = float(b.get("entry_price") or b.get("price") or 0)
        if matching_sells:
            s = matching_sells[0]
            exit_price = float(s.get("exit_price") or 0)
            pnl = float(s.get("pnl") or 0)
            pnl_pct = round((exit_price - entry_price) / entry_price * 100, 2) if entry_price > 0 and exit_price > 0 else 0
            try:
                t_in  = datetime.strptime(buy_time, "%Y-%m-%d %H:%M")
                t_out = datetime.strptime(s.get("time",""), "%Y-%m-%d %H:%M")
                duration_min = int((t_out - t_in).total_seconds() / 60)
            except: duration_min = 0
            rows.append([
                sym, b.get("name",""), b.get("qty",0),
                entry_price, exit_price, pnl, pnl_pct,
                buy_time, s.get("time",""), duration_min,
                s.get("reason",""), b.get("score",0), b.get("confidence",0),
                "ربح" if pnl>0 else ("خسارة" if pnl<0 else "تعادل"),
                b.get("source","")
            ])
        else:
            rows.append([
                sym, b.get("name",""), b.get("qty",0),
                entry_price, "", "", "",
                buy_time, "", "",
                "مفتوح", b.get("score",0), b.get("confidence",0),
                "مفتوح", b.get("source","")
            ])

    output = io.StringIO()
    output.write('\ufeff')  # BOM عشان Excel يقرأ العربي صح
    writer = csv.writer(output)
    writer.writerow([
        "الرمز","الاسم","الكمية",
        "سعر الدخول","سعر الخروج","الربح/الخسارة $","الربح/الخسارة %",
        "وقت الدخول","وقت الخروج","المدة (دقيقة)",
        "سبب الخروج","النقاط","الثقة %",
        "النتيجة","المصدر"
    ])
    writer.writerows(rows)
    csv_bytes = output.getvalue().encode("utf-8-sig")
    from flask import Response
    return Response(
        csv_bytes,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=jalal_radar_trades.csv"}
    )




# ════════════════════════════════════════════════════════════
# التشغيل
# ════════════════════════════════════════════════════════════
# نشغّل المجدول على مستوى الموديول (يشتغل مع gunicorn و python)
try:
    start_scheduler()
except Exception as _e:
    print(f"تحذير: المجدول لم يبدأ: {_e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("="*54)
    print("  ⚡ جلال رادار برو — Alpaca-Native")
    print(f"  أمريكي: {len(US_STOCKS)} سهم · كريبتو: {len(CRYPTO)} عملة")
    print("="*54)
    app.run(host="0.0.0.0", port=port, debug=False)
