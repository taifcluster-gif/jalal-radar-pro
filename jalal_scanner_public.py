# -*- coding: utf-8 -*-
"""
jalal_scanner_public1.py  —  نسخة معدّلة (v2)
════════════════════════════════════════════════════════════
ماسح عام لترشيح الأسهم السعودية (تاسي) والأمريكية والعملات الرقمية،
مع وضع "خشاش سريع" منفصل بمعايير وأهداف خاصة.

⚠️ صفر اتصال بأي حساب تداول (لا Alpaca ولا غيره) — تحليل وعرض فقط.
بيانات yfinance العامة (+ سهمك اختيارياً لتاسي بمتغير SAHMK_KEY_PUBLIC).

الاعتماديات: flask, yfinance, pandas, numpy
متغيرات بيئة اختيارية:
  DATA_DIR            مجلد حفظ المفضلة وسجل الأداء (الافتراضي: مجلد التشغيل)
  SHARIA_US_FILE      مسار ملف sharia_compliant_us.json (الافتراضي: بجانب هذا الملف)
  SAHMK_KEY_PUBLIC    مفتاح سهمك لأسعار تاسي اللحظية (اختياري)

كل العتبات في TREND_CFG و JUNK_CFG أسفل — عدّلها من هناك بدون ما تلمس المنطق.
"""
import os, json, time, threading, csv, io, re
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, request, Response

app = Flask(__name__)
USD_TO_SAR = 3.75
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.getcwd())

# ════════════════════════════════════════════════════════════
# العتبات (قابلة للتعديل) — خُفّفت عن النسخة السابقة
# ════════════════════════════════════════════════════════════
TREND_CFG = {
    "buy_score": 14,    # كان 15
    "cond_score": 11,   # كان 12
    "watch_score": 8,   # كان 9
    "buy_rr": 1.2,      # كان 1.3 (والآن rr يتغير فعلياً حسب المقاومة)
    "cond_rr": 1.0,
    "rsi_lo": 42, "rsi_hi": 72,   # كان 40-70
    "adx_min": 18,      # كان 20
    "vr_min": 1.1,      # كان 1.2
}
JUNK_CFG = {
    "price_min": 0.05, "price_max": 5.0,
    "min_vol_penny": 1_000_000,   # سهم تحت 1$  (كان 2 مليون)
    "min_vol_low": 300_000,       # سهم 1$-5$   (كان 500 ألف)
    "min_dollar_vol": 250_000,    # حد أدنى لقيمة التداول اليومية بالدولار
    "buy_score": 13, "cond_score": 9, "watch_score": 6,
    "vr_spike": 1.4, "vr_big": 2.5,
    "breakout_days": 10,
    "max_ext10": 1.0,        # ارتفع أكثر من 100% بآخر 10 أيام = ركض زايد
    "max_day_chg": 35.0,     # قفزة يومية أكبر من 35% = لا تطارد
    "risk_atr_min": 0.8, "risk_atr_max": 1.6,   # مدى الوقف بوحدات ATR
    "t1_r": 1.5, "t2_r": 3.0,                   # الأهداف كمضاعف للمخاطرة
}

# ════════════════════════════════════════════════════════════
# قوائم الأسهم
# ════════════════════════════════════════════════════════════
SA_STOCKS = {
    "2222":"أرامكو السعودية","1180":"الأهلي السعودي","1120":"الراجحي",
    "2010":"سابك","7010":"stc","4200":"الدريس",
    "1211":"معادن","1030":"الإستثمار","2380":"بترو رابغ",
    "2350":"كيان السعودية","3010":"أسمنت العربية","1010":"الرياض البنك",
    "2290":"ينساب","4001":"أسواق العثيم","2250":"المجموعة السعودية",
    "6010":"نادك","3020":"أسمنت اليمامة","4030":"البحري",
    "8010":"التعاونية","2020":"سابك للمغذيات الزراعية",
}
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
    "DUOL":"Duolingo","TTWO":"Take-Two","ODFL":"Old Dominion",
}
CRYPTO = {
    "BTC-USD":"Bitcoin","ETH-USD":"Ethereum","SOL-USD":"Solana",
    "AVAX-USD":"Avalanche","LINK-USD":"Chainlink","DOT-USD":"Polkadot",
    "LTC-USD":"Litecoin","UNI-USD":"Uniswap","AAVE-USD":"Aave",
    "BCH-USD":"Bitcoin Cash","XRP-USD":"Ripple","DOGE-USD":"Dogecoin",
}
# مجمع مرشحين للخشاش (مستقل عن US_STOCKS). تُفحص أسعارها وسيولتها وقت المسح؛
# أي رمز متوقف أو سعره فوق 5$ يُتجاهل تلقائياً. لتوسيع الكون: وضع "شرعي فقط"
# يمسح قائمة sharia_compliant_us.json كاملة (1680 رمز) ويفلترها بالسعر والسيولة.
JUNK_STOCKS = {
    "SAVA":"Cassava Sciences","SINT":"SINTX Technologies","ATER":"Aterian",
    "CENN":"Cenntro Electric","GSAT":"Globalstar","IMPP":"Imperial Petroleum",
    "AGEN":"Agenus","OCGN":"Ocugen","CTRM":"Castor Maritime","SHIP":"Seanergy Maritime",
    "NNDM":"Nano Dimension","BTBT":"Bit Digital","SOS":"SOS Limited","WKHS":"Workhorse",
    "CLOV":"Clover Health","SNDL":"SNDL Inc","TLRY":"Tilray","ACB":"Aurora Cannabis",
    "CGC":"Canopy Growth","GRWG":"GrowGeneration","PHUN":"Phunware","BBAI":"BigBear.ai",
    "VERB":"Verb Technology","HUT":"Hut 8","BITF":"Bitfarms","CIFR":"Cipher Mining",
    "GREE":"Greenidge","WULF":"TeraWulf","RIOT":"Riot Platforms","MARA":"MARA Holdings",
    "CLSK":"CleanSpark","UEC":"Uranium Energy","DNN":"Denison Mines","FCEL":"FuelCell Energy",
    "PLUG":"Plug Power","BLNK":"Blink Charging","CHPT":"ChargePoint","MVIS":"MicroVision",
    "SPCE":"Virgin Galactic","DNA":"Ginkgo Bioworks","VLD":"Velo3D","NIO":"NIO",
    "XPEV":"XPeng","LCID":"Lucid","RIG":"Transocean","AEMD":"Aethlon Medical",
    "ENVB":"Enveric Biosciences","NAK":"Northern Dynasty","OPEN":"Opendoor","BB":"BlackBerry",
    "NOK":"Nokia","AMC":"AMC Entertainment","FUBO":"fuboTV","GPRO":"GoPro","BARK":"BARK Inc",
    "LUMN":"Lumen","BNGO":"Bionano Genomics","ZOM":"Zomedica",
}

# ════════════════════════════════════════════════════════════
# المؤشرات الفنية
# ════════════════════════════════════════════════════════════
def _f(x, default=None):
    """رقم آمن: يرجّع default لو NaN/inf/غير رقم."""
    try:
        v = float(x)
        return default if (v != v or v in (float("inf"), float("-inf"))) else v
    except Exception:
        return default

def _ema(s, p): return s.ewm(span=p, adjust=False).mean()
def _sma(s, p): return s.rolling(p).mean()

def _rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    r = 100 - 100 / (1 + g / l.replace(0, np.nan))
    r = r.where(~((l == 0) & (g > 0)), 100.0)   # صعود بلا خسائر = 100 (مو NaN)
    r = r.where(~((l == 0) & (g == 0)), 50.0)
    return r

def _macd(s):
    ml = _ema(s, 12) - _ema(s, 26); sg = _ema(ml, 9); return ml, sg, ml - sg

def _adx(h, l, c, p=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    at = tr.rolling(p).mean()
    up, dn = h.diff(), -l.diff()
    dp = pd.Series(np.where((up > dn) & (up > 0), up, 0.), index=h.index).rolling(p).mean()
    dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.), index=h.index).rolling(p).mean()
    dip, dim = 100*dp/at, 100*dm/at
    return (100*(dip-dim).abs()/(dip+dim).replace(0, np.nan)).rolling(p).mean()

def _atr(h, l, c, p=14):
    return pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1).rolling(p).mean()

def _stoch(h, l, c, k=14, d=3):
    kv = 100*(c-l.rolling(k).min())/(h.rolling(k).max()-l.rolling(k).min()).replace(0, np.nan)
    return kv, kv.rolling(d).mean()

def vol_ratio(vol, n=20):
    """نسبة الحجم لمتوسطه. نأخذ الأعلى بين آخر شمعتين: شمعة اليوم أثناء الجلسة
    ناقصة الحجم فتظلم السهم، والبديل آخر شمعة مكتملة. المتوسط لا يشمل الشمعة نفسها."""
    v = pd.to_numeric(vol, errors="coerce").fillna(0).astype(float)
    for _ in range(3):   # شموع أخيرة حجمها صفر/ناقص (قبل الافتتاح) = بيانات غير مكتملة، نتجاهلها
        if len(v) > n + 2 and v.iloc[-1] == 0: v = v.iloc[:-1]
    if len(v) < n + 2: return 1.0
    def r(i):
        base = float(v.iloc[-(n+i):-i].mean())
        return float(v.iloc[-i]) / base if base > 0 else 1.0
    return max(r(1), r(2))

def sanitize_nan(obj):
    """يمنع أي NaN من كسر JSON بالمتصفح."""
    if isinstance(obj, dict): return {k: sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list): return [sanitize_nan(v) for v in obj]
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))): return None
    return obj

def prepare_df(df):
    """تنظيف أي DataFrame وارد (yfinance/سهمك) لشكل موحّد، أو None لو غير صالح."""
    try:
        if df is None or len(df) == 0: return None
        d = df.copy()
        if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
        d.columns = [str(c).capitalize() for c in d.columns]
        for col in ("Open", "High", "Low", "Close", "Volume"):
            if col not in d.columns: return None
            d[col] = pd.to_numeric(d[col], errors="coerce")
        d = d.dropna(subset=["Close", "High", "Low"])
        d["Volume"] = d["Volume"].fillna(0)
        d = d[d["Close"] > 0]
        return d if len(d) else None
    except Exception:
        return None

def _nd(price): return 2 if price >= 10 else (3 if price >= 1 else 5)

# ════════════════════════════════════════════════════════════
# جلب البيانات
# ════════════════════════════════════════════════════════════
def get_history(ticker, period="2y", interval="1d", retries=2):
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
        except Exception:
            if attempt < retries: time.sleep(2+attempt*2); continue
            return pd.DataFrame()
    return pd.DataFrame()

def _split_batch(raw, part):
    """يفكّ نتيجة yf.download لعدة رموز إلى {رمز: DataFrame} بغض النظر عن ترتيب المستويات."""
    res = {}
    if raw is None or getattr(raw, "empty", True): return res
    if isinstance(raw.columns, pd.MultiIndex):
        l0 = set(raw.columns.get_level_values(0)); l1 = set(raw.columns.get_level_values(1))
        for s in part:
            try:
                if s in l0: d = raw[s]
                elif s in l1: d = raw.xs(s, axis=1, level=1)
                else: continue
                d = d.dropna(how="all")
                if not d.empty: res[s] = d
            except Exception:
                continue
    elif len(part) == 1:
        d = raw.dropna(how="all")
        if not d.empty: res[part[0]] = d
    return res

def batch_history(symbols, period="3mo", chunk=40, progress_cb=None):
    """تنزيل دفعي (طلب واحد لكل chunk) — أسرع بكثير من رمز-رمز.
    لو فشلت الدفعة بالكامل (مشكلة شبكة) نرجع للتنزيل الفردي بشرط أن تكون صغيرة."""
    symbols = list(dict.fromkeys(symbols)); out = {}
    for i in range(0, len(symbols), chunk):
        part = symbols[i:i+chunk]; got = {}
        try:
            raw = yf.download(part, period=period, interval="1d", group_by="ticker",
                              auto_adjust=True, progress=False, threads=True, timeout=30)
            got = _split_batch(raw, part)
        except Exception:
            got = {}
        if not got and len(part) <= 60:
            def one(s):
                d = get_history(s, period, retries=0)
                return s, d
            with ThreadPoolExecutor(max_workers=4) as ex:
                for s, d in ex.map(one, part):
                    if d is not None and not d.empty: got[s] = d
        out.update(got)
        if progress_cb: progress_cb(min(i+chunk, len(symbols)), len(symbols))
    return out

SAHMK_BASE = "https://api.sahmk.sa"
SAHMK_KEY = os.environ.get("SAHMK_KEY_PUBLIC", "")

def sahmk_get(path):
    if not SAHMK_KEY: return {"error": "no key"}
    try:
        import urllib.request
        req = urllib.request.Request(SAHMK_BASE+path, headers={"Authorization": f"Bearer {SAHMK_KEY}", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e: return {"error": str(e)}

def sahmk_price(symbol):
    r = sahmk_get(f"/quote/{symbol}/")
    try: return float(r.get("price") or r.get("last_price") or r.get("close") or 0)
    except Exception: return None

def sahmk_history(symbol):
    r = sahmk_get(f"/historical/{symbol}/")
    try:
        if isinstance(r, list) and len(r) >= 50:
            df = pd.DataFrame(r); df.columns = [c.capitalize() for c in df.columns]
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"]); df = df.sort_values("Date").set_index("Date")
            return df
    except Exception: pass
    return None

def fetch_sa(code, period="2y"):
    """(df, live_price) لسهم تاسي: سهمك أولاً ثم yfinance .SR."""
    live = sahmk_price(code) if SAHMK_KEY else None
    df = sahmk_history(code) if SAHMK_KEY else None
    if df is None or len(df) < 60:
        try: df = yf.Ticker(code + ".SR").history(period=period, interval="1d")
        except Exception: df = None
    return df, live

# ════════════════════════════════════════════════════════════
# تحليل الاتجاه (JRF) — موحّد لأمريكي / كريبتو / تاسي
# ════════════════════════════════════════════════════════════
def analyze_trend_df(df, code, name, kind="us", live_price=None):
    """kind: us | crypto | sa. الأوزان مجموعها 20."""
    try:
        d = prepare_df(df)
        if d is None or len(d) < 60: return None
        cfg = TREND_CFG
        c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]
        n = len(d)
        last_close = float(c.iloc[-1]); prev_close = float(c.iloc[-2])
        price = float(live_price) if live_price and live_price > 0 else last_close

        e20, e50 = _ema(c, 20), _ema(c, 50)
        e_long = _ema(c, 200 if n >= 200 else (100 if n >= 100 else 50))
        rv = _rsi(c); ml, sg, mh = _macd(c)
        adv = _adx(h, l, c); sk, sd = _stoch(h, l, c); av = _atr(h, l, c)
        val = lambda s, k=-1: _f(s.iloc[k])
        gt = lambda a, b: (a is not None and b is not None and a > b)

        rsi = val(rv); adx = val(adv); stk = val(sk); std = val(sd)
        vr = vol_ratio(v)
        c1 = gt(price, val(e20))
        c2 = gt(val(e20), val(e50))
        c3 = gt(price, val(e_long))
        c4 = gt(val(ml), val(sg))
        c5 = gt(val(mh), val(mh, -2))
        c6 = rsi is not None and cfg["rsi_lo"] <= rsi <= cfg["rsi_hi"]
        c7 = gt(rsi, val(rv, -2))
        c8 = adx is not None and adx > cfg["adx_min"]
        c9 = stk is not None and std is not None and stk > 20 and stk > std
        c10 = vr > cfg["vr_min"]
        c11 = gt(adx, val(adv, -2))
        score = int(2*c1 + 2*c2 + 2*c3 + 2*c4 + 1*c5 + 2*c6 + 1*c7 + 3*c8 + 2*c9 + 2*c10 + 1*c11)

        # الوقف والأهداف مبنية على هيكل السعر (مو نسب ثابتة) فيتغير rr فعلياً:
        atr = val(av) or price * 0.02
        swing_low = float(l.iloc[-7:].min())
        sl = min(price - 1.0*atr, swing_low - 0.1*atr)
        sl = max(sl, price - 2.5*atr)
        if sl <= 0: sl = price * 0.9
        resistance = float(h.iloc[-21:-1].max())
        t1 = resistance if (resistance - price) >= 0.75*atr else price + 2.0*atr
        t1 = min(t1, price + 4.0*atr)
        t2 = t1 + 1.5*atr
        risk = max(price - sl, price * 0.001)
        rr = round((t1 - price) / risk, 2)
        nd = _nd(price); R = lambda x: round(float(x), nd)
        ptp = round((t1 - price) / price * 100, 2); psl = round((price - sl) / price * 100, 2)

        above20 = bool(c1)
        e100 = val(_ema(c, 100))
        above100 = e100 is not None and price > e100
        if score >= cfg["buy_score"] and above20 and above100: trend = "استثمار"
        elif score >= cfg["cond_score"] and above20: trend = "سوينج"
        elif above20: trend = "مضاربة"
        else: trend = "ضعيف"

        if score >= cfg["buy_score"] and rr >= cfg["buy_rr"] and above20: verdict = "BUY"
        elif score >= cfg["cond_score"] and rr >= cfg["cond_rr"] and above20: verdict = "BUY_COND"
        elif score >= cfg["watch_score"]: verdict = "WATCH"
        else: verdict = "AVOID"
        confidence = int(min(100, round(score/20*80 + min(max(rr, 0), 3)/3*20)))

        flags = []
        if (resistance - price) < 0.75*atr and resistance > price: flags.append("🧱 مقاومة قريبة")
        if rsi is not None and rsi > 75: flags.append("⚠️ RSI مرتفع")
        if vr < 0.8: flags.append("حجم ضعيف")

        daily_move = atr if atr > 0 else price*0.015
        raw_days = max(1, round((t1 - price) / daily_move))
        factor = 0.8 if (adx or 0) >= 30 else (1.0 if (adx or 0) >= 20 else 1.5)
        est = max(1, round(raw_days*factor))
        eta = f"{est}-{est+2} أيام" if est <= 3 else (f"{est}-{est+4} أيام" if est <= 10 else f"{est}-{est+7} يوم")

        is_sa = (kind == "sa"); is_crypto = (kind == "crypto")
        return {
            "code": code, "name": name, "is_crypto": is_crypto, "is_sa": is_sa, "is_junk": False,
            "price": R(price), "price_sar": (None if is_sa else round(price*USD_TO_SAR, 2)),
            "live": bool(live_price), "score": score, "verdict": verdict, "trend": trend,
            "confidence": confidence, "lb": R(price*0.998), "t1": R(t1), "t2": R(t2), "sl": R(sl),
            "rr": rr, "ptp": ptp, "psl": psl, "rsi": round(rsi, 1) if rsi is not None else None,
            "adx": round(adx, 1) if adx is not None else 0, "vr": round(vr, 1),
            "chg": round((price - prev_close)/prev_close*100, 2) if prev_close else 0,
            "eta": eta, "flags": flags,
        }
    except Exception:
        return None

# ════════════════════════════════════════════════════════════
# الخشاش السريع — منطق مستقل تماماً:
#  • سعر أقل من 5$ فقط + بوابة سيولة (حجم + قيمة تداول)
#  • EMA9/EMA20 قصيرة، كسر قمة 10 أيام (أو قربها)، وانفجار حجم
#  • وقف قريب مبني على أقرب قاع + ATR، وأهداف 1.5R/3R (أيام معدودة)
#  • حماية من مطاردة الركض: ارتفاع >100% بـ10 أيام أو قفزة >35% = مراقبة فقط
# ════════════════════════════════════════════════════════════
def analyze_junk_df(df, code, name):
    try:
        d = prepare_df(df)
        cfg = JUNK_CFG
        if d is None or len(d) < 25: return None
        c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]
        price = float(c.iloc[-1]); prev_close = float(c.iloc[-2])
        if not (cfg["price_min"] <= price < cfg["price_max"]): return None

        avg_vol = float(v.iloc[-21:-1].mean()) if len(v) > 21 else float(v.mean())
        min_vol = cfg["min_vol_penny"] if price < 1.0 else cfg["min_vol_low"]
        if avg_vol < min_vol or avg_vol * price < cfg["min_dollar_vol"]: return None

        e9, e20 = _ema(c, 9), _ema(c, 20)
        rv = _rsi(c); adv = _adx(h, l, c); av = _atr(h, l, c)
        val = lambda s, k=-1: _f(s.iloc[k])
        gt = lambda a, b: (a is not None and b is not None and a > b)

        bd = cfg["breakout_days"]
        hh = float(h.iloc[-(bd+1):-1].max())
        breakout = price > hh
        near = (not breakout) and price >= hh * 0.97
        vr = vol_ratio(v)
        rsi = val(rv); adx = val(adv)
        chg = (price - prev_close) / prev_close * 100 if prev_close else 0.0

        c1 = gt(price, val(e9)); c2 = gt(val(e9), val(e20))
        c3 = 4 if breakout else (2 if near else 0)
        c4 = vr >= cfg["vr_spike"]; c5 = vr >= cfg["vr_big"]
        c6 = price > prev_close
        c7 = rsi is not None and 40 <= rsi <= 85
        c8 = gt(rsi, val(rv, -2))
        c9 = adx is not None and adx > 18
        score = int(2*c1 + 2*c2 + c3 + 4*c4 + 2*c5 + 1*c6 + 2*c7 + 1*c8 + 2*c9)

        ext10 = price / float(c.iloc[-10:].min()) - 1
        chasing = (ext10 > cfg["max_ext10"]) or (chg > cfg["max_day_chg"])

        if chasing: verdict = "WATCH" if score >= cfg["watch_score"] else "AVOID"
        elif score >= cfg["buy_score"] and c1 and (breakout or vr >= 1.5): verdict = "BUY"
        elif score >= cfg["cond_score"] and c1 and (breakout or near or vr >= 1.3): verdict = "BUY_COND"
        elif score >= cfg["watch_score"]: verdict = "WATCH"
        else: verdict = "AVOID"

        atr = val(av) or price * 0.05
        swing_low = float(l.iloc[-5:].min())
        risk = min(max(price - swing_low, cfg["risk_atr_min"]*atr), cfg["risk_atr_max"]*atr)
        risk = min(risk, price * 0.45)
        sl = price - risk
        t1 = price + cfg["t1_r"]*risk; t2 = price + cfg["t2_r"]*risk
        nd = _nd(price); R = lambda x: round(float(x), nd)

        flags = ["⏱ اطلع بعد 3 أيام لو ما تحرك"]
        if chasing: flags.insert(0, "⚠️ ركض زايد — لا تطارد")
        if avg_vol * price < 1_000_000: flags.append("سيولة محدودة — انتبه للسبريد")

        confidence = int(min(100, round(score/20*75 + min(vr, 3)/3*25)))
        trend = "اختراق+حجم" if (breakout and c4) else ("زخم" if (c1 and c2) else "ضعيف")
        eta = "1-3 أيام" if score >= cfg["buy_score"] else ("3-6 أيام" if score >= cfg["cond_score"] else "أسبوع+")

        return {
            "code": code, "name": name, "is_crypto": False, "is_sa": False, "is_junk": True,
            "price": R(price), "price_sar": round(price*USD_TO_SAR, 3), "live": False,
            "score": score, "verdict": verdict, "trend": trend, "confidence": confidence,
            "lb": R(price*0.995), "t1": R(t1), "t2": R(t2), "sl": R(sl),
            "rr": cfg["t1_r"], "ptp": round((t1-price)/price*100, 2), "psl": round(risk/price*100, 2),
            "rsi": round(rsi, 1) if rsi is not None else None,
            "adx": round(adx, 1) if adx is not None else 0, "vr": round(vr, 1),
            "chg": round(chg, 2), "eta": eta, "flags": flags,
            "price_tier": "penny" if price < 1 else "low",
        }
    except Exception:
        return None

# ════════════════════════════════════════════════════════════
# القائمة الشرعية (اختيارية)
# ════════════════════════════════════════════════════════════
def load_sharia_us():
    path = os.environ.get("SHARIA_US_FILE", os.path.join(BASE_DIR, "sharia_compliant_us.json"))
    try:
        with open(path, encoding="utf-8") as f: d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}

# ════════════════════════════════════════════════════════════
# حالة المسح + المفضلة + سجل الأداء
# ════════════════════════════════════════════════════════════
MARKETS = ("us", "crypto", "sa", "junk")
def _blank(): return {"data": [], "last": None, "status": "idle", "progress": 0, "scanned": 0, "mode": "", "msg": ""}
scan_state = {m: _blank() for m in MARKETS}

FAV_FILE = os.path.join(DATA_DIR, "public_favorites.json")
SCAN_HISTORY_FILE = os.path.join(DATA_DIR, "public_scan_history.json")
history_lock = threading.Lock()

def load_favs():
    try:
        with open(FAV_FILE) as f: d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}

def save_favs(d):
    with open(FAV_FILE, "w") as f: json.dump(d, f, ensure_ascii=False, indent=2)

def load_scan_history():
    with history_lock:
        try:
            with open(SCAN_HISTORY_FILE) as f: return json.load(f)
        except Exception:
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
            hist.append({"ts": ts, "market": market, "code": s["code"], "name": s.get("name"),
                         "verdict": s.get("verdict"), "score": s.get("score"),
                         "entry": s.get("price"), "t1": s.get("t1"), "t2": s.get("t2"), "sl": s.get("sl")})
        save_scan_history(hist[-1500:])
    except Exception:
        pass

def _bars_since(market, code, days):
    """شموع يومية تغطي آخر `days` يوم لتقييم الترشيحات (High/Low لكل يوم)."""
    period = "1mo" if days <= 25 else ("3mo" if days <= 80 else ("6mo" if days <= 170 else "2y"))
    try:
        if market == "sa":
            df, _ = fetch_sa(code, period)
        else:
            df = get_history(code, period, retries=1)
        d = prepare_df(df)
        if d is None: return None
        idx = pd.to_datetime(d.index)
        if getattr(idx, "tz", None) is not None: idx = idx.tz_localize(None)
        d = d.copy(); d.index = idx.normalize()
        return d
    except Exception:
        return None

def evaluate_pick(rec, bars):
    """يمشي على الشموع بعد يوم الترشيح بالترتيب. لو لمس الوقف والهدف بنفس اليوم نحسبها وقف
    (تحفّظ). بعد هدف 1 نعتبره ناجح ولا نخفّضه لو رجع للوقف."""
    entry, t1, t2, sl = rec.get("entry"), rec.get("t1"), rec.get("t2"), rec.get("sl")
    try: day = pd.Timestamp(datetime.strptime(rec["ts"], "%Y-%m-%d %H:%M").date())
    except Exception: return "unknown", None
    if bars is None or bars.empty: return "unknown", None
    cur = float(bars["Close"].iloc[-1])
    after = bars[bars.index > day]
    status, hit1 = "open", False
    for _, row in after.iterrows():
        lo, hi = float(row["Low"]), float(row["High"])
        if not hit1:
            if sl is not None and lo <= sl: status = "hit_sl"; break
            if t2 is not None and hi >= t2: status = "hit_t2"; break
            if t1 is not None and hi >= t1: hit1 = True; status = "hit_t1"
        else:
            if t2 is not None and hi >= t2: status = "hit_t2"; break
    return status, cur

def compute_scan_history_status(market=None, days=10):
    hist = load_scan_history()
    cutoff = datetime.now() - timedelta(days=days)
    rows, cache = [], {}
    for rec in hist:
        if market and rec.get("market") != market: continue
        try: rec_dt = datetime.strptime(rec["ts"], "%Y-%m-%d %H:%M")
        except Exception: continue
        if rec_dt < cutoff: continue
        key = (rec.get("market"), rec.get("code"))
        if key not in cache: cache[key] = _bars_since(rec.get("market"), rec.get("code"), days)
        status, cur = evaluate_pick(rec, cache[key])
        row = dict(rec); row["status"] = status; row["current"] = cur
        row["days_elapsed"] = (datetime.now() - rec_dt).days
        e = rec.get("entry") or 0
        row["pnl_pct"] = round((cur - e)/e*100, 2) if (cur is not None and e) else None
        rows.append(row)
    rows.sort(key=lambda r: r["ts"], reverse=True)
    return rows

# ════════════════════════════════════════════════════════════
# تنفيذ المسح
# ════════════════════════════════════════════════════════════
MEDALS = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]

def _publish(market, results, sort_key=None, **extra):
    st = scan_state[market]
    sort_key = sort_key or (lambda x: (-x["score"], -x["confidence"], -x["rr"]))
    buys = sorted([s for s in results if s["verdict"] in ("BUY", "BUY_COND")], key=sort_key)
    top = buys[:5]
    if len(top) < 5:
        watch = sorted([s for s in results if s["verdict"] == "WATCH"], key=sort_key)
        top += watch[:5 - len(top)]
    favs = set(load_favs().get(market, []))
    for i, s in enumerate(top):
        s["medal"] = MEDALS[i] if i < len(MEDALS) else str(i+1)
        s["is_fav"] = s["code"] in favs
    st["data"] = top
    st["last"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    st["scanned"] = len(results)
    st["progress"] = 100; st["status"] = "done"
    st.update(extra)
    record_scan_history(market, top)
    return top

def _progress(market, lo=0, hi=90):
    def cb(done, total): scan_state[market]["progress"] = round(lo + (hi-lo)*done/max(total, 1))
    return cb

def run_scan(market="us"):
    stocks = US_STOCKS if market == "us" else CRYPTO
    kind = "crypto" if market == "crypto" else "us"
    data = batch_history(list(stocks), "2y", chunk=25, progress_cb=_progress(market))
    results = [r for r in (analyze_trend_df(df, s, stocks[s], kind) for s, df in data.items()) if r]
    return _publish(market, results)

def run_scan_sa():
    market = "sa"; items = list(SA_STOCKS.items()); results = []
    if SAHMK_KEY:
        def one(it):
            code, name = it
            df, live = fetch_sa(code)
            return analyze_trend_df(df, code, name, "sa", live)
        with ThreadPoolExecutor(max_workers=3) as ex:
            for i, r in enumerate(ex.map(one, items), 1):
                scan_state[market]["progress"] = round(90*i/len(items))
                if r: results.append(r)
    else:
        data = batch_history([c + ".SR" for c, _ in items], "2y", chunk=25, progress_cb=_progress(market))
        for code, name in items:
            r = analyze_trend_df(data.get(code + ".SR"), code, name, "sa")
            if r: results.append(r)
    return _publish(market, results)

def run_scan_junk(sharia=False):
    market = "junk"; st = scan_state[market]
    sh = load_sharia_us()
    if sharia:
        if not sh:
            st.update(status="error", msg="ملف القائمة الشرعية (sharia_compliant_us.json) غير موجود بجانب الملف"); return []
        universe = dict(sh)
    else:
        universe = dict(JUNK_STOCKS)
    st["mode"] = "sharia" if sharia else "list"
    data = batch_history(list(universe), "3mo", chunk=120 if sharia else 40, progress_cb=_progress(market))
    results = []
    for s, df in data.items():
        r = analyze_junk_df(df, s, universe.get(s, s))
        if r:
            if sh and s in sh: r["flags"] = ["🕌 ضمن القائمة الشرعية"] + r["flags"]
            results.append(r)
    return _publish(market, results, sort_key=lambda x: (-x["score"], -x["vr"], -x["confidence"]),
                    mode="sharia" if sharia else "list")

def _guarded(market, fn, *a):
    try:
        fn(*a)
    except Exception as e:
        scan_state[market].update(status="error", msg=f"خطأ أثناء المسح: {type(e).__name__}")

# ════════════════════════════════════════════════════════════
# تحليل سهم واحد (للبحث الفوري والمفضلة)
# ════════════════════════════════════════════════════════════
def analyze_any(code, market):
    """يرجّع (نتيجة, رسالة). السهم الأمريكي تحت 5$ يتحلل تلقائياً بمعايير الخشاش."""
    if market == "sa" or code in SA_STOCKS:
        df, live = fetch_sa(code)
        return analyze_trend_df(df, code, SA_STOCKS.get(code, code), "sa", live), None
    if market == "junk" or code in JUNK_STOCKS:
        r = analyze_junk_df(get_history(code, "3mo", retries=1), code, JUNK_STOCKS.get(code, code))
        return r, (None if r else "ما ينطبق عليه شروط الخشاش (السعر 5$ فأكثر أو سيولة ضعيفة)")
    crypto = (market == "crypto")
    df = get_history(code, "2y", retries=1)
    name = US_STOCKS.get(code) or CRYPTO.get(code) or code
    d = prepare_df(df)
    if d is not None and not crypto and float(d["Close"].iloc[-1]) < JUNK_CFG["price_max"]:
        r = analyze_junk_df(df, code, name)
        if r: return r, "السهم تحت 5$ — حُلّل بمعايير الخشاش"
        return None, "سهم تحت 5$ لكن سيولته ضعيفة — خارج نطاق الأداة"
    return analyze_trend_df(df, code, name, "crypto" if crypto else "us"), None

# ════════════════════════════════════════════════════════════
# مسارات Flask — قراءة/تحليل فقط
# ════════════════════════════════════════════════════════════
CODE_RE = re.compile(r"^[A-Z0-9.\-^=]{1,15}$")
def _market(src=None, default="us"):
    m = (src or request.args).get("market", default)
    return m if m in MARKETS else default

@app.route("/scan")
def scan_route():
    market = _market()
    if scan_state[market]["status"] == "scanning":
        return jsonify({"ok": True, "already": True})
    sharia = request.args.get("sharia") == "1"
    scan_state[market].update(status="scanning", progress=0, msg="")
    if market == "sa": target = (_guarded, market, run_scan_sa)
    elif market == "junk": target = (_guarded, market, run_scan_junk, sharia)
    else: target = (_guarded, market, run_scan, market)
    threading.Thread(target=target[0], args=target[1:], daemon=True).start()
    return jsonify({"ok": True})

@app.route("/scan_status")
def scan_status():
    st = scan_state[_market()]
    return jsonify({"status": st["status"], "progress": st["progress"], "scanned": st["scanned"], "msg": st["msg"]})

@app.route("/results")
def results_route():
    st = scan_state[_market()]
    return jsonify(sanitize_nan({"data": st["data"], "last": st["last"], "scanned": st["scanned"], "mode": st["mode"]}))

@app.route("/api/meta")
def api_meta():
    return jsonify({"sharia_us": len(load_sharia_us())})

@app.route("/analyze_one", methods=["POST"])
def analyze_one():
    d = request.get_json(silent=True) or {}
    code = str(d.get("code", "")).strip().upper()
    market = _market(d)
    if not code or not CODE_RE.match(code): return jsonify({"ok": False, "msg": "رمز غير صالح"})
    r, note = analyze_any(code, market)
    if not r: return jsonify({"ok": False, "msg": note or f"تعذّر تحليل {code}"})
    return jsonify(sanitize_nan({"ok": True, "result": r, "note": note}))

@app.route("/api/favorites")
def get_favorites(): return jsonify(load_favs())

@app.route("/api/favorites/toggle", methods=["POST"])
def toggle_favorite():
    d = request.get_json(silent=True) or {}
    code = str(d.get("code", "")).strip().upper(); market = _market(d)
    if not CODE_RE.match(code): return jsonify({"ok": False})
    favs = load_favs(); lst = favs.get(market, [])
    if code in lst: lst.remove(code); added = False
    else: lst.append(code); added = True
    favs[market] = lst; save_favs(favs)
    return jsonify({"ok": True, "added": added})

@app.route("/api/favorites/analyze")
def analyze_favorites():
    market = _market()
    codes = load_favs().get(market, [])
    if not codes: return jsonify({"data": []})
    out = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        for r, _ in ex.map(lambda c: analyze_any(c, market), codes):
            if r: r["is_fav"] = True; out.append(r)
    out.sort(key=lambda x: (-x["score"], -x["confidence"]))
    return jsonify(sanitize_nan({"data": out}))

@app.route("/api/scan_history")
def api_scan_history():
    market = request.args.get("market", "").strip()
    market = market if market in MARKETS else None
    try: days = int(request.args.get("days", 10))
    except Exception: days = 10
    days = max(1, min(days, 90))
    rows = compute_scan_history_status(market, days)
    buy_rows = [r for r in rows if r.get("verdict") in ("BUY", "BUY_COND")]
    total = len(buy_rows)
    hit = len([r for r in buy_rows if r["status"] in ("hit_t1", "hit_t2")])
    stopped = len([r for r in buy_rows if r["status"] == "hit_sl"])
    closed = hit + stopped
    summary = {"total": total, "hit_target": hit, "hit_stop": stopped,
               "still_open": max(0, total - closed),
               "win_rate": round(hit/closed*100, 1) if closed else 0, "closed": closed}
    return jsonify(sanitize_nan({"rows": rows, "summary": summary}))

@app.route("/api/export_scan_history")
def export_scan_history():
    market = request.args.get("market", "").strip()
    market = market if market in MARKETS else None
    rows = compute_scan_history_status(market, days=365)
    out = io.StringIO(); w = csv.writer(out)
    w.writerow(["التاريخ","السوق","الرمز","الاسم","الحكم","السكور","سعر الدخول","هدف 1","هدف 2","وقف الخسارة","السعر الحالي","التغير %","الحالة","أيام منذ الترشيح"])
    st_ar = {"hit_t1":"✅ لمس هدف 1","hit_t2":"✅ لمس هدف 2","hit_sl":"❌ لمس الوقف","open":"⏳ لسه مفتوح","unknown":"؟ تعذر الجلب"}
    mk_ar = {"us":"أمريكي","crypto":"كريبتو","sa":"تاسي","junk":"خشاش"}
    for r in rows:
        w.writerow([r.get("ts",""), mk_ar.get(r.get("market"), r.get("market","")), r.get("code",""), r.get("name",""),
                    r.get("verdict",""), r.get("score",""), r.get("entry",""), r.get("t1",""), r.get("t2",""),
                    r.get("sl",""), r.get("current",""), r.get("pnl_pct",""),
                    st_ar.get(r.get("status"), r.get("status","")), r.get("days_elapsed","")])
    return Response(out.getvalue().encode("utf-8-sig"), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment;filename=jalal_scanner_history.csv"})

# ════════════════════════════════════════════════════════════
# الواجهة
# ════════════════════════════════════════════════════════════
PAGE = r"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ماسح جلال رادار — عام</title>
<link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap" rel="stylesheet">
<style>
:root{--bg:#0e1015;--surface:#16181d;--surface-2:#1c1f26;--ink:#e8e8e6;--ink-2:#a8a8a4;--ink-3:#6b6f76;--line:#2a2d34;
--brand:#ef9f27;--green:#1d9e75;--green-soft:#10251e;--red:#e2504a;--amber:#ef9f27;--amber-soft:#2a2010;--r:14px;--r-lg:20px}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Tajawal',system-ui,sans-serif;background:var(--bg);color:var(--ink);direction:rtl;line-height:1.6}
.shell{max-width:760px;margin:0 auto;padding:16px 16px 64px}
.hdr{text-align:center;padding:18px 0 10px}.hdr h1{font-size:19px;font-weight:800}.hdr p{color:var(--ink-3);font-size:12px;margin-top:4px}
.mkt-tabs{display:flex;gap:6px;margin-bottom:16px}
.mkt-tab{flex:1;padding:10px 4px;border-radius:var(--r);border:1px solid var(--line);background:var(--surface);color:var(--ink-2);text-align:center;font-size:12.5px;cursor:pointer;font-family:inherit}
.mkt-tab.on{background:var(--brand);color:#2a1a00;border-color:var(--brand);font-weight:700}
.q-box{display:flex;gap:8px;margin-bottom:14px}
.q-box input{flex:1;padding:10px 12px;border-radius:var(--r);border:1px solid var(--line);background:var(--surface);color:var(--ink);font-family:inherit}
.q-box button{padding:10px 16px;border-radius:var(--r);border:0;background:var(--surface-2);color:var(--ink);font-family:inherit;cursor:pointer}
.scan-btn{width:100%;padding:13px;border-radius:var(--r);border:0;background:var(--brand);color:#2a1a00;font-size:14px;font-weight:800;cursor:pointer;margin-bottom:8px;font-family:inherit}
.scan-meta{text-align:center;color:var(--ink-3);font-size:11px;margin-bottom:12px}
.tally{display:flex;gap:8px;margin-bottom:12px}.tally-cell{flex:1;background:var(--surface);border-radius:var(--r);padding:10px;text-align:center}
.tally-num{font-size:19px;font-weight:800}.tally-lbl{font-size:11px;color:var(--ink-3)}
.empty{text-align:center;color:var(--ink-3);padding:40px 0;font-size:13px;line-height:1.8}.empty-ico{font-size:32px;margin-bottom:8px}
.sig{background:var(--surface);border:1px solid var(--line);border-radius:var(--r-lg);margin-top:12px;overflow:hidden}
.sig-verdict{padding:11px 16px;display:flex;align-items:center;justify-content:space-between}
.sig-verdict.buy{background:var(--green-soft);border-bottom:1px solid #1a4a3a}
.sig-verdict.cond{background:var(--amber-soft);border-bottom:1px solid #4a3818}
.sig-verdict.watch{background:var(--surface-2);border-bottom:1px solid var(--line)}
.verdict-main{display:flex;align-items:center;gap:9px}.verdict-label{font-weight:800;font-size:15px}
.verdict-sub{font-size:11px;color:var(--ink-3);margin-top:1px}
.verdict-score{text-align:left}.verdict-score-num{font-size:19px;font-weight:800}.verdict-score-lbl{font-size:10px;color:var(--ink-3)}
.sig-body{padding:14px 16px}.sig-top{display:flex;align-items:flex-start;gap:11px}.sig-medal{font-size:22px}
.sig-id{flex:1;min-width:0}.sig-name{font-weight:700;font-size:14px;display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.sig-code{color:var(--ink-3);font-size:12px;font-weight:400}
.fav-star{cursor:pointer;color:var(--ink-3);font-size:16px}.fav-star.on{color:var(--brand)}
.sig-meta{display:flex;gap:8px;margin-top:4px;flex-wrap:wrap}.mb{font-size:11px;color:var(--ink-3);background:var(--surface-2);padding:2px 6px;border-radius:6px}
.sig-px{text-align:left}.sig-px-main{font-size:17px;font-weight:800}.sig-px-sar{font-size:11px;color:var(--ink-3)}
.sig-bar{height:3px;background:var(--surface-2);margin:10px 0;border-radius:3px;overflow:hidden}.sig-bar-fill{height:100%;background:var(--brand)}
.sig-tags{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:10px}
.tag{font-size:11px;background:var(--surface-2);padding:5px 9px;border-radius:8px;color:var(--ink-2)}
.tag.warn{background:var(--amber-soft);color:var(--amber)}
.verdict-score-num{direction:ltr;unicode-bidi:isolate}
.sig-trade-btn{margin-right:auto;background:var(--green);color:#fff;border:0;border-radius:9px;padding:6px 14px;font-family:inherit;font-size:12px;font-weight:700;text-decoration:none}
.sig-levels{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;background:var(--surface-2);border-radius:var(--r);padding:8px}
.lvl{text-align:center}.lvl-lbl{font-size:10px;color:var(--ink-3)}.lvl-val{font-size:12px;font-weight:700;margin-top:2px}.lvl-pct{font-size:10px;color:var(--ink-3)}
.fav-header{color:var(--brand);font-size:12px;font-weight:700;margin:16px 0 4px}
.loader{display:none;text-align:center;padding:20px;color:var(--ink-3);font-size:13px}.loader.show{display:block}
.junk-box{display:none;background:var(--amber-soft);border:1px solid #4a3818;border-radius:var(--r);padding:10px 12px;font-size:11.5px;color:var(--ink-2);margin-bottom:12px;line-height:1.8}
.junk-box label{display:flex;align-items:center;gap:6px;margin-top:6px;color:var(--ink);cursor:pointer}
.footer-note{text-align:center;color:var(--ink-3);font-size:11px;margin-top:24px;padding-top:14px;border-top:1px solid var(--line)}
.hidden-view{display:none}
</style>
</head>
<body>
<div class="shell">
  <div class="hdr"><h1>⚡ ماسح جلال رادار</h1><p>ترشيح فني آلي (JRF) للأسهم السعودية والأمريكية والعملات — عرض فقط، بدون تنفيذ صفقات</p></div>
  <div class="mkt-tabs">
    <button class="mkt-tab on" id="mt-us" onclick="setMarket('us')">🇺🇸 أمريكي</button>
    <button class="mkt-tab" id="mt-sa" onclick="setMarket('sa')">🇸🇦 تاسي</button>
    <button class="mkt-tab" id="mt-crypto" onclick="setMarket('crypto')">💰 كريبتو</button>
    <button class="mkt-tab" id="mt-junk" onclick="setMarket('junk')">⚡ خشاش</button>
  </div>
  <div class="junk-box" id="junk-box">
    ⚠️ <b>الخشاش السريع</b>: أسهم تحت 5$ عالية التذبذب. معايير منفصلة (اختراق + انفجار حجم)، وقف قريب، أهداف 1.5R/3R لأيام معدودة، ويمنع مطاردة الركض الزايد. مخاطرة أعلى بطبيعته — حجم مركز صغير دائماً.
    <label id="sharia-row" style="display:none"><input type="checkbox" id="sharia-chk"><span>🕌 شرعي فقط — يمسح القائمة الشرعية كاملة (<span id="sharia-n">0</span> سهم) ويفلترها بالسعر والسيولة (أبطأ)</span></label>
  </div>
  <div class="q-box"><input id="q-code" placeholder="الرمز (AAPL أو 2222 أو BTC-USD)"><button onclick="analyzeOne()">تحليل فوري</button></div>
  <div id="instant"></div>
  <button class="scan-btn" id="scan-btn" onclick="startScan()">🔍 مسح السوق الأمريكي</button>
  <div class="scan-meta" id="scan-meta"></div>
  <div class="loader" id="loader"><div id="loader-sub">جاري الفحص...</div></div>
  <div class="mkt-tabs" style="margin-top:4px">
    <button class="mkt-tab on" id="vt-scan" onclick="setView('scan')">🔍 نتائج المسح</button>
    <button class="mkt-tab" id="vt-perf" onclick="setView('perf')">📜 أداء الترشيحات</button>
  </div>
  <div id="view-scan">
    <div id="tally"></div><div id="fav-section"></div>
    <div id="signals"><div class="empty"><div class="empty-ico">📡</div>اضغط مسح لتحليل السوق</div></div>
  </div>
  <div id="view-perf" class="hidden-view">
    <div style="color:var(--ink-3);font-size:12px;margin-bottom:10px;line-height:1.7">
      كل مسح يُحفظ هنا. نمشي على شموع الأيام بعد الترشيح (أعلى/أدنى سعر يومي) ونحكم: وصل الهدف أولاً ولا لمس الوقف؟ هذا اللي تعتمد عليه لمعايرة الشروط بالأرقام.
    </div>
    <div class="q-box">
      <select id="ph-days" style="flex:1;padding:10px 12px;border-radius:var(--r);border:1px solid var(--line);background:var(--surface);color:var(--ink);font-family:inherit" onchange="loadPerformance()">
        <option value="3">آخر 3 أيام</option><option value="5">آخر 5 أيام</option><option value="10" selected>آخر 10 أيام</option>
        <option value="30">آخر 30 يوم</option><option value="90">آخر 90 يوم</option>
      </select>
      <button onclick="loadPerformance()">🔄 تحديث</button>
    </div>
    <a id="ph-export" href="/api/export_scan_history" style="display:inline-block;margin-bottom:12px;padding:8px 14px;border-radius:10px;background:var(--surface-2);color:var(--ink);font-size:12px;text-decoration:none">📥 تصدير Excel (CSV)</a>
    <div id="ph-summary"></div>
    <div id="ph-list"><div class="empty"><div class="empty-ico">📜</div>اضغط تحديث لعرض سجل الأداء</div></div>
  </div>
  <div class="footer-note">أداة تحليل فني تعليمية — ليست توصية استثمارية. تحقق بنفسك قبل أي قرار.</div>
</div>
<div id="toast" style="position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:var(--surface-2);color:var(--ink);padding:10px 18px;border-radius:10px;font-size:13px;opacity:0;pointer-events:none;transition:.2s;z-index:99"></div>
<script>
var market='us', scanPoll=null;
function esc(s){return String(s==null?'':s).replace(/[&<>"']/g,function(ch){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch];});}
function q(id){return document.getElementById(id);}
function bd(x){return '<bdi dir="ltr">'+x+'</bdi>';}
function fmtEta(e){return esc(e||'—').replace(/^(\d+-\d+)/,function(m){return bd(m);});}
function money(v,isSA){return isSA?bd(v)+' ر.س':bd('$'+v);}

function setMarket(m){
  market=m;
  ['us','sa','crypto','junk'].forEach(function(k){q('mt-'+k).classList.toggle('on',k===m);});
  q('junk-box').style.display=(m==='junk')?'block':'none';
  var lbls={us:'🔍 مسح السوق الأمريكي',crypto:'🔍 مسح الكريبتو',sa:'🔍 مسح تاسي 🇸🇦',junk:'⚡ مسح الخشاش السريع'};
  q('scan-btn').innerHTML=lbls[m]||'🔍 مسح';
  q('instant').innerHTML='';
  loadResults(); loadFavorites();
  if(q('vt-perf').classList.contains('on')) loadPerformance();
}
function startScan(){
  var m=market, url='/scan?market='+m;
  if(m==='junk' && q('sharia-chk').checked) url+='&sharia=1';
  q('loader').classList.add('show'); q('loader-sub').textContent='جاري الفحص...';
  fetch(url).then(function(r){return r.json();}).then(function(){
    if(scanPoll) clearInterval(scanPoll);
    scanPoll=setInterval(function(){pollScan(m);},1200);
  });
}
function pollScan(m){
  fetch('/scan_status?market='+m).then(function(r){return r.json();}).then(function(d){
    if(d.status==='done'||d.status==='error'){
      clearInterval(scanPoll); q('loader').classList.remove('show');
      if(d.status==='error') toast(d.msg||'صار خطأ بالمسح');
      if(m===market) loadResults();
    } else if(m===market){ q('loader-sub').textContent='تم '+(d.progress||0)+'%'; }
  });
}
function loadResults(){
  fetch('/results?market='+market).then(function(r){return r.json();}).then(function(d){
    var data=d.data||[];
    q('scan-meta').textContent=d.last?('آخر مسح: '+d.last+' · حُلّل '+d.scanned+(d.mode==='sharia'?' · القائمة الشرعية':'')):'';
    var buys=data.filter(function(s){return s.verdict==='BUY';}).length;
    var conds=data.filter(function(s){return s.verdict==='BUY_COND';}).length;
    q('tally').innerHTML=data.length?
      '<div class="tally"><div class="tally-cell"><div class="tally-num">'+buys+'</div><div class="tally-lbl">شراء</div></div>'+
      '<div class="tally-cell"><div class="tally-num">'+conds+'</div><div class="tally-lbl">مشروط</div></div>'+
      '<div class="tally-cell"><div class="tally-num">'+data.length+'</div><div class="tally-lbl">مرشّح</div></div></div>':'';
    if(data.length) q('signals').innerHTML=data.map(renderSig).join('');
    else if(d.last) q('signals').innerHTML='<div class="empty"><div class="empty-ico">🧊</div>المسح خلص وما فيه سهم يحقق الشروط الحين.<br>هذا طبيعي بسوق ضعيف — لا تنزّل العتبات عشان تشوف شي، راجع تبويب أداء الترشيحات أول.</div>';
    else q('signals').innerHTML='<div class="empty"><div class="empty-ico">📡</div>اضغط مسح لتحليل السوق</div>';
  });
}
function renderSig(s){
  var vClass=s.verdict==='BUY'?'buy':(s.verdict==='BUY_COND'?'cond':'watch');
  var vIco=s.verdict==='BUY'?'🟢':(s.verdict==='BUY_COND'?'🟡':'⏳');
  var vLabel=s.verdict==='BUY'?'شراء':(s.verdict==='BUY_COND'?'شراء مشروط':'مراقبة');
  var isSA=s.is_sa===true;
  var priceDisp='<div class="sig-px-main">'+money(s.price,isSA)+'</div>'+((!isSA&&s.price_sar)?'<div class="sig-px-sar">'+bd(s.price_sar)+' ر.س</div>':'');
  var tvSym=isSA?('TADAWUL-'+s.code):s.code;
  var tags=(s.flags||[]).map(function(f){return '<span class="tag warn">'+esc(f)+'</span>';}).join('');
  var tv='<a class="sig-trade-btn" href="https://www.tradingview.com/symbols/'+encodeURIComponent(tvSym)+'/" target="_blank" rel="noopener">📈 TradingView</a>';
  return '<div class="sig"><div class="sig-verdict '+vClass+'"><div class="verdict-main"><span>'+vIco+'</span>'+
    '<div><div class="verdict-label">'+vLabel+(s.is_junk?' ⚡':'')+'</div><div class="verdict-sub">ثقة '+bd(s.confidence+'%')+' · '+esc(s.trend)+'</div></div></div>'+
    '<div class="verdict-score"><div class="verdict-score-num">'+s.score+'/20</div><div class="verdict-score-lbl">JRF</div></div></div>'+
    '<div class="sig-body"><div class="sig-top"><span class="sig-medal">'+(s.medal||'')+'</span><div class="sig-id">'+
    '<div class="sig-name">'+esc(s.name)+'<span class="sig-code">'+esc(s.code)+'</span>'+
    '<span class="fav-star'+(s.is_fav?' on':'')+'" data-code="'+esc(s.code)+'">'+(s.is_fav?'★':'☆')+'</span></div>'+
    '<div class="sig-meta"><span class="mb">RSI '+bd(s.rsi==null?'—':s.rsi)+'</span><span class="mb">ADX '+bd(s.adx||'—')+'</span><span class="mb">حجم '+bd('×'+s.vr)+'</span><span class="mb">⏱ '+fmtEta(s.eta)+'</span></div></div>'+
    '<div class="sig-px">'+priceDisp+'</div></div>'+
    '<div class="sig-bar"><div class="sig-bar-fill" style="width:'+s.confidence+'%"></div></div>'+
    '<div class="sig-tags"><span class="tag">R:R <b>'+bd(s.rr)+'</b></span><span class="tag">🎯 <b>'+money(s.t1,isSA)+'</b> '+bd('+'+s.ptp+'%')+'</span>'+
    '<span class="tag">🛡 <b>'+money(s.sl,isSA)+'</b> '+bd('-'+s.psl+'%')+'</span>'+tv+'</div>'+
    (tags?'<div class="sig-tags">'+tags+'</div>':'')+
    '<div class="sig-levels"><div class="lvl"><div class="lvl-lbl">دخول</div><div class="lvl-val">'+money(s.lb,isSA)+'</div></div>'+
    '<div class="lvl"><div class="lvl-lbl">هدف 1</div><div class="lvl-val">'+money(s.t1,isSA)+'</div><div class="lvl-pct">'+bd('+'+s.ptp+'%')+'</div></div>'+
    '<div class="lvl"><div class="lvl-lbl">هدف 2</div><div class="lvl-val">'+money(s.t2,isSA)+'</div></div>'+
    '<div class="lvl"><div class="lvl-lbl">وقف</div><div class="lvl-val">'+money(s.sl,isSA)+'</div><div class="lvl-pct">'+bd('-'+s.psl+'%')+'</div></div></div></div></div>';
}
document.addEventListener('click',function(e){
  var t=e.target;
  if(t&&t.classList&&t.classList.contains('fav-star')) toggleFav(t.getAttribute('data-code'));
});
function toggleFav(code){
  fetch('/api/favorites/toggle',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code:code,market:market})})
    .then(function(r){return r.json();}).then(function(d){
      if(d.ok){toast(d.added?'⭐ أُضيف للمفضلة':'تمت الإزالة');loadResults();loadFavorites();}
    });
}
function loadFavorites(){
  var m=market;
  fetch('/api/favorites/analyze?market='+m).then(function(r){return r.json();}).then(function(d){
    if(m!==market) return;
    q('fav-section').innerHTML=(d.data&&d.data.length)?('<div class="fav-header">⭐ مفضلتك ('+d.data.length+')</div>'+d.data.map(renderSig).join('')):'';
  });
}
function analyzeOne(){
  var code=q('q-code').value.trim().toUpperCase();
  if(!code){toast('أدخل الرمز');return;}
  var box=q('instant');
  box.innerHTML='<div style="text-align:center;color:var(--ink-3);padding:14px">جاري تحليل '+esc(code)+'…</div>';
  fetch('/analyze_one',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code:code,market:market})})
    .then(function(r){return r.json();}).then(function(d){
      if(!d.ok){box.innerHTML='<div style="color:var(--red);padding:10px">❌ '+esc(d.msg||'تعذّر')+'</div>';return;}
      box.innerHTML=(d.note?'<div class="tag warn" style="margin-top:8px;display:inline-block">'+esc(d.note)+'</div>':'')+renderSig(d.result);
    });
}
function toast(msg){var t=q('toast');t.textContent=msg;t.style.opacity='1';setTimeout(function(){t.style.opacity='0';},2600);}
function setView(v){
  q('vt-scan').classList.toggle('on',v==='scan'); q('vt-perf').classList.toggle('on',v==='perf');
  q('view-scan').classList.toggle('hidden-view',v!=='scan'); q('view-perf').classList.toggle('hidden-view',v!=='perf');
  if(v==='perf') loadPerformance();
}
function loadPerformance(){
  var days=q('ph-days').value, m=market;
  q('ph-export').href='/api/export_scan_history?market='+m;
  q('ph-list').innerHTML='<div class="empty"><div class="empty-ico">⏳</div>جاري الجلب…</div>';
  fetch('/api/scan_history?market='+m+'&days='+days).then(function(r){return r.json();}).then(function(d){
    var rows=d.rows||[], sum=d.summary||{};
    q('ph-summary').innerHTML=sum.total?
      '<div class="tally"><div class="tally-cell"><div class="tally-num">'+sum.hit_target+'</div><div class="tally-lbl">حقق هدف</div></div>'+
      '<div class="tally-cell"><div class="tally-num">'+sum.hit_stop+'</div><div class="tally-lbl">لمس وقف</div></div>'+
      '<div class="tally-cell"><div class="tally-num">'+sum.still_open+'</div><div class="tally-lbl">لسه مفتوح</div></div>'+
      '<div class="tally-cell"><div class="tally-num">'+sum.win_rate+'%</div><div class="tally-lbl">نجاح (من المحسومة)</div></div></div>':'';
    q('ph-list').innerHTML=rows.length?rows.map(renderHistCard).join(''):'<div class="empty"><div class="empty-ico">📜</div>ما فيه ترشيحات محفوظة بهالفترة لهالسوق</div>';
  });
}
function renderHistCard(r){
  var isSA=r.market==='sa';
  var sm={hit_t1:{i:'✅',l:'لمس هدف 1',c:'buy'},hit_t2:{i:'✅',l:'لمس هدف 2',c:'buy'},hit_sl:{i:'❌',l:'لمس الوقف',c:'watch'},open:{i:'⏳',l:'لسه مفتوح',c:'cond'},unknown:{i:'❓',l:'تعذر جلب السعر',c:'cond'}};
  var st=sm[r.status]||sm.unknown;
  var mk={us:'🇺🇸 أمريكي',crypto:'💰 كريبتو',sa:'🇸🇦 تاسي',junk:'⚡ خشاش'}[r.market]||r.market;
  var pnl=(r.pnl_pct==null)?'—':(r.pnl_pct>=0?'+':'')+r.pnl_pct+'%';
  return '<div class="sig"><div class="sig-verdict '+st.c+'"><div class="verdict-main"><span>'+st.i+'</span><div><div class="verdict-label">'+st.l+'</div><div class="verdict-sub">'+mk+' · '+bd(esc(r.ts))+' · قبل '+r.days_elapsed+' يوم</div></div></div>'+
    '<div class="verdict-score"><div class="verdict-score-num">'+pnl+'</div><div class="verdict-score-lbl">التغيّر</div></div></div>'+
    '<div class="sig-body"><div class="sig-name">'+esc(r.name||r.code)+'<span class="sig-code">'+esc(r.code)+'</span></div>'+
    '<div class="sig-meta" style="margin-bottom:8px"><span class="mb">'+esc(r.verdict||'')+'</span><span class="mb">سكور '+bd(r.score!=null?r.score:'—')+'</span></div>'+
    '<div class="sig-levels"><div class="lvl"><div class="lvl-lbl">دخول</div><div class="lvl-val">'+money(r.entry,isSA)+'</div></div>'+
    '<div class="lvl"><div class="lvl-lbl">هدف 1</div><div class="lvl-val">'+money(r.t1,isSA)+'</div></div>'+
    '<div class="lvl"><div class="lvl-lbl">هدف 2</div><div class="lvl-val">'+money(r.t2,isSA)+'</div></div>'+
    '<div class="lvl"><div class="lvl-lbl">وقف</div><div class="lvl-val">'+money(r.sl,isSA)+'</div></div></div>'+
    '<div class="sig-tags" style="margin-top:8px"><span class="tag">السعر الحالي <b>'+(r.current!=null?money(Number(r.current).toFixed(4),isSA):'—')+'</b></span></div></div></div>';
}
fetch('/api/meta').then(function(r){return r.json();}).then(function(d){
  if(d.sharia_us>0){q('sharia-row').style.display='flex';q('sharia-n').textContent=d.sharia_us;}
}).catch(function(){});
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
