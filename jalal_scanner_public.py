# -*- coding: utf-8 -*-
"""
jalal_scanner_public.py
════════════════════════════════════════════════════════════
نسخة الماسح العامة — مدعومة بفلتر الخشاش والزخم العالي.
"""
import os, json, time, csv, io
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, request, Response

app = Flask(name if 'name' in globals() else __name__)
USD_TO_SAR = 3.75

# قائمة واسعة تشمل أسهم الخشاش والزخم والأسهم الكبرى
US_STOCKS = {
    "AAPL":"Apple","MSFT":"Microsoft","GOOGL":"Alphabet","META":"Meta",
    "NVDA":"NVIDIA","AMD":"AMD","TSLA":"Tesla","AMZN":"Amazon",
    "PLTR":"Palantir","SOFI":"SoFi","MARA":"Marathon Digital","RIOT":"Riot Platforms",
    "CLSK":"CleanSpark","COIN":"Coinbase","HOOD":"Robinhood","NIO":"NIO",
    "XPEV":"XPeng","LCID":"Lucid","BABA":"Alibaba","FSLR":"First Solar",
    "SAVA":"Cassava Sciences","SINT":"Sintx Technologies","MULN":"Mullen Automotive"
}
SA_STOCKS = {
    "2222":"أرامكو السعودية","1180":"الأهلي السعودي","1120":"الراجحي",
    "2010":"سابك","7010":"stc","4200":"الدريس",
    "1211":"معادن","2380":"بترو رابغ","2350":"كيان السعودية",
    "7201":"بحر العرب","4140":"صادرات","1832":"صدر","2130":"صدق","4061":"أنعام القابضة"
}
CRYPTO = {
    "BTC-USD":"Bitcoin","ETH-USD":"Ethereum","SOL-USD":"Solana",
    "PEPE-USD":"Pepe","SHIB-USD":"Shiba Inu","SUI-USD":"Sui","FET-USD":"Fetch.ai"
}

def _ema(s, p): return s.ewm(span=p, adjust=False).mean()
def _rsi(s, p=14):
    d = s.diff(); g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))

def sanitize_nan(obj):
    if isinstance(obj, dict): return {k: sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list): return [sanitize_nan(v) for v in obj]
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))): return None
    return obj

def analyze_stock(code, name, is_crypto=False, is_sa=False, volatile_only=False):
    try:
        ticker = (code + ".SR") if is_sa else code
        df = yf.download(ticker, period="3mo", interval="1d", progress=False, timeout=8)
        if df.empty or len(df) < 15: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        price = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2])
        chg = round((price - prev) / prev * 100, 2)
        
        # فلتر الخشاش / السعر المنخفض والزخم
        if volatile_only and not is_crypto and not is_sa and price > 15.0:
            return None # استبعاد الأسهم الثقيلة إذا طلب المستخدم خشاش أمريكي رخيص

        rv = _rsi(df["Close"], 14)
        rsi_val = round(float(rv.iloc[-1]), 1) if not pd.isna(float(rv.iloc[-1])) else 50.0
        
        vol_now = float(df["Volume"].iloc[-1])
        vol_avg = float(df["Volume"].rolling(20).mean().iloc[-1]) or 1.0
        vr = round(vol_now / vol_avg, 2)

        if volatile_only and vr < 1.1:
            return None # يتطلب زيادة ملحوظة في السيولة

        score = 10
        if vr > 1.3: score += 4
        if chg > 1.0: score += 3
        if volatile_only: score += 5

        verdict = "HIGH_MOMENTUM" if volatile_only else ("BUY" if chg > 0 else "WATCH")
        
        atr = price * 0.03
        t1 = round(price + atr * 1.5, 2)
        sl = round(price - atr * 1.0, 2)
        ptp = round((t1 - price) / price * 100, 1)
        psl = round((price - sl) / price * 100, 1)

        return {
            "code": code, "name": name, "is_crypto": is_crypto, "is_sa": is_sa,
            "price": round(price, 2), "price_sar": round(price * USD_TO_SAR, 2) if not is_sa else round(price, 2),
            "score": score, "verdict": verdict, "trend": "⚡ خشاش / زخم عالي" if volatile_only else "صاعد",
            "confidence": min(95, score * 5), "lb": round(price * 0.99, 2), "t1": t1, "t2": round(price + atr * 3.0, 2),
            "sl": sl, "rr": 1.8, "ptp": ptp, "psl": psl, "rsi": rsi_val, "vr": vr, "chg": chg,
            "eta": "1-2 يوم", "medal": "⚡" if volatile_only else "🥇"
        }
    except:
        return None

FAV_FILE = os.path.join(os.getcwd(), "public_favorites.json")
def load_favs():
    if os.path.exists(FAV_FILE):
        try:
            with open(FAV_FILE) as f: return json.load(f)
        except: return {"us": [], "sa": [], "crypto": []}
    return {"us": [], "sa": [], "crypto": []}

def save_favs(d):
    with open(FAV_FILE, "w") as f: json.dump(d, f, ensure_ascii=False, indent=2)

@app.route("/scan", methods=["GET", "POST"])
def scan_route():
    market = request.args.get("market", "us")
    volatile_only = request.args.get("volatile", "false").lower() == "true"
    
    stocks = US_STOCKS if market=="us" else (SA_STOCKS if market=="sa" else CRYPTO)
    is_sa = (market == "sa")
    is_crypto = (market == "crypto")
    
    results = []
    for code, name in stocks.items():
        res = analyze_stock(code, name, is_crypto, is_sa, volatile_only)
        if res: results.append(res)
        
    results.sort(key=lambda x: (x["score"], x["chg"]), reverse=True)
    top = results[:8]
    
    data = {
        "data": top,
        "last": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "scanned": len(stocks),
        "status": "done"
    }
    app.config[f"scan_res_{market}_{volatile_only}"] = data
    return jsonify(sanitize_nan({"ok": True, **data}))

@app.route("/results", methods=["GET"])
def results_route():
    market = request.args.get("market", "us")
    volatile_only = request.args.get("volatile", "false").lower() == "true"
    if market not in ["us", "sa", "crypto"]: market = "us"
    key = f"scan_res_{market}_{volatile_only}"
    data = app.config.get(key, {"data": [], "last": None, "scanned": 0, "status": "done"})
    return jsonify(sanitize_nan(data))

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
.scan-btn { width:100%; padding:14px; border-radius:var(--r); border:0; background:var(--brand); color:#2a1a00; font-size:15px; font-weight:800; cursor:pointer; margin-bottom:12px; box-shadow:0 4px 12px rgba(239,159,39,0.2); }
.scan-meta { text-align:center; color:var(--ink-3); font-size:11px; margin-bottom:12px; }
.empty { text-align:center; color:var(--ink-3); padding:40px 0; font-size:13px; }
.empty-ico { font-size:32px; margin-bottom:8px; }
.sig { background:var(--surface); border:1px solid var(--line); border-radius:var(--r-lg); margin-top:12px; overflow:hidden; }
.sig-verdict { padding:11px 16px; display:flex; align-items:center; justify-content:space-between; background:var(--green-soft); border-bottom:1px solid #1a4a3a; }
.verdict-main { display:flex; align-items:center; gap:9px; }
.verdict-label { font-weight:800; font-size:15px; }
.verdict-sub { font-size:11px; color:var(--ink-3); }
.sig-body { padding:14px 16px; }
.sig-top { display:flex; align-items:flex-start; gap:11px; }
.sig-id { flex:1; }
.sig-name { font-weight:700; font-size:14px; display:flex; align-items:center; gap:6px; }
.sig-code { color:var(--ink-3); font-size:12px; }
.sig-px { text-align:left; }
.sig-px-main { font-size:17px; font-weight:800; }
.sig-tags { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin:10px 0; }
.tag { font-size:11px; background:var(--surface-2); padding:5px 9px; border-radius:8px; color:var(--ink-2); }
.sig-trade-btn { margin-right:auto; background:var(--green); color:#fff; border:0; border-radius:9px; padding:6px 14px; font-size:12px; font-weight:700; text-decoration:none; }
.sig-levels { display:grid; grid-template-columns:repeat(4,1fr); gap:6px; background:var(--surface-2); border-radius:var(--r); padding:8px; }
.lvl { text-align:center; } .lvl-lbl { font-size:10px; color:var(--ink-3); }
.lvl-val { font-size:12px; font-weight:700; margin-top:2px; }
.loader { display:none; text-align:center; padding:16px; color:var(--brand); font-weight:700; font-size:13px; background:var(--surface); border-radius:var(--r); margin-bottom:12px; border:1px solid var(--line); }
.loader.show { display:block; }
</style>
</head>
<body>
<div class="shell">
  <div class="hdr">
    <h1>⚡ ماسح جلال رادار</h1>
    <p>نسخة الفحص السريع مع فلتر الخشاش والزخم</p>
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

  <button class="scan-btn" id="scan-btn" onclick="startScan()">🔍 مسح السوق الآن</button>
  <div class="scan-meta" id="scan-meta"></div>
  <div class="loader" id="loader">⚡ جارٍ فحص الزخم والخشاش...</div>

  <div id="signals"><div class="empty"><div class="empty-ico">📡</div>اضغط مسح السوق الآن لبدء الفحص</div></div>
</div>

<script>
var market = 'us';
function onMarketChange() {
  market = document.getElementById('market-select').value;
  var lbls={'us':'🔍 مسح السوق الأمريكي','sa':'🔍 مسح السوق السعودي (تاسي)','crypto':'🔍 مسح الكريبتو 💰'};
  document.getElementById('scan-btn').innerHTML = lbls[market]||'🔍 مسح';
}

function startScan() {
  var isVolatile = document.getElementById('volatile-chk').checked;
  var btn = document.getElementById('scan-btn');
  btn.disabled = true;
  document.getElementById('loader').classList.add('show');
  document.getElementById('signals').innerHTML = '';

  fetch('/scan?market='+encodeURIComponent(market)+'&volatile='+(isVolatile?'true':'false'))
    .then(function(r){ return r.json(); })
    .then(function(d){
      btn.disabled = false;
      document.getElementById('loader').classList.remove('show');
      renderResults(d);
    })
    .catch(function(err){
      btn.disabled = false;
      document.getElementById('loader').classList.remove('show');
      alert('حدث خطأ في الاتصال');
    });
}

function renderResults(d) {
  var data = d.data||[];
  document.getElementById('scan-meta').textContent = d.last ? ('آخر مسح: '+d.last+' · عدد الأسهم: '+d.scanned) : '';
  document.getElementById('signals').innerHTML = data.length ?
    data.map(renderSig).join('') :
    '<div class="empty"><div class="empty-ico">📡</div>لا توجد نتائج مطابقة لشروط الخشاش والزخم حالياً</div>';
}

function renderSig(s) {
  var isSA = s.is_sa === true;
  var cur = isSA ? ' ر.س' : '$'; var curPre = isSA ? '' : '$';
  var tvSym = isSA ? ('TADAWUL-'+s.code) : s.code;
  return '<div class="sig">'+
    '<div class="sig-verdict"><div class="verdict-main"><span>'+s.medal+'</span>'+
    '<div><div class="verdict-label">'+s.trend+'</div><div class="verdict-sub">تغير يومي: '+s.chg+'% · سيولة x'+s.vr+'</div></div></div>'+
    '<div class="verdict-score"><b>السكور: '+s.score+'</b></div></div>'+
    '<div class="sig-body"><div class="sig-top">'+
    '<div class="sig-id"><div class="sig-name">'+s.name+' <span class="sig-code">'+s.code+'</span></div>'+
    '<div style="font-size:11px;color:var(--ink-3);margin-top:2px;">RSI: '+s.rsi+'</div></div>'+
    '<div class="sig-px"><div class="sig-px-main">'+curPre+s.price+cur+'</div></div></div>'+
    '<div class="sig-tags"><span class="tag">الهدف: <b>'+curPre+s.t1+cur+'</b></span>'+
    '<span class="tag">الوقف: <b>'+curPre+s.sl+cur+'</b></span>'+
    '<a class="sig-trade-btn" href="https://www.tradingview.com/symbols/'+tvSym+'/" target="_blank">📈 TradingView</a></div>'+
    '<div class="sig-levels"><div class="lvl"><div class="lvl-lbl">دخول</div><div class="lvl-val">'+curPre+s.lb+cur+'</div></div>'+
    '<div class="lvl"><div class="lvl-lbl">هدف 1</div><div class="lvl-val">'+curPre+s.t1+cur+'</div></div>'+
    '<div class="lvl"><div class="lvl-lbl">هدف 2</div><div class="lvl-val">'+curPre+s.t2+cur+'</div></div>'+
    '<div class="lvl"><div class="lvl-lbl">وقف</div><div class="lvl-val">'+curPre+s.sl+cur+'</div></div></div></div></div>';
}
</script>
</body>
</html>"""

@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
