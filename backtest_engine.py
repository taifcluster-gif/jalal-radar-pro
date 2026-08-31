# ════════════════════════════════════════════════════════════
# backtest_engine.py — محرك اختبار تاريخي (v1)
# ────────────────────────────────────────────────────────────
# الفكرة: بدل انتظار شهرين بالوقت الحقيقي عشان نجمع 20-30 صفقة،
# نشغّل نفس منطق الإشارة بالضبط (نفس معادلة الـ11 شرط ونفس نظام
# الـTier) على سنتين من البيانات التاريخية — يعطينا مئات الصفقات
# خلال دقايق.
#
# مهم: هذا الملف يعيد استخدام نفس صيغ المؤشرات (EMA/RSI/MACD/ADX/
# Stochastic) بحساب واحد على كامل السلسلة الزمنية، مو تكرار الحساب
# كل يوم — لأن هذي الصيغ "سببية" (Causal): قيمة المؤشر باليوم i
# تعتمد بس على الأيام اللي قبله، ما تتسرب فيها معلومة من المستقبل.
# هذا يخلي النتيجة عادلة وسريعة بنفس الوقت.
#
# ⚠️ لا يشغّل من هذي البيئة (ما فيه وصول شبكي لـyfinance من هنا) —
# لازم يشتغل من داخل التطبيق نفسه على Render، اللي أصلاً يستخدم
# get_history() لنفس الغرض بالبوت الحي.
# ════════════════════════════════════════════════════════════
import pandas as pd
from datetime import datetime


def _ema(s, p): return s.ewm(span=p, adjust=False).mean()
def _sma(s, p): return s.rolling(p).mean()

def _rsi(s, p=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(p).mean()
    down = -d.clip(upper=0).rolling(p).mean()
    rsv = up / down.replace(0, 1e-10)
    return 100 - (100 / (1 + rsv))

def _macd(s):
    e12, e26 = _ema(s, 12), _ema(s, 26)
    line = e12 - e26
    sig = _ema(line, 9)
    return line, sig, line - sig

def _adx(h, l, c, p=14):
    up = h.diff(); down = -l.diff()
    plus_dm = ((up > down) & (up > 0)) * up
    minus_dm = ((down > up) & (down > 0)) * down
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(p).mean()
    plus_di = 100 * (plus_dm.rolling(p).mean() / atr.replace(0,1e-10))
    minus_di = 100 * (minus_dm.rolling(p).mean() / atr.replace(0,1e-10))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0,1e-10)
    return dx.rolling(p).mean()

def _atr(h, l, c, p=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def _stoch(h, l, c, k=14, d=3):
    lowest = l.rolling(k).min(); highest = h.rolling(k).max()
    pk = 100 * (c - lowest) / (highest - lowest).replace(0,1e-10)
    return pk, pk.rolling(d).mean()


def compute_indicators(df):
    """يحسب كل المؤشرات مرة وحدة على كامل السلسلة — سببية بطبيعتها."""
    c = df.copy()
    ind = {}
    ind["e20"] = _ema(c["Close"], 20)
    ind["e50"] = _ema(c["Close"], 50)
    ind["e100"] = _ema(c["Close"], 100)
    ind["e200"] = _ema(c["Close"], 200)
    ind["rsi"] = _rsi(c["Close"])
    ind["macd_l"], ind["macd_s"], ind["macd_h"] = _macd(c["Close"])
    ind["adx"] = _adx(c["High"], c["Low"], c["Close"])
    ind["stoch_k"], ind["stoch_d"] = _stoch(c["High"], c["Low"], c["Close"])
    ind["vol_avg"] = _sma(c["Volume"], 20)
    ind["atr"] = _atr(c["High"], c["Low"], c["Close"])
    return ind


def score_at(c, ind, i):
    """
    نفس معادلة JRF (11 شرط) بالضبط من analyze_symbol() — بس هنا نقرأ
    من مؤشرات محسوبة سلفاً بدل استدعاء get_history() كل مرة.
    ترجع (score, verdict, entry_tier) أو (None, None, None) لو البيانات ناقصة.
    """
    try:
        price = float(c["Close"].iloc[i])
        e20 = float(ind["e20"].iloc[i]); e50 = float(ind["e50"].iloc[i])
        e100 = float(ind["e100"].iloc[i]); e200 = float(ind["e200"].iloc[i])
        rsi_now = float(ind["rsi"].iloc[i]); rsi_prev = float(ind["rsi"].iloc[i-1])
        ml = float(ind["macd_l"].iloc[i]); sg = float(ind["macd_s"].iloc[i])
        mh_now = float(ind["macd_h"].iloc[i]); mh_prev = float(ind["macd_h"].iloc[i-1])
        adx_now = float(ind["adx"].iloc[i]); adx_prev = float(ind["adx"].iloc[i-1])
        sk = float(ind["stoch_k"].iloc[i]); sd = float(ind["stoch_d"].iloc[i])
        vol_avg = float(ind["vol_avg"].iloc[i])
        atr_v = float(ind["atr"].iloc[i])
        if any(pd.isna(x) for x in [e20,e50,e100,e200,rsi_now,ml,sg,adx_now,sk,vol_avg,atr_v]):
            return None
    except Exception:
        return None

    if vol_avg <= 0 or price <= 0:
        return None

    raw_vr = float(c["Volume"].iloc[i]) / vol_avg
    vr_now = raw_vr  # (تبسيط: بدون تصحيح "شمعة اليوم الجزئية" — غير منطبق على بيانات يومية مقفلة)

    c1 = price > e20
    c2 = e20 > e50
    c3 = price > e200
    c4 = ml > sg
    c5 = mh_now > mh_prev
    c6 = 40 <= rsi_now <= 70
    c7 = rsi_now > rsi_prev
    c8 = adx_now > 20
    c9 = sk > 20 and sk > sd
    c10 = vr_now > 1.2
    c11 = adx_now > adx_prev

    score = (2*c1+2*c2+2*c3+2*c4+1*c5+2*c6+1*c7+2*c8+2*c9+2*c10+2*c11)

    lb = round(price - atr_v*1.0, 2)  # (v3.4) وقف خسارة أضيق — كان 1.5×ATR
    t1 = round(price + atr_v*2.0, 2)
    rr = round((t1-price)/max(price-lb, 0.01), 2)
    ad = price > e20
    aw = price > e100

    if score >= 15 and rr >= 1.3:
        verdict = "BUY"
    elif score >= 12 and rr >= 1.0 and ad:
        verdict = "BUY_COND"
    elif score >= 9:
        verdict = "WATCH"
    else:
        verdict = "AVOID"

    entry_tier = None
    if verdict == "BUY":
        entry_tier = 1
    elif verdict == "BUY_COND" and vr_now >= 1.2:
        entry_tier = 2
    elif verdict == "BUY_COND":
        entry_tier = 3
    elif verdict == "WATCH" and score >= 9 and vr_now >= 2.0 and adx_now > adx_prev:
        entry_tier = 3

    return {"score": score, "verdict": verdict, "entry_tier": entry_tier,
            "sl": lb, "atr": atr_v}


TIER_PCT = {1: 0.02, 2: 0.012, 3: 0.005}  # نفس نسب risk_manager.py


def backtest_symbol(df, symbol, capital=100000.0,
                     tp1_pct=3.0, tp2_pct=5.0, trail_pct=1.5,
                     disable_tier1=True, disable_tier3=True):
    """
    يشغّل محاكاة كاملة (دخول + خروج) على سهم واحد عبر كل تاريخه المتاح.
    يرجع قائمة صفقات مقفولة، كل وحدة فيها: entry_tier, entry_price,
    exit_price, exit_reason, pnl, pnl_pct, entry_date, exit_date.
    """
    if df is None or df.empty or len(df) < 210:
        return []

    ind = compute_indicators(df)
    c = df
    trades = []
    pos = None  # {"tier","entry_price","entry_idx","qty","orig_qty","sold_tp1","sold_tp2","highest","sl"}

    for i in range(210, len(c)):
        date_i = c.index[i]
        high_i = float(c["High"].iloc[i]); low_i = float(c["Low"].iloc[i])
        close_i = float(c["Close"].iloc[i])

        # ── أول شي: لو عندنا مركز مفتوح، نفحص الخروج (وقف خسارة / أهداف / تريلينج) ──
        if pos:
            pnl_pct_now = (high_i - pos["entry_price"]) / pos["entry_price"] * 100
            if high_i > pos["highest"]:
                pos["highest"] = high_i

            # Break-even عند +1.5% قبل TP1
            if not pos["sold_tp1"] and pnl_pct_now >= 1.5:
                pos["sl"] = max(pos["sl"], pos["entry_price"])

            exited = False
            # وقف الخسارة (نتحقق أول شي — تحفّظياً لو انلمس بنفس اليوم مع الهدف)
            if not pos["sold_tp1"] and low_i <= pos["sl"]:
                _close_trade(trades, pos, pos["sl"], date_i, "وقف الخسارة", symbol)
                pos = None; exited = True

            if not exited and not pos_is_none(pos):
                tp1_price = pos["entry_price"] * (1 + tp1_pct/100)
                tp2_price = pos["entry_price"] * (1 + tp2_pct/100)
                if not pos["sold_tp1"] and high_i >= tp1_price:
                    pos["sold_tp1"] = True
                    trades.append(_partial(pos, tp1_price, date_i, f"هدف 1 (+{tp1_pct}%)", symbol, 0.40))
                elif pos["sold_tp1"] and not pos["sold_tp2"] and high_i >= tp2_price:
                    pos["sold_tp2"] = True
                    trades.append(_partial(pos, tp2_price, date_i, f"هدف 2 (+{tp2_pct}%)", symbol, 0.30))
                elif pos["sold_tp1"]:
                    trail_price = pos["highest"] * (1 - trail_pct/100)
                    if low_i <= trail_price:
                        _close_trade(trades, pos, trail_price, date_i, "وقف متحرك", symbol)
                        pos = None

        # ── ثانياً: لو ما فيه مركز، نفحص إشارة دخول جديدة ──
        if not pos:
            sig = score_at(c, ind, i)
            if sig and sig["entry_tier"]:
                tier = sig["entry_tier"]
                if tier == 1 and disable_tier1:
                    continue  # (v3.4) Tier 1 معطّل — نفس قرار البوت الحي
                if tier == 3 and disable_tier3:
                    continue  # (v3.4) Tier 3 معطّل — كان يخسر بعد ما شلنا Tier 1
                alloc = capital * TIER_PCT[tier]
                qty = alloc / close_i
                pos = {"tier": tier, "entry_price": close_i, "entry_idx": i,
                       "qty": qty, "orig_qty": qty, "sold_tp1": False, "sold_tp2": False,
                       "highest": close_i, "sl": sig["sl"], "entry_date": date_i}

    return trades


def pos_is_none(pos):
    return pos is None


def _partial(pos, price, date, reason, symbol, frac):
    q = pos["orig_qty"] * frac
    pnl = (price - pos["entry_price"]) * q
    pos["qty"] -= q
    return {"symbol": symbol, "tier": pos["tier"], "entry_price": round(pos["entry_price"],2),
            "exit_price": round(price,2), "reason": reason, "qty": round(q,4),
            "pnl": round(pnl,2), "pnl_pct": round((price-pos["entry_price"])/pos["entry_price"]*100,2),
            "entry_date": str(pos["entry_date"].date()) if hasattr(pos["entry_date"],"date") else str(pos["entry_date"]),
            "exit_date": str(date.date()) if hasattr(date,"date") else str(date)}


def _close_trade(trades, pos, price, date, reason, symbol):
    q = pos["qty"]
    pnl = (price - pos["entry_price"]) * q
    trades.append({"symbol": symbol, "tier": pos["tier"], "entry_price": round(pos["entry_price"],2),
                    "exit_price": round(price,2), "reason": reason, "qty": round(q,4),
                    "pnl": round(pnl,2), "pnl_pct": round((price-pos["entry_price"])/pos["entry_price"]*100,2),
                    "entry_date": str(pos["entry_date"].date()) if hasattr(pos["entry_date"],"date") else str(pos["entry_date"]),
                    "exit_date": str(date.date()) if hasattr(date,"date") else str(date)})


def summarize(all_trades):
    """يجمّع نتيجة كل الأسهم لتقرير واحد، بما فيه تفصيل حسب الـTier."""
    if not all_trades:
        return {"total_trades": 0, "note": "ما طلعت أي إشارة دخول عبر الفترة المختبرة"}

    wins = [t for t in all_trades if t["pnl"] > 0]
    losses = [t for t in all_trades if t["pnl"] <= 0]
    total_pnl = sum(t["pnl"] for t in all_trades)

    by_tier = {}
    for tier in (1, 2, 3):
        tt = [t for t in all_trades if t["tier"] == tier]
        if not tt: continue
        tw = [t for t in tt if t["pnl"] > 0]
        by_tier[tier] = {
            "trades": len(tt), "win_rate": round(len(tw)/len(tt)*100, 1),
            "total_pnl": round(sum(t["pnl"] for t in tt), 2),
            "avg_win": round(sum(t["pnl"] for t in tw)/len(tw), 2) if tw else 0,
            "avg_loss": round(sum(t["pnl"] for t in tt if t["pnl"]<=0)/max(len(tt)-len(tw),1), 2),
        }

    return {
        "total_trades": len(all_trades),
        "wins": len(wins), "losses": len(losses),
        "win_rate": round(len(wins)/len(all_trades)*100, 1),
        "total_pnl": round(total_pnl, 2),
        "avg_win": round(sum(t["pnl"] for t in wins)/len(wins), 2) if wins else 0,
        "avg_loss": round(sum(t["pnl"] for t in losses)/len(losses), 2) if losses else 0,
        "by_tier": by_tier,
    }
