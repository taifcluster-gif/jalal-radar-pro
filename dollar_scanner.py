# -*- coding: utf-8 -*-
"""
dollar_scanner.py (v2)
==================
موديول مستقل تمامًا عن البوت الأساسي (جلال رادار برو) — استراتيجية جانبية
لأسهم "الدولار" (0.5$ - 2$) اللي بترند صاعد، برأس مال افتراضي 1000$.

لا يلمس أي كود من البوت الرئيسي — تقدر تضيفه بملف منفصل بنفس المستودع
وتشغّله كـ endpoint منفصل أو job منفصل على Render.

(v2 — بعد المراجعة):
- إصلاح: كل مرشح كان يسوي 3 طلبات yfinance منفصلة (سيولة + سكور + أرباح) —
  صار طلب واحد بس (get_history_once) ونعيد استخدامه بكل مكان.
- إصلاح: البوت الرئيسي ما يستخدم مكتبة alpaca-trade-api إطلاقاً (يستخدم HTTP
  مباشر) — أضفت طبقة _alpaca_req() بديلة ما تحتاج المكتبة، عشان يتوافق مع
  بيئة النشر الفعلية بدون تثبيت اعتمادية جديدة.
- إصلاح: فلتر تاريخ الأرباح كان يفترض شكل ثابت لـ ticker.calendar، وهذا
  تغيّر بين إصدارات yfinance ويطلع خطأ بصمت. صار أكثر تحفظاً.
- إضافة: تحميل قائمة الأسهم الشرعية تلقائياً من الملفين اللي رفعتهم.

⚠️ هذا الملف "سكانر" بس — يكتشف الفرص، ما ينفّذ أوامر شراء/بيع فعلية.
تنفيذ الأوامر خطوة منفصلة تحتاج تأكيد صريح قبل ما تُضاف.

الاعتماديات المطلوبة (requirements.txt) — بدون alpaca-trade-api:
    yfinance
    pandas
    numpy
"""

import os
import time
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import yfinance as yf
import urllib.request
import json as _json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dollar_scanner")

CONFIG = {
    "capital_base": 1000.0,
    "max_position_pct": 0.07,
    "price_min": 0.5,
    "price_max": 2.0,
    "min_avg_volume": 500_000,
    "ema_period": 9,
    "breakout_lookback_days": 7,
    "volume_spike_multiplier": 1.5,
    "min_score_to_enter": 2,
    "earnings_blackout_days": 1,
    "stop_loss_pct": 0.05,
    "take_profit_pct": 0.10,
    "sharia_compliant_symbols": None,
    "sharia_file_us": os.path.join(os.getcwd(), "اسهم_شرعية_سوق_امريكي.txt"),
    "sharia_file_sa": os.path.join(os.getcwd(), "اسهم_شرعية_السوق_السعودي.txt"),
    "alpaca_key": os.environ.get("ALPACA_API_KEY_DOLLAR", ""),
    "alpaca_secret": os.environ.get("ALPACA_SECRET_KEY_DOLLAR", ""),
    "alpaca_base_url": os.environ.get("ALPACA_BASE_URL_DOLLAR", "https://paper-api.alpaca.markets"),
}


def load_sharia_list(market="us"):
    """يقرأ ملف الأسهم الشرعية الخام ويرجع set() برموز الأسهم فقط."""
    path = CONFIG["sharia_file_us"] if market == "us" else CONFIG["sharia_file_sa"]
    if not os.path.exists(path):
        logger.warning(f"ملف القائمة الشرعية غير موجود: {path}")
        return set()
    symbols = set()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f.read().splitlines():
                line = line.strip()
                if "\t" in line:
                    parts = line.split("\t")
                    # (v2.1) إصلاح: رموز السوق السعودي أرقام صرفة (4326، 2222...)
                    # — الاستبعاد القديم لأي رمز رقمي كان يفرّغ القائمة السعودية بالكامل.
                    # التمييز الصحيح: نص الترويسة "رقم الصف" لحاله بدون تاب، مو الرمز نفسه.
                    if len(parts) >= 2:
                        sym = parts[0].strip()
                        if sym:
                            symbols.add(sym)
    except Exception as e:
        logger.warning(f"فشل قراءة القائمة الشرعية ({path}): {e}")
    logger.info(f"القائمة الشرعية ({market}): {len(symbols)} رمز")
    return symbols


def init_sharia_filter():
    us_list = load_sharia_list("us")
    if us_list:
        CONFIG["sharia_compliant_symbols"] = us_list
    return us_list


def _alpaca_req(path, base=None):
    base = base or CONFIG["alpaca_base_url"]
    req = urllib.request.Request(
        base + path,
        headers={
            "APCA-API-KEY-ID": CONFIG["alpaca_key"],
            "APCA-API-SECRET-KEY": CONFIG["alpaca_secret"],
        },
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return _json.loads(r.read().decode())


def get_candidate_universe():
    try:
        assets = _alpaca_req("/v2/assets?status=active&asset_class=us_equity")
    except Exception as e:
        logger.error(f"فشل جلب الأصول من Alpaca: {e}")
        return []

    tradable = [a["symbol"] for a in assets
                if a.get("tradable") and a.get("exchange") in ("NASDAQ", "NYSE", "AMEX")]
    logger.info(f"إجمالي الأصول القابلة للتداول من Alpaca: {len(tradable)}")

    if CONFIG["sharia_compliant_symbols"]:
        before = len(tradable)
        tradable = [s for s in tradable if s in CONFIG["sharia_compliant_symbols"]]
        logger.info(f"فلتر الالتزام الشرعي: {before} → {len(tradable)} رمز")

    candidates = []
    batch_size = 100
    for i in range(0, len(tradable), batch_size):
        batch = tradable[i:i + batch_size]
        try:
            symbols_param = ",".join(batch)
            snaps = _alpaca_req(f"/v2/stocks/snapshots?symbols={symbols_param}",
                                 base="https://data.alpaca.markets")
        except Exception as e:
            logger.warning(f"تعذر جلب batch {i}: {e}")
            continue
        for sym, snap in (snaps or {}).items():
            latest_trade = snap.get("latestTrade") or {}
            price = latest_trade.get("p")
            if price and CONFIG["price_min"] <= price <= CONFIG["price_max"]:
                candidates.append(sym)
        time.sleep(0.2)

    logger.info(f"عدد الرموز بعد فلتر السعر: {len(candidates)}")
    return candidates


def get_history_once(symbol, period="30d"):
    try:
        hist = yf.download(symbol, period=period, progress=False, auto_adjust=True)
        if hist.empty or len(hist) < 5:
            return None
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        return hist
    except Exception as e:
        logger.warning(f"{symbol}: خطأ بجلب البيانات التاريخية - {e}")
        return None


def liquidity_filter(symbol, hist):
    if hist is None or hist.empty:
        return False, None
    avg_vol = hist["Volume"].mean()
    return bool(avg_vol >= CONFIG["min_avg_volume"]), float(avg_vol)


def score_symbol(symbol, hist):
    if hist is None or hist.empty or len(hist) < CONFIG["breakout_lookback_days"] + 2:
        return None

    close = hist["Close"]
    volume = hist["Volume"]
    last_close = float(close.iloc[-1])
    last_volume = float(volume.iloc[-1])
    prev_close = float(close.iloc[-2])

    ema = close.ewm(span=CONFIG["ema_period"], adjust=False).mean()
    cond_ema = bool(last_close > float(ema.iloc[-1]))

    lookback = close.iloc[-(CONFIG["breakout_lookback_days"] + 1):-1]
    recent_high = float(lookback.max())
    cond_breakout = bool(last_close > recent_high)

    avg_volume = float(volume.iloc[-11:-1].mean())
    is_green = last_close > prev_close
    cond_volume = bool(
        avg_volume > 0
        and last_volume >= avg_volume * CONFIG["volume_spike_multiplier"]
        and is_green
    )

    score = sum([cond_ema, cond_breakout, cond_volume])

    return {
        "symbol": symbol, "score": score,
        "cond_ema": cond_ema, "cond_breakout": cond_breakout, "cond_volume": cond_volume,
        "last_close": last_close,
        "avg_volume": avg_volume if not np.isnan(avg_volume) else None,
        "pass": score >= CONFIG["min_score_to_enter"],
    }


def exclusion_filter(symbol, hist):
    ok_liquidity, avg_vol = liquidity_filter(symbol, hist)
    if not ok_liquidity:
        return True, f"سيولة ضعيفة (متوسط حجم: {avg_vol})"

    try:
        ticker = yf.Ticker(symbol)
        cal = ticker.calendar
        earnings_date = None
        if isinstance(cal, dict):
            earnings_date = cal.get("Earnings Date")
            if isinstance(earnings_date, list) and earnings_date:
                earnings_date = earnings_date[0]
        elif cal is not None and hasattr(cal, "loc") and "Earnings Date" in getattr(cal, "index", []):
            ed = cal.loc["Earnings Date"]
            earnings_date = ed.iloc[0] if hasattr(ed, "iloc") else ed
        if earnings_date is not None:
            delta_days = abs((pd.Timestamp(earnings_date).date() - datetime.now().date()).days)
            if delta_days <= CONFIG["earnings_blackout_days"]:
                return True, f"قريب من تاريخ الأرباح ({earnings_date})"
    except Exception:
        pass

    return False, None


def calc_position_size(price, capital=None):
    capital = capital or CONFIG["capital_base"]
    max_dollar_amount = capital * CONFIG["max_position_pct"]
    qty = max_dollar_amount / price
    return round(qty, 3), round(max_dollar_amount, 2)


def run_scan():
    if not CONFIG["sharia_compliant_symbols"]:
        init_sharia_filter()

    candidates = get_candidate_universe()
    opportunities = []

    for symbol in candidates:
        hist = get_history_once(symbol)
        if hist is None:
            continue

        excluded, reason = exclusion_filter(symbol, hist)
        if excluded:
            logger.info(f"{symbol}: مستبعد - {reason}")
            continue

        result = score_symbol(symbol, hist)
        if result is None or not result["pass"]:
            continue

        qty, dollar_amount = calc_position_size(result["last_close"])
        result["suggested_qty"] = qty
        result["suggested_dollar_amount"] = dollar_amount
        result["stop_loss_price"] = round(result["last_close"] * (1 - CONFIG["stop_loss_pct"]), 4)
        result["take_profit_price"] = round(result["last_close"] * (1 + CONFIG["take_profit_pct"]), 4)
        opportunities.append(result)

    opportunities.sort(key=lambda x: x["score"], reverse=True)
    logger.info(f"عدد الفرص الجاهزة للدخول: {len(opportunities)}")
    return opportunities


if __name__ == "__main__":
    if not CONFIG["alpaca_key"] or not CONFIG["alpaca_secret"]:
        raise SystemExit(
            "لازم تحط ALPACA_API_KEY_DOLLAR و ALPACA_SECRET_KEY_DOLLAR كمتغيرات بيئة "
            "منفصلة تماماً عن مفاتيح البوت الرئيسي (ALPACA_API_KEY/ALPACA_SECRET_KEY)"
        )
    # ── فحص أمان: نتأكد إن المفاتيح فعلاً مختلفة عن البوت الرئيسي ──
    # لو انحطت نفس المفاتيح بالغلط بمتغيرات "دولار سكانر"، نرفض نشتغل
    # بدل ما نستمر بصمت على نفس حساب البوت الرئيسي.
    _main_key = os.environ.get("ALPACA_API_KEY", "")
    if _main_key and _main_key == CONFIG["alpaca_key"]:
        raise SystemExit(
            "⚠️ توقف أمان: ALPACA_API_KEY_DOLLAR نفس مفتاح البوت الرئيسي بالضبط. "
            "لازم حساب Paper منفصل تماماً — راجع خطوات الإعداد قبل ما تكمل."
        )
    opps = run_scan()
    for o in opps:
        print(o)
