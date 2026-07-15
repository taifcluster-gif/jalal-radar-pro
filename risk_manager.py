# -*- coding: utf-8 -*-
"""
risk_manager.py — وحدة إدارة المخاطر لـ Jalal Radar Pro (JRF v3.3)
====================================================================
الملف جديد بالكامل — ارفعه بجانب الملف الرئيسي في نفس المجلد.

القواعد المطبقة:
  1) الحجم النسبي الذكي (intraday relative volume) — يقارن حجم اليوم
     بمتوسط الحجم *لنفس الوقت من الجلسة* بدل الحجم اليومي الكامل.
  2) حد الصفقات اليومي: 8 بدل 5.
  3) سقف التعرض الكلي: مجموع المراكز المفتوحة لا يتجاوز 15% من رأس المال.
  4) سقف التعرض للسهم الواحد: 3% من رأس المال (شامل المراكز المفتوحة).
  5) حد القطاع: أقصى صفقتين جديدتين لكل قطاع في اليوم الواحد.

كل الدوال تُرجع (True/False, سبب_بالعربي) للتسجيل في اللوق.
"""

from datetime import datetime, time as dtime, timezone

# ==========================================================
# 1) الإعدادات — عدّل الأرقام من هنا فقط
# ==========================================================
MAX_TRADES_PER_DAY      = 8      # كان 5
MAX_TOTAL_EXPOSURE_PCT  = 0.15   # 15% من رأس المال كحد أقصى للمراكز المفتوحة
MAX_SYMBOL_EXPOSURE_PCT = 0.03   # 3% كحد أقصى لتعرض السهم الواحد
MAX_TRADES_PER_SECTOR   = 2      # صفقتان جديدتان لكل قطاع في اليوم
MIN_RELATIVE_VOLUME     = 1.0    # الحد الأدنى للحجم النسبي الذكي (1.0 = مساوٍ للمعتاد)

# ==========================================================
# 2) خريطة القطاعات — غطّ كل أسهم قائمتك (غير المذكور = OTHER)
# ==========================================================
SECTOR_MAP = {
    # تقنية / سايبر / سحابة
    "CRWD": "TECH", "PANW": "TECH", "FTNT": "TECH", "NET": "TECH",
    "SNOW": "TECH", "DDOG": "TECH", "META": "TECH", "AMD": "TECH",
    "NVDA": "TECH", "MSFT": "TECH", "GOOGL": "TECH", "AAPL": "TECH",
    "TXN": "TECH", "AVGO": "TECH", "ADBE": "TECH", "CRM": "TECH",
    # طاقة
    "EOG": "ENERGY", "XOM": "ENERGY", "CVX": "ENERGY", "COP": "ENERGY",
    # صناعة / مواد / نقل
    "NUE": "INDUSTRIAL", "APD": "INDUSTRIAL", "ODFL": "INDUSTRIAL",
    "CAT": "INDUSTRIAL", "DE": "INDUSTRIAL", "UNP": "INDUSTRIAL",
    # صحة
    "TMO": "HEALTH", "ISRG": "HEALTH", "LLY": "HEALTH", "ABT": "HEALTH",
    # استهلاكي
    "SBUX": "CONSUMER", "NKE": "CONSUMER", "COST": "CONSUMER",
    "MCD": "CONSUMER", "PG": "CONSUMER",
}

def get_sector(symbol: str) -> str:
    return SECTOR_MAP.get(symbol.upper(), "OTHER")


# ==========================================================
# 3) الحجم النسبي الذكي (intraday relative volume)
# ==========================================================
def session_elapsed_fraction(now_utc=None) -> float:
    """
    نسبة ما مضى من جلسة السوق الأمريكي (13:30 → 20:00 UTC).
    ترجع قيمة بين 0.05 و 1.0 (لا نسمح بأقل من 0.05 لتفادي القسمة على صفر
    والتضخيم المجنون في أول دقائق).
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    open_minutes  = 13 * 60 + 30   # 13:30 UTC
    close_minutes = 20 * 60        # 20:00 UTC
    now_minutes   = now_utc.hour * 60 + now_utc.minute

    if now_minutes <= open_minutes:
        return 0.05
    if now_minutes >= close_minutes:
        return 1.0
    frac = (now_minutes - open_minutes) / (close_minutes - open_minutes)
    return max(0.05, min(frac, 1.0))


# منحنى شكل الحجم داخل الجلسة (U-shape): أول الجلسة وآخرها أنشط.
# المفتاح = نسبة الوقت المنقضي، القيمة = النسبة التراكمية المتوقعة من حجم اليوم.
_VOLUME_CURVE = [
    (0.00, 0.00), (0.08, 0.15), (0.15, 0.25), (0.25, 0.35),
    (0.40, 0.47), (0.50, 0.55), (0.60, 0.62), (0.75, 0.72),
    (0.85, 0.82), (0.95, 0.93), (1.00, 1.00),
]

def expected_volume_fraction(elapsed: float) -> float:
    """النسبة المتوقعة من حجم اليوم الكامل عند نقطة زمنية معينة (استيفاء خطي)."""
    for i in range(1, len(_VOLUME_CURVE)):
        t0, v0 = _VOLUME_CURVE[i - 1]
        t1, v1 = _VOLUME_CURVE[i]
        if elapsed <= t1:
            if t1 == t0:
                return v1
            return v0 + (v1 - v0) * (elapsed - t0) / (t1 - t0)
    return 1.0


def smart_relative_volume(today_volume: float, avg_daily_volume: float,
                          now_utc=None) -> float:
    """
    الحجم النسبي الذكي:
      = حجم اليوم حتى الآن ÷ (متوسط الحجم اليومي × النسبة المتوقعة لهذا الوقت)

    مثال: الساعة 16:00 UTC مضى ~38% من الجلسة، والمتوقع تراكمياً ~46% من
    حجم اليوم. لو السهم حقق 40% من متوسط حجمه اليومي، النتيجة = 0.40/0.46
    ≈ 0.87x بدل 0.40x بالطريقة القديمة — تقييم عادل لأول الجلسة.
    """
    if not avg_daily_volume or avg_daily_volume <= 0:
        return 0.0
    elapsed = session_elapsed_fraction(now_utc)
    expected = expected_volume_fraction(elapsed)
    expected = max(expected, 0.05)
    return (today_volume / avg_daily_volume) / expected


def check_volume(today_volume: float, avg_daily_volume: float,
                 now_utc=None):
    """(True/False, نسبة, رسالة) — الفحص الجاهز للاستخدام في المسح."""
    rv = smart_relative_volume(today_volume, avg_daily_volume, now_utc)
    if rv >= MIN_RELATIVE_VOLUME:
        return True, rv, f"حجم نسبي ذكي {rv:.1f}x ✓"
    return False, rv, f"حجم منخفض (ذكي {rv:.1f}x)"


# ==========================================================
# 4) بوابة الشراء الموحدة — كل قواعد المخاطر في مكان واحد
# ==========================================================
def can_buy(symbol: str,
            equity: float,
            order_value: float,
            open_positions: dict,
            trades_today: list,
            market_open: bool):
    """
    الفحص النهائي قبل إرسال أي أمر شراء.

    المعاملات:
      symbol         : رمز السهم
      equity         : القيمة الكلية للمحفظة (equity من Alpaca)
      order_value    : قيمة الصفقة المقترحة بالدولار
      open_positions : dict {symbol: market_value} للمراكز المفتوحة حالياً
      trades_today   : list بالرموز اللي انشترت اليوم (مع التكرار إن وجد)
      market_open    : حالة السوق

    ترجع: (True/False, سبب_بالعربي)
    """
    symbol = symbol.upper()

    # (أ) السوق
    if not market_open:
        return False, "السوق مغلق"

    # (ب) نفس السهم مرة واحدة باليوم
    if symbol in trades_today:
        return False, "اشترينا هذا السهم اليوم"

    # (ج) حد الصفقات اليومي
    if len(trades_today) >= MAX_TRADES_PER_DAY:
        return False, f"وصلنا حد الصفقات اليومي ({MAX_TRADES_PER_DAY})"

    # (د) حد القطاع
    sector = get_sector(symbol)
    sector_trades = sum(1 for s in trades_today if get_sector(s) == sector)
    if sector_trades >= MAX_TRADES_PER_SECTOR:
        return False, f"وصلنا حد قطاع {sector} اليوم ({MAX_TRADES_PER_SECTOR} صفقة)"

    # (هـ) سقف التعرض للسهم الواحد (شامل أي مركز مفتوح)
    current_symbol_value = float(open_positions.get(symbol, 0.0))
    max_symbol_value = equity * MAX_SYMBOL_EXPOSURE_PCT
    if current_symbol_value + order_value > max_symbol_value:
        return False, (f"تجاوز سقف السهم الواحد "
                       f"({MAX_SYMBOL_EXPOSURE_PCT*100:.0f}% = "
                       f"${max_symbol_value:,.0f})")

    # (و) سقف التعرض الكلي
    total_exposure = sum(float(v) for v in open_positions.values())
    max_total = equity * MAX_TOTAL_EXPOSURE_PCT
    if total_exposure + order_value > max_total:
        return False, (f"تجاوز سقف التعرض الكلي "
                       f"({MAX_TOTAL_EXPOSURE_PCT*100:.0f}% = "
                       f"${max_total:,.0f}، الحالي ${total_exposure:,.0f})")

    return True, "مسموح ✓"


# ==========================================================
# 5) مساعد: جلب المراكز المفتوحة من Alpaca بصيغة جاهزة لـ can_buy
# ==========================================================
def positions_to_dict(alpaca_positions) -> dict:
    """
    يحول قائمة مراكز Alpaca إلى {symbol: market_value}.
    يشتغل مع alpaca-trade-api (list_positions) ومع alpaca-py.
    """
    result = {}
    for p in alpaca_positions:
        try:
            sym = getattr(p, "symbol", None) or p.get("symbol")
            mv  = getattr(p, "market_value", None) or p.get("market_value", 0)
            result[str(sym).upper()] = abs(float(mv))
        except Exception:
            continue
    return result
