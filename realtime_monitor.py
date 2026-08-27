# ════════════════════════════════════════════════════════════
# realtime_monitor.py — مراقبة لحظية للخروج (Stop-Loss / Trailing)
# ────────────────────────────────────────────────────────────
# الفكرة: بدل انتظار دورة الفحص كل 15 دقيقة عشان نطبّق وقف الخسارة،
# هذي الوحدة تفتح اتصال WebSocket لحظي (Alpaca IEX - مجاني) وتراقب
# بالثانية فقط الأسهم اللي عندنا فيها مراكز مفتوحة فعلياً (عادة أقل
# من 8 رموز، بعيد جداً عن حد الـ30 رمز حق الخطة المجانية).
#
# هذي الوحدة مستقلة تماماً عن jalal_radar_pro.py — ما تعدّل منطق
# الدخول أو الفحص الدوري الحالي إطلاقاً. لو صار فيها خلل أو انقطعت،
# البوت الأساسي يستمر يشتغل عادي بالفحص كل 15 دقيقة كخط دفاع احتياطي.
#
# ⚠️ ملاحظة مهمة: هذا الكود اجتاز فحص الصياغة (compile) بس ما قدرنا
# نختبره ضد اتصال Alpaca فعلي من هذي البيئة (لا يوجد وصول شبكي لـ
# alpaca.markets من بيئة التطوير). لازم اختبار حقيقي على Render.
# ════════════════════════════════════════════════════════════
import json
import threading
import time
import traceback
from datetime import datetime

try:
    import websocket  # pip install websocket-client
except ImportError:
    websocket = None

STREAM_URL_IEX = "wss://stream.data.alpaca.markets/v2/iex"  # خطة مجانية
MAX_FREE_TIER_SYMBOLS = 30  # سقف الخطة المجانية لقنوات trades/quotes


class RealtimeExitMonitor:
    """
    يراقب فقط الرموز المحتفظ بها حالياً (مراكز مفتوحة)، ويستدعي
    on_tick(symbol, price) عند كل صفقة (trade) لحظية — عشان نطبّق
    وقف الخسارة / trailing فوراً بدل الانتظار لدورة الفحص الدورية.
    """

    def __init__(self, api_key, api_secret, on_tick_callback, get_held_symbols_callback,
                 on_status_change=None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.on_tick = on_tick_callback
        self.get_held_symbols = get_held_symbols_callback
        self.on_status_change = on_status_change or (lambda status, detail: None)

        self.ws = None
        self.subscribed = set()
        self.running = False
        self.connected = False
        self._lock = threading.Lock()
        self._last_error = None

    # ── دورة الحياة ──
    def start(self):
        if websocket is None:
            self._log("❌ مكتبة websocket-client غير مثبّتة — أضفها لـ requirements.txt")
            return
        if self.running:
            return
        self.running = True
        threading.Thread(target=self._run_forever, daemon=True).start()
        # مؤقّت خلفي يحدّث الاشتراكات كل 60 ثانية (لو صفقة جديدة انفتحت أو انقفلت)
        threading.Thread(target=self._resubscribe_loop, daemon=True).start()
        self._log("✅ بدأ تشغيل المراقبة اللحظية")

    def stop(self):
        self.running = False
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
        self._log("🛑 أوقفت المراقبة اللحظية")

    # ── الاتصال ──
    def _run_forever(self):
        backoff = 5  # ثواني — يتصاعد لو الفشل تكرر، يرجع للأساس بعد اتصال ناجح
        while self.running:
            hit_conn_limit = False
            try:
                hit_conn_limit = self._connect_and_listen()
            except Exception as e:
                self._last_error = str(e)
                self._log(f"⚠️ خطأ اتصال: {e}")
                traceback.print_exc()
            self.connected = False
            if not self.running:
                break
            if hit_conn_limit:
                # ── تجاوز حد الاتصالات: نعطي وقت أطول عشان الاتصال القديم يتنظف من عند Alpaca ──
                wait = min(backoff * 3, 60)
                self._log(f"⏳ تجاوز حد الاتصالات — ننتظر {wait} ثانية قبل إعادة المحاولة")
                time.sleep(wait)
                backoff = min(backoff * 2, 60)
            else:
                time.sleep(backoff)
                backoff = 5  # نجح لفترة → نرجع للتأخير الأساسي

    def _connect_and_listen(self):
        """يرجع True لو سبب الانقطاع كان تجاوز حد الاتصالات (عشان نطبّق تأخير أطول)."""
        self._hit_conn_limit = False
        self.ws = websocket.WebSocketApp(
            STREAM_URL_IEX,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self.ws.run_forever(ping_interval=20, ping_timeout=10)
        return self._hit_conn_limit

    def _on_open(self, ws):
        self._log("🔌 اتصال WebSocket فُتح — جاري المصادقة...")
        ws.send(json.dumps({"action": "auth", "key": self.api_key, "secret": self.api_secret}))

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
        except Exception:
            return
        if not isinstance(data, list):
            data = [data]
        for msg in data:
            t = msg.get("T")
            if t == "success" and msg.get("msg") == "authenticated":
                self.connected = True
                self._log("✅ تمت المصادقة — جاري الاشتراك بالرموز المحتفظ بها")
                self._sync_subscriptions(ws)
            elif t == "error":
                self._last_error = str(msg)
                self._log(f"⚠️ خطأ من Alpaca: {msg}")
                if msg.get("code") == 406 or "connection limit" in str(msg.get("msg","")).lower():
                    self._hit_conn_limit = True
                    ws.close()
            elif t == "t":  # صفقة لحظية (trade tick)
                sym = msg.get("S")
                price = msg.get("p")
                if sym and price:
                    try:
                        self.on_tick(sym, float(price))
                    except Exception:
                        traceback.print_exc()

    def _on_error(self, ws, error):
        self._last_error = str(error)
        self._log(f"⚠️ WebSocket error: {error}")

    def _on_close(self, ws, code, msg):
        self.connected = False
        self._log(f"🔌 الاتصال انقطع (code={code})")

    # ── الاشتراكات الديناميكية ──
    def _resubscribe_loop(self):
        while self.running:
            time.sleep(60)
            if self.connected and self.ws:
                self._sync_subscriptions(self.ws)

    def _sync_subscriptions(self, ws):
        with self._lock:
            try:
                held = set(self.get_held_symbols())
            except Exception:
                held = set()
            # سقف أمان يطابق حد الخطة المجانية — حتى لو زاد عدد المراكز لأي سبب
            held = set(list(held)[:MAX_FREE_TIER_SYMBOLS])

            to_add = held - self.subscribed
            to_remove = self.subscribed - held
            try:
                if to_add:
                    ws.send(json.dumps({"action": "subscribe", "trades": list(to_add)}))
                    self._log(f"➕ اشتراك لحظي: {', '.join(to_add)}")
                if to_remove:
                    ws.send(json.dumps({"action": "unsubscribe", "trades": list(to_remove)}))
                    self._log(f"➖ إلغاء اشتراك: {', '.join(to_remove)}")
            except Exception as e:
                self._log(f"⚠️ فشل تحديث الاشتراك: {e}")
                return
            self.subscribed = held

    def status(self):
        return {
            "running": self.running,
            "connected": self.connected,
            "subscribed": sorted(self.subscribed),
            "last_error": self._last_error,
        }

    def _log(self, text):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[realtime {ts}] {text}")
        self.on_status_change(self.status(), text)
