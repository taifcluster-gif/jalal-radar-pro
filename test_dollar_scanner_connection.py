# ════════════════════════════════════════════════════════════
# test_dollar_scanner_connection.py — اختبار اتصال سريع فقط
# ────────────────────────────────────────────────────────────
# يتأكد بس إن المفاتيح الجديدة صحيحة ومتصلة بحساب Paper منفصل —
# ما يشغّل الفحص الكامل (اللي بياخذ وقت طويل لأنه يفحص آلاف الأسهم).
#
# طريقة التشغيل (من Shell بـRender، أو جهازك لو فيه بايثون):
#   python3 test_dollar_scanner_connection.py API_KEY SECRET_KEY
# ════════════════════════════════════════════════════════════
import sys
import json
import urllib.request

if len(sys.argv) < 3:
    print("الاستخدام: python3 test_dollar_scanner_connection.py API_KEY SECRET_KEY")
    sys.exit(1)

KEY, SECRET = sys.argv[1], sys.argv[2]
BASE = "https://paper-api.alpaca.markets"

def req(path):
    r = urllib.request.Request(
        BASE + path,
        headers={"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SECRET},
    )
    with urllib.request.urlopen(r, timeout=10) as resp:
        return json.loads(resp.read().decode())

print("🚀 جاري اختبار الاتصال بحساب Alpaca الجديد (دولار سكانر)...")
try:
    acc = req("/v2/account")
    print("✅ الاتصال نجح!")
    print(f"   رقم الحساب: {acc.get('account_number')}")
    print(f"   الرصيد (Cash): ${acc.get('cash')}")
    print(f"   القيمة الكلية: ${acc.get('equity')}")
    print(f"   حالة الحساب: {acc.get('status')}")
except Exception as e:
    print(f"❌ فشل الاتصال: {e}")
    print("   تأكد إن المفتاحين صحيحين ومن الحساب الجديد (مو الرئيسي)")
    sys.exit(1)
