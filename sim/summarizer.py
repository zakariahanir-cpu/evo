import os
import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


def summarize_generation(stats: dict) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    prompt = (
        "أنت راوٍ يلخص ما حدث في جيل واحد من محاكاة عالم افتراضي فيه كائنات ذكاء اصطناعي "
        "تتطور جيناتها عبر الأجيال. هذه إحصائيات الجيل رقم "
        f"{stats['generation']}:\n"
        f"- عدد السكان في نهاية الجيل: {stats['population']} "
        f"(ذكور: {stats['males']}, إناث: {stats['females']})\n"
        f"- الولادات: {stats['births']}, الوفيات: {stats['deaths']}\n"
        f"- أفراد جدد عشوائيون بسبب قلة السكان: {stats['rescued_random_spawns']}\n"
        f"- كمية الطعام المأكولة: {stats['food_eaten']}\n"
        f"- الخشب المُجمّع: {stats['wood_collected']}, المآوي المبنية: {stats['shelters_built']}\n"
        f"- متوسط الطاقة: {stats['avg_energy']:.1f}, متوسط العمر: {stats['avg_age']:.1f}, "
        f"أكبر عمر: {stats['oldest_age']}\n\n"
        "اكتب فقرة قصيرة (٣-٥ أسطر) بالعربية الفصحى المبسطة تلخص كيف كان أداء هذا الجيل، "
        "هل كان جيل ازدهار أم شح موارد أم انقراض شبه كامل، بأسلوب سردي شيّق لكنه دقيق بالأرقام. "
        "لا تخترع أحداثًا غير موجودة في الإحصائيات."
    )

    if not api_key:
        return ("⚠️ لم يتم العثور على GROQ_API_KEY، تم تخطي التلخيص الذكي لهذا الجيل. "
                "أضف السر GROQ_API_KEY في إعدادات المستودع (Settings > Secrets and variables > Actions).")

    try:
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 300,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ تعذّر الاتصال بـ Groq API لتلخيص هذا الجيل ({e})."
