import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from sim.simulation import Simulation
from sim.persistence import load_state, save_state, STATE_PATH
from sim.summarizer import summarize_generation

REPORT_PATH = os.path.join("reports", "generation_log.md")


def append_report(stats, summary):
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    is_new = not os.path.exists(REPORT_PATH)
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        if is_new:
            f.write("# سجل تطور العالم\n\n")
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        f.write(f"## الجيل {stats['generation']} — {now}\n\n")
        f.write(
            f"- 👥 السكان: {stats['population']} (ذكور: {stats['males']}, إناث: {stats['females']})\n"
            f"- 👶 ولادات: {stats['births']} | 💀 وفيات: {stats['deaths']} | "
            f"🌱 أفراد إنقاذ عشوائيون: {stats['rescued_random_spawns']}\n"
            f"- 🍎 طعام مأكول: {stats['food_eaten']} | 🪵 خشب: {stats['wood_collected']} | "
            f"🏠 مآوٍ مبنية: {stats['shelters_built']}\n"
            f"- ⚡ متوسط الطاقة: {stats['avg_energy']:.1f} | 🎂 متوسط العمر: {stats['avg_age']:.1f} "
            f"| 👴 أكبر عمر: {stats['oldest_age']}\n\n"
        )
        f.write(f"{summary}\n\n---\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=10000)
    parser.add_argument("--ticks", type=int, default=None,
                         help="عدد النبضات لكل جيل (للاختبار السريع)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sim = load_state(STATE_PATH)
    if sim is None:
        print("لا توجد حالة محفوظة، بدء محاكاة جديدة...")
        sim = Simulation(seed=args.seed)
    else:
        print(f"تم استكمال المحاكاة من الجيل {sim.generation}")

    for _ in range(args.generations):
        kwargs = {}
        if args.ticks:
            kwargs["ticks"] = args.ticks
        stats = sim.run_generation(**kwargs)
        print(f"[جيل {stats['generation']}] سكان={stats['population']} "
              f"ولادات={stats['births']} وفيات={stats['deaths']}")
        summary = summarize_generation(stats)
        append_report(stats, summary)
        save_state(sim, STATE_PATH)  # نحفظ بعد كل جيل احتياطًا من الانقطاع

    print(f"انتهى. الجيل الحالي: {sim.generation}")


if __name__ == "__main__":
    main()
  
