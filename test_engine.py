"""Run: python test_engine.py  - checks the expert system on the demo patients."""
from engine import ExpertSystem, RULES, count_rules
from samples import SAMPLES, make_patient

es = ExpertSystem()
print(f"Total rules: {len(RULES)}  {count_rules()}\n")

expected = {
    "Patient A - Chest pain on walking (CAD)": "Coronary",
    "Patient B - Breathless, swollen legs (Heart Failure)": "Heart Failure",
    "Patient C - Palpitations and fainting (Arrhythmia)": "Arrhythmia",
    "Patient D - Heart attack warning signs (Emergency)": "Coronary",
    "Patient E - Healthy young adult": "No significant",
}

ok = True
for name in SAMPLES:
    r = es.infer(make_patient(name))
    passed = expected[name] in r.primary
    ok &= passed
    print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    print(f"   CF: {r.cf}  ->  {r.primary}")
    print(f"   Urgency: {r.urgency_text}   Rules fired: {len(r.fired)}\n")

assert len(RULES) >= 50
assert ok, "Some cases failed"
print("All tests passed.")
