"""
CardioExpert - Heart Disease Expert System  (single-file version)
==================================================================
Run locally :  streamlit run app.py

This one file contains the whole project in 3 parts:

  PART 1  KNOWLEDGE BASE + INFERENCE ENGINE
          - 78 IF-THEN rules (Derived 20, CAD 15, HF 14, ARR 12, Decision 17)
          - forward chaining with MYCIN certainty factors
          - detects 3 diseases: Coronary Artery Disease, Heart Failure, Arrhythmia
          - decision making (5 urgency levels) + decision support (tests & advice)

  PART 2  DEMO PATIENTS (default values + 5 sample patients)

  PART 3  STREAMLIT USER INTERFACE (Diagnosis, Knowledge Base, How it works)
"""
import datetime as dt
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import streamlit as st


######################################################################
#                                                                    #
#   PART 1 : KNOWLEDGE BASE + INFERENCE ENGINE                       #
#                                                                    #
######################################################################

# ----------------------------------------------------------------------
# Diseases and urgency levels
# ----------------------------------------------------------------------
DISEASES = {
    "CAD": "Coronary Artery Disease",
    "HF": "Heart Failure",
    "ARR": "Cardiac Arrhythmia",
}

URGENCY = {
    0: "LOW RISK - Routine care",
    1: "PREVENTIVE CHECK-UP advised",
    2: "SEE A DOCTOR SOON (within days)",
    3: "URGENT - See a cardiologist within 24 hours",
    4: "EMERGENCY - Call emergency services (1122) now",
}


# ----------------------------------------------------------------------
# Rule structure
# ----------------------------------------------------------------------
@dataclass
class Rule:
    rid: str                                  # rule id, e.g. "C01"
    group: str                                # Derived / CAD / HF / ARR / Decision
    text: str                                 # human readable IF ... THEN ...
    cond: Callable[[dict], bool]              # IF part (test on working memory)
    then: Optional[Callable[[dict], dict]] = None  # THEN part: new facts
    disease: Optional[str] = None             # THEN part: evidence for a disease
    cf: float = 0.0                           # certainty factor of that evidence
    advice: Optional[str] = None              # THEN part: recommendation
    phase: int = 1


def g(wm: dict, key: str, default=False):
    """Safe get from working memory."""
    return wm.get(key, default)


def has_htn(wm):
    return g(wm, "hypertension_stage1") or g(wm, "hypertension_stage2")


def max_cf(wm):
    return max(g(wm, "cf_CAD", 0.0), g(wm, "cf_HF", 0.0), g(wm, "cf_ARR", 0.0))


def raise_urgency(level):
    """THEN-part helper: urgency only goes up, never down."""
    return lambda wm: {"urgency": max(level, g(wm, "urgency", 0))}


# ----------------------------------------------------------------------
# KNOWLEDGE BASE
# ----------------------------------------------------------------------
RULES: List[Rule] = [
    # ================= PHASE 1: DERIVED FACTS =================
    Rule("D01", "Derived", "IF height and weight are given THEN calculate BMI = weight / height^2",
         lambda wm: g(wm, "height_cm", 0) > 0 and g(wm, "weight_kg", 0) > 0,
         then=lambda wm: {"bmi": round(wm["weight_kg"] / (wm["height_cm"] / 100) ** 2, 1)}),
    Rule("D02", "Derived", "IF BMI >= 30 THEN patient is obese",
         lambda wm: g(wm, "bmi", 0) >= 30, then=lambda wm: {"obese": True}),
    Rule("D03", "Derived", "IF 25 <= BMI < 30 THEN patient is overweight",
         lambda wm: 25 <= g(wm, "bmi", 0) < 30, then=lambda wm: {"overweight": True}),
    Rule("D04", "Derived", "IF (male AND age >= 45) OR (female AND age >= 55) THEN age is a risk factor",
         lambda wm: (g(wm, "sex") == "Male" and g(wm, "age", 0) >= 45)
         or (g(wm, "sex") == "Female" and g(wm, "age", 0) >= 55),
         then=lambda wm: {"age_risk": True}),
    Rule("D05", "Derived", "IF systolic >= 140 OR diastolic >= 90 THEN stage-2 hypertension",
         lambda wm: g(wm, "sys_bp", 0) >= 140 or g(wm, "dia_bp", 0) >= 90,
         then=lambda wm: {"hypertension_stage2": True}),
    Rule("D06", "Derived", "IF systolic 130-139 OR diastolic 80-89 (and not stage 2) THEN stage-1 hypertension",
         lambda wm: not g(wm, "hypertension_stage2")
         and (130 <= g(wm, "sys_bp", 0) < 140 or 80 <= g(wm, "dia_bp", 0) < 90),
         then=lambda wm: {"hypertension_stage1": True}),
    Rule("D07", "Derived", "IF systolic >= 180 OR diastolic >= 120 THEN hypertensive crisis",
         lambda wm: g(wm, "sys_bp", 0) >= 180 or g(wm, "dia_bp", 0) >= 120,
         then=lambda wm: {"hypertensive_crisis": True}),
    Rule("D08", "Derived", "IF total cholesterol >= 240 mg/dL THEN high cholesterol",
         lambda wm: g(wm, "cholesterol", 0) >= 240, then=lambda wm: {"high_cholesterol": True}),
    Rule("D09", "Derived", "IF LDL >= 160 mg/dL THEN high LDL",
         lambda wm: g(wm, "ldl", 0) >= 160, then=lambda wm: {"high_ldl": True}),
    Rule("D10", "Derived", "IF HDL < 40 (male) OR HDL < 50 (female) THEN low HDL",
         lambda wm: 0 < g(wm, "hdl", 0) < (40 if g(wm, "sex") == "Male" else 50),
         then=lambda wm: {"low_hdl": True}),
    Rule("D11", "Derived", "IF known diabetic OR fasting sugar >= 126 mg/dL THEN diabetic",
         lambda wm: g(wm, "diabetes") or g(wm, "fasting_sugar", 0) >= 126,
         then=lambda wm: {"diabetic": True}),
    Rule("D12", "Derived", "IF fasting sugar 100-125 mg/dL (not diabetic) THEN prediabetic",
         lambda wm: not g(wm, "diabetic") and 100 <= g(wm, "fasting_sugar", 0) < 126,
         then=lambda wm: {"prediabetic": True}),
    Rule("D13", "Derived", "IF resting heart rate > 100 bpm THEN tachycardia",
         lambda wm: g(wm, "resting_hr", 0) > 100, then=lambda wm: {"tachycardia": True}),
    Rule("D14", "Derived", "IF resting heart rate < 60 bpm THEN bradycardia",
         lambda wm: 0 < g(wm, "resting_hr", 0) < 60, then=lambda wm: {"bradycardia": True}),
    Rule("D15", "Derived", "IF at least 3 of (obese, hypertension, low HDL, high sugar, high cholesterol) THEN metabolic syndrome",
         lambda wm: sum([bool(g(wm, "obese")), bool(has_htn(wm)), bool(g(wm, "low_hdl")),
                         bool(g(wm, "diabetic") or g(wm, "prediabetic")),
                         bool(g(wm, "high_cholesterol"))]) >= 3,
         then=lambda wm: {"metabolic_syndrome": True}),
    Rule("D16", "Derived", "IF 2 or more of (smoker, sedentary, heavy alcohol) THEN poor lifestyle",
         lambda wm: sum([bool(g(wm, "smoker")), bool(g(wm, "sedentary")), bool(g(wm, "alcohol_heavy"))]) >= 2,
         then=lambda wm: {"poor_lifestyle": True}),
    Rule("D17", "Derived", "IF ejection fraction is known AND EF < 40% THEN reduced pumping (HFrEF range)",
         lambda wm: 0 < g(wm, "ef", 0) < 40, then=lambda wm: {"reduced_ef": True}),
    Rule("D18", "Derived", "IF chest pain is typical angina OR (pain on exertion AND relieved by rest) THEN stable angina pattern",
         lambda wm: g(wm, "chest_pain") == "Typical angina"
         or (g(wm, "pain_exertion") and g(wm, "pain_relieved_rest")),
         then=lambda wm: {"angina_pattern": True}),
    Rule("D19", "Derived", "IF chest pain at rest AND (sweating OR pain spreading to arm/jaw) THEN acute coronary syndrome pattern",
         lambda wm: g(wm, "pain_at_rest") and (g(wm, "sweating") or g(wm, "pain_radiates")),
         then=lambda wm: {"acs_pattern": True}),
    Rule("D20", "Derived", "IF leg swelling AND (rapid weight gain OR breathless lying flat) THEN fluid overload",
         lambda wm: g(wm, "leg_swelling")
         and (g(wm, "weight_gain_fast") or g(wm, "breathless") == "Lying flat"),
         then=lambda wm: {"fluid_overload": True}),

    # ================= PHASE 1: CORONARY ARTERY DISEASE =================
    Rule("C01", "CAD", "IF stable angina pattern THEN CAD (CF 0.60)",
         lambda wm: g(wm, "angina_pattern"), disease="CAD", cf=0.60),
    Rule("C02", "CAD", "IF acute coronary syndrome pattern THEN CAD (CF 0.80)",
         lambda wm: g(wm, "acs_pattern"), disease="CAD", cf=0.80),
    Rule("C03", "CAD", "IF chest pain spreads to left arm / jaw THEN CAD (CF 0.40)",
         lambda wm: g(wm, "pain_radiates"), disease="CAD", cf=0.40),
    Rule("C04", "CAD", "IF ECG shows ST-segment changes THEN CAD (CF 0.70)",
         lambda wm: g(wm, "ecg_st_changes"), disease="CAD", cf=0.70),
    Rule("C05", "CAD", "IF previous heart attack THEN CAD (CF 0.60)",
         lambda wm: g(wm, "previous_mi"), disease="CAD", cf=0.60),
    Rule("C06", "CAD", "IF age is a risk factor THEN CAD (CF 0.20)",
         lambda wm: g(wm, "age_risk"), disease="CAD", cf=0.20),
    Rule("C07", "CAD", "IF smoker THEN CAD (CF 0.30)",
         lambda wm: g(wm, "smoker"), disease="CAD", cf=0.30),
    Rule("C08", "CAD", "IF diabetic THEN CAD (CF 0.30)",
         lambda wm: g(wm, "diabetic"), disease="CAD", cf=0.30),
    Rule("C09", "CAD", "IF high LDL OR high total cholesterol THEN CAD (CF 0.30)",
         lambda wm: g(wm, "high_ldl") or g(wm, "high_cholesterol"), disease="CAD", cf=0.30),
    Rule("C10", "CAD", "IF low HDL THEN CAD (CF 0.20)",
         lambda wm: g(wm, "low_hdl"), disease="CAD", cf=0.20),
    Rule("C11", "CAD", "IF hypertension (stage 1 or 2) THEN CAD (CF 0.20)",
         lambda wm: has_htn(wm), disease="CAD", cf=0.20),
    Rule("C12", "CAD", "IF family history of heart disease THEN CAD (CF 0.25)",
         lambda wm: g(wm, "family_history"), disease="CAD", cf=0.25),
    Rule("C13", "CAD", "IF metabolic syndrome THEN CAD (CF 0.25)",
         lambda wm: g(wm, "metabolic_syndrome"), disease="CAD", cf=0.25),
    Rule("C14", "CAD", "IF chest pain present AND breathless on exertion THEN CAD (CF 0.30)",
         lambda wm: g(wm, "chest_pain", "None") != "None" and g(wm, "breathless") == "On exertion",
         disease="CAD", cf=0.30),
    Rule("C15", "CAD", "IF chest pain is non-anginal AND no ECG change AND no previous MI THEN against CAD (CF -0.40)",
         lambda wm: g(wm, "chest_pain") == "Non-anginal" and not g(wm, "ecg_st_changes")
         and not g(wm, "previous_mi"),
         disease="CAD", cf=-0.40),

    # ================= PHASE 1: HEART FAILURE =================
    Rule("H01", "HF", "IF breathless when lying flat (orthopnea) THEN HF (CF 0.60)",
         lambda wm: g(wm, "breathless") == "Lying flat", disease="HF", cf=0.60),
    Rule("H02", "HF", "IF breathless at rest THEN HF (CF 0.50)",
         lambda wm: g(wm, "breathless") == "At rest", disease="HF", cf=0.50),
    Rule("H03", "HF", "IF swelling of legs / ankles THEN HF (CF 0.40)",
         lambda wm: g(wm, "leg_swelling"), disease="HF", cf=0.40),
    Rule("H04", "HF", "IF fluid overload THEN HF (CF 0.60)",
         lambda wm: g(wm, "fluid_overload"), disease="HF", cf=0.60),
    Rule("H05", "HF", "IF cough at night THEN HF (CF 0.30)",
         lambda wm: g(wm, "night_cough"), disease="HF", cf=0.30),
    Rule("H06", "HF", "IF constant fatigue AND breathless on exertion THEN HF (CF 0.30)",
         lambda wm: g(wm, "fatigue") and g(wm, "breathless") == "On exertion", disease="HF", cf=0.30),
    Rule("H07", "HF", "IF ejection fraction < 40% THEN HF (CF 0.80)",
         lambda wm: g(wm, "reduced_ef"), disease="HF", cf=0.80),
    Rule("H08", "HF", "IF ejection fraction 40-49% THEN HF (CF 0.40)",
         lambda wm: 40 <= g(wm, "ef", 0) < 50, disease="HF", cf=0.40),
    Rule("H09", "HF", "IF stage-2 hypertension THEN HF (CF 0.25)",
         lambda wm: g(wm, "hypertension_stage2"), disease="HF", cf=0.25),
    Rule("H10", "HF", "IF previous heart attack THEN HF (CF 0.35)",
         lambda wm: g(wm, "previous_mi"), disease="HF", cf=0.35),
    Rule("H11", "HF", "IF tachycardia AND any breathlessness THEN HF (CF 0.25)",
         lambda wm: g(wm, "tachycardia") and g(wm, "breathless", "None") != "None", disease="HF", cf=0.25),
    Rule("H12", "HF", "IF obese THEN HF (CF 0.15)",
         lambda wm: g(wm, "obese"), disease="HF", cf=0.15),
    Rule("H13", "HF", "IF diabetic THEN HF (CF 0.15)",
         lambda wm: g(wm, "diabetic"), disease="HF", cf=0.15),
    Rule("H14", "HF", "IF ejection fraction >= 55% (normal) THEN against HF (CF -0.50)",
         lambda wm: g(wm, "ef", 0) >= 55, disease="HF", cf=-0.50),

    # ================= PHASE 1: ARRHYTHMIA =================
    Rule("A01", "ARR", "IF palpitations THEN arrhythmia (CF 0.40)",
         lambda wm: g(wm, "palpitations"), disease="ARR", cf=0.40),
    Rule("A02", "ARR", "IF irregular pulse THEN arrhythmia (CF 0.60)",
         lambda wm: g(wm, "irregular_pulse"), disease="ARR", cf=0.60),
    Rule("A03", "ARR", "IF ECG shows irregular rhythm THEN arrhythmia (CF 0.80)",
         lambda wm: g(wm, "ecg_irregular"), disease="ARR", cf=0.80),
    Rule("A04", "ARR", "IF fainting episodes THEN arrhythmia (CF 0.40)",
         lambda wm: g(wm, "fainting"), disease="ARR", cf=0.40),
    Rule("A05", "ARR", "IF dizziness AND palpitations THEN arrhythmia (CF 0.35)",
         lambda wm: g(wm, "dizziness") and g(wm, "palpitations"), disease="ARR", cf=0.35),
    Rule("A06", "ARR", "IF tachycardia THEN arrhythmia (CF 0.35)",
         lambda wm: g(wm, "tachycardia"), disease="ARR", cf=0.35),
    Rule("A07", "ARR", "IF bradycardia AND (dizziness OR fatigue OR fainting) THEN arrhythmia (CF 0.45)",
         lambda wm: g(wm, "bradycardia") and (g(wm, "dizziness") or g(wm, "fatigue") or g(wm, "fainting")),
         disease="ARR", cf=0.45),
    Rule("A08", "ARR", "IF heavy alcohol use THEN arrhythmia (CF 0.20)",
         lambda wm: g(wm, "alcohol_heavy"), disease="ARR", cf=0.20),
    Rule("A09", "ARR", "IF high stress AND palpitations THEN arrhythmia (CF 0.20)",
         lambda wm: g(wm, "high_stress") and g(wm, "palpitations"), disease="ARR", cf=0.20),
    Rule("A10", "ARR", "IF age >= 65 AND hypertension THEN arrhythmia / AF risk (CF 0.20)",
         lambda wm: g(wm, "age", 0) >= 65 and has_htn(wm), disease="ARR", cf=0.20),
    Rule("A11", "ARR", "IF previous heart attack THEN arrhythmia (CF 0.15)",
         lambda wm: g(wm, "previous_mi"), disease="ARR", cf=0.15),
    Rule("A12", "ARR", "IF palpitations with high stress BUT pulse regular, normal ECG and normal heart rate THEN likely anxiety, against arrhythmia (CF -0.30)",
         lambda wm: g(wm, "palpitations") and g(wm, "high_stress") and not g(wm, "irregular_pulse")
         and not g(wm, "ecg_irregular") and not g(wm, "tachycardia") and not g(wm, "bradycardia"),
         disease="ARR", cf=-0.30),

    # ================= PHASE 2: DECISION SUPPORT =================
    Rule("X01", "Decision", "IF acute coronary syndrome pattern THEN EMERGENCY",
         lambda wm: g(wm, "acs_pattern"), then=raise_urgency(4),
         advice="Possible heart attack: call emergency services (1122) and chew 300 mg aspirin if not allergic.",
         phase=2),
    Rule("X02", "Decision", "IF hypertensive crisis THEN EMERGENCY",
         lambda wm: g(wm, "hypertensive_crisis"), then=raise_urgency(4),
         advice="Blood pressure is dangerously high: go to the emergency department now.", phase=2),
    Rule("X03", "Decision", "IF fainting AND (palpitations OR chest pain) THEN URGENT",
         lambda wm: g(wm, "fainting") and (g(wm, "palpitations") or g(wm, "chest_pain", "None") != "None"),
         then=raise_urgency(3), advice="Fainting with heart symptoms needs same-day cardiac evaluation.", phase=2),
    Rule("X04", "Decision", "IF any disease CF >= 0.70 THEN URGENT",
         lambda wm: max_cf(wm) >= 0.70, then=raise_urgency(3), phase=2),
    Rule("X05", "Decision", "IF highest disease CF is 0.40-0.69 THEN SEE DOCTOR SOON",
         lambda wm: 0.40 <= max_cf(wm) < 0.70, then=raise_urgency(2), phase=2),
    Rule("X06", "Decision", "IF highest CF < 0.40 AND (metabolic syndrome OR poor lifestyle OR age risk with hypertension) THEN PREVENTIVE CHECK-UP",
         lambda wm: max_cf(wm) < 0.40 and (g(wm, "metabolic_syndrome") or g(wm, "poor_lifestyle")
                                          or (g(wm, "age_risk") and has_htn(wm))),
         then=raise_urgency(1), advice="Get a yearly check-up: BP, lipid profile, sugar and ECG.", phase=2),
    Rule("X07", "Decision", "IF CAD CF >= 0.40 THEN recommend CAD tests",
         lambda wm: g(wm, "cf_CAD", 0) >= 0.40,
         advice="Tests for CAD: 12-lead ECG, troponin, exercise stress test (ETT), lipid profile; consider angiography.",
         phase=2),
    Rule("X08", "Decision", "IF HF CF >= 0.40 THEN recommend HF tests and care",
         lambda wm: g(wm, "cf_HF", 0) >= 0.40,
         advice="Tests for heart failure: echocardiogram, NT-proBNP, chest X-ray. Limit salt and weigh yourself daily.",
         phase=2),
    Rule("X09", "Decision", "IF arrhythmia CF >= 0.40 THEN recommend rhythm tests",
         lambda wm: g(wm, "cf_ARR", 0) >= 0.40,
         advice="Tests for arrhythmia: ECG, 24-48 h Holter monitor, thyroid test (TSH), electrolytes. Reduce caffeine.",
         phase=2),
    Rule("X10", "Decision", "IF smoker THEN advise quitting",
         lambda wm: g(wm, "smoker"), advice="Stop smoking - it is the biggest avoidable heart risk.", phase=2),
    Rule("X11", "Decision", "IF obese OR overweight THEN advise weight loss",
         lambda wm: g(wm, "obese") or g(wm, "overweight"),
         advice="Aim to reduce weight by 5-10% through diet and exercise.", phase=2),
    Rule("X12", "Decision", "IF hypertension THEN advise BP control",
         lambda wm: has_htn(wm),
         advice="Monitor BP at home, reduce salt (< 5 g/day) and follow up with a doctor about BP medicine.", phase=2),
    Rule("X13", "Decision", "IF diabetic OR prediabetic THEN advise sugar control",
         lambda wm: g(wm, "diabetic") or g(wm, "prediabetic"),
         advice="Control blood sugar: check HbA1c, reduce sugar and refined carbs.", phase=2),
    Rule("X14", "Decision", "IF high cholesterol OR high LDL OR low HDL THEN advise lipid control",
         lambda wm: g(wm, "high_cholesterol") or g(wm, "high_ldl") or g(wm, "low_hdl"),
         advice="Improve cholesterol: less fried/fatty food, more fibre; a doctor may prescribe a statin.", phase=2),
    Rule("X15", "Decision", "IF sedentary THEN advise exercise",
         lambda wm: g(wm, "sedentary"),
         advice="Do at least 150 minutes per week of moderate exercise (e.g. brisk walking) once cleared by a doctor.",
         phase=2),
    Rule("X16", "Decision", "IF heavy alcohol OR high stress THEN advise lifestyle change",
         lambda wm: g(wm, "alcohol_heavy") or g(wm, "high_stress"),
         advice="Avoid alcohol and manage stress (sleep 7-8 h, relaxation, counselling if needed).", phase=2),
    Rule("X17", "Decision", "IF no disease CF >= 0.20 AND no urgency raised THEN reassure",
         lambda wm: max_cf(wm) < 0.20 and g(wm, "urgency", 0) == 0,
         advice="No significant signs of heart disease. Keep a healthy lifestyle and do routine check-ups.",
         phase=2),
]


# ----------------------------------------------------------------------
# Certainty factor combination (MYCIN)
# ----------------------------------------------------------------------
def combine_cf(cf1: float, cf2: float) -> float:
    if cf1 >= 0 and cf2 >= 0:
        return cf1 + cf2 * (1 - cf1)
    if cf1 < 0 and cf2 < 0:
        return cf1 + cf2 * (1 + cf1)
    return (cf1 + cf2) / (1 - min(abs(cf1), abs(cf2)))


def interpret_cf(cf: float) -> str:
    if cf >= 0.70:
        return "High likelihood"
    if cf >= 0.40:
        return "Moderate likelihood"
    if cf >= 0.20:
        return "Low likelihood"
    return "Unlikely"


# ----------------------------------------------------------------------
# Inference engine (forward chaining)
# ----------------------------------------------------------------------
@dataclass
class Result:
    cf: Dict[str, float]
    urgency_level: int
    urgency_text: str
    primary: str
    advice: List[str]
    fired: List[Rule]
    facts: dict
    trace: Dict[str, List[Rule]] = field(default_factory=dict)


class ExpertSystem:
    def __init__(self, rules: List[Rule] = RULES):
        self.rules = rules

    def infer(self, inputs: dict) -> Result:
        wm = dict(inputs)                         # working memory
        for d in DISEASES:
            wm[f"cf_{d}"] = 0.0
        wm["urgency"] = 0
        fired: List[Rule] = []
        fired_ids = set()
        advice: List[str] = []
        trace = {d: [] for d in DISEASES}

        # Phase 1 (facts + disease evidence), then Phase 2 (decisions).
        for phase in (1, 2):
            changed = True
            while changed:                        # repeat until no new rule fires
                changed = False
                for rule in self.rules:
                    if rule.phase != phase or rule.rid in fired_ids:
                        continue
                    if not rule.cond(wm):         # match IF-part
                        continue
                    # ---- fire the rule (THEN-part) ----
                    fired_ids.add(rule.rid)
                    fired.append(rule)
                    if rule.then:
                        wm.update(rule.then(wm))
                    if rule.disease:
                        key = f"cf_{rule.disease}"
                        wm[key] = combine_cf(wm[key], rule.cf)
                        trace[rule.disease].append(rule)
                    if rule.advice:
                        advice.append(rule.advice)
                    changed = True

        cf = {d: round(max(0.0, wm[f"cf_{d}"]), 2) for d in DISEASES}
        best = max(cf, key=cf.get)
        primary = (f"{DISEASES[best]} ({interpret_cf(cf[best])})" if cf[best] >= 0.40
                   else "No significant heart disease detected")
        level = wm["urgency"]
        derived = {k: v for k, v in wm.items() if k not in inputs and not k.startswith("cf_")}
        return Result(cf, level, URGENCY[level], primary, advice, fired, derived, trace)


def count_rules():
    out = {}
    for r in RULES:
        out[r.group] = out.get(r.group, 0) + 1
    return out


######################################################################
#                                                                    #
#   PART 2 : DEMO PATIENTS                                           #
#                                                                    #
######################################################################
DEFAULTS = {
    "age": 40, "sex": "Male", "height_cm": 170, "weight_kg": 70,
    "smoker": False, "diabetes": False, "family_history": False, "previous_mi": False,
    "sedentary": False, "alcohol_heavy": False, "high_stress": False,
    "sys_bp": 118, "dia_bp": 76, "resting_hr": 75,
    "cholesterol": 180, "ldl": 100, "hdl": 50, "fasting_sugar": 90,
    "chest_pain": "None", "pain_exertion": False, "pain_relieved_rest": False,
    "pain_at_rest": False, "pain_radiates": False, "sweating": False, "nausea": False,
    "breathless": "None", "leg_swelling": False, "fatigue": False,
    "night_cough": False, "weight_gain_fast": False,
    "palpitations": False, "irregular_pulse": False, "dizziness": False, "fainting": False,
    "ecg_st_changes": False, "ecg_irregular": False, "ef": 0,
}

SAMPLES = {
    "Patient A - Chest pain on walking (CAD)": {
        "age": 58, "sex": "Male", "height_cm": 172, "weight_kg": 88, "smoker": True,
        "family_history": True, "sedentary": True, "sys_bp": 145, "dia_bp": 92,
        "cholesterol": 255, "ldl": 170, "hdl": 35, "fasting_sugar": 110,
        "chest_pain": "Typical angina", "pain_exertion": True, "pain_relieved_rest": True,
        "pain_radiates": True, "breathless": "On exertion",
    },
    "Patient B - Breathless, swollen legs (Heart Failure)": {
        "age": 67, "sex": "Female", "height_cm": 158, "weight_kg": 82, "diabetes": True,
        "previous_mi": True, "sys_bp": 150, "dia_bp": 88, "resting_hr": 104,
        "fasting_sugar": 140, "breathless": "Lying flat", "leg_swelling": True,
        "fatigue": True, "night_cough": True, "weight_gain_fast": True, "ef": 32,
    },
    "Patient C - Palpitations and fainting (Arrhythmia)": {
        "age": 45, "sex": "Female", "height_cm": 162, "weight_kg": 60,
        "alcohol_heavy": True, "high_stress": True, "resting_hr": 118,
        "palpitations": True, "irregular_pulse": True, "dizziness": True,
        "fainting": True, "ecg_irregular": True,
    },
    "Patient D - Heart attack warning signs (Emergency)": {
        "age": 62, "sex": "Male", "height_cm": 175, "weight_kg": 90, "smoker": True,
        "diabetes": True, "sys_bp": 160, "dia_bp": 95, "fasting_sugar": 150,
        "chest_pain": "Typical angina", "pain_at_rest": True, "pain_radiates": True,
        "sweating": True, "nausea": True, "ecg_st_changes": True,
    },
    "Patient E - Healthy young adult": {
        "age": 24, "sex": "Female", "height_cm": 160, "weight_kg": 55, "hdl": 60,
    },
}


def make_patient(name: str) -> dict:
    p = dict(DEFAULTS)
    p.update(SAMPLES.get(name, {}))
    return p


######################################################################
#                                                                    #
#   PART 3 : STREAMLIT USER INTERFACE                                #
#                                                                    #
######################################################################
st.set_page_config(page_title="CardioExpert - Heart Disease Expert System",
                   page_icon="❤️", layout="wide")

st.markdown("""
<style>
.big-title {font-size: 2.1rem; font-weight: 700; margin-bottom: 0;}
.sub {color: #888; margin-top: 0;}
.card {border: 1px solid rgba(128,128,128,.3); border-radius: 12px; padding: 14px 16px;}
</style>
""", unsafe_allow_html=True)

# ---------------- session state ----------------
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("result", None)


def load_sample():
    """Callback: fill all input widgets with a demo patient."""
    name = st.session_state["sample_choice"]
    patient = make_patient(name) if name in SAMPLES else dict(DEFAULTS)
    for k, v in patient.items():
        st.session_state[k] = v
    st.session_state["result"] = None


# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown("## ❤️ CardioExpert")
    page = st.radio("Go to", ["🩺 Diagnosis", "📚 Knowledge Base", "⚙️ How it works"])
    st.divider()
    st.selectbox("Load a demo patient", ["-- Blank form --"] + list(SAMPLES),
                 key="sample_choice", on_change=load_sample)
    st.caption(f"{len(RULES)} rules · 3 diseases · forward chaining · certainty factors")
    st.divider()
    st.caption("⚠️ Educational project only. Not a substitute for a real doctor.")


# =====================================================================
# PAGE 1: DIAGNOSIS
# =====================================================================
def diagnosis_page():
    st.markdown('<p class="big-title">❤️ CardioExpert</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub">Rule-based expert system for Coronary Artery Disease, '
                'Heart Failure and Cardiac Arrhythmia</p>', unsafe_allow_html=True)

    with st.form("patient_form"):
        t1, t2, t3, t4 = st.tabs(["1️⃣ Personal & lifestyle", "2️⃣ Vitals & lab",
                                  "3️⃣ Symptoms", "4️⃣ Test reports (optional)"])
        with t1:
            c1, c2, c3 = st.columns(3)
            c1.number_input("Age (years)", 1, 120, key="age")
            c1.selectbox("Sex", ["Male", "Female"], key="sex")
            c2.number_input("Height (cm)", 50, 230, key="height_cm")
            c2.number_input("Weight (kg)", 10, 250, key="weight_kg")
            c3.checkbox("Smoker", key="smoker")
            c3.checkbox("Known diabetic", key="diabetes")
            c3.checkbox("Family history of heart disease", key="family_history")
            c3.checkbox("Previous heart attack", key="previous_mi")
            c1.checkbox("Sedentary (little exercise)", key="sedentary")
            c2.checkbox("Heavy alcohol use", key="alcohol_heavy")
            c2.checkbox("High stress", key="high_stress")
        with t2:
            c1, c2, c3 = st.columns(3)
            c1.number_input("Systolic BP (mmHg)", 70, 260, key="sys_bp")
            c1.number_input("Diastolic BP (mmHg)", 40, 160, key="dia_bp")
            c1.number_input("Resting heart rate (bpm)", 30, 220, key="resting_hr")
            c2.number_input("Total cholesterol (mg/dL)", 80, 500, key="cholesterol")
            c2.number_input("LDL (mg/dL)", 30, 400, key="ldl")
            c3.number_input("HDL (mg/dL)", 10, 120, key="hdl")
            c3.number_input("Fasting blood sugar (mg/dL)", 50, 500, key="fasting_sugar")
        with t3:
            c1, c2, c3 = st.columns(3)
            c1.markdown("**Chest pain**")
            c1.selectbox("Type of chest pain", ["None", "Typical angina", "Atypical", "Non-anginal"],
                         key="chest_pain",
                         help="Typical angina = pressure-like pain in chest on effort, goes away with rest.")
            c1.checkbox("Pain comes on exertion", key="pain_exertion")
            c1.checkbox("Pain relieved by rest", key="pain_relieved_rest")
            c1.checkbox("Pain even at rest", key="pain_at_rest")
            c1.checkbox("Pain spreads to arm / jaw / back", key="pain_radiates")
            c1.checkbox("Cold sweating", key="sweating")
            c1.checkbox("Nausea / vomiting", key="nausea")
            c2.markdown("**Breathing & fluid**")
            c2.selectbox("Shortness of breath", ["None", "On exertion", "At rest", "Lying flat"],
                         key="breathless")
            c2.checkbox("Swelling of legs / ankles", key="leg_swelling")
            c2.checkbox("Constant tiredness", key="fatigue")
            c2.checkbox("Cough at night", key="night_cough")
            c2.checkbox("Rapid weight gain (2 kg in 3 days)", key="weight_gain_fast")
            c3.markdown("**Heart rhythm**")
            c3.checkbox("Palpitations (racing / fluttering heart)", key="palpitations")
            c3.checkbox("Irregular pulse", key="irregular_pulse")
            c3.checkbox("Dizziness", key="dizziness")
            c3.checkbox("Fainting episodes", key="fainting")
        with t4:
            st.caption("Leave these if no report is available.")
            c1, c2 = st.columns(2)
            c1.checkbox("ECG shows ST-segment changes", key="ecg_st_changes")
            c1.checkbox("ECG shows irregular rhythm", key="ecg_irregular")
            c2.number_input("Ejection fraction % from echo (0 = not known)", 0, 80, key="ef")

        submitted = st.form_submit_button("🔍 Run expert system", type="primary",
                                          use_container_width=True)

    if submitted:
        inputs = {k: st.session_state[k] for k in DEFAULTS}
        st.session_state["result"] = ExpertSystem().infer(inputs)
        st.session_state["last_inputs"] = inputs

    r = st.session_state["result"]
    if r is None:
        st.info("Fill the form (or load a demo patient from the sidebar) and press **Run expert system**.")
        return
    show_result(r)


def show_result(r):
    st.divider()
    st.subheader("📋 Result")

    # ---- decision: urgency banner ----
    box = {4: st.error, 3: st.error, 2: st.warning, 1: st.info, 0: st.success}[r.urgency_level]
    box(f"**Decision:** {r.urgency_text}")
    st.markdown(f"### Conclusion: {r.primary}")

    # ---- disease certainty ----
    cols = st.columns(3)
    icons = {"CAD": "🫀", "HF": "💧", "ARR": "📈"}
    for col, (code, name) in zip(cols, DISEASES.items()):
        with col:
            st.markdown(f"**{icons[code]} {name}**")
            st.progress(min(r.cf[code], 1.0))
            st.markdown(f"Certainty: **{r.cf[code]*100:.0f}%** — {interpret_cf(r.cf[code])}")

    # ---- recommendations ----
    st.subheader("✅ Recommendations")
    for i, a in enumerate(r.advice, 1):
        st.markdown(f"{i}. {a}")

    # ---- explanation facility ----
    st.subheader("🧠 Why? (explanation)")
    for code, name in DISEASES.items():
        rules = r.trace[code]
        with st.expander(f"{name}: {len(rules)} rule(s) gave evidence → {r.cf[code]*100:.0f}%"):
            if not rules:
                st.write("No rule matched for this disease.")
            for rule in rules:
                st.markdown(f"- **{rule.rid}**: {rule.text}")

    with st.expander(f"All {len(r.fired)} rules fired (in order)"):
        st.table([{"Rule": x.rid, "Group": x.group, "Rule text": x.text} for x in r.fired])

    with st.expander("Facts inferred by the system"):
        facts = {k: v for k, v in r.facts.items() if v not in (False, 0) or k == "urgency"}
        st.json(facts)

    st.download_button("⬇️ Download report (.txt)", make_report(r),
                       file_name="cardioexpert_report.txt", use_container_width=True)


def make_report(r):
    lines = ["CARDIOEXPERT - HEART DISEASE EXPERT SYSTEM REPORT",
             f"Date: {dt.datetime.now():%Y-%m-%d %H:%M}", "",
             f"Decision: {r.urgency_text}", f"Conclusion: {r.primary}", "",
             "Certainty factors:"]
    lines += [f"  {DISEASES[c]}: {v*100:.0f}% ({interpret_cf(v)})" for c, v in r.cf.items()]
    lines += ["", "Recommendations:"] + [f"  {i}. {a}" for i, a in enumerate(r.advice, 1)]
    lines += ["", "Rules fired:"] + [f"  {x.rid}: {x.text}" for x in r.fired]
    lines += ["", "Disclaimer: educational project, not a medical diagnosis."]
    return "\n".join(lines)


# =====================================================================
# PAGE 2: KNOWLEDGE BASE
# =====================================================================
def kb_page():
    st.title("📚 Knowledge Base")
    counts = count_rules()
    cols = st.columns(len(counts) + 1)
    cols[0].metric("Total rules", len(RULES))
    for col, (grp, n) in zip(cols[1:], counts.items()):
        col.metric(grp, n)
    names = {"Derived": "Derived facts", "CAD": "Coronary Artery Disease",
             "HF": "Heart Failure", "ARR": "Arrhythmia", "Decision": "Decision support"}
    search = st.text_input("Search rules", placeholder="e.g. cholesterol, ECG, fainting")
    for grp, title in names.items():
        rows = [{"ID": r.rid, "Rule": r.text, "CF": f"{r.cf:+.2f}" if r.disease else "-"}
                for r in RULES if r.group == grp and search.lower() in r.text.lower()]
        if rows:
            with st.expander(f"{title} ({len(rows)})", expanded=bool(search)):
                st.table(rows)


# =====================================================================
# PAGE 3: HOW IT WORKS
# =====================================================================
def how_page():
    st.title("⚙️ How CardioExpert works")
    st.markdown("""
**Expert system** = a program that copies the decision-making of a human expert
(here, a cardiologist) using a **knowledge base** of IF-THEN rules and an **inference engine**.

**Components**
1. **User interface** (Streamlit) – collects patient facts.
2. **Working memory** – stores the facts of the current patient.
3. **Knowledge base** – 78 IF-THEN rules in 5 groups.
4. **Inference engine** – *forward chaining*: keeps firing every rule whose IF-part is true
   until no new rule can fire.
5. **Certainty factors** – each disease rule has a CF (−1 to +1). Evidence is combined with
   the MYCIN formula `CF = CF1 + CF2 × (1 − CF1)`.
6. **Explanation facility** – shows *why* a conclusion was reached (rules fired).
7. **Decision support** – sets an urgency level and recommends tests & lifestyle changes.

**Two phases**
- Phase 1: raw data → derived facts (BMI, hypertension, …) → disease evidence.
- Phase 2: decision rules use the final certainty to decide urgency and advice.
""")
    st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=box, style="rounded"];
  UI [label="Streamlit UI\\n(patient data)"];
  WM [label="Working memory\\n(facts)"];
  KB [label="Knowledge base\\n78 rules"];
  IE [label="Inference engine\\nforward chaining + CF"];
  OUT [label="Diagnosis + urgency\\n+ recommendations"];
  EXP [label="Explanation\\n(why?)"];
  UI -> WM -> IE; KB -> IE; IE -> WM; IE -> OUT; IE -> EXP;
}""")
    st.caption("Semester project – Heart Disease Expert System. Educational use only.")


if page.startswith("🩺"):
    diagnosis_page()
elif page.startswith("📚"):
    kb_page()
else:
    how_page()
