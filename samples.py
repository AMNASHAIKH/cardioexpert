"""Default input values and demo patients (useful in the viva)."""

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
