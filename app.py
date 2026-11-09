"""
CardioExpert - Streamlit user interface
Run locally:  streamlit run app.py
"""
import datetime as dt

import streamlit as st

from engine import DISEASES, RULES, ExpertSystem, count_rules, interpret_cf
from samples import DEFAULTS, SAMPLES, make_patient

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
