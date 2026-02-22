#!/usr/bin/env python3
"""
Menopause KP-CDST: Clinical Decision Support Tool
===================================================
Streamlit app for clinicians to input patient profile and receive
a KP risk assessment with treatment pathway recommendation and
cost-offset analysis.

Foundation: Metri et al. (2023) normative KP data (N=8,089)
COI Model: Gannott (2025) — First published COI for menopausal
           cognitive & mood symptoms (Australia)

Usage:
    pip install streamlit plotly pandas
    streamlit run Menopause_KP_CDST.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Menopause KP-CDST",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
# NORMATIVE DATA (Metri et al. 2023, Int J Tryptophan Res)
# N=8,089 across 120 studies, 31 countries
# ══════════════════════════════════════════════════════════════

# Grand weighted means ± SD
NORM = {
    'serum_trp': {'mean': 60.52, 'sd': 15.38, 'unit': 'μM', 'label': 'Serum Tryptophan'},
    'serum_kyn': {'mean': 1.96, 'sd': 0.51, 'unit': 'μM', 'label': 'Serum Kynurenine'},
    'plasma_trp': {'mean': 51.45, 'sd': 10.47, 'unit': 'μM', 'label': 'Plasma Tryptophan'},
    'plasma_kyn': {'mean': 1.82, 'sd': 0.54, 'unit': 'μM', 'label': 'Plasma Kynurenine'},
}

# Australian-specific normative ranges (Metri 2023, regional data)
NORM_AU = {
    'serum_trp': {'mean': 67.26, 'sd': 11.19, 'unit': 'μM'},
    'serum_kyn': {'mean': 2.43, 'sd': 0.59, 'unit': 'μM'},
    'plasma_trp': {'mean': 42.87, 'sd': 8.51, 'unit': 'μM'},
    'plasma_kyn': {'mean': 2.12, 'sd': 0.52, 'unit': 'μM'},
}

# Age regression coefficients (Metri 2023)
AGE_EFFECTS = {
    'serum_trp':  {'beta': -0.20, 'p': 0.036},
    'serum_kyn':  {'beta': 0.01, 'p': 0.002},
    'plasma_trp': {'beta': -0.74, 'p': 0.001},
    'plasma_kyn': {'beta': 0.02, 'p': 0.001},
}

# Sex effects (female vs male)
SEX_EFFECTS = {
    'serum_trp':  {'beta': -0.22, 'p': 0.012},
    'serum_kyn':  {'beta': -0.05, 'p': 0.001},
}

# KP-linked conditions and their AU annual burden
KP_CONDITIONS = {
    'Major Depression': {'burden_b': 12.6, 'kp_link': 'Elevated KYN/TRP; reduced serotonin', 'risk_hr': 2.5, 'evidence': 'A'},
    'Alzheimer\'s/Dementia': {'burden_b': 18.0, 'kp_link': 'Neurotoxic QUIN accumulation', 'risk_hr': 1.46, 'evidence': 'A'},
    'Type 2 Diabetes': {'burden_b': 6.3, 'kp_link': 'IDO upregulation; insulin resistance', 'risk_hr': 1.3, 'evidence': 'B'},
    'Cardiovascular Disease': {'burden_b': 12.5, 'kp_link': 'KYN predicts cardiovascular events', 'risk_hr': 1.4, 'evidence': 'A'},
    'Anxiety Disorders': {'burden_b': 5.8, 'kp_link': 'Serotonin depletion via KP shunt', 'risk_hr': 1.58, 'evidence': 'A'},
    'Osteoporosis': {'burden_b': 3.4, 'kp_link': 'TRP→serotonin→bone metabolism', 'risk_hr': 1.2, 'evidence': 'B'},
}

# Treatment options
TREATMENTS = {
    'iTBS': {
        'label': 'Intermittent Theta-Burst Stimulation',
        'annual_cost': 7500, 'mbs_rebate': 4080,
        'oop': 3420, 'evidence_mood': 'B', 'evidence_cog': 'GAP',
        'desc': 'Non-invasive brain stimulation targeting DLPFC. MBS Items 14216-14220. '
                'MenoStim trial (Metri PhD, NICM HRI) is FIRST to test for menopausal symptoms.',
        'color': '#E74C3C',
    },
    'MHT (HRT)': {
        'label': 'Menopausal Hormone Therapy',
        'annual_cost': 380, 'mbs_rebate': 0,
        'oop': 380, 'evidence_mood': 'A', 'evidence_cog': 'B',
        'desc': 'Estrogen ± progesterone. PBS $31.60/mo from March 2025. '
                'First-line for VMS; may benefit mood and cognition during critical window (Maki 2013).',
        'color': '#3498DB',
    },
    'SSRI/SNRI': {
        'label': 'Antidepressant Medication',
        'annual_cost': 300, 'mbs_rebate': 0,
        'oop': 300, 'evidence_mood': 'A', 'evidence_cog': 'D',
        'desc': 'Escitalopram, desvenlafaxine. PBS generic. Evidence for mood but NOT cognition. '
                'May worsen cognitive symptoms in some women.',
        'color': '#F39C12',
    },
    'CBT (Better Access)': {
        'label': 'Cognitive Behavioural Therapy',
        'annual_cost': 560, 'mbs_rebate': 560,
        'oop': 0, 'evidence_mood': 'A', 'evidence_cog': 'C',
        'desc': 'MBS Better Access: 6 sessions/yr. Evidence for mood and hot flush coping. '
                'Limited direct evidence for menopausal cognitive symptoms.',
        'color': '#27AE60',
    },
    'Monitoring Only': {
        'label': 'Watchful Waiting + Lifestyle',
        'annual_cost': 320, 'mbs_rebate': 165,
        'oop': 155, 'evidence_mood': 'C', 'evidence_cog': 'C',
        'desc': 'GP monitoring + lifestyle (exercise, sleep, stress management). '
                'Cognitive symptoms are transient — reverse postmenopause (SWAN longitudinal).',
        'color': '#95A5A6',
    },
}

# Productivity loss (from COI Stromberg decomposition, Stromberg et al. 2017)
PER_WOMAN_INDIRECT = 25_917  # AUD/yr (Stromberg-adjusted)
STROMBERG = {
    'Base_A': 196,       # Employee absenteeism
    'Employer_A': 190,   # Employer replacement (SA=0.97)
    'Base_P': 13_166,    # Employee presenteeism
    'Employer_P': 7_110, # Employer friction (SP=0.54)
    'WEP': 5_256,        # Workplace environment problems (SWEP=0.72 x employer-side)
}

# Treatment evidence matrix (0-10 scale for radar chart)
TX_EVIDENCE = {
    'iTBS': {'mood': 6, 'cognition': 3, 'vms': 1, 'cost_eff': 5, 'access': 4, 'safety': 8},
    'MHT (HRT)': {'mood': 7, 'cognition': 6, 'vms': 10, 'cost_eff': 9, 'access': 9, 'safety': 7},
    'SSRI/SNRI': {'mood': 9, 'cognition': 2, 'vms': 5, 'cost_eff': 9, 'access': 10, 'safety': 6},
    'CBT (Better Access)': {'mood': 8, 'cognition': 4, 'vms': 6, 'cost_eff': 8, 'access': 7, 'safety': 10},
    'Monitoring Only': {'mood': 2, 'cognition': 3, 'vms': 1, 'cost_eff': 10, 'access': 10, 'safety': 10},
}

# ARIA-H / Neurovascular risk data
DEMENTIA_LIFETIME_COST = 442_000  # AUD, NATSEM estimate
ARIA_H_PREV_POSTMENO = 0.12  # ~10-15% cerebral microbleeds in postmenopausal women
KP_POSITIVE_PREV = 0.30  # GAP estimate — KP dysregulation in perimenopausal women

# Research gaps — for the Gaps tab
RESEARCH_GAPS = [
    {
        'gap': 'KP dysregulation prevalence in perimenopausal women',
        'current': '30% estimate (GAP) — extrapolated from Metri 2023 age-sex regression, not menopause-specific',
        'fundable': 'Cross-sectional KP profiling of 200-500 women stratified by menopausal stage (STRAW+10)',
        'fills': 'BIM population funnel; CDST threshold calibration',
        'owner': 'MenoStim trial can answer this with baseline KP data from n=72',
        'priority': 'HIGH',
    },
    {
        'gap': 'Cognitive symptom attribution to productivity loss',
        'current': '35% estimate (GAP) — based on Griffiths 2013 UK survey, not validated',
        'fundable': 'WPAI-M (menopause-specific work productivity instrument) validation study',
        'fills': 'COI cognitive burden sheet; per-symptom economic attribution',
        'owner': 'Could be nested within MenoStim as secondary outcome',
        'priority': 'HIGH',
    },
    {
        'gap': 'iTBS efficacy for menopausal cognitive/mood symptoms',
        'current': 'NO DATA — MenoStim is the FIRST trial',
        'fundable': 'Phase III multi-site RCT following MenoStim pilot (n=72)',
        'fills': 'BIM efficacy parameter; CDST treatment ranking validation',
        'owner': 'Metri PhD — MenoStim trial (ACTRN12625000030471)',
        'priority': 'CRITICAL',
    },
    {
        'gap': 'KP biomarker response to iTBS',
        'current': 'No data on whether iTBS modifies KYN/TRP ratio',
        'fundable': 'Pre/post KP profiling within MenoStim (add-on to existing protocol)',
        'fills': 'Mechanistic link; CDST biomarker-guided selection validation',
        'owner': 'Metri — if MenoStim collects bloods pre/post, this is answerable',
        'priority': 'CRITICAL',
    },
    {
        'gap': 'Menopause-attributable fraction for dementia',
        'current': '3% estimate (GAP) — speculative, from Rocca observational HRs',
        'fundable': 'Longitudinal cohort linking perimenopause KP levels to 10-year dementia incidence',
        'fills': 'Dementia cost avoidance model; VDC estimation',
        'owner': 'Requires ALSWH or 45-and-Up linkage — beyond single trial',
        'priority': 'MEDIUM',
    },
    {
        'gap': 'KYNA/QUIN ratio in menopausal women',
        'current': 'Metri 2023 measured TRP and KYN only — not downstream metabolites',
        'fundable': 'Extended KP metabolome (KYNA, QUIN, 3-HK, picolinic acid) in menopause cohort',
        'fills': 'Neuroprotective vs neurotoxic balance; precision risk stratification',
        'owner': 'Metri 2023 explicitly recommends this as future work',
        'priority': 'HIGH',
    },
    {
        'gap': 'ARIA-H prevalence in KP-dysregulated perimenopausal women',
        'current': 'No data linking cerebral microbleeds to KP status in menopause — separate literatures',
        'fundable': 'MRI substudy within MenoStim (n=72): brain MRI + KP bloods at baseline and post-iTBS',
        'fills': 'Triple-hit model validation; neurovascular risk stratification; dementia cost avoidance denominator',
        'owner': 'Nestable within MenoStim if MRI added to protocol; or ALSWH/45-and-Up linkage study',
        'priority': 'CRITICAL',
    },
]

# ══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

def z_score(value, mean, sd):
    """Compute z-score from normative data."""
    if sd == 0:
        return 0
    return (value - mean) / sd

def kp_risk_score(trp, kyn, sample_type='serum', use_au=True, patient_age=51):
    """
    Compute KP risk score from TRP and KYN values.
    Age-adjusted using Metri 2023 regression coefficients.

    Returns dict with z-scores, KYN/TRP ratio, risk level, and interpretation.
    """
    norms = NORM_AU if use_au else NORM
    trp_key = f'{sample_type}_trp'
    kyn_key = f'{sample_type}_kyn'

    # Age-adjusted normative means (Metri 2023 regression)
    adj_trp_mean = age_adjust(patient_age, norms[trp_key]['mean'], AGE_EFFECTS[trp_key]['beta'])
    adj_kyn_mean = age_adjust(patient_age, norms[kyn_key]['mean'], AGE_EFFECTS[kyn_key]['beta'])

    trp_z = z_score(trp, adj_trp_mean, norms[trp_key]['sd'])
    kyn_z = z_score(kyn, adj_kyn_mean, norms[kyn_key]['sd'])

    # KYN/TRP ratio (higher = more KP activation = more risk)
    kyn_trp = kyn / trp if trp > 0 else 0
    # Age-adjusted normative KYN/TRP
    norm_kyn_trp = adj_kyn_mean / adj_trp_mean if adj_trp_mean > 0 else norms[kyn_key]['mean'] / norms[trp_key]['mean']
    kyn_trp_z = (kyn_trp - norm_kyn_trp) / (norm_kyn_trp * 0.25)  # ~25% CV

    # Composite risk: low TRP + high KYN + high ratio = high risk
    # TRP below normal = bad → invert sign
    composite = (-trp_z + kyn_z + kyn_trp_z) / 3

    if composite > 1.5:
        level = 'HIGH'
        color = '#E74C3C'
        interpretation = 'Significant KP dysregulation. Elevated neurotoxic shift. Consider intervention.'
    elif composite > 0.5:
        level = 'MODERATE'
        color = '#F39C12'
        interpretation = 'Moderate KP activation. Monitor and consider targeted intervention if symptomatic.'
    elif composite > -0.5:
        level = 'LOW-MODERATE'
        color = '#F1C40F'
        interpretation = 'Mild KP changes consistent with normal perimenopause transition.'
    else:
        level = 'LOW'
        color = '#27AE60'
        interpretation = 'KP within normal range. Standard menopause management recommended.'

    return {
        'trp_z': round(trp_z, 2),
        'kyn_z': round(kyn_z, 2),
        'kyn_trp': round(kyn_trp, 4),
        'kyn_trp_z': round(kyn_trp_z, 2),
        'composite': round(composite, 2),
        'level': level,
        'color': color,
        'interpretation': interpretation,
        'norm_trp': norms[trp_key]['mean'],
        'norm_kyn': norms[kyn_key]['mean'],
        'adj_trp': round(adj_trp_mean, 2),
        'adj_kyn': round(adj_kyn_mean, 2),
        'norm_kyn_trp': round(norm_kyn_trp, 4),
    }

def age_adjust(age, base_mean, beta):
    """Age-adjust a normative value using regression coefficient."""
    # Centered at mean study age 47.35
    return base_mean + beta * (age - 47.35)

def recommend_treatment(kp_result, symptoms, age, stage):
    """Generate treatment pathway recommendation based on KP profile + symptoms."""
    recommendations = []
    scores = {}

    for tx_name, tx in TREATMENTS.items():
        score = 50  # Base score

        # KP risk level adjustments
        if kp_result['level'] == 'HIGH':
            if tx_name == 'iTBS':
                score += 30  # Strong KP dysregulation → brain stimulation most targeted
            elif tx_name == 'MHT (HRT)':
                score += 20  # Estrogen modulates KP
            elif tx_name == 'Monitoring Only':
                score -= 20
        elif kp_result['level'] == 'MODERATE':
            if tx_name == 'MHT (HRT)':
                score += 20
            elif tx_name == 'iTBS':
                score += 10
        elif kp_result['level'] == 'LOW':
            if tx_name == 'Monitoring Only':
                score += 20
            elif tx_name == 'iTBS':
                score -= 15

        # Symptom adjustments
        if 'Cognitive fog' in symptoms:
            if tx_name == 'iTBS':
                score += 15
            elif tx_name == 'SSRI/SNRI':
                score -= 10  # May worsen cognitive symptoms
        if 'Depression' in symptoms:
            if tx_name in ('SSRI/SNRI', 'CBT (Better Access)'):
                score += 15
            if tx_name == 'iTBS':
                score += 10
        if 'Anxiety' in symptoms:
            if tx_name in ('SSRI/SNRI', 'CBT (Better Access)'):
                score += 10
        if 'Hot flushes/VMS' in symptoms:
            if tx_name == 'MHT (HRT)':
                score += 25  # First-line for VMS
        if 'Sleep disturbance' in symptoms:
            if tx_name == 'MHT (HRT)':
                score += 10
            if tx_name == 'CBT (Better Access)':
                score += 5

        # Menopausal stage
        if stage == 'Late perimenopause':
            if tx_name == 'MHT (HRT)':
                score += 10  # Critical window
        elif stage == 'Early postmenopause':
            if tx_name == 'MHT (HRT)':
                score += 5
        elif stage == 'Late postmenopause (>5yr)':
            if tx_name == 'MHT (HRT)':
                score -= 15  # Past critical window
            if tx_name == 'Monitoring Only':
                score += 10  # Cognitive symptoms often resolve

        # Age factor
        if age > 55:
            if tx_name == 'Monitoring Only':
                score += 5  # Closer to resolution

        scores[tx_name] = max(0, min(100, score))

    # Sort by score
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked, scores

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

st.sidebar.title("Patient Profile")

st.sidebar.markdown("**Demographics**")
age = st.sidebar.slider("Age", 40, 65, 51, help="Patient age in years")
stage = st.sidebar.selectbox("Menopausal Stage", [
    'Early perimenopause',
    'Late perimenopause',
    'Early postmenopause (<5yr)',
    'Late postmenopause (>5yr)',
    'Surgical menopause',
], index=1)

st.sidebar.divider()
st.sidebar.markdown("**Symptoms** (select all that apply)")
symptom_options = [
    'Cognitive fog',
    'Memory problems',
    'Depression',
    'Anxiety',
    'Hot flushes/VMS',
    'Sleep disturbance',
    'Fatigue',
    'Difficulty concentrating at work',
]
symptoms = []
for s in symptom_options:
    if st.sidebar.checkbox(s, key=f'sym_{s}'):
        symptoms.append(s)

st.sidebar.divider()
st.sidebar.markdown("**KP Biomarkers** (if available)")
has_kp = st.sidebar.toggle("KP blood test results available", value=False,
    help="If the patient has had serum/plasma TRP and KYN measured")

sample_type = 'serum'
trp_val = 60.52
kyn_val = 1.96

if has_kp:
    sample_type = st.sidebar.selectbox("Sample type", ['serum', 'plasma'])
    norm_ref = NORM_AU[f'{sample_type}_trp']
    kyn_ref = NORM_AU[f'{sample_type}_kyn']
    trp_val = st.sidebar.number_input(
        f"TRP ({norm_ref['unit']})",
        min_value=5.0, max_value=150.0,
        value=norm_ref['mean'],
        step=1.0,
        help=f"AU normative: {norm_ref['mean']} ± {norm_ref['sd']} {norm_ref['unit']}"
    )
    kyn_val = st.sidebar.number_input(
        f"KYN ({kyn_ref['unit']})",
        min_value=0.1, max_value=10.0,
        value=kyn_ref['mean'],
        step=0.1,
        help=f"AU normative: {kyn_ref['mean']} ± {kyn_ref['sd']} {kyn_ref['unit']}"
    )

st.sidebar.divider()
st.sidebar.markdown("**Risk Factors**")
risk_factors = []
rf_options = [
    ('Early/surgical menopause (<45)', 'early_meno'),
    ('Family history of dementia', 'fam_dementia'),
    ('Bilateral oophorectomy', 'oophorectomy'),
    ('No current MHT use', 'no_mht'),
    ('History of depression', 'hx_depression'),
]
for label, key in rf_options:
    if st.sidebar.checkbox(label, key=f'rf_{key}'):
        risk_factors.append(label)

st.sidebar.divider()
st.sidebar.markdown("**Neuroimaging / Genetics** (if available)")
has_mri = st.sidebar.toggle("MRI neuroimaging available", value=False,
    help="If the patient has had brain MRI — enables ARIA-H risk scoring")

cmb_count = 0
has_wmh = False
has_siderosis = False
apoe_status = 'Unknown'

if has_mri:
    cmb_count = st.sidebar.number_input("Cerebral microbleeds (CMB count)", 0, 50, 0,
        help="Number of cerebral microbleeds on SWI/T2* MRI")
    has_wmh = st.sidebar.checkbox("White matter hyperintensities (Fazekas 2-3)", key='rf_wmh',
        help="Moderate-severe WMH on FLAIR MRI")
    has_siderosis = st.sidebar.checkbox("Superficial siderosis", key='rf_siderosis',
        help="Cortical superficial siderosis — marker of cerebral amyloid angiopathy")

apoe_status = st.sidebar.selectbox("APOE e4 status (if known)", [
    'Unknown', 'Non-carrier', 'Heterozygous (e3/e4)', 'Homozygous (e4/e4)'
], help="Apolipoprotein E genotype — strongest genetic risk factor for AD")

st.sidebar.divider()
use_au = st.sidebar.toggle("Use Australian normative ranges", value=True,
    help="Metri 2023: AU serum TRP 67.26±11.19 vs global 60.52±15.38")

st.sidebar.divider()
st.sidebar.markdown("**Foundation:** Metri et al. 2023")
st.sidebar.markdown("**COI Model:** Gannott 2025")
st.sidebar.markdown("**Trial:** MenoStim (Metri PhD)")
st.sidebar.caption("Research prototype. Not validated for clinical use.")

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

st.title("Menopause KP-CDST")
st.markdown("*Kynurenine Pathway Clinical Decision Support — Menopausal Cognitive & Mood Symptoms*")

if not symptoms and not has_kp:
    st.info("Select symptoms and/or enter KP biomarker values in the sidebar to generate a clinical profile.")
    st.markdown("---")

    st.subheader("About This Tool")
    col1, col2, col3 = st.columns(3)
    col1.metric("KP Normative Studies", "120")
    col2.metric("Individuals", "8,089")
    col3.metric("Countries", "31")

    st.markdown("""
    This tool uses the **kynurenine pathway (KP) normative reference data** from
    Metri et al. (2023) to generate a personalised risk assessment for women
    experiencing menopausal cognitive and mood symptoms.

    **The Menopause-KP Connection:**
    - Women have significantly **lower baseline TRP** (β = −0.22, p=.012)
    - TRP **falls with age** (β = −0.74, p<.001) while KYN **rises** (β = 0.02, p<.001)
    - This shifts the pathway toward **neurotoxic metabolites** at precisely the menopause transition
    - KP dysregulation is linked to depression, dementia, CVD, and T2DM

    **Economic Context:**
    - National burden: **$41.9B AUD/yr** (first published COI model)
    - Per-woman productivity loss: **$25,917 AUD/yr** (Stromberg-adjusted)
    - KP screening cost: **$80 AUD** (standard pathology HPLC)
    - KP-targeted iTBS reduces cost per responder by **61%** vs unselected

    **Research Throughline:**
    1. KP normative data → Metri et al. 2023 (diagnostic foundation)
    2. Complementary therapy landscape → Metri et al. 2026, Climacteric (gap identification)
    3. iTBS for menopausal symptoms → MenoStim trial, Metri PhD (intervention)
    4. Economic framework → Gannott 2025 COI + BIM (commercial case)
    """)
    st.stop()

# ── Compute KP score ──
kp = kp_risk_score(trp_val, kyn_val, sample_type, use_au, patient_age=age)

# ── Compute treatment recommendations ──
ranked, scores = recommend_treatment(kp, symptoms, age, stage)

# ── Header metrics ──
st.markdown("### Patient Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Age", f"{age} years")
col2.metric("Stage", stage.split('(')[0].strip())
col3.metric("Symptoms", f"{len(symptoms)}")
col4.metric("Risk Factors", f"{len(risk_factors)}")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "KP Risk Profile", "Treatment Pathway", "Cost Analysis",
    "Dementia Risk", "Research Gaps", "Clinical Summary"
])

# ── TAB 1: KP Risk Profile ──
with tab1:
    st.subheader("Kynurenine Pathway Risk Assessment")

    if has_kp:
        # KP score display
        col1, col2, col3 = st.columns(3)
        col1.metric("KP Risk Level", kp['level'],
                    help="Based on composite of TRP z-score, KYN z-score, and KYN/TRP ratio")
        col2.metric("KYN/TRP Ratio", f"{kp['kyn_trp']:.4f}",
                    f"{'↑' if kp['kyn_trp'] > kp['norm_kyn_trp'] else '↓'} vs norm {kp['norm_kyn_trp']:.4f}")
        col3.metric("Composite Score", f"{kp['composite']:.2f}",
                    help=">1.5 = HIGH, 0.5-1.5 = MODERATE, -0.5-0.5 = LOW-MOD, <-0.5 = LOW")

        st.markdown(f"**Interpretation:** {kp['interpretation']}")

        # Z-score chart
        fig_z = go.Figure()
        labels = ['TRP (inverted)', 'KYN', 'KYN/TRP Ratio']
        values = [-kp['trp_z'], kp['kyn_z'], kp['kyn_trp_z']]
        colors = ['#E74C3C' if v > 1 else '#F39C12' if v > 0.5 else '#27AE60' for v in values]

        fig_z.add_trace(go.Bar(
            x=labels, y=values,
            marker_color=colors,
            text=[f"z = {v:.2f}" for v in values],
            textposition='outside',
        ))
        fig_z.add_hline(y=1.0, line_dash='dash', line_color='red',
                       annotation_text='Risk threshold (+1 SD)')
        fig_z.add_hline(y=0, line_color='gray')
        fig_z.update_layout(
            height=350, yaxis_title='Z-Score (deviation from normative)',
            title='KP Biomarker Profile vs Normative Range',
        )
        st.plotly_chart(fig_z, width='stretch')

        # Normative comparison table — AGE-ADJUSTED
        st.markdown(f"**Normative Comparison — Age-Adjusted to {age}yr (Metri et al. 2023 regression)**")
        ref_label = "Australian" if use_au else "Global"
        norms = NORM_AU if use_au else NORM
        comparison = pd.DataFrame([
            {
                'Metabolite': 'Tryptophan (TRP)',
                'Patient': f"{trp_val:.1f} μM",
                f'Age-Adj Mean (age {age})': f"{kp['adj_trp']:.2f} μM",
                f'{ref_label} Population Mean': f"{norms[f'{sample_type}_trp']['mean']:.2f} μM",
                'Z-Score': f"{kp['trp_z']:+.2f}",
                'Status': '⚠️ LOW' if kp['trp_z'] < -1 else '✅ Normal' if abs(kp['trp_z']) < 1 else '↑ High',
            },
            {
                'Metabolite': 'Kynurenine (KYN)',
                'Patient': f"{kyn_val:.2f} μM",
                f'Age-Adj Mean (age {age})': f"{kp['adj_kyn']:.2f} μM",
                f'{ref_label} Population Mean': f"{norms[f'{sample_type}_kyn']['mean']:.2f} μM",
                'Z-Score': f"{kp['kyn_z']:+.2f}",
                'Status': '⚠️ HIGH' if kp['kyn_z'] > 1 else '✅ Normal' if abs(kp['kyn_z']) < 1 else '↓ Low',
            },
            {
                'Metabolite': 'KYN/TRP Ratio',
                'Patient': f"{kp['kyn_trp']:.4f}",
                f'Age-Adj Mean (age {age})': f"{kp['norm_kyn_trp']:.4f}",
                f'{ref_label} Population Mean': f"{norms[f'{sample_type}_kyn']['mean'] / norms[f'{sample_type}_trp']['mean']:.4f}",
                'Z-Score': f"{kp['kyn_trp_z']:+.2f}",
                'Status': '⚠️ ELEVATED' if kp['kyn_trp_z'] > 1 else '✅ Normal' if abs(kp['kyn_trp_z']) < 1 else 'Low',
            },
        ])
        st.dataframe(comparison, width='stretch', hide_index=True)
        st.caption(f"Age adjustment: TRP β={AGE_EFFECTS[f'{sample_type}_trp']['beta']}/yr, "
                  f"KYN β={AGE_EFFECTS[f'{sample_type}_kyn']['beta']}/yr, "
                  f"centered at study mean age 47.35yr (Metri 2023)")

    else:
        st.warning("No KP biomarker data entered. Risk assessment is based on symptoms and demographics only.")
        st.markdown("""
        **To enhance this assessment:**
        - Order serum TRP and KYN (standard HPLC at any pathology lab, ~$80 AUD)
        - Results scored against Metri et al. (2023) normative ranges
        - KP-guided treatment selection reduces cost per responder by 61%
        """)

        # Age-based expected KP trajectory
        ages = list(range(40, 66))
        expected_trp = [age_adjust(a, NORM_AU['serum_trp']['mean'], AGE_EFFECTS['serum_trp']['beta']) for a in ages]
        expected_kyn = [age_adjust(a, NORM_AU['serum_kyn']['mean'], AGE_EFFECTS['serum_kyn']['beta']) for a in ages]

        fig_traj = go.Figure()
        fig_traj.add_trace(go.Scatter(x=ages, y=expected_trp, name='Expected TRP (μM)',
                                      line=dict(color='#3498DB', width=3)))
        fig_traj.add_vline(x=age, line_dash='dash', line_color='red',
                          annotation_text=f'Patient age: {age}')
        fig_traj.update_layout(
            height=300, title='Expected Serum TRP Trajectory with Age (Metri 2023 regression)',
            xaxis_title='Age', yaxis_title='Serum TRP (μM)',
        )
        st.plotly_chart(fig_traj, width='stretch')

    # Symptom profile
    st.markdown("---")
    st.markdown("**Symptom Profile**")
    if symptoms:
        cog_symptoms = [s for s in symptoms if s in ('Cognitive fog', 'Memory problems', 'Difficulty concentrating at work')]
        mood_symptoms = [s for s in symptoms if s in ('Depression', 'Anxiety')]
        physical = [s for s in symptoms if s not in cog_symptoms + mood_symptoms]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Cognitive**")
            for s in cog_symptoms:
                st.markdown(f"- {s}")
            if not cog_symptoms:
                st.caption("None reported")
        with col2:
            st.markdown("**Mood**")
            for s in mood_symptoms:
                st.markdown(f"- {s}")
            if not mood_symptoms:
                st.caption("None reported")
        with col3:
            st.markdown("**Physical**")
            for s in physical:
                st.markdown(f"- {s}")
            if not physical:
                st.caption("None reported")
    else:
        st.caption("No symptoms selected")

# ── TAB 2: Treatment Pathway ──
with tab2:
    st.subheader("Treatment Pathway Recommendation")
    st.caption("Ranked by clinical suitability score (symptoms + KP profile + menopausal stage)")

    # Horizontal bar chart
    tx_names = [r[0] for r in ranked]
    tx_scores = [r[1] for r in ranked]
    tx_colors = [TREATMENTS[t]['color'] for t in tx_names]

    fig_tx = go.Figure()
    fig_tx.add_trace(go.Bar(
        y=list(reversed(tx_names)),
        x=list(reversed(tx_scores)),
        orientation='h',
        marker_color=list(reversed(tx_colors)),
        text=[f"{s}/100" for s in reversed(tx_scores)],
        textposition='outside',
    ))
    fig_tx.update_layout(
        height=350,
        xaxis_title='Suitability Score',
        xaxis=dict(range=[0, 110]),
        margin=dict(l=10, r=80, t=30, b=40),
    )
    st.plotly_chart(fig_tx, width='stretch')

    # Detailed cards
    st.markdown("---")
    for i, (tx_name, score) in enumerate(ranked):
        tx = TREATMENTS[tx_name]
        rank_label = '🥇' if i == 0 else '🥈' if i == 1 else '🥉' if i == 2 else f'#{i+1}'

        with st.expander(f"{rank_label} {tx_name} — Score: {score}/100", expanded=(i == 0)):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Annual Cost", f"${tx['annual_cost']:,}")
            col2.metric("MBS Rebate", f"${tx['mbs_rebate']:,}")
            col3.metric("Patient OOP", f"${tx['oop']:,}")
            col4.metric("Evidence (Mood/Cog)", f"{tx['evidence_mood']}/{tx['evidence_cog']}")
            st.markdown(tx['desc'])

    # Radar chart — evidence comparison
    st.markdown("---")
    st.subheader("Evidence Profile Comparison")
    st.caption("Scores (0-10): higher = stronger evidence or better performance in that domain")

    radar_cats = ['Mood', 'Cognition', 'VMS', 'Cost-Eff.', 'Access', 'Safety']
    fig_radar = go.Figure()
    for tx_name in [r[0] for r in ranked[:3]]:  # Top 3 treatments
        ev = TX_EVIDENCE[tx_name]
        vals = [ev['mood'], ev['cognition'], ev['vms'], ev['cost_eff'], ev['access'], ev['safety']]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],  # Close the polygon
            theta=radar_cats + [radar_cats[0]],
            fill='toself',
            name=tx_name,
            line=dict(color=TREATMENTS[tx_name]['color']),
            opacity=0.6,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        height=450, showlegend=True,
        title='Top 3 Treatments — Evidence Radar',
    )
    st.plotly_chart(fig_radar, width='stretch')

    # Key note about iTBS
    if ranked[0][0] == 'iTBS':
        st.info("**Note:** iTBS is the #1 recommendation based on this patient's KP profile. "
                "The MenoStim trial (Metri PhD, NICM HRI/WSU) is the FIRST to test iTBS for "
                "menopausal cognitive symptoms. Efficacy is NOT yet established.")

# ── TAB 3: Cost Analysis ──
with tab3:
    st.subheader("Per-Patient Cost-Offset Analysis")

    # Stromberg decomposition
    st.markdown(f"**Annual productivity loss (Stromberg-adjusted): ${PER_WOMAN_INDIRECT:,} AUD**")
    st.caption("Stromberg et al. (2017); applied per Tewhaiti-Smith, Gannott et al. (2025)")

    with st.expander("Stromberg Decomposition (5 buckets)", expanded=False):
        strom_data = pd.DataFrame([
            {'Bucket': '1. Employee Absenteeism', 'AUD/yr': STROMBERG['Base_A'], 'Type': 'Employee'},
            {'Bucket': '2. Employer Replacement (SA=0.97)', 'AUD/yr': STROMBERG['Employer_A'], 'Type': 'Employer'},
            {'Bucket': '3. Employee Presenteeism', 'AUD/yr': STROMBERG['Base_P'], 'Type': 'Employee'},
            {'Bucket': '4. Employer Friction (SP=0.54)', 'AUD/yr': STROMBERG['Employer_P'], 'Type': 'Employer'},
            {'Bucket': '5. WEP (SWEP=0.72 x employer-side)', 'AUD/yr': STROMBERG['WEP'], 'Type': 'Employer'},
        ])
        fig_strom = go.Figure()
        colors = ['#3498DB' if t == 'Employee' else '#E67E22' for t in strom_data['Type']]
        fig_strom.add_trace(go.Bar(
            x=strom_data['Bucket'], y=strom_data['AUD/yr'],
            marker_color=colors,
            text=[f"${v:,}" for v in strom_data['AUD/yr']],
            textposition='outside',
        ))
        fig_strom.update_layout(
            height=350, yaxis_title='AUD/year per symptomatic employed woman',
            title='Productivity Loss Decomposition (Stromberg 2017)',
        )
        st.plotly_chart(fig_strom, width='stretch')
        st.caption("Blue = employee-side cost | Orange = employer-side cost. "
                   "WEP computed from employer-side values only (0.72 x [Employer_A + Employer_P]).")

    # Treatment cost vs offset for each option
    efficacy_assumptions = {
        'iTBS': 0.22 if (has_kp and kp['level'] in ('HIGH', 'MODERATE')) else 0.12,
        'MHT (HRT)': 0.15,
        'SSRI/SNRI': 0.10,
        'CBT (Better Access)': 0.08,
        'Monitoring Only': 0.03,
    }

    cost_data = []
    for tx_name, tx in TREATMENTS.items():
        eff = efficacy_assumptions[tx_name]
        offset = round(PER_WOMAN_INDIRECT * eff)
        net = tx['annual_cost'] - offset
        roi = round((offset / tx['annual_cost'] - 1) * 100, 1) if tx['annual_cost'] > 0 else 0
        be_years = round(tx['annual_cost'] / offset, 1) if offset > 0 else float('inf')
        cost_data.append({
            'Treatment': tx_name,
            'Annual Cost': tx['annual_cost'],
            'Assumed Efficacy': f"{eff:.0%}",
            'Productivity Offset': offset,
            'Net Annual': net,
            'Break-Even (yrs)': be_years,
        })

    df_cost = pd.DataFrame(cost_data)

    # Grouped bar: cost vs offset
    fig_cost = go.Figure()
    fig_cost.add_trace(go.Bar(
        name='Treatment Cost',
        x=df_cost['Treatment'],
        y=df_cost['Annual Cost'],
        marker_color='#E74C3C',
    ))
    fig_cost.add_trace(go.Bar(
        name='Productivity Offset',
        x=df_cost['Treatment'],
        y=df_cost['Productivity Offset'],
        marker_color='#27AE60',
    ))
    fig_cost.update_layout(
        height=400, barmode='group',
        yaxis_title='AUD/year',
        title='Treatment Cost vs Productivity Offset',
    )
    st.plotly_chart(fig_cost, width='stretch')

    st.dataframe(df_cost, width='stretch', hide_index=True)

    if has_kp and kp['level'] in ('HIGH', 'MODERATE'):
        st.success(f"**KP-targeted iTBS:** With biomarker selection, assumed efficacy rises from "
                  f"12% to 22%, making iTBS cost-offset at ${round(PER_WOMAN_INDIRECT * 0.22):,}/yr "
                  f"against a ${TREATMENTS['iTBS']['annual_cost']:,} annual cost.")

    # National scaling
    st.markdown("---")
    st.subheader("National Scaling (if treatment adopted)")
    uptake_pct = st.slider("Assumed uptake (% of eligible)", 0.5, 10.0, 2.0, 0.5)
    eligible = 360_000
    treated = round(eligible * uptake_pct / 100)
    top_tx = ranked[0][0]
    top_eff = efficacy_assumptions[top_tx]
    national_offset = treated * round(PER_WOMAN_INDIRECT * top_eff)
    national_cost = treated * TREATMENTS[top_tx]['annual_cost']
    national_net = national_cost - national_offset

    col1, col2, col3 = st.columns(3)
    col1.metric("Patients Treated", f"{treated:,}")
    col2.metric(f"National Offset ({top_tx})", f"${national_offset:,}")
    col3.metric("Net Budget Impact", f"${national_net:,}",
               delta=f"{'Cost' if national_net > 0 else 'Saving'}",
               delta_color='inverse')

# ── TAB 4: Dementia Risk + ARIA-H ──
with tab4:
    st.subheader("Downstream Dementia Risk Assessment")
    st.caption("EXPLORATORY — Rocca et al. (2007, 2021) + Metri et al. (2023) + ARIA-H neurovascular risk")

    # ── Classical risk factor scoring ──
    dementia_score = 0
    risk_items = []

    if 'Bilateral oophorectomy' in risk_factors:
        dementia_score += 3
        risk_items.append(('Bilateral oophorectomy <menopause', 'HR = 1.46', 'Rocca 2007', '+3'))
    if 'Early/surgical menopause (<45)' in risk_factors:
        dementia_score += 3
        risk_items.append(('Early menopause (<45)', 'aOR = 2.21 for MCI', 'Rocca 2021', '+3'))
    if 'Family history of dementia' in risk_factors:
        dementia_score += 2
        risk_items.append(('Family history', 'OR ~2.0', 'Literature', '+2'))
    if 'No current MHT use' in risk_factors:
        dementia_score += 1
        risk_items.append(('No MHT during critical window', '~30% risk reduction missed', 'Maki 2013', '+1'))
    if has_kp and kp['level'] == 'HIGH':
        dementia_score += 2
        risk_items.append(('KP dysregulation (HIGH)', 'Neurotoxic shift', 'Metri 2023 + Giil 2016', '+2'))
    elif has_kp and kp['level'] == 'MODERATE':
        dementia_score += 1
        risk_items.append(('KP activation (MODERATE)', 'Elevated KYN/TRP', 'Metri 2023', '+1'))
    if 'Cognitive fog' in symptoms or 'Memory problems' in symptoms:
        dementia_score += 1
        risk_items.append(('Current cognitive symptoms', 'Subjective', 'Self-report', '+1'))

    # ── ARIA-H neurovascular scoring ──
    aria_score = 0
    if has_mri:
        if cmb_count >= 5:
            aria_score += 3
            risk_items.append(('Cerebral microbleeds (>=5)', f'{cmb_count} CMBs on MRI', 'ARIA-H literature', '+3'))
        elif cmb_count >= 1:
            aria_score += 2
            risk_items.append(('Cerebral microbleeds (1-4)', f'{cmb_count} CMBs on MRI', 'ARIA-H literature', '+2'))
        if has_wmh:
            aria_score += 2
            risk_items.append(('WMH (Fazekas 2-3)', 'BBB compromise marker', 'Cerebrovascular lit.', '+2'))
        if has_siderosis:
            aria_score += 3
            risk_items.append(('Superficial siderosis', 'CAA marker — high BBB vulnerability', 'ARIA-H literature', '+3'))

    if apoe_status == 'Homozygous (e4/e4)':
        aria_score += 3
        risk_items.append(('APOE e4/e4 homozygous', 'OR ~12 for AD; BBB permeability', 'Literature', '+3'))
    elif apoe_status == 'Heterozygous (e3/e4)':
        aria_score += 2
        risk_items.append(('APOE e3/e4 heterozygous', 'OR ~3.2 for AD', 'Literature', '+2'))

    total_score = dementia_score + aria_score
    max_score = 12 + 11  # Classical max 12 + ARIA max 11

    # ── Neurovascular vulnerability composite ──
    if aria_score >= 5:
        nv_level = 'HIGH'
        nv_color = '#E74C3C'
    elif aria_score >= 2:
        nv_level = 'MODERATE'
        nv_color = '#F39C12'
    else:
        nv_level = 'LOW'
        nv_color = '#27AE60'

    # Combined risk level
    if total_score >= 10:
        risk_level = 'CRITICAL'
        risk_color = '#C0392B'
    elif total_score >= 6:
        risk_level = 'ELEVATED'
        risk_color = '#E74C3C'
    elif total_score >= 3:
        risk_level = 'MODERATE'
        risk_color = '#F39C12'
    else:
        risk_level = 'POPULATION-LEVEL'
        risk_color = '#27AE60'

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Classical Risk", f"{dementia_score}/12")
    col2.metric("ARIA-H / Neurovasc.", f"{aria_score}/11")
    col3.metric("Combined Score", f"{total_score}/23")
    col4.metric("Risk Level", risk_level)

    if risk_items:
        st.markdown("**Contributing Factors:**")
        df_risk = pd.DataFrame(risk_items, columns=['Factor', 'Effect Size', 'Source', 'Points'])
        st.dataframe(df_risk, width='stretch', hide_index=True)

    # ── Triple-Hit Model ──
    st.markdown("---")
    st.subheader("The Triple-Hit Model")
    st.caption("Why menopause + KP dysregulation + neurovascular vulnerability = accelerated neurodegeneration")

    st.markdown("""
    **Hit 1 — Estrogen withdrawal:** BBB becomes more permeable at menopause
    (Bake & Segal 2007). Estrogen normally maintains tight junctions and limits
    neuroinflammatory infiltration. Loss recapitulates an aged BBB phenotype.

    **Hit 2 — KP dysregulation:** Elevated circulating kynurenine crosses the BBB
    via the L-system amino acid transporter. In the brain, microglia convert KYN →
    quinolinic acid (QUIN), an NMDA receptor agonist and excitotoxin. Metri et al.
    (2023) showed TRP falls and KYN rises with age — the steepest shift occurs at
    menopause.

    **Hit 3 — ARIA-H neurovascular vulnerability:** Cerebral microbleeds, WMH,
    and superficial siderosis indicate pre-existing BBB compromise. APOE e4 carriers
    have leakier vessels. In this population, KYN flux into the brain is amplified
    and QUIN-mediated excitotoxicity is compounded.

    **The intersection of all three hits defines the highest-risk, most cost-effective
    population to intervene in.**
    """)

    # Triple-hit funnel visualization
    fig_funnel = go.Figure(go.Funnel(
        y=['All perimenopausal women (2.5M)',
           'Symptomatic (cognitive/mood) (~1.4M)',
           'KP-dysregulated (~30%: 420K)',
           'ARIA-H positive (~12%: 50K)',
           'KP + ARIA-H overlap (~15-50K)'],
        x=[2_500_000, 1_400_000, 420_000, 50_000, 30_000],
        textinfo='value+text',
        marker=dict(color=['#27AE60', '#F1C40F', '#F39C12', '#E74C3C', '#C0392B']),
    ))
    fig_funnel.update_layout(
        height=400,
        title='Precision Medicine Funnel: Population to High-Risk Subgroup',
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_funnel, width='stretch')

    # ── Cost Avoidance Calculator ──
    st.markdown("---")
    st.subheader("Dementia Cost Avoidance Model")
    st.caption("Precision targeting converts speculative population-level estimates into defensible subgroup economics")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Population-Level Estimate**")
        pop_af = st.slider("Population AF (%)", 1.0, 10.0, 3.0, 0.5,
            help="Menopause-attributable fraction for dementia — no published estimate exists (GAP)") / 100
        pop_n = 2_500_000
        pop_avoidable = round(pop_n * pop_af * DEMENTIA_LIFETIME_COST / 1e9, 1)
        st.metric("Population", f"{pop_n:,}")
        st.metric("Avoidable Lifetime Cost", f"${pop_avoidable:.1f}B AUD")
        st.caption(f"AF = {pop_af:.1%} | Source quality: D (GAP — no published data)")

    with col2:
        st.markdown("**Precision Subgroup (ARIA-H + KP-targeted)**")
        default_sub_af = 20.0 if nv_level == 'HIGH' else 12.0 if nv_level == 'MODERATE' else 5.0
        sub_af = st.slider("Subgroup AF (%)", 5.0, 40.0, default_sub_af, 1.0,
            help="Higher AF defensible in defined high-risk subgroup (Rocca HRs in surgical menopause)") / 100
        sub_n_slider = st.slider("High-risk subgroup size", 15_000, 125_000, 50_000, 5_000,
            help="ARIA-H+ and KP-dysregulated perimenopausal women")
        sub_avoidable = round(sub_n_slider * sub_af * DEMENTIA_LIFETIME_COST / 1e9, 1)
        st.metric("Avoidable Lifetime Cost", f"${sub_avoidable:.1f}B AUD")
        st.caption(f"AF = {sub_af:.0%} | Source quality: C (defensible, screenable population)")

    # Per-patient intervention value
    per_patient_value = round(sub_af * DEMENTIA_LIFETIME_COST)
    st.info(f"**Per-patient intervention value:** At {sub_af:.0%} AF, avoidable cost per patient is "
            f"**${per_patient_value:,} AUD** — against a treatment cost of $7,500/yr. "
            f"{'Strongly cost-effective' if per_patient_value > 50000 else 'Cost-effective'} "
            f"by any ICER threshold.")

    st.markdown("---")
    st.markdown("**KP Connection to Neurodegeneration (Metri et al. 2023):**")
    st.markdown("""
    The kynurenine pathway is a central mediator between neuroinflammation and neurodegeneration:
    - **Tryptophan** is degraded via KP → **kynurenine** → branch point
    - Neuroprotective branch: → **kynurenic acid (KYNA)** (NMDA antagonist)
    - Neurotoxic branch: → **3-HK** → **quinolinic acid (QUIN)** (NMDA agonist, excitotoxin)
    - With aging and estrogen withdrawal, the balance shifts toward **QUIN** (Giil et al. 2016)
    - Metri et al. (2023) established the normative reference enabling detection of this shift
    - **ARIA-H markers indicate the BBB is already compromised** — QUIN flux amplified

    **Intervention during the critical window (perimenopause) may modify this trajectory.**
    """)

    st.warning("**EXPLORATORY:** Dementia risk scoring is not validated. The triple-hit model "
              "is hypothesis-generating. ARIA-H risk in KP-dysregulated perimenopausal women "
              "has never been studied — this is a fundable research question (see Research Gaps). "
              "The MenoStim trial can begin to answer it if MRI + KP bloods are collected.")

# ── TAB 5: Research Gaps ──
with tab5:
    st.subheader("Research Gaps & Fundable Questions")
    st.caption("Where Metri's research fills evidence gaps — and what remains to be answered")

    st.markdown("""
    Every **GAP** estimate in the COI/BIM model represents a **fundable research question**.
    The MenoStim trial (n=72) can answer several of these directly. Others require
    larger cohorts or data linkage studies.
    """)

    for i, gap in enumerate(RESEARCH_GAPS):
        priority_color = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡'}.get(gap['priority'], '⚪')
        with st.expander(f"{priority_color} {gap['gap']} [{gap['priority']}]", expanded=(i < 2)):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Current evidence:** {gap['current']}")
                st.markdown(f"**What it fills:** {gap['fills']}")
            with col2:
                st.markdown(f"**Fundable study:** {gap['fundable']}")
                st.markdown(f"**Who can answer it:** {gap['owner']}")

    st.markdown("---")
    st.subheader("The Grant Narrative")
    st.markdown("""
    **NHMRC / MRFF application structure:**

    1. **Burden:** $41.9B AUD/yr national cost of menopausal cognitive & mood symptoms
       (Gannott 2025 — first published COI)
    2. **Mechanism:** KP dysregulation accelerates at menopause — lower TRP, rising KYN,
       neurotoxic shift (Metri et al. 2023; Giil et al. 2016)
    3. **Gap:** No validated intervention for menopausal cognitive symptoms;
       complementary therapy evidence is low/very low (Metri et al. 2026)
    4. **Intervention:** iTBS — first trial for this indication (MenoStim, Metri PhD)
    5. **Biomarker:** KP profiling enables targeted patient selection
       (cost per responder drops 61% with biomarker guidance)
    6. **Translation:** CDST algorithm converts research into clinical tool
    7. **Economic case:** BIM shows KP-targeted iTBS is near cost-neutral within 5 years

    **Key question for Phase III funding:**
    *"Does iTBS in KP-dysregulated perimenopausal women modify the kynurenine-neurodegeneration
    trajectory and reduce cognitive/mood symptom burden?"*

    If MenoStim collects pre/post KP biomarkers, the pilot alone can establish
    whether iTBS shifts the KYN/TRP ratio — the mechanistic proof needed for a
    larger efficacy trial.
    """)

# ── TAB 6: Clinical Summary ──
with tab6:
    st.subheader("Clinical Summary Report")

    # Generate summary text
    summary_lines = []
    summary_lines.append(f"**Patient:** {age}yo female, {stage}")
    summary_lines.append(f"**Symptoms:** {', '.join(symptoms) if symptoms else 'None reported'}")
    summary_lines.append(f"**Risk Factors:** {', '.join(risk_factors) if risk_factors else 'None'}")

    if has_kp:
        summary_lines.append(f"**KP Biomarkers ({sample_type}):** TRP {trp_val:.1f} μM, KYN {kyn_val:.2f} μM, KYN/TRP {kp['kyn_trp']:.4f}")
        summary_lines.append(f"**KP Risk Level:** {kp['level']} (composite z = {kp['composite']:.2f})")
        summary_lines.append(f"**Interpretation:** {kp['interpretation']}")
    else:
        summary_lines.append("**KP Biomarkers:** Not available — recommend serum TRP/KYN ($80 AUD)")

    summary_lines.append("")
    summary_lines.append(f"**Recommended Treatment:** {ranked[0][0]} (score {ranked[0][1]}/100)")
    summary_lines.append(f"**Alternative:** {ranked[1][0]} (score {ranked[1][1]}/100)")
    summary_lines.append("")
    summary_lines.append(f"**Estimated annual productivity loss:** ${PER_WOMAN_INDIRECT:,} AUD")
    eff = efficacy_assumptions[ranked[0][0]]
    summary_lines.append(f"**Potential offset with {ranked[0][0]}:** ${round(PER_WOMAN_INDIRECT * eff):,} AUD ({eff:.0%} improvement)")
    summary_lines.append("")
    summary_lines.append(f"**Dementia risk score:** {total_score}/23 ({risk_level})")
    summary_lines.append(f"  Classical: {dementia_score}/12 | ARIA-H/Neurovasc: {aria_score}/11")
    if nv_level != 'LOW':
        summary_lines.append(f"  Neurovascular vulnerability: {nv_level}")
    if has_mri and cmb_count > 0:
        summary_lines.append(f"  CMBs: {cmb_count} | WMH: {'Yes' if has_wmh else 'No'} | Siderosis: {'Yes' if has_siderosis else 'No'}")
    if apoe_status not in ('Unknown', 'Non-carrier'):
        summary_lines.append(f"  APOE: {apoe_status}")

    for line in summary_lines:
        st.markdown(line)

    st.markdown("---")
    st.markdown("**References:**")
    st.markdown("""
    - Metri NJ, et al. (2023). Normative data on serum/plasma TRP and KYN. *Int J Tryptophan Res*, 16. (KP normative)
    - Metri NJ, et al. (2026). Complementary therapies for menopausal symptoms. *Climacteric*. (Therapy landscape)
    - Metri NJ, et al. (2025). The MenoStim Trial. ACTRN12625000030471. (iTBS pilot)
    - Gannott M (2025). COI: Menopausal Cognitive & Mood Symptoms in Australia. (Economic model)
    - Tewhaiti-Smith JMK, Gannott M, et al. (2025). Cost of Endometriosis and CPP in NZ. *Women (MDPI)*. (Stromberg method)
    - Rocca WA, et al. (2007, 2021). Oophorectomy and dementia risk. *Neurology; JAMA Netw Open*. (Dementia pathway)
    - Maki PM (2013). Critical window hypothesis. *Menopause*. (HRT timing)
    - Bake S, Segal M (2007). 17β-estradiol differentially regulates BBB permeability. *Endocrinology*. (BBB + estrogen)
    - Giil LM, et al. (2016). KP metabolites in dementia. (KP → neurodegeneration)
    - Guillemin GJ (2012). Quinolinic acid neurotoxicity review. (QUIN excitotoxicity)
    """)

    # Copy-friendly text
    st.markdown("---")
    plain_text = "\n".join([line.replace('**', '') for line in summary_lines])
    st.text_area("Copy-friendly text:", value=plain_text, height=300)

# ── Footer ──
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; font-size: 0.85em;'>
<b>Menopause KP-CDST v1.0</b><br>
KP Normative Data: Metri et al. 2023 (Int J Tryptophan Res, N=8,089)<br>
COI Model: Gannott 2025 (First published model) | Stromberg: Tewhaiti-Smith, Gannott et al. 2025<br>
ARIA-H Triple-Hit Model: Bake & Segal 2007; Giil 2016; Guillemin 2012<br>
Trial: MenoStim (Metri PhD, NICM HRI, WSU) — ACTRN12625000030471<br>
<br>
<b>Research prototype — not validated for clinical use.</b><br>
For clinical decision support only. Not a substitute for clinical judgement.
</div>
""", unsafe_allow_html=True)
