# Menopause KP-CDST: Clinical Decision Support Tool
## Kynurenine Pathway-Guided Treatment Selection for Menopausal Cognitive & Mood Symptoms

### Overview

The Menopause KP-CDST is the first clinical decision support tool to integrate kynurenine pathway (KP) biomarker data with menopausal symptom profiling for treatment pathway selection. Built on the normative reference dataset established by Metri et al. (2023) — the largest systematic review and meta-analysis of serum and plasma tryptophan and kynurenine concentrations (N=8,089 across 120 studies, 31 countries) — this tool enables clinicians to score individual patients against population-level norms and generate personalised treatment recommendations.

The tool accompanies the first published Cost of Illness model for menopausal cognitive and mood symptoms in Australia (Gannott, 2025), which quantifies the national burden at AUD $41.9 billion annually and the per-woman productivity loss at AUD $25,917/year (Stromberg-adjusted).

### Clinical Rationale

The menopause transition represents a convergence point for KP dysregulation:

- Women have significantly lower baseline serum tryptophan than men (beta = -0.22, p=.012; Metri et al. 2023)
- Tryptophan falls with age (plasma beta = -0.74, p<.001) while kynurenine rises (beta = 0.02, p<.001)
- The mean age of the Metri et al. normative dataset (47.35 years) sits precisely at the perimenopause window
- Estrogen modulates KP enzyme activity; menopause-related estrogen withdrawal accelerates the shift toward neurotoxic metabolites

This compounding effect — lower female baseline, accelerating age-related degradation, and hormonal withdrawal — creates the steepest KP shift toward neurotoxicity at exactly the time women report the highest rates of cognitive and mood symptoms (44-65% reporting cognitive difficulties; 37% depression/anxiety).

### Functionality

The tool provides five integrated modules:

**1. KP Risk Profile**
Scores patient serum or plasma TRP and KYN values against Metri et al. (2023) normative ranges (global or Australian-specific). Computes z-scores for TRP, KYN, and the KYN/TRP ratio. Generates a composite risk level (LOW / LOW-MODERATE / MODERATE / HIGH) reflecting the degree of KP dysregulation. When biomarkers are unavailable, displays age-adjusted expected trajectories from the Metri regression coefficients.

**2. Treatment Pathway**
Ranks five treatment options by clinical suitability score (0-100) based on the patient's KP profile, symptom constellation, menopausal stage, and risk factors:
- Intermittent Theta-Burst Stimulation (iTBS) — MBS Items 14216-14220
- Menopausal Hormone Therapy (MHT) — PBS from March 2025
- SSRI/SNRI antidepressants — PBS generic
- CBT via Better Access — MBS Items 10968-10970
- Monitoring + lifestyle

**3. Cost-Offset Analysis**
Calculates per-patient treatment cost versus productivity offset for each option. Uses Stromberg-adjusted productivity loss (Tewhaiti-Smith, Gannott et al. 2025) and efficacy assumptions that adjust based on KP profile — KP-targeted iTBS assumes 22% productivity improvement (NNT=3) versus 12% unselected (NNT=8). Includes national scaling with adjustable uptake slider.

**4. Dementia Risk Assessment**
Exploratory scoring (0-12) based on Rocca et al. (2007: HR=1.46 for bilateral oophorectomy; 2021: aOR=2.21 for MCI) and KP dysregulation status. Connects to the kynurenine-neurodegeneration pathway established in the broader literature and contextualised by Metri et al. (2023) normative data. Clearly flagged as hypothesis-generating.

**5. Clinical Summary**
Generates a copy-friendly text report suitable for medical records, including KP interpretation, treatment recommendation, cost-offset estimate, and dementia risk score with references.

### Research Throughline

The tool operationalises a coherent research narrative:

1. **Metri et al. (2023)**, Int J Tryptophan Res — Normative KP data providing the diagnostic foundation (N=8,089)
2. **Metri et al. (2026)**, Climacteric — IMS systematic review identifying the evidence gap for brain stimulation in menopausal symptoms (158 studies reviewed; cognitive outcomes not a primary endpoint; brain stimulation not covered)
3. **The MenoStim Trial** (ACTRN12625000030471) — First randomised, sham-controlled, double-blinded pilot clinical trial of iTBS for menopausal cognitive and mood symptoms (n=72; Metri, Cavaleri, Alhassani, Ee, Steiner-Lim)
4. **Gannott (2025) COI + BIM** — Economic framework quantifying the $41.9B burden and demonstrating that KP-targeted iTBS is near cost-neutral within 5 years

### Competitive Moat

The CDST represents the commercialisation layer in a three-tier moat stack:
- **Layer 1:** Normative KP reference ranges (Metri 2023 — published, citable)
- **Layer 2:** Menopause-specific KP thresholds and validated iTBS protocol (MenoStim trial — pending)
- **Layer 3:** The CDST algorithm matching KP biomarker profile + symptoms + risk factors to treatment recommendation (this tool)

Without the CDST, iTBS is a commodity (any clinic can purchase a TMS device). With the CDST, the algorithm becomes the defensible intellectual property, creating a data flywheel where every patient treated improves the model's predictive accuracy.

### Technical Requirements

```
pip install streamlit plotly pandas
streamlit run Menopause_KP_CDST.py
```

### Key References

- Metri NJ, Butt AS, Murali A, Steiner-Lim GZ, Lim CK (2023). Normative data on serum and plasma tryptophan and kynurenine concentrations from 8089 individuals across 120 studies. *Int J Tryptophan Res*, 16:11786469231211184.
- Metri NJ, et al. (2026). Complementary therapies for management of menopausal symptoms: IMS systematic review. *Climacteric*. DOI:10.1080/13697137.2025.2584061.
- Tewhaiti-Smith JMK, Gannott M, et al. (2025). The cost of endometriosis and chronic pelvic pain burden in New Zealand. *Women (MDPI)*, 5(4):47.
- Stromberg C, et al. (2017). Estimating the burden and economic cost of exposure to occupational hazards.
- Rocca WA, et al. (2007). Increased risk of cognitive impairment or dementia in women who underwent oophorectomy before menopause. *Neurology*, 69(11):1074-1083.
- Rocca WA, et al. (2021). Premenopausal bilateral oophorectomy and risk of MCI. *JAMA Netw Open*, 4(11):e2131448.

### Disclaimer

Research prototype. Not validated for clinical use. All efficacy assumptions for iTBS in menopausal symptoms are conditional on the outcomes of the MenoStim trial. The KP risk scoring algorithm has not been validated against clinical outcomes. For clinical decision support only — not a substitute for clinical judgement.

---
*Developed by M. Gannott (Adjunct Fellow, NICM Health Research Institute, Western Sydney University)*
*For N-J. Metri (NICM HRI, WSU) — February 2025*
