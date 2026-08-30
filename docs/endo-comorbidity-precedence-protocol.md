# Testing the "Endometriosis Is Upstream" Hypothesis

**Status:** design memo, not a result. Written to scope whether the hypothesis is testable
with existing data, and what would have to be true for a positive result to mean anything.

**Hypothesis (as stated):** endometriosis temporally precedes, and therefore drives, the bulk
of its associated comorbidities. If true, earlier diagnosis and effective intervention should
bend the downstream comorbidity curve — which converts endometriosis from a cost line into an
upstream prevention target.

---

## 1. The short answer

Yes, the data exists. National registers and multi-site EHR networks record dated diagnoses
across a whole life course, and several groups have already used them on exactly this question.

But the temporal test as framed cannot deliver the conclusion, for one structural reason:

> **The exposure's measurement error is correlated with the outcome.**

Endometriosis diagnosis date is not onset date. Mean diagnostic delay is ~6.6 years across
studies (range 1.5–11.3); recent cohorts report ~10 years. Worse, delay is *longer in women
with more comorbidities* (+0.3 years per comorbidity, p < 0.0001, ComPaRe-Endometriosis).

That is differential misclassification, not noise. Every year of delay mechanically relocates
comorbidity diagnoses to *before* the endometriosis index date, and it does so most strongly
in precisely the women the hypothesis is about. Consequences:

- A naive analysis is biased **against** the hypothesis. Finding "comorbidities came first" is
  uninformative — it is the expected artefact.
- Finding "endometriosis came first" *despite* that bias would be notable, but only if the
  analysis can show the result survives the plausible delay distribution.

So the design question is not "can we see the order of diagnoses." It is "can we recover the
order of *onsets* from data that only records diagnoses."

---

## 2. What is already published

Worth knowing before committing resources — parts of this are done.

| Study | Design | Bearing on the hypothesis |
|---|---|---|
| Surrey et al. 2018, *J Womens Health* | 26,961 cases / 107,844 matched controls | All 22 comorbidities elevated; ≥2× for 9 (infertility, ovarian cyst, fibroids, PID, interstitial cystitis, IBS, constipation/dyschezia, ovarian and endometrial cancer). This is the effect-size backbone. |
| Rossi / Uimari et al. 2022, *Fertil Steril* — NFBC1966 | Whole birth cohort, 349 endo / 3,499 referents, ICD codes 1968–2016 | Near-lifetime capture from birth. Diagnoses accumulated at a significantly *younger age* in endometriosis. Note the framing: younger, not "after endo." |
| Finnish nationwide register 2025, *Hum Reprod* (deaf032) | 2,680 women **surgically verified before age 25**, 5,338 matched referents, followed to 2019 | The short-delay subgroup design, already executed. 40 vs 18 hospital visits post-index. **Pre-existing depression/anxiety predicted higher HRs for subsequent somatic disorders** — the psychiatric comorbidity preceded and modified rather than followed. |
| Cell Rep Med 2025 (UC 6-centre EHR) | 40,000+ endometriosis patients | Hundreds of associated conditions; clustering into distinct subphenotypes (psychiatric cluster, autoimmune cluster). Implies "endo drives comorbidities" is wrong as a single statement — it is cluster-specific at best. |
| *Hum Reprod Update* 2023, 29(5):655 | MR + genetic correlation across comorbidities | **Migraine–endometriosis is non-causal** — shared genetically controlled mechanisms. |
| Adewuyi et al. 2023, *Hum Genet* | Genomic overlap, 76 comorbidities | Predominantly **pleiotropic**, not causal. |
| Bidirectional MR, depression | | Evidence runs **depression → endometriosis**, the reverse of the hypothesised direction. |
| MR, cardiovascular/cerebrovascular 2025 | | **Null.** No causal effect of endometriosis on CVD, HF, stroke, hypertension — against a well-known observational HR ≈ 1.6 (NHSII). |

### What that adds up to

The best available causal evidence — Mendelian randomization, which is immune to diagnostic
delay because genotype is fixed at conception — mostly does **not** support endometriosis as the
upstream driver for the headline comorbidities. It supports a **shared liability** (inflammatory
and pain-sensitisation) of which endometriosis is one downstream manifestation among several.

The observational/MR gap (CVD HR 1.6 observational, null in MR) is itself the tell: it is the
signature of confounding or ascertainment, not of a causal effect that MR is underpowered to see.

---

## 3. The reframe that survives the evidence

The economic and policy argument does **not** require endometriosis to be causally upstream.
It requires endometriosis to be the **earliest reliably diagnosable and treatable marker** of a
shared multi-system liability.

- **"Endometriosis causes the comorbidities"** — not currently supportable; MR contradicts it for
  migraine, depression and CVD.
- **"Endometriosis is the clinical entry point to a lifetime multimorbidity trajectory, and the
  only point in that trajectory where a specific, treatable lesion exists"** — supportable today
  on published evidence, and carries the same policy implication.

The second claim is weaker causally and stronger strategically: it survives adversarial review,
and it still justifies early diagnosis as a system intervention.

### Cost-model consequence (flagged, not resolved)

If endometriosis and depression share a common cause rather than one causing the other, then
depression costs **cannot** be attributed to endometriosis in a COI model. This is the same
double-counting problem the NZ paper handled between CPP and endometriosis via the 43.4% DAC
scalar — but applied across the full comorbidity set, where the attribution fraction is unknown
and probably much lower. Any comorbidity-inclusive burden figure inherits this exposure.

---

## 4. If the temporal analysis is run anyway — the design that would hold up

### 4.1 Anchoring: do not use the endometriosis diagnosis date

Replace the index date with a symptom-onset proxy, constructed from the pre-diagnostic record:

- first dysmenorrhoea / pelvic pain code
- first NSAID or combined oral contraceptive prescribed for pain
- first gynaecology referral

UK primary care data (CPRD Aurum) captures this journey; hospital-only registers do not.

Use **time since menarche** as the analysis clock rather than calendar age.

### 4.2 Delay-shift sensitivity analysis (mandatory)

Re-run the primary analysis shifting the endometriosis index date back by 0, 3, 5, 8 and 11
years, spanning the empirical delay distribution. Report the lag at which the precedence
conclusion flips.

- Survives at lag 0 → strong.
- Requires ≥8 years of assumed lag → the answer was assumed, not measured.

### 4.3 Bias calibration via neurodevelopmental negative controls

This sharpens the original instinct about late-onset conditions, and inverts it.

ADHD and autism are **fixed at birth** but in women are routinely diagnosed in the twenties and
thirties. Endometriosis therefore *cannot* cause them. So:

> Any apparent endometriosis → incident-ADHD hazard is a **direct measurement of the surveillance
> and ascertainment bias in your own pipeline.**

Estimate that hazard ratio and treat it as the noise floor. **Any outcome whose HR does not
clear the ADHD floor is not evidence.**

Caveat to state explicitly: endometriosis–ADHD also carries genuine shared-genetic correlation,
so this is an *upper bound* on surveillance bias rather than a clean estimate. That makes it
conservative, which is the right direction.

**Second negative control — timing:** conditions diagnosed pre-menarche (type 1 diabetes,
congenital conditions, childhood asthma). If endometriosis "predicts" these, ascertainment is
contaminating the pipeline.

**Positive control:** endometriosis → clear-cell and endometrioid ovarian cancer. Established and
mechanistically direct. If the pipeline does not recover it at approximately the known magnitude,
the pipeline is broken and nothing else it produces counts.

### 4.4 The comparison that actually decides it

The dominant rival hypothesis is not reverse causation. It is:

> Chronic pain, systemic inflammation, opioid exposure, sleep disruption and healthcare-system
> attrition drive the comorbidities — and endometriosis is one cause of chronic pain among many.

Under that model endometriosis is a **marker**, not a driver, and treating lesions specifically
would not outperform treating the pain.

Test it with a **negative control exposure**: a pain-matched non-endometriosis comparator —
chronic low back pain, or primary dysmenorrhoea without endometriosis — matched on pain-years,
opioid MME and healthcare contact frequency.

- Endometriosis hazard **exceeds** the pain-matched comparator → something endometriosis-specific.
- It does **not** → endometriosis is a marker, and the "treat endo, prevent downstream disease"
  thesis fails at the first hurdle.

This single comparison is worth more than the entire temporal analysis. It should be run first.

---

## 5. The intervention arm — the strongest idea, and its traps

Testing whether effective treatment bends the comorbidity curve is the genuinely powerful design,
because it is the only one that speaks directly to modifiability.

**Structure it as a target trial emulation:** excision vs ablation vs medical management vs
watchful waiting, new-user active-comparator.

Traps that will otherwise sink it:

1. **Immortal time bias.** Surgery is a time-varying exposure — a woman must remain
   comorbidity-free long enough to receive it. This alone can manufacture the entire effect.
   Handle with clone-censor-weight or landmark analysis. Non-negotiable.
2. **Confounding by indication.** Who receives excision? Severe disease, better insurance,
   specialist centres, higher health literacy — all independently predictive of the outcomes.
3. **Follow-up truncation.** US claims data has median continuous enrolment of ~2–3 years, which
   is fatal for long-horizon outcomes like CVD and dementia.

**The instrument worth pursuing:** surgeon or centre preference for excision vs ablation, or
distance to a specialist endometriosis centre. Preference-based IV is standard in
pharmacoepidemiology and would be novel applied here.

**Bridge to the existing KP work in this repository:** endometriosis is a leading indication for
premenopausal oophorectomy, and the CDST already carries the Rocca estimates (HR 1.46 for
bilateral oophorectomy; aOR 2.21 for MCI). Endometriosis → oophorectomy → accelerated oestrogen
withdrawal → KP dysregulation → cognitive outcome is a *mechanistically specified, surgically
dated* causal chain. Unlike the diffuse comorbidity claim, every link is separately evidenced and
the exposure has an unambiguous date in the record. **This is the strongest version of the
upstream argument available, and it connects the endometriosis work to the menopause model.**

---

## 6. Data sources, ranked for this question

1. **Finnish national registers (+ FinnGen).** Lifetime linkage via personal identity code,
   hospital discharge register back to 1968, surgical verification, sibling linkage — and FinnGen
   supplies genotypes in the same population, so the temporal analysis and the MR can run in one
   cohort. Given the Finland COI work and Eir Accelerator relationships already in train, this is
   the obvious play.
2. **Danish / Swedish registers.** Same virtues; sibling and discordant-twin designs control
   shared genetics and childhood environment.
3. **UK CPRD Aurum.** Best available capture of the *pre-diagnostic* symptom journey (primary
   care from ~1995), linked to HES. Required if the symptom-onset anchoring in §4.1 is used.
4. **UK Biobank / All of Us.** Genotype plus linked records; good for MR, thinner longitudinal
   depth, volunteer selection.
5. **US claims (Merative, Optum, TriNetX).** Fast and large. Enrolment churn truncates follow-up.
   Adequate for description, weak for the causal claim.

---

## 7. Recommended sequence

1. **Negative control exposure first** (§4.4). Pain-matched comparator. Cheapest, and it is the
   load-bearing test — if endometriosis does not beat a pain-matched control, the rest is moot.
2. **Bias calibration** (§4.3). Establish the ADHD noise floor before interpreting any hazard.
3. **Oophorectomy chain** (§5). Dated exposure, specified mechanism, connects to the KP model.
4. **Temporal precedence with delay-shift sensitivity** (§4.1–4.2) — as supporting evidence only,
   never as the headline.
5. **Target trial emulation of treatment** (§5) — the highest-value output, and the one that needs
   the register data and the preference IV.

Frame findings as *"endometriosis is the earliest treatable node in a shared liability"* rather
than *"endometriosis causes the comorbidities."* The first is defensible now. The second will not
survive a reviewer who knows the MR literature.

---

## 8. Open items

- **Citation not verified.** The "2022 PLOS" comorbidity paper could not be located. Closest
  matches to the description: Surrey et al. 2018 (*J Womens Health*, the 22-comorbidity matched
  cohort) for the effect sizes, or Rossi/Uimari et al. 2022 (*Fertil Steril*, NFBC1966) for the
  2022 date plus temporal analysis. Needs confirming against the source.
- **HERA white paper** not located in Drive or this repository; content not incorporated.
- **Local comorbidity working files** are not in this repository. The NZ cost-of-illness paper
  (Tewhaiti-Smith, Gannott et al. 2025) is cited here only for the Stromberg multipliers and does
  not carry a comorbidity analysis.
