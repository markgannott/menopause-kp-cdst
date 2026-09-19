# Methods

## The annual cost of illness of fifty women's health conditions in Finland

**Model:** `Finland_WH_COI_REBUILT_AgeBanded (Gannott 2026).xlsx` — frozen central estimate
€15,877,300,767.24
**Reference year (quantities):** 2024 · **Price level (euros):** 2026
**Author:** Mark Gannott (Suncoast Ventures) · **Editor:** Anastasiya Markvarde (Eir Accelerator)

---

## 1. Study design and perspective

We conducted a prevalence-based, bottom-up cost-of-illness (COI) analysis of fifty
conditions that occur exclusively, predominantly or differentially in women, estimating
the recurring annual economic burden borne by Finland.

The analysis takes a **societal perspective** and applies the **human-capital
(opportunity-cost) approach** to productivity valuation: time lost from paid work is
valued at the gross wage the woman would otherwise have earned. The friction-cost method
is not used, and no adjustment is made for labour-market slack or worker replacement.
This choice is deliberate and is the more expansive of the two conventions; it is
declared here because it is the single largest methodological determinant of the
indirect-cost layer.

The estimating engine is an adaptation of the published New Zealand endometriosis and
chronic-pelvic-pain COI engine (Tewhaiti-Smith, Gannott et al., *Women* [MDPI]
2025;5(4):47), extended here from a single-condition survey instrument to a fifty-condition,
register-anchored, age-banded national account.

The analysis is **bottom-up and condition-level**: each of the fifty conditions is costed
independently from its own population denominator, its own unit costs and its own
productivity inputs, and the national total is the sum of the fifty rows. No component of
the estimate is derived by apportioning a global or supranational figure to Finland by
population share. This is the principal contrast with the previously circulating Finnish
women's-health figures of €4.76 billion and €3.9 billion, both of which are the McKinsey
Health Institute global \$1 trillion estimate scaled by Finland's share of world
population.

### 1.1 Cost boundary

The headline estimate comprises four cost accounts:

| Account | Symbol | Definition | 2026 €M |
|---|---|---|---|
| Direct health costs | `D` | Health-sector resource use attributable to the condition | 8,096.39 |
| Absenteeism | `A` | Paid work time lost to the condition through absence | 2,450.89 |
| Presenteeism | `P` | Paid work time lost while the woman remains at work | 3,561.88 |
| Employer-borne productivity | `W` | Production loss accruing to the employer beyond the wage | 1,768.14 |
| **Indirect subtotal** | `A+P+W` | | **7,780.91** |
| **Annual COI** | `D+A+P+W` | | **15,877.30** |

Three categories are **computed but deliberately excluded from the headline**, because
they are not components of the same annual bill (Section 8): the diagnostic-delay layer
(€28,925.8M, an accumulated rather than annual burden), the informal-care layer
(€2,299.45M, unpaid rather than market production), and premature mortality.

**No cost of premature mortality is included anywhere in the model.** There is no
years-of-life-lost, value-of-statistical-life or future-earnings-stream component. The
estimate is therefore a floor with respect to fatal conditions, and the cancer cluster in
particular (Section 3.3) is compressed by this exclusion.

Out-of-pocket patient expenditure, informal-sector care purchases, travel and carer
absenteeism are likewise excluded.

---

## 2. Currency, price level and vintage alignment

Two distinct conventions are applied and kept separate, so that a quantity and a price
are never conflated.

**Quantity vintage.** Every count — prevalence, incidence, register day-count, employment
rate — is fixed to the reference year **2024**, the latest complete Finnish register year
at the time of the freeze. Where 2024 was not published, the closest published year is
used and flagged (Finnish Cancer Registry 2023). Demographic inputs that had drifted to
2025 during development were reverted to exact 2024 values in a dedicated vintage pass.

**Price level.** Every euro is then restated to the **2026** price level using published
indices, printed in the model rather than retrieved at run time, so that the headline
cannot move because an API happened to respond on a given afternoon:

| Restates | Index | Factor (2024 → 2026) | Status |
|---|---|---|---|
| Wage-based components (`A`, `P`, `W`) | StatFin index of wage and salary earnings | 1.051282 | 2026Q1, preliminary — flagged |
| Price-based components (27 direct unit-cost inputs) | Eurostat HICP Finland (`prc_hicp_aind`), Jan–Jun 2026 mean 124.425 | 1.038346 | Year-to-date — flagged |
| Non-euro source inputs (2013 USD) | ECB reference rate, then HICP | 1.3281 USD/EUR (2013) | Exact |

Both 2026 indices are partial-year and are declared as such. Where a single source blends
wage and price components, HICP is applied to the whole; because HICP is the lower of the
two factors, this understates the productivity component, which is the conservative
direction. One direct input — stroke, €25,000 — is carried un-normalised pending
retrieval of its original-euro figure and is flagged in the model's price-normalisation
pass.

---

## 3. Case definition and population denominators

For each condition *i* the model carries a single **costed population** `N_i`: the number
of Finnish women to whom the condition's unit costs are applied in the reference year.
`N_i` is a costed count, not a prevalence estimate, and the two are not interchangeable.

Denominators are exact StatFin 2024 populations. The principal ones are women aged 15–49
= 1,199,617 and women aged 45–65 = 721,153. The per-woman figure reported alongside the
headline (€5,580.38) uses the total Finnish female population, ≈2.845 million.

`N_i` is constructed on one of four explicitly declared bases, and the basis is recorded
per row:

### 3.1 Register-counted populations

Where a Finnish register publishes the count directly, it is used unmodified. Examples:
type 2 diabetes `N` = 183,119 women (Kela special reimbursement entitlement 103, exact
2024); cardiovascular disease `N` = 54,944 (coronary women); multiple sclerosis `N` =
9,371 (national MS register 12,494 × ≈75% female).

Register counts inherit the register's own case definition, and where that definition is
broader than the condition being named, this is recorded as a white-space flag rather
than silently corrected. The diabetes row is the clearest instance: Kela entitlement 103
includes type 1 diabetes, and no sex-split type-2-only extract exists.

### 3.2 Prevalence × denominator

Where no register count exists, `N_i` = published prevalence rate × the exact StatFin
denominator for the age range the rate was measured in. Prevalence enters as a **band with
a midpoint central estimate**, not as a point estimate, and the model's own documentation
declares these as directional approximations rather than actuarial predictions.

Prevalence-type discipline is enforced explicitly: each rate is identified as point,
12-month or lifetime, and only point or 12-month rates are applied as current cases. This
correction alone moved depression from 511,200 to 173,000 costed women, anxiety from
624,800 to 211,000, and eating disorders from 107,400 to 47,000, when lifetime rates that
had been applied as current caseloads were replaced.

Sex-disaggregation is enforced on the same principle: a both-sex denominator is never
applied to a women-only row.

### 3.3 Incidence basis (cancer)

All five cancer rows are costed on **annual incidence**, not prevalence: the model counts
women newly diagnosed in the reference year and applies first-year-of-treatment costs
(breast 5,173; endometrial 916; ovarian 550; thyroid 423; cervical 193).

This is internally consistent — an incident count is matched to an incident-year unit
cost — but it makes the cancer cluster **not comparable** with the prevalence-based
clusters, and the direction is unambiguous: ongoing treatment, survivorship, recurrence
and (as in every row) premature mortality fall outside the boundary. The resulting cancer
total of €286.7M, 1.81% of the annual bill, is a first-year-incidence figure and is
labelled as such wherever it is reported.

### 3.4 Modelled or constructed populations

Where neither a register count nor a transferable prevalence rate exists, `N_i` is
constructed from documented components and carries the lowest grade. Two examples define
the class:

- **Menopause and perimenopause** is costed not on the age-band denominator but on a
  narrow symptomatic tier: moderate-to-severe vasomotor symptoms at 12% × 721,153 women
  aged 45–65 = 86,711. The broad-population construction is documented as an upside
  scenario and is not used in the central estimate.
- **Uterine fibroids** has no population-representative Finnish prevalence figure at all.
  `N` = 43,941 is built from the Zimmermann 2012 diagnosed-prevalence gradient × StatFin
  2024 populations × a 53.7% symptom-impact share, corroborated against a FinnGen lifetime
  figure of ≈14.6%. It is recorded as a white space, not as a measurement.

---

## 4. Direct costs

For each condition:

```
D_i  =  N_i  ×  c_i
```

where `c_i` is the annual direct health cost per affected woman in 2026 euros, and `N_i`
is the **full** costed population — direct costs are not conditioned on employment.

Unit costs `c_i` are drawn, in descending order of preference, from (a) Finnish register
or national cost studies, (b) Nordic studies restated to Finnish prices, (c) other
European studies restated to Finnish prices and population. Each is normalised from its
own source price year to 2026 euros by the HICP chain in Section 2.

Two disciplines are applied to the unit costs:

**Cost-label hygiene.** A unit cost is matched to the population basis it was measured on.
A first-year-of-treatment cost is applied only to an incident count; a steady-state
prevalent cost only to a prevalent count; a total (direct + indirect) cost is never
applied as a direct cost. The endometriosis row is the worked example: the previously
carried €6,800 was a *total* cost figure, which double-counted productivity loss that the
model computes separately in the indirect layer, and it was replaced by the EndoCost
direct-only figure (Simoens et al., *Hum Reprod* 2012;27(5):1292–99), corroborated
independently against a Finnish hospital-district figure of ≈€3,114 and normalised to
€3,916 per woman in 2026 euros.

**Vintage annotation.** Unit costs older than approximately ten years are annotated in the
row rather than silently uplifted; the HICP restatement corrects the price level but not
the change in clinical practice, and this residual is disclosed.

---

## 5. Indirect costs

The indirect layer is the methodological core of the model, and it differs from the direct
layer in one decisive respect: **indirect costs are scaled only to the employed share of
the affected population, resolved by age band.** A woman who is not in paid work
contributes no absenteeism and no presenteeism cost. This is what locates the burden peak
in the 35–44 band rather than in the oldest and largest prevalence pools.

### 5.1 Employment exposure

Each condition carries a six-band age distribution `d_i = (d_i1 … d_i6)` over the bands
15–24, 25–34, 35–44, 45–54, 55–64 and 65+, with `Σ_b d_ib = 1`. Each band carries the
Finnish female employment rate `e_b` for that band (StatFin Labour Force Survey, exact
female 2024 rates):

| Band | 15–24 | 25–34 | 35–44 | 45–54 | 55–64 | 65+ |
|---|---|---|---|---|---|---|
| Female employment rate `e_b` | 0.4429 | 0.7501 | 0.8170 | 0.8429 | 0.7301 | 0.0537 |

The condition's **employment-exposure factor** is the distribution-weighted employment
rate:

```
φ_i  =  Σ_b  d_ib · e_b
```

and the **employed affected population** — the base for every indirect component — is

```
E_i  =  N_i · φ_i
```

Worked example. Menopause carries `d = (0, 0, 0, 0.15, 0.65, 0.20)`, giving
`φ = 0.15(0.8429) + 0.65(0.7301) + 0.20(0.0537) = 0.6117`, so 86,711 costed women become
53,041 employed affected women.

Each of the fifty age distributions is a literature or age-gradient input, not a register
extract, and every one is flagged in the model as replaceable by HILMO diagnosis-date
distributions once register access is granted.

### 5.2 Wage base

| Parameter | Value | Source |
|---|---|---|
| Female full-time median monthly earnings, 2024 | €3,376 | StatFin Structure of Earnings |
| Annual earnings | €40,512 | derived |
| Working days per year, `Δ` | 249 | assumption — flagged |
| Daily wage, 2024 | €162.70 | derived |
| **Daily wage, 2026 EUR, `w`** | **€171.04** | × 1.051282 earnings index |

The 249-day figure is an assumption and is declared as one: Finland publishes no official
standard, and the 2025 calendar yields 251 weekdays net of holidays and de facto eves. The
parameter is retained at 249 pending resolution. It scales every day-valued euro
proportionally and is therefore the quietest high-leverage parameter in the model.

### 5.3 Absenteeism

Each condition carries `s_i`, condition-attributable absence days per affected employed
woman per year. Where possible `s_i` is derived as a **register floor**: Kela
sickness-allowance compensated days for women in the relevant ICD-10 chapter, allocated to
the condition by its share of that chapter. Depression, for example, takes Kela F30–F39
women's days (2,715,083) × a 67.2% recipient share = 1,825,278 days, distributed across
the costed population.

```
A_i  =  E_i · s_i · w · m_A · J_i
```

where `m_A = 1.97` is the Strömberg absenteeism multiplier and `J_i` the overlap factor
(Section 6).

### 5.4 Presenteeism

Each condition carries `p_i`, the proportion of working time lost to impaired performance
while at work, applied across the full working year:

```
P_i  =  E_i · p_i · Δ · w · m_P · J_i
```

with `m_P = 1.54`, the Strömberg presenteeism multiplier.

`p_i` is admitted **only** where a published day-equivalent or impairment percentage
exists for that condition. Finland maintains no register of presenteeism — a woman can be
present at work, fully employed in official statistics, and materially impaired, and no
administrative record captures it — so where the literature supplies no figure, `p_i` is
held at **zero** rather than imputed (Section 10). Thirty-eight of the fifty rows carry
presenteeism at zero on this rule.

### 5.5 Employer-borne productivity (WEP)

The third Strömberg parameter, work-environment-related productivity loss, is **not**
applied to the wage base. It is applied only to the *employer-borne increment* — the
production loss over and above the wage that the absenteeism and presenteeism multipliers
themselves identify:

```
W_i  =  m_W · [ (m_A − 1)·E_i·s_i·w  +  (m_P − 1)·E_i·p_i·Δ·w ] · J_i
```

with `m_W = 0.72`. Equivalently, `W_i = 0.72 · [ 0.97·A_i^wage + 0.54·P_i^wage ]`, where
the superscript denotes the unmultiplied wage value.

This construction has a property worth stating explicitly, because it is what makes the
sensitivity analysis in Section 11 interpretable: **the WEP layer is entirely
multiplier-derived and collapses to zero when the Strömberg multipliers are switched
off.** It is not an independent employer-overhead assumption. An earlier build of the
model did carry a free-standing employer-overhead input in euros per patient per year;
that construction was retired in favour of this one.

### 5.6 Full estimating equation

For condition *i*:

```
Total_i = N_i·c_i
        + J_i · E_i · w · [ s_i·m_A  +  p_i·Δ·m_P
                            + m_W·( s_i·(m_A−1) + p_i·Δ·(m_P−1) ) ]

    where  E_i = N_i · Σ_b d_ib·e_b
```

and the national estimate is `Σ_i Total_i` over the fifty conditions.

### 5.7 The Strömberg multipliers

| Multiplier | Value | Applied to |
|---|---|---|
| Absenteeism `m_A` | 1.97 | Wage value of absence days |
| Presenteeism `m_P` | 1.54 | Wage value of impaired working time |
| Work-environment `m_W` | 0.72 | Employer-borne increment only |

Source: Strömberg C, Aboagye E, Hagberg J, Bergström G, Lohela-Karlsson M. "Estimating the
Effect and Economic Impact of Absenteeism, Presenteeism, and Work Environment-Related
Problems on Reductions in Productivity from a Managerial Perspective." *Value in Health*
2017;20(8):1058–64.

Two limitations of this transfer are declared rather than defended. First, the multipliers
were elicited from **758 Swedish managers** describing a general workforce; they are not
condition-specific, not women-specific, and not Finnish. Their application to women's
health conditions is a methodological extension first made in Tewhaiti-Smith, Gannott et
al. (2025) and carried forward here. Second, because a multiplier above unity asserts that
lost work costs more than the wage it displaced, the entire increment above the pure
human-capital floor rests on that single transferred instrument. The model therefore
publishes the multiplier-off estimate as a first-class result rather than a footnote
(Section 11).

---

## 6. Overlap and double-counting

Women receive multiple, overlapping diagnoses that can describe the same underlying
illness and the same lost work. Summing fifty independently costed rows without correction
would count that work twice.

An interim **menstrual-cluster overlap factor** `J = 0.7935` is applied to five rows —
endometriosis, adenomyosis, heavy menstrual bleeding, dysmenorrhea and uterine fibroids —
and is applied to **indirect cost only**, on the reasoning that duplicated diagnoses
duplicate lost work more reliably than they duplicate health-service contacts.

The effect is material and is reported: without the dedup the national estimate would be
€16,546.6M rather than €15,877.3M, a reduction of €669.3M (4.0%).

No overlap correction is applied across the other forty-five conditions, and none is
applied at the person level. The full cross-condition overlap matrix is an open design
item, and the per-woman lifetime figures the model reports elsewhere are therefore
single-condition exposures, not additive personal totals. Because the uncorrected
directions of comorbidity (depression with endometriosis, anxiety with depression, type 2
diabetes with gestational diabetes) are overwhelmingly positive, the absence of a full
matrix biases the estimate **upward**, and this is the one identified bias in the model
that runs against conservatism.

*Documentation note: the model's published narrative describes the dedup as applying to
four conditions; the engine's own data array applies `J = 0.7935` to five, uterine fibroids
included, consistent with the pass in which fibroids was added as the fiftieth condition.
The arithmetic above follows the engine. The prose should be reconciled to it.*

---

## 7. Allocation by age band

Indirect cost is allocated across the six age bands in proportion to each band's share of
the condition's employed affected population:

```
Indirect_ib  =  Indirect_i ·  (d_ib · e_b) / φ_i
```

Direct costs are not age-allocated. Summing over conditions gives the age profile of the
€7,780.9M indirect layer:

| Band | 15–24 | 25–34 | 35–44 | 45–54 | 55–64 | 65+ |
|---|---|---|---|---|---|---|
| Indirect cost, €M | 625.3 | 2,077.8 | **2,293.7** | 1,799.4 | 963.6 | 21.1 |
| Share | 8.0% | 26.7% | 29.5% | 23.1% | 12.4% | 0.3% |

Women aged 25–44 carry **56.2%** of all indirect cost, and the burden peaks at 35–44 —
the years of highest employment, earnings, caregiving load and pension accumulation.

This profile is a **modelled allocation**, not a measurement. It is the direct product of
the fifty literature-derived age distributions, and every one is flagged
HILMO-replaceable. A conservation check in the engine enforces that the six bands sum
exactly to the indirect total.

---

## 8. Layers computed but held outside the headline

Three quantities are computed on the same engine and reported separately, because
combining them with an annual flow would be a category error. Keeping them out is a
deliberate refusal, and it is the specific respect in which this estimate is smaller than
it could have been made.

**Diagnostic delay — €28,925.8M.** For conditions with a documented time-to-diagnosis
`y_i`, the delay layer is `Indirect_i × y_i`: the productivity burden accumulated between
symptom onset and diagnosis, across the cohort currently within its delay window.
Endometriosis alone, at 8.5 years, carries the largest share. This is an **accumulated
stock**, not an annual flow, and cannot be added to the €15.877B.

**Informal care — €2,299.45M.** The value of unpaid care, overwhelmingly supplied by
women, valued by the Displaced Labor method (Gannott 2026) at a blended opportunity-cost
wage of €21.80/hour (Statistics Finland Structure of Earnings 2024; blended 67/33 female
€20.32 / male €23.40, uprated to 2025 euros). This is non-market production and is not
part of a market-cost account. The layer was rebuilt on 1 September 2026 from a previously
published €1,776.97M; the single largest correction was cardiovascular informal care,
carried at €40M against a Finnish national figure of €937M (Luengo-Fernandez et al., *Eur
Heart J* 2023) and corrected to €450M at a 48% women's share. Two method items remain
open: cardiovascular disease and stroke are top-down national shares while every other row
is bottom-up, and that inconsistency is disclosed rather than smoothed.

**Premature mortality — not computed.** See Section 1.1.

---

## 9. Evidence grading

Every condition carries a source grade, and the grade is reported wherever the condition
is reported. An unbadged number is read as sourced; that is why every row carries a letter.

| Grade | Definition | Rows | Share of € |
|---|---|---|---|
| **A** | Register extract or official statistic — Kela, HILMO, THL, Finnish Cancer Registry, StatFin. Checkable by anyone with the same access. | 6 | 12.5% |
| **B** | Peer-reviewed literature, Finnish or Nordic where it exists, or a register-adjacent cohort (FinnGen, FINRISK). Attributable, not a primary record. | 21 | 57.0% |
| **C** | Transferred or derived — non-Nordic literature restated to Finnish prices and population, or arithmetic on graded inputs. Directional. | 21 | 30.4% |
| **D** | Weak, older than ten years, or contested. Carried only where nothing better exists. | 2 | 0.1% |

**69.5% of the central estimate is graded A or B.** The share is weighted by euro, not by
row count, because what matters is not how many conditions sit in a grade but how much of
the total rests on it.

Twenty-one rows carry a red flag — a cost component the engine is wired for that no source
can fill — and fifteen carry a white-space flag, meaning no figure exists in the
literature at all.

---

## 10. The zero-input rule

Where no published source supports a cost component, that component is **held at zero**
rather than imputed, assumed or carried forward from an analogous condition.

The consequence is stated plainly because it determines how the headline should be read:

- **38 of 50** rows carry presenteeism at zero.
- **16 of 50** rows carry no indirect cost at all.
- **0 of 50** rows carry zero total cost.

In most of these cases zero identifies an evidence gap, not an absence of economic burden.
The estimate is therefore a **floor**: every objection a reader can raise about a
zero-valued component moves the total up, not down. An illustrative ceiling exists — the
engine's tornado sheet fills 37 of the 38 zero-presenteeism rows at a Schoep-half analog
for a further €1,590.2M — and it is never used as a central value.

*Reconciliation note: 38 and 16 are the counts computed directly from the engine's
fifty-row array and are the correct figures. The published site table has separately
displayed 18 and 20 for what should be the same tier; those are erroneous and should be
corrected to 38 and 16.*

---

## 11. Sensitivity analysis

One-way sensitivity is reported around the €15,877.3M central estimate. The ranges are the
model's own scenario bounds — they are **not** confidence intervals, and no probabilistic
or Monte Carlo analysis was performed. Each bound declares its own provenance.

| Parameter varied | Low €M | High €M | Swing €M | Basis |
|---|---|---|---|---|
| Strömberg multipliers off (pure human-capital floor) | **11,653.4** | 15,877.3 | 4,223.9 | Sourced — multipliers on vs. off; register day-counts identical either side |
| Employment exposure (×0.85 / toward full affected population) | 14,710.2 | 17,250.4 | 2,540.2 | Judgment — model's own scenario bound |
| Red-flagged presenteeism filled (illustrative ceiling) | 15,877.3 | 17,467.5 | 1,590.2 | Judgment — 37 zero rows at a Schoep-half analog |
| Wage ±10% | 15,099.2 | 16,655.4 | 1,556.2 | Judgment — symmetric band on the StatFin median, flagged as free-hand |
| Menstrual dedup removed (`J` → 1) | 15,877.3 | 16,546.6 | 669.3 | Model — the engine's own dedup |

Two results carry most of the interpretive weight.

**The human-capital floor.** Switching the Strömberg multipliers off — valuing lost time at
the wage and nothing more — gives **€11,653.4M**, of which €8,096.4M is direct and
€3,556.9M indirect. The WEP layer vanishes entirely, by construction (Section 5.5). The
€4.2B difference between floor and central is the entire contribution of one transferred
Swedish managerial instrument, and it is the largest single source of uncertainty in the
study. Readers who do not accept the multiplier transfer should read €11.65B.

**Employment exposure** is the second-largest lever, which follows directly from the
decision to scale indirect cost to employed women only.

---

## 12. Validation and audit

The model was subjected to a documented hostile audit on 11 August 2026 under a single
instruction: do not keep a wrong number. The pre-audit central estimate of **€93,415.2M**
did not survive it. Two failures were disqualifying:

1. **55.1%** of the pre-audit total rested on fifty unsourced presenteeism inputs, which
   the zero-input rule (Section 10) subsequently eliminated.
2. The model implied **37.8 million** absence days against a Kela national ceiling of
   **8.30 million** compensated sick days for all Finnish women in 2024 — an internal
   impossibility.

Nine numbered passes followed, each recorded in the engine's change log before any
published artefact was touched, comprising **237 cell-level edits**: register/literature
rebuild; price normalisation to a single price level; vintage alignment to a single
reference year; pre-freeze structural locks (including repair of an off-by-one column
reference that had caused every age band to inherit the adjacent band's employment rate);
addition of uterine fibroids as the fiftieth condition; and application of the eight-cluster
taxonomy.

**Register-ceiling validation.** The published model implies **7,273,509** absence
day-equivalents against the Kela 2024 ceiling of 8.30 million compensated sick days for
Finnish women — the model sits inside the register, and is constrained to. Presenteeism
adds **13,522,317** day-equivalents, for which no register exists. The total,
**20,796,099** day-equivalents, is 83,518 full-time-equivalent workers at 249 days — 3.21%
of Finland's 2,602,000 employed, or one worker in every 31.

*Reconciliation note: an intermediate pass records the absence layer at 6.25M
day-equivalents; the frozen central implies 7.27M following the later passes. Both sit
below the register ceiling, but the pass-2 annotation should be updated to the frozen
figure.*

**Arithmetic validation.** The published fifty-row array reproduces the engine's cached
totals to the cent, and every equation in Section 5 has been independently recomputed from
the published per-condition inputs (`N_i`, `φ_i`, `s_i`, `p_i`, `c_i`, `J_i`) against the
published per-condition outputs, with agreement to rounding on all fifty rows and on all
five national aggregates.

---

## 13. Limitations

1. **The productivity multipliers are transferred.** They come from 758 Swedish managers
   describing a general workforce, not from Finnish women with these conditions. They
   determine €4.2B of the €15.9B (Section 11).
2. **Presenteeism has no register anywhere in the design.** Thirty-eight rows carry it at
   zero and one row in the dominant cluster carries it from a Dutch survey instrument.
3. **Age distributions are modelled, not measured.** All fifty are literature or
   age-gradient inputs, and the age profile in Section 7 is only as good as they are.
4. **The overlap correction is partial.** Five rows in one cluster, indirect cost only. The
   remaining comorbidity is uncorrected and biases the total upward — the single
   non-conservative element of the design.
5. **Unit-cost transfers dominate the direct layer.** Grade C carries 30.4% of the total,
   comprising non-Nordic literature restated to Finnish prices.
6. **No premature mortality, no out-of-pocket costs, no carer absenteeism** (Sections 1.1,
   8).
7. **Working days per year (249) is an assumption**, and it scales every day-valued euro.
8. **The 2026 price indices are partial-year** and will need restating when full-year
   figures publish.
9. **Prevalence enters as banded midpoints.** These are directional approximations, not
   actuarial predictions, and the model says so in its own documentation.
10. **Direct cost stands proxy for current condition spend** in the funding comparison,
    pending reconciliation against THL disease-based expenditure.

The design of the model is asymmetric with respect to these limitations. Items 2, 6 and 10
all push the true figure up; only item 4 pushes it down. **€15.877 billion should be read
as a floor.**

---

## 14. Reproducibility and register replacement

The estimating engine is a fully formula-driven workbook — every derived value is a live
formula, no derived value is hardcoded, and every parameter in Sections 2, 5.2 and 5.7 is
an editable input on a single assumptions sheet. Published artefacts are regenerated from
a single extracted data file rather than transcribed, so a number that moves in the model
moves in every downstream artefact on the next build.

Four register operations would convert the largest modelled components into measured ones,
and each is a query Finland can put to data it already holds:

1. **Split Kela sickness allowance by sex and ICD-10.** The Kelasto cube publishes days by
   diagnosis chapter but not by sex within chapter; this single publication change would
   replace the model's largest allocation step with a measured line.
2. **Give endometriosis a code Kela can see.** The largest condition in the model, at
   €1,831M, is invisible to the register that pays for it; pre-diagnosis leave is coded to
   other chapters.
3. **Repair the gestational-diabetes register series**, which feeds the type 2 diabetes
   line downstream.
4. **Pull HILMO diagnosis dates and ETK disability pensions**, replacing all fifty modelled
   age distributions and confirming the pension-exit signal in the 55–64 band.

---

## Appendix A — Parameter summary

| Parameter | Symbol | Value | Source / status |
|---|---|---|---|
| Conditions costed | — | 50 | 8 clusters (12/8/8/6/5/5/3/3) |
| Reference year (quantities) | — | 2024 | Latest complete register year |
| Price level | — | 2026 EUR | HICP / earnings index |
| Women 15–49 | — | 1,199,617 | StatFin 2024, exact |
| Women 45–65 | — | 721,153 | StatFin 2024, exact |
| Female median monthly earnings | — | €3,376 (€3,549 in 2026 EUR) | StatFin Structure of Earnings 2024 |
| Working days per year | `Δ` | 249 | **Assumption — flagged** |
| Daily wage, 2026 EUR | `w` | €171.04 | Derived |
| Female employment rate by band | `e_b` | .4429 / .7501 / .8170 / .8429 / .7301 / .0537 | StatFin LFS 2024, female, exact |
| Absenteeism multiplier | `m_A` | 1.97 | Strömberg 2017 |
| Presenteeism multiplier | `m_P` | 1.54 | Strömberg 2017 |
| Work-environment multiplier | `m_W` | 0.72 | Strömberg 2017, on increment only |
| Menstrual overlap factor | `J` | 0.7935 | Model, 5 rows, indirect only |
| Earnings index 2024→2026 | — | 1.051282 | StatFin, 2026Q1 preliminary |
| HICP Finland 2024→2026 | — | 1.038346 | Eurostat, Jan–Jun 2026 mean |
| USD/EUR 2013 | — | 1.3281 | ECB reference rate |
| Kela absence ceiling, women 2024 | — | 8.30M days | Validation constraint |

## Appendix B — Results summary

| Quantity | Value |
|---|---|
| Annual COI | €15,877.30M |
| Direct | €8,096.39M (51.0%) |
| Indirect | €7,780.91M (49.0%) |
| — Absenteeism | €2,450.89M |
| — Presenteeism | €3,561.88M |
| — Employer-borne | €1,768.14M |
| Indirect : direct ratio | €0.961 per €1 |
| Absence day-equivalents | 7,273,509 |
| Presenteeism day-equivalents | 13,522,317 |
| Total day-equivalents | 20,796,099 (83,518 FTE; 3.21% of employed) |
| Per Finnish woman | €5,580.38 |
| Human-capital floor (multipliers off) | €11,653.4M |
| Gross of menstrual dedup | €16,546.6M |
| Diagnostic-delay layer (separate) | €28,925.8M |
| Informal-care layer (separate) | €2,299.45M |
| Largest cluster — menstrual, pelvic, urogynecologic | €4,902.2M (30.9%) |
| Largest condition — endometriosis | €1,831.1M |
| Top 5 conditions | €6,381M (40.2%) |
| Top 10 conditions | €10,293M (64.8%) |
| Share graded A or B | 69.5% |

## Appendix C — Cluster totals

| Cluster | Rows | €M | Share |
|---|---|---|---|
| Menstrual, pelvic & urogynecologic | 12 | 4,902.2 | 30.9% |
| Mental health & neurodevelopmental | 6 | 3,560.3 | 22.4% |
| Neurologic & cognitive | 3 | 2,132.5 | 13.4% |
| Autoimmune & musculoskeletal | 8 | 2,077.8 | 13.1% |
| Cardiometabolic | 3 | 1,849.7 | 11.7% |
| Reproductive & endocrine | 5 | 790.4 | 5.0% |
| Cancer (incidence basis) | 5 | 286.7 | 1.8% |
| Pregnancy & maternal | 8 | 277.8 | 1.7% |
| **Total** | **50** | **15,877.3** | **100%** |

---

## References

Strömberg C, Aboagye E, Hagberg J, Bergström G, Lohela-Karlsson M. Estimating the Effect
and Economic Impact of Absenteeism, Presenteeism, and Work Environment-Related Problems on
Reductions in Productivity from a Managerial Perspective. *Value in Health*.
2017;20(8):1058–1064. doi:10.1016/j.jval.2017.05.008

Tewhaiti-Smith J, Gannott M, Semprini A, et al. The Cost of Endometriosis and Chronic
Pelvic Pain Burden in New Zealand (Aotearoa): Results from a Nationwide Survey. *Women*.
2025;5(4):47. doi:10.3390/women5040047

Simoens S, Dunselman G, Dirksen C, et al. The burden of endometriosis: costs and quality
of life of women with endometriosis and treated in referral centres. *Human Reproduction*.
2012;27(5):1292–1299. doi:10.1093/humrep/des073

Saavalainen L, Tikka T, But A, et al. Trends in the incidence rate, type and treatment of
surgically verified endometriosis — a nationwide cohort study. *Acta Obstetricia et
Gynecologica Scandinavica*. 2018;97(1):59–67. doi:10.1111/aogs.13244

Schoep ME, Adang EMM, Maas JWM, et al. Productivity loss due to menstruation-related
symptoms: a nationwide cross-sectional survey among 32,748 women. *BMJ Open*.
2019;9:e026186. doi:10.1136/bmjopen-2018-026186

Suvitie PA, Hallamaa MK, Matomäki JM, et al. Prevalence of Pain Symptoms Suggestive of
Endometriosis Among Finnish Adolescent Girls (TEENMAPS). *Journal of Pediatric and
Adolescent Gynecology*. 2016;29(2):97–103. doi:10.1016/j.jpag.2015.07.001

Taipale H, Lähteenvuo M, Tanskanen A, et al. Healthcare utilization, costs, and
productivity losses in treatment-resistant depression in Finland — a matched cohort study.
*BMC Psychiatry*. 2022;22:484. doi:10.1186/s12888-022-04115-7

Winkelmann A, Perrot S, Schaefer C, et al. Impact of fibromyalgia severity on health
economic costs. *Applied Health Economics and Health Policy*. 2011;9(2):125–136.

Luengo-Fernandez R, Walli-Attaei M, Gray A, et al. Economic burden of cardiovascular
diseases in the European Union. *European Heart Journal*. 2023.

Finnish Institute for Health and Welfare (THL). FinHealth 2017 Study.
urn.fi/URN:ISBN:978-952-343-105-8

Arffman M, Ilanne-Parikka P, Keskimäki I, et al. FinDM database on diabetes in Finland.
THL Discussion Paper 19/2020. urn.fi/URN:ISBN:978-952-343-492-9

Social Insurance Institution of Finland (Kela). Sickness Insurance Statistics 2024,
Tables 5–7 and 12–13.

Statistics Finland. Population Structure tables 11rc and 11rd; Labour Force Survey table
13aj; Structure of Earnings table 15aw.

Finnish Cancer Registry. Cancer in Finland 2023.

Eurostat. Harmonised Index of Consumer Prices, Finland (`prc_hicp_aind`).
