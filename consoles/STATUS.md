# Console work — overnight status, 19 September 2026

Six things were asked for. Two are done, one is researched and specced, two are
**blocked on source access**, and one is a finding that needs a ruling. Detail below,
worst news first.

---

## 1. Blocked — Liz / Svelte console (the HMB task)

**I could not do this one, and it is not a research gap — it is an access gap.**

The `svelte-console` Vercel project:

- has **no linked GitHub repository** (deployment metadata is empty — it was pushed
  from a local folder with the Vercel CLI, not from git);
- is **SSO-protected** on all except custom domains, so I cannot fetch the rendered
  page either;
- `live: false` on the project record, latest production deploy 19 Sep 2026.

So I can neither read what is on Liz's console today nor tell whether HMB is already
covered, let alone add anything. **What I need from you:** either push that folder to
GitHub, or tell me the local path and I'll work from it.

**What I did prepare.** I read the HMB & Pelvic Health Cluster doc you put in Drive at
11:02 this morning and pulled out what a Svelte/Liz console would need to carry. On your
own framing — *"Svelte fits the cluster, but Liz may fit even more than Svelte does"* —
the console should probably not be a Svelte product console at all. It should be a
**sourcing-and-integration console**: the physical-product layer across the pelvic-health
lifecycle, with Svelte as one node rather than the subject.

Proposed spine, straight from your doc:

| Layer | Console section | Node |
|---|---|---|
| Lifecycle | Postpartum → HMB/pelvic → perimenopause/GSM → continence | the acquisition logic ("lifetime familiarity with a trusted product category") |
| Pathway | recognise → quantify bleeding → detect anaemia → assess iron → intervene → treat → support → follow | where a product sits at each step |
| Sourcing | what Metluma and others need to integrate without becoming product companies | Liz's actual role |
| Distribution | institutional buyer behaviour, scale | Aunt Flow / Claire as peer |
| Evidence | what is measured vs asserted for each product claim | the grade discipline from the Finland build |

HMB is not "one more use case" on that structure — it is the middle of the pathway and
the entry point for the anaemia/iron layer (Sanguina hemoglobin, Ferrivia ferritin). If
Liz's current console treats HMB as a product use case rather than a pathway position, that
is the change worth making, and it is bigger than "add HMB".

---

## 2. Blocked — Next Life Sciences console

`next-engine` deploys from `markgannott/next-engine`, but **that GitHub repo is empty** —
the clone returns "you appear to have cloned an empty repository", while Vercel still
references commit `ee18f53`. Either the repo was emptied after the last deploy or the
live build is served from a source that is no longer in git. Same ask as above: point me
at the source.

**Material I do have** for when it is unblocked: the NEXT LIFE SCIENCES POAP (V2), and
your *NEXT Shots on Goal Executive Summary* (19 Aug, for Darlene Walley and Joe Bailey) —
which already has the structure a console wants: an existing endometriosis/fibroid
drug-delivery platform, a set of uterine-therapeutics "shots", and Mike Armour's costings
at A$150k per shot for the first study and A$250–350k for the next stage, with the
Australian R&D offset modelled. That is a portfolio console: shots on goal, cost per
shot, net of offset, value-inflection path per asset. It is close to the ART
stage-adjusted weighting you already built.

---

## 3. Done — the Endometriosis Console (v1)

`consoles/endo/index.html` — self-contained, Suncoast tokens, responsive, light/dark.

Condition-first and **deliberately company-agnostic**, since you haven't picked the
company. Nothing in it needs to change when you do; a company overlay drops on top.

What it carries, all independently recomputed rather than transcribed:

- **Three national counts side by side** — Finland €1,831.1M (endo alone, 119,129 costed
  women, grade B), Australia Int$6.50B (Armour et al., PLOS ONE 2019, endo + CPP),
  New Zealand NZ$22B+ (Tewhaiti-Smith, Gannott et al. 2025, endo + CPP). Shown as three
  findings, not one number — different scopes and currencies.
- **The agreement that matters**: productivity is 74.5% of the Finnish total, 75–84% of
  the Australian, 65–75% of the New Zealand. Three methods, three countries, same answer.
- **Finland decomposed** — presenteeism (€684.0M) is larger than all direct health
  spending on the condition (€466.5M), and it is the one account with no register
  anywhere.
- **The delay layer** — €14,618.4M, half the entire national delay figure, at €1,720M per
  year of delay.
- **The cluster** — ten pelvic/menstrual rows, €4,722.8M, endo 38.8% of it.
- **Age profile** — 80% of cases land before 45.
- **Evidence table** with grades and the white spaces, including the one that matters
  most: *Kela has no sickness-allowance line endometriosis can be seen under.*

Weakest input carrying the largest number: the 8.5-year diagnostic delay is grade C —
international literature, no Finnish time-to-diagnosis series. Flagged on the page.

---

## 4. Finding that needs a ruling — the delay layer is un-deduplicated

Found while computing the endo numbers, not previously flagged anywhere.

The annual figure applies the menstrual-cluster overlap factor `J = 0.7935`. **The delay
layer divides it back out.**

```
endometriosis delay = (indirect ÷ J) × years
                    = (1,364.7 ÷ 0.7935) × 8.5
                    = €14,618.4M          ✓ matches the engine exactly
```

Same for adenomyosis (€740.4M), dysmenorrhea (€1,052.6M), HMB (€641.1M) and fibroids —
and therefore for the **€28,925.8M national delay total**, which is the number in the
white paper.

This may well be deliberate: the delay window predates the overlapping diagnoses, so
arguably there is nothing yet to deduplicate. But it is not stated anywhere, and it means
the annual layer and the delay layer are not on the same basis while being quoted in the
same sentence. Worth a ruling before release, and it is adjacent to the delay-figure
reconciliation already open on your checklist (€28.9B vs the site's €17.1B "delay engine").

---

## 5. Done — NGS console review

`markgannott/nfl-ngs`, Next.js, 14 pages, 8 Python build scripts, ~57MB of pre-built JSON.

**What it is.** The same evidence architecture as the Finland build, pointed at football —
and the about page says so outright: *"the same evidence machinery we point at healthcare."*
Build scripts emit JSON to `public/data`, the app is a thin reader over it with a promise
cache in `lib/data.ts`, and there is a `/sources` page carrying "sources, grades and every
assumption". That grading discipline transferring to a side project is the strongest
signal in the repo.

**What is good.**
- Clean separation: Python builds facts, TypeScript renders them. Same discipline as
  `build_paper.py` regenerating the white paper from `chart_data.json`.
- A reusable component vocabulary — `StatTile`, `LeaderTable`, `Sparkline`,
  `TrajectoryChart`, `RangePlot`, `EraChart`, `WindowBar`, `Slicers`. **This is the most
  portable asset you have across all the consoles** and it is currently locked in the
  football repo.
- The page titles are genuinely good: "Is the AFC actually weak?", "Who has actually won
  anything", "Everything you need to hold a football conversation, for people who have
  never watched one". That is the same voice as "The bill arrives in pieces."

**What I'd raise.**
- `README.md` is still the stock `create-next-app` boilerplate. Every other repo of yours
  carries a real working-rules file; this one doesn't, and it is the only place the
  window/slicer conventions are documented at all.
- 57MB of JSON committed to git. Fine now, awkward later.
- No `CLAUDE.md` of substance — it just points at the Next.js agent rules.

**The actionable bit.** If the components library were extracted into a small shared
package, the Metluma, endo and NLS consoles would all start from a working chart
vocabulary instead of hand-rolled bars. I'd rather do that once than four times. Say the
word.

---

## 6. Researched and specced — Metluma

**Company.** Doctor-led collaborative care for perimenopause and menopause. Co-founded by
**Georgie Drury** (CEO), **Dr Nicole Avard** and **Jarrah Eddy**. Custom digital platform
integrating physical, behavioural and lifestyle care delivered by GPs, nurses and allied
health. Products: Press Pause seminars, a six-week menopause coaching programme, and a
24/7 community. Research partnerships with Australian universities and Digital Health CRC.
Georgie previously founded and scaled **Springday** (wellbeing measurement, 13 countries).

**The asset that makes a console possible.** Metluma has its own instrument and its own
cohort — the **UMA40 assessment**, and the **2026 Australian Menopause Experience Report**
built on **1,468 women**. Headline findings:

- **43%** triggered at least one clinical red flag at baseline — meaning referral to a
  doctor within 24–48 hours.
- **More than one in five** presented with *multiple* red flags, associated with a more
  complex symptom burden.
- High-risk symptom set includes heart palpitations, debilitating migraine and dizziness.
- The framing: women are already in clinically significant distress by the time they ask
  for help.

That is a real dataset with a real finding, which is what separates a console from a
brochure. **Note:** `metluma.com` and the Digital Health CRC site are both blocked by this
environment's network egress proxy, so the above is assembled from search results and your
Drive. Before anything is published, the UMA40 figures need confirming against Metluma's
own report — I would not put 43% on a page on this basis alone.

**Proposed console — "The Metluma Console: what 1,468 women were carrying before anyone
asked."**

| Section | Content | Status |
|---|---|---|
| The red-flag finding | 43% / 1-in-5 multiple, by flag category | needs source confirmation |
| Who arrives how | symptom clusters at baseline, by age band | needs cohort data |
| The cost of arriving late | UMA40 severity × the Finland/AU productivity engine — **this is the bridge only you can build**, because the Strömberg decomposition is already yours | buildable now |
| Care-pathway position | product-light orchestration; where the sourced products and the anaemia/iron layer attach | from your HMB doc |
| Research bridge | NICM — GSL, Joelle Metri, MenoStim; Metluma's completed Western study and appetite for another | from your HMB doc |
| Evidence grades | same A/B/C/D discipline | method already exists |

The third row is the one worth building. Metluma has severity at baseline; you have a
published engine that converts severity into productivity cost. Nobody else can put those
two together, and it turns "43% are at clinical risk" into a number a Finnish or Australian
payer has to respond to.

---

## What I need from you, in priority order

1. **Source for `svelte-console`** (Liz) — a repo or a path. Blocking the HMB task entirely.
2. **Source for `next-engine`** (Next Life Sciences) — the GitHub repo is empty.
3. **A ruling on the delay-layer dedup** (§4) before the white paper goes out.
4. **Metluma's UMA40 report**, or permission to cite the public summary as-is.
5. Whether to **extract the NGS component library** into something the other consoles share.
