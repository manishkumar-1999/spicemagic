# Off-topic (tier 0) decision rules — DRAFT, needs your ratification

Off-topic causes **12 of the 25 remaining test errors (48%)**, and 9 of those 12
get promoted to a *human* tier — the single most damaging error this project can
make, because it inflates the human-evidence denominator.

It also has **291 training examples (the most of any class) and the worst F1
(0.737)**, while `rct` reaches perfect recall from 85. That rules out "not enough
labels" as the cause. The cause is that tier 0 is the only class defined by a
convention rather than by a study design — so the convention has to be written
down before more annotation can help.

Below is the convention I actually applied to the 400 round-2 papers, reverse-
engineered from my own labels. **Mark each rule keep / change.** Where we already
disagreed during calibration, I adopted your call — those are flagged.

---

## The rules I applied

**R1 — No plant, herb, spice or food at all → tier 0.**
Pure clinical or biomedical papers the association-miner caught by accident.
> #5 Guidance prostheses for partial mandibulectomy · #30 Remote ADHD monitoring
> protocol · #397 Narrow complex tachycardia ECG

**R2 — The plant is the crop, not the medicine → tier 0.**
Agronomy, soil science, crop pathology, pest management, yield, storage, ripening.
> #20 Fennel aphids and their predators · #163 Film mulching and garlic yield
> · #328 Trichoderma biocontrol in maize

**R3 — Measuring the food, not its effect → tier 0.**
Analytical chemistry, adulteration detection, residues, packaging, sensors.
> #173 Argemone adulteration in mustard oil · #289 Honey adulteration by HS-SPME
> · #244 Acephate residues on mango

**R4 — Making the compound, not testing it → tier 0.**
Bioproduction, metabolic engineering, synthesis with no bioactivity assay.
> #50 Glycyrrhetinic acid in engineered yeast · #174 Artemisinic acid in yeast
> · #393 Sanguinarine from Papaver cell cultures

**R5 — The plant word is a false match → tier 0.**
> #275 Giant "onion stone" (a stone shape) · #109 mWasabi (a fluorescent protein)
> · #44 PAPRIKA-HF (a trial acronym) · #143 sulfur mustard (a warfare agent)

**R6 — Editorial notices → tier 0.**
Corrections, retractions, errata, comments, news items.
> #240 Retraction notice · #105 Correction · #150 Ig Nobel news

---

## The two boundary rules — these are the ones that matter

These came from our calibration disagreements (4 of 7 were exactly here).
**I adopted your convention; confirm it is what you want.**

**R7 — A plant-extract bioactivity assay inside a non-health paper → tier 5
(in-vitro), NOT tier 0.**
Your call on the solar-dryer paper that ran an antioxidant assay.
> Applied to: #2 antifungals on grapevine wounds · #78 essential oils on stone
> fruit rot · #94 neem pesticide on sunflower mildew

*Risk:* this pulls a large number of agricultural papers into tier 5, where they
sit alongside genuine pharmacological cell work. If you would rather these were
tier 0, say so — it changes roughly 20–30 of my 400 labels.

**R8 — A human diet/nutrition study with no named spice → tier 3
(human-observational), NOT tier 0.**
Your call on the Myanmar infant-feeding survey.
> Applied to: #60 ultraprocessed food and cognition · #250 fussy eating cohort
> · #354 dietary patterns and childhood overweight

*Risk:* this is the rule most likely behind the 9 off-topic→human-tier errors,
because it makes tier 0 and tier 3 overlap in exactly the region the model has
to separate. Tightening it to "must name a specific spice or herb" would sharpen
the boundary — but it contradicts your calibration call, so it is your decision.

---

## Cases I was NOT consistent about — please rule

| # | case | I chose | alternative |
|---|---|---|---|
| a | Human **allergy** study where the allergen is a plant food (#4, #81, #169, #312) | tier 3 | tier 0 |
| b | Plant used as a **test organism** for toxicology (Allium test, #206, #326, #342) | mixed 5 / 0 | pick one |
| c | **Ethnobotanical surveys** of traditional use (#80, #148, #389) | tier 3 | tier 6 or 0 |
| d | Plant lectin/enzyme used as a **lab reagent** (#307, #380) | tier 0 | tier 5 |
| e | **Veterinary** case reports and outbreaks (#146, #302, #374) | tier 4 / 0 | pick one |

Case (b) is the clearest inconsistency in my 400: I labelled #206 and #326 as
in-vitro but #342 as off-topic, on a distinction (is the plant the subject or the
instrument?) that I never stated. Whichever way you rule, it should be uniform.

---

## What happens once you ratify

1. I re-check my 400 round-2 labels against the ratified rules and correct the
   ones that disagree — expect 20–40 changes, concentrated in R7/R8 and case (b).
2. The same rules go into the round-3 sampler as the annotation guideline, so
   the next batch is consistent from the start.
3. Round 3 is **class-targeted**: papers near the tier-0 boundary specifically,
   not the lowest-margin papers anywhere. Round 2 used margin sampling, which is
   why `rct` got 4 new examples and `synthesis` got 1.
