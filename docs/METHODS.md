# Methods — ventilator QI composite score

Companion to [`../README.md`](../README.md). What each metric means, the one place this pipeline
deviates from upstream, how the composite and the per-patient exposure are formed, and the caveats
a reader should know before using the score.

This document carries no results. Every number the pipeline produces is written to
`output/<site>/final_no_phi/`, with its provenance block.

Definitions are those of [`AartikSarma/clif-dashboards`](https://github.com/AartikSarma/clif-dashboards)
at the commit pinned in `definitions/SOURCE`, read from `definitions/respiratory.yaml` (a verbatim,
hash-checked copy). Upstream code paths are cited as `src/clif_dash/...`.

---

## 1. Cohort

| Step | Rule | Source |
|---|---|---|
| Encounter linking | Hospitalizations < 6 h apart become one `encounter_block` | clifpy `stitch_encounters`; `platform.yaml` `stitch_hours` |
| Adults | `age_at_admission >= 18` | `platform.yaml` `age_min` |
| Ventilator timeline | clifpy respiratory-support waterfall: hourly scaffold at `HH:59:59`, device/mode repair, forward-fill within mode blocks | `respiratory.yaml` `imv_source` |
| IMV run | Contiguous rows with `device_category = imv` | `upstream/prep.py` `imv_runs` |
| IMV episode | New episode after >= 24 h off invasive ventilation | `respiratory.yaml` `episode_gap_hours` |
| ICU requirement | Episode must overlap an ADT stay with `location_category = icu` | `respiratory.yaml` `require_icu` |
| Outside intubation | Excluded when the first ADT location is not the ED **and** the first recorded device is IMV **and** it is the encounter's first episode | `respiratory.yaml` `exclude_outside_intubation` |
| Outliers | Out-of-range values nulled per clifpy `outlier_config.yaml`; `fio2_set` divided by 100 when its whole-table mean > 1 | `upstream/clif_io.py` |

Time attribution follows upstream: a record's period comes from its own timestamp in **site-local**
time, an episode's from its start. The only change is widening the grain from month to quarter
(`period` in the site profile).

---

## 2. Metric definitions

### 2.1 LTVV — low tidal volume ventilation

- **Denominator:** hourly-scaffold IMV hours inside the episode whose `mode_category` is
  volume-targeted (`assist control-volume control`, `pressure-regulated volume control`, `simv`,
  `volume support`), with a non-null `tidal_volume_set` **and** a known predicted body weight. An
  episode with no PBW contributes nothing (`missing_height: exclude`).
- **Numerator:** those hours with `tidal_volume_set / PBW <= 8` mL/kg (tolerance `1e-9`). Rates at
  6 and 7 mL/kg are emitted as `ltvv_rate_at_6` / `ltvv_rate_at_7` for sensitivity but are not part
  of any score.
- **PBW:** ARDSNet — `45.5 (female) or 50.0 (otherwise) + 0.91 x (height_cm - 152.4)`.
- **Height:** the **first** `vitals` `height_cm` between 120 and 220 cm in the encounter block.
- **Grain:** one IMV hour. Source: `upstream/metrics.py`, `pbw()`.

### 2.2 SAT — spontaneous awakening trial

- **Vent-day:** a site-local calendar day with **>= 6** hourly IMV scaffold rows inside the episode.
- **Denominator:** vent-days on a continuous sedative infusion (propofol, midazolam, lorazepam,
  dexmedetomidine, ketamine — opioid-only infusions do *not* qualify a day), with **no** paralytic
  infusion overlapping the day, and not under comfort care.
- **Numerator:** a documented SAT (`patient_assessments.assessment_category =
  sat_delivery_pass_fail`, pass or fail — the value is not inspected) **or** the EHR phenotype: all
  continuous sedatives at zero for >= 30 min while still on IMV, after having been on that day.
- **Grain:** one eligible vent-day. Source: `upstream/metrics.py`, `_sat_phenotype`.

### 2.3 SBT — spontaneous breathing trial

- **Denominator:** vent-days where at **some** IMV hour that day `fio2_set <= 0.5`, `peep_set <= 8`,
  no paralytic covers that instant, and the norepinephrine-equivalent dose is `<= 0.2` mcg/kg/min;
  and the day is not under comfort care. A missing FiO2 or PEEP cannot establish eligibility
  (`NaN <= x` is false).
- **Numerator:** a documented SBT (`sbt_delivery_pass_fail`) **or** the phenotype: pressure
  support/CPAP with `pressure_support_set <= 8` and `peep_set <= 5`, or `blow by`, sustained
  >= 30 min and beginning that day, judged on real (non-scaffold) rows.
- **Grain:** one eligible vent-day. Source: `upstream/metrics.py`, `_sbt_phenotype`.

### 2.4 Proning

- **Cohort (the denominator):** first IMV episode of an adult ICU encounter; not intubated at an
  outside hospital; no tracheostomy in the first 24 h; a first qualifying arterial blood gas
  (`P/F < 150` with `PEEP >= 5` and `FiO2 >= 0.6`) within 36 h of IMV start; confirmed either by a
  second qualifying ABG 12-24 h later or by proning within 24 h of the first; not proned before the
  first qualifying ABG; and not in a `procedural` location in the preceding 48 h.
- **Enrollment:** the earlier of the second qualifying ABG and the first prone session.
- **Numerator (`p12`):** first prone session starting <= 12 h after enrollment.
- **Prone session:** only `position_category = prone`; a session ends at the next documented
  position change, or at the end of the ventilation episode for the last row.
- **Grain:** one patient-encounter. Source: `upstream/proning.py`.

---

## 3. The one deviation: time-resolved unit attribution

Upstream attributes an entire IMV episode to the ICU where it **started**, labelled by
`location_type` (`upstream/prep.py` `attribute_unit`). For a score used as a model exposure that is
too coarse in two ways: a transferred patient's whole course is credited to the sending unit, and
physically distinct units sharing a `location_type` collapse together, so keying on the categorical
can merge two separate ICUs of the same type.

This pipeline instead credits each record to the ICU that delivered it:

| Metric | Attributed to | Rationale |
|---|---|---|
| `ltvv` | ICU covering the hour's **midpoint** (`t - 30 min`) | Scaffold rows are stamped at `HH:59:59` and stand for the hour they end; upstream's own `_in_segments` tests the same instant |
| `sat` / `sbt` | ICU covering the **earliest IMV scaffold hour** of that calendar day | SAT and SBT are morning, nursing-driven; the unit holding the trial opportunity is the one the patient enters the day in. Tie-free, so order-invariant |
| `prone_12h` | ICU covering the **enrollment** instant | Enrollment is when the proning decision becomes due |

ADT locations are mutually exclusive in time, so "the ICU covering instant *t*" is unambiguous; the
lookup is a backward as-of join with a half-open `[in_dttm, out_dttm)` test.

**Attribution only redistributes.** Stage 5 asserts that denominators summed over every unit cell
plus the unattributed remainder equal the cohort total, for all four metrics.

### Unattributed care

Records falling in no named ICU stay are excluded from the panel and reported in
`records/manifest.json`. This is expected: an episode need only *overlap* an ICU stay, so parts of
it occur on the ward, in the OR, or in the ED. `unattributed_placement` in that manifest breaks the
excluded records down by `location_category`, separating care delivered elsewhere from records with
no ADT coverage at all. Excluding them is deliberate — it is real care, but not care any ICU can be
scored for.

---

## 4. Scoring

**Grain.** The primary exposure is one score per ICU, pooled over every available period
(`unit_panel.csv`). A unit x period panel is retained for sensitivity (`unit_period_panel.csv`).
Pooling the window re-divides the SUMMED counts rather than averaging the periodic rates, so a
near-empty period cannot move the unit score — material wherever a unit comes online part-way
through the window. Coarsening also raises the proning denominator per cell by more than an order
of magnitude, which is what makes the four-metric score computable at all.

**Pooled rates.** `rate = sum(numerator) / sum(denominator)` over the cell — never a mean of
per-patient percentages. A unit's LTVV rate is hour-weighted and its SAT/SBT rates are
vent-day-weighted, matching upstream's aggregation.

**Composite.** A weighted mean of pooled rates times 100, with weights normalized within each
score. A score requires **every** one of its components to have a non-zero denominator in that
cell; otherwise it is null and counted. Renormalizing over whatever happened to be present would
make the score mean different things in different cells — unacceptable for a regression exposure.
`composite(..., require_all=False)` exists for sensitivity work.

**Thin cells are flagged, never dropped.** Dropping them would bias the panel toward high-volume
units. `<metric>_low` marks a cell below that metric's floor; `<score>_low` ORs only the floors of
the metrics that score actually uses. The global `low_denominator` ORs all of them and is therefore
the wrong filter for a score that excludes a rare metric, since the rare metric's floor will mark
most cells.

**No leave-one-out adjustment.** An earlier version subtracted each patient's own numerator and
denominator from their cell, to remove the mechanical component of an exposure-outcome
association. It was dropped once the primary exposure moved to unit grain: with denominators that
large, one patient's own care moves their own exposure by a negligible amount. The correction was a
quarter-grain concern.

**Exposure assignment.** A patient can contribute to more than one cell, so two artifacts are
emitted: a long `(patient, cell)` table, and a primary-cell assignment using the cell where the
patient accrued the most **vent-days** (the `sat` denominator — the grain SAT and SBT are measured
on).

---

## 5. Outcome model

`code/08_outcome_model.R`. One row per patient, their first qualifying IMV hospitalisation,
restricted to complete cases on outcome, exposure, SOFA, Charlson and age; the remaining
covariates are complete by construction, which stage 7 and stage 8 both assert.

**Model chain.** Three nested mixed-effects logistic models with a random intercept for the
attributed ICU: `m1` null, `m2` adding the primary adjustment, `m3` adding the unit score centred
on the mean of the **unit** values so its coefficient reads per 1-point increase.

**Primary adjustment.** Standardised age, SOFA, and six case-mix covariates anchored at the first
intubation. SOFA is classic SOFA from the CLIF-epidemiology-of-CRRT calculator, vendored verbatim
in `vendor/crrt_sofa/` at the commit its `SOURCE` file records, with the respiratory subscore
recomputed as stage 7 describes. The covariates are built in stage 7 (`common/covariates.py`):

| Covariate | Definition | Reference |
|---|---|---|
| `cci_poa` | Charlson index from `hospital_diagnosis` rows with `poa_present = 1`, same implementation as `cci`. Replaces `cci`, which also counts conditions acquired during the stay | — |
| `hours_to_intubation` | Admission to first intubation: ≤ 0 h, 0–24 h, 24–72 h, 72 h–7 d, > 7 d. ≤ 0 h marks intubation before arrival | 0–24 h |
| `location_at_intubation` | `adt.location_category` of the row spanning intubation on half-open intervals; `ed`, `icu`, `procedural` kept, no covering row → `unknown`, every other category → `ward_other` | `icu` |
| `procedural_within_24h` | Any `procedural` ADT stay overlapping the 24 h before intubation — the post-operative case mix | 0 |
| `limited_code_status` | Last `code_status_category` at or before intubation is a DNR-type order (DNR, DNAR, UDNR, DNR/DNI, DNAR/DNI, AND). Full and no recorded order form one level: recording of any order varies by year and by unit, so separating them would adjust for documentation practice. Orders after intubation are excluded as mediators. A category in neither set (e.g. `Other`) stops stage 7 until classified | 0 |
| `admission_type` | `admission_type_category` of the encounter's first hospitalization | `ed` |

The covariates were chosen after `code/07b_covariate_probe.py` tabulated each candidate's coverage
by unit and year; the probe reports coverage and crude mortality only, not model fit. Calendar year
was considered and not included.

**Covariate blocks** (`covariate_blocks.csv`) add the covariates cumulatively in a fixed order —
original adjustment (age, `cci`, SOFA) → `cci_poa` in place of `cci` → intubation setting (time,
location, procedural stay) → limited code status → admission type — and report, per block, the
between-unit variance with covariates alone, the share of the null variance it removes, and the
score's OR. With this few units the order is a presentation choice, not a test. Between-unit variation is reported as the raw log-odds variance, the ICC and the
median odds ratio; the MOR is the one to quote, since the ICC's denominator carries the
logistic distribution's fixed individual-level variance and reads low by construction.

**Table 1** is stratified by tertile of the ICUs *ranked by score*, not by tertile of the patient
distribution. With this few units the score is lumpy enough that a patient-weighted cut can leave
a stratum holding a single ICU, which would publish that ICU's case mix in a column of its own.
Strata therefore differ in size; `table1_tertiles.csv` records the composition with anonymised
labels. An absent categorical value is reported as its own `Unknown` level rather than folded
into another category. The p-values treat patients as independent and so ignore clustering within
ICUs — they are descriptive. Cells are not small-cell suppressed.

**Sensitivity analyses** (`sensitivity_analyses.csv`) refit the full specification under six
prespecified variations: the original adjustment (age, `cci`, SOFA) alone, the six SOFA subscores
in place of the total (an absent subscore entering as 0, as in the total), `score_full` as the
exposure, units meeting every `score_core` denominator floor, patients cared for in exactly one
ICU, and SOFA with the liver component removed (the component with the widest between-unit
charting gap). Each refits its **own** null
model on its own subset, so `var_explained_vs_null` compares like with like; `n_excluded` is
reported so a restriction that happens to exclude nobody reads as vacuous rather than as applied.

**The exposure-response figure** holds every covariate at its cohort mean and the random effect
at zero, and traces predicted mortality across the observed score range from `m3`. The band is
the 95% interval from the fixed-effect covariance alone. ICUs are overlaid at their own score and
at the adjusted mortality implied by the conditional mode from `m2` — the model that has not seen
the score — so the points are not fitted to the line they are plotted against.

---

## 6. Caveats

1. **SAT may be effectively a phenotype measure.** Where `sat_delivery_pass_fail` is sparsely
   documented, the SAT numerator is driven almost entirely by the sedative-interruption phenotype
   rather than by documentation. `sat_doc` / `sat_pheno` / `sat_num` are stored separately so the
   split is visible per site; check it before interpreting the SAT rate.
2. **PBW defaults unknown sex to male.** Upstream's `pbw()` tests
   `str(sex).lower().startswith("f")`, so `Unknown`, `None` and `nan` all receive the male base of
   50.0 kg. Inherited deliberately for definitional identity; it biases PBW upward and therefore
   LTVV adherence upward for patients of unrecorded sex.
3. **The SAT phenotype does not detect dose halving.** A >= 50% sedative dose reduction is not
   counted as an SAT (`respiratory.yaml` `sat.phenotype`), so the SAT rate is a lower bound.
4. **Proning rates are case-mix dependent.** The denominator is a severity-defined cohort, so a
   cardiac or neuro ICU may have near-zero eligible patients over years. `score_full` should be
   read as a sensitivity analysis, not as a second primary.
5. **`location_name` is optional and free text in CLIF 2.1.** It has no permissible-value list and
   may be absent, so this pipeline is not guaranteed to run at an arbitrary site. Stage 0 fails with
   a clear message rather than falling back silently.
6. **Respiratory SOFA is limited by paired FiO2, not by oxygenation.** `clif_respiratory_support` is
   event-based, so `fio2_set` appears only on rows where it was charted; many IMV rows carry a null
   because nothing changed. The analysis dataset therefore computes oxygenation from the waterfall,
   which forward-fills FiO2 within mode blocks, with an S/F fallback (Rice 2007 equivalents) for the
   pre-intubation window where no ventilator exists. SpO2 is not the constraint; it is densely
   charted throughout.
7. **A missing SOFA subscore is summed as 0**, i.e. as normal. Per-component and per-unit
   missingness is reported by stage 7 in `missingness_report.csv` and `missingness_by_unit.csv`;
   read those before adjusting on SOFA, since unit-differential charting becomes unit-differential
   measured severity.
8. **A benign library warning.** `pandas` with `numpy` at the pinned versions emits
   `DeprecationWarning: The 'generic' unit for NumPy timedelta is deprecated` from inside pandas'
   own `Timestamp` arithmetic. The unit at every call site here is explicit and the arithmetic is
   asserted correct in `tests/test_scoring.py`; the warning is filtered narrowly in stage 2.
9. **The score is a level-2 exposure, so a single site cannot power it.** Its coefficient is
   estimated from as many points as the site has ICUs, however many patients contribute, and the
   confidence band in the exposure-response figure reflects fixed-effect uncertainty only — read
   it as a lower bound. Single-site results are descriptive; the fix is more participating sites,
   not more patients.
