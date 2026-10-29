# Methods — ventilator QI composite score

Companion to [`../README.md`](../README.md). Covers what each metric means, the one place this
pipeline deviates from upstream, how the composite and leave-one-out exposure are formed, and the
caveats a reader should know before using the score.

Definitions are those of [`AartikSarma/clif-dashboards`](https://github.com/AartikSarma/clif-dashboards)
at commit `f55c379`, read from `definitions/respiratory.yaml` (a verbatim copy, hash-pinned in
`definitions/SOURCE`). Upstream code paths are cited as `src/clif_dash/...`.

---

## 1. Cohort

| Step | Rule | Source |
|---|---|---|
| Encounter linking | Hospitalizations < 6 h apart become one `encounter_block` | clifpy `stitch_encounters`; `platform.yaml` `stitch_hours` |
| Adults | `age_at_admission >= 18` | `platform.yaml` `age_min` |
| Ventilator timeline | clifpy respiratory-support waterfall: hourly scaffold at `HH:59:59`, device/mode repair, forward-fill within mode blocks | `respiratory.yaml` `imv_source` |
| IMV run | Contiguous rows with `device_category = imv` | `upstream/prep.py:163` `imv_runs` |
| IMV episode | New episode after >= 24 h off invasive ventilation | `respiratory.yaml` `episode_gap_hours` |
| ICU requirement | Episode must overlap an ADT stay with `location_category = icu` | `respiratory.yaml` `require_icu` |
| Outside intubation | Excluded when first ADT location is not the ED **and** first recorded device is IMV **and** `episode_seq == 1` | `respiratory.yaml` `exclude_outside_intubation` |
| Outliers | Out-of-range values nulled per clifpy `outlier_config.yaml`; `fio2_set` divided by 100 when its whole-table mean > 1 | `upstream/clif_io.py:125` |

At UChicago this yields **16,228** IMV episodes across **13,495** encounter blocks, **12,969** of
them first episodes, spanning 2018-Q1 to 2024-Q4.
<!-- src: output/uchicago/intermediate_phi/cohort/manifest.json → counts; cross-checked against clif-dashboards/output/uchicago/respiratory/summary.json funnel "IMV episodes analysed" = 16228 and proning_funnel step 1 = 12969 -->

Time attribution follows upstream: a record's period is derived from its own timestamp in
**site-local** time, and an episode's period from its start. The only change is widening the grain
from month to quarter (`period` in the site profile).

---

## 2. Metric definitions

### 2.1 LTVV — low tidal volume ventilation

- **Denominator:** hourly-scaffold IMV hours inside the episode whose `mode_category` is
  volume-targeted (`assist control-volume control`, `pressure-regulated volume control`, `simv`,
  `volume support`), with a non-null `tidal_volume_set` **and** a known predicted body weight.
  An episode with no PBW contributes nothing (`missing_height: exclude`).
- **Numerator:** those hours with `tidal_volume_set / PBW <= 8` mL/kg (tolerance `1e-9`).
  Rates at 6 and 7 mL/kg are emitted as `ltvv_rate_at_6` / `ltvv_rate_at_7` for sensitivity but are
  not part of any score.
- **PBW:** ARDSNet — `45.5 (female) or 50.0 (otherwise) + 0.91 x (height_cm - 152.4)`.
- **Height:** the **first** `vitals` `height_cm` between 120 and 220 cm in the encounter block.
- **Grain:** one IMV hour. Source: `upstream/metrics.py:189-208`, `pbw` at `:17`.

### 2.2 SAT — spontaneous awakening trial

- **Vent-day:** a site-local calendar day with **>= 6** hourly IMV scaffold rows inside the episode.
- **Denominator:** vent-days on a continuous sedative infusion (propofol, midazolam, lorazepam,
  dexmedetomidine, ketamine — opioid-only infusions do *not* qualify a day), with **no** paralytic
  infusion overlapping the day, and not under comfort care.
- **Numerator:** a documented SAT (`patient_assessments.assessment_category =
  sat_delivery_pass_fail`, pass or fail — the value is not inspected) **or** the EHR phenotype: all
  continuous sedatives at zero for >= 30 min while still on IMV, after having been on that day.
- **Grain:** one eligible vent-day. Source: `upstream/metrics.py:224-279`, phenotype at `:413`.

### 2.3 SBT — spontaneous breathing trial

- **Denominator:** vent-days where at **some** IMV hour that day `fio2_set <= 0.5`,
  `peep_set <= 8`, no paralytic covers that instant, and the norepinephrine-equivalent dose is
  `<= 0.2` mcg/kg/min; and the day is not under comfort care. A missing FiO2 or PEEP cannot
  establish eligibility (`NaN <= x` is false).
- **Numerator:** a documented SBT (`sbt_delivery_pass_fail`) **or** the phenotype: pressure
  support/CPAP with `pressure_support_set <= 8` and `peep_set <= 5`, or `blow by`, sustained
  >= 30 min and beginning that day, judged on real (non-scaffold) rows.
- **Grain:** one eligible vent-day. Source: `upstream/metrics.py:263-274`, phenotype at `:455`.

### 2.4 Proning

- **Cohort (the denominator):** first IMV episode of an adult ICU encounter; not intubated at an
  outside hospital; no tracheostomy in the first 24 h; a first qualifying arterial blood gas
  (`P/F < 150` with `PEEP >= 5` and `FiO2 >= 0.6`) within 36 h of IMV start; confirmed either by a
  second qualifying ABG 12-24 h later or by proning within 24 h of the first; not proned before
  the first qualifying ABG; and not in a `procedural` location in the preceding 48 h.
- **Enrollment:** the earlier of the second qualifying ABG and the first prone session.
- **Numerator (`p12`):** first prone session starting <= 12 h after enrollment.
- **Prone session:** only `position_category = prone`; a session ends at the next documented
  position change, or at the end of the ventilation episode for the last row.
- **Grain:** one patient-encounter. Source: `upstream/proning.py:52-159`.

At UChicago the cohort is **682** patients with **111** proned within 12 h.
<!-- src: output/uchicago/final_no_phi/verification.json → checks "proning cohort size" / "proned <= 12 h (p12)", both PASS against clif-dashboards/output/uchicago/respiratory/proning.parquet -->

---

## 3. The one deviation: time-resolved unit attribution

Upstream attributes an entire IMV episode to the ICU where it **started**, labelled by
`location_type` (`upstream/prep.py:209` `attribute_unit`). For a score used as a model exposure that
is too coarse in two ways: a transferred patient's whole course is credited to the sending unit, and
physically distinct units sharing a type collapse together. At UChicago `N09S` and `N09N` are both
`medical_icu`, so keying on the categorical would merge two separate MICUs.
<!-- src: output of code/00_probe_units.py, [units] table — 7 distinct ICU location_name values, N09S and N09N both under location_type medical_icu -->

This pipeline instead credits each record to the ICU that delivered it:

| Metric | Attributed to | Rationale |
|---|---|---|
| `ltvv` | ICU covering the hour's **midpoint** (`t - 30 min`) | Scaffold rows are stamped at `HH:59:59` and stand for the hour they end; upstream's own `_in_segments` (`metrics.py:439`) tests the same instant |
| `sat` / `sbt` | ICU covering the **earliest IMV scaffold hour** of that calendar day | SAT and SBT are morning, nursing-driven; the unit holding the trial opportunity is the one the patient enters the day in. Tie-free, so order-invariant |
| `prone_12h` | ICU covering the **enrollment** instant | Enrollment is when the proning decision becomes due |

ADT locations are mutually exclusive in time, so "the ICU covering instant *t*" is unambiguous; the
lookup is a backward as-of join with a half-open `[in_dttm, out_dttm)` test.

**Attribution only redistributes.** Stage 5 asserts that denominators summed over every unit cell
plus the unattributed remainder equal the cohort total, for all four metrics.

### Unattributed care

Records falling in no named ICU stay are reported and excluded from the panel. At UChicago:

| Metric | Unattributed | Where it actually was |
|---|---|---|
| `ltvv` hours | 10.73% | ward 44%, procedural 25%, ED 23%, no ADT coverage 8% |
| vent-days | 16.35% | ward 43%, ED 34%, procedural 16%, no ADT coverage 7% |
| proning | 1.91% | — |

<!-- src: output/uchicago/intermediate_phi/records/manifest.json → unattributed_share and unattributed_placement; conservation confirmed in output/uchicago/final_no_phi/verification.json section [2] -->

This is expected: an episode need only *overlap* an ICU stay, so parts of it occur on the ward, in
the OR, or in the ED. Only 7-8% of the unattributed records have no ADT coverage at all. Excluding
them is a deliberate choice — it is real care, but not care any ICU can be scored for.

---

## 4. Scoring

**Pooled rates.** `rate = sum(numerator) / sum(denominator)` over the cell — never a mean of
per-patient percentages. So a unit-quarter's LTVV rate is hour-weighted and its SAT/SBT rates are
vent-day-weighted, matching upstream's aggregation (`clif_dash/server/aggregate.py`).

**Composite.** A weighted mean of pooled rates times 100, with weights normalized within each score.
A score requires **every** one of its components to have a non-zero denominator in that cell;
otherwise it is null and counted. Renormalizing over whatever happened to be present would make the
score mean different things in different cells — unacceptable for a regression exposure.
`composite(..., require_all=False)` exists for sensitivity work.

**Thin cells are flagged, never dropped.** Dropping them would bias the panel toward high-volume
units. `<metric>_low` marks a cell below that metric's floor; `<score>_low` ORs only the floors of
the metrics that score actually uses. The global `low_denominator` ORs all of them and is therefore
the *wrong* filter for `score_core` — at UChicago the proning floor alone marks 139 of 195 cells,
none of which bears on `score_core`.
<!-- src: output of code/03_build_panel.py — floor prone_12h flagged 139/195; score_core below its own floors 8/195 -->

**Leave-one-out.** A unit's measured quality partly reflects the care given to the very patient
whose outcome is being modelled, so regressing an outcome on the raw cell rate builds in a
mechanical association. The LOO rate removes **all** of that patient's numerator and denominator
from the cell:

```
loo_rate = (cell_num - own_num) / (cell_den - own_den)
```

A patient who contributed nothing to a metric keeps the cell's full rate for it (removing zero
changes nothing). Where removing the patient leaves no denominator, the rate and the score are null
— not zero. At UChicago the median `|score_core - LOO|` is 0.08 points, p95 0.56, max 23.67: small
on average, but the tail is exactly where it matters.
<!-- src: output of code/04_exposure_assignment.py — score_core LOO computable 14,988/14,991 -->

**Exposure assignment.** 20.9% of patients contribute to more than one cell, so two artifacts are
emitted: a long `(patient, cell)` table, and a primary-cell assignment using the cell where the
patient accrued the most **vent-days** (`sat` denominator — the grain SAT and SBT are measured on).
<!-- src: output of code/04_exposure_assignment.py — 11,702 patients, 14,991 (patient, cell) rows, 2,451 in >1 cell -->

---

## 5. Caveats

1. **SAT at UChicago is effectively a phenotype measure.** Only **254** `sat_delivery_pass_fail`
   rows exist in the whole dataset, against **3,508** for SBT, so the SAT numerator is driven almost
   entirely by the sedative-interruption phenotype rather than by documentation.
   `sat_doc` / `sat_pheno` / `sat_num` are stored separately so this stays visible.
   <!-- src: clif-dashboards/output/uchicago/respiratory/summary.json → data_quality.sat_documented_rows = 254, data_quality.sbt_documented_rows = 3508 -->
2. **PBW defaults unknown sex to male.** Upstream's `pbw()` tests
   `str(sex).lower().startswith("f")`, so `Unknown`, `None` and `nan` all receive the male base of
   50.0 kg. Inherited deliberately for definitional identity; it biases PBW upward and therefore
   LTVV adherence upward for patients of unrecorded sex.
3. **The SAT phenotype does not detect dose halving.** A >= 50% sedative dose reduction is not
   counted as an SAT (`respiratory.yaml` `sat.phenotype`), so the SAT rate is a lower bound.
4. **Proning rates are case-mix dependent.** The denominator is a severity-defined cohort, so a
   cardiac or neuro ICU may have near-zero eligible patients over years. `score_full` should be read
   as a sensitivity analysis, not as a second primary.
5. **`location_name` is optional and free text in CLIF 2.1.** It has no permissible-value list and
   may be absent, so this pipeline is not guaranteed to run at an arbitrary site. Stage 0 fails with
   a clear message rather than falling back silently.
6. **A benign library warning.** `pandas 2.3.3` with `numpy 2.5.4` (the upstream's pins) emits
   `DeprecationWarning: The 'generic' unit for NumPy timedelta is deprecated` from inside pandas'
   own `Timestamp` arithmetic. The unit at every call site here is explicit and the arithmetic is
   asserted correct in `tests/test_scoring.py`; the warning is filtered narrowly in stage 2.

---

## 6. Source map

| Claim | Value | Source |
|---|---|---|
| IMV episodes analysed | 16,228 | `intermediate_phi/cohort/manifest.json` `counts.imv_episodes` |
| Distinct ICU `location_name` | 7 | `code/00_probe_units.py` `[units]` |
| Panel cells | 195 (7 units x 28 quarters) | `final_no_phi/panel_provenance.json` `cells` |
| LTVV denominator (attributed) | 893,991 IMV hours | `code/03_build_panel.py` `[contributions]` |
| LTVV pooled rate (attributed) | 92.47% | `final_no_phi/verification.json` section [3] |
| SAT denominator (attributed) | 37,694 vent-days | `code/03_build_panel.py` `[contributions]` |
| SAT pooled rate (attributed) | 47.80% | `final_no_phi/verification.json` section [3] |
| SBT denominator (attributed) | 61,309 vent-days | `code/03_build_panel.py` `[contributions]` |
| SBT pooled rate (attributed) | 20.69% | `final_no_phi/verification.json` section [3] |
| Proning cohort | 682 patients, 111 proned <= 12 h | `final_no_phi/verification.json` section [1] |
| `score_core` computable | 193/195 cells, median 54.0 | `code/03_build_panel.py` `[panel]` |
| `score_full` computable | 153/195 cells, median 43.1 | `code/03_build_panel.py` `[panel]` |
| Equivalence checks | 13/13 PASS, 0 episodes differing | `final_no_phi/verification.json` |
| Cached vs cold-build panel | identical, 195 cells x 28 columns | `code/06_compare_tiers.py --compare` |
| Runtime, cached / cold | 12m27s / 17m19s (build 5m15s, 1.1 GB) | `output/uchicago/run_timings.csv` |
