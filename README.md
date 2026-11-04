# clif-vent-qi-score

*Composite ventilator quality scores for individual ICUs, built to serve as a modeling exposure.*

A reproducible CLIF pipeline that scores a single ICU — identified by `adt.location_name` — over a
calendar period, from four respiratory quality metrics: **low tidal volume ventilation (LTVV)**,
**spontaneous awakening trial (SAT) rate**, **spontaneous breathing trial (SBT) rate**, and
**proning rate**. Each metric contributes a pooled rate; equally weighted, they form a composite
quality score per unit-period.

![CLIF 2.1](https://img.shields.io/badge/CLIF-2.1-blue) ![python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue) ![definitions f55c379](https://img.shields.io/badge/definitions-clif--dashboards%40f55c379-informational)

---

## Objective

Produce a **composite quality score per ICU** suitable for use as an exposure in a downstream
outcome model, plus a per-patient exposure assignment.

Two grains are emitted from the same contribution table:

| Grain | File | Role |
|---|---|---|
| **unit** | `unit_panel.csv` | **PRIMARY exposure** — one score per ICU, pooled over all available time |
| unit x period | `unit_period_panel.csv` | secondary — the same scores per quarter, for sensitivity / time-varying analysis |

Pooling the whole window is **not** the mean of the periodic rates: it re-divides the summed
numerators and denominators, so a near-empty quarter cannot pull a unit's score around.

**This is not a dashboard.** The metric definitions come from
[`AartikSarma/clif-dashboards`](https://github.com/AartikSarma/clif-dashboards), pinned at commit
`f55c379`; its predicate functions are imported unchanged. This repo changes exactly one thing —
**the grain at which those predicates are emitted** — so care can be attributed to the ICU that
actually delivered it rather than to the ICU an episode happened to begin in.

| Metric | Measures | Denominator | Reference |
|---|---|---|---|
| `ltvv` | Set tidal volume <= 8 mL/kg predicted body weight | IMV hours | ARDSNet; ATS/ESICM/SCCM 2017 |
| `sat` | Sedation interruption on eligible ventilated days | eligible vent-days | ABCDEF bundle; CLIF SAT phenotyping |
| `sbt` | Breathing trial on eligible ventilated days | eligible vent-days | ABCDEF bundle; CLIF SBT phenotyping |
| `prone_12h` | Proned within 12 h of enrollment | PROSEVA-eligible patients | PROSEVA; CLIF_Proning_Incidence_Severe_ARF |

The pipeline is **descriptive**: it computes the exposure. Outcome and causal modeling are
deliberately out of scope.

### Two scores, and why

`score_core` is the **primary exposure**; `score_full` is computed alongside it on the unit panel (stage 8 no longer fits it as a sensitivity row).

```
score_core = 100 x mean(ltvv_rate, sat_rate, sbt_rate)
score_full = 100 x mean(ltvv_rate, sat_rate, sbt_rate, prone_12h_rate)
```

Proning is excluded from the primary score because its denominator is a *severity-defined cohort*
typically orders of magnitude smaller than the vent-day denominators. At equal weight, proning
would contribute a quarter of the score while being estimated with far less precision, dominating
the composite's noise rather than its signal. All four metrics are computed and stored regardless;
only score membership differs, and that is a one-line config edit in
`definitions/composite.json`. Check the per-metric denominators your site produces (stage 3 prints
them) before reading `score_full`.

---

## Required CLIF tables and fields

CLIF **spec version 2.1**. The cohort and all four metrics are built from the tables below.

**Shared by every metric**

| Table | Fields / categorical values |
|---|---|
| `patient` | `patient_id`, `sex_category` (`Male` / `Female` / `Unknown`) |
| `hospitalization` | `patient_id`, `hospitalization_id`, `admission_dttm`, `discharge_dttm`, `age_at_admission`, `discharge_category` |
| `adt` | `hospitalization_id`, `in_dttm`, `out_dttm`, `location_category` = `icu`, **`location_name`**, `location_type`, `hospital_id` |
| `respiratory_support` | `recorded_dttm`, `device_category` = `imv`, `mode_category`, `tracheostomy`, `fio2_set`, `peep_set` |

**Per metric**

| Metric | Additional fields |
|---|---|
| `ltvv` | `respiratory_support.tidal_volume_set`; `vitals.vital_category` = `height_cm` (for predicted body weight) |
| `sat` | `medication_admin_continuous.med_category` in {propofol, midazolam, lorazepam, dexmedetomidine, ketamine} and the paralytics; `med_dose`, `mar_action_category`; `patient_assessments.assessment_category` = `sat_delivery_pass_fail`; `code_status.code_status_category` |
| `sbt` | `respiratory_support.pressure_support_set`, `mode_category` in {`pressure support/cpap`, `blow by`}; `patient_assessments.assessment_category` = `sbt_delivery_pass_fail`; vasoactive `med_category` values for the norepinephrine-equivalent dose |
| `prone_12h` | `labs.lab_category` = `po2_arterial`; `position.position_category` = `prone`; `adt.location_category` = `procedural` (recent-OR exclusion) |

**Outcome model covariates** (stage 7; built in `common/covariates.py`)

| Table | Fields / categorical values |
|---|---|
| `hospital_diagnosis` | `hospitalization_id`, `diagnosis_code`, `diagnosis_code_format` = `icd10cm` (Charlson), **`poa_present`** = 1 (present-on-admission Charlson; required) |
| `adt` | `location_category` spanning intubation (`ed` / `icu` / `procedural` kept, others pooled); `procedural` in the 24 h before intubation |
| `hospitalization` | `admission_type_category` |
| `code_status` | `patient_id`, `start_dttm`, `code_status_category`; the last order between admission and intubation (earlier admissions ignored). Limited = DNR, DNAR, UDNR, DNR/DNI, DNAR/DNI, AND; not limited = Full, Presume Full, Other, none recorded. Any other value stops stage 7 until it is classified in `common/covariates.py` |
| `patient_assessments` | `hospitalization_id`, `recorded_dttm`, `assessment_category` = `gcs_total`, `numerical_value`; the worst GCS in the 24 h before intubation (primary physiology). The other physiology terms reuse the SOFA inputs: `labs` `po2_arterial`, `platelet_count`, `bilirubin_total`, `creatinine`; `vitals` `spo2`, `map`; vasopressors as norepinephrine equivalents. A missing `clif_patient_assessments` file stops stage 7 |

> **`location_name` is optional in CLIF 2.1** and is free text with no permissible-value list, so it
> is site-specific and may be absent entirely. Stage 0 checks for it and fails with a clear message
> rather than silently falling back. Stage 0 reports how many ICU rows carry a name and how many
> distinct units they resolve to, and asserts that each name nests within exactly one
> `location_type` — two physically distinct ICUs can share a type and would be collapsed by keying
> on the categorical.

Values are nulled outside clifpy's `outlier_config.yaml` ranges before any metric is computed, and
`fio2_set` is rescaled by 1/100 when its whole-table mean exceeds 1 — both exactly as upstream.

---

## Cohort identification

Unit of analysis for scoring is the **(ICU, calendar period)** cell; the underlying records are IMV
hours, ventilated days, and patients.

- Hospitalizations are linked into `encounter_block`s with clifpy `stitch_encounters` (< 6 h apart).
- The clifpy respiratory-support **waterfall** builds an hourly ventilator timeline; contiguous IMV
  stretches become **runs**, and a new **episode** starts after >= 24 h off invasive ventilation.
- **Included:** adults (age >= 18 at admission) whose IMV episode overlaps an ADT stay with
  `location_category = icu`.
- **Excluded:** episodes intubated at an outside hospital (first ADT location is not the ED *and*
  the first recorded device is IMV *and* it is the encounter's first episode).
- **Per metric**, each record is credited to the ICU that delivered it:

  | Metric | Record | Attributed to |
  |---|---|---|
  | `ltvv` | one eligible IMV scaffold hour | the ICU covering that hour's midpoint |
  | `sat` / `sbt` | one vent-day (>= 6 IMV hours in a site-local calendar day) | the ICU the patient *began* that ventilated day in |
  | `prone_12h` | one PROSEVA-eligible patient | the ICU covering the enrollment instant |

  SAT and SBT are morning, nursing-driven processes, so the unit holding the trial opportunity is
  the one the patient enters the day in. The pick is tie-free and order-invariant.

- Records falling in **no named ICU stay** are reported and then excluded — real care, but not care
  any ICU can be scored for. Stage 2 reports the share excluded and breaks it down by
  `location_category`, separating ward / procedural / ED time from records with no ADT coverage.

Output formats: Parquet for row-level working files, **CSV + Parquet** for the shareable panel.

---

## Configuration

Two layers, deliberately separated:

- **`definitions/`** — the shared, versioned clinical spec. Committed; sites do not edit it.
  `respiratory.yaml` and `platform.yaml` are verbatim copies of `clif-dashboards`'
  `config/decisions/` at the pinned SHA, with hashes recorded in `definitions/SOURCE`.
  `composite.json` declares which metrics make up which score.
- **`sites/<site>.json`** — per-site access only (data path, timezone, optional cache). Gitignored.

```bash
cp sites/uchicago.example.json sites/<site>.json   # then edit data_path
./run_scores.sh --site <site>                      # or: export CLIF_SITE=<site>
```

| Field | Meaning |
|---|---|
| `site_id` | Site label stamped into provenance |
| `clif_version` | The CLIF **spec** version (e.g. `2.1.0`). A conversion/dataset release goes in `dataset_version` — never here |
| `data_path` | Folder holding `clif_*.parquet` |
| `file_prefix` / `file_format` | Default `clif_` / `parquet` |
| `timezone` | Site-local tz; all calendar days and periods are binned in it |
| `stage_dir` | **Optional** cache — a `clif-dashboards` respiratory build's `stage/` folder. `null` is fine |
| `period` | Grain of the **secondary** panel: `quarter` (default) or `month`. The primary has no period dimension |
| `min_denominator` | Nested by grain (`unit`, `unit_period`), then by metric: the denominator floor below which a cell is **flagged**, not dropped. Stated per grain because a quarter-calibrated floor can never bind on a seven-year pooled denominator |
| `unit_include` / `unit_collapse` | Restrict to, or pool, specific `location_name` values |

`definitions/` is also what the upstream builder reads: the generated site YAML points
`SiteConfig.decisions_dir` at this folder, so there is no second copy of a threshold to drift from.
Stage 0 re-diffs `definitions/` against the pinned upstream commit and fails loudly on divergence.

### Where outputs land

| Path | Contents | Shareable? |
|---|---|---|
| `output/<site>/intermediate_phi/` | Row-level records, contributions, per-patient exposure, the analysis dataset, unit crosswalk | **Never** |
| `output/<site>/upstream_build/` | Stage cache written by a cold build (~1 GB) | **Never** |
| `output/<site>_backup_<timestamp>/` | The previous output, moved aside by `run_scores --fresh`; delete once the new run succeeds | **Never** |
| `output/<site>/final_no_phi/unit_panel.csv` | **The primary exposure** — one row per ICU | Yes, site-internal |
| `output/<site>/final_no_phi/unit_period_panel.csv` | Secondary, one row per ICU x period | Yes, site-internal |
| `output/<site>/final_no_phi/*_anonymized.csv` | Both, with `location_name` stripped (`Unit 1..n` only) | **Yes — use these externally** |
| `output/<site>/final_no_phi/verification.json` | Equivalence check results | Yes |
| `output/<site>/final_no_phi/08_outcome_model/` (`_<cohort>/` for a restricted `CLIF_COHORT`) | Model coefficients, ICC/MOR (with variance LRT and MOR CI), per-unit effects, Table 1, three figures, `captions.md` and `timings.csv`; covariate blocks and sensitivity analyses for the primary cohort (`early_nonop`) only | Site-internal; Table 1 still needs a small-cell pass |
| `output/<site>/final_no_phi/07b_covariate_probe/` | Coverage and crude mortality of candidate case-mix covariates, overall, by `unit_id` and by intubation year; counts below `small_cell_threshold` (`.clif-prepush.json`) suppressed | Site-internal |
| `output/pooled/` | Multi-site stack from stage 7c: `intermediate_phi/analysis/` (all patients, never shared), `intermediate_phi/site_key.json` (letter-to-site key, never shared), `final_no_phi/unit_panel.parquet`, `unit_panel_anonymized.csv`, `pooled_provenance.json`, `08_outcome_model/` | `final_no_phi/` site-anonymized |
| `output/<site>/final_no_phi/logs/` | Per-step logs, aggregate only | Yes |
| `output/<site>/run_timings.csv` | Per-stage wall clock + git SHA | Yes |

Everything under `08_outcome_model/` labels ICUs as `Unit 1..n` except `unit_summary.csv`,
`unit_effects.csv` and `deaths_by_unit.png`, which carry `location_name` for site-internal
reading. Anonymise those before they go anywhere external. `unit_caterpillar.png` labels ICUs
`ICU 1..n` in plot order with no site or unit name; the `icu` column of `unit_effects.csv` is
the key.

---

## Prerequisites

Python **3.12** (matching the upstream's `requires-python >= 3.12`).

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

R dependencies are pinned with **renv** (`renv.lock`, 38 packages incl. arrow 26.0.0,
lme4 2.1-0, ggplot2 4.0.3). On a new machine:

```r
renv::restore()
```

The compute stack is **pinned exactly** — `clif-dashboards` at a commit, plus `clifpy==2.1.0`,
`pandas==2.3.3`, `numpy==2.5.4`, `pyarrow==26.0.0`, `duckdb==1.5.6`, `polars==1.44.2` — because
every site must compute identical numbers. The `clif-dashboards` pin is a *definitional* pin, not
merely a dependency pin: bumping it can change what the metrics mean, so re-run stage 5 after any
bump.

Stage 7 computes classic SOFA with the calculator from
[`CLIF-epidemiology-of-CRRT`](https://github.com/Common-Longitudinal-ICU-data-Format/CLIF-epidemiology-of-CRRT),
vendored verbatim under `vendor/crrt_sofa/` (Apache-2.0, license included), so no CRRT checkout
is needed. `vendor/crrt_sofa/SOURCE` records the upstream commit and a sha256 per file, and
`tests/test_vendor_crrt.py` fails if any vendored byte changes. One upstream gap is patched from
`common/sofa_crrt.py` instead: the P/F pairing aligns lab and respiratory-support time zones but
not time units, so both are cast to microseconds first. To update, re-copy from a new
commit and replace the hashes and SHA together.

---

## Running the pipeline

**macOS / Linux**

```bash
./run_scores.sh --site uchicago
./run_scores.sh --site uchicago --rebuild        # ignore any cache, build from raw CLIF
./run_scores.sh --site uchicago --fresh          # back up output/<site>, then --rebuild
./run_scores.sh --site uchicago --from 07        # rerun stage 7 onward, reusing stages 0-4
./run_scores.sh --site uchicago --probe          # also run the 07b covariate probe
./run_scores.sh --site uchicago --require-stage  # fail instead of cold-building (CI)
./show_results.sh --site uchicago                # print the shareable, aggregate outputs
```

**Windows (PowerShell)**

```powershell
.\run_scores.ps1 -Site uchicago
.\run_scores.ps1 -Site uchicago -Rebuild
.\run_scores.ps1 -Site uchicago -Fresh
.\run_scores.ps1 -Site uchicago -From 07
.\run_scores.ps1 -Site uchicago -Probe
.\run_scores.ps1 -Site uchicago -RequireStage
.\show_results.ps1 -Site uchicago
```

`--from` accepts `00`, `01`, `02`, `03`, `04` or `07` and needs an earlier run of the site to
resume from; `--rebuild` and `--fresh` rebuild the cohort in stage 0, so they need `--from 00`.
`--fresh` moves the old output aside rather than deleting it. `show_results` prints only named
aggregate files under `final_no_phi/` (the probe log, the pooling log, the model summaries) and
leaves out the per-unit tables, whose counts can fall below the small-cell threshold.

The outcome model is a separate R step, run after the Python pipeline. It reads the site from
`CLIF_SITE` (default `uchicago`) and the outcome cohort from `CLIF_COHORT` (default `all`); the
sensitivity analyses and covariate blocks run only for the primary cohort, `early_nonop`:

```bash
CLIF_COHORT=early_nonop Rscript code/08_outcome_model.R
```

To redraw only the caterpillar plot from a finished stage 8 (no refit), run
`Rscript code/caterpillar.R` with the same `CLIF_SITE`.

Tests (synthetic frames only, no CLIF data):

```bash
.venv/bin/python -m pytest tests/ -q
```

### Pooling several sites

On a machine that holds every member site's data (under a data use agreement), run stages 0-7 for
each site with its own profile, then stack their patient rows and unit panels and fit the model
once on the stack:

```bash
./run_scores.sh --site site_one --probe                     # and each other member
./run_pooled.sh --members site_one,site_two,site_three      # order sets Site A, B, ...
./show_results.sh --site pooled --cohort early_nonop        # the primary cohort
# refit one cohort on the existing pooled data:
./run_pooled.sh --model-only --cohort early_nonop
```

```powershell
.\run_scores.ps1 -Site site_one -Probe
.\run_pooled.ps1 -Members site_one,site_two,site_three
.\show_results.ps1 -Site pooled -Cohort early_nonop
.\run_pooled.ps1 -ModelOnly -Cohort early_nonop
```

`run_pooled` writes `sites/pooled.json` from `--members` (or uses the existing one), checks that
R and its packages load before the long run, then runs stage 7c and the pooled stage 8;
`--pool-only` stops after 7c, `--model-only` skips it, and `--cohort` picks which of stage 8's
outcome cohorts to fit (default all three: `all`, `early_nonop`, `early_nonop_hypoxemia`;
definitions in `docs/METHODS.md`), each into its own `08_outcome_model_<cohort>/`. `early_nonop`
is the primary cohort and the only one fit with the sensitivity analyses and covariate blocks.
The cohorts run in parallel, each logging to `output/pooled/final_no_phi/logs/08_outcome_model_<cohort>.log`
rather than the terminal (Windows splits errors into a separate `.err.log`), and each writes
`timings.csv`, the seconds spent in each section. Run by
hand, the steps are
`code/07c_pool_sites.py` followed by `CLIF_SITE=pooled CLIF_COHORT=<cohort> Rscript
code/08_outcome_model.R`.

Raw CLIF tables are never merged. Stage 7c refuses to pool sites whose definitions, CLIF major.minor version (patch releases may differ)
or upstream SHA differ, or whose row- and panel-building code differs between the commits they
ran (`git diff` over the files each one imports: panels compare only what stages 0-4 import, so a
stage-7 fix means rerunning stage 7, not a cold build; unrelated commits never block). Keys are prefixed with
the site letter and units are labelled `Site A · Unit n`; the letter-to-site key is written to
`output/pooled/intermediate_phi/site_key.json` only. In pooled mode stage 8 adds a site fixed
effect to every specification and leave-one-site-out sensitivity rows.

### The stage cache is an optimization, never a dependency

A `clif-dashboards` respiratory build leaves a `stage/` folder holding the stitched encounters, the
IMV episodes and the hourly waterfall. Reusing it skips the expensive re-stitch and re-waterfall.
Stage 1 resolves its input in three tiers and reports which it used:

1. **`stage`** — the cache exists, is complete, **and** was built under the same decisions. The
   match is verified against the decisions tree embedded in the build's own `summary.json`, i.e.
   against what the cache actually did, not a hash of what it was supposed to read.
2. **`build`** — otherwise, the upstream respiratory build is invoked in-process to create one.
   This is the *same code* that produced the cache, so there is exactly one staging implementation.
   Expect a full waterfall rebuild and ~1 GB of output.
3. **fail** — no usable cache and no readable `data_path`, with remediation text.

Only stage 0 resolves the source; it records the result and later stages reuse it, so a cold build
runs **once** per invocation rather than once per stage.

`code/06_compare_tiers.py` checks that a cached run and a `--rebuild` run produce identical
panels at both grains. Per-stage wall clock is appended to `output/<site>/run_timings.csv` on
every run.

---

## Pipeline steps

| Step | Script | Output |
|---|---|---|
| 0 | `code/00_probe_units.py` | Aggregate-only probe: definition hashes, composite coverage, config-key coverage, ICU unit inventory, panel shape |
| 1 | `code/01_build_cohort.py` | `cohort/` — episodes with period labels, IMV runs, encounters, ICU intervals keyed by `location_name` |
| 2 | `code/02_metric_records.py` | `records/` — `ltvv_hours`, `vent_days`, `prone_patients`, each attributed to a unit; asserts equivalence per chunk |
| 3 | `code/03_build_panel.py` | `panel/contributions.parquet` + `final_no_phi/{unit,unit_period}_panel.{csv,parquet}` at both grains |
| 4 | `code/04_exposure_assignment.py` | `exposure/exposure_{unit,unit_period}_{long,primary}.parquet` — per-patient exposure |
| 5 | `code/05_verify_equivalence.py` | `final_no_phi/verification.json` — cross-run equivalence vs the dashboard's persisted output |
| 7 | `code/07_analysis_dataset.py` | `analysis/analysis_dataset.{csv,parquet}` — one row per patient for the outcome model, plus `_extended` with demographics, the case-mix covariates, SOFA components, physiology values (incl. pre-intubation GCS) and ICU-stay diagnostics |
| 7b | `code/07b_covariate_probe.py` | `final_no_phi/07b_covariate_probe/covariate_coverage.csv` + `logs/07b_covariate_probe.log` — on demand: coverage of candidate covariates (location at intubation, recent procedural stay, admission type, pre-intubation code status, admission-to-intubation time, Charlson conditions with and without present-on-admission, primary-diagnosis coverage and ICD-10 chapter) in the model sample, plus which hospital each ICU sits in (anonymized) |
| 7c | `code/07c_pool_sites.py` | `output/pooled/` — on demand, multi-site: stacks member sites' stage-7 patient rows and unit panels (site-prefixed keys, anonymized labels) after checking versions, build code, required columns and reference levels; `./run_pooled.sh` runs it and then `CLIF_SITE=pooled Rscript code/08_outcome_model.R` |
| 6 | `code/06_compare_tiers.py` | `tier_comparison/` — on demand, proves a cold build reproduces the cached-tier panel exactly |
| 8 | `code/08_outcome_model.R` | `final_no_phi/08_outcome_model/` — the multilevel outcome model (R), adjusted for physiology at intubation: coefficients, ICC/MOR with a variance LRT and profile CI, per-unit effects, Table 1, three figures; cumulative covariate blocks and sensitivity analyses for the primary cohort (`CLIF_COHORT=early_nonop`). Run after the Python pipeline |

Stages 0-5 and 7 run on every build via the runner (`--from` resumes partway). Stage 7b is a
diagnostic, run on demand after stage 7 (`./run_scores.sh --site uchicago --probe`, or
`.venv/bin/python code/07b_covariate_probe.py --site uchicago` alone). Stage 7c pools several
sites and runs on demand (`./run_pooled.sh`). Stage 6 is a verification tool invoked manually
around a `--rebuild` run:

```bash
.venv/bin/python code/06_compare_tiers.py --site uchicago --snapshot
./run_scores.sh --site uchicago --rebuild
.venv/bin/python code/06_compare_tiers.py --site uchicago --compare
```

---

## Project structure

```
clif-vent-qi-score/
├── bundle_config.py            # site profile (+) definitions resolver; reads thresholds from the YAML
├── requirements.txt            # pinned Python compute stack
├── renv.lock                   # pinned R compute stack (renv::restore())
├── run_scores.sh / .ps1        # one-command site build, per-stage timings; --from / --fresh / --probe
├── run_pooled.sh / .ps1        # stage 7c + pooled stage 8 for the members of sites/pooled.json
├── show_results.sh / .ps1      # print the shareable, aggregate outputs for a site or pooled
├── definitions/                # SHARED, versioned clinical spec (committed)
│   ├── respiratory.yaml        #   verbatim from clif-dashboards @ pinned SHA
│   ├── platform.yaml           #   ditto
│   ├── composite.json          #   metric -> weight maps: score_core, score_full
│   └── SOURCE                  #   upstream SHA + sha256 per file
├── sites/                      # per-site access profiles (real ones gitignored)
│   ├── uchicago.example.json
│   └── pooled.example.json     #   members to stack in stage 7c
├── common/
│   ├── cohort_source.py        # stage-cache validation / cold upstream build / hard fail
│   ├── attribution.py          # ICU intervals by location_name; hour + day attribution
│   ├── covariates.py           # case-mix covariates anchored at intubation; small-cell suppression
│   ├── sofa_crrt.py            # classic SOFA via the vendored calculator, waterfall respiratory
│   ├── oxygenation.py          # P/F with S/F fallback for the respiratory subscore
│   └── scoring.py              # pooled rates, composite, floor flags
├── vendor/crrt_sofa/           # CRRT classic SOFA calculator, verbatim; SOURCE = commit + sha256
│   ├── code/                   #   sofa_calculator.py, outlier_utils.py
│   ├── config/                 #   outlier_config.json
│   └── LICENSE                 #   Apache-2.0
├── code/                       # 00..07c Python, each standalone-runnable
│   ├── 08_outcome_model.R      #   the outcome model (R)
│   └── caterpillar.R           #   caterpillar plot; sourced by stage 8 or run alone to redraw
├── tests/                      # synthetic frames, no CLIF data
│   ├── test_scoring.py         #   pooled rates, composite, grain, attribution
│   ├── test_cohort_source.py   #   stage-cache completeness + decision matching
│   ├── test_analysis_dataset.py#   the extended export cannot lose a column silently
│   ├── test_covariates.py      #   intubation-anchored covariates + suppression
│   ├── test_covariate_probe.py #   anonymized hospital-to-ICU summary
│   ├── test_pool_sites.py      #   pooling: prefixing, collisions, version refusal
│   ├── test_vendor_crrt.py     #   vendored SOFA calculator is byte-identical to its pin
│   ├── test_sofa_crrt.py       #   P/F time-unit workaround; pre-intubation GCS window
│   └── test_phi_gate.py        #   the shareable-output identifier gate actually fires
├── docs/METHODS.md             # definitions, deviations, known caveats
└── output/<site>/              # gitignored
    ├── intermediate_phi/       #   row-level — never shared
    ├── upstream_build/         #   stage cache from a cold build — never shared
    └── final_no_phi/           #   the panel, verification, model outputs, logs/
```

---

## Definitions & provenance

Every shareable artifact carries a provenance block: `site_id`, `code_version` (git SHA),
`clif_version`, `dataset_version`, `definition_version`, `upstream_sha`, `cohort_source` (which
tier produced the cohort), `period`, and `generated`.

**Reproducing a definition is asserted, not claimed.** Two independent checks:

- *In-process* (stage 2) — for every chunk, the upstream `metrics.compute()` runs alongside the
  fine-grain emission and our records are re-aggregated to episode grain and compared to its output.
  A mismatch raises immediately.
- *Cross-run* (stage 5) — the same comparison against the dashboard's **persisted**
  `episodes.parquet` and `proning.parquet` from a separate build, plus attribution conservation.

Results of the cross-run check are written to `final_no_phi/verification.json`, one row per
property, with the count of episodes differing on each.

See [`docs/METHODS.md`](docs/METHODS.md) for the full definitions, the deviation from upstream,
and the known caveats.

---

## Data safety

- `intermediate_phi/` holds one row per patient-hour and patient-day. It is never shared and never
  read with a display tool — only by code that prints aggregates.
- `final_no_phi/` is aggregate by construction, and stage 3 **enforces** it: a gate aborts the run
  if `hospitalization_id`, `patient_id`, `encounter_block` or `episode_id` appears in any shareable
  file. Every stage prints counts only; no raw CLIF table or patient-level value reaches stdout.
- `sites/*.json` carries local data paths and is gitignored; only `*.example.json` is committed.
- **For anything leaving the site** — manuscripts, posters, coordinating-centre uploads — use
  `unit_period_panel_anonymized.csv`, which carries `Unit 1..n` instead of `location_name`. The
  crosswalk stays in `intermediate_phi/panel/unit_crosswalk.csv`.

## Acknowledgements

Metric definitions from [clif-dashboards](https://github.com/AartikSarma/clif-dashboards)
(GPL-3.0-or-later). Built on [CLIF](https://clif-consortium.github.io/website/) and
[clifpy](https://github.com/Common-Longitudinal-ICU-data-Format/clifpy). The unit-day attribution
idiom follows `clif-ventilator-qi-dashboard`.
