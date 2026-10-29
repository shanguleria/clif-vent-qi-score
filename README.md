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

Produce a **unit x period panel of composite quality scores** suitable for use as an exposure in a
downstream outcome model, plus a per-patient exposure assignment with leave-one-out variants.

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

`score_core` is the **primary exposure**; `score_full` is a prespecified **sensitivity analysis**.

```
score_core = 100 x mean(ltvv_rate, sat_rate, sbt_rate)
score_full = 100 x mean(ltvv_rate, sat_rate, sbt_rate, prone_12h_rate)
```

Proning is excluded from the primary score because its denominator is a *severity-defined cohort*
roughly two orders of magnitude smaller than the vent-day denominators. At UChicago, 2018-2024:
682 PROSEVA-eligible patients against ~44,400 SAT-eligible vent-days. At equal weight, proning
would contribute a quarter of the score while being estimated with ~1/100th the precision,
dominating the composite's noise rather than its signal. Empirically, the proning floor flags
**139 of 195** unit-quarter cells, against **8 of 195** for all of `score_core`'s components
combined. All four metrics are computed and stored regardless; only score membership differs, and
that is a one-line config edit in `definitions/composite.json`.

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

> **`location_name` is optional in CLIF 2.1** and is free text with no permissible-value list, so it
> is site-specific and may be absent entirely. Stage 0 checks for it and fails with a clear message
> rather than silently falling back. At UChicago it is populated on 100% of ICU rows and resolves to
> 7 units — two of which (`N09S`, `N09N`) share `location_type = medical_icu` and would have been
> collapsed had the pipeline keyed on the categorical instead.

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
  any ICU can be scored for. At UChicago that is 10.7% of IMV hours and 16.3% of vent-days, which
  sit on the ward (~44%), in procedural areas (~16-25%) and in the ED (~23-34%).

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
| `period` | `quarter` (default) or `month` |
| `min_denominator` | Per-metric floor below which a cell is **flagged**, not dropped |
| `unit_include` / `unit_collapse` | Restrict to, or pool, specific `location_name` values |

`definitions/` is also what the upstream builder reads: the generated site YAML points
`SiteConfig.decisions_dir` at this folder, so there is no second copy of a threshold to drift from.
Stage 0 re-diffs `definitions/` against the pinned upstream commit and fails loudly on divergence.

### Where outputs land

| Path | Contents | Shareable? |
|---|---|---|
| `output/<site>/intermediate_phi/` | Row-level records, contributions, per-patient exposure, unit crosswalk | **Never** |
| `output/<site>/upstream_build/` | Stage cache written by a cold build (~1 GB) | **Never** |
| `output/<site>/final_no_phi/unit_period_panel.csv` | The panel — aggregate by construction | Yes, site-internal |
| `output/<site>/final_no_phi/unit_period_panel_anonymized.csv` | Same, with `location_name` stripped (`Unit 1..n` only) | **Yes — use this externally** |
| `output/<site>/final_no_phi/verification.json` | Equivalence check results | Yes |
| `output/<site>/run_timings.csv` | Per-stage wall clock + git SHA | Yes |

---

## Prerequisites

Python **3.12** (matching the upstream's `requires-python >= 3.12`).

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The compute stack is **pinned exactly** — `clif-dashboards` at a commit, plus `clifpy==2.1.0`,
`pandas==2.3.3`, `numpy==2.5.4`, `pyarrow==26.0.0`, `duckdb==1.5.6` — because every site must
compute identical numbers. The `clif-dashboards` pin is a *definitional* pin, not merely a
dependency pin: bumping it can change what the metrics mean, so re-run stage 5 after any bump.

---

## Running the pipeline

**macOS / Linux**

```bash
./run_scores.sh --site uchicago
./run_scores.sh --site uchicago --rebuild        # ignore any cache, build from raw CLIF
./run_scores.sh --site uchicago --require-stage  # fail instead of cold-building (CI)
```

**Windows (PowerShell)**

```powershell
.\run_scores.ps1 -Site uchicago
.\run_scores.ps1 -Site uchicago -Rebuild
.\run_scores.ps1 -Site uchicago -RequireStage
```

Tests (synthetic frames only, no CLIF data):

```bash
.venv/bin/python -m pytest tests/ -q
```

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

**Verified at UChicago.** A cached run and a `--rebuild` run produce **identical panels** — all 195
cells x 28 columns, checked by `code/06_compare_tiers.py`. Measured cost: 12m27s cached vs 17m19s
cold, of which the upstream build (encounter stitching + the clifpy waterfall over 4,034,216
respiratory_support rows) is 5m15s and writes 1.1 GB.

---

## Pipeline steps

| Step | Script | Output |
|---|---|---|
| 0 | `code/00_probe_units.py` | Aggregate-only probe: definition hashes, composite coverage, config-key coverage, ICU unit inventory, panel shape |
| 1 | `code/01_build_cohort.py` | `cohort/` — episodes with period labels, IMV runs, encounters, ICU intervals keyed by `location_name` |
| 2 | `code/02_metric_records.py` | `records/` — `ltvv_hours`, `vent_days`, `prone_patients`, each attributed to a unit; asserts equivalence per chunk |
| 3 | `code/03_build_panel.py` | `panel/contributions.parquet` + `final_no_phi/unit_period_panel.{csv,parquet}` |
| 4 | `code/04_exposure_assignment.py` | `exposure/` — per-patient exposure, long and primary-cell, with leave-one-out |
| 5 | `code/05_verify_equivalence.py` | `final_no_phi/verification.json` — cross-run equivalence vs the dashboard's persisted output |
| 6 | `code/06_compare_tiers.py` | `tier_comparison/` — on demand, proves a cold build reproduces the cached-tier panel exactly |

Stages 0-5 run on every build via the runner. Stage 6 is a verification tool invoked manually
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
├── requirements.txt            # pinned compute stack
├── run_scores.sh / .ps1        # one-command build, per-stage timings
├── definitions/                # SHARED, versioned clinical spec (committed)
│   ├── respiratory.yaml        #   verbatim from clif-dashboards @ pinned SHA
│   ├── platform.yaml           #   ditto
│   ├── composite.json          #   metric -> weight maps: score_core, score_full
│   └── SOURCE                  #   upstream SHA + sha256 per file
├── sites/                      # per-site access profiles (real ones gitignored)
│   └── uchicago.example.json
├── common/
│   ├── cohort_source.py        # stage-cache validation / cold upstream build / hard fail
│   ├── attribution.py          # ICU intervals by location_name; hour + day attribution
│   └── scoring.py              # pooled rates, composite, leave-one-out
├── code/                       # 00..06, each standalone-runnable
├── tests/                      # synthetic frames, no CLIF data
│   ├── test_scoring.py         #   pooled rates, composite, leave-one-out, attribution
│   ├── test_cohort_source.py   #   stage-cache completeness + decision matching
│   └── test_phi_gate.py        #   the shareable-output identifier gate actually fires
├── docs/METHODS.md             # definitions, deviations, known caveats
└── output/<site>/              # gitignored
    ├── intermediate_phi/       #   row-level — never shared
    ├── upstream_build/         #   stage cache from a cold build — never shared
    └── final_no_phi/           #   the panel + verification
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

At UChicago both pass: 13/13 checks, **zero episodes differing** on any of `ltvv_num/den`,
`sat_num/den`, `sbt_num/den`, `vent_days`, proning cohort size, or `p12`.

See [`docs/METHODS.md`](docs/METHODS.md) for the full definitions, the deviations from upstream,
and the known caveats (notably: at UChicago SAT is effectively a *phenotype* measure, since only
254 `sat_delivery_pass_fail` rows exist in seven years).

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
