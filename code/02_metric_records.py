"""Stage 2 — emit per-hour, per-vent-day and per-patient metric records, attributed to an ICU.

clif-dashboards' `metrics.compute()` encodes all four definitions but collapses them to per-EPISODE
totals, which cannot be split across units. This stage keeps the definitions and changes only the
grain:

  ltvv_hours.parquet    one row per eligible IMV scaffold hour   -> unit covering that hour
  vent_days.parquet     one row per vent-day, SAT+SBT flags      -> unit the day began in
  prone_patients.parquet one row per PROSEVA-eligible patient    -> unit at enrollment

Every predicate is the upstream's, imported unchanged. For each chunk we ALSO run the upstream
`metrics.compute()` itself and assert our fine-grain records re-aggregate to its per-episode
numerators and denominators exactly. That turns "we reproduced the definition" from a claim into an
in-process assertion, and it costs one extra pass over data already in memory.

The per-chunk context mirrors `respiratory/build.py::phase_b_chunk` so each encounter sees exactly
the rows the dashboard's own build showed it (same staged chunk, same outlier handling, same
whole-table FiO2 percent test).

Prints only aggregate counts.

Run:  .venv/bin/python code/02_metric_records.py [--site uchicago] [--chunks 0,1,2]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

# numpy 2.5 warns about the generic timedelta unit inside pandas' Timestamp arithmetic. The call
# sites here are the upstream's own day-boundary idiom (d0 + Timedelta(days=1)); the unit IS
# explicit, so the warning is a library-internal artefact, not a correctness signal.
warnings.filterwarnings("ignore", message=".*generic.*unit for NumPy timedelta.*")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bundle_config as bc                                             # noqa: E402
from common import attribution as attr                                 # noqa: E402
from common import cohort_source                                       # noqa: E402

from clif_dash import stage                                            # noqa: E402
from clif_dash.upstream import clif_io, metrics, prep, proning         # noqa: E402

#: Staged phase-B tables the four measures read. Mirrors respiratory/build.py TABLES.
TABLES = ("vitals", "labs", "position", "patient_assessments",
          "medication_admin_continuous", "code_status")

_noop = lambda *_a, **_k: None


def fio2_recorded_as_percent(cfg) -> bool:
    """Whole-table FiO2 unit test, as the upstream computes it (build.py::site_statistics).

    Must be a WHOLE-table statistic: deciding per chunk would rescale some chunks and not others.
    """
    con = stage.connect()
    try:
        mean = con.execute(
            f"SELECT avg(fio2_set) FROM read_parquet('{cfg.table_path('respiratory_support')}')"
        ).fetchone()[0]
    finally:
        con.close()
    return bool(mean is not None and mean > 1)


# --------------------------------------------------------------------------- #
# Fine-grain emission — same predicates as metrics.compute(), finer output
# --------------------------------------------------------------------------- #
def emit_records(episodes, ctx, tables, D, tz):
    """Return (ltvv_hour_rows, vent_day_rows) for one chunk's episodes."""
    dm = D["metrics"]
    dval = lambda m, i: next(d["value"] for d in dm[m]["decisions"] if d["id"] == i)

    vt_modes = set(dval("ltvv", "modes"))
    vt_thr = dval("ltvv", "threshold_ml_kg")
    vt_thrs = sorted(set(next((d["value"] for d in dm["ltvv"]["decisions"]
                               if d["id"] == "thresholds_selectable"), []) + [vt_thr]))
    sbt_elig = dval("sbt", "eligibility")
    ps_max = dval("sbt", "ps_max")
    peep_trial = dval("sbt", "peep_max_trial")

    heights, weights = ctx["heights"], ctx["weights"]
    iv, pa_by, cs_by = ctx["iv"], ctx["pa_by"], ctx["cs_by"]
    w, runs, enc = ctx["w"], ctx["runs"], ctx["enc"]
    w_by = {k: g for k, g in w.groupby("encounter_block")}
    runs_by = {k: g for k, g in runs.groupby("encounter_block")}
    enc_i = enc.set_index("encounter_block")

    ltvv_rows, day_rows = [], []
    for e in episodes.itertuples():
        blk = e.encounter_block
        E = enc_i.loc[blk]
        g = w_by[blk]
        # Identical slice to metrics.compute(): IMV rows inside [start, end)
        inep = g[(g["recorded_dttm"] >= e.start) & (g["recorded_dttm"] < e.end) & g["is_imv"]]
        iv_b = iv.get(blk)
        pa_b = pa_by.get(blk)
        h = heights.loc[blk] if blk in heights.index else None
        height = h["height_cm"] if h is not None else np.nan
        p_bw = metrics.pbw(height, E["sex_category"])

        # ---- LTVV: one row per eligible IMV hour --------------------------
        hrs = inep[inep["is_scaffold"] & inep["mode_category"].isin(vt_modes)
                   & inep["tidal_volume_set"].notna()]
        if not np.isnan(p_bw) and len(hrs):
            ratio = (hrs["tidal_volume_set"] / p_bw).to_numpy()
            rec = hrs["recorded_dttm"].to_numpy()
            row = {"episode_id": e.episode_id, "encounter_block": blk}
            for i in range(len(hrs)):
                r = dict(row, recorded_dttm=rec[i], vt_pbw=float(ratio[i]))
                for t in vt_thrs:
                    r[metrics.ltvv_key(t)] = int(ratio[i] <= t + 1e-9)
                ltvv_rows.append(r)

        # ---- SAT / SBT: one row per vent-day ------------------------------
        sed_on = prep.on_intervals(iv_b, "sedative")
        par_on = prep.on_intervals(iv_b, "paralytic")
        my_runs = runs_by[blk][runs_by[blk]["episode_id"] == e.episode_id]
        sc = inep[inep["is_scaffold"]].copy()
        if sc.empty:
            continue
        sc["day"] = sc["recorded_dttm"].dt.tz_convert(tz).dt.date
        for day, dg in sc.groupby("day"):
            if len(dg) < 6:                       # vent_day = calendar_day_6h
                continue
            d0 = pd.Timestamp(day, tz=tz).tz_convert("UTC")
            d1 = d0 + pd.Timedelta(days=1)
            code = metrics.code_status_at(cs_by, E["patient_id"], d1)
            comfort = code.startswith("Comfort")
            on_sed = prep.overlaps(sed_on, d0, d1)
            on_par = prep.overlaps(par_on, d0, d1)
            docs = pa_b[(pa_b["recorded_dttm"] >= d0) & (pa_b["recorded_dttm"] < d1)] if pa_b is not None else None

            sat_den = int(on_sed and not on_par and not comfort)
            sat_doc = sat_pheno = 0
            if sat_den:
                sat_doc = int(bool(docs is not None and docs["cat_l"].eq("sat_delivery_pass_fail").any()))
                sat_pheno = int(metrics._sat_phenotype(sed_on, my_runs, d0, d1))

            ok = dg[(dg["fio2_set"] <= sbt_elig["fio2_max"]) & (dg["peep_set"] <= sbt_elig["peep_max"])]
            elig = False
            for t in ok["recorded_dttm"]:
                if not prep.covers(par_on, t) and prep.nee_at(iv_b, t) <= sbt_elig["nee_max"]:
                    elig = True
                    break
            sbt_den = int(elig and not comfort)
            sbt_doc = sbt_pheno = 0
            if sbt_den:
                sbt_doc = int(bool(docs is not None and docs["cat_l"].eq("sbt_delivery_pass_fail").any()))
                sbt_pheno = int(metrics._sbt_phenotype(g, my_runs, d0, d1, ps_max, peep_trial))

            day_rows.append({
                "episode_id": e.episode_id, "encounter_block": blk, "day": day,
                "first_hour": dg["recorded_dttm"].min(), "imv_hours": int(len(dg)),
                "sat_den": sat_den, "sat_doc": sat_doc, "sat_pheno": sat_pheno,
                "sat_num": int(sat_den and (sat_doc or sat_pheno)),
                "sbt_den": sbt_den, "sbt_doc": sbt_doc, "sbt_pheno": sbt_pheno,
                "sbt_num": int(sbt_den and (sbt_doc or sbt_pheno)),
            })
    return pd.DataFrame(ltvv_rows), pd.DataFrame(day_rows)


def assert_equivalent(ltvv, days, episode_metrics, vt_thr, chunk):
    """Our fine-grain records must re-aggregate to the upstream's per-episode num/den exactly."""
    ref = episode_metrics.set_index("episode_id")
    problems = []

    got_den = ltvv.groupby("episode_id").size() if len(ltvv) else pd.Series(dtype=int)
    got_num = (ltvv.groupby("episode_id")[metrics.ltvv_key(vt_thr)].sum()
               if len(ltvv) else pd.Series(dtype=int))
    for name, got, refcol in (("ltvv_den", got_den, "ltvv_den"), ("ltvv_num", got_num, "ltvv_num")):
        exp = ref[refcol].fillna(0).astype(int)
        mine = got.reindex(exp.index).fillna(0).astype(int)
        bad = exp[exp != mine]
        if len(bad):
            problems.append(f"{name}: {len(bad)} episodes differ (e.g. {bad.index[0]}: "
                            f"upstream={exp.loc[bad.index[0]]} ours={mine.loc[bad.index[0]]})")

    if len(days):
        agg = days.groupby("episode_id")[["sat_den", "sat_num", "sat_doc", "sat_pheno",
                                          "sbt_den", "sbt_num", "sbt_doc", "sbt_pheno"]].sum()
    else:
        agg = pd.DataFrame(columns=["sat_den", "sat_num", "sat_doc", "sat_pheno",
                                    "sbt_den", "sbt_num", "sbt_doc", "sbt_pheno"], dtype=int)
    for col in ("sat_den", "sat_num", "sbt_den", "sbt_num"):
        exp = ref[col].fillna(0).astype(int)
        mine = agg[col].reindex(exp.index).fillna(0).astype(int)
        bad = exp[exp != mine]
        if len(bad):
            problems.append(f"{col}: {len(bad)} episodes differ (e.g. {bad.index[0]}: "
                            f"upstream={exp.loc[bad.index[0]]} ours={mine.loc[bad.index[0]]})")

    # vent_days is a count of days with >= 6 IMV hours; our day rows are exactly those
    exp = ref["vent_days"].fillna(0).astype(int)
    mine = (days.groupby("episode_id").size() if len(days) else pd.Series(dtype=int)
            ).reindex(exp.index).fillna(0).astype(int)
    bad = exp[exp != mine]
    if len(bad):
        problems.append(f"vent_days: {len(bad)} episodes differ (e.g. {bad.index[0]}: "
                        f"upstream={exp.loc[bad.index[0]]} ours={mine.loc[bad.index[0]]})")

    if problems:
        raise AssertionError(f"chunk {chunk} definitional drift:\n  " + "\n  ".join(problems))


# --------------------------------------------------------------------------- #
def process_chunk(cfg, src, chunk, statistics, tz, D, site):
    work = src.stage
    chunk_blocks = pd.read_parquet(work / "chunks_b.parquet").query("chunk == @chunk")["encounter_block"]
    in_chunk = lambda f: f[f["encounter_block"].isin(chunk_blocks)].reset_index(drop=True)
    mapping = in_chunk(pd.read_parquet(work / "mapping.parquet"))
    adt = in_chunk(pd.read_parquet(work / "adt.parquet"))
    encounters = in_chunk(pd.read_parquet(work / "encounters.parquet"))
    episodes = in_chunk(pd.read_parquet(work / "episodes_a.parquet"))
    runs = in_chunk(pd.read_parquet(work / "runs_a.parquet"))
    waterfall = pd.read_parquet(work / "waterfall" / f"chunk_{chunk}.parquet")
    waterfall = waterfall.drop(columns=["waterfall_row", "phase_a_chunk", "chunk"]).reset_index(drop=True)

    tables = {t: stage.load_staged_chunk(cfg, "phase_b", t, chunk) for t in TABLES}
    clif_io.apply_outliers(tables, log=_noop, fio2_percent=statistics["fio2_recorded_as_percent"])

    ctx = dict(T=tables, enc=encounters, mapping=mapping, adt=adt, w=waterfall,
               runs=runs, ep=episodes, cci=None)
    # Running the upstream function populates ctx (heights/weights/iv/pa_by/cs_by/labs_by/vit_by)
    # AND gives the reference per-episode totals our records are checked against.
    episode_metrics = metrics.compute(tables, ctx, D, tz, log=_noop, fit_mortality=False)

    ltvv, days = emit_records(episodes, ctx, tables, D, tz)
    vt_thr = next(d["value"] for d in D["metrics"]["ltvv"]["decisions"] if d["id"] == "threshold_ml_kg")
    assert_equivalent(ltvv, days, episode_metrics, vt_thr, chunk)

    ctx["ep"] = episode_metrics
    prone, prone_funnel, _ = proning.compute(tables, ctx, D, log=_noop, expected=False)
    return ltvv, days, prone, prone_funnel, episode_metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site")
    ap.add_argument("--chunks", help="comma-separated subset, for debugging")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--require-stage", action="store_true")
    args = ap.parse_args()
    site = bc.active_site(args.site)
    prof = bc.load_profile(site)
    tz, period = prof["timezone"], prof.get("period", "quarter")

    print(f"=== 02 metric records: site {site} ===")
    src = cohort_source.resolve(site, rebuild=args.rebuild, require_stage=args.require_stage)
    cfg = src.cfg(site)
    D = bc.decisions(site)

    statistics = {"fio2_recorded_as_percent": fio2_recorded_as_percent(cfg)}
    print(f"  fio2 recorded as percent : {statistics['fio2_recorded_as_percent']}")

    chunks = ([int(c) for c in args.chunks.split(",")] if args.chunks
              else sorted(int(p.stem.split("_")[1]) for p in src.waterfall_chunks()))
    print(f"  chunks to process        : {len(chunks)}")

    L, Dy, P, ref = [], [], [], []
    funnel = None
    for i, chunk in enumerate(chunks, 1):
        ltvv, days, prone, pf, em = process_chunk(cfg, src, chunk, statistics, tz, D, site)
        L.append(ltvv); Dy.append(days); ref.append(em)
        if len(prone):
            P.append(prone)
        funnel = pf if funnel is None else [
            {**a, "n": a["n"] + b["n"]} for a, b in zip(funnel, pf)]
        print(f"    [{i:>2}/{len(chunks)}] chunk {chunk:<3} "
              f"ltvv_hours={len(ltvv):>7,}  vent_days={len(days):>6,}  prone={len(prone):>4,}", flush=True)

    ltvv = pd.concat(L, ignore_index=True) if any(len(x) for x in L) else pd.DataFrame()
    days = pd.concat(Dy, ignore_index=True) if any(len(x) for x in Dy) else pd.DataFrame()
    prone = pd.concat(P, ignore_index=True) if P else pd.DataFrame()
    episode_ref = pd.concat(ref, ignore_index=True)

    # ---- attribute every record to the ICU that delivered the care -----------
    icu = pd.read_parquet(bc.phi_dir(site) / "cohort" / "icu_intervals.parquet")

    ltvv["t"] = attr.hour_midpoints(ltvv["recorded_dttm"])
    ltvv = attr.unit_at(ltvv, icu)
    ltvv["period"] = attr.period_label(ltvv["recorded_dttm"], tz, period)

    days["t"] = attr.hour_midpoints(days["first_hour"])
    days = attr.unit_at(days, icu)
    days["period"] = attr.period_label(days["first_hour"], tz, period)

    if len(prone):
        prone["t"] = pd.to_datetime(prone["t_enroll"], utc=True)
        prone = attr.unit_at(prone, icu)
        prone["period"] = attr.period_label(prone["t_enroll"], tz, period)

    out = bc.phi_dir(site) / "records"
    out.mkdir(parents=True, exist_ok=True)
    ltvv.drop(columns=["t"]).to_parquet(out / "ltvv_hours.parquet", index=False)
    days.drop(columns=["t"]).to_parquet(out / "vent_days.parquet", index=False)
    if len(prone):
        prone.drop(columns=["t"]).to_parquet(out / "prone_patients.parquet", index=False)
    episode_ref.to_parquet(out / "episode_reference.parquet", index=False)

    def where_unattributed(df, tcol, adt):
        """Aggregate location_category of records that fell in no named ICU stay."""
        un = df[df["unit_name"] == attr.UNATTRIBUTED]
        if not len(un):
            return {}
        allloc = adt.copy()
        allloc["unit_name"] = allloc["location_category"].astype("string").str.lower()
        allloc["unit_type"] = allloc["location_type"].astype("string")
        allloc["encounter_block"] = attr.norm_block(allloc["encounter_block"])
        allloc = allloc[allloc["encounter_block"].notna()]
        allloc["in_dttm"] = attr.as_ns(allloc["in_dttm"])
        allloc["out_dttm"] = attr.as_ns(allloc["out_dttm"]).fillna(attr.FAR_FUTURE)
        allloc = allloc[allloc["out_dttm"] > allloc["in_dttm"]].sort_values(
            ["encounter_block", "in_dttm"])
        probe = un[["encounter_block"]].copy()
        probe["t"] = attr.hour_midpoints(un[tcol])
        got = attr.unit_at(probe, allloc)
        return {str(k): int(v) for k, v in got["unit_name"].value_counts().items()}

    def unattr(df, label):
        if not len(df):
            return 0.0
        share = float((df["unit_name"] == attr.UNATTRIBUTED).mean())
        print(f"  {label:<12} {len(df):>9,} records   unattributed {share * 100:>5.2f}%")
        return share

    print(f"\n  definitional equivalence : PASS on all {len(chunks)} chunks "
          f"(re-aggregated to episode grain == upstream metrics.compute)")
    print(f"\n[records] attributed to {icu['unit_name'].nunique()} ICU units")
    shares = {
        "ltvv_hours": unattr(ltvv, "ltvv_hours"),
        "vent_days": unattr(days, "vent_days"),
        "prone": unattr(prone, "prone"),
    }
    adt_all = src.read("adt")
    placement = {
        "ltvv_hours": where_unattributed(ltvv, "recorded_dttm", adt_all),
        "vent_days": where_unattributed(days, "first_hour", adt_all),
    }
    print("\n[unattributed] where those records actually were (location_category):")
    for label, counts in placement.items():
        if counts:
            total = sum(counts.values())
            parts = ", ".join(f"{k} {100 * v / total:.0f}%" for k, v in
                              sorted(counts.items(), key=lambda kv: -kv[1]))
            print(f"  {label:<12} {parts}")

    (out / "manifest.json").write_text(json.dumps({
        "provenance": bc.provenance(site, cohort_source=src.tier),
        "counts": {"ltvv_hours": int(len(ltvv)), "vent_days": int(len(days)),
                   "prone_patients": int(len(prone)), "episodes": int(len(episode_ref))},
        "unattributed_share": shares,
        "unattributed_placement": placement,
        "proning_funnel": funnel,
        "fio2_recorded_as_percent": statistics["fio2_recorded_as_percent"],
    }, indent=2, default=str))
    print(f"\n  wrote {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
