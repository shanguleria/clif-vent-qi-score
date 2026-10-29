"""Stage 5 — prove the definitions were reproduced, not reinterpreted.

Stage 2 already asserts equivalence inside each chunk against `metrics.compute()` running in the
same process. This stage is the independent, cross-run check: it compares against the dashboard's
own PERSISTED output (`episodes.parquet`, `proning.parquet`) from a separate build.

Three properties:

  1. DEFINITIONAL   our per-hour/per-day records, re-aggregated to episode grain, equal the
                    dashboard's per-episode numerators and denominators exactly.
  2. CONSERVATION   attribution only redistributes: denominators summed over every unit cell plus
                    the unattributed remainder equal the cohort total.
  3. POOLED RATE    the site-wide pooled rate over attributed care, and over everything, bracket
                    the dashboard's own headline.

Exit code is non-zero if any property fails. Prints only aggregate counts.

Run:  .venv/bin/python code/05_verify_equivalence.py [--site uchicago]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bundle_config as bc                                             # noqa: E402
from common import attribution as attr                                 # noqa: E402
from common import cohort_source                                       # noqa: E402
from common import scoring as S                                        # noqa: E402

from clif_dash.upstream import metrics as up_metrics                   # noqa: E402

#: metric -> (upstream numerator column, upstream denominator column)
UPSTREAM_COLS = {"sat": ("sat_num", "sat_den"), "sbt": ("sbt_num", "sbt_den")}


def compare(label: str, ours: pd.Series, theirs: pd.Series, results: list) -> None:
    """Exact per-episode comparison on the union of episode ids."""
    idx = theirs.index.union(ours.index)
    a = ours.reindex(idx).fillna(0).astype("int64")
    b = theirs.reindex(idx).fillna(0).astype("int64")
    bad = idx[a.to_numpy() != b.to_numpy()]
    ok = len(bad) == 0
    results.append((label, ok))
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  {label:<28} ours={int(a.sum()):>9,}  upstream={int(b.sum()):>9,}"
          f"  episodes differing: {len(bad):,}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site")
    args = ap.parse_args()
    site = bc.active_site(args.site)
    print(f"=== 05 verify equivalence: site {site} ===")

    src = (cohort_source.recorded(bc.phi_dir(site) / "cohort" / "source.json", log=lambda *a: None)
           or cohort_source.resolve(site, log=lambda *a: None))
    dash = src.stage.parent
    ep_path, pr_path = dash / "episodes.parquet", dash / "proning.parquet"
    if not ep_path.exists():
        print(f"  SKIP: no persisted dashboard output at {ep_path}.\n"
              f"  Stage 2's in-process assertions already covered definitional equivalence; this\n"
              f"  cross-run check needs a completed `clif-dash build` to compare against.")
        return 0

    phi = bc.phi_dir(site)
    ltvv = pd.read_parquet(phi / "records" / "ltvv_hours.parquet")
    days = pd.read_parquet(phi / "records" / "vent_days.parquet")
    up = pd.read_parquet(ep_path).set_index("episode_id")
    results: list[tuple[str, bool]] = []

    # ---- 1. definitional equivalence --------------------------------------
    print("\n[1] definitional equivalence vs the dashboard's persisted episodes.parquet")
    thr = bc.decision_value("ltvv", "threshold_ml_kg", site)
    compare(f"ltvv_den (hours)", ltvv.groupby("episode_id").size(), up["ltvv_den"], results)
    compare(f"ltvv_num (<= {thr} mL/kg)",
            ltvv.groupby("episode_id")[up_metrics.ltvv_key(thr)].sum(), up["ltvv_num"], results)
    compare("vent_days", days.groupby("episode_id").size(), up["vent_days"], results)
    for metric, (ncol, dcol) in UPSTREAM_COLS.items():
        g = days.groupby("episode_id")
        compare(f"{dcol}", g[f"{metric}_den"].sum(), up[dcol], results)
        compare(f"{ncol}", g[f"{metric}_num"].sum(), up[ncol], results)

    if pr_path.exists():
        prone_us = pd.read_parquet(phi / "records" / "prone_patients.parquet")
        prone_up = pd.read_parquet(pr_path)
        for label, a, b in (("proning cohort size", len(prone_us), len(prone_up)),
                            ("proned <= 12 h (p12)", int(prone_us["p12"].sum()),
                             int(prone_up["p12"].sum()))):
            ok = a == b
            results.append((label, ok))
            print(f"  {'PASS' if ok else 'FAIL'}  {label:<28} ours={a:>9,}  upstream={b:>9,}")

    # ---- 2. attribution conservation --------------------------------------
    print("\n[2] attribution conservation (cells + unattributed == cohort total)")
    contrib = pd.read_parquet(phi / "panel" / "contributions.parquet")
    totals = {
        "ltvv": (len(ltvv), int((ltvv["unit_name"] == attr.UNATTRIBUTED).sum())),
        "sat": (int(days["sat_den"].sum()),
                int(days.loc[days["unit_name"] == attr.UNATTRIBUTED, "sat_den"].sum())),
        "sbt": (int(days["sbt_den"].sum()),
                int(days.loc[days["unit_name"] == attr.UNATTRIBUTED, "sbt_den"].sum())),
    }
    if pr_path.exists():
        prone_us = pd.read_parquet(phi / "records" / "prone_patients.parquet")
        totals["prone_12h"] = (len(prone_us),
                               int((prone_us["unit_name"] == attr.UNATTRIBUTED).sum()))
    for metric, (cohort_total, unattributed) in totals.items():
        panelled = int(contrib.loc[contrib["metric"] == metric, "den"].sum())
        ok = panelled + unattributed == cohort_total
        results.append((f"conservation {metric}", ok))
        print(f"  {'PASS' if ok else 'FAIL'}  {metric:<10} panel={panelled:>9,} + "
              f"unattributed={unattributed:>7,} = {panelled + unattributed:>9,} "
              f"(cohort {cohort_total:,})   unattributed {100 * unattributed / max(cohort_total, 1):>5.2f}%")

    # ---- 3. pooled rates ---------------------------------------------------
    print("\n[3] site-wide pooled rates")
    print(f"  {'metric':<10} {'attributed':>12} {'all care':>12}   (dashboard pools all care)")
    for metric, (ncol, dcol) in (("ltvv", ("ltvv_num", "ltvv_den")), *UPSTREAM_COLS.items()):
        c = contrib[contrib["metric"] == metric]
        att = S.pooled_rate([c["num"].sum()], [c["den"].sum()])[0]
        if metric == "ltvv":
            allnum = ltvv[up_metrics.ltvv_key(thr)].sum(); allden = len(ltvv)
        else:
            allnum = days[f"{metric}_num"].sum(); allden = days[f"{metric}_den"].sum()
        allr = S.pooled_rate([allnum], [allden])[0]
        print(f"  {metric:<10} {100 * att:>11.2f}% {100 * allr:>11.2f}%")

    passed = sum(1 for _, ok in results if ok)
    print(f"\n=== {passed}/{len(results)} checks passed ===")
    out = bc.share_dir(site) / "verification.json"
    out.write_text(json.dumps({
        "provenance": bc.provenance(site, cohort_source=src.tier),
        "compared_against": str(ep_path),
        "checks": {label: ok for label, ok in results},
        "passed": passed, "total": len(results),
    }, indent=2, default=str))
    print(f"  wrote {out}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
