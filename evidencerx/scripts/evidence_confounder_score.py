"""
EvidenceRx -- Phase 9 (score): turn the filled audit sheet into a measured
reliability figure for the Phase 6/7 headline numbers.

The point of the three buckets is the comparison between them:
  A_hype / B_contested  -- error rate among the pairs the analysis singles out
  C_random              -- error rate among ordinary pairs

If A and B are no worse than C, the interesting findings are as trustworthy as
the rest of the database. If they are worse, the headline numbers are partly an
artefact of the pairs that get surfaced, and the report must say so.

Reads : evidence_confounder_audit.xlsx  (Pairs sheet, judgements filled in)
Writes: evidence_confounder_results.csv

Usage:
    python evidence_confounder_score.py
"""

from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

BASE = Path(__file__).parent
XLSX = BASE / "evidence_confounder_audit.xlsx"
OUT = BASE / "evidence_confounder_results.csv"

CHECKS = ["disease_ok", "tier_ok", "direction_ok"]


def wilson(k, n, z=1.96):
    """Wilson interval -- honest at the small n an audit produces, unlike the
    normal approximation, which can run past 0 or 1."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    if not XLSX.exists():
        raise SystemExit(f"{XLSX.name} missing -- run "
                         f"evidence_confounder_audit.py first")
    wb = load_workbook(XLSX, read_only=True, data_only=True)
    ws = wb["Pairs"]
    head = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    df = pd.DataFrame(list(ws.iter_rows(min_row=2, values_only=True)),
                      columns=head)
    for c in CHECKS + ["verdict"]:
        df[c] = df[c].astype(str).str.strip().str.lower().replace("none", "")

    done = df[df.verdict != ""]
    print(f"Pairs judged: {len(done)} of {len(df)}")
    if done.empty:
        raise SystemExit("Nothing filled in yet -- fill the Pairs sheet first.")
    if len(done) < len(df):
        print(f"  ({len(df)-len(done)} still blank; scoring the completed ones)\n")

    # ---- per-check error rates, overall and by bucket ----
    print("=== Error rate per check ===")
    print(f"  {'check':16s} {'bucket':13s} {'n':>4s} {'bad':>4s} "
          f"{'err%':>6s}  95% CI")
    rows = []
    for chk in CHECKS:
        for bucket in ["ALL"] + sorted(done.bucket.unique()):
            sub = done if bucket == "ALL" else done[done.bucket == bucket]
            sub = sub[sub[chk].isin(["yes", "no", "unsure"])]
            n = len(sub)
            bad = int((sub[chk] == "no").sum())
            if n == 0:
                continue
            lo, hi = wilson(bad, n)
            print(f"  {chk:16s} {bucket:13s} {n:4d} {bad:4d} "
                  f"{100*bad/n:5.1f}%  [{100*lo:.0f}, {100*hi:.0f}]")
            rows.append({"check": chk, "bucket": bucket, "n": n, "bad": bad,
                         "err_pct": round(100 * bad / n, 1),
                         "ci_lo": round(100 * lo, 1), "ci_hi": round(100 * hi, 1)})
        print()

    # ---- verdicts ----
    print("=== Overall verdict ===")
    vt = pd.crosstab(done.bucket, done.verdict)
    print(vt.to_string())
    usable = done.verdict.isin(["ok", "minor"]).sum()
    lo, hi = wilson(usable, len(done))
    print(f"\n  usable (ok or minor): {usable}/{len(done)} "
          f"= {100*usable/len(done):.0f}%  [95% CI {100*lo:.0f}-{100*hi:.0f}]")

    # ---- the comparison that matters ----
    print("\n=== Are the headline pairs worse than ordinary ones? ===")
    rnd = done[done.bucket == "C_random"]
    tgt = done[done.bucket != "C_random"]
    if len(rnd) and len(tgt):
        r_bad = (rnd.verdict == "bad").mean() * 100
        t_bad = (tgt.verdict == "bad").mean() * 100
        print(f"  random control  : {r_bad:.0f}% bad (n={len(rnd)})")
        print(f"  surfaced pairs  : {t_bad:.0f}% bad (n={len(tgt)})")
        d = t_bad - r_bad
        if d > 15:
            print(f"  -> surfaced pairs are {d:.0f} pp worse. The Phase 6/7 "
                  f"highlights are partly an artefact; say so in the report "
                  f"and exclude the 'bad' pairs from the headline figures.")
        elif d < -15:
            print(f"  -> surfaced pairs are {-d:.0f} pp BETTER, which is odd; "
                  f"check the sampling before relying on it.")
        else:
            print(f"  -> comparable ({d:+.0f} pp). The highlighted findings "
                  f"are as reliable as the database generally.")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    bad_pairs = done[done.verdict == "bad"][["pair_id", "bucket", "spice",
                                             "disease", "notes"]]
    if len(bad_pairs):
        print(f"\n=== Pairs to exclude from the headline figures ({len(bad_pairs)}) ===")
        print(bad_pairs.to_string(index=False))
    print(f"\nWrote {OUT.name}")


if __name__ == "__main__":
    main()
