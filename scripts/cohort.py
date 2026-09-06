"""Regenerate reports/cohort.csv from the DICOM headers.

    python scripts/cohort.py

Reads data/splits.csv, samples DICOMs, and writes the age / sex / view table that
demonstrates the dataset is adult.  Split out from prepare_data.py so the table
can be rebuilt in seconds without redoing CLAHE over 26,684 images.

Also writes reports/age_histogram.png, which is a better figure for the report
than three summary numbers.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from config import Config
from preprocess import dicom_metadata


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--sample", type=int, default=3000,
                    help="how many DICOM headers to read (0 = all)")
    Config.add_args(ap)
    args = ap.parse_args()
    cfg = Config.from_args(args)

    splits = Path(cfg.splits_csv)
    if not splits.exists():
        print(f"{splits} not found - run scripts/prepare_data.py first.")
        return 1
    df = pd.read_csv(splits)

    missing = [p for p in df.path.head(5) if not Path(p).exists()]
    if missing:
        print("the original DICOMs are gone, e.g.\n  " + missing[0]
              + "\nRe-run scripts/prepare_data.py --download to fetch them.")
        return 1

    sample = df if not args.sample else df.sample(min(args.sample, len(df)),
                                                  random_state=cfg.seed)
    print(f"reading {len(sample):,} DICOM headers ...")
    m = pd.DataFrame([dicom_metadata(p) for p in sample.path])

    ages = m.age.dropna()
    if ages.empty:
        seen = m.age_raw.dropna().unique()[:8]
        print("could not parse any PatientAge. Raw values seen: "
              + (", ".join(map(repr, seen)) if len(seen) else "tag absent entirely"))
        print("If the tag is absent, drop the age claim from the report and rely "
              "on the dataset's published description instead.")
        return 1

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame([{
        "n_sampled": len(m),
        "n_with_age": int(len(ages)),
        "age_min": round(float(ages.min()), 1),
        "age_q1": round(float(ages.quantile(0.25)), 1),
        "age_median": round(float(ages.median()), 1),
        "age_q3": round(float(ages.quantile(0.75)), 1),
        "age_max": round(float(ages.max()), 1),
        "pct_under_18": round(100.0 * (ages < 18).mean(), 2),
        "pct_male": round(100.0 * (m.sex == "M").mean(), 2),
        "view_PA": round(100.0 * (m.view == "PA").mean(), 2),
        "view_AP": round(100.0 * (m.view == "AP").mean(), 2),
    }])
    table.to_csv(out / "cohort.csv", index=False)
    print("\n" + table.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.hist(ages, bins=range(0, int(ages.max()) + 5, 5),
            color="#1b6ca8", edgecolor="white")
    ax.axvline(18, color="#c0392b", ls="--", lw=1.6, label="18 years")
    ax.set(xlabel="patient age (years)", ylabel="images",
           title=f"RSNA cohort age distribution (n={len(ages):,} sampled)")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out / "age_histogram.png", dpi=150)
    plt.close(fig)

    print(f"\nwrote {out / 'cohort.csv'}")
    print(f"wrote {out / 'age_histogram.png'}")
    print(f"\n{100 - table.pct_under_18.iloc[0]:.1f}% of sampled patients are adults.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
