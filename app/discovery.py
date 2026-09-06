"""Finding trained checkpoints and the metrics that belong to them.

Deliberately free of Streamlit: both the demo and `scripts/prepare_space.py` need
this, and importing a Streamlit script from a plain process executes its entire
UI. Keeping discovery separate means the deploy script can ask "which model is
best?" without starting a web app.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SEARCH_DIRS = [ROOT / "checkpoints", ROOT / "results" / "checkpoints"]
REPORT_DIRS = [ROOT / "reports", ROOT / "results" / "reports"]

#: Checkpoints that exist for the pipeline's benefit, not for demonstrating.
#: `_phase1` is the frozen-backbone warm-up kept so a fine-tuning failure is
#: recoverable; `_rehearsal` is the deliberately under-trained subsample run.
#: Offering either invites showing an examiner the wrong model.
INTERNAL = ("_phase1", "_rehearsal")


def find_threshold(run_name: str) -> tuple[float, dict | None]:
    """The operating point tuned on validation, plus that run's test metrics."""
    for d in REPORT_DIRS:
        path = d / run_name / "metrics.json"
        if path.is_file():
            m = json.loads(path.read_text(encoding="utf-8"))
            return float(m["test"]["threshold"]), m
    return 0.5, None


def find_config(run_name: str) -> dict | None:
    """The run's saved Config, which records the input size among other things."""
    for d in REPORT_DIRS:
        path = d / run_name / "config.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def find_checkpoints() -> list[Path]:
    found = []
    for d in SEARCH_DIRS:
        if d.is_dir():
            found += [p for p in sorted(d.glob("*.keras"))
                      if not any(tag in p.stem for tag in INTERNAL)]
    return found


def catalogue() -> list[dict]:
    """Offerable checkpoints, best first.

    Ordered by the test AUROC each run recorded, not by where the file happens to
    sit - otherwise the default depends on directory search order and the demo can
    open on the weaker model.
    """
    rows = []
    for path in find_checkpoints():
        threshold, metrics = find_threshold(path.stem)
        rows.append({"path": path, "name": path.stem, "threshold": threshold,
                     "metrics": metrics,
                     "auroc": (metrics or {}).get("test", {}).get("auroc")})
    rows.sort(key=lambda r: (r["auroc"] is not None, r["auroc"] or 0.0,
                             r["path"].stat().st_mtime), reverse=True)
    return rows
