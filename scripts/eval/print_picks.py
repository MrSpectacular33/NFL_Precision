#!/usr/bin/env python3
import argparse
import pandas as pd
from pathlib import Path
import numpy as np
import sys
import re

"""
Robust + diagnostic picks printer.

- Detects likely margin/total columns from many variants.
- If chosen columns look all zeros/NaN, prints a short diagnostics summary
  of other promising columns so you can fix pipeline mappings quickly.
"""

PREFERRED_MARGIN = [
    # most likely
    "pred_margin_ols", "pred_margin_ridgecv",
    # variants we've seen
    "pred_margin", "margin_ols", "margin_ridgecv", "margin",
    "pred_margin_home", "pred_margin_away",
    # very generic fallbacks
    "yhat_margin", "predicted_margin"
]

PREFERRED_TOTAL = [
    # most likely
    "pred_total_ols_home", "pred_total_ols", "pred_total_ridgecv",
    # variants we've seen
    "pred_total", "total_ols", "total_ridgecv", "total",
    # very generic fallbacks
    "yhat_total", "predicted_total"
]

TEAM_COL_PREFS = [
    ("team", "opponent"),
    ("home_team", "away_team"),
    ("home", "away"),
    ("team_home", "team_away"),
]

def best_present(df: pd.DataFrame, names: list[str]) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    # case-insensitive search
    lowmap = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lowmap:
            return lowmap[n.lower()]
    return None

def pick_team_cols(df: pd.DataFrame) -> tuple[str | None, str | None]:
    for a,b in TEAM_COL_PREFS:
        if a in df.columns and b in df.columns:
            return a,b
    # last-ditch: look for any two columns with "team" & "opp" substrings
    cols = [c for c in df.columns]
    a = next((c for c in cols if re.search(r'home|team', c, re.I)), None)
    b = next((c for c in cols if re.search(r'away|opp', c, re.I)), None)
    return a,b

def is_all_zero_or_na(s: pd.Series) -> bool:
    if s is None:
        return True
    try:
        x = pd.to_numeric(s, errors="coerce")
    except Exception:
        return True
    return bool((x.fillna(0.0) == 0.0).all())

def summarize_candidates(df: pd.DataFrame, kind: str) -> list[str]:
    """Return a few promising column names for debugging output."""
    pats = {
        "margin": ("margin|spread|pred.*marg|marg.*pred", PREFERRED_MARGIN),
        "total":  ("total|pred.*tot|tot.*pred", PREFERRED_TOTAL),
    }
    regex_pat, preferred = pats[kind]
    num_cols = []
    for c in df.columns:
        if re.search(regex_pat, c, re.I):
            try:
                x = pd.to_numeric(df[c], errors="coerce")
                nz = int((x.fillna(0.0) != 0.0).sum())
                num_cols.append((c, nz))
            except Exception:
                pass
    # ensure preferred names get shown first if present
    name2score = {c:nz for c,nz in num_cols}
    ordered = []
    for p in preferred:
        if p in name2score:
            ordered.append((p, name2score[p]))
    for c,nz in num_cols:
        if c not in [x for x,_ in ordered]:
            ordered.append((c,nz))
    # top 6
    return [f"{c} (nonzero_rows={nz})" for c,nz in ordered[:6]]

def select_or_fallback(df: pd.DataFrame, pref: list[str], kind: str) -> str | None:
    cand = best_present(df, pref)
    if cand and not is_all_zero_or_na(df[cand]):
        return cand

    # heuristic fallback: any numeric column that matches kind words
    pat = r'margin|spread' if kind == "margin" else r'total'
    candidates = []
    for c in df.columns:
        if re.search(pat, c, re.I):
            try:
                x = pd.to_numeric(df[c], errors="coerce")
                if (~x.isna()).sum() > 0 and not is_all_zero_or_na(x):
                    candidates.append(c)
            except Exception:
                pass
    if candidates:
        # prefer shorter, more exact names
        candidates.sort(key=lambda n: (len(n), n))
        return candidates[0]
    return cand  # might be None or zero-only; diagnostics will trigger

def fmt_num(x) -> str:
    try:
        if pd.isna(x):
            return "—"
        xn = float(x)
        # avoid printing -0.00
        if abs(xn) < 1e-12:
            xn = 0.0
        return f"{xn:.2f}"
    except Exception:
        return "—"

def main():
    ap = argparse.ArgumentParser(description="Print picks from predictions file (robust)")
    ap.add_argument("--file", required=True, help="Path to predictions CSV or Parquet")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # choose columns
    margin_col = select_or_fallback(df, PREFERRED_MARGIN, "margin")
    total_col  = select_or_fallback(df, PREFERRED_TOTAL, "total")
    gid_col    = "game_id" if "game_id" in df.columns else None
    team_col, opp_col = pick_team_cols(df)

    # diagnostics if needed
    diag_msgs = []
    if margin_col is None or is_all_zero_or_na(df.get(margin_col, pd.Series(dtype=float))):
        diag = summarize_candidates(df, "margin")
        diag_msgs.append("Margin column uncertain or zero-only. Candidates: " + (", ".join(diag) if diag else "(none)"))
    if total_col is None or is_all_zero_or_na(df.get(total_col, pd.Series(dtype=float))):
        diag = summarize_candidates(df, "total")
        diag_msgs.append("Total column uncertain or zero-only. Candidates: " + (", ".join(diag) if diag else "(none)"))

    print("[picks]")
    for _, r in df.iterrows():
        if team_col and opp_col:
            label = f"{r[team_col]} vs {r[opp_col]}"
        elif "home_team" in df.columns and "away_team" in df.columns:
            label = f"{r['home_team']} vs {r['away_team']}"
        elif "team" in df.columns and "opponent" in df.columns:
            label = f"{r['team']} vs {r['opponent']}"
        else:
            label = "MATCHUP"

        m = fmt_num(r[margin_col]) if margin_col in df.columns else "—"
        t = fmt_num(r[total_col])  if total_col  in df.columns else "—"
        gid = f"{r[gid_col]}: " if gid_col else ""
        print(f"  {gid}{label} | margin(OLS)={m}  total(OLS)={t}")

    # print diagnostics at end (once)
    if diag_msgs:
        print("\n[diagnostics]")
        for msg in diag_msgs:
            print("  - " + msg)

if __name__ == "__main__":
    main()
