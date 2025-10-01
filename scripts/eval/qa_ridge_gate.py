import argparse, json, sys, pathlib

def load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)   # e.g., reports/cv/total_ols.json
    ap.add_argument("--candidate", required=True)  # e.g., reports/cv/total_ridgecv.json
    ap.add_argument("--tolerance", type=float, default=1.02)  # allow 2% worse at most
    args = ap.parse_args()

    base = load(args.baseline)
    cand = load(args.candidate)

    # Prefer CV where available
    base_rmse = base.get("rmse_cv", base.get("rmse"))
    cand_rmse = cand.get("rmse_cv", cand.get("rmse"))

    if base_rmse is None or cand_rmse is None:
        print("[qa_ridge_gate] Missing RMSE fields.", file=sys.stderr)
        sys.exit(2)

    max_allowed = float(base_rmse) * args.tolerance
    ok = cand_rmse <= max_allowed

    report = {
        "baseline_rmse": base_rmse,
        "candidate_rmse": cand_rmse,
        "tolerance": args.tolerance,
        "max_allowed": max_allowed,
        "pass": ok
    }
    outp = pathlib.Path("reports/diagnostics/qa_gate.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[qa_ridge_gate] base={base_rmse:.3f} cand={cand_rmse:.3f} tol x{args.tolerance:.3f} => {'PASS' if ok else 'FAIL'} -> {outp}")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
