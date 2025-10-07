import json, pathlib, pandas as pd
OUT_DIR = pathlib.Path("reports/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def dump(path_in, tag):
    with open(path_in, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    coefs = cfg["coefficients"].copy()
    const = coefs.pop("const", None)
    rows = [{"feature": k, "coef": float(v), "abs_coef": abs(float(v))} for k,v in coefs.items()]
    df = pd.DataFrame(rows).sort_values("abs_coef", ascending=False).reset_index(drop=True)
    if const is not None:
        df.loc[len(df)] = {"feature":"const","coef":float(const),"abs_coef":abs(float(const))}
    out_csv = OUT_DIR / f"ols_coefficients_{tag}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[feature_audit] {tag}: rmse={cfg.get('rmse'):.3f}, r2={cfg.get('r2'):.3f} -> {out_csv}")
    print(df.head(15).to_string(index=False))

def main():
    dump("reports/cv/margin_ols.json", "margin")
    dump("reports/cv/total_ols.json",  "total")

if __name__ == "__main__":
    main()
