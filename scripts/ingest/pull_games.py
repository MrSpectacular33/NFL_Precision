import argparse, pathlib
import pandas as pd
import nfl_data_py as nfl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season-start", type=int, default=2019)
    ap.add_argument("--season-end", type=int, default=2025)
    args = ap.parse_args()

    out_dir = pathlib.Path("data/raw"); out_dir.mkdir(parents=True, exist_ok=True)

    seasons = list(range(args.season_start, args.season_end + 1))
    parts = []
    fetched, skipped = [], []

    # Try each season separately to avoid failing the whole run if a parquet is missing
    for y in seasons:
        try:
            df_y = nfl.import_weekly_data([y])
            if df_y is None or len(df_y) == 0:
                skipped.append((y, "empty"))
                continue
            # Normalize columns
            df_y.columns = [c.strip().lower().replace(" ", "_") for c in df_y.columns]
            parts.append(df_y)
            fetched.append(y)
        except Exception as e:
            skipped.append((y, str(e)))

    if not parts:
        raise SystemExit(f"No weekly data available for any season in {seasons}. Skipped={skipped}")

    df = pd.concat(parts, ignore_index=True)
    path = out_dir / f"weekly_stats_{min(fetched)}_{max(fetched)}.parquet"
    df.to_parquet(path, index=False)
    print(f"[pull_games] seasons fetched: {fetched}")
    if skipped:
        print(f"[pull_games] seasons skipped: {skipped}")
    print(f"[pull_games] wrote {len(df):,} rows -> {path}")

if __name__ == "__main__":
    main()
