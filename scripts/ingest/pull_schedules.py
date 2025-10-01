import argparse, pathlib
import nfl_data_py as nfl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season-start", type=int, default=2019)
    ap.add_argument("--season-end", type=int, default=2025)
    args = ap.parse_args()

    out_dir = pathlib.Path("data/raw"); out_dir.mkdir(parents=True, exist_ok=True)
    seasons = list(range(args.season_start, args.season_end + 1))
    df = nfl.import_schedules(seasons)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    path = out_dir / f"schedules_{seasons[0]}_{seasons[-1]}.parquet"
    df.to_parquet(path, index=False)
    print(f"[pull_schedules] wrote {len(df):,} rows -> {path}")

if __name__ == "__main__":
    main()
