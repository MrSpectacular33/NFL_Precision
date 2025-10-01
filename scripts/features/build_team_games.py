import argparse, pathlib, json
import pandas as pd

REQ_SCHED = ['game_id','season','week','home_team','away_team','home_score','away_score']

DIV = {
    "ARI":"NFCW","SEA":"NFCW","SF":"NFCW","LAR":"NFCW",
    "DAL":"NFCE","PHI":"NFCE","NYG":"NFCE","WAS":"NFCE",
    "GB":"NFCN","MIN":"NFCN","CHI":"NFCN","DET":"NFCN",
    "TB":"NFCS","ATL":"NFCS","CAR":"NFCS","NO":"NFCS",
    "KC":"AFCW","LAC":"AFCW","DEN":"AFCW","LV":"AFCW","OAK":"AFCW",
    "BUF":"AFCE","MIA":"AFCE","NYJ":"AFCE","NE":"AFCE",
    "BAL":"AFCN","PIT":"AFCN","CLE":"AFCN","CIN":"AFCN",
    "TEN":"AFCS","JAX":"AFCS","HOU":"AFCS","IND":"AFCS",
}

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--season-start', type=int, default=2019)
    ap.add_argument('--season-end', type=int, default=2025)
    args = ap.parse_args()

    raw_dir = pathlib.Path('data/raw')
    proc_dir = pathlib.Path('data/processed'); proc_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = pathlib.Path('reports/diagnostics'); diag_dir.mkdir(parents=True, exist_ok=True)

    sched_paths = sorted(raw_dir.glob(f'schedules_{args.season_start}_{args.season_end}.parquet')) or \
                  sorted(raw_dir.glob('schedules_*.parquet'))
    if not sched_paths:
        raise SystemExit('No schedules_*.parquet found. Run pull_schedules.py first.')
    spath = sched_paths[-1]
    s = pd.read_parquet(spath)
    s.columns = [c.strip().lower().replace(' ', '_') for c in s.columns]

    missing = [c for c in REQ_SCHED if c not in s.columns]
    if missing:
        raise SystemExit(f'schedules missing columns: {missing}')

    # Try to detect market columns from schedules (nfl_data_py often has these)
    total_col  = find_col(s, ['over_under_line','total_line','over_under','total'])
    spread_col = find_col(s, ['spread_line','spread'])

    # Build home/away rows
    home = pd.DataFrame({
        'season': s['season'],
        'week': s['week'],
        'game_id': s['game_id'].astype('string'),
        'team': s['home_team'].astype('string'),
        'opponent': s['away_team'].astype('string'),
        'points_for': s['home_score'],
        'points_against': s['away_score'],
        'is_home': 1
    })
    away = pd.DataFrame({
        'season': s['season'],
        'week': s['week'],
        'game_id': s['game_id'].astype('string'),
        'team': s['away_team'].astype('string'),
        'opponent': s['home_team'].astype('string'),
        'points_for': s['away_score'],
        'points_against': s['home_score'],
        'is_home': 0
    })

    # Divisional flag
    for df_ in (home, away):
        df_['division'] = df_['team'].map(DIV)
        df_['opp_division'] = df_['opponent'].map(DIV)
        df_['is_divisional'] = (df_['division'] == df_['opp_division']).astype('Int8')

    out = pd.concat([home, away], ignore_index=True)

    # Targets
    out['points_for'] = pd.to_numeric(out['points_for'], errors='coerce')
    out['points_against'] = pd.to_numeric(out['points_against'], errors='coerce')
    out['margin'] = out['points_for'] - out['points_against']
    out['total_points'] = out['points_for'] + out['points_against']

    # Rest days + short week if date exists
    date_col = None
    for c in ['gameday','game_date','gamedate','game_time']:
        if c in s.columns:
            date_col = c; break
    if date_col is not None:
        sd = s.copy()
        sd['_date'] = pd.to_datetime(sd[date_col], errors='coerce')
        dhome = sd[['game_id','home_team','_date']].rename(columns={'home_team':'team'})
        daway = sd[['game_id','away_team','_date']].rename(columns={'away_team':'team'})
        d = pd.concat([dhome, daway], ignore_index=True)
        out = out.merge(d, on=['game_id','team'], how='left')
        out = out.sort_values(['team','_date']).reset_index(drop=True)
        out['rest_days'] = out.groupby('team')['_date'].diff().dt.days
        out['short_week'] = (out['rest_days'] < 7).astype('Int8')
    else:
        out['rest_days'] = pd.NA
        out['short_week'] = pd.NA

    # === Market anchors ===
    # Total: same for both teams
    if total_col is not None:
        tdf = s[['game_id', total_col]].rename(columns={total_col: 'market_total'})
        out = out.merge(tdf, on='game_id', how='left')
        out['market_total'] = pd.to_numeric(out['market_total'], errors='coerce')

    # Spread: usually home-centric (home negative if favored)
    if spread_col is not None:
        sdf = s[['game_id', spread_col]].rename(columns={spread_col: '_spread_home'})
        out = out.merge(sdf, on='game_id', how='left')
        out['_spread_home'] = pd.to_numeric(out['_spread_home'], errors='coerce')
        # Convert to the spread for this row's team
        out['market_spread'] = out.apply(
            lambda r: r['_spread_home'] if r['is_home'] == 1 else (-r['_spread_home'] if pd.notna(r['_spread_home']) else pd.NA),
            axis=1
        )
        out.drop(columns=['_spread_home'], inplace=True)

    # Keep schema stable
    for col in ['total_yards_for','total_yards_against','turnovers_for','turnovers_against']:
        if col not in out.columns:
            out[col] = pd.NA

    out = out.sort_values(['season','week','game_id','team']).reset_index(drop=True)
    feat_path = proc_dir / 'team_games_features.parquet'
    out.to_parquet(feat_path, index=False)

    diag = {
        'source_path': str(spath),
        'rows': len(out),
        'cols': out.columns.tolist()
    }
    with open(diag_dir / 'build_team_games_schema.json','w',encoding='utf-8') as f:
        json.dump(diag, f, indent=2)

    print(f'[build_team_games] wrote {len(out):,} rows -> {feat_path}')

if __name__ == '__main__':
    main()
