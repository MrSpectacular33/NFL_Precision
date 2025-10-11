import sys, yaml, json
with open(r"configs/risk/adaptive_kelly.yml","r",encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

need = {
  "top": ["kelly_input","bankroll_target"],
  "kelly": ["base_fraction","min_fraction","max_fraction"],
  "regimes": ["vol_high_sigma","vol_low_sigma","dd_soft","dd_hard"],
  "smoothing": ["ema_alpha","dd_alpha"],
  "correlation": ["same_game_penalty","penalize_positive_rho","rho_same_game"],
  "constraints": ["per_bet_cap","min_edge","max_bets_per_game","max_weekly_risk"],
}

errors=[]
for k in need["top"]:
    if k not in cfg: errors.append(f"missing top-level: {k}")

for sect in ("kelly","regimes","smoothing","correlation","constraints"):
    if not isinstance(cfg.get(sect), dict):
        errors.append(f"missing section: {sect}")
        continue
    for k in need[sect]:
        if k not in cfg[sect]:
            errors.append(f"missing {sect}.{k}")

print(json.dumps({"ok": not errors, "errors": errors}, indent=2))
sys.exit(0 if not errors else 2)
