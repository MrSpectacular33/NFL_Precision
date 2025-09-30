import argparse, json, pathlib
p = argparse.ArgumentParser()
p.add_argument("--season", type=int, required=True)
args = p.parse_args()

pathlib.Path("data/processed").mkdir(parents=True, exist_ok=True)
with open("reports/diagnostics/hello_features.json","w",encoding="utf-8") as f:
    json.dump({"phase":"features","season":args.season}, f)
print("[hello_features] OK", {"season":args.season})
