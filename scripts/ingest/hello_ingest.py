import argparse, json, os, pathlib
p = argparse.ArgumentParser()
p.add_argument("--season", type=int, required=True)
p.add_argument("--week", type=int, required=True)
args = p.parse_args()

pathlib.Path("data/raw").mkdir(parents=True, exist_ok=True)
out = {"phase":"ingest","season":args.season,"week":args.week}
with open("reports/diagnostics/hello_ingest.json","w",encoding="utf-8") as f:
    json.dump(out,f)
print("[hello_ingest] OK", out)
