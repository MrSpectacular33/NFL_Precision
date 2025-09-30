import json, pathlib
pathlib.Path("reports/calibration").mkdir(parents=True, exist_ok=True)
with open("reports/calibration/hello_eval.json","w",encoding="utf-8") as f:
    json.dump({"phase":"eval","status":"ok"}, f)
print("[hello_eval] OK")
