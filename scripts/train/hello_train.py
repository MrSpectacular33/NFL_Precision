import json, pathlib
pathlib.Path("reports/cv").mkdir(parents=True, exist_ok=True)
with open("reports/cv/hello_train.json","w",encoding="utf-8") as f:
    json.dump({"phase":"train","status":"ok"}, f)
print("[hello_train] OK")
