param()
$py = ".\.venv\Scripts\python.exe"
& $py .\scripts\features\add_rolling_form_v5.py
& $py .\scripts\features\add_sched_context_v5.py
& $py .\scripts\features\add_weather_v5.py
& $py .\scripts\features\finalize_v5.py
