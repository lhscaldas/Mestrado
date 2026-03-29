import subprocess

# scripts = ["cusum_main.py", "changepoint_plot.py", "changepoint_metrics.py"]
scripts = ["vwcd_main.py", "cusum_main.py", "pelt_main.py", "changepoint_metrics.py"]

for script in scripts:
    print(f"Running {script}...")
    subprocess.run(["python", script], check=True)