import subprocess

# scripts = ["cusum_main.py", "changepoint_plot.py", "changepoint_metrics.py"]
scripts = ["cusum_main.py", "changepoint_metrics.py"]

for script in scripts:
    print(f"Running {script}...")
    subprocess.run(["python", script], check=True)