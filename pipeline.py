import subprocess

"""
IMPORTANTE: Checar os cenários e métodos a serem processados em cada script antes de rodar este pipeline, para evitar processamento desnecessário.
"""
# scripts = ["cusum_main.py", "changepoint_plot.py", "changepoint_metrics.py"]
scripts = ["vwcd_main.py", "cusum_main.py", "pelt_main.py", "changepoint_plot.py"]

for script in scripts:
    print(f"Running {script}...")
    subprocess.run(["python", script], check=True)