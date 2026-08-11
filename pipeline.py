import subprocess

"""
IMPORTANTE: Checar os cenários e métodos a serem processados em cada script antes de rodar este pipeline, para evitar processamento desnecessário.
"""

scripts = [f"main_{method}.py" for method in ["vwcd", "cusum", "pelt"]] # + ["plot_changepoint.py"]

for script in scripts:
    print(f"Running {script}...")
    subprocess.run(["python", script], check=True)