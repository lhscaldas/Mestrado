import ruptures as rpt
import numpy as np
import pandas as pd
import os
from changepoint_plot import plot_changepoint
import shutil

def single_file(cenario, file):
    series_folder = os.path.join("series", cenario)
    results_folder = os.path.join("results", cenario, "pelt")
    os.makedirs(results_folder, exist_ok=True)
    
    serie_file = os.path.join(series_folder, file)
    pelt_file = os.path.join(results_folder, file)

    data = pd.read_csv(serie_file)
    timestamps = data.iloc[:, 0].values
    X = data.iloc[:, 1].values

    # Apply PELT algorithm
    min_size = 4
    algo = rpt.Pelt(model='rbf', min_size=min_size).fit(X)
    pen = 2 * np.log(len(X)) # BIC-like penalty
    result = algo.predict(pen=pen)
    CP = np.array(result[:-1]).astype(int)
    CP = CP.tolist()

    results = pd.DataFrame({
        'timestamp': timestamps,
        'value': X,
        'CP': [1 if i in CP else 0 for i in range(len(X))]
    })
    results.to_csv(pelt_file, index=False)

def multiple_files(cenario):
    serie_folder = os.path.join("series", cenario)
    files = os.listdir(serie_folder)

    results_folder = os.path.join("results", cenario, "pelt")
    if os.path.exists(results_folder):
        shutil.rmtree(results_folder)
    os.makedirs(results_folder, exist_ok=True)
    
    plots_folder = os.path.join("plots", cenario, "pelt")
    if os.path.exists(plots_folder):
        shutil.rmtree(plots_folder)
    os.makedirs(plots_folder, exist_ok=True)

    for file in files:
        if file.endswith(".csv"):
            single_file(cenario, file)
            plot_changepoint(cenario, "pelt", file, save=True)

if __name__ == "__main__":
    print("Starting PELT change point detection...")
    import time
    # begin = time.time()

    # cenario = "teste"
    # file = "teste01.csv"
    # single_file(cenario,file)

    # cenario = "teste"
    # multiple_files(cenario, threshold)

    # end = time.time()
    # print(f"Tempo total: {end - begin:.2f} segundos")

    NDT_folder = "NDT_OUT"
    cenarios = [f"{NDT_folder}/packet_loss", f"{NDT_folder}/tp_up", f"{NDT_folder}/rtt_down", f"{NDT_folder}/tp_down", f"{NDT_folder}/rtt_up"]
    for cenario in cenarios:
        begin = time.time()
        multiple_files(cenario)
        end = time.time()
        print(f"Cenário '{cenario}' processado em {end - begin:.2f} segundos")