import ruptures as rpt
import numpy as np
import pandas as pd
import os
from changepoint_plot import plot_changepoint

def single_file(cenario, file, W):
    series_folder = os.path.join("series", cenario)
    results_folder = os.path.join("results", cenario, "pelt")
    os.makedirs(results_folder, exist_ok=True)
    
    serie_file = os.path.join(series_folder, file)
    pelt_file = os.path.join(results_folder, file)

    data = pd.read_csv(serie_file)
    
    if W > 0:
        T = data.shape[0]
        if T <= 2 * W:
            raise ValueError(f"Window W={W} too large for T={T}. Need T > 2W.")
        data = data.iloc[W:-W].reset_index(drop=True)
        
    timestamps = data.iloc[:, 0].values
    X = data.iloc[:, 1].values

    # Apply PELT algorithm
    algo = rpt.Pelt(model='rbf').fit(X)
    pen = 3
    result = algo.predict(pen=pen)
    CP = np.array(result[:-1]).astype(int)
    CP = CP.tolist()


    results = pd.DataFrame({
        'timestamp': timestamps,
        'value': X,
        'CP': [1 if i in CP else 0 for i in range(len(X))]
    })
    results.to_csv(pelt_file, index=False)

def multiple_files(cenario, W=20):
    serie_folder = os.path.join("series", cenario)
    files = os.listdir(serie_folder)
    for file in files:
        if file.endswith(".csv"):
            try:
                single_file(cenario, file, W=W)
                plot_changepoint(cenario, "pelt", file, save=True)
            except Exception as e:
                continue

if __name__ == "__main__":
    import time
    # begin = time.time()

    # cenario = "teste"
    # file = "teste01.csv"
    # single_file(cenario,file)

    # cenario = "teste"
    # multiple_files(cenario, threshold)

    # end = time.time()
    # print(f"Tempo total: {end - begin:.2f} segundos")

    cenarios = ["teste_m110", "teste_m130", "teste_m150", "NDT_tp_down", "NDT_rtt_up"]
    for cenario in cenarios:
        begin = time.time()
        multiple_files(cenario, W=20)
        end = time.time()
        print(f"Cenário '{cenario}' processado em {end - begin:.2f} segundos")