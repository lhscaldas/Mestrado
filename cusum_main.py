import numpy as np
import pandas as pd
import os
import scipy.stats as stats
from changepoint_plot import plot_changepoint


def wl_cusum(X, w0=20, w1=10, h=3.0):
    """
    Window-limited CUSUM - Based on the provided logic.
    Assumes Gaussian distribution and uses logpdf for statistic calculation.
    """
    def logpdf(x, mean, std):
        return stats.norm.logpdf(x, loc=mean, scale=std)

    lcp = 0
    CP = []
    St = 0
    
    # Ensure there is enough data for the initial window
    if len(X) < w0:
        return []

    for t, y_t in enumerate(X):
        # Only start processing after accumulating the reference window (w0) since the last CP
        if t >= lcp + w0:
            if np.isnan(St):
                St = 0 
                
            # Phase 1: Null hypothesis parameters (m0, s0)
            # Calculated once when the w0 window is reached
            if t == lcp + w0:
                m0 = X[lcp:t].mean()
                s0 = X[lcp:t].std(ddof=1)
                if np.round(s0, 3) == 0:
                    s0 = 0.001
                Ht = h * s0
            
            # Phase 2: Alternative hypothesis parameters (m1, s1) via sliding window w1
            # Ensures the w1 window does not cross the last change point
            start_w1 = max(t - w1, lcp)
            m1 = X[start_w1:t+1].mean()
            s1 = X[start_w1:t+1].std(ddof=1)
            if np.round(s1, 3) == 0:
                s1 = 0.001
            
            # Update CUSUM statistic using Log-Likelihood Ratio
            LLR = logpdf(y_t, m1, s1) - logpdf(y_t, m0, s0) # type: ignore
            St = max(0, St + LLR)
            
            # Threshold check
            if St > Ht: # type: ignore
                lcp = t
                CP.append(lcp)
                St = 0 # Statistic reset
        else:
            St = np.nan
            
    return CP

def single_file(cenario, file):
    series_folder = os.path.join("series", cenario)
    results_folder = os.path.join("results", cenario, "cusum")
    os.makedirs(results_folder, exist_ok=True)

    serie_file = os.path.join(series_folder, file)
    cusum_file = os.path.join(results_folder, file)
    
    data = pd.read_csv(serie_file)
    timestamps = data.iloc[:, 0].values
    X = data.iloc[:, 1].values

    # Detecção via CUSUM com inferência de parâmetros (MLE)
    CP = wl_cusum(X, w0=20, w1=10, h=5.0)

    results = pd.DataFrame({
        'timestamp': timestamps,
        'value': X,
        'CP': [1 if i in CP else 0 for i in range(len(X))]
    })
    results.to_csv(cusum_file, index=False)

def multiple_files(cenario):
    serie_folder = os.path.join("series", cenario)
    files = os.listdir(serie_folder)
    for file in files:
        if file.endswith(".csv"):
            try:
                single_file(cenario, file)
                plot_changepoint(cenario, "cusum", file, save=True)
            except Exception:
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
        multiple_files(cenario)
        end = time.time()
        print(f"Cenário '{cenario}' processado em {end - begin:.2f} segundos")