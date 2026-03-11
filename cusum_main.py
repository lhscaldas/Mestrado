import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.stats as stats

def plot_results(cenario, file):
    # Define folders
    results_folder = os.path.join("results", cenario, "cusum")
    plots_folder = os.path.join("plots", cenario, "cusum")
    os.makedirs(plots_folder, exist_ok=True)

    # Define files
    cusum_file = os.path.join(results_folder, file)
    plot_file = os.path.join(plots_folder, file.replace(".csv", ".png"))
    
    # Read the CUSUM results
    results = pd.read_csv(cusum_file)

    # Convert timestamps to datetime to avoid conversion errors
    results['timestamp'] = pd.to_datetime(results['timestamp'])
    timestamps = results['timestamp'].values
    values = results['value'].values
    CP = results[results['CP'] == 1]['timestamp'].values

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, values, label='Data')  # type: ignore
    
    for i, cp in enumerate(CP):
        plt.axvline(x=cp, color='red', linestyle='--', label='Change Point' if i == 0 else "")
        
    plt.title(f'CUSUM - {file}')
    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(plot_file)
    plt.close()

def wl_cusum(X, w0=20, w1=10, h=5.0):
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
    plot_results(cenario, file)

def multiple_files(cenario):
    serie_folder = os.path.join("series", cenario)
    files = os.listdir(serie_folder)
    for file in files:
        if file.endswith(".csv"):
            try:
                single_file(cenario, file)
            except Exception:
                continue

if __name__ == "__main__":
    import time
    begin = time.time()

    cenario = "teste"
    # file = "teste01.csv"
    # single_file(cenario,file)
    multiple_files(cenario)

    end = time.time()
    print(f"Tempo total: {end - begin:.2f} segundos")