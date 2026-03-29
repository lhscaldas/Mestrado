import numpy as np
import pandas as pd
import os
import scipy.stats as stats
from changepoint_plot import plot_changepoint
import optuna
import shutil

def wl_cusum(X, w0=30, w1=20, h=5.0):
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
            # Phase 1: Null hypothesis parameters (m0, s0)
            # Calculated once when the w0 window is reached
            if t == lcp + w0:
                m0 = X[lcp:t].mean()
                s0 = X[lcp:t].std(ddof=1)
                s0 = max(s0, 1e-6)  # Avoid division by zero
                Ht = h * s0
            
            # Phase 2: Alternative hypothesis parameters (m1, s1) via sliding window w1
            # Ensures the w1 window does not cross the last change point
            start_w1 = max(t - w1, lcp)
            m1 = X[start_w1:t].mean()
            s1 = X[start_w1:t].std(ddof=1) 
            s1 = max(s1, 1e-6) # Avoid division by zero
            
            # Update CUSUM statistic using Log-Likelihood Ratio
            LLR = logpdf(y_t, m1, s0) - logpdf(y_t, m0, s0) # type: ignore
            St = max(0, St + LLR)
            
            # Threshold check
            if St > Ht: # type: ignore
                lcp = t
                CP.append(lcp-w1//2)
                St = 0 # Statistic reset
            
    return CP

# def generate_synthetic_data(n_points=300):
#     np.random.seed(42)
#     X = np.random.normal(0, 1, n_points)
#     true_cps = [100, 200]
#     X[100:200] += 2.0
#     X[200:] -= 1.0
#     return X, true_cps

# def evaluate_cps(true_cps, pred_cps, tolerance=10):
#     matched_preds = set()
#     tp = 0
#     for tcp in true_cps:
#         for pcp in pred_cps:
#             if pcp not in matched_preds and abs(tcp - pcp) <= tolerance:
#                 tp += 1
#                 matched_preds.add(pcp)
#                 break
#     fp = len(pred_cps) - tp
#     fn = len(true_cps) - tp
    
#     if tp == 0:
#         return 0.0
    
#     precision = tp / (tp + fp)
#     recall = tp / (tp + fn)
#     f1 = 2 * (precision * recall) / (precision + recall)
#     return f1

# def objective(trial):
#     w0 = trial.suggest_int("w0", 10, 50)
#     w1 = trial.suggest_int("w1", 3, w0 - 1)
#     h = trial.suggest_float("h", 0.5, 15.0)
    
#     X, true_cps = generate_synthetic_data(n_points=300)
#     pred_cps = wl_cusum(X, w0=w0, w1=w1, h=h)
    
#     score = evaluate_cps(true_cps, pred_cps, tolerance=10)
#     return score

# def find_best_hyperparameters():
#     optuna.logging.set_verbosity(optuna.logging.WARNING)
#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=200)
    
#     print("Melhores hiperparâmetros para séries de até 300 pontos:")
#     print(f"w0: {study.best_params['w0']}")
#     print(f"w1: {study.best_params['w1']}")
#     print(f"h: {study.best_params['h']:.3f}")
#     print(f"F1-Score: {study.best_value:.3f}")

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
    CP = wl_cusum(X)

    results = pd.DataFrame({
        'timestamp': timestamps,
        'value': X,
        'CP': [1 if i in CP else 0 for i in range(len(X))]
    })
    results.to_csv(cusum_file, index=False)

def multiple_files(cenario):
    serie_folder = os.path.join("series", cenario)
    files = os.listdir(serie_folder)

    results_folder = os.path.join("results", cenario, "cusum")
    if os.path.exists(results_folder):
        shutil.rmtree(results_folder)
    os.makedirs(results_folder, exist_ok=True)
    
    plots_folder = os.path.join("plots", cenario, "cusum")
    if os.path.exists(plots_folder):
        shutil.rmtree(plots_folder)
    os.makedirs(plots_folder, exist_ok=True)

    for file in files:
        if file.endswith(".csv"):
            single_file(cenario, file)
            plot_changepoint(cenario, "cusum", file, save=True)

if __name__ == "__main__":
    print("Starting CUSUM change point detection...")
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

    # find_best_hyperparameters()