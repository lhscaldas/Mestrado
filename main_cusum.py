import numpy as np
import pandas as pd
import os
import scipy.stats as stats
from plot_changepoint import plot_changepoint
import optuna
import shutil
from river.drift import PageHinkley
from detecta import detect_cusum
from scipy.ndimage import median_filter

#------------------ Otimização de hiperparâmetros para CUSUM ------------------

def generate_synthetic_data(n_points=300):
    np.random.seed(42)
    X = np.random.normal(0, 1, n_points)
    true_cps = [100, 200]
    X[100:200] += 2.0
    X[200:] -= 1.0
    return X, true_cps

def evaluate_cps(true_cps, pred_cps, tolerance=10):
    matched_preds = set()
    tp = 0
    for tcp in true_cps:
        for pcp in pred_cps:
            if pcp not in matched_preds and abs(tcp - pcp) <= tolerance:
                tp += 1
                matched_preds.add(pcp)
                break
    fp = len(pred_cps) - tp
    fn = len(true_cps) - tp
    
    if tp == 0:
        return 0.0
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def objective(trial):
    w0 = trial.suggest_int("w0", 10, 50)
    w1 = trial.suggest_int("w1", 3, w0 - 1)
    h = trial.suggest_float("h", 0.5, 15.0)
    
    X, true_cps = generate_synthetic_data(n_points=300)
    pred_cps = wl_cusum(X, w0=w0, w1=w1, h=h) # type: ignore
    
    score = evaluate_cps(true_cps, pred_cps, tolerance=10)
    return score

def find_best_hyperparameters():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200)
    
    print("Melhores hiperparâmetros para séries de até 300 pontos:")
    print(f"w0: {study.best_params['w0']}")
    print(f"w1: {study.best_params['w1']}")
    print(f"h: {study.best_params['h']:.3f}")
    print(f"F1-Score: {study.best_value:.3f}")

#------------------ Métodos de detecção de CP CUSUM based ---------------------

def river_cusum(X, min_instances=30, delta=0.005, threshold=50): # Lib River (baseado no Page-Hinkley)
    cusum_ph = PageHinkley(min_instances=min_instances, delta=delta, threshold=threshold)
    cp = []
    
    for t, y_t in enumerate(X):
        cusum_ph.update(y_t)
        if cusum_ph.drift_detected:
            cp.append(t)
            
    return cp

def detecta_cusum(X, threshold=5.0, drift=0.5): # Lib detecta (baseado no cusum)
    baseline_window = 100
    if len(X) < baseline_window:
        return []
    s0 = np.std(X[:baseline_window], ddof=1)
    s0 = max(s0, 1e-6)  # Evita divisão por zero

    threshold = threshold * s0
    drift = drift * s0

    ta, _, _, _ = detect_cusum(
        x=X,
        threshold=threshold, # type: ignore
        drift=drift, # type: ignore
        ending=True,
        show=False
    )
    
    return list(ta)

def detecta_cusum_tratado(X, threshold=5.0, drift=0.7, window_suavizacao=5):
    """
    Pipeline com pré-tratamento de sinal (suavização + normalização)
    para detecção universal via CUSUM clássico.
    """
    X_arr = np.asarray(X, dtype=float)
    if len(X_arr) < 30:
        return []
    
    # 1. Tratamento: Média Móvel para eliminar ruído local e spikes isolados
    # Usamos pandas para lidar facilmente com as bordas (min_periods=1)
    X_smooth = pd.Series(X_arr).rolling(window=window_suavizacao, center=True, min_periods=1).mean().to_numpy()
    
    # 2. Tratamento: Normalização Z-Score Global da série suavizada
    # Isso traz qualquer RTT (seja base 8ms ou base 120ms) para a mesma escala estatística
    std_smooth = np.std(X_smooth, ddof=1)
    if std_smooth < 1e-6:
        return [] # Série perfeitamente plana, sem changepoints
        
    X_norm = (X_smooth - np.mean(X_smooth)) / std_smooth
    
    # 3. Execução do CUSUM na série limpa e normalizada
    # Como a série está normalizada, threshold=5.0 significa acumular 5 desvios padrões de erro.
    ta, _, _, _ = detect_cusum(
        x=X_norm,
        threshold=threshold, # type: ignore
        drift=drift, # type: ignore
        ending=True,
        show=False
    )
    
    # 4. Pós-processamento: Agrupar detecções redundantes na mesma transição
    CP_filtrados = []
    if len(ta) > 0:
        ta_sorted = sorted(list(ta))
        CP_filtrados.append(ta_sorted[0])
        for i in range(1, len(ta_sorted)):
            # Evita registrar múltiplos CPs na subida/descida do mesmo degrau
            if ta_sorted[i] - ta_sorted[i-1] > 10:
                CP_filtrados.append(ta_sorted[i])
                
    return CP_filtrados

def detecta_cusum_universal(X, threshold=4.0, drift=0.2, window_rolling=30):
    """
    CUSUM adaptativo com normalização por janela móvel (Rolling Scale).
    Funciona universalmente para micro-degraus e macro-degraus de RTT.
    """
    X_arr = np.asarray(X, dtype=float)
    if len(X_arr) < window_rolling:
        return []
    
    # 1. Filtro de mediana leve para eliminar picos de curtíssima duração (spikes)
    X_smooth = median_filter(X_arr, size=3)
    
    # Transformar em Pandas para usar estatística móvel eficiente
    s_smooth = pd.Series(X_smooth)
    
    # 2. Estatísticas Móveis (Rolling) para capturar o contexto local
    rolling_median = s_smooth.rolling(window=window_rolling, center=False, min_periods=1).median()
    rolling_q25 = s_smooth.rolling(window=window_rolling, center=False, min_periods=1).quantile(0.25)
    rolling_q75 = s_smooth.rolling(window=window_rolling, center=False, min_periods=1).quantile(0.75)
    
    rolling_iqr = rolling_q75 - rolling_q25
    # Definir um piso para o IQR local para evitar divisões por zero ou hipersensibilidade
    rolling_iqr = np.maximum(rolling_iqr.to_numpy(), 1.5)
    
    # 3. Normalização Local Adaptativa
    X_norm = (X_smooth - rolling_median.to_numpy()) / rolling_iqr
    
    # 4. Execução do CUSUM clássico na série adaptada
    ta, _, _, _ = detect_cusum(
        x=X_norm,
        threshold=threshold, # type: ignore
        drift=drift, # type: ignore
        ending=True,
        show=False
    )
    
    # 5. Pós-processamento: Agrupamento de CPs redundantes na mesma transição
    CP_filtrados = []
    if len(ta) > 0:
        ta_sorted = sorted(list(ta))
        CP_filtrados.append(ta_sorted[0])
        for i in range(1, len(ta_sorted)):
            if ta_sorted[i] - ta_sorted[i-1] > 10:
                CP_filtrados.append(ta_sorted[i])
                
    return CP_filtrados

def wl_cusum(X, w0=20, w1=20, h=5.0, d=0.5): # Meu
    """
    Window-limited CUSUM - Based on the provided logic.
    Assumes Gaussian distribution and uses logpdf for statistic calculation.
    """
    def logpdf(x, mean, std):
        return stats.norm.logpdf(x, loc=mean, scale=std)

    lcp = 0 
    CP = []
    St = 0
    last_zero_t = 0
    
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
                Dt = d * s0
            
            # Phase 2: Alternative hypothesis parameters (m1, s1) via sliding window w1
            # Ensures the w1 window does not cross the last change point
            start_w1 = max(t - w1, lcp)
            m1 = X[start_w1:t].mean()
            s1 = X[start_w1:t].std(ddof=1)
            s1 = max(s1, 1e-6) # Avoid division by zero
            
            # Update CUSUM statistic using Log-Likelihood Ratio
            LLR = logpdf(y_t, m1, s1) - logpdf(y_t, m0, s0) # type: ignore
            St = max(0, St + LLR - Dt) # type: ignore
            last_zero_t = t if St == 0 else last_zero_t
            
            # Threshold check
            if St > Ht: # type: ignore
                lcp = t
                CP.append(last_zero_t)
                St = 0 # Statistic reset
            
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

    # Detecção via CUSUM
    CP = wl_cusum(X)
    # CP = river_cusum(X)
    # CP = detecta_cusum(X)
    # CP = detecta_cusum_universal(X)


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

    # NDT_folder = "NDT"
    # cenarios = [f"{NDT_folder}/{p}" for p in os.listdir(f"series/{NDT_folder}") if os.path.isdir(f"series/{NDT_folder}/{p}") and p != "full"]
    cenarios = ['teste_m110', 'teste_m130', 'teste_m150']
    for cenario in cenarios:
        begin = time.time()
        multiple_files(cenario)
        end = time.time()
        print(f"Cenário '{cenario}' processado em {end - begin:.2f} segundos")

    # find_best_hyperparameters()