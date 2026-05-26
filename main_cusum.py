import numpy as np
import pandas as pd
import os
import scipy.stats as stats
from plot_changepoint import plot_changepoint
import optuna
import shutil
from river.drift import PageHinkley
from detecta import detect_cusum

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

def detecta_cusum(X, threshold=10.0, drift=0.5): # Lib detecta (baseado no cusum)
    baseline_window = 20
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

def detecta_cusum_auto(X, c_threshold=3.0, drift_desejado=0.5):
    if len(X) < 20:
        return []
    
    # 1. Usar o desvio padrão global da série (mais estável que a janela de 20)
    # ou uma métrica robusta como o MAD (Median Absolute Deviation)
    s_global = np.std(X, ddof=1)
    s_global = max(s_global, 1e-6)
    
    # 2. Threshold adaptativo baseado no comprimento da série (estilo BIC do PELT)
    # Quanto maior a série, maior o threshold necessário para evitar falsos positivos acumulados
    threshold_adaptativo = c_threshold * np.log(len(X))
    
    # 3. Escalar os parâmetros pelo desvio global
    threshold = threshold_adaptativo * s_global
    drift = drift_desejado * s_global

    # Executa a função interna da biblioteca
    ta, _, _, _ = detect_cusum(
        x=X,
        threshold=threshold,
        drift=drift, # type: ignore
        ending=True,
        show=False
    )
    
    return list(ta)

import numpy as np
from scipy import stats

def detecta_cusum_rede(X, threshold_factor=4.5, drift_factor=0.5):
    """
    Versão robusta do CUSUM adaptada para telemetria de rede (RTT).
    Usa estatística robusta (MAD) e threshold adaptativo ao tamanho da série.
    """
    X_arr = np.asarray(X)
    N = len(X_arr)
    if N < 20:
        return []

    # 1. Filtro leve para ignorar spikes isolados que quebram o CUSUM
    # Se quiser testar o CUSUM puro primeiro, comente as duas linhas abaixo
    from scipy.ndimage import median_filter
    X_filtrado = median_filter(X_arr, size=3)
    
    # 2. Calcular a escala usando MAD (Median Absolute Deviation)
    # É muito mais robusto a picos isolados do que o desvio padrão clássico
    mediana = np.median(X_filtrado)
    mad = np.median(np.abs(X_filtrado - mediana))
    # Consistência para distribuição normal
    s_robusto = mad * 1.4826 
    s_robusto = max(s_robusto, 1e-6)

    # 3. Threshold adaptativo misto (Estilo BIC do PELT + Limite de Controle)
    # O log(N) garante que séries longas não acumulem falsos positivos por deriva natural
    threshold_adaptativo = threshold_factor * np.log(N) * s_robusto
    drift_adaptativo = drift_factor * s_robusto

    # 4. Executa o algoritmo interno da biblioteca detecta
    ta, _, _, _ = detect_cusum(
        x=X_filtrado,
        threshold=threshold_adaptativo,
        drift=drift_adaptativo,
        ending=True,
        show=False
    )
    
    return list(ta)

import numpy as np
from scipy import stats
from scipy.ndimage import median_filter

def detecta_cusum_estavel(X, threshold_factor=6.0, drift_factor=1.0):
    """
    CUSUM adaptativo com proteção contra saturação de falsos positivos
    e tratamento de piso de ruído para séries de RTT.
    """
    X_arr = np.asarray(X, dtype=float)
    N = len(X_arr)
    if N < 30:
        return []

    # 1. Filtro de mediana para limpar spikes secos (essencial para RTT)
    X_filtrado = median_filter(X_arr, size=3)

    # 2. Cálculo robusto da variância (MAD)
    mediana_global = np.median(X_filtrado)
    mad = np.median(np.abs(X_filtrado - mediana_global))
    s_robusto = mad * 1.4826
    
    # PROTEÇÃO 1: Evita que séries muito "justas" quebrem o algoritmo.
    # Se o desvio robusto for menor que 2.0 ms, assume 2.0 ms como ruído natural de rede.
    s_robusto = max(s_robusto, 2.0)

    # 3. Parametrização robusta baseada no tamanho da série
    # Subir o threshold_factor para 6.0 remove o acúmulo de ruído comum.
    threshold = threshold_factor * np.log(N) * s_robusto
    drift = drift_factor * s_robusto

    # 4. Execução do CUSUM original da biblioteca
    ta, _, _, _ = detect_cusum(
        x=X_filtrado,
        threshold=threshold,
        drift=drift,
        ending=True,
        show=False
    )
    
    # PROTEÇÃO 2: Filtragem de adjacência (Cessa o efeito metralhadora)
    # Se houver alarmes em sequência direta, mantém apenas o primeiro da transição
    CP_filtrados = []
    if len(ta) > 0:
        ta_sorted = sorted(list(ta))
        CP_filtrados.append(ta_sorted[0])
        for i in range(1, len(ta_sorted)):
            # Se o changepoint atual for muito colado ao anterior (ex: menos de 7 pontos),
            # ignora porque é apenas o resíduo da mesma transição acumulada
            if ta_sorted[i] - ta_sorted[i-1] > 7:
                CP_filtrados.append(ta_sorted[i])

    return CP_filtrados

def wl_cusum(X, w0=20, w1=20, h=10.0, d=0.5): # Meu
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

import numpy as np
import pandas as pd

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
        threshold=threshold,
        drift=drift,
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

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

def detecta_cusum_final(X, threshold=3.0, drift=0.2, window_suavizacao=5):
    """
    Pipeline com filtragem não-linear por mediana e escalonamento robusto (IQR).
    Garante o espurgo de outliers isolados e foca em mudanças de patamar estáveis.
    """
    X_arr = np.asarray(X, dtype=float)
    if len(X_arr) < 30:
        return []
    
    # 1. TRATAMENTO 1: Filtro de Mediana Móvel (Elimina matematicamente picos isolados)
    # Diferente da média, a mediana apaga o spike se ele durar menos que metade da janela
    X_smooth = median_filter(X_arr, size=window_suavizacao)
    
    # 2. TRATAMENTO 2: Normalização Robusta via IQR (Substitui o Z-Score)
    # Isola os picos remanescentes e dá peso real para a variação dos blocos estáveis
    q25, q75 = np.percentile(X_smooth, [25, 75])
    iqr = q75 - q25
    
    # Se a série for excessivamente plana no miolo, define um piso para o IQR
    iqr = max(iqr, 1.0) 
    
    # Série centralizada na mediana e escalonada pelo IQR
    X_norm = (X_smooth - np.median(X_smooth)) / iqr
    
    # 3. Execução do CUSUM clássico na série blindada
    ta, _, _, _ = detect_cusum(
        x=X_norm,
        threshold=threshold,
        drift=drift,
        ending=True,
        show=False
    )
    
    # 4. Pós-processamento para agrupar múltiplos disparos no mesmo degrau
    CP_filtrados = []
    if len(ta) > 0:
        ta_sorted = sorted(list(ta))
        CP_filtrados.append(ta_sorted[0])
        for i in range(1, len(ta_sorted)):
            if ta_sorted[i] - ta_sorted[i-1] > 15: # Janela de tolerância de transição
                CP_filtrados.append(ta_sorted[i])
                
    return CP_filtrados

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

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
        threshold=threshold,
        drift=drift,
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
    # CP = wl_cusum(X)
    # CP = river_cusum(X)
    # CP = detecta_cusum(X)
    # CP = detecta_cusum_auto(X)
    # CP = detecta_cusum_rede(X)
    # CP = detecta_cusum_estavel(X)
    # CP = detecta_cusum_tratado(X)
    # CP = detecta_cusum_final(X)
    CP = detecta_cusum_universal(X)


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

    NDT_folder = "NDT"
    cenarios = [f"{NDT_folder}/{p}" for p in os.listdir(f"series/{NDT_folder}") if os.path.isdir(f"series/{NDT_folder}/{p}") and p != "full"]
    for cenario in cenarios:
        begin = time.time()
        multiple_files(cenario)
        end = time.time()
        print(f"Cenário '{cenario}' processado em {end - begin:.2f} segundos")

    # find_best_hyperparameters()