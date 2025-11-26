from VWCD import *
import pandas as pd
import numpy as np
import ruptures as rpt
import time
import os

from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
changepoint_np = importr('changepoint.np')
changepoint = importr('changepoint')

def pelt_wrapper(X, mode='rbf', penalty=None, **kwargs):
    startTime = time.time()

    if mode == 'rbf':
        algo = rpt.Pelt(model='rbf').fit(X)
        pen = 3
        if penalty == 'BIC':
         pen = np.log(len(X)) # BIC
        elif penalty == 'AIC':
         pen = 2
        result = algo.predict(pen=pen)
        CP = np.array(result[:-1]).astype(int)
    elif mode == 'ed':
        CP = [int(i) for i in changepoint.cpts(changepoint_np.cpt_np(FloatVector(X), penalty='MBIC', minseglen=4))]
        CP = np.array(CP[:-1]).astype(int)
    else:
        raise ValueError("Modo desconhecido. Use 'rbf' ou 'ed'.")
   
    endTime = time.time()
    elapsedTime = endTime - startTime

    # Estes conceitos não existem no PELT
    vote_counts = np.zeros_like(X)
    agg_probs = np.zeros_like(X)

    return CP.tolist(), elapsedTime, vote_counts, agg_probs

def detect_changepoints(input_dir, output_dir, detection_func, default_params):

    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(input_dir, file)
            df = pd.read_csv(file_path, parse_dates=['timestamp'])
            df = df.drop(columns=['state', 'value_cp'], errors='ignore')

            for column in df.select_dtypes(include=[np.number]).columns:
                y = df[column].dropna().to_numpy()

                current_params = default_params.copy()
                
                current_params['X'] = y

                CP, elapsedTime, vote_counts, agg_probs = detection_func(**current_params)
                CP = np.array(CP)
                
                changepoints = np.zeros(len(y), dtype=int)
                if len(CP) > 0:
                    changepoints[CP.astype(int)] = 1

                local_means = np.zeros(len(y))
                local_stds = np.zeros(len(y))

                # Calcular M0 e S0
                if len(CP) > 0:
                    cps = CP.astype(int)
                    all_cps = np.concatenate(([0], cps, [len(y)]))
                    for i in range(len(all_cps) - 1):
                        start_idx = all_cps[i]
                        end_idx = all_cps[i+1]
                        segment = y[start_idx:end_idx]
                        if len(segment) > 0:
                            local_means[start_idx:end_idx] = np.mean(segment)
                            local_stds[start_idx:end_idx] = np.std(segment, ddof=1) if len(segment) > 1 else 0

                valid_indices = df[column].dropna().index
                df.loc[valid_indices, f'{column}_cp'] = changepoints
                df.loc[valid_indices, f'{column}_votes'] = vote_counts
                df.loc[valid_indices, f'{column}_agg_probs'] = agg_probs
                df.loc[valid_indices, f'{column}_local_mean'] = local_means
                df.loc[valid_indices, f'{column}_local_std'] = local_stds
                
            output_file = os.path.join(output_dir, file)
            df.to_csv(output_file, index=False)

    print(f"Processamento concluído. Resultados salvos em: {output_dir}")

def recalculate_means_and_stds_by_reference(input_dir, output_dir, reference_feature):
    """
    Recalcula as médias e desvios padrão locais para todas as colunas numéricas, 
    baseando os segmentos nos changepoints de uma 'reference_feature' para o cálculo das estatísticas de cada segmento.

    Parâmetros:
    -----------
    input_dir : str
        Diretório contendo os arquivos CSV processados (com pelo menos uma coluna '_cp').
    output_dir : str
        Diretório para salvar os arquivos CSV com as estatísticas recalculadas.
    reference_feature : str
        Nome da variável cuja coluna '_cp' definirá os segmentos 
        (ex: 'throughput_download').

    Retorna:
    --------
    None
        Salva os arquivos CSV atualizados no diretório de saída.
    """
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"Iniciando recálculo de estatísticas (baseado em '{reference_feature}_cp')...")

    for file in os.listdir(input_dir):
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)
        
        try:
            df = pd.read_csv(file_path)
            df.reset_index(drop=True, inplace=True) # Garante índice posicional
            df['test_time'] = pd.to_datetime(df['timestamp'], errors='coerce')
        except Exception as e:
            print(f"Erro ao ler o arquivo {file_path}: {e}")
            continue

        # 1. Encontrar os segmentos com base na feature de referência
        reference_cp_column = f'{reference_feature}_cp'
        if reference_cp_column not in df.columns:
            print(f"AVISO: Coluna de referência '{reference_cp_column}' não encontrada em '{file}'. Pulando.")
            continue
            
        print(f"  -> Processando '{file}'...")

        changepoint_indices = df.index[df[reference_cp_column] == 1].tolist()
        all_segment_boundaries = sorted(list(set([0] + changepoint_indices + [len(df)])))

        # 2. Identificar todas as colunas numéricas originais
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        # Excluir colunas auxiliares que não são features originais
        original_features = [
            col for col in numeric_cols 
            if not col.endswith(('_cp', '_votes', '_vote_probs', '_agg_probs', '_local_mean', '_local_std'))
        ]
        
        # Prepara as colunas de saída
        for var in original_features:
            df[f'{var}_local_mean'] = np.nan
            df[f'{var}_local_std'] = np.nan

        # 3. Iterar sobre cada segmento e calcular as estatísticas usando a janela w0
        for i in range(len(all_segment_boundaries) - 1):
            start_idx = all_segment_boundaries[i] # Início do segmento (inclusive)
            end_idx = all_segment_boundaries[i+1] # Fim do segmento (exclusive)
            
            # Define a janela de cálculo: os primeiros w0 pontos DENTRO do segmento
            # O início da janela é o start_idx (ou 0 se for o primeiro ponto)
            window_start = start_idx
            # O fim da janela é w0 pontos depois, limitado pelo fim do segmento
            window_end = end_idx

            # Pega a fatia do DataFrame correspondente à JANELA DE CÁLCULO
            df_window_slice = df.iloc[window_start:window_end]

            if df_window_slice.empty:
                continue

            # Calcula a média e std DENTRO da janela w0 para todas as features originais
            segment_means = df_window_slice[original_features].mean().to_dict()
            segment_stds = df_window_slice[original_features].std(ddof=1).to_dict()

            # 4. Atribui as estatísticas calculadas a TODO o segmento no DataFrame original
            for var in original_features:
                mean_col = f'{var}_local_mean'
                std_col = f'{var}_local_std'
                
                # Obtém a posição da coluna pelo nome para usar com .iloc
                mean_col_idx = df.columns.get_loc(mean_col)
                std_col_idx = df.columns.get_loc(std_col)

                # Atribui a média/std calculada na janela w0 a todas as linhas do segmento atual
                df.iloc[start_idx:end_idx, mean_col_idx] = segment_means.get(var, np.nan) # type: ignore
                df.iloc[start_idx:end_idx, std_col_idx] = segment_stds.get(var, np.nan)   # type: ignore
        
        # 5. Salva o DataFrame atualizado em um novo arquivo CSV
        df.to_csv(output_path, index=False)

    print(f"\n✅ Processo concluído. Arquivos atualizados salvos em: {output_dir}")

import numpy as np

def cusum(X, f0_params, f1_params, threshold):

    def gaussian_log_pdf(x, mean, std):
        return -0.5 * np.log(2 * np.pi) - np.log(std) - 0.5 * ((x - mean) / std)**2

    n = len(X)
    g = np.zeros(n)
    alarm_time = None
    
    current_g = 0.0
    
    # Define as funções com os parâmetros fixos passados
    f0_log = lambda x: gaussian_log_pdf(x, mean=f0_params[0], std=f0_params[1])
    f1_log = lambda x: gaussian_log_pdf(x, mean=f1_params[0], std=f1_params[1])

    for t in range(n):
        # 1. Log-Likelihood Ratio
        llr = f1_log(X[t]) - f0_log(X[t])
        
        # 2. Atualização Recursiva
        current_g = max(0, current_g + llr)
        g[t] = current_g
        
        # 3. Verificação do Limiar (CORRIGIDO)
        if current_g >= threshold:
            alarm_time = t
            break
            
    return alarm_time, g

def cusum_wrapper(X, f0_params=(0,1), f1_params=(1,1), threshold=5, **kwargs):
    startTime = time.time()
    alarm_time, g = cusum(X, f0_params, f1_params, threshold)
    CP = []
    if alarm_time is not None:
        last_zero = np.where(g[:alarm_time] == 0)[0]
        est_cp = last_zero[-1] + 1 if last_zero.size > 0 else 0
        CP.append(est_cp)
    endTime = time.time()
    elapsedTime = endTime - startTime
    vote_counts = np.zeros_like(X) # Estes conceitos não existem no CuSum
    agg_probs = np.zeros_like(X) # Estes conceitos não existem no CuSum
    return CP, elapsedTime, vote_counts, agg_probs

if __name__ == '__main__':
    THRESHOLD = 0.90
    WINDOW_SIZE = 50
    
    cenario_dir = 'teste'
    input_dir = 'time_series/' + cenario_dir
    output_dir = 'changepoints/' + cenario_dir
    
    # cenario_1 = {'m0': 0, 'mb': 0.5, 'mc': -0.5}
    # cenario_2 = {'m0': 0, 'mb': 1.2, 'mc': 0.7}
    # cenario_3 = {'m0': 0, 'mb': 0.5, 'mc': 1}
    # cenario_params = cenario_1
    # detect_changepoints(
    #     input_dir=input_dir,
    #     output_dir=output_dir+'/cusum_b',
    #     detection_func=cusum_wrapper,
    #     default_params={
    #         'f0_params': (cenario_params['m0'], 1),
    #         'f1_params': (cenario_params['mb'], 1),
    #         'threshold': np.log(1000)
    #     },
    # )

    detect_changepoints(
        input_dir=input_dir,
        output_dir=output_dir+'/otima_H_l1',
        detection_func=vwcd_MAP,
        default_params={
            'w': WINDOW_SIZE,
            'vote_p_thr': THRESHOLD,
            'aggreg': 'otima_H',
            'lamb': 1,
            'verbose': False
        },
    )

    # detect_changepoints(
    #     input_dir=input_dir,
    #     output_dir=output_dir+'/otima_H_l0',
    #     detection_func=vwcd,
    #     default_params={
    #         'w': WINDOW_SIZE,
    #         'vote_p_thr': THRESHOLD,
    #         'aggreg': 'otima_H',
    #         'lamb': 0,
    #         'verbose': False
    #     },
    # )

    # detect_changepoints(
    #     input_dir=input_dir,
    #     output_dir=output_dir+'/otima_H_l05',
    #     detection_func=vwcd,
    #     default_params={
    #         'w': WINDOW_SIZE,
    #         'vote_p_thr': THRESHOLD,
    #         'aggreg': 'otima_H',
    #         'lamb': 0.5,
    #         'verbose': False
    #     },
    # )

    # detect_changepoints(
    #     input_dir=input_dir,
    #     output_dir=output_dir+'/otima_H_l01',
    #     detection_func=vwcd,
    #     default_params={
    #         'w': WINDOW_SIZE,
    #         'vote_p_thr': THRESHOLD,
    #         'aggreg': 'otima_H',
    #         'lamb': 0.1,
    #         'verbose': False
    #     },
    # )


    # detect_changepoints(
    #     input_dir=input_dir,
    #     output_dir=output_dir+'/otima_H_l1000',
    #     detection_func=vwcd_MAP,
    #     default_params={
    #         'w': WINDOW_SIZE,
    #         'vote_p_thr': THRESHOLD,
    #         'aggreg': 'otima_H',
    #         'lamb': 1000,
    #         'verbose': False
    #     },
    # )

    detect_changepoints(
        input_dir=input_dir,
        output_dir=output_dir+'/logaritmica_H',
        detection_func=vwcd_MAP,
        default_params={
            'w': WINDOW_SIZE,
            'vote_p_thr': THRESHOLD,
            'aggreg': 'logaritmica_H',
            'verbose': False
        },
    )

    # recalculate_means_and_stds_by_reference(
    #     input_dir=output_dir+'/logaritmica_KL/',
    #     output_dir=output_dir+'/logaritmica_KL_recalc/',
    #     reference_feature='throughput_download'
    # )