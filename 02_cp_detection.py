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
    """
    Função adaptadora (wrapper) para usar o algoritmo PELT com a função
    genérica detect_changepoints_generic.

    Esta função calcula os changepoints, as médias/desvios dos segmentos,
    e retorna os outros valores como placeholders para manter a compatibilidade.

    Parâmetros:
    -----------
    X : np.ndarray
        A série temporal.
    pen : int or float, optional
        A penalidade para o algoritmo PELT. Padrão 3.
    **kwargs : dict
        Argumentos adicionais (ignorados pelo PELT, mas necessários para a interface).

    Retorna:
    --------
    tuple
        Uma tupla de 7 elementos compatível com a função genérica.
    """
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

    # 3. Criar placeholders para as outras saídas
    # Estes conceitos não existem no PELT
    vote_counts = np.zeros_like(X)
    agg_probs = np.zeros_like(X)

    return CP.tolist(), elapsedTime, vote_counts, agg_probs

def detect_changepoints(input_dir, output_dir, detection_func, default_params):
    """
    Executa uma função de detecção de changepoints em colunas numéricas de arquivos CSV,
    permitindo parâmetros específicos por tipo de variável.

    Parâmetros:
    ----------
    input_dir : str
        Diretório contendo os arquivos CSV.
    output_dir : str
        Diretório para salvar os resultados.
    detection_func : function
        A função de detecção a ser usada (ex: vwcd, pelt_wrapper).
    default_params : dict
        Dicionário com os parâmetros padrão a serem usados para todas as variáveis.

    Retorna:
    -------
    None
        Salva novos arquivos CSV no `output_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)
    if 'w' not in default_params:
        raise ValueError("O tamanho da janela 'w' deve ser fornecido em default_params.")
    w_size = default_params['w']

    for file in os.listdir(input_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(input_dir, file)
            df = pd.read_csv(file_path, parse_dates=['timestamp'])

            for column in df.select_dtypes(include=[np.number]).columns:
                y = df[column].dropna().to_numpy()

                if len(y) < w_size:
                    continue

                current_params = default_params.copy()
                
                current_params['X'] = y

                CP, elapsedTime, vote_counts, agg_probs = detection_func(**current_params)
                CP = np.array(CP)
                
                changepoints = np.zeros(len(y), dtype=int)
                if len(CP) > 0:
                    changepoints[CP.astype(int)] = 1

                local_means = np.zeros(len(y))
                local_stds = np.zeros(len(y))

                # Recalcular M0 e S0 manualmente
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

def cusum_standard_benchmark(X, log_pdf_alvo, log_pdf_base, threshold):
    """
    Implementação do CUSUM Padrão (Benchmark do Artigo).
    Baseado na Eq. (5): S[t] = max(0, S[t-1] + log(f_alvo(x)/f_base(x)))
    
    Parâmetros:
    ----------
    X : array-like
        A série temporal de observações.
    log_pdf_alvo : function
        Função que retorna o log da densidade da hipótese de mudança (ex: fB).
    log_pdf_base : function
        Função que retorna o log da densidade da hipótese atual/normal (ex: f0).
    threshold : float
        Limiar de detecção (h ou b).
        
    Retorno:
    -------
    alarm_time : int ou None
        Índice onde o alarme disparou (primeiro t tal que g_t >= h).
    trajectory : np.array
        A trajetória da estatística g_t para visualização.
    """
    n = len(X)
    g = np.zeros(n)
    alarm_time = None
    
    # O artigo inicializa em 0 (Eq. 4)
    current_g = 0.0 
    
    for t in range(n):
        # 1. Calcula a Log-Likelihood Ratio Instantânea
        # Z_t = ln( f_alvo(X_t) / f_base(X_t) )
        llr = log_pdf_alvo(X[t]) - log_pdf_base(X[t])
        
        # 2. Atualização Recursiva (Fórmula de Lindley)
        # g_t = max(0, g_{t-1} + Z_t)
        current_g = max(0, current_g + llr)
        g[t] = current_g
        
        # 3. Verificação do Limiar
        if current_g >= threshold and alarm_time is None:
            alarm_time = t
            # Em benchmarks de "Quickest Detection", geralmente paramos no primeiro alarme.
            # Se quiser continuar monitorando, deve-se resetar current_g = 0 aqui.
            break
            
    return alarm_time, g

# # --- Exemplo de Configuração (Cenário Gaussiano) ---
# if __name__ == "__main__":
#     # Definição auxiliar para log-pdf Gaussiana
#     def gaussian_log_pdf(x, mean, std):
#         return -0.5 * np.log(2 * np.pi) - np.log(std) - 0.5 * ((x - mean) / std)**2

#     # Parâmetros do "Cenário 1" do artigo (Página 6, VI. Numerical Results) [cite: 312]
#     # f0: Normal(0, 1)
#     # fB: Normal(0.5, 1)
    
#     # Funções lambda prontas para passar ao algoritmo
#     f0_log = lambda x: gaussian_log_pdf(x, mean=0.0, std=1.0)
#     fB_log = lambda x: gaussian_log_pdf(x, mean=0.5, std=1.0)
    
#     # Gerando dados sintéticos para teste
#     # 50 amostras normais (f0) + 50 amostras com mudança (fB)
#     np.random.seed(42)
#     dados = np.concatenate([
#         np.random.normal(0, 1, 50),
#         np.random.normal(0.5, 1, 50)
#     ])
    
#     # Rodando o Benchmark CuSum(fB, f0)
#     t_alarme, trajetoria = cusum_standard_benchmark(dados, fB_log, f0_log, threshold=5.0)
    
#     print(f"Mudança real inicia em: t=50")
#     print(f"Alarme disparado em: t={t_alarme}")

if __name__ == '__main__':
    THRESHOLD = 0.98
    WINDOW_SIZE = 24
    input_dir = 'artificial_time_series'
    output_dir = 'artificial_changepoints'
    # input_dir = 'time_series'
    # output_dir = 'changepoints'
    
    # detect_changepoints(
    #     input_dir=input_dir,
    #     output_dir=output_dir+'/pelt_ed',
    #     detection_func=pelt_wrapper,
    #     default_params={
    #         'w': WINDOW_SIZE,
    #         'mode': 'ed'
    #     },
    # )

    # detect_changepoints(
    #     input_dir=input_dir,
    #     output_dir=output_dir+'/pelt_rbf_bic',
    #     detection_func=pelt_wrapper,
    #     default_params={
    #         'w': WINDOW_SIZE,
    #         'mode': 'rbf',
    #         'penalty':'BIC'
    #     },
    # )

    # detect_changepoints(
    #     input_dir=input_dir,
    #     output_dir=output_dir+'/pelt_rbf_aic',
    #     detection_func=pelt_wrapper,
    #     default_params={
    #         'w': WINDOW_SIZE,
    #         'mode': 'rbf',
    #         'penalty':'AIC'
    #     },
    # )

    # detect_changepoints(
    #     input_dir=input_dir,
    #     output_dir=output_dir+'/pelt_rbf_p3',
    #     detection_func=pelt_wrapper,
    #     default_params={
    #         'w': WINDOW_SIZE,
    #         'mode': 'rbf',
    #         'penalty':None
    #     },
    # )


    detect_changepoints(
        input_dir=input_dir,
        output_dir=output_dir+'/mean',
        detection_func=vwcd,
        default_params={
            'w': WINDOW_SIZE,
            'vote_p_thr': THRESHOLD,
            'aggreg': 'mean',
            'verbose': False
        },
    )

    detect_changepoints(
        input_dir=input_dir,
        output_dir=output_dir+'/multiplicativa',
        detection_func=vwcd,
        default_params={
            'w': WINDOW_SIZE,
            'vote_p_thr': THRESHOLD,
            'aggreg': 'multiplicativa',
            'verbose': False
        },
    )


    detect_changepoints(
        input_dir=input_dir,
        output_dir=output_dir+'/logaritmica_KL',
        detection_func=vwcd,
        default_params={
            'w': WINDOW_SIZE,
            'vote_p_thr': THRESHOLD,
            'aggreg': 'logaritmica_KL',
            'verbose': False
        },
    )

    detect_changepoints(
        input_dir=input_dir,
        output_dir=output_dir+'/logaritmica_H',
        detection_func=vwcd,
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



