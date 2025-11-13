from VWCD import *
import pandas as pd
import numpy as np
import ruptures as rpt
import time
import os

def pelt_wrapper(X, model="rbf", pen=3, **kwargs):
    """
    Função adaptadora (wrapper) para usar o algoritmo PELT com a função
    genérica detect_changepoints_generic.

    Esta função calcula os changepoints, as médias/desvios dos segmentos,
    e retorna os outros valores como placeholders para manter a compatibilidade.

    Parâmetros:
    -----------
    X : np.ndarray
        A série temporal.
    model : str, optional
        O modelo de custo para o PELT ('rbf', 'l1', 'l2', etc.). Padrão 'rbf'.
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

    # 1. Executar o algoritmo PELT
    algo = rpt.Pelt(model=model).fit(X)
    result = algo.predict(pen=pen)
    
    # O resultado do PELT inclui o final da série, que removemos para compatibilidade
    CP = np.array(result[:-1]).astype(int)

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

                # if len(CP) > 0:
                #     cps = CP.astype(int)
                #     all_cps = np.concatenate(([0], cps, [len(y)]))
                #     for i in range(len(all_cps) - 1):
                #         start_idx = all_cps[i]
                #         end_idx = all_cps[i+1]
                #         if i < len(M0):
                #             local_means[start_idx:end_idx] = M0[i]
                #             local_stds[start_idx:end_idx] = S0[i]
                # else:
                #     if len(y) > 0:
                #         local_means[:] = np.mean(y)
                #         local_stds[:] = np.std(y, ddof=1) if len(y) > 1 else 0

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

if __name__ == '__main__':
    THRESHOLD = 0.98
    WINDOW_SIZE = 24
    
    detect_changepoints(
        input_dir='time_series',
        output_dir='changepoints/pelt',
        detection_func=pelt_wrapper,
        default_params={
            'w': WINDOW_SIZE,
        },
    )

    detect_changepoints(
        input_dir='time_series',
        output_dir='changepoints/mean',
        detection_func=vwcd,
        default_params={
            'w': WINDOW_SIZE,
            'vote_p_thr': THRESHOLD,
            'aggreg': 'mean',
            'verbose': False
        },
    )

    detect_changepoints(
        input_dir='time_series',
        output_dir='changepoints/multiplicativa',
        detection_func=vwcd,
        default_params={
            'w': WINDOW_SIZE,
            'vote_p_thr': THRESHOLD,
            'aggreg': 'multiplicativa',
            'verbose': False
        },
    )


    detect_changepoints(
        input_dir='time_series',
        output_dir='changepoints/logaritmica_KL',
        detection_func=vwcd,
        default_params={
            'w': WINDOW_SIZE,
            'vote_p_thr': THRESHOLD,
            'aggreg': 'logaritmica_KL',
            'verbose': False
        },
    )

    detect_changepoints(
        input_dir='time_series',
        output_dir='changepoints/logaritmica_H',
        detection_func=vwcd,
        default_params={
            'w': WINDOW_SIZE,
            'vote_p_thr': THRESHOLD,
            'aggreg': 'logaritmica_H',
            'verbose': False
        },
    )

    # recalculate_means_and_stds_by_reference(
    #     input_dir='changepoints/logaritmica_KL/',
    #     output_dir='changepoints/logaritmica_KL_recalc/',
    #     reference_feature='throughput_download'
    # )



