import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def segmentar_dataset(input_dir, output_dir, output_filename, feature, max_gap_days=3):
    """
    Cria um dataset de segmentado a partir de séries temporais em formato CSV.
    """
    
    def _get_feature_value(df, col_name, start_idx, end_idx):
        val = df.at[start_idx, col_name]
        if pd.isna(val):
            valid_vals = df.loc[start_idx:end_idx, col_name].dropna()
            return valid_vals.iloc[0] if not valid_vals.empty else np.nan
        return val

    os.makedirs(output_dir, exist_ok=True)
    
    segment_data = []

    for file in os.listdir(input_dir):
        if not file.endswith(".csv"):
            continue
            
        file_path = os.path.join(input_dir, file)
        try:
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['test_time'], errors='coerce')
            df.dropna(subset=['timestamp'], inplace=True)
        except Exception as e:
            print(f"Erro ao ler ou processar 'test_time' no arquivo {file_path}: {e}")
            continue

        client, server = file.split('.')[0].split('_', 1)
        
        df.sort_values(by='timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)

        changepoint_column = f'{feature}_cp'
        if changepoint_column not in df.columns:
            print(f"AVISO: Coluna '{changepoint_column}' não encontrada em {file}. Pulando.")
            continue
        
        changepoint_indices = df.index[df[changepoint_column] == 1].tolist()
        
        # --- CORREÇÃO 1 ---
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds() / (24 * 3600)  # type: ignore
        
        gap_indices = df.index[df['time_diff'] > max_gap_days].tolist()
        
        split_points = sorted(list(set([0, len(df) - 1] + changepoint_indices + gap_indices)))
        
        for start_idx, end_idx in zip(split_points[:-1], split_points[1:]):
            if start_idx >= end_idx:
                continue

            start_time = df.at[start_idx, 'timestamp']
            end_time = df.at[end_idx, 'timestamp']
            
            # --- CORREÇÃO 2 ---
            duration = (end_time - start_time).total_seconds() / (24 * 3600) # type: ignore

            event = 1 if end_idx in changepoint_indices else 0
            if end_idx in gap_indices:
                event = 0

            features_to_get = [
                'throughput_download_local_mean', 'throughput_upload_local_mean',
                'rtt_download_local_mean', 'rtt_upload_local_mean',
                'throughput_download_local_std', 'throughput_upload_local_std',
                'rtt_download_local_std', 'rtt_upload_local_std',
                'packet_loss_local_mean', 'packet_loss_local_std'
            ]
            
            interval_data = {
                'client': client, 'server': server,
                'timestamp_start': start_time, 'timestamp_end': end_time,
                'time': duration, 'event': event
            }
            
            for f_name in features_to_get:
                clean_name = f_name.replace('_local', '')
                if f_name in df.columns:
                    interval_data[clean_name] = _get_feature_value(df, f_name, start_idx, end_idx)
                else:
                    interval_data[clean_name] = np.nan
            
            segment_data.append(interval_data)

    segment_df = pd.DataFrame(segment_data)
    
    if not segment_df.empty:
        segment_df = pd.get_dummies(segment_df, columns=['client', 'server'], dtype=int)
        
        output_path = os.path.join(output_dir, output_filename)
        segment_df.to_csv(output_path, index=False)
        print(f"\n✅ Dataset segmentado salvo em: {output_path}")
    else:
        print("\nNenhum dado foi gerado.")
        
    return segment_df

def clusterizar_e_salvar_kmeans(input_csv_path, n_clusters, output_dir, output_filename):
    """
    Clusteriza dados pré-processados usando K-means e salva o resultado
    adicionando a coluna 'cluster' ao CSV original.

    Esta função assume que as linhas em X_scaled correspondem às linhas
    no arquivo CSV original (ou seja, o pré-processamento não removeu linhas).

    Parâmetros:
    -----------
    input_csv_path : str
        Caminho completo para o arquivo CSV *original* do dataset de sobrevivência
        (o mesmo que foi passado para `preprocessar_dataset_survival`).
    n_clusters : int
        O número de clusters (k) a serem encontrados pelo K-means.
    output_dir : str
        Diretório para salvar o novo CSV com a coluna de clusters.
    output_filename : str
        Nome do arquivo CSV de saída (ex: 'survival_kmeans_clusters.csv').

    Retorna:
    --------
    pd.DataFrame or None
        O DataFrame original com a coluna 'cluster' adicionada, ou None se ocorrer erro.
    """
    
    df = pd.read_csv(input_csv_path)
    client_cols = [col for col in df.columns if col.startswith('client_')]
    server_cols = [col for col in df.columns if col.startswith('server_')]
    onehot_cols = client_cols + server_cols

    features = onehot_cols + ['throughput_download_mean', 'throughput_upload_mean', 'rtt_download_mean', 'rtt_upload_mean', 'throughput_download_std', 'throughput_upload_std', 'rtt_download_std', 'rtt_upload_std', 'packet_loss_mean', 'packet_loss_std']
    X = df[features]

    features_to_scale = ['throughput_download_mean', 'throughput_upload_mean', 'rtt_download_mean', 'rtt_upload_mean', 'throughput_download_std', 'throughput_upload_std', 'rtt_download_std', 'rtt_upload_std', 'packet_loss_mean', 'packet_loss_std']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[features_to_scale])

    print(f"Aplicando K-means com k={n_clusters}...")
    try:
        # Aplica o K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # n_init=10 é o padrão em versões recentes
        clusters = kmeans.fit_predict(X_scaled)
        print("Clusterização K-means concluída.")
    except Exception as e:
        print(f"Erro durante a execução do K-means: {e}")
        return None

    try:
        # Lê o DataFrame original novamente
        df_original = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo original não encontrado em {input_csv_path}")
        return None
    except Exception as e:
        print(f"Erro ao ler o arquivo original {input_csv_path}: {e}")
        return None

    # Verifica se o número de clusters corresponde ao número de linhas
    if len(clusters) != len(df_original):
        print(f"Erro crítico: O número de clusters gerados ({len(clusters)}) não corresponde"
              f" ao número de linhas no CSV original ({len(df_original)}). "
              "Verifique se o pré-processamento removeu linhas.")
        return None

    # Adiciona a coluna de clusters ao DataFrame original
    df_clustered = df_original.copy()
    df_clustered['cluster'] = clusters
    print(f"Distribuição dos clusters:\n{df_clustered['cluster'].value_counts()}")

    # Constrói o caminho de saída e salva
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    try:
        df_clustered.to_csv(output_path, index=False)
        print(f"Dataset com clusters K-means salvo em: {output_path}")
    except Exception as e:
        print(f"Erro ao salvar o arquivo {output_path}: {e}")
        return None

    return df_clustered

if __name__ == "__main__":
    input_name = 'vwcd_all_votes'
    # segmentar_dataset(
    #     input_dir = input_name + '/',
    #     output_dir = 'datasets_segmentados/',
    #     output_filename = 'segmentos_' + input_name + '.csv',
    #     feature = 'throughput_download',
    #     max_gap_days = 3)

    clusterizar_e_salvar_kmeans(
        input_csv_path='datasets_segmentados/segmentos_' + input_name + '.csv',
        n_clusters=4,
        output_dir='clusters',
        output_filename='clusters_' + input_name + '_4s.csv',
    )

