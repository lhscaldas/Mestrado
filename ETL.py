import pandas as pd
import numpy as np
import os

def transform_artificial_csv_files(folder_path):
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    
    for idx, file in enumerate(files, start=1):
        file_path = os.path.join(folder_path, file)
        
        df = pd.read_csv(file_path)
        df = df[['timestamp', 'value', 'value_cp']]
        
        new_file_name = f'teste{idx:02d}.csv'
        new_file_path = os.path.join(folder_path, new_file_name)
        
        df.to_csv(new_file_path, index=False)
        
        if file_path != new_file_path:
            os.remove(file_path)

def clean_and_transform_NDT(path_folder: str, file_name: str):
    # Carrega o CSV
    df = pd.read_csv(path_folder + f'/{file_name}')

    # ORDENA pelo timestamp
    df = df.sort_values(by='timestamp')

    # Pega apenas as coluna ["timestamp","client_ip","server_ip","download_tp_bps","latency_download_sec","upload_tp_bps","latency_upload_sec","mac_address","download_retrans_percent","test_uuid"]
    df = df[["timestamp","client_ip","server_ip","download_tp_bps","latency_download_sec","upload_tp_bps","latency_upload_sec","mac_address","download_retrans_percent","test_uuid"]]

    # Renomeia a coluna download_retrans_percent para loss_rate
    df = df.rename(columns={"download_retrans_percent": "loss_rate"})

    # Criando a coluna nome do cliente a partir do MAC
    df_device = pd.read_csv(path_folder + '/devices.csv')
    df['client_name'] = df['mac_address'].map(df_device.set_index('mac')['owner'])

    # Criando a coluna nome do servidor a partir do IP
    df_server = pd.read_csv(path_folder + '/servers.csv')
    df['server_name'] = df['server_ip'].map(df_server.set_index('server_ip')['name'])

    # Convertendo download_tp_bps e upload_tp_bps para Mbps
    df['download_tp_Mbps'] = df['download_tp_bps'] / 1_000_000
    df['upload_tp_Mbps'] = df['upload_tp_bps'] / 1_000_000

    # Convertendo latency_download_sec e latency_upload_sec para ms
    df['latency_download_ms'] = df['latency_download_sec'] * 1000
    df['latency_upload_ms'] = df['latency_upload_sec'] * 1000

    # Verifica se há colunas com valores nulos, printa o quantitativo e remove essas linhas
    null_counts = df.isnull().sum()
    print("Contagem de valores nulos por coluna:")
    print(null_counts[null_counts > 0])
    df = df.dropna()

    # Verifica se há colunas com valores negativos, printa e remove essas linhas
    negative_conditions = (df[['download_tp_bps', 'latency_download_ms', 'upload_tp_bps', 'latency_upload_ms', 'loss_rate']] < 0).any(axis=1)
    negative_counts = negative_conditions.sum()
    print(f"Quantidade de linhas com valores negativos: {negative_counts}")
    df = df[~negative_conditions]

    # Remove os clientes LandTeste e Gigalink
    df = df[~df['client_name'].isin(['LandTeste', 'Gigalink'])]

    # Reordenando as colunas de forma lógica
    logical_order = [
        'timestamp', 'test_uuid', 
        'client_name', 'client_ip', 'mac_address',
        'server_name', 'server_ip', 
        'download_tp_Mbps', 'download_tp_bps', 'latency_download_ms',
        'upload_tp_Mbps', 'upload_tp_bps', 'latency_upload_ms',
        'loss_rate'
    ]
    df = df[logical_order]

    # Salva em outro CSV
    clean_file_name = file_name.replace('raw', 'clean')
    clean_file_path = os.path.join(path_folder, clean_file_name)
    df.to_csv(clean_file_path, index=False)

def export_time_series_NDT(df_pandas, output_dir, metadata_csv_filename):

    os.makedirs(output_dir, exist_ok=True)

    clients = df_pandas['client_name'].unique()
    sites = df_pandas['server_name'].unique()
    med = []

    # converte timestamp para datetime
    df_pandas['timestamp'] = pd.to_datetime(df_pandas['timestamp'])

    for c in clients:
        for s in sites:
            df_pair = df_pandas[(df_pandas.client_name == c) & (df_pandas.server_name == s)]
                      
            if len(df_pair) >= 100:
                df_ts = pd.DataFrame({
                    'timestamp': df_pair['timestamp'].values,
                    'rtt_download': df_pair['latency_download_ms'].values,
                    'throughput_download': df_pair['download_tp_Mbps'].values,
                    'rtt_upload': df_pair['latency_upload_ms'].values,
                    'throughput_upload': df_pair['upload_tp_Mbps'].values,
                    'packet_loss': df_pair['loss_rate'].values
                })
                df_ts.sort_values(by='timestamp', inplace=True)

                output_file = f"{output_dir}/{c}_{s}.csv" 

                # verifica se o df_ts possui timestamps repetidos e, se sim, pular a exportação desse arquivo
                if df_ts['timestamp'].duplicated().any():
                    print(f"Série temporal para cliente '{c}' e site '{s}' possui timestamps duplicados. Pulando a exportação deste arquivo.")
                    continue

                else:
                    df_ts.to_csv(output_file, index=False) 

                    df_pair_sorted = df_pair.sort_values(by='timestamp')
                    inicio = df_pair_sorted['timestamp'].iloc[0]
                    fim = df_pair_sorted['timestamp'].iloc[-1]
                    num_med = len(df_pair)
                    mean_time = np.round(df_pair_sorted['timestamp'].diff().mean().total_seconds() / 3600, 1)
                    file_prefix = f"{c}_{s}"
                    
                    quant = {
                        "client": c, "site": s, "inicio": inicio, "fim": fim,
                        "num_med": num_med, "mean_time": mean_time, "file_prefix": file_prefix
                    }
                    med.append(quant)

    df_metadata = pd.DataFrame(med)
    df_metadata.to_csv(metadata_csv_filename, index=False)
    
    print(f"Metadados salvos com sucesso em: {metadata_csv_filename}")
    print(f"Séries temporais (.csv) salvas em: {output_dir}")

def split_NDT_metrics_csv(input_folder: str):
    col_to_suffix = {
        'rtt_download': 'rtt_down',
        'throughput_download': 'tp_down',
        'rtt_upload': 'rtt_up',
        'throughput_upload': 'tp_up',
        'packet_loss': 'pl'
    }
    
    input_folder = input_folder.rstrip('/\\')
    parent_dir = os.path.dirname(input_folder)
    
    output_folders = {}
    for col, suffix in col_to_suffix.items():
        new_folder = os.path.join(parent_dir, suffix)
        
        os.makedirs(new_folder, exist_ok=True)
        output_folders[col] = new_folder
        
    for file in os.listdir(input_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(input_folder, file)
            df = pd.read_csv(file_path)

            # Verificar se o df possui timestamps repetidos e, se sim, pular o arquivo
            if df['timestamp'].duplicated().any():
                print(f"Arquivo '{file}' possui timestamps duplicados. Pulando este arquivo.")
                continue

            # Verificar se o df resultante está vazio após o filtro e, se sim, pular o arquivo
            if df.empty:
                print(f"Arquivo '{file}' não possui dados para o mês de outubro. Pulando este arquivo.")
                continue
            
            for col, out_folder in output_folders.items():
                if col in df.columns:
                    df_subset = df[['timestamp', col]]
                    out_path = os.path.join(out_folder, file)
                    df_subset.to_csv(out_path, index=False)            


if __name__ == "__main__":
    path_folder = 'NDT dataset'
    file_name = 'NOV_ABR_raw.csv'
    clean_and_transform_NDT(path_folder, file_name)

    # Carrega o CSV limpo
    clean_path = os.path.join(path_folder, file_name.replace('raw', 'clean'))
    df_clean = pd.read_csv(clean_path)

    # Exporta as séries temporais e metadados
    output_dir = os.path.join('series', 'NDT_NOV_ABR', 'full')
    export_time_series_NDT(
        df_pandas=df_clean,
        output_dir=output_dir, 
        metadata_csv_filename=clean_path.replace('clean', 'metadata')
    )

    split_NDT_metrics_csv(input_folder=output_dir)