import pandas as pd
import numpy as np
import os

def clean_and_transform(path_folder: str):
    # Carrega o CSV
    df = pd.read_csv(path_folder + '/ndt_tests_raw.csv')

    # ORDENA pelo timestamp
    df = df.sort_values(by='timestamp')

    # Pega apenas as coluna ["timestamp","client_ip","server_ip","download_tp_bps","latency_download_sec","upload_tp_bps","latency_upload_sec","mac_address","download_retrans_percent","test_uuid"]
    df = df[["timestamp","client_ip","server_ip","download_tp_bps","latency_download_sec","upload_tp_bps","latency_upload_sec","mac_address","download_retrans_percent","test_uuid"]]

    # Filtra a coluna timestamp até 2025-08-31
    df = df[df['timestamp'] <= '2025-08-31']

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
    df.to_csv(path_folder + '/ndt_tests_clean.csv', index=False)

def export_time_series(df_pandas, output_dir, metadata_csv_filename):
    """
    Exporta séries temporais das variáveis de download e upload para cada par cliente-servidor.

    Para cada par de cliente e servidor no DataFrame de entrada:
    - Cria um DataFrame contendo a série temporal das métricas de download e upload.
    - Cada DataFrame de série temporal é salvo em um arquivo .csv individual.
    - Gera metadados consolidados, que são salvos em um único arquivo .csv.

    Parâmetros:
    ----------
    df_pandas : pd.DataFrame
        O DataFrame do pandas contendo os dados de entrada.
    output_dir : str
        O caminho do diretório onde os arquivos de séries temporais (.csv) serão salvos.
        Este diretório será criado se não existir.
    metadata_csv_filename : str
        O nome completo do arquivo (com caminho, se necessário) para salvar os metadados em formato CSV.

    Retorno:
    -------
    pd.DataFrame
        Um DataFrame contendo os metadados das séries temporais, incluindo:
        - client (str): Identificação do cliente.
        - site (str): Identificação do servidor.
        - inicio (datetime): Timestamp da primeira medição.
        - fim (datetime): Timestamp da última medição.
        - num_med (int): Número de medições.
        - mean_time (float): Intervalo médio entre medições, em horas.
        - file_prefix (str): Prefixo usado nos nomes dos arquivos gerados.

    Arquivos gerados:
    -----------------
    - Séries temporais para cada par cliente-servidor: <client>_<site>.csv
    - Metadados: O arquivo especificado no parâmetro `metadata_csv_filename`.
    """
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

if __name__ == "__main__":
    path_folder = 'datasets'
    clean_and_transform(path_folder)

    # Contagem de valores nulos por coluna:
    # client_name     3505
    # server_name    15210
    # Quantidade de linhas com valores negativos: 466

    # Carrega o CSV limpo
    df_clean = pd.read_csv(path_folder + '/ndt_tests_clean.csv')

    # Exporta as séries temporais e metadados
    export_time_series(
        df_pandas=df_clean,
        output_dir='time_series', 
        metadata_csv_filename=path_folder + '/metadata_time_series.csv'
    )