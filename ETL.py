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

def NDT_transform(path_folder: str, file_name: str):
    # Carrega o CSV
    df = pd.read_csv(path_folder + f'/{file_name}')

    # ORDENA pelo timestamp
    df = df.sort_values(by='timestamp')

    # Pega apenas as coluna ["timestamp","client_ip","server_ip","download_tp_bps","latency_download_sec","upload_tp_bps","latency_upload_sec","mac_address","download_retrans_percent","test_uuid"]
    df = df[["timestamp","client_ip","server_ip","server_fqdn","download_tp_bps","latency_download_sec","upload_tp_bps","latency_upload_sec","mac_address","download_retrans_percent","test_uuid"]]

    # Renomeia a coluna download_retrans_percent para loss_rate
    df = df.rename(columns={"download_retrans_percent": "loss_rate"})

    # Criando a coluna nome do cliente a partir do MAC
    df_device = pd.read_csv(path_folder + '/devices.csv')
    df['client_name'] = df['mac_address'].map(df_device.set_index('mac')['owner'])
    df['tipo'] = df['mac_address'].map(df_device.set_index('mac')['tipo'])

    # Criando a coluna nome do servidor a partir do fqdn
    df_server = pd.read_csv(path_folder + '/servers.csv')
    df['server_name'] = df['server_fqdn'].map(df_server.set_index('server_fqdn')['apelido'])

    # Mapeamento para garantir consistência e manter a distribuição
    mask_client = df['client_name'].notna()
    client_map = {val: f"client{i+1:02d}" for i, val in enumerate(df.loc[mask_client, 'client_name'].unique())}
    df.loc[mask_client, 'client_name'] = df.loc[mask_client, 'client_name'].map(client_map)
    mask_server = df['server_name'].notna()
    server_map = {val: f"server{i+1:02d}" for i, val in enumerate(df.loc[mask_server, 'server_name'].unique())}
    df.loc[mask_server, 'server_name'] = df.loc[mask_server, 'server_name'].map(server_map)

    # Salvar o mapeamento para possível reversão
    import json
    mapping_data = {'client_map': client_map, 'server_map': {v: k for k, v in server_map.items()}} #
    mapping_data = {'client_map': {v: k for k, v in client_map.items()}, 'server_map': {v: k for k, v in server_map.items()}}
    with open(os.path.join(path_folder, 'mapping_log.json'), 'w') as f:
        json.dump(mapping_data, f, indent=4)

    # Convertendo download_tp_bps e upload_tp_bps para Mbps e removendo as colunas originais
    df['download_tp_Mbps'] = df['download_tp_bps'] / 1_000_000
    df['upload_tp_Mbps'] = df['upload_tp_bps'] / 1_000_000
    df = df.drop(['download_tp_bps', 'upload_tp_bps'], axis=1)

    # Convertendo latency_download_sec e latency_upload_sec para ms e removendo as colunas originais
    df['latency_download_ms'] = df['latency_download_sec'] * 1000
    df['latency_upload_ms'] = df['latency_upload_sec'] * 1000
    df = df.drop(['latency_download_sec', 'latency_upload_sec'], axis=1)

    # Reordenando as colunas de forma lógica
    logical_order = [
        'timestamp', 'test_uuid', 'tipo',
        'client_name', 'client_ip', 'mac_address',
        'server_name', 'server_ip', 'server_fqdn',
        'upload_tp_Mbps', 'latency_upload_ms', 
        'download_tp_Mbps', 'latency_download_ms', 'loss_rate'
    ]
    df = df[logical_order]

    # Renomeando para rtt_download,throughput_download,rtt_upload,throughput_upload,packet_loss
    df = df.rename(columns={
        'latency_download_ms': 'rtt_download',
        'download_tp_Mbps': 'throughput_download',
        'latency_upload_ms': 'rtt_upload',
        'upload_tp_Mbps': 'throughput_upload',
        'loss_rate': 'packet_loss'
    })

    # Salva em outro CSV
    clean_file_name = file_name.replace('raw', 'transformed')
    clean_file_path = os.path.join(path_folder, clean_file_name)
    df.to_csv(clean_file_path, index=False)

def NDT_clean(
    path_folder: str, 
    file_name: str,
    limite_horas: int,
    seg_min: int
):
    
    metricas = [
        'rtt_download', 'throughput_download', 
        'rtt_upload', 'throughput_upload', 'packet_loss'
    ]
        
    # Carrega o CSV
    file_path = os.path.join(path_folder, file_name)
    df = pd.read_csv(file_path)

    # manter apenas o tipo 'raspberry'
    df = df[df['tipo'] == 'raspberry'].copy()

    # remover os valores nulos e negativos antes das lógicas de tempo
    df = df.dropna()
    for col in metricas:
        if col in df.columns:
            df = df[df[col] >= 0]

    # Criar coluna de par cliente-servidor
    df['client_server_pair'] = df['client_name'] + " -> " + df['server_name']

    # Conversão de timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['data'] = df['timestamp'].dt.date

    # 1. Remover duplicatas ANTES do cálculo de tempo
    duplicatas = df.duplicated(subset=['client_server_pair', 'timestamp'], keep=False)
    df = df[~duplicatas].copy()

    # 2. Ordenar por tempo e par
    df = df.sort_values(by=['client_server_pair', 'timestamp']).reset_index(drop=True)

    # 3. Criar coluna com a diferença de tempo (agora os pulos de tempo serão reais)
    df['delta_tempo_horas'] = df.groupby('client_server_pair')['timestamp'].diff().dt.total_seconds() / 3600

    # filtrar seguimentos curtos
    condicao_quebra = df['delta_tempo_horas'] > limite_horas
    df['id_bloco'] = condicao_quebra.groupby(df['client_server_pair']).cumsum()
    df['tamanho_bloco'] = df.groupby(['client_server_pair', 'id_bloco'])['id_bloco'].transform('count')
    df = df[df['tamanho_bloco'] >= seg_min].copy()
    df['id_bloco'] = df.groupby('client_server_pair')['id_bloco'].rank(method='dense').astype(int)

    # Salva em outro CSV
    clean_file_name = file_name.replace('transformed', 'clean')
    clean_file_path = os.path.join(path_folder, clean_file_name)
    df.to_csv(clean_file_path, index=False)


def NDT_export(df_pandas, output_dir, metadata_csv_filename):

    os.makedirs(output_dir, exist_ok=True)

    clients = df_pandas['client_name'].unique()
    sites = df_pandas['server_name'].unique()
    med = []

    # converte timestamp para datetime
    df_pandas['timestamp'] = pd.to_datetime(df_pandas['timestamp'])

    for c in clients:
        for s in sites:
            df_pair = df_pandas[(df_pandas.client_name == c) & (df_pandas.server_name == s)]
                      
            df_ts = pd.DataFrame({
                'timestamp': df_pair['timestamp'].values,
                'rtt_download': df_pair['rtt_download'].values,
                'throughput_download': df_pair['throughput_download'].values,
                'rtt_upload': df_pair['rtt_upload'].values,
                'throughput_upload': df_pair['throughput_upload'].values,
                'packet_loss': df_pair['packet_loss'].values
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

def NDT_split(input_folder: str):
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
                print(f"Arquivo '{file}' não possui dados para o período. Pulando este arquivo.")
                continue
            
            for col, out_folder in output_folders.items():
                if col in df.columns:
                    df_subset = df[['timestamp', col]]
                    out_path = os.path.join(out_folder, file)
                    df_subset.to_csv(out_path, index=False)            


if __name__ == "__main__":
    # Transforma o CSV bruto em um CSV com colunas mais amigáveis e unidades convertidas
    path_folder = 'NDT dataset'
    file_name = 'NDT_raw.csv'
    # NDT_transform(path_folder, file_name)

    # Limpa o CSV transformado, removendo linhas com valores nulos ou negativos, e outros tipos de limpeza
    NDT_clean(
    path_folder=path_folder, 
    file_name=file_name.replace('raw', 'transformed'),
    limite_horas=12,
    seg_min=1000
    )

    # Exporta as séries temporais e metadados
    # clean_path = os.path.join(path_folder, file_name.replace('raw', 'clean'))
    # df_clean = pd.read_csv(clean_path)
    # output_dir = os.path.join('series', 'NDT_NOV_ABR', 'full')
    # NDT_export(
    #     df_pandas=df_clean,
    #     output_dir=output_dir, 
    #     metadata_csv_filename=clean_path.replace('clean', 'metadata')
    # )

    # # Separa as séries temporais em arquivos distintos por métrica
    # split_NDT_metrics_csv(input_folder=output_dir)