import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import re

def cluster_and_save_results(scenario, method, ref_metric, threshold, n_clusters=3, clean=False):
    # 1. Caminhos de entrada e saída
    input_dir = os.path.join("features", scenario)
    output_dir = os.path.join("clusters", scenario)
    os.makedirs(output_dir, exist_ok=True)
    
    file_name = f"features_{method}_{ref_metric}_{threshold}.csv"
    input_path = os.path.join(input_dir, file_name)
    
    if not os.path.exists(input_path):
        print(f"Erro: Arquivo {input_path} not found.")
        return

    # 2. Carregamento dos dados
    df = pd.read_csv(input_path)
    feature_cols = ['d_rtt_down', 'd_tp_down', 'd_rtt_up', 'd_tp_up', 'd_pl', 'sync_score']
    df['d_rtt_down'] = df['d_rtt_down_rel']

    # 3. Pré-processamento
    df_clean = df.dropna().copy()
    corte = 10 # ms
     
    if clean:
        df_clean = df_clean[
        ((df_clean['d_rtt_down_abs'] > 0) & (df_clean['d_rtt_down_abs'] >= corte)) |
        ((df_clean['d_rtt_down_abs'] < 0) & (df_clean['d_rtt_down_abs'] <= -corte))|
        ((df_clean['d_rtt_up_abs'] > 0) & (df_clean['d_rtt_up_abs'] >= corte)) |
        ((df_clean['d_rtt_up_abs'] < 0) & (df_clean['d_rtt_up_abs'] <= -corte))  
        ] 
        low = df_clean[feature_cols].quantile(0.01)
        high = df_clean[feature_cols].quantile(0.95)
        df_clean = df_clean[((df_clean[feature_cols] >= low) & (df_clean[feature_cols] <= high)).all(axis=1)]

    df_clean = df_clean.drop(columns=['d_rtt_down_abs', 'd_rtt_up_abs', 'd_rtt_down_rel', 'd_rtt_up_rel'])
    
    # 4. Normalização
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df_clean[feature_cols])
    
    # 5. Treinamento do GMM
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
    df_clean['cluster'] = gmm.fit_predict(x_scaled)

    # 6. Salvamento do CSV
    base_output_name = f"clusters_K{n_clusters:02d}_{file_name}"
    csv_path = os.path.join(output_dir, base_output_name)
    df_clean.to_csv(csv_path, index=False)

    # 7. Geração do Relatório Técnico (TXT)
    report_path = csv_path.replace(".csv", ".txt")
    with open(report_path, "w") as f:
        f.write(f"RELATORIO DE CLUSTERIZACAO GMM - {scenario}\n")
        f.write("="*60 + "\n")
        f.write(f"Metrica de Referencia: {ref_metric}\n")
        f.write(f"Algoritmo CPD: {method}\n")
        f.write(f"Numero de Clusters (K): {n_clusters}\n")
        f.write(f"Total de Eventos: {len(df_clean)}\n\n")

        # Metricas de Ajuste do Modelo
        f.write("METRICAS DE AJUSTE (Qualidade do Modelo):\n")
        f.write(f"AIC (Akaike Information Criterion): {gmm.aic(x_scaled):.2f}\n")
        f.write(f"BIC (Bayesian Information Criterion): {gmm.bic(x_scaled):.2f}\n")
        f.write(f"Log-Likelihood: {gmm.score(x_scaled) * len(df_clean):.2f}\n\n")

        f.write("RESUMO DOS CLUSTERS:\n")
        f.write("-" * 30 + "\n")
        counts = df_clean['cluster'].value_counts().sort_index()
        for cluster_id, count in counts.items():
            f.write(f"Cluster {cluster_id}:\n")
            f.write(f"  - Quantidade de Eventos: {count} ({count/len(df_clean)*100:.1f}%)\n")
            f.write(f"  - Peso (Mix Proportion): {gmm.weights_[cluster_id]:.4f}\n\n")  # type: ignore
            
            # Médias (des-normalizadas para facilitar interpretação física)
            # scaler.inverse_transform converte de volta para a escala original (ms, %, etc)
            means_orig = scaler.inverse_transform(gmm.means_)
            f.write("  - Medias (Escala Original):\n")
            for i, col in enumerate(feature_cols):
                f.write(f"    {col}: {means_orig[cluster_id][i]:.4f}\n") # type: ignore
            
            f.write("\n  - Matriz de Covariancia (Espaco Z-Score):\n")
            f.write(str(np.round(gmm.covariances_[cluster_id], 4)) + "\n") # type: ignore
            f.write("-" * 30 + "\n")

    print(f"Processamento concluído para {ref_metric}.")
    print(f"CSV salvo em: {csv_path}")
    print(f"Relatorio salvo em: {report_path}")

def compile_cluster_reports(scenario, method, ref_metric, threshold):
    path = os.path.join("clusters", scenario)
    if not os.path.exists(path):
        print(f"Erro: Pasta {path} não encontrada.")
        return

    data = []
    
    # Regex para capturar K do nome do arquivo e os valores dentro do TXT
    file_pattern = rf"clusters_K(\d+)_features_{method}_{ref_metric}_{threshold}\.txt"    

    for file in os.listdir(path):
        match = re.match(file_pattern, file)
        if match:
            k_val = int(match.group(1))
            full_path = os.path.join(path, file)
            
            with open(full_path, 'r') as f:
                content = f.read()
                
                # Extração via Regex dos valores numéricos
                aic = re.search(r"AIC.*: ([\d\.-]+)", content)
                bic = re.search(r"BIC.*: ([\d\.-]+)", content)
                ll  = re.search(r"Log-Likelihood: ([\d\.-]+)", content)
                events = re.search(r"Total de Eventos: (\d+)", content)
                
                if aic and bic and ll:
                    data.append({
                        'K': k_val,
                        'Log-Likelihood': float(ll.group(1)),
                        'AIC': float(aic.group(1)),
                        'BIC': float(bic.group(1)),

                    })

    if not data:
        print("Nenhum relatório encontrado para os parâmetros fornecidos.")
        return

    # Criar DataFrame e ordenar por K
    df = pd.DataFrame(data).sort_values('K').reset_index(drop=True)
    
    # Calcular a diferença (Delta) entre o K atual e o anterior para ver o ganho
    df['Delta_BIC'] = df['BIC'].diff()
    df['Delta_LL']  = df['Log-Likelihood'].diff()

    print(f"\nCOMPARAÇÃO DE MODELOS - CENÁRIO: {scenario}")
    print(f"Métrica: {ref_metric} | Threshold: {threshold}")
    print("=" * 85)
    print(df.to_string(index=False))
    print("=" * 85)
    print("\n* Delta_BIC negativo indica ganho de qualidade.")
    print("* Delta_LL positivo indica melhor ajuste aos dados.") 

if __name__ == "__main__":
    scenario = "NDT"
    method = "vwcd_fp1"
    ref_metric = "rtt_down"
    threshold = 0.95
    for k in range(2, 11):
        cluster_and_save_results(
            scenario=scenario,
            method=method,
            ref_metric=ref_metric,
            threshold=threshold,
            n_clusters=k,
            clean=True
        )
    compile_cluster_reports(
        scenario=scenario,
        method=method,
        ref_metric=ref_metric,
        threshold=threshold
    )