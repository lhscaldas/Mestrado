import os
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import seaborn as sns

def plot_changepoints(input_cp_dir, client, site, variable, ylim=None,
                      plot_votes=False, plot_probs=False, save_fig=False):
    """
    Plota os valores de uma variável ao longo do tempo com changepoints destacados,
    lendo os dados de um arquivo CSV.

    Parâmetros:
    ----------
    input_cp_dir : str
        Caminho para o diretório onde os arquivos CSV com os dados de changepoint estão armazenados.
    client : str
        Identificador do cliente.
    site : str
        Identificador do site (servidor).
    variable : str
        Nome da variável a ser plotada (ex: 'throughput_download').
    ylim : tuple or None, optional
        Limites do eixo Y no formato (y_min, y_max).
    plot_votes : bool
        Se True, inclui o gráfico com o número de votos.
    plot_probs : bool
        Se True, inclui o gráfico com as probabilidades dos votos.
    save_fig : bool
        Se True, salva a figura em um arquivo PNG no diretório 'imgs/'.

    Retorna:
    -------
    None
        Exibe ou salva o gráfico com os subplots selecionados.
    """
    # Construir o caminho para o arquivo .csv
    file_name = f"{client}_{site}.csv"
    file_path = os.path.join(input_cp_dir, file_name)

    if not os.path.exists(file_path):
        print(f"Arquivo não encontrado: {file_path}")
        return

    # Carregar o arquivo CSV, garantindo que o timestamp seja convertido para datetime
    try:
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
    except Exception as e:
        print(f"Erro ao ler o arquivo {file_path}: {e}")
        return

    # Nomes das colunas de análise
    changepoint_column = f"{variable}_cp"
    votes_column = f"{variable}_votes"
    probs_column = f"{variable}_agg_probs"
    
    # Verificar se as colunas necessárias existem
    required_columns = [variable, changepoint_column]
    if plot_votes:
        required_columns.append(votes_column)
    if plot_probs:
        required_columns.append(probs_column)
    
    if not all(col in df.columns for col in required_columns):
        print(f"Erro: O arquivo {file_name} não contém todas as colunas necessárias para a variável '{variable}'.")
        return

    # Obter os timestamps dos changepoints
    changepoints = df['timestamp'][df[changepoint_column] == 1]

    # Determinar o número de subplots e a proporção de altura
    n_plots = 1 + plot_votes + plot_probs
    height_ratios = [3] + [1.5] * (n_plots - 1)
    
    # Criar figura e eixos, compartilhando o eixo X
    fig, axes = plt.subplots(n_plots, 1, figsize=(15, 2.5 * n_plots + 2),
                             sharex=True, gridspec_kw={'height_ratios': height_ratios})

    # Garantir que 'axes' seja sempre uma lista para facilitar a indexação
    if n_plots == 1:
        axes = [axes]

    plt.subplots_adjust(hspace=0.1)
    current_ax_idx = 0

    # --- Subplot 1: Valores da variável ---
    ax = axes[current_ax_idx]
    # ax.plot(df['timestamp'], df[variable], label='Valores', color='gray', alpha=0.8)
    ax.plot(df['timestamp'], df[variable], label='Valores', alpha=0.8)
    # Adiciona linhas verticais para cada changepoint
    for i, cp in enumerate(changepoints):
        ax.axvline(x=cp, color='red', linestyle='--', label='Ponto de mudança' if i == 0 else "")
    ax.set_ylabel(variable)
    ax.set_title(f"Análise de Changepoint para '{variable}'\nCliente: {client} | Servidor: {site}")
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    if ylim:
        ax.set_ylim(ylim)
    current_ax_idx += 1

    # --- Subplot 2: Número de votos ---
    if plot_votes:
        ax = axes[current_ax_idx]
        ax.plot(df['timestamp'], df[votes_column], color='green', marker='.', linestyle='None', label='Votos')
        for cp in changepoints:
            ax.axvline(x=cp, color='red', linestyle='--')
        ax.set_ylabel('Nº de Votos')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        current_ax_idx += 1

    # --- Subplot 3: Probabilidades dos votos ---
    if plot_probs:
        ax = axes[current_ax_idx]
        ax.plot(df['timestamp'], df[probs_column], color='purple', marker='.', linestyle='None', label='Probabilidade Agregada')
        for cp in changepoints:
            ax.axvline(x=cp, color='red', linestyle='--')
        ax.set_ylabel('Probabilidade')
        ax.set_ylim(0, 1.05)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        current_ax_idx += 1

    # Configurar o rótulo do eixo X apenas no último subplot
    axes[-1].set_xlabel('Tempo')
    fig.autofmt_xdate() # Melhora a formatação das datas

    if save_fig:
        output_dir = 'imgs'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{client}_{site}_{variable}.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Figura salva em: {output_file}")
    
    plt.show()

def plot_changepoint_comparison(source_dirs, client, server, variables, ref_method, ylim=None, save_fig=False):
    """
    Plota uma comparação de análises de changepoint de múltiplas fontes,
    com legenda e cores consistentes para cada método.
    Trata o método 'ref_method' como ground truth (linha preta) e os demais
    como marcadores circulares com deslocamento vertical.

    Parâmetros:
    -----------
    source_dirs : dict
        Dicionário com nomes para a legenda e caminhos para os diretórios dos CSVs.
        Ex: {'Pelt': 'cp_pelt', 'VWCD Otimizado': 'cp_vwcd_2'}
    client : str
        O nome do cliente (singular).
    server : str
        O nome do site/servidor (singular).
    variables : list
        Uma lista com as métricas a serem plotadas.
    ref_method : str
        O nome (chave do source_dirs) do método a ser usado como referência 
        (plotado como linha preta tracejada).
    ylim : tuple or None, optional
        Limites do eixo Y para o gráfico principal.
    save_fig : bool
        Se True, salva a figura em um arquivo PNG no diretório 'imgs_comparativo/'.
    """

    # --- 1. Mapeamento Fixo de Cores e Marcadores ---
    method_labels = list(source_dirs.keys())
    base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_map = {}
    ref_label = ref_method
    non_ref_color_idx_map = 0 # Contador para cores dos métodos não-referência

    for i, label in enumerate(method_labels):
        if label == ref_label:
            color_map[label] = 'black'
        else:
            color_map[label] = base_colors[non_ref_color_idx_map % len(base_colors)]
            non_ref_color_idx_map += 1
    # -------------------------------------------------
    
    for variable in variables:
        data_sources = {}
        for label, input_dir in source_dirs.items():
            file_path = os.path.join(input_dir, f"{client}_{server}.csv")
            if not os.path.exists(file_path):
                print(f"AVISO: Arquivo não encontrado para '{label}': {file_path}")
                continue
            try:
                # Lê o CSV e trata datas de forma robusta
                df_temp = pd.read_csv(file_path)
                # Assumindo que a coluna de tempo se chama 'timestamp' ou 'test_time'
                time_col = 'timestamp' if 'timestamp' in df_temp.columns else 'test_time'
                df_temp[time_col] = pd.to_datetime(df_temp[time_col], errors='coerce')
                data_sources[label] = df_temp
            except Exception as e:
                print(f"Erro ao ler o arquivo {file_path} para '{label}': {e}")

        if not data_sources:
            print(f"Nenhum dado pôde ser carregado para '{variable}' do par ({client}, {server}). Pulando gráfico.")
            continue

        fig, ax = plt.subplots(1, 1, figsize=(18, 6))
        
        base_df = None
        if data_sources:
             base_df = next(iter(data_sources.values()))
        else:
            continue # Pula se nenhum dado foi carregado

        time_col = 'timestamp' if 'timestamp' in base_df.columns else 'test_time'
        
        if variable not in base_df.columns:
            print(f"AVISO: Coluna '{variable}' não encontrada no arquivo base. Pulando gráfico.")
            continue
            
        ax.plot(base_df[time_col], base_df[variable], label=f'Valores de {variable}', alpha=0.6, color='grey')

        y_values = base_df[variable].dropna()
        y_range = y_values.max() - y_values.min() if len(y_values) > 1 else 1
        gap = y_range * 0.05 if y_range > 0 else 1

        # --- 2. Criação de Elementos para a Legenda Consistente ---
        legend_elements = [mlines.Line2D([0], [0], color='grey', alpha=0.6, label=f'Valores de {variable}')]
        non_ref_methods_in_legend = [] # Para manter a ordem correta na legenda

        if ref_label in method_labels:
            legend_elements.append(mlines.Line2D([0], [0], color='black', linestyle='--', label=f'CP {ref_label}'))

        for label in method_labels:
                if label != ref_label:
                    legend_elements.append(mlines.Line2D([0], [0], marker='o', color=color_map.get(label, 'grey'), label=f'CP {label}', linestyle='None', markersize=10))
                    non_ref_methods_in_legend.append(label) # Guarda a ordem
        # ------------------------------------------------------------

        # --- 3. Plotagem dos Changepoints Reais (sem label) ---
        non_ref_method_counter = 0 # Usado para o deslocamento vertical na ordem correta
        for label, df in data_sources.items():
            cp_col = f"{variable}_cp"
            if cp_col in df.columns:
                changepoint_data = df[df[cp_col] == 1].copy() # Usa .copy() para evitar SettingWithCopyWarning

                if not changepoint_data.empty:
                    # Lógica especial para o método de REFERÊNCIA
                    if label == ref_label:
                        for cp_time in changepoint_data[time_col]:
                            ax.axvline(x=cp_time, color='black', linestyle='--')
                    # Lógica para os outros métodos
                    else:
                        try:
                            plot_order_index = non_ref_methods_in_legend.index(label)
                        except ValueError:
                            plot_order_index = non_ref_method_counter

                        offset_multiplier = ((-1)**(plot_order_index + 1)) * ((plot_order_index + 1) // 2)
                        
                        if variable in changepoint_data.columns:
                            changepoint_data.loc[:, 'y_with_offset'] = changepoint_data[variable] + (offset_multiplier * gap)
                            ax.plot(changepoint_data[time_col], changepoint_data['y_with_offset'],
                                    linestyle='None',
                                    marker='o',
                                    color=color_map.get(label, 'grey'),
                                    markersize=10)
                        
                        non_ref_method_counter += 1
        # ------------------------------------------------------

        ax.set_ylabel(variable)
        ax.set_title(f"Comparativo de Changepoints para '{variable}'\nCliente: {client} | Servidor: {server}")
        # --- 4. Usa os handles customizados para a legenda ---
        ax.legend(handles=legend_elements)
        # ----------------------------------------------------
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        if ylim:
            ax.set_ylim(ylim)

        ax.set_xlabel('Tempo')
        fig.autofmt_xdate()

        if save_fig:
            output_dir = 'imgs_comparativo'
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"COMP_{client}_{server}_{variable}.png")
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Figura salva em: {output_file}")

        plt.show()

def plot_feature_with_means(input_dir, client, site, variable, figsize=(20, 7)):
    """
    Plota uma série temporal, seus changepoints (se disponíveis), e a média/desvio padrão
    de cada segmento detectado.
    """
    file_path = os.path.join(input_dir, f"{client}_{site}.csv")

    if not os.path.exists(file_path):
        print(f"AVISO: Arquivo não encontrado: {file_path}")
        return

    try:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
    except Exception as e:
        print(f"Erro ao ler ou processar o arquivo {file_path}: {e}")
        return

    # --- MODIFICAÇÃO 1: Separar colunas essenciais das opcionais ---
    # Colunas que SÃO OBRIGATÓRIAS para o gráfico funcionar
    essential_cols = [
        'timestamp', 
        variable, 
        f'{variable}_local_mean', 
        f'{variable}_local_std'
    ]

    # Verifica se as colunas essenciais existem
    if not all(col in df.columns for col in essential_cols):
        print(f"AVISO: O arquivo {file_path} não contém as colunas essenciais (valor, média, std) para a variável '{variable}'.")
        return
    # ----------------------------------------------------------------

    fig, ax = plt.subplots(figsize=figsize)

    # 1. Plota a série temporal original
    ax.plot(df['timestamp'], df[variable], color='grey', alpha=0.7, label=f'Valores de {variable}')

    # 2. Plota a média local de cada segmento
    ax.plot(df['timestamp'], df[f'{variable}_local_mean'], color='#1f77b4', linestyle='-', 
            linewidth=2, label='Média Local do Segmento')

    # 3. Plota a área do desvio padrão
    mean_series = df[f'{variable}_local_mean']
    std_series = df[f'{variable}_local_std']
    ax.fill_between(df['timestamp'], 
                    mean_series - std_series, 
                    mean_series + std_series, 
                    color='#1f77b4', alpha=0.2, label='Média ± 1 Desvio Padrão')

    # --- MODIFICAÇÃO 2: Tornar a plotagem dos CPs condicional ---
    # 4. Plota os changepoints como linhas verticais, APENAS SE a coluna existir
    changepoint_column = f'{variable}_cp'
    if changepoint_column in df.columns:
        changepoints = df['timestamp'][df[changepoint_column] == 1]
        for i, cp_time in enumerate(changepoints):
            ax.axvline(x=cp_time, color='red', linestyle='--', 
                       label='Ponto de Mudança' if i == 0 else "")
    # ------------------------------------------------------------

    # --- Formatação Final ---
    ax.set_title(f"Análise de Segmentos para '{variable}'\nCliente: {client} | Servidor: {site}")
    ax.set_xlabel("Tempo")
    ax.set_ylabel(variable)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.autofmt_xdate()

    plt.show()

def summarize_and_plot_total_cps(source_dirs):
    """
    Calcula o número total de changepoints detectados por diferentes métodos 
    e plota um gráfico de barras comparativo, usando cores consistentes 
    com plot_changepoint_comparison (Pelt=vermelho, outros do ciclo padrão).

    Parâmetros:
    -----------
    source_dirs : dict
        Dicionário onde as chaves são os nomes dos métodos (para a legenda) 
        e os valores são os caminhos relativos para os diretórios 
        contendo os arquivos CSV processados.

    Retorna:
    --------
    dict
        Um dicionário com o nome de cada método e sua contagem total de changepoints.
    """
    total_counts = {}

    # --- 1. Calcula a contagem (lógica inalterada) ---
    for method_name, input_dir in source_dirs.items():
        method_total = 0
        
        if not os.path.isdir(input_dir):
            print(f"    AVISO: Diretório não encontrado: {input_dir}. Pulando método.")
            total_counts[method_name] = 0
            continue

        for file in os.listdir(input_dir):
            if file.endswith(".csv"):
                file_path = os.path.join(input_dir, file)
                try:
                    df = pd.read_csv(file_path)
                    cp_columns = [col for col in df.columns if col.endswith('_cp')]
                    file_total = 0
                    if cp_columns:
                        try:
                            file_total = df[cp_columns].sum().sum()
                        except TypeError: 
                            file_total = sum(df[col].astype(float).sum() for col in cp_columns)
                    method_total += file_total
                except Exception as e:
                    print(f"    Erro ao processar o arquivo {file_path}: {e}")

        total_counts[method_name] = int(method_total)

    # --- 2. Preparação para Plotagem com Cores Consistentes ---
    if not total_counts:
        print("\nNenhum dado para plotar.")
        return total_counts

    # Cria o mapeamento de cores (lógica replicada de plot_changepoint_comparison)
    method_labels_original_order = list(source_dirs.keys())
    base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_map = {}
    non_pelt_color_idx_map = 0
    for label in method_labels_original_order:
        if 'pelt' in label.lower():
            color_map[label] = 'black'
        else:
            color_map[label] = base_colors[non_pelt_color_idx_map % len(base_colors)]
            non_pelt_color_idx_map += 1

    # Ordena os métodos pela contagem para a plotagem
    sorted_methods = sorted(total_counts.items(), key=lambda item: item[1])
    method_names_sorted = [item[0] for item in sorted_methods]
    counts_sorted = [item[1] for item in sorted_methods]
    # Pega as cores correspondentes na ordem correta
    bar_colors = [color_map[name] for name in method_names_sorted]

    # --- 3. Plotagem ---
    fig, ax = plt.subplots(figsize=(10, len(method_names_sorted) * 0.6))
    bars = ax.barh(method_names_sorted, counts_sorted, color=bar_colors) # Usa as cores mapeadas
    
    ax.bar_label(bars, padding=3)
    ax.set_xlabel('Número Total de Changepoints Detectados')
    ax.set_title('Comparativo da Contagem Total de Changepoints por Método')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def count_cps(input_dir, metricas):
    all_rows_data = []
    for file in os.listdir(input_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(input_dir, file)
            df = pd.read_csv(file_path)
            cliente, servidor = file.split('.')[0].split('_', 1)
            row_data = {'cliente': cliente, 'servidor': servidor}
            for metrica in metricas:
                cp_column = metrica + '_cp'
                row_data[metrica] = float(df[cp_column].sum())
            all_rows_data.append(row_data)
    return pd.DataFrame(all_rows_data)

def boxplot_cps(input_dir, metricas, name):
    df_cps = count_cps(input_dir, metricas)
    df_long = df_cps.melt(id_vars=['cliente', 'servidor'], value_vars=metricas, var_name='metrica', value_name='contagem')
    plt.figure(figsize=(10, len(metricas) * 0.7 + 1))
    sns.boxplot(data=df_long, x='contagem', y='metrica')
    plt.title(f'Distribuição de Change Points em {name}')
    plt.xlabel('Número de Change Points')
    plt.ylabel('Métrica')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()