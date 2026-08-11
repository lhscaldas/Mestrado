import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_metrics_folder(cenario: str, method: str, threshold: float = 0.95, tolerance: int = 10) -> pd.DataFrame:
    serie_folder = os.path.join("series", cenario)
    files = sorted([f for f in os.listdir(serie_folder) if f.endswith(".csv")])
    
    results = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    
    for file in files:
        serie_file = os.path.join(serie_folder, file)
        df_serie = pd.read_csv(serie_file)[['timestamp', 'value_cp']]
        df_serie['timestamp'] = pd.to_datetime(df_serie['timestamp'])

        if method.startswith('vwcd'):
            res_file = os.path.join("results", cenario, method, "tail_probability_theta_ge_Tstar", file)
        else:
            res_file = os.path.join("results", cenario, method, file)

        if not os.path.exists(res_file):
            print(f"Warning: Result file '{res_file}' not found. Skipping.")
            continue
            
        df_res = pd.read_csv(res_file)
        df_res['timestamp'] = pd.to_datetime(df_res['timestamp'])

        df_merged = pd.merge(df_serie, df_res, on='timestamp', how='left').fillna(0)
        y_true = df_merged['value_cp'].astype(int).values
        
        if method.startswith('vwcd'):
            y_pred = (df_merged['P_theta_ge_Tstar'] > threshold).astype(int).values
        else:
            y_pred = df_merged['CP'].astype(int).values

        true_indices = np.where(y_true == 1)[0]
        pred_indices = np.where(y_pred == 1)[0]
        
        used_preds = set()
        tp = 0
        fn = 0
        
        for ti in true_indices:
            valid_preds = [pi for pi in pred_indices if abs(pi - ti) <= tolerance]
            if valid_preds:
                tp += 1
                used_preds.update(valid_preds)
            else:
                fn += 1
                
        fp = len(pred_indices) - len(used_preds)
        tn = len(y_true) - tp - fp - fn
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

    if (total_tp + total_fn + total_fp + total_tn) == 0:
        raise ValueError("Nenhum dado encontrado.")

    accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn)
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0.0

    results.append({
        'Method': method,
        'TP': total_tp, 'FP': total_fp, 'TN': total_tn, 'FN': total_fn,
        'Accuracy': accuracy, 'Recall': recall, 'Specificity': specificity,
        'Precision': precision, 'F1-Score': f1
    })

    metrics_df = pd.DataFrame(results)

    return metrics_df

def threshold_selection(cenario: str, method: str = 'vwcd', tolerance: int = 3):
    thresholds = np.arange(0.01, 1.01, 0.01)
    results_list = []
    
    for threshold in thresholds:
        # Passa o método específico recebido como argumento
        metrics_df = calculate_metrics_folder(cenario, method, threshold, tolerance=tolerance)
        metrics_df.insert(0, 'Threshold', threshold)
        results_list.append(metrics_df)

    results_df = pd.concat(results_list, ignore_index=True)
    
    os.makedirs(os.path.join("metrics", cenario), exist_ok=True)
    # O nome do arquivo agora é dinâmico baseado no nome do método
    file_path = os.path.join("metrics", cenario, f"thresholds_{method}.csv")
    results_df.to_csv(file_path, index=False)

def plot_f1_score(cenario: str, method: str, Show: bool = True, Save: bool = False):
    results_df = pd.read_csv(os.path.join("metrics", cenario, f"thresholds_{method}.csv"))
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Threshold'], results_df['F1-Score'], marker='o')
    plt.title('F1-Score vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('F1-Score')
    plt.grid()
    if Save:
        plt.savefig(f"metrics/{cenario}/f1-score_vs_threshold_{method}.png")
    if Show:
        plt.show()

def plot_roc_curve(cenario: str, method: str, Show: bool = True, Save: bool = False):
    results_df = pd.read_csv(os.path.join("metrics", cenario, f"thresholds_{method}.csv"))
    
    results_df['FPR'] = np.where(
        (results_df['FP'] + results_df['TN']) > 0,
        results_df['FP'] / (results_df['FP'] + results_df['TN']),
        0.0
    )
    
    results_df = results_df.sort_values(by='FPR')
    
    auc_value = np.trapezoid(results_df['Recall'], results_df['FPR'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['FPR'], results_df['Recall'], marker='o', linestyle='-', color='b')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    
    plt.text(0.7, 0.2, f'AUC = {auc_value:.4f}', fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
             
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.grid()
    
    if Save:
        os.makedirs(os.path.join("metrics", cenario), exist_ok=True)
        plt.savefig(f"metrics/{cenario}/roc_curve_{method}.png")
    if Show:
        plt.show()
    else:
        plt.close()

def plot_grouped_roc_curve(cenarios: list, methods: list, cenario_labels: dict = None, method_labels: dict = None, Show: bool = True, Save: bool = False, save_dir: str = "metrics"): # type: ignore
    if cenario_labels is None:
        cenario_labels = {c: c for c in cenarios}
    if method_labels is None:
        method_labels = {m: m for m in methods}
        
    n_cenarios = len(cenarios)
    fig, axes = plt.subplots(1, n_cenarios, figsize=(6 * n_cenarios, 6))
    
    if n_cenarios == 1:
        axes = [axes]
        
    for ax, cenario in zip(axes, cenarios):
        for method in methods:
            file_path = os.path.join("metrics", cenario, f"thresholds_{method}.csv")
            if not os.path.exists(file_path):
                print(f"Warning: File not found {file_path}")
                continue
                
            df = pd.read_csv(file_path)
            
            df['FPR'] = np.where(
                (df['FP'] + df['TN']) > 0,
                df['FP'] / (df['FP'] + df['TN']),
                0.0
            )
            df = df.sort_values(by='FPR')

            # Para completar a curva ROC, adicionamos os pontos (0,0) e (1,1)
            df = pd.concat([
                pd.DataFrame({'FPR': [0.0], 'Recall': [0.0]}),
                df[['FPR', 'Recall']],
                pd.DataFrame({'FPR': [1.0], 'Recall': [1.0]})
            ], ignore_index=True).drop_duplicates()
            
            auc_value = np.trapezoid(df['Recall'], df['FPR'])
            m_label = method_labels.get(method, method)
            
            ax.plot(df['FPR'], df['Recall'], linestyle='-', label=f"{m_label} (AUC = {auc_value:.4f})")
            
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
        
        c_label = cenario_labels.get(cenario, cenario)
        ax.set_title(f"Scenario: {c_label}")
        ax.set_xlabel('False Positive Rate (FPR)')
        ax.set_ylabel('True Positive Rate (Recall)')
        ax.legend(loc='lower right', fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.7)
        
    plt.tight_layout()
    
    if Save:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "grouped_roc_curve.png"), dpi=300, bbox_inches='tight')
        
    if Show:
        plt.show()
    else:
        plt.close()

def plot_precision_recall_curve(cenario: str, method: str, Show: bool = True, Save: bool = False):
    results_df = pd.read_csv(os.path.join("metrics", cenario, f"thresholds_{method}.csv"))
    
    results_df = results_df.sort_values(by='Recall')
    
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Recall'], results_df['Precision'], marker='o', linestyle='-', color='g')
    
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid()
    
    if Save:
        os.makedirs(os.path.join("metrics", cenario), exist_ok=True)
        plt.savefig(f"metrics/{cenario}/precision_recall_curve_{method}.png")
    if Show:
        plt.show()
    else:
        plt.close()

def plot_grouped_precision_recall_curve(cenarios: list, methods: list, cenario_labels: dict = None, method_labels: dict = None, Show: bool = True, Save: bool = False, save_dir: str = "metrics"): #type: ignore
    if cenario_labels is None:
        cenario_labels = {c: c for c in cenarios}
    if method_labels is None:
        method_labels = {m: m for m in methods}
        
    n_cenarios = len(cenarios)
    fig, axes = plt.subplots(1, n_cenarios, figsize=(6 * n_cenarios, 6))
    
    if n_cenarios == 1:
        axes = [axes]
        
    for ax, cenario in zip(axes, cenarios):
        for method in methods:
            file_path = os.path.join("metrics", cenario, f"thresholds_{method}.csv")
            if not os.path.exists(file_path):
                print(f"Warning: File not found {file_path}")
                continue
                
            df = pd.read_csv(file_path)
            
            df = df.sort_values(by='Recall')
            
            auc_value = np.trapezoid(df['Precision'], df['Recall'])
            
            m_label = method_labels.get(method, method)
            ax.plot(df['Recall'], df['Precision'], linestyle='-', label=f"{m_label} (AUC = {auc_value:.4f})")
            
        c_label = cenario_labels.get(cenario, cenario)
        ax.set_title(f"Scenario: {c_label}")
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        
        ax.legend(loc='best', fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.7)
        
    plt.tight_layout()
    
    if Save:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "grouped_precision_recall_curve.png"), dpi=300, bbox_inches='tight')
        
    if Show:
        plt.show()
    else:
        plt.close()

def save_latex_metrics(cenario: str, method: str, columns: list = [None]):
    csv_file = os.path.join("metrics", cenario, f"metrics_{method}.csv")
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_file}")
        
    df = pd.read_csv(csv_file)
    
    if columns[0] is None:
        columns = df.columns.tolist()
    else:
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Colunas não encontradas no DataFrame: {missing_cols}")
        df = df[columns]

    def format_val(val):
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val)

    num_cols = len(columns)
    col_format = "c" * num_cols
    
    latex_lines = []
    latex_lines.append(f"\\begin{{tabular}}{{{col_format}}}")
    latex_lines.append("\\hline")
    
    header = " & ".join([f"{col}" for col in columns]) + " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\hline")
    
    for _, row in df.iterrows():
        row_str = " & ".join([format_val(row[col]) for col in columns]) + " \\\\"
        latex_lines.append(row_str)
        
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}\n")
    
    latex_content = "\n".join(latex_lines)
    
    out_dir = os.path.join("metrics", cenario)
    os.makedirs(out_dir, exist_ok=True)
    
    filename = os.path.join(out_dir, f"metrics_{method}.tex")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(latex_content)

def plot_grouped_metrics(cenario: str, metrics: list = ['TP', 'FP', 'FN'], scenario_label: str = "", Show: bool = True, Save: bool = False):
    csv_file = os.path.join("metrics", cenario, "metrics_compare.csv")
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"File not found: {csv_file}")

    df = pd.read_csv(csv_file)
    
    missing_cols = [col for col in metrics if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    if 'Method' not in df.columns:
        raise ValueError("Column 'Method' not found in CSV file.")

    # --- Lógica para adicionar espaçamento entre as barras do mesmo grupo ---
    n_methods = len(df)
    n_metrics = len(metrics)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y = np.arange(n_methods)  # Posições dos métodos no eixo Y
    total_group_height = 0.7  # Espaço total que as barras de um método ocupam
    inter_bar_gap = 0.02      # O espaçamento entre as barras de TP, FP, FN
    
    # Cálculo da altura de cada barra individual descontando os gaps
    bar_height = (total_group_height - (n_metrics - 1) * inter_bar_gap) / n_metrics
    
    # Plotagem manual de cada métrica para garantir o controle do offset
    for i, metric in enumerate(metrics):
        # Calcula o deslocamento para cada barra dentro do grupo
        offset = -total_group_height/2 + i * (bar_height + inter_bar_gap) + bar_height/2
        ax.barh(y + offset, df[metric], bar_height, label=metric, edgecolor='black')

    # Configurações do gráfico
    title_scenario = scenario_label if scenario_label != "" else cenario    
    ax.set_title(f'Evaluation Metrics - Scenario: {title_scenario}')
    ax.set_ylabel('Method')
    ax.set_xlabel('Count')
    
    ax.set_yticks(y)
    ax.set_yticklabels(df['Method'])
    ax.invert_yaxis() # Opcional: mantém a ordem do CSV de cima para baixo
    
    ax.legend(title='Metrics')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if Save:
        out_dir = os.path.join("metrics", cenario)
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, "grouped_metrics_compare.png"))
    
    if Show:
        plt.show()
    else:
        plt.close()

def generate_latex_table_rows(target_threshold: float) -> str:
    rows_config = [
        (20, 1, 1, "vwcd_w20_fp1"),
        (20, 2, 1, "vwcd_w20_fp2"),
        (20, 1, 2, "vwcd_w20_fn2"),
        (24, 1, 1, "vwcd_w24_fp1"),
        (24, 2, 1, "vwcd_w24_fp2"),
        (24, 1, 2, "vwcd_w24_fn2"),
    ]
    cenarios = ["teste_m110", "teste_m130", "teste_m150"]
    
    latex_lines = []
    
    for w, cfp, cfn, method in rows_config:
        row_str = f"{w} & {cfp} & {cfn}"
        
        for cenario in cenarios:
            file_path = os.path.join("metrics", cenario, f"thresholds_{method}.csv")
            
            if not os.path.exists(file_path):
                row_str += " & - & - & -"
                continue
                
            df = pd.read_csv(file_path)
            
            idx = (df['Threshold'] - target_threshold).abs().idxmin()
            row_data = df.loc[idx]
            
            tp = int(row_data['TP']) # type: ignore
            fp = int(row_data['FP']) # type: ignore
            fn = int(row_data['FN']) # type: ignore
            
            row_str += f" & {tp} & {fp} & {fn}"
            
        row_str += r" \\"
        latex_lines.append(row_str)
        
    return "\n".join(latex_lines)

if __name__ == "__main__":
    print("Starting change point metrics calculation...\n")
    # methods = ["cusum", "pelt"]
    methods = ["cusum"]
    cenarios = ["teste_m110", "teste_m130", "teste_m150"]
    
    latex_output = ["\\midrule"]
    
    for method in methods:
        method_metrics = []
        for cenario in cenarios:
            df = calculate_metrics_folder(cenario, method, threshold=0.95, tolerance=10)
            print(f"Metrics for {method} in {cenario}:\n{df}\n")
            
            # Extraindo os valores diretamente da primeira linha do DataFrame consolidado
            tp = int(df['TP'].iloc[0])
            fp = int(df['FP'].iloc[0])
            fn = int(df['FN'].iloc[0])
            
            method_metrics.extend([str(tp), str(fp), str(fn)])
            
        row_str = f"{method.upper()} & {' & '.join(method_metrics)} \\\\"
        latex_output.append(row_str)
        
    latex_output.append("\\bottomrule")
    
    print("\n".join(latex_output))


# if __name__ == "__main__":
#     thresholds_to_print = [0.80, 0.90, 0.95]
#     for th in thresholds_to_print:
#         print(f"% --- Rows for Threshold {th*100:.0f}% ---")
#         print(generate_latex_table_rows(th))
#         print("\n")

# if __name__ == "__main__":
#     print("Starting change point metrics calculation...")
#     methods = ["vwcd_w20_fp1", "vwcd_w20_fp2", "vwcd_w20_fn2", "vwcd_w24_fn1", "vwcd_w24_fp2", "vwcd_w24_fn2"]
#     for method in methods:
#         cenarios = ["teste_m110", "teste_m130", "teste_m150"]
#         for cenario in cenarios: # calcular para um limiar fixo
#             calculate_metrics_folder(cenario, method, threshold=0.95, tolerance=10, save=True)
#         for cenario in cenarios: # calcular para uma faixa de limiares
#             threshold_selection(cenario=cenario, method=method, tolerance=10)
#         for cenario in cenarios: # plotar gráficos 
#             plot_f1_score(cenario=cenario, method=method, Show=False, Save=True)
#             plot_roc_curve(cenario=cenario, method=method, Show=False, Save=True)
#             plot_precision_recall_curve(cenario=cenario, method=method, Show=False, Save=True)



    # cenario_labels = {
    #     "teste_m110": r"$1 \sigma$ shift",
    #     "teste_m130": r"$3 \sigma$ shift",
    #     "teste_m150": r"$5 \sigma$ shift"
    # }
    # for cenario, label in cenario_labels.items(): # type: ignore
    #     plot_grouped_metrics(cenario=cenario, metrics=['TP', 'FP', 'FN'], scenario_label=label, Show=False, Save=True)


# if __name__ == "__main__":
#     methods = [
#         "vwcd_w20_fp1", 
#         "vwcd_w20_fp2", 
#         "vwcd_w20_fn2", 
#         "vwcd_w24_fp1", 
#         "vwcd_w24_fp2", 
#         "vwcd_w24_fn2"
#     ]
    
#     cenarios = [
#         "teste_m110", 
#         "teste_m130", 
#         "teste_m150"
#     ]

#     cenario_labels = {
#         "teste_m110": r"$1\sigma$ Shift",
#         "teste_m130": r"$3\sigma$ Shift",
#         "teste_m150": r"$5\sigma$ Shift"
#     }

#     method_labels = {
#         "vwcd_w20_fp1": r"VWCD ($W=20, C_{FP}=1$)",
#         "vwcd_w20_fp2": r"VWCD ($W=20, C_{FP}=2$)",
#         "vwcd_w20_fn2": r"VWCD ($W=20, C_{FN}=2$)",
#         "vwcd_w24_fp1": r"VWCD ($W=24, C_{FP}=1$)",
#         "vwcd_w24_fp2": r"VWCD ($W=24, C_{FP}=2$)",
#         "vwcd_w24_fn2": r"VWCD ($W=24, C_{FN}=2$)"
#     }

    # print("Generating grouped ROC curves...")
    # plot_grouped_roc_curve(
    #     cenarios=cenarios,
    #     methods=methods,
    #     cenario_labels=cenario_labels,
    #     method_labels=method_labels,
    #     Show=True,
    #     Save=True
    # )
    # plot_grouped_precision_recall_curve(
    #     cenarios=cenarios,
    #     methods=methods,
    #     cenario_labels=cenario_labels,
    #     method_labels=method_labels,
    #     Show=True,
    #     Save=True
    # )



        