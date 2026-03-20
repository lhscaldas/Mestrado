import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_metrics_folder(cenario: str, method: str, threshold: float = 0.95, tolerance: int = 3, save = False) -> pd.DataFrame:
    if method not in ['vwcd', 'cusum', 'pelt', 'compare']:
        raise ValueError("Method must be one of 'vwcd', 'cusum', 'pelt', or 'compare'.")

    methods_to_evaluate = ['vwcd', 'cusum', 'pelt'] if method == 'compare' else [method]
    serie_folder = os.path.join("series", cenario)
    files = sorted([f for f in os.listdir(serie_folder) if f.endswith(".csv")])
    
    results = []

    for m in methods_to_evaluate:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0
        
        for file in files:
            serie_file = os.path.join(serie_folder, file)
            df_serie = pd.read_csv(serie_file)[['timestamp', 'value_cp']]
            df_serie['timestamp'] = pd.to_datetime(df_serie['timestamp'])

            if m == 'vwcd':
                res_file = os.path.join("results", cenario, "vwcd", "tail_probability_theta_ge_Tstar", file)
            elif m == 'cusum':
                res_file = os.path.join("results", cenario, "cusum", file)
            elif m == 'pelt':
                res_file = os.path.join("results", cenario, "pelt", file)
            else:
                continue

            if not os.path.exists(res_file):
                print(f"Warning: Result file '{res_file}' not found. Skipping.")
                continue
                
            df_res = pd.read_csv(res_file)
            df_res['timestamp'] = pd.to_datetime(df_res['timestamp'])

            df_merged = pd.merge(df_serie, df_res, on='timestamp', how='inner')

            y_true = df_merged['value_cp'].astype(int).values
            
            if m == 'vwcd':
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

        total_points = total_tp + total_fp + total_fn + total_tn
        if total_points == 0:
            continue

        accuracy = (total_tp + total_tn) / total_points
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0.0

        results.append({
            'Method': m.upper(),
            'TP': total_tp,
            'FP': total_fp,
            'TN': total_tn,
            'FN': total_fn,
            'Accuracy': accuracy,
            'Recall': recall,
            'Specificity': specificity,
            'Precision': precision,
            'F1-Score': f1
        })

    if not results:
        raise ValueError("Nenhum dado encontrado para calcular as métricas.")

    metrics_df = pd.DataFrame(results)

    if save:
        os.makedirs("metrics", exist_ok=True)
        os.makedirs(os.path.join("metrics", cenario), exist_ok=True)
        file_path = os.path.join("metrics", cenario, f"metrics_{method}.csv")
        metrics_df.to_csv(file_path, index=False)

    return metrics_df

def threshold_selection(cenario: str):
    thresholds = np.arange(0.01, 1.01, 0.01)
    results_list = []
    
    for threshold in thresholds:
        metrics_df = calculate_metrics_folder(cenario,'vwcd',threshold) # type: ignore
        metrics_df.insert(0, 'Threshold', threshold)
        results_list.append(metrics_df)

    results_df = pd.concat(results_list, ignore_index=True)
    
    os.makedirs("metrics", exist_ok=True)
    os.makedirs(os.path.join("metrics", cenario), exist_ok=True)
    file_path = os.path.join("metrics", cenario, f"thresholds_vwcd.csv")
    results_df.to_csv(file_path, index=False)

def plot_f1_score(cenario: str, Show: bool = True, Save: bool = False):
    results_df = pd.read_csv(os.path.join("metrics", cenario, f"thresholds_vwcd.csv"))
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Threshold'], results_df['F1-Score'], marker='o')
    plt.title('F1-Score vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('F1-Score')
    plt.grid()
    if Save:
        plt.savefig(f"metrics/{cenario}/f1-score_vs_threshold_vwcd.png")
    if Show:
        plt.show()

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

if __name__ == "__main__":
    cenario = "teste"
    method = "vwcd"
    # results = calculate_metrics_folder(cenario, method, threshold=0.95, tolerance=10, save=True)
    # print(results)
    # threshold_selection(cenario)
    # plot_f1_score(cenario=cenario, Show=True, Save=True)
    save_latex_metrics(cenario, method, columns=['Method', 'TP', 'FP', 'FN', 'Recall', 'Precision', 'F1-Score'])


        