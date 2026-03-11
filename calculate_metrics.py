import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_metrics_folder(cenario: str, threshold: float, tolerance: int = 3) -> pd.DataFrame:
    serie_folder = os.path.join("series", cenario)
    results_folder = os.path.join("results", cenario)
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    
    files = sorted([f for f in os.listdir(serie_folder) if f.endswith(".csv")])
    
    for file in files:
        serie_file = os.path.join(serie_folder, file)
        tail_file = os.path.join(results_folder, f"tail_probability_theta_ge_Tstar_{file}")
        
        if not os.path.exists(tail_file):
            continue
            
        df_serie = pd.read_csv(serie_file)[['timestamp', 'value_cp']]
        df_serie['timestamp'] = pd.to_datetime(df_serie['timestamp'])

        df_tail = pd.read_csv(tail_file)[['timestamp', 'P_theta_ge_Tstar']]
        df_tail['timestamp'] = pd.to_datetime(df_tail['timestamp'])

        df_merged = pd.merge(df_serie, df_tail, on='timestamp', how='inner')

        y_true = df_merged['value_cp'].astype(int).values
        y_pred = (df_merged['P_theta_ge_Tstar'] > threshold).astype(int).values

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
        raise ValueError("Nenhum dado encontrado para calcular as métricas.")

    accuracy = (total_tp + total_tn) / total_points
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0.0

    metrics_df = pd.DataFrame({
        'TP': [total_tp],
        'FP': [total_fp],
        'TN': [total_tn],
        'FN': [total_fn],
        'Accuracy': [accuracy],
        'Recall': [recall],
        'Specificity': [specificity],
        'Precision': [precision],
        'F1-Score': [f1]
    })

    return metrics_df

def save_latex_metrics(cenario: str, threshold: float):
    os.makedirs("metrics", exist_ok=True)
    
    df = calculate_metrics_folder(cenario, threshold)
    
    tp = int(df['TP'].iloc[0])
    fp = int(df['FP'].iloc[0])
    tn = int(df['TN'].iloc[0])
    fn = int(df['FN'].iloc[0])
    acc = float(df['Accuracy'].iloc[0])
    rec = float(df['Recall'].iloc[0])
    spec = float(df['Specificity'].iloc[0])
    prec = float(df['Precision'].iloc[0])
    f1 = float(df['F1-Score'].iloc[0])

    latex_content = f"""% --- Tabela Completa ---
\\begin{{tabular}}{{ccccccccc}}
\\hline
{'TP':<5} & {'FP':<5} & {'TN':<6} & {'FN':<5} & {'Accuracy':<8} & {'Recall':<6} & {'Specificity':<11} & {'Precision':<9} & {'F1-Score':<8} \\\\
\\hline
{tp:<5} & {fp:<5} & {tn:<6} & {fn:<5} & {acc:<8.4f} & {rec:<6.4f} & {spec:<11.4f} & {prec:<9.4f} & {f1:<8.4f} \\\\
\\hline
\\end{{tabular}}

% --- Tabela Resumida ---
\\begin{{tabular}}{{cccccc}}
\\hline
{'TP':<5} & {'FP':<5} & {'FN':<5} & {'Recall':<6} & {'Precision':<9} & {'F1-Score':<8} \\\\
\\hline
{tp:<5} & {fp:<5} & {fn:<5} & {rec:<6.4f} & {prec:<9.4f} & {f1:<8.4f} \\\\
\\hline
\\end{{tabular}}

% --- Tabela Super Resumida ---
\\begin{{tabular}}{{ccc}}
\\hline
{'TP':<5} & {'FP':<5} & {'FN':<5} \\\\
\\hline
{tp:<5} & {fp:<5} & {fn:<5} \\\\
\\hline
\\end{{tabular}}
"""

    filename = os.path.join("metrics", f"metrics_{cenario}_{threshold:.2f}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(latex_content)

def threshold_selection(cenario: str):
    thresholds = np.arange(0.01, 1.01, 0.01)
    results_list = []
    
    for threshold in thresholds:
        metrics_df = calculate_metrics_folder(cenario, threshold) # type: ignore
        metrics_df.insert(0, 'Threshold', threshold)
        results_list.append(metrics_df)

    results_df = pd.concat(results_list, ignore_index=True)
    
    os.makedirs("metrics", exist_ok=True)
    file_path = os.path.join("metrics", f"metrics_{cenario}.csv")
    results_df.to_csv(file_path, index=False)

def plot_f1_score(cenario: str, Show: bool = True, Save: bool = False):
    results_df = pd.read_csv(os.path.join("metrics", f"metrics_{cenario}.csv"))
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Threshold'], results_df['F1-Score'], marker='o')
    plt.title('F1-Score vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('F1-Score')
    plt.grid()
    if Save:
        plt.savefig(f"metrics/f1-score_vs_threshold_{cenario}.png")
    if Show:
        plt.show()

if __name__ == "__main__":
    cenarios = ["teste_m110","teste_m130", "teste_m150"]
    # cenario = "teste_m130"
    for cenario in cenarios:
        # results_df = threshold_selection(cenario)
        # plot_f1_score(cenario=cenario, Show=False, Save=True)
        save_latex_metrics(cenario, threshold=0.95)


        