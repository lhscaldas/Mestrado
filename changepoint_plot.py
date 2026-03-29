import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_changepoint(cenario, method, file, threshold=0.95, tail_plot=False, show=False, save=False):
    if method not in ['vwcd', 'cusum', 'pelt', 'compare']:
        raise ValueError("Method must be one of 'vwcd', 'cusum', 'pelt', or 'compare'.")
    if method in ['pelt', 'cusum'] and tail_plot:
        raise ValueError("Tail plot is only available for 'vwcd' or 'compare' methods.")

    serie_file = f'series/{cenario}/{file}'
    df_serie = pd.read_csv(serie_file)
    df_serie['timestamp'] = pd.to_datetime(df_serie['timestamp'])
    value_column = [col for col in df_serie.columns if col != 'timestamp'][0]

    df_tail, df_cusum, df_pelt = None, None, None
    if method == 'vwcd' or method == 'compare':
        tail_file = f'results/{cenario}/vwcd/tail_probability_theta_ge_Tstar/{file}'
        df_tail = pd.read_csv(tail_file)
        df_tail['timestamp'] = pd.to_datetime(df_tail['timestamp'])
    if method == 'cusum' or method == 'compare':    
        cusum_file = f'results/{cenario}/cusum/{file}'
        df_cusum = pd.read_csv(cusum_file)
        df_cusum['timestamp'] = pd.to_datetime(df_cusum['timestamp'])
    if method == 'pelt' or method == 'compare':
        pelt_file = f'results/{cenario}/pelt/{file}'
        df_pelt = pd.read_csv(pelt_file)
        df_pelt['timestamp'] = pd.to_datetime(df_pelt['timestamp'])

    ax2=None
    if tail_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=(12, 5))
    
    ax1.plot(df_serie['timestamp'], df_serie[value_column], label='Data', 
             linewidth=1, color='gray', marker='o', 
             markerfacecolor='black', markeredgecolor='black', markersize=5)

    if df_tail is not None:    
        CP_vwcd = df_tail[df_tail['P_theta_ge_Tstar'] > threshold]['timestamp']
        for i, cp in enumerate(CP_vwcd):
            ax1.axvline(x=cp, color='red', linestyle='-', linewidth=4, alpha=0.7, zorder=2, label='CP VWCD' if i == 0 else "")
    if df_cusum is not None:
        CP_cusum = df_cusum[df_cusum['CP'] == 1]['timestamp']
        for i, cp in enumerate(CP_cusum):
            ax1.axvline(x=cp, color='green', linestyle='--', linewidth=2, zorder=3, label='CP CUSUM' if i == 0 else "")
    if df_pelt is not None:
        CP_pelt = df_pelt[df_pelt['CP'] == 1]['timestamp']
        for i, cp in enumerate(CP_pelt):
            ax1.axvline(x=cp, color='blue', linestyle=':', linewidth=2.5, zorder=4, label='CP PELT' if i == 0 else "")

    if method == 'compare':
        ax1.set_title(f'Change Point Detection Comparison - {file}')
    else:
        ax1.set_title(f'{method.upper()} - {file}')
    ax1.set_ylabel(value_column.capitalize())
    ax1.set_xlabel('Time')
    ax1.legend(loc='lower right')

    if (ax2 is not None) and (df_tail is not None):
        ax2.plot(df_tail['timestamp'], df_tail['P_theta_ge_Tstar'], color='gray', label='Tail Probability')
        ax2.axhline(y=threshold, color='red', linestyle='-', label=f'Threshold ({threshold})')
        
        ax2.set_title(r"Tail Probability Over Time: $P(\theta \geq T^*)$")
        ax2.set_ylabel(r"$P(\theta \geq T^*)$")
        ax2.set_xlabel('Time')
        ax2.legend(loc='center right')

    fig.tight_layout()

    if save:
        if tail_plot:
            dir_path_serie = os.path.join('plots', cenario, method, 'ts_tail_plot')
        elif method in ['vwcd', 'compare']:
            dir_path_serie = os.path.join('plots', cenario, method, 'ts_plot')
        else:
            dir_path_serie = os.path.join('plots', cenario, method)
        os.makedirs(dir_path_serie, exist_ok=True)
        plot_file_serie = os.path.join(dir_path_serie, file.replace('.csv', '.png'))
        if os.path.exists(plot_file_serie):
            os.remove(plot_file_serie)
        fig.savefig(plot_file_serie)

    if show:
        plt.show()
    
    plt.close("all")

def plot_one(cenario, method, file, threshold, save=False, tail_plot=False, show=False):
    try:
        plot_changepoint(cenario, method, file, threshold, save=save, show=show, tail_plot=tail_plot)
    except Exception as e:
        print(f"Error plotting {file}: {e}")
    

def plot_folder(cenario, method, threshold, save=True, tail_plot=False):
    serie_folder = f'series/{cenario}'
    try:
        if method not in ['vwcd', 'cusum', 'pelt', 'compare']:
            raise ValueError("Method must be one of 'vwcd', 'cusum', 'pelt', or 'compare'.")
        if method in ['pelt', 'cusum'] and tail_plot:
            raise ValueError("Tail plot is only available for 'vwcd' or 'compare' methods.")
        
        for file in os.listdir(serie_folder):
            if file.endswith('.csv'):
                plot_one(cenario, method, file, threshold, save=save, show=False, tail_plot=tail_plot)
    except ValueError as e:
        print(f"Error plotting folder '{serie_folder}': {e}")

if __name__ == "__main__":
    print("Starting change point plotting...")
    # cenario = "teste"
    # method = "compare"
    # file = "teste01.csv"
    # plot_one(cenario, method, file, 0.95, save=False, show=True, tail_plot=False)

    cenarios = ["teste_m110", "teste_m130", "teste_m150", "NDT_tp_down", "NDT_rtt_up"]
    method = "compare"
    for cenario in cenarios:
        plot_folder(cenario, method, 0.95, save=True, tail_plot=False)
        plot_folder(cenario, method, 0.95, save=True, tail_plot=True)
