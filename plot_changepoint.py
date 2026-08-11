import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_changepoint(cenario, methods, file, threshold=0.95, tail_plot=False, show=False, save=False, slice_window=["", ""], alias="", metric=None):
    if isinstance(methods, str):
        methods = [methods] if methods else []
        
    base_types = []
    vwcd_method = None
    for m in methods:
        if m.startswith('vwcd'):
            base_types.append('vwcd')
            vwcd_method = m
        elif m.startswith('cusum'):
            base_types.append('cusum')
        elif m.startswith('pelt'):
            base_types.append('pelt')
        else:
            raise ValueError(f"Method '{m}' must start with 'vwcd', 'cusum', or 'pelt'.")
            
    for b_type in set(base_types):
        if base_types.count(b_type) > 1:
            raise ValueError(f"Cannot pass more than one method of type '{b_type}'.")

    if tail_plot and not vwcd_method:
        raise ValueError("Tail plot is only available if a 'vwcd' method is provided.")

    start = pd.to_datetime(slice_window[0]) if (len(slice_window) > 0 and slice_window[0]) else None
    end = pd.to_datetime(slice_window[1]) if (len(slice_window) > 1 and slice_window[1]) else None

    def load_and_slice_df(filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
        if start:
            df = df[df['timestamp'] >= start]
        if end:
            df = df[df['timestamp'] <= end]
        return df

    metric_val = metric if metric else cenario
    serie_file = os.path.join('series', metric_val, file)
    df_serie = load_and_slice_df(serie_file)

    if df_serie.empty:
        return
    
    value_column = [col for col in df_serie.columns if col != 'timestamp'][0]

    df_methods = {}
    for m in methods:
        if m.startswith('vwcd'):
            path = os.path.join('results', cenario, m, 'tail_probability_theta_ge_Tstar', file)
        elif m.startswith('cusum'):
            path = os.path.join('results', cenario, m, file)
        elif m.startswith('pelt'):
            path = os.path.join('results', cenario, m, file)
        df_methods[m] = load_and_slice_df(path)  # type: ignore

    ax2 = None
    if tail_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=(12, 5))
    
    ax1.plot(df_serie['timestamp'], df_serie[value_column], label='Data', 
             linewidth=1, color='gray', marker='o', 
             markerfacecolor='black', markeredgecolor='black', markersize=5)

    for m in methods:
        df_m = df_methods[m]
        if m.startswith('vwcd'):
            cps = df_m[df_m['P_theta_ge_Tstar'] > threshold]['timestamp']
            color, ls, lw, zo, lbl, alpha = 'red', '-', 4, 2, 'CP VWCD', 0.7
        elif m.startswith('cusum'):
            cps = df_m[df_m['CP'] == 1]['timestamp']
            color, ls, lw, zo, lbl, alpha = 'green', '--', 2, 3, 'CP CUSUM', 1.0
        elif m.startswith('pelt'):
            cps = df_m[df_m['CP'] == 1]['timestamp']
            color, ls, lw, zo, lbl, alpha = 'blue', ':', 2.5, 4, 'CP PELT', 1.0

        for i, cp in enumerate(cps): # type: ignore
            ax1.axvline(x=cp, color=color, linestyle=ls, linewidth=lw,  # type: ignore
                        alpha=alpha, zorder=zo, label=lbl if i == 0 else "") # type: ignore

    file_alias = f"{file} ({alias})" if alias != "" else file
    if len(methods) == 0:
        ax1.set_title(f'Time Series - {file_alias}')
    elif len(methods) > 1:
        ax1.set_title(f'Change Point Detection Comparison - {file_alias}')
    else:
        ax1.set_title(f'{methods[0].upper()} - {file_alias}')
        
    ax1.set_ylabel(value_column.capitalize())
    ax1.set_xlabel('Time')
    ax1.legend(loc='lower right')

    if tail_plot and vwcd_method and ax2 is not None:
        df_tail = df_methods[vwcd_method]
        ax2.plot(df_tail['timestamp'], df_tail['P_theta_ge_Tstar'], color='gray', label='Tail Probability')
        ax2.axhline(y=threshold, color='red', linestyle='-', label=f'Threshold ({threshold})')
        ax2.set_title(r"Tail Probability Over Time: $P(\theta \geq T^*)$")
        ax2.set_ylabel(r"$P(\theta \geq T^*)$")
        ax2.set_xlabel('Time')
        ax2.legend(loc='center right')

    fig.tight_layout()

    if save:
        if len(methods) == 0:
            method_str = "pure"
        elif len(methods) > 1:
            method_str = "compare"
        else:
            method_str = methods[0]
            
        method_alias = f"{method_str}_{alias}" if alias != "" else method_str
        
        if tail_plot:
            dir_path_serie = os.path.join('plots', cenario, method_alias, 'ts_tail_plot')
        elif vwcd_method or len(methods) > 1:
            dir_path_serie = os.path.join('plots', cenario, method_alias, 'ts_plot')
        else:
            dir_path_serie = os.path.join('plots', cenario, method_alias)
            
        os.makedirs(dir_path_serie, exist_ok=True)
        plot_file_serie = os.path.join(dir_path_serie, file.replace('.csv', '.png'))
        if os.path.exists(plot_file_serie):
            os.remove(plot_file_serie)
        fig.savefig(plot_file_serie)

    if show:
        plt.show()
    
    plt.close("all")

def plot_one(cenario, method, file, threshold, save=False, tail_plot=False, show=False, alias="", slice_window=["",""]):
    try:
        plot_changepoint(cenario, method, file, threshold, save=save, show=show, tail_plot=tail_plot, alias=alias, slice_window=slice_window)
    except Exception as e:
        print(f"Error plotting {file}: {e}")
    

def plot_folder(cenario, method, threshold, save=True, tail_plot=False, alias="", slice_window=["",""], show=False):
    serie_folder = f'series/{cenario}'
    try:        
        for file in os.listdir(serie_folder):
            if file.endswith('.csv'):
                plot_one(cenario, method, file, threshold, save=save, show=show, tail_plot=tail_plot, alias=alias, slice_window=slice_window)
    except ValueError as e:
        print(f"Error plotting folder '{serie_folder}': {e}")

if __name__ == "__main__":
    print("Starting change point plotting...")
    # cenario = 'teste_m130'
    # method = "cusum"
    # file = "teste01.csv"
    # plot_one(cenario, method, file, 0.95, save=True, show=True, tail_plot=False)

    method = ["cusum", "pelt", "vwcd_w24_fp2"]
    threshold = 0.9

    # NDT_folder = "NDT"
    # cenarios = [f"{NDT_folder}/{p}" for p in os.listdir(f"series/{NDT_folder}") if os.path.isdir(f"series/{NDT_folder}/{p}") and p != "full"]
    # cenarios = ["teste_m110", "teste_m130", "teste_m150"]
    # for cenario in cenarios:
        # plot_folder(cenario, method, threshold, save=False, tail_plot=True, show=True) # iterativo
        # plot_folder(cenario, method, threshold, save=True, tail_plot=False) # comparação
        # plot_folder(cenario, method, threshold, save=True, tail_plot=True) # comparação
        # plot_folder(cenario, [], threshold, save=True, tail_plot=False) # série pura 

    slices = [
        ["2025-10-01", "2025-10-31"],
        ["2025-11-01", "2025-11-30"],
        ["2025-12-01", "2025-12-31"],
        ["2026-01-01", "2026-01-31"],
        ["2026-02-01", "2026-02-28"],
        ["2026-03-01", "2026-03-31"],
        ["2026-04-01", "2026-04-30"]
    ]
    aliases = ["october", "november", "december", "january", "february", "march", "april"]
    for slice_window, alias in zip(slices, aliases):
        cenario = "NDT/rtt_down"
        plot_folder(cenario, method, threshold, save=True, tail_plot=False, alias=alias, slice_window=slice_window) # slice
        plot_folder(cenario, method, threshold, save=True, tail_plot=True, alias=alias, slice_window=slice_window) # slice

    # # cenarios = ["NDT_OUT/packet_loss", "NDT_OUT/tp_up","NDT_OUT/rtt_down"]
    # # method = "pure"
    # # for cenario in cenarios:
    # #     plot_folder(cenario, method, threshold, save=True, tail_plot=False)
