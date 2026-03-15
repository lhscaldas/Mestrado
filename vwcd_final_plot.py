import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_tail_analysis(cenario, file, threshold, show=False, save=False):
    serie_file = f'series/{cenario}/{file}'
    tail_file = f'results/{cenario}/vwcd/tail_probability_theta_ge_Tstar/{file}'

    df_serie = pd.read_csv(serie_file)
    df_serie['timestamp'] = pd.to_datetime(df_serie['timestamp'])
    value_column = [col for col in df_serie.columns if col != 'timestamp'][0]

    df_tail = pd.read_csv(tail_file)
    df_tail['timestamp'] = pd.to_datetime(df_tail['timestamp'])

    CP = df_tail[df_tail['P_theta_ge_Tstar'] > threshold]['timestamp']

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    
    ax1.plot(df_serie['timestamp'], df_serie[value_column], label='Data', 
             linewidth=1, marker='o', markerfacecolor='black', markeredgecolor='black', markersize=5)
    
    for i, cp in enumerate(CP):
        ax1.axvline(x=cp, color='red', linestyle='--', label='Change Point' if i == 0 else "")
        
    ax1.set_title(f'VWCD - {file}')
    ax1.set_ylabel(value_column.capitalize())
    ax1.set_xlabel('Time')
    ax1.legend(loc='lower right')

    fig1.tight_layout()

    if save:
        dir_path_serie = os.path.join('plots', cenario, 'vwcd/ts_plot')
        os.makedirs(dir_path_serie, exist_ok=True)
        plot_file_serie = os.path.join(dir_path_serie, file.replace('.csv', '.png'))
        if os.path.exists(plot_file_serie):
            os.remove(plot_file_serie)
        fig1.savefig(plot_file_serie)

    if show:
        plt.show()

    fig2, (ax2_1, ax2_2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    ax2_1.plot(df_serie['timestamp'], df_serie[value_column], label='Data', 
               linewidth=1, marker='o', markerfacecolor='black', markeredgecolor='black', markersize=5)
    
    for i, cp in enumerate(CP):
        ax2_1.axvline(x=cp, color='red', linestyle='--', label='Change Point' if i == 0 else "")
        
    ax2_1.set_title(f'VWCD - {file}')
    ax2_1.set_ylabel(value_column.capitalize())
    ax2_1.legend(loc='lower right')

    ax2_2.plot(df_tail['timestamp'], df_tail['P_theta_ge_Tstar'], label='Tail Probability')
    ax2_2.axhline(y=threshold, color='red', linestyle='-', label=f'Threshold ({threshold})')
    
    ax2_2.set_title(r"Tail Probability Over Time: $P(\theta \geq T^*)$")
    ax2_2.set_ylabel(r"$P(\theta \geq T^*)$")
    ax2_2.set_xlabel('Time')
    ax2_2.legend(loc='center right')

    fig2.tight_layout()

    if save:
        dir_path_tail = os.path.join('plots', cenario, 'vwcd/ts_tail_plot')
        os.makedirs(dir_path_tail, exist_ok=True)
        plot_file_tail = os.path.join(dir_path_tail, file.replace('.csv', '.png'))
        if os.path.exists(plot_file_tail):
            os.remove(plot_file_tail)
        fig2.savefig(plot_file_tail)

    if show:
        plt.show()
        
    plt.close("all")

def plot_one(cenario, file, threshold, save=False):
    try:
        plot_tail_analysis(cenario, file, threshold, save=save)
    except Exception as e:
        print(f"Error plotting {file}: {e}")
    

def plot_folder(cenario, threshold, save=True):
    serie_folder = f'series/{cenario}'

    for file in os.listdir(serie_folder):
        if file.endswith('.csv'):
            plot_one(cenario, file, threshold, save=save)

if __name__ == "__main__":
    cenario = "teste"
    # file = "teste01.csv"
    # plot_one(cenario, file, 1.01, save=False, show_serie=True, show_tail=False)

    plot_folder(cenario, 0.95, save=True)
