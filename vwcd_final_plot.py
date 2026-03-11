import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_tail_analysis(serie_file, tail_file, threshold, show_serie=True, show_tail=True, save=False):
    if not show_serie and not show_tail:
        raise ValueError("At least one display option (show_serie or show_tail) must be True.")

    df_serie = pd.read_csv(serie_file)
    df_serie['timestamp'] = pd.to_datetime(df_serie['timestamp'])
    value_column = [col for col in df_serie.columns if col != 'timestamp'][0]

    df_tail = pd.read_csv(tail_file)
    df_tail['timestamp'] = pd.to_datetime(df_tail['timestamp'])

    anomalies = df_tail[df_tail['P_theta_ge_Tstar'] > threshold]['timestamp']

    num_subplots = sum([show_serie, show_tail])
    fig, axes = plt.subplots(num_subplots, 1, figsize=(12, 5 * num_subplots), sharex=True)

    if num_subplots == 1:
        axes = [axes]

    idx = 0

    if show_serie:
        ax_serie = axes[idx]
        ax_serie.plot(df_serie['timestamp'], df_serie[value_column], label='Serie', linewidth=2, marker='o', markerfacecolor='black', markeredgecolor='black', markersize=5)
        
        for t in anomalies:
            ax_serie.axvline(x=t, color='red', linestyle='--', alpha=0.6)
            
        ax_serie.set_title('Serie')
        ax_serie.set_ylabel(value_column.capitalize())
        ax_serie.legend(loc='lower right')
        if num_subplots == 1:
            ax_serie.set_xlabel('Time')
        idx += 1

    if show_tail:
        ax_tail = axes[idx]
        ax_tail.plot(df_tail['timestamp'], df_tail['P_theta_ge_Tstar'], label='Tail Probability')
        ax_tail.axhline(y=threshold, color='red', linestyle='-', label=f'Threshold ({threshold})')
        
        ax_tail.set_title('Tail Probability')
        ax_tail.set_ylabel('Probability')
        ax_tail.set_xlabel('Time')
        ax_tail.legend(loc='lower right')

    plt.tight_layout()
    if save:
        dir_path = os.path.join('plots', cenario, 'serie_plot')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        plot_file = os.path.join(dir_path, os.path.basename(serie_file).replace('.csv', '.png'))
        if os.path.exists(plot_file):
            os.remove(plot_file)
        plt.savefig(plot_file)
    else:
        plt.show()
    plt.close("all")

def plot_one(cenario, file, threshold, save=False, show_serie=True, show_tail=True):
    serie_file = f'series/{cenario}/{file}'
    tail_file = f'results/{cenario}/tail_probability_theta_ge_Tstar/{file}'
    try:
        plot_tail_analysis(serie_file, tail_file, threshold, show_serie=show_serie, show_tail=show_tail, save=save)
    except Exception as e:
        print(f"Error plotting {file}: {e}")
    

def plot_folder(cenario, threshold, save=True, show_serie=True, show_tail=True):
    serie_folder = f'series/{cenario}'

    for file in os.listdir(serie_folder):
        if file.endswith('.csv'):
            plot_one(cenario, file, threshold, save=save, show_serie=show_serie, show_tail=show_tail)

if __name__ == "__main__":
    cenario = "teste"
    # file = "teste01.csv"
    # plot_one(cenario, file, 1.01, save=False, show_serie=True, show_tail=False)

    plot_folder(cenario, 0.95, save=True)
