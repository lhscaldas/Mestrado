from vwcd_votes import window_votes
from vwcd_aggregation import aggregation_from_input_votes_confs
import os
from changepoint_plot import plot_changepoint

def single_file(cenario, method, file):
    # Define folders
    series_folder = os.path.join("series", cenario)
    results_folder = os.path.join("results", cenario, method)
    plots_folder = os.path.join("plots", cenario, method)

    # Create folders if they don't exist
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)
    os.makedirs(results_folder + "/votes/", exist_ok=True)
    os.makedirs(results_folder + "/confs/", exist_ok=True)
    os.makedirs(results_folder + "/aggregated_entropy_kl/", exist_ok=True)
    os.makedirs(results_folder + "/aggregated_entropy_kl_full_details/", exist_ok=True)
    os.makedirs(results_folder + "/tail_probability_theta_ge_Tstar/", exist_ok=True)
    os.makedirs(plots_folder + "/stack_plot/", exist_ok=True)
    os.makedirs(plots_folder + "/stack_plot_conf/", exist_ok=True)
    os.makedirs(plots_folder + "/agg_plot/", exist_ok=True)
    os.makedirs(plots_folder + "/tail_plot/", exist_ok=True)

    # Define files
    serie_file = series_folder + "/" + file
    votes_file = results_folder + "/votes/" + file
    conf_file = results_folder + "/confs/" + file
    agg_csv = results_folder + "/aggregated_entropy_kl/" + file
    agg_csv_detail = results_folder + "/aggregated_entropy_kl_full_details/" + file
    agg_tail_csv = results_folder + "/tail_probability_theta_ge_Tstar/" + file
    stack_plot_file = plots_folder + "/stack_plot/" + file.replace(".csv", ".png")
    stack_plot_conf_file = plots_folder + "/stack_plot_conf/" + file.replace(".csv", ".png")
    agg_plot_file = plots_folder + "/agg_plot/" + file.replace(".csv", ".png")
    tail_plot_file = plots_folder + "/tail_plot/" + file.replace(".csv", ".png")

    window_votes(serie_file, votes_file, conf_file)
    aggregation_from_input_votes_confs(votes_file,conf_file,agg_csv,agg_csv_detail,agg_tail_csv, stack_plot_file, stack_plot_conf_file, agg_plot_file, tail_plot_file, plot=True)

def multiple_files(cenario, method, threshold=0.95):
    serie_folder = "series/"+ cenario
    files = os.listdir(serie_folder)
    for file in files:
        if file.endswith(".csv"):
            try:
                single_file(cenario, method, file)
                plot_changepoint(cenario, method, file, threshold=threshold, save=True)
                plot_changepoint(cenario, method, file, threshold=threshold, save=True, tail_plot=True)
            except Exception as e:
                continue

if __name__ == "__main__":
    import time
    # begin = time.time()

    method = "vwcd"
    threshold = 0.95

    # cenario = "teste"
    # file = "teste01.csv"
    # single_file(cenario,file)

    # cenario = "teste"
    # multiple_files(cenario, method, threshold)

    # end = time.time()
    # print(f"Tempo total: {end - begin:.2f} segundos")

    cenarios = ["teste_m110", "teste_m130", "teste_m150", "NDT_tp_down", "NDT_rtt_up"]
    for cenario in cenarios:
        begin = time.time()
        multiple_files(cenario, method, threshold)
        end = time.time()
        print(f"Cenário '{cenario}' processado em {end - begin:.2f} segundos")