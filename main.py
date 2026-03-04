from votes_commonvar import window_votes
from aggregation_from_input_votes_confs import aggregation_from_input_votes_confs
import os

def single_file(cenario,file):
    # Define folders
    series_folder = "series/"+ cenario
    results_folder = "results/" + cenario
    plots_folder = "plots/" + cenario

    # Create folders if they don't exist
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)

    # Define files
    serie_file = series_folder + "/" + file
    votes_file = results_folder + "/votes_" + file
    conf_file = results_folder + "/conf_" + file
    agg_csv = results_folder + "/aggregated_entropy_kl_" + file
    agg_csv_detail = results_folder + "/aggregated_entropy_kl_full_details_" + file
    agg_tail_csv = results_folder + "/tail_probability_theta_ge_Tstar_" + file
    stack_plot_file = plots_folder + "/stack_plot_" + file.replace(".csv", ".png")
    stack_plot_conf_file = plots_folder + "/stack_plot_conf_" + file.replace(".csv", ".png")
    agg_plot_file = plots_folder + "/agg_plot_" + file.replace(".csv", ".png")
    tail_plot_file = plots_folder + "/tail_plot_" + file.replace(".csv", ".png")

    window_votes(serie_file, votes_file, conf_file)
    aggregation_from_input_votes_confs(votes_file,conf_file,agg_csv,agg_csv_detail,agg_tail_csv, stack_plot_file, stack_plot_conf_file, agg_plot_file, tail_plot_file, plot=True)

def multiple_files(cenario):
    serie_folder = "series/"+ cenario
    files = os.listdir(serie_folder)
    for file in files:
        if file.endswith(".csv"):
            single_file(cenario, file)

if __name__ == "__main__":
    import time
    begin = time.time()

    cenario = "teste_m130"
    # file = "teste01.csv"
    # single_file(cenario,file)
    multiple_files(cenario)

    end = time.time()
    print(f"Tempo total: {end - begin:.2f} segundos")