from votes_commonvar import window_votes
from aggregation_from_input_votes_confs import aggregation_from_input_votes_confs
import os

# Define folders
cenario = "teste"
serie_folder = "series/"+ cenario
votes_folder = "windows/" + cenario
conf_folder = "windows/" + cenario
agg_folder = "aggregated/" + cenario
plots_folder = "plots/" + cenario

# Create folders if they don't exist
os.makedirs(votes_folder, exist_ok=True)
os.makedirs(conf_folder, exist_ok=True)
os.makedirs(agg_folder, exist_ok=True)
os.makedirs(plots_folder, exist_ok=True)

# Define files
file = "teste01.csv"
serie_file = serie_folder + "/" + file
votes_file = votes_folder + "/votes_" + file
conf_file = conf_folder + "/conf_" + file
agg_csv = agg_folder + "/aggregated_entropy_kl_" + file
agg_csv_detail = agg_folder + "/aggregated_entropy_kl_full_details_" + file
agg_tail_csv = agg_folder + "/tail_probability_theta_ge_Tstar_" + file
stack_plot_file = plots_folder + "/stack_plot_" + file.replace(".csv", ".png")
stack_plot_conf_file = plots_folder + "/stack_plot_conf_" + file.replace(".csv", ".png")
agg_plot_file = plots_folder + "/agg_plot_" + file.replace(".csv", ".png")
tail_plot_file = plots_folder + "/tail_plot_" + file.replace(".csv", ".png")


window_votes(serie_file, votes_file, conf_file)
aggregation_from_input_votes_confs(votes_file,conf_file,agg_csv,agg_csv_detail,agg_tail_csv, stack_plot_file, stack_plot_conf_file, agg_plot_file, tail_plot_file, plot=True)