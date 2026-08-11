from aux_vwcd_votes import window_votes
from aux_vwcd_aggregation import aggregation_from_input_votes_confs
import os
from plot_changepoint import plot_changepoint
import shutil

def single_file(W, C_FP, C_FN, cenario, method, file):
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

    window_votes(W, serie_file, votes_file, conf_file)
    aggregation_from_input_votes_confs(W, C_FP, C_FN, votes_file,conf_file,agg_csv,agg_csv_detail,agg_tail_csv, stack_plot_file, stack_plot_conf_file, agg_plot_file, tail_plot_file, plot=True)

def multiple_files(W, C_FP, C_FN, cenario, method, threshold=0.95):
    serie_folder = "series/"+ cenario
    files = os.listdir(serie_folder)

    results_folder = os.path.join("results", cenario, method)
    if os.path.exists(results_folder):
        shutil.rmtree(results_folder)
    os.makedirs(results_folder, exist_ok=True)
    
    plots_folder = os.path.join("plots", cenario, method)
    if os.path.exists(plots_folder):
        shutil.rmtree(plots_folder)
    os.makedirs(plots_folder, exist_ok=True)

    for file in files:
        if file.endswith(".csv"):
            try:
                single_file(W, C_FP, C_FN, cenario, method, file)
                plot_changepoint(cenario, method, file, threshold=threshold, save=True)
                plot_changepoint(cenario, method, file, threshold=threshold, save=True, tail_plot=True)
            except Exception as e:
                print(f"Erro ao processar o arquivo '{file}': {e}")
                continue



if __name__ == "__main__":
    print("Starting VWCD change point detection...")
    import time
    method = "vwcd_w24_fn2"
    threshold = 0.95 # Só influencia nos plots
    W = 24
    C_FP = 1.0
    C_FN = 2.0


    NDT_folder = "NDT"
    cenarios = [f"{NDT_folder}/{p}" for p in os.listdir(f"series/{NDT_folder}") if os.path.isdir(f"series/{NDT_folder}/{p}") and p != "full"]
    # cenarios = ['teste_m110', 'teste_m130', 'teste_m150']
    for cenario in cenarios:
        begin = time.time()
        multiple_files(W, C_FP, C_FN, cenario, method, threshold)
        end = time.time()
        print(f"Cenário '{cenario}' processado em {end - begin:.2f} segundos")



# if __name__ == "__main__":
#     print("Starting VWCD change point detection...")
#     import time
    
#     threshold = 0.95 # Só influencia nos plots
    
#     # Parâmetros para testar
#     W_values = [20, 24]
#     cost_combinations = [
#         (1.0, 1.0), # C_FP=1, C_FN=1
#         (2.0, 1.0), # C_FP=2, C_FN=1
#         (1.0, 2.0)  # C_FP=1, C_FN=2
#     ]

#     cenarios = ['teste_m110', 'teste_m130', 'teste_m150']
    
#     for W in W_values:
#         for C_FP, C_FN in cost_combinations:
#             # Definição do sufixo com base nos custos
#             if C_FP == 1.0 and C_FN == 1.0:
#                 sufixo = "fp1"
#             elif C_FP == 2.0 and C_FN == 1.0:
#                 sufixo = "fp2"
#             elif C_FP == 1.0 and C_FN == 2.0:
#                 sufixo = "fn2"
            
#             method = f"vwcd_w{W}_{sufixo}"  # type: ignore
            
#             print(f"\n--- Iniciando testes para: {method} (W={W}, C_FP={C_FP}, C_FN={C_FN}) ---")
            
#             for cenario in cenarios:
#                 begin = time.time()
#                 multiple_files(W, C_FP, C_FN, cenario, method, threshold)
#                 end = time.time()
#                 print(f"Cenário '{cenario}' processado em {end - begin:.2f} segundos")