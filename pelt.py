import ruptures as rpt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results(cenario, file):
    # Define folders
    results_folder = "results/" + cenario
    plots_folder = "plots/" + cenario

    # Create folders if they don't exist
    os.makedirs(plots_folder, exist_ok=True)

    # Define files
    pelt_file = results_folder + "/pelt_" + file
    plot_file = plots_folder + "/pelt_" + file.replace(".csv", ".png")
    
    # Read the PELT results
    results = pd.read_csv(pelt_file)
    
    # Convert timestamps to datetime to avoid conversion errors
    results['timestamp'] = pd.to_datetime(results['timestamp'])
    
    timestamps = results['timestamp'].values
    values = results['value'].values
    CP = results[results['CP'] == 1]['timestamp'].values

    # Plotting
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, values, label='Data')  # type: ignore
    
    for i, cp in enumerate(CP):
        plt.axvline(x=cp, color='red', linestyle='--', label='Change Point' if i == 0 else "")
        
    plt.title(f'PELT - {file}')
    plt.xlabel('Timestamp')
    plt.ylabel(results.columns[1])
    plt.legend()
    plt.savefig(plot_file)
    plt.close()

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
    pelt_file = results_folder + "/pelt_" + file
    
    # Read the series data
    data = pd.read_csv(serie_file)
    timestamps = data.iloc[:, 0].values
    X = data.iloc[:, 1].values

    # Apply PELT algorithm
    algo = rpt.Pelt(model='rbf').fit(X)
    pen = 3
    result = algo.predict(pen=pen)
    CP = np.array(result[:-1]).astype(int)
    CP = CP.tolist()

    # Save in csv file with timestamps, X, and CP column with 1 for change points and 0 for non-change points
    results = pd.DataFrame({
        'timestamp': timestamps,
        'value': X,
        'CP': [1 if i in CP else 0 for i in range(len(X))]
    })
    results.to_csv(pelt_file, index=False)

    # Plot results
    plot_results(cenario, file)

def multiple_files(cenario):
    serie_folder = "series/"+ cenario
    files = os.listdir(serie_folder)
    for file in files:
        if file.endswith(".csv"):
            try:
                single_file(cenario, file)
            except Exception as e:
                continue

if __name__ == "__main__":
    import time
    begin = time.time()

    cenario = "teste"
    # file = "teste01.csv"
    # single_file(cenario,file)
    multiple_files(cenario)

    end = time.time()
    print(f"Tempo total: {end - begin:.2f} segundos")