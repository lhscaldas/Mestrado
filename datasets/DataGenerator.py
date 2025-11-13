import numpy as np
import pandas as pd
import os

def generate_series(N, means, stds):
    
    n_states = 3

    if len(means) != n_states or len(stds) != n_states:
        raise ValueError("As listas 'means' e 'stds' devem ter 3 elementos.")
    
    p = 0.99
    A = np.array([
        [p, (1 - p) / 2, (1 - p) / 2],
        [(1 - p) / 2, p, (1 - p) / 2],
        [(1 - p) / 2, (1 - p) / 2, p]
    ])

    pi = np.array([1/3, 1/3, 1/3])

    states = np.zeros(N, dtype=int)
    time_series = np.zeros(N)

    states[0] = np.random.choice(n_states, p=pi)
    time_series[0] = np.random.normal(means[states[0]], stds[states[0]])

    for t in range(1, N):
        prev_state = states[t-1]
        transition_probs = A[prev_state, :]
        states[t] = np.random.choice(n_states, p=transition_probs)
        
        mean = means[states[t]]
        std = stds[states[t]]
        time_series[t] = np.random.normal(mean, std)

    return time_series, states

# plotar a serie com a cor da linha mudando de acordo com o estado
def plot_series(time_series, states):
    import matplotlib.pyplot as plt
    N = len(time_series)
    colors = ['blue', 'orange', 'green']
    plt.figure(figsize=(12, 6))
    for i in range(1, N):
        plt.plot([i-1, i], [time_series[i-1], time_series[i]], color=colors[states[i]])
    plt.title("Série Temporal Gerada por Cadeia de Markov Oculta")
    plt.xlabel("Tempo")
    plt.ylabel("Valor da Série")
    plt.grid()
    plt.show()

# criar um dataframe com a serie, os estados, uma coluna de timestamp, uma de cliente e uma de servidor
def create_dataframe(time_series, states, client, server):
    N = len(time_series)
    # timestamps de 30 em 30minutos a partir de 2025-08-01
    timestamps = pd.date_range(start='2025-08-01', periods=N, freq='30T')
    client_col = [client] * N
    server_col = [server] * N

    df = pd.DataFrame({
        'timestamp': timestamps,
        'value': time_series,
        'state': states,
        'client_name': client_col,
        'server_name': server_col
    })
    return df

# criar multiplos dfs para diferentes clientes e servidores e salvar em diferentes arquivos csv
def generate_csvs(num_pairs, N, means, stds):
    for i in range(num_pairs):
        client = f'Cliente_{i+1:02d}'
        server = f'Server_{i+1:02d}'
        time_series, states = generate_series(N, means, stds)
        df = create_dataframe(time_series, states, client, server)
        os.makedirs('artificial_time_series', exist_ok=True)
        df.to_csv(f'artificial_time_series/{client}_{server}.csv', index=False)


if __name__ == "__main__":
    N = 300
    means = [50, 100, 200]
    stds = [5, 10, 5]

    # time_series, states = generate_series(N, means, stds)
    # plot_series(time_series, states)
    
    generate_csvs(num_pairs=5, N=N, means=means, stds=stds)