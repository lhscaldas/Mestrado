import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def generate_series(N, means, stds, p):
    
    n_states = len(means)

    if len(means) != len(stds):
        raise ValueError("As listas 'means' e 'stds' devem ter o mesmo tamanho.")
    
    A = np.full((n_states, n_states), (1-p) / (n_states - 1))
    np.fill_diagonal(A, p)

    pi = np.full(n_states, 1 / n_states)

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

def plot_series(time_series, states):
    N = len(time_series)
    n_states = np.max(states) + 1
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(n_states)]
    fig, ax = plt.subplots(figsize=(12, 6))
    legend_added = [False] * n_states
    for i in range(1, N):
        current_state = states[i]
        current_color = colors[current_state]
        label = None
        if not legend_added[current_state]:
            label = f"Estado {current_state}"
            legend_added[current_state] = True
        ax.plot([i-1, i], 
                [time_series[i-1], time_series[i]], 
                color=current_color, 
                label=label)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title("Série Temporal Gerada por Cadeia de Markov Oculta")
    ax.set_xlabel("Tempo")
    ax.set_ylabel("Valor")
    ax.grid()
    plt.tight_layout()
    plt.show()

def create_dataframe(time_series, states, client, server):
    N = len(time_series)
    
    timestamps = pd.date_range(start='2025-08-01', periods=N, freq='30min')
    client_col = [client] * N
    server_col = [server] * N

    state_series = pd.Series(states)
    label_CP = (state_series != state_series.shift(1)).astype(int)
    label_CP.iloc[0] = 0


    df = pd.DataFrame({
        'timestamp': timestamps,
        'value': time_series,
        'state': states,
        'value_cp': label_CP,
        'client_name': client_col,
        'server_name': server_col
    })
    return df


def generate_csvs(num_pairs, N, means, stds, p):
    if not os.path.exists('artificial_time_series'):
        os.makedirs('artificial_time_series')
    else:
        for file in os.listdir('artificial_time_series'):
            file_path = os.path.join('artificial_time_series', file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    for i in range(num_pairs):
        client = f'Cliente{i+1:02d}'
        server = f'Server{i+1:02d}'
        time_series, states = generate_series(N, means, stds, p)
        df = create_dataframe(time_series, states, client, server)
        df.to_csv(f'artificial_time_series/{client}_{server}.csv', index=False)


if __name__ == "__main__":
    N = 1440
    means = [50, 50, 100, 100]
    stds = [1, 3, 1, 3]
    p = 0.995

    # time_series, states = generate_series(N, means, stds, p)
    # plot_series(time_series, states)
    
    generate_csvs(num_pairs=5, N=N, means=means, stds=stds, p=p)