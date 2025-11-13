import numpy as np
from scipy.stats import betabinom
import time

def vwcd_original(X, w, ab, p_thr, vote_p_thr, vote_n_thr, y0, yw, aggreg, verbose=False):
    """
    Detecta pontos de mudança em uma série temporal usando o algoritmo Voting Windows Changepoint Detection.

    Parâmetros:
        X (array-like): Série temporal.
        w (int): Tamanho da janela deslizante de votação.
        ab (float): Hiperparâmetros alfa e beta da distribuição beta-binomial
        p_thr (float): Limiar de probabilidade para o voto de uma janela ser registrado.
        vote_p_thr (float): Limiar de probabilidade para definir um ponto de mudança após a agregação dos votos.
        vote_n_thr (float): Fração mínima da janela que precisa votar.
        y0 (float): Probabilidade a priori da função logística (início da janela).
        yw (float): Probabilidade a priori da função logística (início da janela).
        aggreg (str): Função de agregação para os votos ('posterior' ou 'mean').
        verbose (bool): Se True, exibe informações sobre os pontos de mudança detectados.

    Retorna:
        tuple: 
            - CP (list): Lista de índices dos pontos de mudança detectados.
            - M0 (list): Lista de médias estimadas nas janelas.
            - S0 (list): Lista de desvios padrão estimados nas janelas.
            - elapsedTime (float): Tempo total de execução do algoritmo.
            - vote_counts (array): Array com o número de votos para cada ponto.
            - agg_probs (array): Array com as probabilidades após agregação dos votos.
    """
    def pos_fun(ll, prior, tau):
        c = np.nanmax(ll)
        lse = c + np.log(np.nansum(prior * np.exp(ll - c)))
        p = ll[tau] + np.log(prior[tau]) - lse
        return np.exp(p)

    def votes_pos(vote_list, prior_v):
        vote_list = np.array(vote_list)
        prod1 = vote_list.prod() * prior_v
        prod2 = (1 - vote_list).prod() * (1 - prior_v)
        p = prod1 / (prod1 + prod2)
        return p

    def logistic_prior(x, w, y0, yw):
        a = np.log((1 - y0) / y0)
        b = np.log((1 - yw) / yw)
        k = (a - b) / w
        x0 = a / k
        y = 1 / (1 + np.exp(-k * (x - x0)))
        return y
    
    def loglik(x, loc, scale):
        n = len(x)
        c = 1 / np.sqrt(2 * np.pi)
        y = n * np.log(c / scale) - (1 / (2 * scale**2)) * ((x - loc) ** 2).sum()
        return y

    N = len(X)
    vote_n_thr = np.floor(w * vote_n_thr)
    i_ = np.arange(0, w - 3)
    prior_w = betabinom.pmf(i_, n=w - 4, a=ab, b=ab)
    x_votes = np.arange(1, w + 1)
    prior_v = logistic_prior(x_votes, w, y0, yw)

    votes = {i: [] for i in range(N)}
    lcp = 0
    CP = []
    
    vote_counts = np.zeros(N)      # Array para armazenar o número de votos
    agg_probs = np.zeros(N)        # Array para armazenar probabilidades agregadas

    startTime = time.time()
    for n in range(N):
        if n >= w - 1:
            Xw = X[n - w + 1 : n + 1]
            LLR_h = []
            for nu in range(1, w - 3 + 1):
                x1 = Xw[: nu + 1]
                m1 = x1.mean()
                s1 = x1.std(ddof=1)
                if np.round(s1, 3) == 0:
                    s1 = 0.001
                logL1 = loglik(x1, loc=m1, scale=s1)

                x2 = Xw[nu + 1 :]
                m2 = x2.mean()
                s2 = x2.std(ddof=1)
                if np.round(s2, 3) == 0:
                    s2 = 0.001
                logL2 = loglik(x2, loc=m2, scale=s2)

                llr = logL1 + logL2
                LLR_h.append(llr)

            LLR_h = np.array(LLR_h)
            pos = [pos_fun(LLR_h, prior_w, nu) for nu in range(w - 3)]
            pos = [np.nan] + pos + [np.nan] * 2
            pos = np.array(pos)

            p_vote_h = np.nanmax(pos)
            nu_map_h = np.nanargmax(pos)

            if p_vote_h >= p_thr:
                j = n - w + 1 + nu_map_h
                votes[j].append(p_vote_h)
                vote_counts[j] += 1

            votes_list = votes[n - w + 1]
            num_votes = len(votes_list)
            if num_votes >= vote_n_thr:
                if aggreg == 'posterior':
                    agg_vote = votes_pos(votes_list, prior_v[num_votes - 1])
                elif aggreg == 'mean':
                    agg_vote = np.mean(votes_list)
                else:
                    raise ValueError(f"Método de agregação desconhecido: '{aggreg}'. Use 'posterior' ou 'mean'.")
                agg_probs[n - w + 1] = agg_vote


                if agg_vote > vote_p_thr:
                    if verbose:
                        print(f'Changepoint at n={n-w+1}, p={agg_vote}, n={num_votes} votes')
                    lcp = n - w + 1
                    CP.append(lcp)

    endTime = time.time()
    elapsedTime = endTime - startTime
    return CP, elapsedTime, vote_counts, agg_probs


def vwcd(X, w, vote_p_thr, aggreg, verbose=False):
    """
    Detecta pontos de mudança em uma série temporal usando o algoritmo Voting Windows Changepoint Detection.
    Modificado por Henrique Caldas:
      - Remove os limiares de filtro de voto individual ('p_thr') e contagem mínima ('vote_n_thr') da lógica original.
      - Remove os parâmetros relacionados à prior logística ('y0', 'yw') e fixa internamente os parâmetros 'w0' e 'ab'.
      - Adiciona novos métodos de agregação alinhados com "Agregação Probabilística" (Siqueira, 2025):
        1. 'multiplicativa': Agregação Multiplicativa pelo Teorema de Bayes (Seção 4.1.3).
        2. 'logaritmica': Agregação Logarítmica (Seção 4.1.2) com pesos definidos pela
           Ponderação Informacional baseada na Divergência Média aos Pares (Seção 4.1.4).

    Parâmetros:
        X (array-like): Série temporal.
        w (int): Tamanho da janela deslizante de votação.
        ab (float): Hiperparâmetros alfa e beta da distribuição beta-binomial
        vote_p_thr (float): Limiar de probabilidade para definir um ponto de mudança após a agregação dos votos.
        aggreg (str): Função de agregação para os votos ('mean', 'multiplicativa', 'logaritmica_KL' ou 'logaritmica_H').
        verbose (bool): Se True, exibe informações sobre os pontos de mudança detectados.

    Retorna:
        tuple: 
            - CP (list): Lista de índices dos pontos de mudança detectados.
            - elapsedTime (float): Tempo total de execução do algoritmo.
            - vote_counts (array): Array com o número de votos para cada ponto.
            - agg_probs (array): Array com as probabilidades após agregação dos votos.
    """

    def loglik(x, loc, scale):
        n = len(x)
        c = 1 / np.sqrt(2 * np.pi)
        y = n * np.log(c / scale) - (1 / (2 * scale**2)) * ((x - loc) ** 2).sum()
        return y

    def pos_fun(ll, prior, tau):
        c = np.nanmax(ll)
        lse = c + np.log(np.nansum(prior * np.exp(ll - c)))
        p = ll[tau] + np.log(prior[tau]) - lse
        return np.exp(p)

    def agg_multiplicativa(vote_list):
        vote_list = np.array(vote_list)
        prod1 = vote_list.prod()
        prod2 = (1 - vote_list).prod()
        p = prod1 / (prod1 + prod2)
        return p
        
    def entropy(votes_list):
        votes_list = np.clip(np.array(votes_list), 1e-9, 1 - 1e-9)
        n = len(votes_list)
        if n <= 1:
            return np.ones(n)
        H = np.zeros(n)
        for i in range(n):
            p_i = votes_list[i]
            H[i] = - (p_i * np.log(p_i) + (1 - p_i) * np.log(1 - p_i))
        ws_brutos = 1 / H
        ws_normalizados = ws_brutos / ws_brutos.sum()
        return ws_normalizados
    
    def div_media(votes_list):
        votes_list = np.clip(np.array(votes_list), 1e-9, 1 - 1e-9)
        n = len(votes_list)
        if n <= 1:
            return np.ones(n)
        D_bar = np.zeros(n)
        for i in range(n):
            p_i = votes_list[i]
            p_not_i = np.delete(votes_list, i)
            D = np.zeros(n-1)
            for j in range(n-1):
                p_j = p_not_i[j]
                D[j] = p_i * np.log(p_i / p_j) + (1 - p_i) * np.log((1 - p_i) / (1 - p_j))
            D_bar[i] = D.mean()
        ws_brutos = 1 / D_bar
        ws_normalizados = ws_brutos / ws_brutos.sum()   
        return ws_normalizados
    
    def agg_logaritmica(vote_list, ws):
        vote_list = np.array(vote_list)
        prod1 = np.prod(vote_list ** ws)
        prod2 = np.prod((1 - vote_list) ** ws)
        p = prod1 / (prod1 + prod2)
        return p
    
    ab = 2.0
    
    N = len(X)
    i_ = np.arange(0, w - 3)
    prior_w = betabinom.pmf(i_, n=w - 4, a=ab, b=ab)

    votes = {i: [] for i in range(N)}
    lcp = 0
    CP = []
    
    vote_counts = np.zeros(N)      # Array para armazenar o número de votos
    agg_probs = np.zeros(N)        # Array para armazenar probabilidades agregadas

    startTime = time.time()
    for n in range(N):
        if n >= w - 1:
            Xw = X[n - w + 1 : n + 1]
            LLR_h = []

            s_janela = np.std(Xw, ddof=1)
            min_std = s_janela * 0.05
            if min_std < 1e-8: min_std = 1e-8

            for nu in range(1, w - 3 + 1):
                x1 = Xw[: nu + 1]
                m1 = x1.mean()
                s1 = x1.std(ddof=1)
                s1 = max(s1, min_std)
                logL1 = loglik(x1, loc=m1, scale=s1)

                x2 = Xw[nu + 1 :]
                m2 = x2.mean()
                s2 = x2.std(ddof=1)
                s2 = max(s2, min_std)
                logL2 = loglik(x2, loc=m2, scale=s2)

                llr = logL1 + logL2
                LLR_h.append(llr)

            LLR_h = np.array(LLR_h)
            pos = [pos_fun(LLR_h, prior_w, nu) for nu in range(w - 3)]
            pos = [np.nan] + pos + [np.nan] * 2
            pos = np.array(pos)

            p_vote_h = np.nanmax(pos)
            nu_map_h = np.nanargmax(pos)

            j = n - w + 1 + nu_map_h
            votes[j].append(p_vote_h)
            vote_counts[j] += 1

            votes_list = votes[n - w + 1]
            num_votes = len(votes_list)

            if aggreg == 'mean':
                agg_vote = np.mean(votes_list) if num_votes > 0 else 0
            elif aggreg == 'multiplicativa':
                agg_vote = agg_multiplicativa(votes_list) if num_votes > 0 else 0
            elif aggreg == 'logaritmica_KL':
                ws = div_media(votes_list)
                agg_vote = agg_logaritmica(votes_list, ws) if num_votes > 0 else 0
            elif aggreg == 'logaritmica_H':
                ws = entropy(votes_list)
                agg_vote = agg_logaritmica(votes_list, ws) if num_votes > 0 else 0
            else:
                raise ValueError(f"Método de agregação desconhecido: '{aggreg}'. Use 'mean', 'multiplicativa', 'logaritmica_KL' ou 'logaritmica_H'.")
            agg_probs[n - w + 1] = agg_vote

            if agg_vote > vote_p_thr:
                if verbose:
                    print(f'Changepoint at n={n-w+1}, p={agg_vote}, n={num_votes} votes')
                lcp = n - w + 1
                CP.append(lcp)

    endTime = time.time()
    elapsedTime = endTime - startTime
    return CP, elapsedTime, vote_counts, agg_probs