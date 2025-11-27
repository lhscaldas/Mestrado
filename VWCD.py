import numpy as np
from scipy.stats import betabinom
import time

# Normalização
def pos_fun(ll, prior, tau):
    c = np.nanmax(ll)
    lse = c + np.log(np.nansum(prior * np.exp(ll - c)))
    p = ll[tau] + np.log(prior[tau]) - lse
    return np.exp(p)

# Agregações
def agg_linear(vote_list):
    vote_list = np.array(vote_list)
    p = vote_list.mean()
    return p

def agg_multiplicativa(vote_list):
    vote_list = np.array(vote_list)
    prod1 = vote_list.prod()
    prod2 = (1 - vote_list).prod()
    p = prod1 / (prod1 + prod2)
    return p

def agg_logaritmica(vote_list, ws):
    vote_list = np.array(vote_list)
    prod1 = np.prod(vote_list ** ws)
    prod2 = np.prod((1 - vote_list) ** ws)
    p = prod1 / (prod1 + prod2)
    return p

def agg_otima(vote_list, ws, lamb):
    vote_list = np.array(vote_list)
    alfa = lamb / (1+lamb)
    prod1 = np.prod(vote_list ** (alfa * ws))
    prod2 = np.prod((1 - vote_list) ** (alfa * ws))
    p = prod1 / (prod1 + prod2)
    return p

# Pesos
def ws_H_exp(H_list):
    beta = 5
    ws_raw = np.exp(- beta * np.array(H_list))
    ws = ws_raw / ws_raw.sum()
    return ws

def ws_H(H_list):
    ws_raw = 1 / (np.array(H_list) + 1e-9)
    ws = ws_raw / ws_raw.sum()
    return ws

def ws_U(vote_list):
    ws_raw = np.ones_like(vote_list)
    ws = ws_raw / (len(vote_list) + 1e-9)
    return ws

# VWCD
def vwcd(X, w, vote_p_thr, ab=2, aggreg=agg_linear, pesos=None, lamb=None, verbose=False):
    def loglik(x, loc, scale):
        n = len(x)
        c = 1 / np.sqrt(2 * np.pi)
        y = n * np.log(c / scale) - (1 / (2 * scale**2)) * ((x - loc) ** 2).sum()
        return y

    N = len(X)
    i_ = np.arange(0, w - 3)
    prior_w = betabinom.pmf(i_, n=w - 4, a=ab, b=ab)

    votes = {i: [] for i in range(N)}
    entropies = {i: [] for i in range(N)}
    lcp = 0
    CP = []
    windows = [] # Armazena votos de cada janela
    
    vote_counts = np.zeros(N)      # Array para armazenar o número de votos
    agg_probs = np.zeros(N)        # Array para armazenar probabilidades agregadas

    startTime = time.time()
    for n in range(w - 1, N):
        Xw = X[n - w + 1 : n + 1]
        LLR_h = []
        min_std = 1e-9

        for nu in range(2, w - 1):
            # Hipótese HA
            x1 = Xw[:nu]
            m1 = x1.mean()
            s1 = x1.std(ddof=1)
            s1 = max(s1, min_std)
            logL1 = loglik(x1, loc=m1, scale=s1)
            x2 = Xw[nu:]
            m2 = x2.mean()
            s2 = x2.std(ddof=1)
            s2 = max(s2, min_std)
            logL2 = loglik(x2, loc=m2, scale=s2)

            # Cálculo do LLR
            llr = logL1 + logL2
            LLR_h.append(llr)

        LLR_h = np.array(LLR_h)
        pos = [pos_fun(LLR_h, prior_w, nu) for nu in range(w - 3)]
        pos = [np.nan] * 2 + pos + [np.nan]
        pos = np.array(pos)
        
        pos_valid = pos[~np.isnan(pos)]
        windows.append(pos_valid.copy()) 
        pos_safe = np.clip(pos_valid, 1e-10, 1.0)
        H_janela = -np.sum(pos_safe * np.log(pos_safe))

        for nu in range(2, w - 1):
            p_vote_h = pos[nu]
            j = n - w + 1 + nu
            votes[j].append(p_vote_h)
            entropies[j].append(H_janela)
            vote_counts[j] += 1

        votes_list = votes[n - w + 1]
        H_list = entropies[n - w + 1]
        num_votes = len(votes_list)

        if num_votes > 0:
            if pesos is not None:
                ws = pesos(H_list)
                if lamb is not None:
                    agg_vote = agg_otima(votes_list, ws, lamb)
                else:
                    agg_vote = aggreg(votes_list, ws)
            else:
                agg_vote = aggreg(votes_list)
        else:
            agg_vote = 0.0

        agg_probs[n - w + 1] = agg_vote

        if agg_vote > vote_p_thr:
            if verbose:
                print(f'Changepoint at n={n-w+1}, p={agg_vote}, n={num_votes} votes')
            lcp = n - w + 1
            CP.append(lcp)

    endTime = time.time()
    elapsedTime = endTime - startTime
    return CP, elapsedTime, vote_counts, agg_probs, votes, windows