"""
Decision-support tool: ingest expert votes + confidences, plot stacked views,
then aggregate expert probabilities using Entropy–KL (confidence prior + KL regularization),
and finally plot/save the aggregated time series.

This script is intended to be student-friendly and heavily commented.

-----------------------------------------------------------------------
PARTS IMPLEMENTED
(1) Read votes CSV:
    - col 1: timestamp
    - col 2: real value (not used yet for aggregation, but kept)
    - col 3..: vote_1..vote_Nexp  (probabilities in [0,1])

(2) Read confidence CSV:
    - col 1: timestamp
    - col 2..: confs_vote_1..confs_vote_Nexp (probabilities in [0,1])

(3) Ask user:
    - votes CSV filename
    - confidence CSV filename
    - number of ignored experts at beginning (k_begin) and end (k_end)
    - window length W (rows), to ignore first W and last W rows

(4) Checks:
    - votes/confidences in [0,1]
    - timestamp alignment
    - ignored experts columns are zero on the trimmed region

(5) Plots:
    - stacked votes over time (trimmed)
    - stacked votes*confidence over time (trimmed)

(6) NEW: Entropy–KL aggregation (per prompt)
    For each time t:
      q_i(t) = c_i(t) / sum_j c_j(t)
      Choose w(t) by minimizing:
        sum_i w_i rho(p_i; mu) + gamma * sum_i w_i log(w_i/q_i)
      with rho(p_i;mu) = (p_i - mu)^2 and mu = sum_i w_i p_i.
      Use fixed-point iterations using the softmax-style update:
        w_i ∝ q_i * exp(-(p_i-mu)^2 / gamma)

    Gamma selection:
      gamma = -delta^2 / log(r)  (delta>0, 0<r<1)

(7) Plot aggregated mu(t) vs time, and write CSV:
    - col1 timestamp
    - col2 aggregated probability mu(t)

-----------------------------------------------------------------------
TO BE CONTINUED: later you can add Bayesian precision aggregation, variance inflation,
and decision rules.
"""

from __future__ import annotations

import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os


# ---------------------------- Helpers ----------------------------

def _prompt_int(prompt: str, min_value: int = 0) -> int:
    """Prompt for an integer >= min_value."""
    while True:
        s = input(prompt).strip()
        try:
            v = int(s)
            if v < min_value:
                print(f"Please enter an integer >= {min_value}.")
                continue
            return v
        except ValueError:
            print("Please enter a valid integer.")


def _prompt_float(prompt: str, min_value: float | None = None, max_value: float | None = None) -> float:
    """Prompt for a float (optionally bounded)."""
    while True:
        s = input(prompt).strip()
        try:
            v = float(s)
            if min_value is not None and v < min_value:
                print(f"Please enter a value >= {min_value}.")
                continue
            if max_value is not None and v > max_value:
                print(f"Please enter a value <= {max_value}.")
                continue
            return v
        except ValueError:
            print("Please enter a valid number.")


def _infer_vote_columns(df_votes: pd.DataFrame) -> Tuple[str, str, List[str]]:
    """
    Infer column names:
      timestamp column = first column
      value column     = second column
      vote columns     = remaining columns
    """
    if df_votes.shape[1] < 3:
        raise ValueError("Votes CSV must have at least 3 columns: timestamp, value, and >=1 vote column.")
    ts_col = df_votes.columns[0]
    val_col = df_votes.columns[1]
    vote_cols = list(df_votes.columns[2:])
    return ts_col, val_col, vote_cols


def _infer_conf_columns(df_conf: pd.DataFrame, n_exp: int) -> Tuple[str, List[str]]:
    """
    Infer confidence columns:
      timestamp column = first column
      confidence columns = next n_exp columns (must exist)
    """
    if df_conf.shape[1] < 1 + n_exp:
        raise ValueError(
            f"Confidence CSV must have at least {1+n_exp} columns: timestamp + {n_exp} confidence columns."
        )
    ts_col = df_conf.columns[0]
    conf_cols = list(df_conf.columns[1:1 + n_exp])
    return ts_col, conf_cols


def _check_probabilities(df: pd.DataFrame, cols: List[str], name: str) -> None:
    """Check all entries in df[cols] are probabilities in [0,1] (no NaNs)."""
    arr = df[cols].to_numpy(dtype=float)
    bad = np.isnan(arr) | (arr < 0.0) | (arr > 1.0)
    if np.any(bad):
        idxs = np.argwhere(bad)
        sample = idxs[:10]
        details = []
        for r, c in sample:
            details.append(f"(row={int(r)}, col={cols[int(c)]}, val={arr[r, c]})")
        raise ValueError(f"{name}: Found values outside [0,1] or NaN. Examples: " + ", ".join(details))


def _align_on_timestamp(
    df_votes: pd.DataFrame, ts_votes: str,
    df_conf: pd.DataFrame, ts_conf: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensure timestamps match exactly (same set and order) by inner-merge.
    This is important because we will multiply vote(t,i) * conf(t,i).
    """
    df_votes = df_votes.copy()
    df_conf = df_conf.copy()
    df_votes[ts_votes] = pd.to_datetime(df_votes[ts_votes])
    df_conf[ts_conf] = pd.to_datetime(df_conf[ts_conf])

    # --- Detect duplicates: MUST be one-to-one for safe alignment ---
    if df_votes[ts_votes].duplicated().any():
        dups = df_votes.loc[df_votes[ts_votes].duplicated(), ts_votes].head(5).to_list()
        raise ValueError(f"Votes file has duplicate timestamps (examples): {dups}")

    if df_conf[ts_conf].duplicated().any():
        dups = df_conf.loc[df_conf[ts_conf].duplicated(), ts_conf].head(5).to_list()
        raise ValueError(f"Confidence file has duplicate timestamps (examples): {dups}")


    merged = pd.merge(
        df_votes, df_conf,
        left_on=ts_votes, right_on=ts_conf,
        how="inner",
        suffixes=("", "_conf"),
        validate="one_to_one"   # <--- critical: errors if duplicates exist
    )

    if merged.shape[0] != df_votes.shape[0] or merged.shape[0] != df_conf.shape[0]:
        raise ValueError(
            "Votes and confidence files do not align by timestamp. "
            "Check for missing/extra timestamps or formatting differences."
        )

    aligned_votes = merged[df_votes.columns].reset_index(drop=True)
    aligned_conf = merged[df_conf.columns].reset_index(drop=True)

    if not np.all(aligned_votes[ts_votes].to_numpy() == aligned_conf[ts_conf].to_numpy()):
        raise ValueError("Timestamp ordering mismatch after merge (unexpected).")

    return aligned_votes, aligned_conf


def _trim_by_window(df: pd.DataFrame, W: int) -> pd.DataFrame:
    """Ignore the first W rows and the last W rows."""
    if W <= 0:
        return df.copy().reset_index(drop=True)
    T = df.shape[0]
    if T <= 2 * W:
        raise ValueError(f"Window W={W} too large for T={T}. Need T > 2W.")
    return df.iloc[W:-W].reset_index(drop=True)


def _check_ignored_are_zero(
    df: pd.DataFrame,
    cols: List[str],
    k_begin: int,
    k_end: int,
    name: str,
    eps: float = 0.0
) -> None:
    """
    REQUIRED check (applied on TRIMMED time range):
      Ignored experts at beginning and end must be exactly zero (or within eps).
    """
    n = len(cols)
    if k_begin + k_end > n:
        raise ValueError(f"{name}: k_begin + k_end = {k_begin+k_end} exceeds number of experts {n}.")

    begin_cols = cols[:k_begin]
    end_cols = cols[n - k_end:] if k_end > 0 else []

    def _has_nonzero(subcols: List[str]) -> np.ndarray:
        if not subcols:
            return np.zeros((df.shape[0],), dtype=bool)
        arr = df[subcols].to_numpy(dtype=float)
        return np.any(np.abs(arr) > eps, axis=1)

    bad_rows = np.where(_has_nonzero(begin_cols) | _has_nonzero(end_cols))[0]
    if bad_rows.size > 0:
        r0 = int(bad_rows[0])
        msg = [f"{name}: Ignored columns are not all zero on the trimmed time range."]
        msg.append(f"First offending row index (trimmed): {r0}")
        if begin_cols:
            msg.append(f"  Beginning ignored cols: {df.loc[df.index[r0], begin_cols].to_dict()}")
        if end_cols:
            msg.append(f"  Ending ignored cols: {df.loc[df.index[r0], end_cols].to_dict()}")
        raise ValueError("\n".join(msg))


def _stackplot_time_series(
    times: np.ndarray,
    series: np.ndarray,
    labels: List[str],
    title: str,
    ylabel: str,
    file = None
) -> None:
    """Stacked plot: each expert series stacked on top of the previous ones."""
    plt.figure(figsize=(12, 6))
    ys = [series[:, j] for j in range(series.shape[1])]
    plt.stackplot(times, ys, labels=labels)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    if len(labels) <= 12:
        plt.legend(loc="upper left")
    else:
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, ncol=1)
    plt.tight_layout()
    if file:
        if os.path.exists(file):
            os.remove(file)
        plt.savefig(file)
    else:
        plt.show()


def _plot_time_series(times: np.ndarray, y: np.ndarray, title: str, ylabel: str, file = None) -> None:
    """Simple line plot for aggregated probability over time."""
    plt.figure(figsize=(12, 4))
    plt.plot(times, y, color="gray")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.tight_layout()
    if file:
        if os.path.exists(file):
            os.remove(file)
        plt.savefig(file)
    else:
        plt.show()


# ---------------------------- Entropy–KL aggregation (NEW) ----------------------------

def gamma_from_downweight(delta: float, r: float) -> float:
    """
    Set gamma by the rule:
        exp(-delta^2 / gamma) = r   =>   gamma = -delta^2 / log(r)

    delta > 0: desired deviation amount
    r in (0,1): desired multiplicative downweighting factor
    """
    if delta <= 0:
        raise ValueError("delta must be > 0.")
    if not (0 < r < 1):
        raise ValueError("r must be in (0,1).")
    return - (delta ** 2) / np.log(r)


def entropy_kl_aggregate_one_time(
    p: np.ndarray,
    c: np.ndarray,
    gamma: float,
    eps: float = 1e-12,
    max_iter: int = 200,
    tol: float = 1e-12
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute the Entropy–KL aggregate for a single time t.

    Inputs:
      p: shape (n,), expert probabilities p_i
      c: shape (n,), expert confidences c_i
      gamma: regularization parameter (>0)
      eps: numerical floor to avoid division by zero / log(0)
      max_iter, tol: fixed-point stopping controls

    Implements:
      q_i = c_i / sum_j c_j  (confidence prior on simplex)
      w_i ∝ q_i * exp(-(p_i - mu)^2 / gamma)
      mu = sum_i w_i p_i

    Returns:
      mu: aggregated probability
      w: final weights (sum to 1)
      q: confidence prior (sum to 1)
    """
    n = p.shape[0]
    if c.shape[0] != n:
        raise ValueError("p and c must have same length.")
    if gamma <= 0:
        raise ValueError("gamma must be > 0.")

    # ---- (1) Confidence prior q on the simplex ----
    # If sum(c) is zero, we fall back to a uniform prior (no information).
    c_pos = np.clip(c.astype(float), 0.0, 1.0)
    s = float(np.sum(c_pos))
    if s <= 0.0:
        q = np.ones(n, dtype=float) / n
    else:
        q = c_pos / s

    # Numerical safety: prevent q_i = 0 exactly, because we will form w_i/q_i inside a KL.
    q = np.clip(q, eps, 1.0)
    q = q / np.sum(q)

    # ---- (2) Fixed-point loop: w -> mu -> w ----
    # A good initialization is w=q (trust confidence before considering disagreement).
    w = q.copy()
    mu = float(np.dot(w, p))

    for _ in range(max_iter):
        # squared loss disagreements: rho_i = (p_i - mu)^2
        rho = (p - mu) ** 2

        # softmax-style update: w_i ∝ q_i * exp(-rho_i/gamma)
        logits = -rho / gamma

        # numerical stabilization for exponentials: subtract max(logits)
        logits = logits - np.max(logits)
        w_new = q * np.exp(logits)
        w_new_sum = float(np.sum(w_new))

        if w_new_sum <= 0.0 or not np.isfinite(w_new_sum):
            # This should not happen under normal conditions, but we guard anyway.
            w_new = q.copy()
            w_new_sum = float(np.sum(w_new))

        w_new = w_new / w_new_sum

        mu_new = float(np.dot(w_new, p))

        # stopping condition: mu and w stable
        if abs(mu_new - mu) < tol and np.max(np.abs(w_new - w)) < tol:
            w, mu = w_new, mu_new
            break

        w, mu = w_new, mu_new

    return mu, w, q


def entropy_kl_aggregate_time_series(
    votes_mat: np.ndarray,
    conf_mat: np.ndarray,
    gamma: float,
    eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Entropy–KL aggregation for every time row.

    votes_mat: shape (T, n)
    conf_mat:  shape (T, n)
    returns:
      mu_series: shape (T,), aggregated probability per time
      w_series:  shape (T, n), weights per time
    """
    T, n = votes_mat.shape
    if conf_mat.shape != (T, n):
        raise ValueError("votes_mat and conf_mat must have the same shape (T, n).")

    mu_series = np.zeros(T, dtype=float)
    w_series = np.zeros((T, n), dtype=float)

    for t in range(T):
        p = votes_mat[t, :]
        c = conf_mat[t, :]
        mu_t, w_t, _q_t = entropy_kl_aggregate_one_time(p=p, c=c, gamma=gamma, eps=eps)
        mu_series[t] = mu_t
        w_series[t, :] = w_t

    return mu_series, w_series


# ---------------------------- Main script ----------------------------

def aggregation_from_input_votes_confs(votes_file,conf_file,out_csv,out_csv_detail,tail_csv,stack_plot_file=None, stack_plot_conf_file=None, agg_plot_file=None,tail_plot_file=None,            plot=False) -> None:
    # print("\n=== Expert Vote + Confidence Ingestion + Entropy–KL Aggregation ===\n")

    # votes_file = input("Enter VOTES CSV filename/path: ").strip()
    # conf_file = input("Enter CONFIDENCE CSV filename/path: ").strip()

    # k_begin = _prompt_int("How many experts to IGNORE at the BEGINNING (k_begin)? ", min_value=0)
    # k_end = _prompt_int("How many experts to IGNORE at the END (k_end)? ", min_value=0)
    # W = _prompt_int("Enter WINDOW length W (rows) to ignore at start and end: ", min_value=0)

    k_begin = 5
    k_end = 4
    W = 20

    # Gamma selection parameters (per prompt)
    # print("\nGamma selection (per downweighting target):")
    # delta = _prompt_float("Enter delta (e.g., 0.2): ", min_value=1e-12)
    # r = _prompt_float("Enter r in (0,1) (e.g., 0.5): ", min_value=1e-12, max_value=1.0 - 1e-12)
    delta = 0.2
    r = 0.5

    gamma = gamma_from_downweight(delta=delta, r=r)
    # print(f"\nComputed gamma = -delta^2 / log(r) = {gamma:.6f}\n")

    # --- Read CSVs ---
    df_votes = pd.read_csv(votes_file)
    df_conf = pd.read_csv(conf_file)

    # --- for safety, ensure timestamp columns are datetime ---
    df_votes['ts_votes'] = pd.to_datetime(df_votes['ts_votes'])
    df_conf['ts_conf'] = pd.to_datetime(df_conf['ts_conf'])

    # --- Infer columns and Nexp ---
    ts_votes, val_col, vote_cols = _infer_vote_columns(df_votes)
    n_exp = len(vote_cols)
    ts_conf, conf_cols = _infer_conf_columns(df_conf, n_exp)

    # print("=== Inferred structure ===")
    # print(f"Nexp (from votes columns): {n_exp}")
    # print(f"k_begin={k_begin}, k_end={k_end}, W={W}")
    # print(f"Vote columns: {vote_cols[0]} ... {vote_cols[-1]}")
    # print(f"Conf columns: {conf_cols[0]} ... {conf_cols[-1]}")
    # print()

    # --- Align timestamps first ---
    df_votes, df_conf = _align_on_timestamp(df_votes, ts_votes, df_conf, ts_conf)

    # --- Probability checks on FULL data ---
    _check_probabilities(df_votes, vote_cols, name="Votes (FULL)")
    _check_probabilities(df_conf, conf_cols, name="Confidences (FULL)")

    # --- Trim by window boundaries (ignore first W and last W rows) ---
    df_votes_t = _trim_by_window(df_votes, W)
    df_conf_t = _trim_by_window(df_conf, W)

    # --- Now check ignored-experts zero property on the TRIMMED time range ---
    _check_ignored_are_zero(df_votes_t, vote_cols, k_begin, k_end, name="Votes (ignored check, TRIMMED)", eps=0.0)
    _check_ignored_are_zero(df_conf_t, conf_cols, k_begin, k_end, name="Confidences (ignored check, TRIMMED)", eps=0.0)

    # print("All ingestion checks PASSED (with ignored-expert check on trimmed region).\n")

    # --- Prepare arrays for plotting and aggregation (TRIMMED region) ---
    times = pd.to_datetime(df_votes_t[ts_votes]).to_numpy()

    votes_mat = df_votes_t[vote_cols].to_numpy(dtype=float)      # (T, n)
    conf_mat = df_conf_t[conf_cols].to_numpy(dtype=float)        # (T, n)
    votes_times_conf = votes_mat * conf_mat                       # (T, n)

    if plot:
        # --- Plot 1: stacked votes (TRIMMED) ---
        _stackplot_time_series(
            times=times,
            series=votes_mat,
            labels=vote_cols,
            title="Stacked Expert Votes Over Time (trimmed by window boundaries)",
            ylabel="Stacked vote probability",
            file=stack_plot_file
        )

        # --- Plot 2: stacked votes * confidence (TRIMMED) ---
        _stackplot_time_series(
            times=times,
            series=votes_times_conf,
            labels=[f"{vc}×conf" for vc in vote_cols],
            title="Stacked Expert Votes × Confidence Over Time (trimmed by window boundaries)",
            ylabel="Stacked (vote × confidence)",
            file=stack_plot_conf_file
        )

    # ----------------------------
    # (NEW) Entropy–KL aggregation
    # ----------------------------
    # print("=== Entropy–KL aggregation (time series) ===")
    # print(f"gamma (from delta={delta}, r={r}) = {gamma:.6f}")

    mu_series, w_series = entropy_kl_aggregate_time_series(votes_mat=votes_mat, conf_mat=conf_mat, gamma=gamma)

    # print(f"Aggregated mu(t) summary: min={mu_series.min():.6f}, max={mu_series.max():.6f}, mean={mu_series.mean():.6f}")
    # print("Example at first trimmed time:")
    # print(f"  time = {times[0]}")
    # print(f"  mu   = {mu_series[0]:.6f}")
    # print(f"  sum(w)= {w_series[0].sum():.12f}  (should be 1 up to numerical precision)\n")

    if plot:
        # --- Plot 3: aggregated probability over time ---
        _plot_time_series(
            times=times,
            y=mu_series,
            title="Aggregated Probability Over Time (Entropy–KL)",
            ylabel="Aggregated probability mu(t)",
            file=agg_plot_file
        )

    # --- Write output CSV with timestamp + aggregated probability ---
    # out_csv = "aggregated_entropy_kl.csv"
    df_out = pd.DataFrame({
        "timestamp": times,
        "mu_agg": mu_series
    })
    df_out.to_csv(out_csv, index=False)
    # print(f"Wrote aggregated time series to: {out_csv}")

    # ------------------------------------------------------------
    # NEW: Write a "full details" CSV per timestamp:
    #   timestamp, mu_agg,
    #   then for i=1..Nexp: vote_i, conf_i, q_i, w_i
    # ------------------------------------------------------------

    # Recompute q(t,i) for every time t (the confidence prior at each timestamp)
    eps = 1e-12
    q_series = np.zeros_like(conf_mat, dtype=float)  # shape (T, Nexp)

    for t in range(conf_mat.shape[0]):
        c_row = np.clip(conf_mat[t, :], 0.0, 1.0)
        s = float(np.sum(c_row))
        if s <= 0.0:
            q_row = np.ones_like(c_row) / c_row.size
        else:
            q_row = c_row / s
        # numerical safety (same as in entropy_kl_aggregate_one_time)
        q_row = np.clip(q_row, eps, 1.0)
        q_row = q_row / np.sum(q_row)
        q_series[t, :] = q_row

    # Build the detailed output table
    detail_cols = {
        "timestamp": times,
        "mu_agg": mu_series
    }

    for j, vcol in enumerate(vote_cols):
        # vote column name already like "vote_1", "vote_2", ...
        detail_cols[f"{vcol}"] = votes_mat[:, j]
        detail_cols[f"conf_{vcol}"] = conf_mat[:, j]
        detail_cols[f"q_{vcol}"] = q_series[:, j]
        detail_cols[f"w_{vcol}"] = w_series[:, j]

    df_detail = pd.DataFrame(detail_cols)

    # out_csv_detail = "aggregated_entropy_kl_full_details.csv"
    df_detail.to_csv(out_csv_detail, index=False)
    # print(f"Wrote full per-timestamp details to: {out_csv_detail}")

    # ------------------------------------------------------------
    # NEW: Risk-based thresholding using costs and plotting P(theta >= T*)
    # Plug this RIGHT AFTER you compute mu_series (and after you have `times`)
    # and RIGHT BEFORE the final "TO BE CONTINUED".
    # Requires: scipy (for Beta CDF) if you want a true posterior tail probability.
    # If you don't have scipy, this will fall back to a simple 0/1 indicator mu>=T*.
    # ------------------------------------------------------------
    
    # Ask user for decision costs
    # C_FP = _prompt_float("Enter cost of FALSE POSITIVE (C_FP) (e.g., 1): ", min_value=0.0)
    # C_FN = _prompt_float("Enter cost of FALSE NEGATIVE (C_FN) (e.g., 1): ", min_value=0.0)
    C_FP = 2.0
    C_FN = 1.0
    
    if C_FP == 0.0 and C_FN == 0.0:
        raise ValueError("At least one of C_FP or C_FN must be > 0.")
    
    # Bayes-optimal cost threshold
    T_star = C_FP / (C_FP + C_FN)
    # print(f"Cost-based threshold T* = C_FP/(C_FP+C_FN) = {T_star:.6f}")
    
    # We will plot P(theta >= T*) over time.
    # If you have a Beta posterior available, compute the tail probability exactly.
    # Otherwise, fallback to a simple proxy: 1{mu >= T*}.
    #
    # NOTE: In this Entropy–KL-only script, we do NOT have a posterior for theta yet.
    # So we do the proxy by default. If you later add Beta–Bernoulli aggregation,
    # replace the proxy with the exact Beta tail probability.
    
    try:
        from scipy.stats import beta as beta_dist  # type: ignore
    
        # Optional: a simple "pseudo-posterior" using confidence as pseudo-counts.
        # This gives a meaningful tail probability even without outcomes.
        #
        # You can adjust kappa_max (strength of confidence) to control how sharp
        # the implied posterior is. Here we ask for it explicitly.
        # kappa_max = _prompt_float(
        #     "Enter kappa_max for pseudo-posterior (e.g., 50 means full confidence ~50 pseudo-obs. Suggestion: use 10): ",
        #     min_value=1e-9
        # )
        kappa_max = 10.0

        # use_entropy_weights = input(
        #     "Use entropy–KL weights w_i in evidence (Method A)? [y/N]: "
        # ).strip().lower() == "y"
        use_entropy_weights = "y"
    
        # Build a Beta posterior per timestamp from (p_i, c_i):
        # kappa_i = kappa_max * c_i
        # alpha = alpha0 + sum_i kappa_i * p_i
        # beta  = beta0  + sum_i kappa_i * (1-p_i)
        alpha0, beta0 = 1.0, 1.0  # uniform prior; change if you want a different prior
        if use_entropy_weights:
            # Method A: evidence contribution uses confidence AND entropy–KL weights
            kappa_mat = kappa_max * np.clip(conf_mat, 0.0, 1.0) * w_series  # shape (T,N)
            # print("Using Method A: kappa_i = kappa_max * c_i * w_i")
        else:
            # Baseline: evidence contribution uses confidence only
            kappa_mat = kappa_max * np.clip(conf_mat, 0.0, 1.0)   # shape (T,N)
            # print("Using baseline: kappa_i = kappa_max * c_i")

        alpha_t = alpha0 + np.sum(kappa_mat * votes_mat, axis=1)
        beta_t  = beta0  + np.sum(kappa_mat * (1.0 - votes_mat), axis=1)
    
        # Tail probability: P(theta >= T*) = 1 - F_beta(T*)
        p_tail = 1.0 - beta_dist.cdf(T_star, a=alpha_t, b=beta_t)
    
        # print("Computed tail probability using Beta pseudo-posterior.")
    except Exception:
        # Fallback: no scipy installed; use a simple indicator based on the point estimate mu.f
        p_tail = (mu_series >= T_star).astype(float)
        print("SciPy not available (or Beta tail failed). Using proxy p_tail = 1{mu >= T*}.")
    
    if plot:
        # Plot P(theta >= T*) over time, with theta in LaTeX
        plt.figure(figsize=(12, 4))
        plt.plot(times, p_tail,color="gray")
        plt.title(r"Tail Probability Over Time: $P(\theta \geq T^*)$")
        plt.xlabel("Time")
        plt.ylabel(r"$P(\theta \geq T^*)$")
        plt.tight_layout()
        if tail_plot_file:
            if os.path.exists(tail_plot_file):
                os.remove(tail_plot_file)
            plt.savefig(tail_plot_file)
        else:
            plt.show()
    
    # Save tail probability CSV
    # tail_csv = "tail_probability_theta_ge_Tstar.csv"
    df_tail = pd.DataFrame({
        "timestamp": times,
        "T_star": np.full_like(p_tail, T_star, dtype=float),
        "P_theta_ge_Tstar": p_tail
    })
    df_tail.to_csv(tail_csv, index=False)
    # print(f"Wrote tail probability time series to: {tail_csv}")
    
    # TO BE CONTINUED to vote aggregation procedure.
    # print("\nDone (plots displayed).")
    # print("TO BE CONTINUED to vote aggregation procedure.\n")

    # Destruct all plots
    plt.close("all")

# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as e:
#         print(f"\nERROR: {e}\n", file=sys.stderr)
#         sys.exit(1)
