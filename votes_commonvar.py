"""
% ============================================================
% The code implements a Normal (Gaussian) change-point model with a mean change and common variance
% Windowed posterior change-location votes (Normal mean+var constant)
% Two segments can have SAME variance estimated per segment.
%
% Input CSV:
%   col1: timestamp "YYYY-MM-DD HH:MM:SS"
%   col2: positive real value
%
% For each sliding window i (length W), computes posterior over tau:
% p_i(tau) ∝ exp( \\tilde{\\ell}_i(tau) )
% where \\tilde{\\ell}_i(tau) is the profile log-likelihood under
% a Normal (Gaussian) model with a change in the mean and common variance over the Window.
%
% Output:
%   votes_matrix(t, k) = k-th vote for time t (k=1..W),
%   corresponding to the k-th window that contains time t.
%   one vote per window cont') 
%   one vote per window containing t (0 padding at edges).
% ============================================================
% Note:
% votes_matrix(t, k) is the k-th vote for global time index t
% The windows that vote for time t are ordered as: win_ids = i_start:i_end;
% the vote index k corresponds to: i = win_ids(k)
% Let: t = global time index; k = vote index (column in votes_matrix); i = window index that cast this vote;
% i_start = max(1, t - W + 1); i_end   = min(num_windows, t); win_ids = i_start:i_end; i = win_ids(k);
% num_windows = T - W + 1;
% win_ids = i_start:i_end; creates a row vector containing a range of integers and
% generates a sequence of numbers starting from i_start and ending at i_end, incrementing by 1 by default.
% Initial time of the voting window [t_i, t_i + W -1] t_i = i = i_start + (k-1)
% Position of t inside that window j = t - i + 1 = t-t_i +1
% Number of points to the left of t in the window {t_i ,t_i +1 ,..., t−1} = n_left = t-t_i 
% Number of points to the right of t in the window {t+1 ,t+2,..., t_i + W −1} = n_right = (t_i + W - 1) - t = W-j
% Calculations of p_tau includes value at t in n_left (i.e., number of data points in the segment [ti,t] = n_left+1 )

"""

import math
import csv
import numpy as np

def window_votes(csv_file, votes_file, conf_file):
    """
    % ----------------------------
    % User parameters
    % ----------------------------
    """
    # csv_file = input("Enter input CSV filename: ").strip()

    W = 20                   # window length
    min_seg = 5              # min samples in EACH segment
    tol_prob = 1e-10         # tolerance for probability checks
    tol_sum  = 1e-8          # tolerance for sum-to-one checks
    eps_var  = 1e-12         # variance floor for numerical stability


    """
    % ------------------------------------------------------------
    % Robust CSV reader: timestamp + numeric value
    % ------------------------------------------------------------
    """
    timestamps = []
    x_list = []

    try:
        with open(csv_file, "r", newline="") as f:
            reader = csv.reader(f, delimiter=",")
        
            # Skip header
            header = next(reader, None)
        
            for row in reader:
                if not row:
                    continue
        
                # Use only columns 1 (timestamp) and 2 (value)
                ts = row[0].strip()
                val_str = row[1].strip()
                if ts == "" and val_str == "":
                    continue
                try:
                    val = float(val_str)
                except Exception as e:
                    raise RuntimeError(f"NaN detected in numeric column — check CSV formatting. Offending row: {row}") from e
                timestamps.append(ts)
                x_list.append(val)
    except FileNotFoundError:
        raise RuntimeError(f"Could not open file: {csv_file}")

    x = np.asarray(x_list, dtype=float)
    T = x.shape[0]

    """
    % Sanity checks
    """
    if np.any(np.isnan(x)):
        raise RuntimeError("NaN detected in numeric column — check CSV formatting.")

    if not np.all(np.isfinite(x)):
        raise RuntimeError("Non-finite values detected in numeric column.")

    """
    % Optional: convert timestamps to datenums (for plotting/handling)
    % dn = datenum(timestamps, "yyyy-mm-dd HH:MM:SS");
    """


    """
    % ----------------------------
    % 2) Prefix sums for fast SSE computation
    % ----------------------------
    % Prefix sums of x and x^2
    """
    S1 = np.concatenate(([0.0], np.cumsum(x)))
    S2 = np.concatenate(([0.0], np.cumsum(x**2)))


    """
    % Helper anonymous functions for segment sums over [a..b]. Note S is the array when function is called
    % seg_sum  = @(S,a,b) (S(b+1) - S(a));
    % seg_mean = @(S,a,b) (seg_sum(S,a,b) / (b-a+1));
    %
    % Sum of squared errors around segment mean, using prefix sums:
    % SSE(a,b) = sum_{t=a..b} x_t^2 - (sum_{t=a..b} x_t)^2 / n
    % SSE(a,b) = sum x^2 - (sum x)^2 / n
    % seg_sse = @(a,b) (seg_sum(S2,a,b) - (seg_sum(S1,a,b)^2)/(b-a+1));
    """
    # NOTE: Octave indices are 1-based (a,b in 1..T, inclusive). We keep that convention here.
    def seg_sum(S, a, b):
        # In Octave with S=[0; cumsum(x)] they used: S(b+1)-S(a)
        # With our S[0]=0 and S[k]=sum_{t=1..k} x_t, this becomes: S[b]-S[a-1]
        return S[b] - S[a-1]

    def seg_mean(S, a, b):
        return seg_sum(S, a, b) / (b - a + 1)

    def seg_sse(a, b):
        n = (b - a + 1)
        s1 = seg_sum(S1, a, b)
        s2 = seg_sum(S2, a, b)
        return s2 - (s1**2) / n


    """
    % ----------------------------
    % 3) Sliding windows: posterior over tau with common variances
    % ----------------------------
    """
    num_windows = T - W + 1
    if num_windows < 1:
        raise RuntimeError(f"Time series length T={T} smaller than W={W}.")

    """
    % p_tau(i, j) = posterior probability for tau at position j within window i
    % where j=1..(W-1) corresponds to global (time) tau = i + j - 1
    """
    p_tau = np.zeros((num_windows, W-1), dtype=float)   # posterior over tau positions per window
    tau_hat = np.zeros((num_windows, 1), dtype=int)

    const_term = -(math.log(2*math.pi) + 1.0) / 2.0     # multiplied by n later

    for i in range(1, num_windows+1):
        """
        % current window: [ti, te]
        """
        ti = i
        te = i + W - 1

        """
        % for each candidate tau, compute SSE1 + SSE2.
        """
        ll = np.full((W-1,), -np.inf, dtype=float)

        for j in range(1, W):
            tau = ti + j - 1

            """
            % number of data points in the segment [ti,tau] 
            % number of data points in the segment (tau,te] 
            % Enforce minimum segment length
            """
            n1 = tau - ti + 1
            n2 = te - (tau + 1) + 1

            if (n1 < min_seg) or (n2 < min_seg):
                ll[j-1] = -np.inf
                continue

            SSE1 = seg_sse(ti, tau)
            SSE2 = seg_sse(tau+1, te)

            """
            % MLE variances per segment: sigma^2 = SSE / n
            % sig1 = max(SSE1 / n1, eps_var);
            % sig2 = max(SSE2 / n2, eps_var);
            % If the model considers a single variance for the window, replace the 2 lines above by the one below
            """
            sig = max((SSE1 + SSE2) / (n1 + n2), eps_var)

            """
            % Profile log-likelihood with separate variances (constants included)
            % ll(j) = n1 * (const_term - 0.5*log(sig1)) + n2 * (const_term - 0.5*log(sig2));
            % If the model considers a single variance for the window, replace the line above by the one below
            """
            ll[j-1] = (n1 + n2) * (const_term - 0.5 * math.log(sig))

        """
        % Convert to posterior: p ∝ exp(ll)
        """
        m = np.max(ll)
        if np.isinf(m):
            """
            % Fallback: all candidates invalid -> uniform posterior
            """
            p = np.ones((W-1,), dtype=float) / (W-1)
        else:
            u = np.exp(ll - m)
            Z = np.sum(u)
            if not (np.isfinite(Z) and (Z > 0)):
                raise RuntimeError(f"Window {i}: invalid normalization constant Z={Z}")
            p = u / Z

        """
        % ---- Validate posterior properties ----
        """
        if not np.all(np.isfinite(p)):
            raise RuntimeError(f"Window {i}: posterior has non-finite entries.")
        if np.any(p < -tol_prob) or np.any(p > 1 + tol_prob):
            bad = int(np.where((p < -tol_prob) | (p > 1 + tol_prob))[0][0]) + 1
            raise RuntimeError(f"Window {i}: posterior out of [0,1]. Example p({bad})={p[bad-1]}")
        s = float(np.sum(p))
        if abs(s - 1.0) > tol_sum:
            raise RuntimeError(f"Window {i}: posterior does not sum to 1 (sum={s}).")

        """
        % tau_hat in this window
        """
        p_tau[i-1, :] = p
        jmax = int(np.argmax(p)) + 1
        tau_hat[i-1, 0] = ti + jmax - 1


    """
    % ----------------------------
    % 4) Build per-time-point votes matrix (T x W) (T rows, W votes each)
    % Each time t belongs to up to W windows.
    % votes_matrix(t, k) is the k-th vote for t (0 if not available).
    %
    % calculate and a "confidence" of the vote based on Fisher information:
    % For each window iwind conf(t,k) = (1/(n_left+1) + 1/n_right)^(-1) / Normalized 
    % conf(t,k) = ((n_left+1) * n_right)/W) / Normalized
    % ( ((n_left+1) * n_right)/W) / (W/2)^2 / W = ( ((n_left+1) * n_right)/W) / (W/4) = ( ((n_left+1) * n_right)) / (W/2)^2 
    % Furthermore, if vote_windows(t,k) = zero, then confidence 
    % = zero.
    % Obs:
    % Initial time of the voting window [t_i, t_i + W -1] t_i = i = i_start + (k-1)
    % Position of t inside that window j = t - i + 1 = t-t_i +1
    % Number of points to the left of t in the window {t_i ,t_i +1 ,..., t−1} = n_left = t-t_i 
    % Number of points to the right of t in the window {t+1 ,t+2,..., t_i + W −1} = n_right = (t_i + W - 1) - t = W-j
    % here we use conf(t,k)^gamma, gamma=2 "tempered confidence weight" (stronger penalty), but gamma=1 can be used.
    %
    % ----------------------------
    """
    votes_matrix = np.zeros((T, W), dtype=float)
    confs_matrix = np.zeros((T, W), dtype=float)
    gamma = 2.0

    for t in range(1, T+1):
        """
        % windows containing t: i in [max(1,t-W+1), min(num_windows,t)]
        % Windows i such that i <= t <= i+W-1  =>  i in [t-W+1, t]
        """
        i_start = max(1, t - W + 1)
        i_end   = min(num_windows, t)

        win_ids = list(range(i_start, i_end+1))
        """
        % number of windows containing t (<= W)
        """
        K = len(win_ids)

        """
        % Collect vote from each window i: posterior mass for tau=t (if allowed)
        % In window i, tau corresponds to j = t - i + 1, and is valid only if j<=W-1
        """
        votes = np.zeros((K,), dtype=float)
        confs = np.zeros((K,), dtype=float)

        for k in range(1, K+1):
            iwin = win_ids[k-1]
            # % local index of tau=t inside window iwin
            j = t - iwin + 1

            n_left = t - (i_start + (k-1))
            n_right = W - j

            if (j >= 1) and (j <= (W-1)):
                votes[k-1] = p_tau[iwin-1, j-1]
                if votes[k-1] > 0:
                    confs[k-1] = ((((n_left+1) * n_right)) / ((W/2.0)**2)) ** gamma
                else:
                    """
                    % confs(k) is 0 if votes(k) = 0
                    """
                    confs[k-1] = 0.0
            else:
                """
                % t is last point of that window -> tau undefined
                % confs(k) is 0 if votes(k) = 0 
                """
                votes[k-1] = 0.0
                confs[k-1] = 0.0

        """
        % Validate votes
        """
        if not np.all(np.isfinite(votes)):
            raise RuntimeError(f"Time t={t}: non-finite vote encountered.")
        if np.any(votes < -tol_prob) or np.any(votes > 1 + tol_prob):
            bad = int(np.where((votes < -tol_prob) | (votes > 1 + tol_prob))[0][0]) + 1
            raise RuntimeError(f"Time t={t}: vote out of [0,1]. Example vote={votes[bad-1]}")

        """
        % Clear row then store left-justified votes
        """
        votes_matrix[t-1, :] = 0.0
        votes_matrix[t-1, 0:K] = votes

        confs_matrix[t-1, :] = 0.0
        confs_matrix[t-1, 0:K] = confs


    """
    % ----------------------------
    % 5) Write output CSV: timestamp, value, then W votes
    %    Write output CSV: timestamp, then W confidences
    % ----------------
    """
    # out_file = f"votes_W{W}_commonvar.csv"
    with open(votes_file, "w", newline="") as f:
        w = csv.writer(f)
        """
        % Header
        """
        header = ["timestamp", "value"] + [f"vote_{k}" for k in range(1, W+1)]
        w.writerow(header)

        """
        % Rows
        """
        for t in range(1, T+1):
            row = [timestamps[t-1], f"{x[t-1]:.15g}"] + [f"{votes_matrix[t-1, k-1]:.15g}" for k in range(1, W+1)]
            w.writerow(row)

    print(f"Done.\nT={T}, W={W}, num_windows={num_windows}\nOutput: {votes_file}")


    """
    % ----------------------------
    % 5.1) Write outpt CSV: timestamp, then confidences
    """
    # out_file = f"confidences_W{W}_commonvar.csv"
    with open(conf_file, "w", newline="") as f:
        w = csv.writer(f)

        """
        % Header
        """
        header = ["timestamp"] + [f"confs_vote_{k}" for k in range(1, W+1)]
        w.writerow(header)

        """
        % Rows
        """
        for t in range(1, T+1):
            row = [timestamps[t-1]] + [f"{confs_matrix[t-1, k-1]:.15g}" for k in range(1, W+1)]
            w.writerow(row)

    print(f"Done.\nT={T}, W={W}, num_windows={num_windows}\nOutput: {conf_file}")


    """
    % ----------------------------
    %
    %
    %=====================================
    % The code below is only for the original way to select change points, using tau_hat (max posterior)
    % It is only used for comparisson, if needed.
    %=====================================
    """


    """
    % ------------------------------------------------------------
    % 7) Report per-time selected change-points and window posteriors
    % ------------------------------------------------------------
    %
    % For each global time t, list all windows i such that:
    %   tau_hat(i) == t
    % and print the corresponding posterior probability
    %   p_i(tau = t | D_i)
    %
    % Output format (CSV):
    % t, timestamp, num_windows,
    %   win_i_1, post_i_1, win_i_2, post_i_2, ...
    % ------------------------------------------------------------
    """
    # cp_file = f"chosen_change_points_W{W}_commonvar.csv"
    # with open(cp_file, "w", newline="") as f:
    #     w = csv.writer(f)

    #     """
    #     % Header
    #     """
    #     header = ["t", "timestamp", "num_windows"]
    #     for k in range(1, W+1):
    #         header += [f"win_{k}", f"post_{k}"]
    #     w.writerow(header)

    #     for t in range(1, T+1):
    #         """
    #         % Find windows that selected t as MAP change-point
    #         """
    #         win_sel = np.where(tau_hat[:, 0] == t)[0] + 1  # back to 1-based window ids
    #         K = int(win_sel.size)

    #         """
    #         % Print basic info
    #         """
    #         row = [str(t), timestamps[t-1], str(K)]

    #         """
    #         % Print window index and posterior for each window
    #         """
    #         for k in range(1, K+1):
    #             iwin = int(win_sel[k-1])
    #             # % local tau index inside window
    #             j = t - iwin + 1
    #             if (j >= 1) and (j <= (W-1)):
    #                 post = float(p_tau[iwin-1, j-1])
    #             else:
    #                 post = 0.0  # should not happen, but safe
    #             row += [str(iwin), f"{post:.15g}"]

    #         """
    #         % Pad remaining columns (for rectangular CSV)
    #         """
    #         for k in range(K+1, W+1):
    #             row += ["", ""]

    #         w.writerow(row)

    # print(f"Chosen change-point report written to: {cp_file}")


    """
    %=====================================
    %
    % ------------------------------------------------------------
    % (SPACE RESERVED) Aggregation across windows
    % ------------------------------------------------------------
    % Later: incorporate confidence weights c_i(tau) and define a fusion rule.
    % Example placeholder:
    %   score(t) = AGGREGATE( votes_matrix(t,:) , confidences(t,:) )
    % ------------------------------------------------------------
    """
