# ============================================================
# Experiment 4 – Computational Overhead Comparison 
# Updates added:
#   1) Force single-thread math via env vars (set BEFORE numpy/scipy import).
#   2) Warm-up calls to reduce first-run/cache effects.
#   3) Use MEDIAN wall-clock times (robust to spikes).
#   4) Fit moment/MLE trends in LOG-LOG space (power-law fit).
#   5) KSD extrapolation ANCHORED as quadratic scaling from last measured point.
#
# Notes:
# - Seeds make the DATA deterministic, not the TIMING. These changes stabilize timing.
# - For best results, still run with threads=1 from shell too (see comment near top).
# ============================================================

# ----------------------------
# (1) Force single-thread math
# ----------------------------
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# If you want maximum control, run from terminal with:
#   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python script.py

import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.stats import kendalltau, rankdata
from scipy.optimize import minimize_scalar, bisect


# ============================================================
# RNG helpers
# ============================================================
def make_rng(seed: int):
    return np.random.default_rng(int(seed))


# ============================================================
# Pseudo-observations 
# ============================================================
def to_pseudo_obs(x2d: np.ndarray) -> np.ndarray:
    """
    Convert (n,2) sample to pseudo-observations in (0,1)^2 via mid-ranks:
      U_i = rank(x_i)/(n+1), V_i = rank(y_i)/(n+1)
    """
    x2d = np.asarray(x2d, dtype=float)
    n = x2d.shape[0]
    u = rankdata(x2d[:, 0], method="average") / (n + 1.0)
    v = rankdata(x2d[:, 1], method="average") / (n + 1.0)
    uv = np.column_stack([u, v])
    return np.clip(uv, 1e-12, 1.0 - 1e-12)


# ============================================================
# Copula sampling (Archimedean via K^{-1} inversion)
# ============================================================
# Gumbel pieces
def phi_gumbel(t, theta):
    t = np.clip(t, 1e-12, 1.0 - 1e-12)
    return (-np.log(t)) ** theta

def dphi_gumbel(t, theta):
    t = np.clip(t, 1e-12, 1.0 - 1e-12)
    return -(theta / t) * ((-np.log(t)) ** (theta - 1.0))

def phi_gumbel_inv(y, theta):
    y = np.maximum(y, 0.0)
    return np.exp(-(y ** (1.0 / theta)))

def theta_to_tau_gumbel(theta):
    return (theta - 1.0) / theta

def tau_to_theta_gumbel(tau):
    # IMPORTANT: Gumbel is nonnegative dependence only
    tau_clipped = np.clip(tau, 0.0, 0.999999)
    return 1.0 / (1.0 - tau_clipped)

# Clayton pieces
def phi_clayton(t, theta):
    t = np.clip(t, 1e-12, 1.0 - 1e-12)
    return (t ** (-theta) - 1.0) / theta

def dphi_clayton(t, theta):
    t = np.clip(t, 1e-12, 1.0 - 1e-12)
    return -(t ** (-theta - 1.0))

def phi_clayton_inv(y, theta):
    y = np.maximum(y, 0.0)
    return (1.0 + theta * y) ** (-1.0 / theta)

copula_families = {
    "Gumbel":  {"phi": phi_gumbel,  "dphi": dphi_gumbel,  "phi_inv": phi_gumbel_inv},
    "Clayton": {"phi": phi_clayton, "dphi": dphi_clayton, "phi_inv": phi_clayton_inv}
}

def K_inverse(t, phi, dphi, theta, tol=1e-10):
    """
    Inverts K(x) = x - phi(x)/phi'(x) via bisection.
    """
    def f(x):
        d = dphi(x, theta)
        if not np.isfinite(d) or abs(d) < 1e-300:
            return np.nan
        return x - (phi(x, theta) / d) - t

    a, b = 1e-12, 1.0 - 1e-12
    try:
        fa, fb = f(a), f(b)
        if not (np.isfinite(fa) and np.isfinite(fb)):
            return float(np.clip(t, a, b))
        if fa * fb > 0:
            return float(np.clip(t, a, b))
        return float(bisect(f, a, b, xtol=tol, maxiter=200))
    except Exception:
        return float(np.clip(t, a, b))

def sample_archimedean_copula(copula_name: str, theta: float, n: int, rng) -> np.ndarray:
    funcs = copula_families[copula_name]
    phi, dphi, phi_inv = funcs["phi"], funcs["dphi"], funcs["phi_inv"]

    s_vals = rng.random(n)
    t_vals = rng.random(n)

    w_vals = np.empty(n, dtype=float)
    for i in range(n):
        w_vals[i] = K_inverse(t_vals[i], phi, dphi, theta)

    phi_w = phi(w_vals, theta)
    u = phi_inv(s_vals * phi_w, theta)
    v = phi_inv((1.0 - s_vals) * phi_w, theta)

    uv = np.column_stack([u, v])
    return np.clip(uv, 1e-12, 1.0 - 1e-12)


# ============================================================
# Discrepancies 
# ============================================================
def estimate_moment_cd(sample_raw: np.ndarray, target_tau: float = 0.6) -> float:
    """
    Moment-CD timing:
      - sample -> pseudo-observations
      - tau_emp on ranks
      - theta_q = tau_to_theta_gumbel(tau_emp) with tau clipped to [0,1)
      - tau_q = theta_to_tau_gumbel(theta_q)
      - return |target_tau - tau_q|
    """
    uv = to_pseudo_obs(sample_raw)
    tau_emp, _ = kendalltau(uv[:, 0], uv[:, 1])
    if not np.isfinite(tau_emp):
        return np.nan

    tau_emp = float(np.clip(tau_emp, 0.0, 0.999999))
    theta_q = float(max(tau_to_theta_gumbel(tau_emp), 1.0001))
    tau_q = float(theta_to_tau_gumbel(theta_q))
    return float(abs(target_tau - tau_q))


def estimate_mle_cd(sample_raw: np.ndarray) -> float:
    """
    MLE fit (Clayton) timing:
      - sample -> pseudo-observations
      - fit Clayton theta by MLE
      - return theta_hat (timing-only)
    """
    uv = to_pseudo_obs(sample_raw)
    u, v = uv[:, 0], uv[:, 1]

    def clayton_log_likelihood(theta, u_, v_):
        if theta <= 0:
            return -np.inf
        u_ = np.clip(u_, 1e-12, 1.0 - 1e-12)
        v_ = np.clip(v_, 1e-12, 1.0 - 1e-12)
        inner = u_ ** (-theta) + v_ ** (-theta) - 1.0
        if np.any(inner <= 0):
            return -np.inf
        term1 = np.log(1.0 + theta)
        term2 = -(theta + 1.0) * (np.log(u_) + np.log(v_))
        term3 = -((2.0 * theta + 1.0) / theta) * np.log(inner)
        return np.sum(term1 + term2 + term3)

    objective = lambda th: -clayton_log_likelihood(th, u, v)
    res = minimize_scalar(objective, bounds=(1e-3, 50.0), method="bounded")
    return float(res.x)


def estimate_ksd_imq(sample_raw: np.ndarray, theta_p: float = 3.0, c: float = 1.0, beta: float = -0.5) -> float:
    """
    IMQ-KSD (U-statistic, diagonal excluded) timing:
      - sample -> pseudo-observations
      - compute squared KSD under Clayton(theta_p) score on (0,1)^2
    Vectorized O(n^2) (still heavy), used only for timing.
    """
    X = to_pseudo_obs(sample_raw)
    n, d = X.shape
    if n < 2:
        return 0.0

    u = np.clip(X[:, 0], 1e-12, 1.0 - 1e-12)
    v = np.clip(X[:, 1], 1e-12, 1.0 - 1e-12)

    ut = u ** (-theta_p)
    vt = v ** (-theta_p)
    S = np.maximum(ut + vt - 1.0, 1e-300)

    score = np.column_stack([
        -(theta_p + 1.0) / u + (2.0 * theta_p + 1.0) * u ** (-theta_p - 1.0) / S,
        -(theta_p + 1.0) / v + (2.0 * theta_p + 1.0) * v ** (-theta_p - 1.0) / S,
    ])  # (n,2)

    diffs = X[:, None, :] - X[None, :, :]          # (n,n,2)
    r2 = np.sum(diffs ** 2, axis=2)                # (n,n)
    base = (c * c + r2)                            # (n,n)
    K = base ** beta                               # (n,n)

    grad_x_k = 2.0 * beta * diffs * (base[:, :, None] ** (beta - 1.0))   # (n,n,2)

    tr_hess_x = (
        2.0 * beta * d * (base ** (beta - 1.0))
        + 4.0 * beta * (beta - 1.0) * r2 * (base ** (beta - 2.0))
    )  # (n,n)

    term1 = (score @ score.T) * K                                  # (n,n)
    s_i_dot_gradx = np.einsum("id,ijd->ij", score, grad_x_k)       # (n,n)
    s_j_dot_gradx = np.einsum("jd,ijd->ij", score, grad_x_k)       # (n,n)
    cross_trace = -tr_hess_x

    kappa = term1 + (s_j_dot_gradx - s_i_dot_gradx) + cross_trace

    np.fill_diagonal(kappa, 0.0)
    return float(np.sum(kappa) / (n * (n - 1)))


# ============================================================
# Timing experiment settings
# ============================================================
sample_sizes = np.logspace(2, 3.3, 10).astype(int)  # ~100 to ~2000
n_time_reps = 20                                     # bump reps for stability (was 10)

KSD_CAP = 500                                        # cap for O(n^2) method
BASE_SEED = 777000


# ------------------------------------------------------------
# (2) Warm-up (reduces first-call / cache effects)
# ------------------------------------------------------------
_warm_rng = make_rng(BASE_SEED)
_warm_sample = sample_archimedean_copula("Clayton", theta=3.0, n=200, rng=_warm_rng)
_ = estimate_moment_cd(_warm_sample, target_tau=0.6)
_ = estimate_mle_cd(_warm_sample)
_ = estimate_ksd_imq(_warm_sample, theta_p=3.0)


# ============================================================
# Run timing
# ============================================================
times_moment = {n: [] for n in sample_sizes}
times_mle    = {n: [] for n in sample_sizes}
times_ksd    = {n: [] for n in sample_sizes}

for n_idx, n in enumerate(tqdm(sample_sizes, desc="Timing Experiment")):
    for r in range(n_time_reps):
        rng = make_rng(BASE_SEED + 10_000 * n_idx + r)

        # same base sample each rep for all methods (fair timing comparison)
        sample_raw = sample_archimedean_copula("Clayton", theta=3.0, n=n, rng=rng)

        t0 = time.perf_counter()
        estimate_moment_cd(sample_raw, target_tau=0.6)
        times_moment[n].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        estimate_mle_cd(sample_raw)
        times_mle[n].append(time.perf_counter() - t0)

        if n <= KSD_CAP:
            t0 = time.perf_counter()
            estimate_ksd_imq(sample_raw, theta_p=3.0)
            times_ksd[n].append(time.perf_counter() - t0)


# ------------------------------------------------------------
# (3) Median times (robust)
# ------------------------------------------------------------
med_times_moment = np.array([np.median(times_moment[n]) for n in sample_sizes])
med_times_mle    = np.array([np.median(times_mle[n])    for n in sample_sizes])

ksd_sizes = np.array(sorted([n for n in sample_sizes if len(times_ksd[n]) > 0]))
med_times_ksd = np.array([np.median(times_ksd[n]) for n in ksd_sizes])


# ------------------------------------------------------------
# (4) Log-log fits for Moment/MLE: t(n) = A * n^p
# ------------------------------------------------------------
# Fit log(t) = p*log(n) + log(A)
coef_moment = np.polyfit(np.log(sample_sizes), np.log(med_times_moment), 1)
coef_mle    = np.polyfit(np.log(sample_sizes), np.log(med_times_mle),    1)

p_moment, logA_moment = float(coef_moment[0]), float(coef_moment[1])
p_mle,    logA_mle    = float(coef_mle[0]),    float(coef_mle[1])

A_moment = float(np.exp(logA_moment))
A_mle    = float(np.exp(logA_mle))

def t_moment_fit(n):
    n = np.asarray(n, dtype=float)
    return A_moment * (n ** p_moment)

def t_mle_fit(n):
    n = np.asarray(n, dtype=float)
    return A_mle * (n ** p_mle)


# ------------------------------------------------------------
# (5) KSD extrapolation anchored: t(n) = t(n0) * (n/n0)^2
# ------------------------------------------------------------
if len(ksd_sizes) > 0:
    anchor_n = float(ksd_sizes[-1])
    anchor_t = float(med_times_ksd[-1])

    def t_ksd_fit(n):
        n = np.asarray(n, dtype=float)
        return anchor_t * (n / anchor_n) ** 2
else:
    anchor_n, anchor_t = np.nan, np.nan
    def t_ksd_fit(n):
        return np.full_like(np.asarray(n, dtype=float), np.nan)


# ============================================================
# Equal-time budget throughput (1 second) based on fits
# ============================================================
BUDGET_SEC = 1.0

def n_capacity_powerlaw(A, p, T):
    # solve A*n^p = T  -> n = (T/A)^(1/p)
    if not np.isfinite(A) or not np.isfinite(p) or A <= 0 or p <= 0:
        return np.nan
    return float((T / A) ** (1.0 / p))

def n_capacity_ksd_quadratic(anchor_n, anchor_t, T):
    # solve anchor_t*(n/anchor_n)^2 = T -> n = anchor_n*sqrt(T/anchor_t)
    if not np.isfinite(anchor_n) or not np.isfinite(anchor_t) or anchor_t <= 0:
        return np.nan
    return float(anchor_n * np.sqrt(T / anchor_t))

n_moment_1s = n_capacity_powerlaw(A_moment, p_moment, BUDGET_SEC)
n_mle_1s    = n_capacity_powerlaw(A_mle,    p_mle,    BUDGET_SEC)
n_ksd_1s    = n_capacity_ksd_quadratic(anchor_n, anchor_t, BUDGET_SEC)

print("\n=== Fit summaries (median timings) ===")
print(f"Moment-CD: t(n) ≈ {A_moment:.3e} * n^{p_moment:.3f}")
print(f"MLE-CD:    t(n) ≈ {A_mle:.3e} * n^{p_mle:.3f}")
if np.isfinite(anchor_n):
    print(f"KSD: anchored quadratic from n0={anchor_n:.0f}, t0={anchor_t:.3e} (t ~ n^2)")

print("\n=== Equal-time (1 second) capacity based on fitted trends ===")
print(f"Moment-based CD ≈ n_max = {n_moment_1s:,.0f}")
print(f"MLE-based    CD ≈ n_max = {n_mle_1s:,.0f}")
if np.isfinite(n_ksd_1s):
    print(f"KSD (anchored n^2) ≈ n_max = {n_ksd_1s:,.0f} (anchor uses measured n ≤ {KSD_CAP})")


# ============================================================
# Plot
# ============================================================
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(sample_sizes, med_times_moment, marker='o', linestyle='-', label='Moment-based CD (median)')
ax.plot(sample_sizes, med_times_mle,    marker='s', linestyle='-', label='MLE-based CD (median)')
ax.plot(ksd_sizes,    med_times_ksd,    marker='^', linestyle='-', label=f'KSD (median, n ≤ {KSD_CAP})')

grid = np.linspace(sample_sizes.min(), sample_sizes.max(), 300)
ax.plot(grid, t_moment_fit(grid), linestyle='--', alpha=0.8, label='Moment-based CD (log-log fit)')
ax.plot(grid, t_mle_fit(grid),    linestyle='--', alpha=0.8, label='MLE-based CD (log-log fit)')
if len(ksd_sizes) > 0:
    ax.plot(grid, t_ksd_fit(grid), linestyle='--', alpha=0.8, label='KSD (anchored quadratic extrap.)')

ax.set_xlabel('Sample Size (n)', fontsize=14)
ax.set_ylabel('Wall-clock Time (seconds)', fontsize=14)
ax.set_title('Computational Overhead of Discrepancy Measures', fontsize=16)
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(fontsize=11)
ax.grid(True, which="both", ls="-", alpha=0.4)
plt.tight_layout()
plt.show()

print(
    "\nFigure caption stub:\n"
    "Wall-clock time versus sample size (log–log). All discrepancy measures operate on pseudo-observations "
    "(rank-based empirical copula) for theoretical alignment. Points show median timings over repeated runs. "
    f"KSD is computed using an IMQ U-statistic and is O(n^2); therefore we report measured timings up to n={KSD_CAP} "
    "and show a dashed quadratic extrapolation anchored at the largest measured KSD point. Moment- and MLE-based "
    "Copula Discrepancy scale close to a power law with exponents estimated by log–log regression; dashed lines "
    "show the fitted trends."
)
