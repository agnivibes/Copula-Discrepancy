# ============================================================
# Experiment 2 – Hyperparameter Selection
# ============================================================

import numpy as np
import random
import os
import matplotlib.pyplot as plt

import statsmodels.api as sm
from scipy.stats import kendalltau, rankdata, t, sem
from scipy.optimize import bisect


# ============================================================
# Reproducibility helpers
# ============================================================
def set_seed(seed=123):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

set_seed(123)

def make_rng(seed: int):
    return np.random.default_rng(int(seed))


# ============================================================
# Pseudo-observations (MATCH Code 1: mid-ranks / average ranks)
# ============================================================
def to_pseudo_observations(samples_2d: np.ndarray) -> np.ndarray:
    """
    Convert a 2D sample to pseudo-observations (U,V) via mid-ranks:
      U_i = rank(x1_i)/(n+1), V_i = rank(x2_i)/(n+1),
    using method="average" to handle ties.
    """
    samples_2d = np.asarray(samples_2d, dtype=float)
    n = samples_2d.shape[0]

    u = rankdata(samples_2d[:, 0], method="average") / (n + 1.0)
    v = rankdata(samples_2d[:, 1], method="average") / (n + 1.0)

    uv = np.column_stack([u, v])
    return np.clip(uv, 1e-12, 1.0 - 1e-12)


# ============================================================
# Gumbel copula pieces
# ============================================================
def tau_to_theta_gumbel(tau):
    # Gumbel tau in [0,1); clip for numerical stability
    tau_clipped = np.clip(tau, 0.0, 0.999999)
    return 1.0 / (1.0 - tau_clipped)

def theta_to_tau_gumbel(theta):
    return (theta - 1.0) / theta

def phi_gumbel(t, theta):
    return (-np.log(t)) ** theta

def dphi_gumbel(t, theta):
    return -(theta / t) * ((-np.log(t)) ** (theta - 1))

def phi_gumbel_inv(y, theta):
    return np.exp(-y ** (1.0 / theta))


# ============================================================
# Archimedean sampling (bivariate) using K-inverse method
# K(w) = w - phi(w)/phi'(w)
# ============================================================
def K_inverse(t, phi, dphi, theta, tol=1e-10):
    def f(x):
        d = dphi(x, theta)
        if not np.isfinite(d) or abs(d) < 1e-300:
            return np.nan
        return x - (phi(x, theta) / d) - t

    a, b = 1e-12, 1.0 - 1e-12
    try:
        fa = f(a)
        fb = f(b)
        if not (np.isfinite(fa) and np.isfinite(fb)):
            return np.clip(t, a, b)
        if fa * fb > 0:
            return np.clip(t, a, b)
        return bisect(f, a, b, xtol=tol, maxiter=200)
    except Exception:
        return np.clip(t, a, b)

def sample_gumbel_copula(theta, n=1000, rng=None):
    """
    Sample (U,V) ~ Gumbel(theta) in (0,1)^2 using K^{-1} inversion.
    """
    if rng is None:
        rng = make_rng(123)

    if theta <= 1.0 or not np.isfinite(theta):
        raise ValueError("Gumbel theta must be > 1.")

    s_vals = rng.random(n)
    t_vals = rng.random(n)

    w_vals = np.empty(n, dtype=float)
    for i in range(n):
        w_vals[i] = K_inverse(t_vals[i], phi_gumbel, dphi_gumbel, theta)

    phi_w = phi_gumbel(w_vals, theta)
    u = phi_gumbel_inv(s_vals * phi_w, theta)
    v = phi_gumbel_inv((1.0 - s_vals) * phi_w, theta)

    uv = np.column_stack([u, v])
    return np.clip(uv, 1e-12, 1.0 - 1e-12)


# ============================================================
# Gumbel log-density 
# ============================================================
def log_c_gumbel(theta, u, v):
    if theta <= 1.0 or not np.isfinite(theta):
        return np.full_like(np.asarray(u), -np.inf, dtype=float)

    u = np.clip(np.asarray(u, dtype=float), 1e-12, 1.0 - 1e-12)
    v = np.clip(np.asarray(v, dtype=float), 1e-12, 1.0 - 1e-12)

    x = -np.log(u)
    y = -np.log(v)

    x = np.clip(x, 1e-300, np.inf)
    y = np.clip(y, 1e-300, np.inf)

    a = x ** theta
    b = y ** theta
    S = a + b
    S = np.clip(S, 1e-300, np.inf)

    S_pow = S ** (1.0 / theta)   # S^(1/θ)
    log_C = -S_pow               # log C(u,v)

    term_xy = (theta - 1.0) * (np.log(x) + np.log(y))
    term_uv = -np.log(u) - np.log(v)
    term_S  = (1.0 / theta - 2.0) * np.log(S)
    term_last = np.log((theta - 1.0) + S_pow)

    out = log_C + term_xy + term_uv + term_S + term_last
    return np.where(np.isfinite(out), out, -np.inf)


# ============================================================
# Monte Carlo entropy for Gumbel: H(Cθ) = -E_{Cθ}[log cθ(U,V)]
# ============================================================
def estimate_H_gumbel_mc(theta, m=100_000, seed=12345):
    rng = make_rng(seed)
    uv = sample_gumbel_copula(theta, n=m, rng=rng)
    logs = log_c_gumbel(theta, uv[:, 0], uv[:, 1])
    return -float(np.mean(logs))


# ============================================================
# ESS helpers
# ============================================================
def ess(samples_1d):
    n = len(samples_1d)
    if n < 2:
        return 0.0
    try:
        acf_vals = sm.tsa.acf(samples_1d, nlags=n - 1, fft=True)
        rho_sum = 0.0
        for t_lag in range(1, n):
            rho = acf_vals[t_lag]
            if t_lag > 1 and t_lag % 2 == 1:
                if acf_vals[t_lag - 1] + rho > 0:
                    rho_sum += acf_vals[t_lag - 1] + rho
                else:
                    break
        tau_int = 1.0 + 2.0 * rho_sum
        return n / tau_int if tau_int > 1.0 else float(n)
    except Exception:
        return float(n)

def min_ess(samples_2d):
    return float(np.min([ess(samples_2d[:, i]) for i in range(samples_2d.shape[1])]))


# ============================================================
# Target Distribution: Bimodal Gaussian Mixture
# ============================================================
print("--- Setting up Experiment 2: Hyperparameter Selection ---")

MEAN1 = np.array([-1.5, -1.5])
MEAN2 = np.array([ 1.5,  1.5])

def log_prob_gmm(x):
    """
    Unnormalized log density of symmetric 2-component Gaussian mixture.
    """
    cov_inv = np.eye(2) * 10.0  # covariance 0.1 * I
    log_p1 = -0.5 * (x - MEAN1).T @ cov_inv @ (x - MEAN1)
    log_p2 = -0.5 * (x - MEAN2).T @ cov_inv @ (x - MEAN2)
    max_log = np.maximum(log_p1, log_p2)
    return max_log + np.log(np.exp(log_p1 - max_log) + np.exp(log_p2 - max_log)) - np.log(2.0)

def score_gmm(x):
    """
    Gradient of log mixture density (up to normalization).
    """
    cov_inv = np.eye(2) * 10.0
    e1 = np.exp(-0.5 * (x - MEAN1).T @ cov_inv @ (x - MEAN1))
    e2 = np.exp(-0.5 * (x - MEAN2).T @ cov_inv @ (x - MEAN2))
    grad1 = -cov_inv @ (x - MEAN1)
    grad2 = -cov_inv @ (x - MEAN2)
    denom = (e1 + e2 + 1e-15)
    return (e1 * grad1 + e2 * grad2) / denom


# ============================================================
# SGLD sampler (uses rng)
# ============================================================
def sgld_sampler(score_func, epsilon, n_samples, n_burnin, initial_point, rng):
    samples = np.zeros((n_samples, 2))
    x_t = np.copy(initial_point)

    for i in range(n_samples + n_burnin):
        noise = rng.standard_normal(2)
        x_t = x_t + (epsilon / 2.0) * score_func(x_t) + np.sqrt(epsilon) * noise
        if i >= n_burnin:
            samples[i - n_burnin, :] = x_t
    return samples


# ============================================================
# Metropolis-Hastings for gold-standard sample (uses rng)
# Two-chain approach (one chain started at each mode) to avoid tau ~ 0
# ============================================================
def metropolis_hastings_gmm(n_samples, n_burnin, proposal_std, rng, initial_point):
    samples = np.zeros((n_samples, 2))
    x_t = np.array(initial_point, dtype=float)
    log_p_t = log_prob_gmm(x_t)

    for i in range(n_samples + n_burnin):
        x_prop = x_t + rng.normal(0.0, proposal_std, 2)
        log_p_prop = log_prob_gmm(x_prop)

        if np.log(rng.random()) < (log_p_prop - log_p_t):
            x_t, log_p_t = x_prop, log_p_prop

        if i >= n_burnin:
            samples[i - n_burnin, :] = x_t

    return samples


# ============================================================
# Step 1: Determine Theta_P for the GMM via gold-standard MH
# ============================================================
print("Generating gold-standard sample to find Theta_P...")

gold_rng = make_rng(20250101)

# two MH chains, one per mode (better mixing for dependence estimation)
gold1 = metropolis_hastings_gmm(
    n_samples=25_000,
    n_burnin=5_000,
    proposal_std=0.8,
    rng=gold_rng,
    initial_point=MEAN1
)

gold2 = metropolis_hastings_gmm(
    n_samples=25_000,
    n_burnin=5_000,
    proposal_std=0.8,
    rng=gold_rng,
    initial_point=MEAN2
)

gold_standard_samples = np.vstack([gold1, gold2])

uv_gold = to_pseudo_observations(gold_standard_samples)
tau_emp_gold, _ = kendalltau(uv_gold[:, 0], uv_gold[:, 1])
if not np.isfinite(tau_emp_gold):
    tau_emp_gold = 0.0

# ---- hard guard: Gumbel requires theta > 1 ----
tau_emp_gold = float(np.clip(tau_emp_gold, 1e-6, 0.999999))
THETA_P = float(max(tau_to_theta_gumbel(tau_emp_gold), 1.0001))
TAU_P = float(theta_to_tau_gumbel(THETA_P))

print(f"Estimated Tau_P for GMM: {TAU_P:.6f} (Theta_P={THETA_P:.6f})")


# ============================================================
# Step 2: Shannon precomputation for CED
# ============================================================
MC_M = 100_000
MC_SEED = 4242

print("--- Precomputing target entropy H(C_{Theta_P}) by Monte Carlo ---")
H_target_hat = estimate_H_gumbel_mc(THETA_P, m=MC_M, seed=MC_SEED)

_entropy_cache = {}
def H_gumbel_cached(theta, m=MC_M):
    key = (round(float(theta), 10), int(m))
    if key not in _entropy_cache:
        seed = int(MC_SEED + 1000 + (abs(hash(key)) % 1_000_000))
        _entropy_cache[key] = estimate_H_gumbel_mc(theta, m=m, seed=seed)
    return _entropy_cache[key]


# ============================================================
# Step 3: Experiment loop
# CKL = Forward KL (Fit || Target):
#   CKL = E_{Fit}[ log c_fit(U,V) - log c_target(U,V) ]
# Approx via MC draws from FITTED copula
# ============================================================
epsilons = np.logspace(-5, -1, 10)

n_replications = 100
n_samples_sgld = 2000
n_burnin_sgld = 500

MC_KL = 20_000  # KL MC size (runtime lever)

results_cd  = {eps: [] for eps in epsilons}
results_ess = {eps: [] for eps in epsilons}
results_ckl = {eps: [] for eps in epsilons}
results_ced = {eps: [] for eps in epsilons}

sample_collections = {}

BASE_RUN_SEED = 900000

for eps_idx, eps in enumerate(epsilons):
    print(f"Running for epsilon = {eps:.2e}...")
    rep_last_samples = None

    for r in range(n_replications):
        # deterministic per-replication RNG
        rng = make_rng(BASE_RUN_SEED + 10_000 * eps_idx + r)

        n_half = n_samples_sgld // 2

        init1 = MEAN1 + 0.1 * rng.standard_normal(2)
        init2 = MEAN2 + 0.1 * rng.standard_normal(2)

        samples1 = sgld_sampler(score_gmm, eps, n_half, n_burnin_sgld, init1, rng)
        samples2 = sgld_sampler(score_gmm, eps, n_half, n_burnin_sgld, init2, rng)
        samples = np.vstack([samples1, samples2])

        rep_last_samples = samples

        # pseudo-observations (mid-ranks)
        uv = to_pseudo_observations(samples)

        # theta_q via tau inversion on pseudo-observations
        tau_emp, _ = kendalltau(uv[:, 0], uv[:, 1])
        if not np.isfinite(tau_emp):
            tau_emp = 0.0

        # ---- hard guard: keep within Gumbel domain ----
        tau_emp = float(np.clip(tau_emp, 1e-6, 0.999999))
        theta_q = float(max(tau_to_theta_gumbel(tau_emp), 1.0001))
        tau_q   = float(theta_to_tau_gumbel(theta_q))

        # CD
        results_cd[eps].append(abs(TAU_P - tau_q))

        # ESS (computed on original chain scale)
        results_ess[eps].append(min_ess(samples))

        # CKL = Forward KL Fit||Target:
        # draw MC sample from fitted copula C_{theta_q}
        uv_mc = sample_gumbel_copula(theta_q, n=MC_KL, rng=rng)

        log_fit = log_c_gumbel(theta_q, uv_mc[:, 0], uv_mc[:, 1])
        log_tgt = log_c_gumbel(THETA_P, uv_mc[:, 0], uv_mc[:, 1])

        ckl_val = float(np.mean(log_fit - log_tgt))
        results_ckl[eps].append(ckl_val)

        # CED
        H_hat_q = H_gumbel_cached(theta_q)
        results_ced[eps].append(abs(H_hat_q - H_target_hat))

    # store one representative sample set for plotting
    sample_collections[eps] = rep_last_samples


# ============================================================
# Confidence intervals
# ============================================================
def get_mean_and_ci(data, confidence=0.95):
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan, np.nan, np.nan
    n = x.size
    mean = float(np.mean(x))
    se = float(sem(x))
    moe = se * float(t.ppf((1 + confidence) / 2.0, n - 1))
    return mean, mean - moe, mean + moe

mean_cd,  lower_cd,  upper_cd  = zip(*[get_mean_and_ci(results_cd[eps])  for eps in epsilons])
mean_ess, lower_ess, upper_ess = zip(*[get_mean_and_ci(results_ess[eps]) for eps in epsilons])
mean_ckl, lower_ckl, upper_ckl = zip(*[get_mean_and_ci(results_ckl[eps]) for eps in epsilons])
mean_ced, lower_ced, upper_ced = zip(*[get_mean_and_ci(results_ced[eps]) for eps in epsilons])

# Log10 for CD plotting (protect against CI dipping below 0)
mean_cd_arr  = np.maximum(np.array(mean_cd),  1e-300)
lower_cd_arr = np.maximum(np.array(lower_cd), 1e-300)
upper_cd_arr = np.maximum(np.array(upper_cd), 1e-300)

log_mean_cd  = np.log10(mean_cd_arr)
log_lower_cd = np.log10(lower_cd_arr)
log_upper_cd = np.log10(upper_cd_arr)


# ============================================================
# Print results
# ============================================================
print("\n--- Mean Copula Discrepancy (CD) with 95% Confidence Intervals ---")
for eps, m, lo, up in zip(epsilons, mean_cd, lower_cd, upper_cd):
    print(f"Epsilon: {eps:.2e} | Mean CD: {m:.6e} | 95% CI: [{lo:.6e}, {up:.6e}]")

print("\n--- Mean Effective Sample Size (ESS) with 95% Confidence Intervals ---")
for eps, m, lo, up in zip(epsilons, mean_ess, lower_ess, upper_ess):
    print(f"Epsilon: {eps:.2e} | Mean ESS: {m:.6e} | 95% CI: [{lo:.6e}, {up:.6e}]")

print("\n--- Mean Copula KL (CKL = Forward KL Fit||Target) with 95% Confidence Intervals ---")
for eps, m, lo, up in zip(epsilons, mean_ckl, lower_ckl, upper_ckl):
    print(f"Epsilon: {eps:.2e} | Mean CKL: {m:.6e} | 95% CI: [{lo:.6e}, {up:.6e}]")

print("\n--- Mean Copula Entropy Gap (CED) with 95% Confidence Intervals ---")
for eps, m, lo, up in zip(epsilons, mean_ced, lower_ced, upper_ced):
    print(f"Epsilon: {eps:.2e} | Mean CED: {m:.6e} | 95% CI: [{lo:.6e}, {up:.6e}]")


# ============================================================
# Plots
# ============================================================
print("\n--- Generating Figure 2 ---")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: CD (log10) + ESS
ax1 = axes[0]
ax1.plot(epsilons, log_mean_cd, marker="o", linestyle="-", label="Log10 Mean CD (lower is better)")
ax1.fill_between(epsilons, log_lower_cd, log_upper_cd, alpha=0.2)
ax1.set_xscale("log")
ax1.set_xlabel("SGLD Step-Size (epsilon)")
ax1.set_ylabel("Log10 Mean CD")
ax1.grid(True, which="both", alpha=0.4)

ax2 = ax1.twinx()
ax2.plot(epsilons, mean_ess, marker="s", linestyle="--", label="Mean ESS (higher is better)")
ax2.fill_between(epsilons, lower_ess, upper_ess, alpha=0.2)
ax2.set_ylabel("Mean ESS")

# combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)

# Right: Sample clouds for best eps (by CD vs ESS)
ax3 = axes[1]

best_eps_cd  = epsilons[int(np.argmin(mean_cd))]
best_eps_ess = epsilons[int(np.argmax(mean_ess))]

sample_cd  = sample_collections[best_eps_cd]
sample_ess = sample_collections[best_eps_ess]

ax3.scatter(sample_ess[:, 0], sample_ess[:, 1], s=20, alpha=0.4, marker="^",
            label=f"ESS-selected (eps={best_eps_ess:.1e})")
ax3.scatter(sample_cd[:, 0], sample_cd[:, 1], s=25, alpha=0.7, marker="o",
            label=f"CD-selected (eps={best_eps_cd:.1e})")

ax3.set_title("SGLD Sample Quality")
ax3.set_xlabel("x1")
ax3.set_ylabel("x2")
ax3.legend(fontsize=9)
ax3.set_xlim(-3, 3)
ax3.set_ylim(-3, 3)
ax3.grid(True, alpha=0.4)

plt.tight_layout()
plt.show()

print("\n--- Generating Figure 2b (CKL & CED) ---")
fig2, (bx1, bx2) = plt.subplots(1, 2, figsize=(12, 5))

bx1.plot(epsilons, mean_ckl, marker="o", linestyle="-", label="Mean CKL (Forward KL)")
bx1.fill_between(epsilons, lower_ckl, upper_ckl, alpha=0.2)
bx1.set_xscale("log")
bx1.set_xlabel("SGLD Step-Size (epsilon)")
bx1.set_ylabel("CKL estimate")
bx1.set_title("Copula KL (CKL)")
bx1.grid(True, which="both", alpha=0.4)
bx1.legend(fontsize=9)

bx2.plot(epsilons, mean_ced, marker="s", linestyle="--", label="Mean CED")
bx2.fill_between(epsilons, lower_ced, upper_ced, alpha=0.2)
bx2.set_xscale("log")
bx2.set_xlabel("SGLD Step-Size (epsilon)")
bx2.set_ylabel("CED estimate")
bx2.set_title("Copula Entropy Gap (CED)")
bx2.grid(True, which="both", alpha=0.4)
bx2.legend(fontsize=9)

plt.tight_layout()
plt.show()


