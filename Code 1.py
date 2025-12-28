# ============================================================
# Experiment 1
# ============================================================

import numpy as np
import random
import os
import matplotlib.pyplot as plt

from scipy.stats import t, sem, rankdata
from scipy.optimize import bisect, minimize_scalar
from joblib import Parallel, delayed
from tqdm import tqdm


# ============================================================
# Reproducibility (parallel part uses deterministic per-rep seeds)
# ============================================================
def set_seed(seed=123):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

set_seed(123)

def make_rng(seed: int):
    return np.random.default_rng(int(seed))


# ============================================================
# Pseudo-observations (rank transform)
# ============================================================
def to_pseudo_observations(sample: np.ndarray) -> np.ndarray:
    """
    Convert bivariate sample to pseudo-observations (ranks) in (0,1).
    Uses rank/(n+1) to avoid 0/1 boundaries.
    """
    sample = np.asarray(sample, dtype=float)
    n = sample.shape[0]
    u = sample[:, 0]
    v = sample[:, 1]

    u_hat = rankdata(u, method="average") / (n + 1.0)
    v_hat = rankdata(v, method="average") / (n + 1.0)

    uv_hat = np.column_stack([u_hat, v_hat])
    return np.clip(uv_hat, 1e-12, 1.0 - 1e-12)


# ============================================================
# Copula generator functions (Gumbel & Clayton)
# ============================================================
def phi_gumbel(t, theta):
    return (-np.log(t)) ** theta

def dphi_gumbel(t, theta):
    return -(theta / t) * ((-np.log(t)) ** (theta - 1))

def phi_gumbel_inv(y, theta):
    return np.exp(-y ** (1.0 / theta))

def tau_to_theta_gumbel(tau):
    tau_clipped = np.clip(tau, -0.999, 0.999)
    return 1.0 / (1.0 - tau_clipped)

def theta_to_tau_gumbel(theta):
    return (theta - 1.0) / theta


def phi_clayton(t, theta):
    return (t ** (-theta) - 1.0) / theta

def dphi_clayton(t, theta):
    return -(t ** (-theta - 1.0))

def phi_clayton_inv(y, theta):
    return (1.0 + theta * y) ** (-1.0 / theta)

def tau_to_theta_clayton(tau):
    tau_clipped = np.clip(tau, -0.999, 0.999)
    return 2.0 * tau_clipped / (1.0 - tau_clipped)

def theta_to_tau_clayton(theta):
    return theta / (theta + 2.0)


copula_families = {
    "Gumbel": {
        "phi": phi_gumbel,
        "dphi": dphi_gumbel,
        "phi_inv": phi_gumbel_inv,
        "tau_to_theta": tau_to_theta_gumbel,
        "theta_to_tau": theta_to_tau_gumbel,
    },
    "Clayton": {
        "phi": phi_clayton,
        "dphi": dphi_clayton,
        "phi_inv": phi_clayton_inv,
        "tau_to_theta": tau_to_theta_clayton,
        "theta_to_tau": theta_to_tau_clayton,
    }
}


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
            return t
        if fa * fb > 0:
            return t
        return bisect(f, a, b, xtol=tol, maxiter=200)
    except Exception:
        return t


def sample_archimedean_copula(copula_name, theta, n=1000, rng=None):
    if rng is None:
        rng = make_rng(123)

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
# Gumbel log-density and log-likelihood
# ============================================================
def log_c_gumbel(theta, u, v):
    if theta <= 1.0 or not np.isfinite(theta):
        return np.full_like(u, -np.inf, dtype=float)

    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    v = np.clip(v, 1e-12, 1.0 - 1e-12)

    x = -np.log(u)
    y = -np.log(v)

    x = np.clip(x, 1e-300, np.inf)
    y = np.clip(y, 1e-300, np.inf)

    a = x ** theta
    b = y ** theta
    S = np.clip(a + b, 1e-300, np.inf)

    S_pow = S ** (1.0 / theta)      # S^(1/theta)
    log_C = -S_pow                  # log C(u,v)

    term_xy = (theta - 1.0) * (np.log(x) + np.log(y))
    term_uv = -np.log(u) - np.log(v)
    term_S = (1.0 / theta - 2.0) * np.log(S)
    term_last = np.log((theta - 1.0) + S_pow)

    out = log_C + term_xy + term_uv + term_S + term_last
    return np.where(np.isfinite(out), out, -np.inf)


def gumbel_log_likelihood(theta, u, v):
    return float(np.sum(log_c_gumbel(theta, u, v)))


def estimate_theta_mle_gumbel(sample_uv):
    u = sample_uv[:, 0]
    v = sample_uv[:, 1]

    def objective(th):
        return -gumbel_log_likelihood(th, u, v)

    bounds = (1.0001, 30.0)
    res = minimize_scalar(objective, bounds=bounds, method="bounded", options={"xatol": 1e-6})
    if (not res.success) or (not np.isfinite(res.x)):
        return 1.0001
    return float(res.x)


# ============================================================
# CI helper
# ============================================================
def get_mean_and_ci(data, confidence=0.95):
    data_clean = np.asarray(data, dtype=float)
    data_clean = data_clean[np.isfinite(data_clean)]
    if data_clean.size < 2:
        return np.nan, np.nan, np.nan

    n = data_clean.size
    mean = float(np.mean(data_clean))
    se = float(sem(data_clean))
    moe = se * float(t.ppf((1 + confidence) / 2.0, n - 1))
    return mean, mean - moe, mean + moe


# ============================================================
# Globals / Experiment config
# ============================================================
TARGET_COPULA = "Gumbel"
OFF_TARGET_COPULA = "Clayton"
TARGET_TAU = 0.6

THETA_P_GUMBEL = tau_to_theta_gumbel(TARGET_TAU)
THETA_P_CLAYTON = tau_to_theta_clayton(TARGET_TAU)

# Monte Carlo size used INSIDE each replication to estimate CKL and entropy of fitted model
# (Feel free to raise if you want smoother curves; this is the runtime knob.)
MC_M_REP = 5000

# One-time MC size for estimating H(C_{theta_P}) if you want MC-based CED baseline
MC_M_TARGET = 100000
MC_SEED_BASE = 4242


# ============================================================
# One-time estimate of target entropy H(C_{theta_P}) for CED baseline
# H(C) = -E_{C}[log c_theta(U,V)]
# ============================================================
print("--- Precomputing target entropy H(C_thetaP) by Monte Carlo ---")
UV_target_entropy = sample_archimedean_copula(
    "Gumbel", THETA_P_GUMBEL, n=MC_M_TARGET, rng=make_rng(MC_SEED_BASE)
)
H_target_hat = -float(np.mean(log_c_gumbel(THETA_P_GUMBEL, UV_target_entropy[:, 0], UV_target_entropy[:, 1])))


# ============================================================
# Single replication:
# - Fit theta_hat by MLE on pseudo-observations
# - CD = |tau(theta_P) - tau(theta_hat)|
# - CKL = E_{C_{theta_hat}}[ log c_{theta_hat} - log c_{theta_P} ]  
# - CED = | H(C_{theta_hat}) - H(C_{theta_P}) |
# ============================================================
def run_single_replication_mle(
    target_copula,
    theta_p_target,
    target_tau,
    off_target_copula,
    theta_p_off_target,
    n,
    seed,
):
    rng = make_rng(seed)

    # -------------------------
    # On-target: sample from Gumbel(theta_P)
    # -------------------------
    sample1 = sample_archimedean_copula(target_copula, theta_p_target, n=n, rng=rng)
    sample1_po = to_pseudo_observations(sample1)

    theta_q1_mle = estimate_theta_mle_gumbel(sample1_po)
    tau_q1 = theta_to_tau_gumbel(theta_q1_mle)
    cd1 = abs(target_tau - tau_q1)

    # PAPER-consistent CKL: sample from fitted copula C_{theta_q1_mle}
    rng_ckl1 = make_rng(seed + 111)
    uv_mc1 = sample_archimedean_copula("Gumbel", theta_q1_mle, n=MC_M_REP, rng=rng_ckl1)
    log_q1 = log_c_gumbel(theta_q1_mle, uv_mc1[:, 0], uv_mc1[:, 1])
    log_p1 = log_c_gumbel(theta_p_target, uv_mc1[:, 0], uv_mc1[:, 1])

    ckl1 = float(np.mean(log_q1 - log_p1))
    H_hat_q1 = -float(np.mean(log_q1))
    ced1 = abs(H_hat_q1 - H_target_hat)

    # -------------------------
    # Off-target: sample from Clayton(theta_p_off_target) but FIT Gumbel
    # -------------------------
    sample2 = sample_archimedean_copula(off_target_copula, theta_p_off_target, n=n, rng=rng)
    sample2_po = to_pseudo_observations(sample2)

    theta_q2_mle = estimate_theta_mle_gumbel(sample2_po)
    tau_q2 = theta_to_tau_gumbel(theta_q2_mle)
    cd2 = abs(target_tau - tau_q2)

    rng_ckl2 = make_rng(seed + 222)
    uv_mc2 = sample_archimedean_copula("Gumbel", theta_q2_mle, n=MC_M_REP, rng=rng_ckl2)
    log_q2 = log_c_gumbel(theta_q2_mle, uv_mc2[:, 0], uv_mc2[:, 1])
    log_p2 = log_c_gumbel(theta_p_target, uv_mc2[:, 0], uv_mc2[:, 1])

    ckl2 = float(np.mean(log_q2 - log_p2))
    H_hat_q2 = -float(np.mean(log_q2))
    ced2 = abs(H_hat_q2 - H_target_hat)

    return cd1, cd2, ckl1, ckl2, ced1, ced2


# ============================================================
# Experiment Setup (Matched Tau)
# ============================================================
print("--- Running Experiment 1 (MLE-based with Matched Tau) ---")

sample_sizes = np.logspace(2, 4, 15).astype(int)
n_replications = 100

results_cd_on = {n: [] for n in sample_sizes}
results_cd_off = {n: [] for n in sample_sizes}
results_ckl_on = {n: [] for n in sample_sizes}
results_ckl_off = {n: [] for n in sample_sizes}
results_ced_on = {n: [] for n in sample_sizes}
results_ced_off = {n: [] for n in sample_sizes}

BASE_RUN_SEED = 900000

for n in tqdm(sample_sizes, desc="Running Experiment 1"):
    seeds = [BASE_RUN_SEED + 10_000 * int(n) + r for r in range(n_replications)]

    out = Parallel(n_jobs=-1)(
        delayed(run_single_replication_mle)(
            TARGET_COPULA, THETA_P_GUMBEL, TARGET_TAU,
            OFF_TARGET_COPULA, THETA_P_CLAYTON,
            n, seeds[r]
        )
        for r in range(n_replications)
    )

    cd1, cd2, ckl1, ckl2, ced1, ced2 = zip(*out)

    results_cd_on[n] = list(cd1)
    results_cd_off[n] = list(cd2)
    results_ckl_on[n] = list(ckl1)
    results_ckl_off[n] = list(ckl2)
    results_ced_on[n] = list(ced1)
    results_ced_off[n] = list(ced2)


# ============================================================
# Process results with CIs
# ============================================================
print("\n--- Processing results with Confidence Intervals ---")
mean_on, lower_on, upper_on = zip(*[get_mean_and_ci(results_cd_on[n]) for n in sample_sizes])
mean_off, lower_off, upper_off = zip(*[get_mean_and_ci(results_cd_off[n]) for n in sample_sizes])

mean_ckl_on, lower_ckl_on, upper_ckl_on = zip(*[get_mean_and_ci(results_ckl_on[n]) for n in sample_sizes])
mean_ckl_off, lower_ckl_off, upper_ckl_off = zip(*[get_mean_and_ci(results_ckl_off[n]) for n in sample_sizes])

mean_ced_on, lower_ced_on, upper_ced_on = zip(*[get_mean_and_ci(results_ced_on[n]) for n in sample_sizes])
mean_ced_off, lower_ced_off, upper_ced_off = zip(*[get_mean_and_ci(results_ced_off[n]) for n in sample_sizes])


# ============================================================
# Print numerical CIs
# ============================================================
print("\n--- 95% Confidence Intervals for Copula Discrepancy (CD) ---")
print("Sample Size | On-Target (Gumbel) | Off-Target (Clayton)")
print("-" * 65)
for i, n in enumerate(sample_sizes):
    print(f"{n:10d} | {mean_on[i]:.6f} [{lower_on[i]:.6f}, {upper_on[i]:.6f}] | "
          f"{mean_off[i]:.6f} [{lower_off[i]:.6f}, {upper_off[i]:.6f}]")

print("\n--- 95% Confidence Intervals for Copula KL (CKL) ---")
print("Sample Size | On-Target (Gumbel) | Off-Target (Clayton)")
print("-" * 65)
for i, n in enumerate(sample_sizes):
    print(f"{n:10d} | {mean_ckl_on[i]:.6f} [{lower_ckl_on[i]:.6f}, {upper_ckl_on[i]:.6f}] | "
          f"{mean_ckl_off[i]:.6f} [{lower_ckl_off[i]:.6f}, {upper_ckl_off[i]:.6f}]")

print("\n--- 95% Confidence Intervals for Copula Entropy Gap (CED) ---")
print("Sample Size | On-Target (Gumbel) | Off-Target (Clayton)")
print("-" * 65)
for i, n in enumerate(sample_sizes):
    print(f"{n:10d} | {mean_ced_on[i]:.6f} [{lower_ced_on[i]:.6f}, {upper_ced_on[i]:.6f}] | "
          f"{mean_ced_off[i]:.6f} [{lower_ced_off[i]:.6f}, {upper_ced_off[i]:.6f}]")


# ============================================================
# Figures
# ============================================================
print("\n--- Generating Figure 1 (CD) ---")
fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(sample_sizes, mean_on, marker="o", linestyle="-", label="On-Target Sample (Gumbel)")
ax.fill_between(sample_sizes, lower_on, upper_on, alpha=0.2)

ax.plot(sample_sizes, mean_off, marker="s", linestyle="--", label="Off-Target Sample (Clayton)")
ax.fill_between(sample_sizes, lower_off, upper_off, alpha=0.2)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Sample Size (n)", fontsize=14)
ax.set_ylabel("Copula Discrepancy (CD)", fontsize=14)
ax.set_title("Distinguishing Dependence Structures (CD)", fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, which="both", alpha=0.4)
plt.tight_layout()
plt.show()

print("\n--- Generating Figure 1b (CKL & CED) ---")
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# CKL
ax1.plot(sample_sizes, mean_ckl_on, marker="o", linestyle="-", label="CKL On-Target (Gumbel)")
ax1.fill_between(sample_sizes, lower_ckl_on, upper_ckl_on, alpha=0.2)
ax1.plot(sample_sizes, mean_ckl_off, marker="s", linestyle="--", label="CKL Off-Target (Clayton)")
ax1.fill_between(sample_sizes, lower_ckl_off, upper_ckl_off, alpha=0.2)
ax1.set_xscale("log")
ax1.set_xlabel("Sample Size (n)", fontsize=13)
ax1.set_ylabel("CKL estimate", fontsize=13)
ax1.set_title("Copula KL Divergence (CKL)", fontsize=14)
ax1.grid(True, which="both", alpha=0.4)
ax1.legend(fontsize=10)

# CED
ax2.plot(sample_sizes, mean_ced_on, marker="o", linestyle="-", label="CED On-Target (Gumbel)")
ax2.fill_between(sample_sizes, lower_ced_on, upper_ced_on, alpha=0.2)
ax2.plot(sample_sizes, mean_ced_off, marker="s", linestyle="--", label="CED Off-Target (Clayton)")
ax2.fill_between(sample_sizes, lower_ced_off, upper_ced_off, alpha=0.2)
ax2.set_xscale("log")
ax2.set_xlabel("Sample Size (n)", fontsize=13)
ax2.set_ylabel("CED estimate", fontsize=13)
ax2.set_title("Copula Entropy Gap (CED)", fontsize=14)
ax2.grid(True, which="both", alpha=0.4)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.show()
