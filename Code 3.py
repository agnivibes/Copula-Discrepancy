# ============================================================
# Experiment 3: Detecting Tail Dependence Mismatch
# ============================================================

import numpy as np
import random
import os
import matplotlib.pyplot as plt

from scipy.stats import kendalltau, t, sem, rankdata
from scipy.optimize import bisect, minimize_scalar
import statsmodels.api as sm
from tqdm import tqdm


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
# 1) Copula definitions (Gumbel & Clayton)
# ============================================================
def phi_gumbel(t, theta):
    return (-np.log(t)) ** theta

def dphi_gumbel(t, theta):
    return -(theta / t) * ((-np.log(t)) ** (theta - 1))

def phi_gumbel_inv(y, theta):
    return np.exp(-y ** (1.0 / theta))

def tau_to_theta_gumbel(tau):
    tau_clipped = np.clip(tau, 0.0, 0.999999)
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
    tau_clipped = np.clip(tau, -0.999999, 0.999999)
    return 2.0 * tau_clipped / (1.0 - tau_clipped)

def theta_to_tau_clayton(theta):
    return theta / (theta + 2.0)


copula_families = {
    "Gumbel": {
        "phi": phi_gumbel, "dphi": dphi_gumbel, "phi_inv": phi_gumbel_inv,
        "tau_to_theta": tau_to_theta_gumbel, "theta_to_tau": theta_to_tau_gumbel
    },
    "Clayton": {
        "phi": phi_clayton, "dphi": dphi_clayton, "phi_inv": phi_clayton_inv,
        "tau_to_theta": tau_to_theta_clayton, "theta_to_tau": theta_to_tau_clayton
    }
}


# ============================================================
# 2) Archimedean sampling via K^{-1} inversion (rng-based)
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

def sample_archimedean_copula(copula_name, theta, n=1000, rng=None):
    """
    Returns uv array of shape (n,2), values in (0,1)^2.
    """
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
# 3) Pseudo-observations
# ============================================================
def to_pseudo_observations_uv(uv: np.ndarray) -> np.ndarray:
    """
    Convert uv in (0,1)^2 to pseudo-observations using mid-ranks.
    """
    uv = np.asarray(uv, dtype=float)
    n = uv.shape[0]
    u_po = rankdata(uv[:, 0], method="average") / (n + 1.0)
    v_po = rankdata(uv[:, 1], method="average") / (n + 1.0)
    return np.clip(np.column_stack([u_po, v_po]), 1e-12, 1.0 - 1e-12)


# ============================================================
# 4) MLE for Clayton (used) + Clayton log-density
# ============================================================
def clayton_log_likelihood(theta, u, v):
    if theta <= 0 or not np.isfinite(theta):
        return -np.inf

    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    v = np.clip(v, 1e-12, 1.0 - 1e-12)

    common = u**(-theta) + v**(-theta) - 1.0
    if np.any(common <= 0):
        return -np.inf

    term1 = np.log(1.0 + theta)
    term2 = -(theta + 1.0) * (np.log(u) + np.log(v))
    term3 = -((2.0 * theta + 1.0) / theta) * np.log(common)

    return float(np.sum(term1 + term2 + term3))

def estimate_theta_mle_clayton(u, v):
    objective = lambda th: -clayton_log_likelihood(th, u, v)
    bounds = (1e-3, 50.0)
    res = minimize_scalar(objective, bounds=bounds, method="bounded", options={"xatol": 1e-6})
    if (not res.success) or (not np.isfinite(res.x)):
        return 1e-3
    return float(res.x)

def log_c_clayton(theta, u, v):
    """
    Pointwise log-density for Clayton:
      c(u,v) = (1+θ) (uv)^-(1+θ) (u^-θ + v^-θ - 1)^(-2 - 1/θ)
    """
    if theta <= 0 or not np.isfinite(theta):
        return np.full_like(np.asarray(u), -np.inf, dtype=float)

    u = np.clip(np.asarray(u, dtype=float), 1e-12, 1.0 - 1e-12)
    v = np.clip(np.asarray(v, dtype=float), 1e-12, 1.0 - 1e-12)

    common = u**(-theta) + v**(-theta) - 1.0
    common = np.clip(common, 1e-15, np.inf)

    term1 = np.log(1.0 + theta)
    term2 = -(1.0 + theta) * (np.log(u) + np.log(v))
    term3 = -(2.0 + 1.0 / theta) * np.log(common)

    return term1 + term2 + term3


# ============================================================
# 5) KSD (Clayton target): score + IMQ-KSD
# ============================================================
def score_clayton_copula(X, theta):
    u, v = X[:, 0], X[:, 1]
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    v = np.clip(v, 1e-12, 1.0 - 1e-12)

    common = u**(-theta) + v**(-theta) - 1.0
    common = np.clip(common, 1e-15, np.inf)

    du = -(1.0 + theta) / u + (2.0 * theta + 1.0) * u**(-theta - 1.0) / common
    dv = -(1.0 + theta) / v + (2.0 * theta + 1.0) * v**(-theta - 1.0) / common
    return np.column_stack([du, dv])

def ksd_imq(X, score_func, c=1.0, beta=-0.5):
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if n < 2:
        return 0.0

    S = score_func(X)                         # (n, d)
    diffs = X[:, None, :] - X[None, :, :]     # (n, n, d)
    r2 = np.sum(diffs**2, axis=2)             # (n, n)
    base = (c**2 + r2)                        # (n, n)
    K = base**beta                            # (n, n)

    grad_x_k = 2 * beta * diffs * base[:, :, None]**(beta - 1)
    tr_hess_x = 2 * beta * d * base**(beta - 1) + 4 * beta * (beta - 1) * r2 * base**(beta - 2)
    cross_trace = -tr_hess_x

    term1 = (S @ S.T) * K
    s_i_dot_gradx = np.einsum("id,ijd->ij", S, grad_x_k)
    s_j_dot_gradx = np.einsum("jd,ijd->ij", S, grad_x_k)
    term23 = (s_j_dot_gradx - s_i_dot_gradx)

    kappa = term1 + term23 + cross_trace
    np.fill_diagonal(kappa, 0.0)

    return float(np.sum(kappa) / (n * (n - 1)))


# ============================================================
# 6) Shannon entropy MC for Clayton (rng-based)
# ============================================================
def estimate_H_clayton_mc(theta, m=100_000, seed=7777):
    rng = make_rng(seed)
    uv = sample_archimedean_copula("Clayton", theta, n=m, rng=rng)
    logs = log_c_clayton(theta, uv[:, 0], uv[:, 1])
    return -float(np.mean(logs))

_entropy_cache = {}
def H_clayton_cached(theta, m=100_000, seed_base=202402):
    key = (round(float(theta), 10), int(m))
    if key not in _entropy_cache:
        seed = int(seed_base + 2000 + (abs(hash(key)) % 1_000_000))
        _entropy_cache[key] = estimate_H_clayton_mc(theta, m=m, seed=seed)
    return _entropy_cache[key]


# ============================================================
# 7) CI helper
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


# ============================================================
# 8) One replication (mismatch Gumbel, target Clayton)
# ============================================================
def run_single_replication_exp3(
    mismatch_copula, theta_mismatch,
    theta_p_target, target_tau,
    n, seed, mc_kl=20_000
):
    rng = make_rng(seed)

    # 1) sample from mismatch Q (Gumbel) in (0,1)^2
    uv_raw = sample_archimedean_copula(mismatch_copula, theta_mismatch, n=n, rng=rng)

    # 2) pseudo-observations (mid-ranks)
    uv = to_pseudo_observations_uv(uv_raw)
    u, v = uv[:, 0], uv[:, 1]

    # 3) naive tau discrepancy (on ranks)
    tau_emp, _ = kendalltau(u, v)
    if not np.isfinite(tau_emp):
        tau_emp = 0.0
    naive = abs(target_tau - tau_emp)

    # 4) CD: fit Clayton theta via MLE on ranks; compare implied tau
    theta_q_mle = estimate_theta_mle_clayton(u, v)
    tau_q = theta_to_tau_clayton(theta_q_mle)
    cd = abs(target_tau - tau_q)

    # 5) KSD against true target Clayton(theta_p_target), on ranks
    X = np.column_stack([u, v])
    score_func = lambda x: score_clayton_copula(x, theta=theta_p_target)
    ksd_val = ksd_imq(X, score_func)

    # 6) CKL: Forward KL (Fit || Target)
    # CKL = E_{C_theta_q}[ log c_theta_q - log c_theta_p ] estimated by MC from fitted Clayton
    uv_fit = sample_archimedean_copula("Clayton", theta_q_mle, n=mc_kl, rng=rng)
    log_fit = log_c_clayton(theta_q_mle, uv_fit[:, 0], uv_fit[:, 1])
    log_tgt = log_c_clayton(theta_p_target, uv_fit[:, 0], uv_fit[:, 1])
    ckl = float(np.mean(log_fit - log_tgt))

    return naive, cd, ksd_val, theta_q_mle, ckl


# ============================================================
# 9) Run Experiment 3
# ============================================================
print("--- Running Experiment 3: Detecting Tail Dependence Mismatch ---")

TARGET_COPULA = "Clayton"
MISMATCH_COPULA = "Gumbel"
TARGET_TAU = 0.6

THETA_P = tau_to_theta_clayton(TARGET_TAU)
THETA_MISMATCH = tau_to_theta_gumbel(TARGET_TAU)

print(f"Target P: {TARGET_COPULA} with tau={TARGET_TAU:.2f} (theta={THETA_P:.2f})")
print(f"Mismatch Q: {MISMATCH_COPULA} with tau={TARGET_TAU:.2f} (theta={THETA_MISMATCH:.2f})")

# Precompute target entropy for CED
MC_M = 100_000
MC_SEED = 202402

print("--- Precomputing target entropy H(C_{Theta_P}) [Clayton] by Monte Carlo ---")
H_target_hat = estimate_H_clayton_mc(THETA_P, m=MC_M, seed=MC_SEED)

# Experiment parameters
sample_sizes = np.logspace(2, 4, 15).astype(int)
n_replications = 100
MC_KL = 20_000

results = {n: {"naive": [], "cd": [], "ksd": []} for n in sample_sizes}
results_ckl = {n: [] for n in sample_sizes}
results_ced = {n: [] for n in sample_sizes}

BASE_RUN_SEED = 910000

for n in tqdm(sample_sizes, desc="Experiment 3 over n"):
    for r in range(n_replications):
        seed = BASE_RUN_SEED + 10_000 * int(n) + r

        naive_res, cd_res, ksd_res, theta_q_mle, ckl_res = run_single_replication_exp3(
            mismatch_copula=MISMATCH_COPULA,
            theta_mismatch=THETA_MISMATCH,
            theta_p_target=THETA_P,
            target_tau=TARGET_TAU,
            n=n,
            seed=seed,
            mc_kl=MC_KL
        )

        results[n]["naive"].append(naive_res)
        results[n]["cd"].append(cd_res)
        results[n]["ksd"].append(ksd_res)
        results_ckl[n].append(ckl_res)

        # CED
        H_hat_q = H_clayton_cached(theta_q_mle, m=MC_M, seed_base=MC_SEED)
        results_ced[n].append(abs(H_hat_q - H_target_hat))


# ============================================================
# 10) Process results
# ============================================================
processed_results = {}
for n in sample_sizes:
    processed_results[n] = {
        "cd": get_mean_and_ci(results[n]["cd"]),
        "naive": get_mean_and_ci(results[n]["naive"]),
        "ksd": get_mean_and_ci(results[n]["ksd"]),
        "ckl": get_mean_and_ci(results_ckl[n]),
        "ced": get_mean_and_ci(results_ced[n]),
    }

print("\n--- Results Table for Experiment 3 ---")
print("-" * 110)
print(f"{'Sample Size':<15} | {'Metric':<30} | {'Mean':<15} | {'Lower CI':<15} | {'Upper CI':<15}")
print("-" * 110)

for n in sample_sizes:
    for key, label in [
        ("cd", "Copula Discrepancy (CD)"),
        ("naive", "Naive Tau Discrepancy"),
        ("ksd", "Kernel Stein Discrepancy"),
        ("ckl", "Copula KL (CKL, Fit||Target)"),
        ("ced", "Copula Entropy Gap (CED)"),
    ]:
        mean, lower, upper = processed_results[n][key]
        print(f"{n:<15} | {label:<30} | {mean:<15.4e} | {lower:<15.4e} | {upper:<15.4e}")
    print("-" * 110)


# ============================================================
# 11) Figures
# ============================================================
print("\n--- Generating Figure 3 (CD, Naive, KSD vs n) ---")
fig, ax1 = plt.subplots(figsize=(10, 7))

mean_cd, lower_cd, upper_cd = zip(*[processed_results[n]["cd"] for n in sample_sizes])
p1, = ax1.plot(sample_sizes, mean_cd, marker="o", linestyle="-", label="CD (MLE-based)")
ax1.fill_between(sample_sizes, lower_cd, upper_cd, alpha=0.2)

mean_naive, lower_naive, upper_naive = zip(*[processed_results[n]["naive"] for n in sample_sizes])
p2, = ax1.plot(sample_sizes, mean_naive, marker="s", linestyle="--", label="Naive Tau Discrepancy")
ax1.fill_between(sample_sizes, lower_naive, upper_naive, alpha=0.2)

ax1.set_xlabel("Sample Size (n)", fontsize=14)
ax1.set_ylabel("Discrepancy (CD & Naive)", fontsize=14)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.grid(True, which="both", alpha=0.3)

ax2 = ax1.twinx()
mean_ksd, lower_ksd, upper_ksd = zip(*[processed_results[n]["ksd"] for n in sample_sizes])
p3, = ax2.plot(sample_sizes, mean_ksd, marker="^", linestyle=":", label="KSD")
ax2.fill_between(sample_sizes, lower_ksd, upper_ksd, alpha=0.2)
ax2.set_yscale("log")
ax2.set_ylabel("KSD", fontsize=14)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
fig.legend(lines1 + lines2, labels1 + labels2, loc="upper center", ncol=3)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()

print("\n--- Generating Figure 3b (CKL & CED vs n) ---")
fig2, (bx1, bx2) = plt.subplots(1, 2, figsize=(12, 5))

mean_ckl, lower_ckl, upper_ckl = zip(*[processed_results[n]["ckl"] for n in sample_sizes])
bx1.plot(sample_sizes, mean_ckl, marker="o", linestyle="-", label="Mean CKL (Fit||Target)")
bx1.fill_between(sample_sizes, lower_ckl, upper_ckl, alpha=0.2)
bx1.set_xscale("log")
bx1.set_xlabel("Sample Size (n)")
bx1.set_ylabel("CKL estimate")
bx1.set_title("Copula KL (CKL)")
bx1.grid(True, which="both", alpha=0.3)
bx1.legend(fontsize=9)

mean_ced, lower_ced, upper_ced = zip(*[processed_results[n]["ced"] for n in sample_sizes])
bx2.plot(sample_sizes, mean_ced, marker="s", linestyle="--", label="Mean CED")
bx2.fill_between(sample_sizes, lower_ced, upper_ced, alpha=0.2)
bx2.set_xscale("log")
bx2.set_xlabel("Sample Size (n)")
bx2.set_ylabel("CED estimate")
bx2.set_title("Copula Entropy Gap (CED)")
bx2.grid(True, which="both", alpha=0.3)
bx2.legend(fontsize=9)

plt.tight_layout()
plt.show()

