# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 00:03:22 2025

@author: fiagb
"""

#==============================================================================
#================== IMPORTING THE NECESSARY LIBRARIES =========================
#==============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv, solve
import math
from numpy.linalg import cholesky
from scipy.stats import norm
from caas_jupyter_tools import display_dataframe_to_user



## Set working directory
directory = '/Users/fiagb/OneDrive/Desktop/PhD LIBRARY/FALL 2025/Asymptotic/HW1'
os.chdir(directory)
print(os.getcwd())



#==============================================================================
#=================== PROBLEM 1 (Textbook Problem 3a, 3b) ======================
#==============================================================================

# Proof demonstration via simulation and plots
rng = np.random.default_rng(42)

# Parameters
r = 2.0          # larger moment
r_prime = 1.2    # smaller moment 
p = r_prime / r

# ----------------- Part (a): E|X|^{r'} <= (E|X|^r)^{r'/r} -----------------------
# We'll sample from a heavy-tailed-but-r-moment-finite distribution: Student t with df=6 (finite moments up to order 6-1=5)
# Implemented by sampling Z ~ t_nu; equivalently, Normal / sqrt(ChiSq/nu)
nu = 6
N_vals = np.logspace(2, 5, 12, dtype=int)  # sample sizes from 1e2 to 1e5
emp_E_r = []
emp_E_rp = []
emp_bound = []

for N in N_vals:
    Z = rng.standard_normal(N)
    V = rng.chisquare(nu, size=N) / nu
    X = Z / np.sqrt(V)  # Student-t_nu
    mr = np.mean(np.abs(X)**r)
    mrp = np.mean(np.abs(X)**r_prime)
    emp_E_r.append(mr)
    emp_E_rp.append(mrp)
    emp_bound.append(mr**p)

# Creating a summary DataFrame (last row highlights large-sample behavior)
summary_a = pd.DataFrame({
    "N": N_vals,
    f"mean|X|^{r_prime:.2f}": emp_E_rp,
    f"(mean|X|^{r:.2f})^{p:.3f} (bound)": emp_bound
})

summary_a.round(6)

# Plot: empirical E|X|^{r'} vs the bound (E|X|^r)^{r'/r} across sample sizes
plt.figure()
plt.plot(N_vals, emp_E_rp, marker='o', label=f"Empirical E|X|^{r_prime}")
plt.plot(N_vals, emp_bound, marker='s', linestyle='--', label=f"Bound: (E|X|^{r})^{r_prime/r}")
plt.xscale('log')
plt.xlabel("Sample size N (log scale)")
plt.ylabel("Value")
plt.title("Hölder-type inequality demonstration")
plt.legend()
plt.tight_layout()
plt.show()




# ----------------- Part (b): If X_n -> X in L^r, then X_n -> X in L^{r'} -------------------
# Constructing X and X_n = X + noise_n, with noise variance decreasing so that E|X_n - X|^r -> 0.
M = 25  # number of steps in the sequence
N = 20000  # samples per step for stable estimates

# Base signal X
X = rng.standard_normal(N)

sigmas = np.geomspace(1.0, 1e-3, M)  # shrinking noise scales
Er = []
Erp = []
Bound = []

for s in sigmas:
    noise = s * rng.standard_normal(N)
    Xn = X + noise
    diff = np.abs(Xn - X)
    mr = np.mean(diff**r)
    mrp = np.mean(diff**r_prime)
    Er.append(mr)
    Erp.append(mrp)
    Bound.append(mr**p)

summary_b = pd.DataFrame({
    "step": np.arange(1, M+1),
    "sigma_noise": sigmas,
    f"E|Xn-X|^{r:.2f}": Er,
    f"E|Xn-X|^{r_prime:.2f}": Erp,
    f"(E|Xn-X|^{r:.2f})^{p:.3f} (bound)": Bound
})

summary_b.round(8)

# Plot: both E|Xn-X|^r and E|Xn-X|^{r'} as noise shrinks (step index increases)
plt.figure()
plt.plot(range(1, M+1), Er, marker='o', label=f"E|Xn - X|^{r}")
plt.plot(range(1, M+1), Erp, marker='s', linestyle='--', label=f"E|Xn - X|^{r_prime}")
plt.xlabel("Sequence step (noise scale decreases)")
plt.ylabel("Moment value")
plt.title("Convergence of moments (L^r ⇒ L^{r'})")
plt.legend()
plt.tight_layout()
plt.show()

# Plot: comparing E|Xn-X|^{r'} to the bound (E|Xn-X|^{r})^{r'/r}
plt.figure()
plt.plot(range(1, M+1), Erp, marker='o', label=f"E|Xn - X|^{r_prime}")
plt.plot(range(1, M+1), Bound, marker='^', linestyle=':', label="Bound via Hölder")
plt.xlabel("Sequence step (noise scale decreases)")
plt.ylabel("Value")
plt.title("Hölder bound along the sequence")
plt.legend()
plt.tight_layout()
plt.show()

last_row = summary_b.iloc[-1]
last_row




#==============================================================================
#==================== PROBLEM 2: Modes of convergence =========================
#==============================================================================

# Empirical demonstration of SIS sure screening in an NP-dimensional regime
# Using lighter settings to ensure smooth execution in this environment.

rng = np.random.default_rng(7)

def sis_retain_indices(X, y, d):
    scores = np.abs(X.T @ y)  # proportional to marginal correlations
    top = np.argpartition(scores, -d)[-d:]
    return np.sort(top)

def trial_once(n, s=5, snr=2.0):
    p = int(np.round(np.exp(n**0.4)))
    support = rng.choice(p, size=s, replace=False)
    beta = np.zeros(p)
    beta[support] = rng.choice([-1.0, 1.0], size=s) * 1.0  # moderate effects
    var_signal = np.sum(beta**2)
    sigma2 = var_signal / snr
    X = rng.standard_normal((n, p))
    y = X @ beta + rng.normal(0.0, np.sqrt(sigma2), size=n)
    d = int(np.floor(n / np.log(n)))
    idx = sis_retain_indices(X, y, d)
    success = set(support).issubset(set(idx))
    return p, d, success

def estimate_prob(n, M=12, s=5, snr=2.0):
    succ = 0
    p_last, d_last = None, None
    for _ in range(M):
        p, d, ok = trial_once(n, s=s, snr=snr)
        succ += int(ok)
        p_last, d_last = p, d
    return p_last, d_last, succ / M

n_values = [120, 160, 200, 260, 300]
records = []
for n in n_values:
    p, d, prob = estimate_prob(n, M=12, s=5, snr=2.0)
    records.append({"n": n, "p (≈exp(n^0.4))": p, "SIS retain d=floor(n/log n)": d, "Pr(S* ⊆ M^) (empirical)": prob})

df = pd.DataFrame(records)
display_dataframe_to_user("SIS sure-screening — probability by n", df)

plt.figure()
plt.plot(df["n"], df["Pr(S* ⊆ M^) (empirical)"], marker='o')
plt.ylim(0, 1.05)
plt.xlabel("n")
plt.ylabel("Empirical probability  P(S* ⊆ M^)")
plt.title("SIS sure screening in NP-dimensional regime  p = exp(n^0.4), s=5")
plt.tight_layout()
plt.show()

df



#==============================================================================
#=============== PROBLEM 3: Separation in logistic regression =================
#==============================================================================

rng = np.random.default_rng(0)

# Generating linearly separable data
n = 200
x1_pos = rng.normal( 1.2, 0.3, n//2)
x1_neg = rng.normal(-1.2, 0.3, n//2)
x2_pos = rng.normal(0.0, 0.7, n//2)
x2_neg = rng.normal(0.0, 0.7, n//2)
X1 = np.concatenate([x1_pos, x1_neg])
X2 = np.concatenate([x2_pos, x2_neg])
y  = np.concatenate([np.ones(n//2), np.zeros(n//2)])
X = np.column_stack([np.ones(n), X1, X2])

def sigmoid(z): return 1.0/(1.0 + np.exp(-z))

def mle_newton(X, y, max_iter=35):
    beta = np.zeros(X.shape[1]); path=[beta.copy()]; norms=[]
    for t in range(max_iter):
        mu = sigmoid(X@beta)
        W = mu*(1-mu)
        S = X.T * W @ X
        g = X.T @ (y - mu)
        try:
            step = solve(S + 1e-8*np.eye(S.shape[0]), g)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(S) @ g
        beta += step
        path.append(beta.copy()); norms.append(np.linalg.norm(beta))
    return np.array(path), np.array(norms)

mle_path, mle_norms = mle_newton(X, y, max_iter=40)

def ridge_logit(X, y, lam, tol=1e-8, max_iter=200):
    p = X.shape[1]
    beta = np.zeros(p); I = np.eye(p)
    for t in range(max_iter):
        mu = sigmoid(X@beta)
        W = mu*(1-mu)
        S = X.T * W @ X + lam*I
        g = X.T @ (y - mu) - lam*beta
        delta = solve(S, g)
        beta_new = beta + delta
        if np.linalg.norm(delta) < tol*(1 + np.linalg.norm(beta)):
            beta = beta_new; break
        beta = beta_new
    return beta, t+1

ridge_lams = np.logspace(-3, 2, 12)
ridge_betas = []; ridge_iters=[]
for lam in ridge_lams:
    b, it = ridge_logit(X, y, lam); ridge_betas.append(b); ridge_iters.append(it)
ridge_betas = np.array(ridge_betas)
ridge_df = pd.DataFrame(ridge_betas, columns=["intercept","x1","x2"])
ridge_df.insert(0,"lambda",ridge_lams); ridge_df["iters"]=ridge_iters
ridge_df.round(6)

def firth_logit(X, y, tol=1e-8, max_iter=200):
    p = X.shape[1]; beta = np.zeros(p)
    for t in range(max_iter):
        mu = sigmoid(X@beta)
        W = mu*(1-mu)
        S = X.T * W @ X
        Sinv = inv(S + 1e-10*np.eye(p))
        WX = (W**0.5)[:,None] * X
        H = WX @ Sinv @ WX.T
        h = np.clip(np.diag(H), 0.0, 1.0)
        Ustar = X.T @ (y - mu + h*(0.5 - mu))
        delta = Sinv @ Ustar
        beta_new = beta + delta
        if np.linalg.norm(delta) < tol*(1 + np.linalg.norm(beta)):
            beta = beta_new; break
        beta = beta_new
    return beta, t+1

firth_beta, firth_iters = firth_logit(X, y)

# Plots
plt.figure()
plt.plot(np.arange(mle_path.shape[0]), mle_path[:,0], marker='o', label="intercept", color='blue')
plt.plot(np.arange(mle_path.shape[0]), mle_path[:,1], marker='s', label="x1", color='orange')
plt.plot(np.arange(mle_path.shape[0]), mle_path[:,2], marker='^', label="x2", color='green')
plt.xlabel("Iteration"); plt.ylabel("Coefficient value")
plt.title("Unpenalized logistic regression under separation\n(Newton scoring diverges)")
plt.legend(); plt.tight_layout(); plt.show()

plt.figure()
plt.plot(np.arange(len(mle_norms)), mle_norms, marker='o', color='blue')
plt.xlabel("Iteration"); plt.ylabel("||β||₂")
plt.title("Explosion of coefficient norm under separation (MLE)")
plt.tight_layout(); plt.show()

plt.figure()
plt.semilogx(ridge_lams, ridge_betas[:,1], marker='o', label="x1", color='orange')
plt.semilogx(ridge_lams, ridge_betas[:,2], marker='s', label="x2", color='green')
plt.semilogx(ridge_lams, ridge_betas[:,0], marker='^', label="intercept", color='blue')
plt.xlabel("λ (log scale)"); plt.ylabel("Coefficient")
plt.title("Ridge-penalized logistic regression\nStable finite β̂_λ across λ")
plt.legend(); plt.tight_layout(); plt.show()

def firth_path_collect(X, y, max_iter=80):
    p = X.shape[1]; beta = np.zeros(p); path=[beta.copy()]
    for t in range(max_iter):
        mu = sigmoid(X@beta)
        W = mu*(1-mu)
        S = X.T * W @ X
        Sinv = inv(S + 1e-10*np.eye(p))
        WX = (W**0.5)[:,None] * X
        H = WX @ Sinv @ WX.T
        h = np.clip(np.diag(H), 0.0, 1.0)
        Ustar = X.T @ (y - mu + h*(0.5 - mu))
        delta = Sinv @ Ustar
        beta = beta + delta; path.append(beta.copy())
        if np.linalg.norm(delta) < 1e-8*(1 + np.linalg.norm(beta)):
            break
    return np.array(path)

fp = firth_path_collect(X, y, max_iter=80)
plt.figure()
plt.plot(np.arange(fp.shape[0]), fp[:,0], marker='o', label="intercept", color='blue')
plt.plot(np.arange(fp.shape[0]), fp[:,1], marker='s', label="x1", color='orange')
plt.plot(np.arange(fp.shape[0]), fp[:,2], marker='^', label="x2", color='green')
plt.xlabel("Iteration"); plt.ylabel("Coefficient value")
plt.title("Firth-penalized logistic regression\nConverges to finite β̂ (no separation)")
plt.legend(); plt.tight_layout(); plt.show()

# Summaries
summary = pd.DataFrame({
    "method": ["Unpenalized (last iter)","Ridge (λ=0.1)","Ridge (λ=1)","Firth"],
    "intercept":[mle_path[-1,0], ridge_df.iloc[(ridge_df['lambda']-0.1).abs().argmin()]["intercept"], 
                 ridge_df.iloc[(ridge_df['lambda']-1.0).abs().argmin()]["intercept"], firth_beta[0]],
    "x1":[mle_path[-1,1], ridge_df.iloc[(ridge_df['lambda']-0.1).abs().argmin()]["x1"],
          ridge_df.iloc[(ridge_df['lambda']-1.0).abs().argmin()]["x1"], firth_beta[1]],
    "x2":[mle_path[-1,2], ridge_df.iloc[(ridge_df['lambda']-0.1).abs().argmin()]["x2"],
          ridge_df.iloc[(ridge_df['lambda']-1.0).abs().argmin()]["x2"], firth_beta[2]]
}).reset_index(drop=True)
summary.round(6)




#==============================================================================
#=============== PROBLEM 4: Debiased Lasso (one coordinate) ===================
#==============================================================================

# Debiased Lasso (one coordinate) with nodewise Lasso direction
# Verify asymptotic N(0,1) of sqrt(n)(tilde_beta_j - beta*_j) / (sigma * sqrt(Theta_hat_jj))

# You may change these to experiment
n = 200                 # samples
p = 600                 # dimensions (p >> n)
s = 10                  # sparsity of beta*
rho = 0.3               # equicorrelation in Sigma
sigma = 1.0             # noise sd
j0 = 3                  # coordinate to infer/debias (0-indexed)
R = 400                 # Monte Carlo replicates (vary epsilon and y, fix X)

rng = np.random.default_rng(7)

# --- Building Gaussian design with equicorrelation and standardize columns to unit variance ---
Sigma = (1-rho)*np.eye(p) + rho*np.ones((p,p))
A = cholesky(Sigma)  # Sigma = A A^T
Z = rng.standard_normal((n,p))
X_raw = Z @ A.T
# center and scale
X = X_raw - X_raw.mean(axis=0, keepdims=True)
col_scales = np.sqrt((X**2).sum(axis=0) / n)
X = X / col_scales

# --- Sparse true beta* ---
beta_star = np.zeros(p)
support = rng.choice(p, size=s, replace=False)
beta_star[support] = rng.normal(0, 1.0, size=s)
# ensure j0 in support (to also test nonzero effect); if not, force a moderate signal
if j0 not in support:
    support[0] = j0
    beta_star[j0] = 0.8

# --- Helper: Lasso via scikit-learn (if available) else simple coordinate descent ---
def fit_lasso(X, y, alpha, max_iter=2000, tol=1e-6):
    try:
        from sklearn.linear_model import Lasso
        m = Lasso(alpha=alpha, fit_intercept=False, max_iter=max_iter, tol=tol, selection="cyclic")
        m.fit(X, y)
        return m.coef_
    except Exception:
        # fallback: basic coordinate descent (L1), not optimized but fine for this size
        n, p = X.shape
        beta = np.zeros(p)
        L = (X**2).sum(axis=0) / n  # Lipschitz constants
        for it in range(max_iter):
            beta_old = beta.copy()
            r = y - X @ beta
            for j in range(p):
                r += X[:,j]*beta[j]
                zj = (X[:,j] @ r) / n
                bj = zj / L[j]
                # soft-threshold
                beta[j] = np.sign(bj) * max(abs(bj) - alpha/L[j], 0.0)
                r -= X[:,j]*beta[j]
            if np.linalg.norm(beta - beta_old) < tol*(1+np.linalg.norm(beta_old)):
                break
        return beta

# --- Nodewise Lasso for a single column j0 to estimate Theta_j ---
def nodewise_theta_j(X, j, alpha_node):
    n, p = X.shape
    idx = np.ones(p, dtype=bool)
    idx[j] = False
    Xj = X[:, j]
    X_minus = X[:, idx]
    gamma = fit_lasso(X_minus, Xj, alpha=alpha_node)
    r = Xj - X_minus @ gamma
    tau2 = (r @ r)/n + alpha_node * np.linalg.norm(gamma, 1)  # van de Geer correction term
    theta = np.zeros(p)
    theta[j] = 1.0
    theta[idx] = -gamma
    theta = theta / tau2
    theta_jj = 1.0 / tau2
    return theta, theta_jj

# --- Tuning parameters (lasso penalty scale: 1/(2n)||y-Xb||^2 + alpha ||b||_1) ---
lam_scale = math.sqrt(2.0 * math.log(p) / n)
alpha_primal = 1.1 * lam_scale       # for β̂
alpha_node = 1.0 * lam_scale         # for nodewise Lasso

# Direction for j0 (fixed X): compute once
theta_j, theta_jj = nodewise_theta_j(X, j0, alpha_node)

# --- Monte Carlo over new epsilons ---
Z_stats = []
tilde_vals = []
lasso_vals = []

for r in range(R):
    eps = rng.normal(0.0, sigma, size=n)
    y = X @ beta_star + eps
    beta_hat = fit_lasso(X, y, alpha=alpha_primal)
    resid = y - X @ beta_hat
    # debiased one-coordinate
    tilde_beta_j = beta_hat[j0] + (theta_j @ (X.T @ resid)) / n
    Z = math.sqrt(n) * (tilde_beta_j - beta_star[j0]) / (sigma * math.sqrt(theta_jj))
    Z_stats.append(Z)
    tilde_vals.append(tilde_beta_j)
    lasso_vals.append(beta_hat[j0])

Z_stats = np.array(Z_stats)
tilde_vals = np.array(tilde_vals)
lasso_vals = np.array(lasso_vals)

# --- Summaries ---
summary = pd.DataFrame({
    "true_beta_j": [beta_star[j0]],
    "nodewise_tau2_inv(Theta_jj)": [theta_jj],
    "Z_mean": [Z_stats.mean()],
    "Z_std": [Z_stats.std(ddof=1)],
    "RMSE_lasso_j": [np.sqrt(np.mean((lasso_vals - beta_star[j0])**2))],
    "RMSE_debiased_j": [np.sqrt(np.mean((tilde_vals - beta_star[j0])**2))],
    "coverage_95%": [np.mean(np.abs(Z_stats) <= 1.96)]
})
summary.round(4)

# --- Plot 1: histogram of Z with standard normal pdf overlay ---
plt.figure()
count, bins, _ = plt.hist(Z_stats, bins=30, density=True, alpha=0.6)
x = np.linspace(Z_stats.min()-0.5, Z_stats.max()+0.5, 400)
pdf = (1/math.sqrt(2*math.pi)) * np.exp(-x**2 / 2.0)
plt.plot(x, pdf, linewidth=2, label="N(0,1) pdf")
plt.xlabel("Z")
plt.ylabel("Density")
plt.title("Debiased Lasso one-coordinate: distribution check (p >> n)")
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot 2: Q-Q plot against standard normal ---
plt.figure()
Z_sorted = np.sort(Z_stats)
q = norm.ppf((np.arange(1, R+1)-0.5)/R)
plt.plot(q, Z_sorted, marker='o', linestyle='')
plt.plot(q, q, linewidth=2, label="y=x")
plt.xlabel("Theoretical quantiles (N(0,1))")
plt.ylabel("Empirical quantiles of Z")
plt.title("Q-Q plot: debiased statistic vs N(0,1)")
plt.legend()
plt.tight_layout()
plt.show()




## ALternative Approach with different settings
def run_debiased_lasso_demo(n=360, p=800, s=10, rho=0.3, sigma=1.0, j0=3, R=120, seed=22):
    rng = np.random.default_rng(seed)
    Sigma = (1-rho)*np.eye(p) + rho*np.ones((p,p))
    A = cholesky(Sigma)
    Z = rng.standard_normal((n,p))
    X_raw = Z @ A.T
    X = X_raw - X_raw.mean(axis=0, keepdims=True)
    col_scales = np.sqrt((X**2).sum(axis=0) / n)
    X = X / col_scales

    beta_star = np.zeros(p)
    support = rng.choice(p, size=s, replace=False)
    beta_star[support] = rng.normal(0, 1.0, size=s)
    if j0 not in support: beta_star[j0] = 0.8

    def fit_lasso(X, y, alpha, max_iter=3000, tol=1e-6):
        try:
            from sklearn.linear_model import Lasso
            m = Lasso(alpha=alpha, fit_intercept=False, max_iter=max_iter, tol=tol, selection="cyclic")
            m.fit(X, y); return m.coef_
        except Exception:
            n, p = X.shape; beta=np.zeros(p); L=(X**2).sum(axis=0)/n
            for it in range(max_iter):
                beta_old=beta.copy(); r = y - X@beta
                for j in range(p):
                    r += X[:,j]*beta[j]
                    zj = (X[:,j]@r)/n; bj = zj/L[j]
                    beta[j] = np.sign(bj)*max(abs(bj)-alpha/L[j],0.0)
                    r -= X[:,j]*beta[j]
                if np.linalg.norm(beta-beta_old) < tol*(1+np.linalg.norm(beta_old)): break
            return beta

    def nodewise_theta_j(X, j, alpha_node):
        n, p = X.shape; idx=np.ones(p, dtype=bool); idx[j]=False
        Xj = X[:,j]; Xminus = X[:,idx]
        gamma = fit_lasso(Xminus, Xj, alpha=alpha_node)
        r = Xj - Xminus@gamma
        tau2 = (r@r)/n + alpha_node*np.linalg.norm(gamma,1)
        theta = np.zeros(p); theta[j]=1.0; theta[idx]=-gamma; theta/=tau2
        theta_jj = 1.0/tau2
        return theta, theta_jj

    lam_scale = math.sqrt(2.0*math.log(p)/n)
    alpha_primal = 1.05*lam_scale; alpha_node = 1.00*lam_scale
    theta_j, theta_jj = nodewise_theta_j(X, j0, alpha_node)

    Z_stats=[]
    for r in range(R):
        eps = rng.normal(0.0, sigma, size=n)
        y = X@beta_star + eps
        beta_hat = fit_lasso(X, y, alpha=alpha_primal)
        resid = y - X@beta_hat
        tilde_beta_j = beta_hat[j0] + (theta_j @ (X.T @ resid))/n
        Z = math.sqrt(n)*(tilde_beta_j - beta_star[j0])/(sigma*math.sqrt(theta_jj))
        Z_stats.append(Z)

    Z_stats = np.array(Z_stats)
    summary = pd.DataFrame({
        "n":[n],"p":[p],"s":[s],"rho":[rho],"Theta_hat_jj":[theta_jj],
        "Z_mean":[Z_stats.mean()],"Z_sd":[Z_stats.std(ddof=1)],
        "coverage_95%":[np.mean(np.abs(Z_stats)<=1.96)]
    })
    return Z_stats, summary

Z_big, summary_big = run_debiased_lasso_demo()
summary_big.round(4)

plt.figure()
plt.hist(Z_big, bins=24, density=True, alpha=0.65)
x = np.linspace(min(-4, Z_big.min()-0.5), max(4, Z_big.max()+0.5), 400)
pdf = (1/np.sqrt(2*np.pi)) * np.exp(-x**2 / 2.0)
plt.plot(x, pdf, linewidth=2, label="N(0,1) pdf")
plt.xlabel("Z (debiased one-coordinate, standardized)")
plt.ylabel("Density")
plt.title("Debiased Lasso: larger n,p — histogram vs N(0,1)")
plt.legend(); plt.tight_layout(); plt.show()

from scipy.stats import norm
plt.figure()
Z_sorted = np.sort(Z_big)
q = norm.ppf((np.arange(1, len(Z_big)+1)-0.5)/len(Z_big))
plt.plot(q, Z_sorted, marker='o', linestyle='')
plt.plot(q, q, linewidth=2, label="y=x")
plt.xlabel("Theoretical quantiles (N(0,1))"); plt.ylabel("Empirical quantiles of Z")
plt.title("Debiased Lasso: larger n,p — Q-Q vs N(0,1)")
plt.legend(); plt.tight_layout(); plt.show()


