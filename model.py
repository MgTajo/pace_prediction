"""
The pace model: a Bayesian dynamic linear model (structural time series).

WHAT IT IS
----------
Exactly the latent-fitness state-space model from the project brief, written
in its batch (smoother) form so it is solvable in closed form.

    log v_i  =  phi(date_i)            # latent log-vVO2max "fitness"
              + beta_r * 1[threshold]  # physiological threshold<->vVO2max offset
              + beta1 * P1_i           # heat penalty (linear in effective temp)
              + beta2 * P2_i           # heat penalty (quadratic)
              + eps_i,  eps_i ~ N(0, sigma^2)

phi follows a random walk in calendar time:

    phi(d_k) - phi(d_{k-1}) ~ N(0, tau^2 * (d_k - d_{k-1}))

i.e. fitness drifts slowly and smoothly.  A small `tau` (the prior that
fitness changes slowly) is precisely what lets the model separate the slow
fitness trend from the fast day-to-day weather variation -- without it the
seasonal temperature swing and the seasonal fitness gain would be confounded.

WHY THIS FORM
-------------
Because every term is *linear in the unknown parameters* and every noise is
Gaussian, the latent-fitness smoother is identical to the posterior of a
Bayesian linear regression.  That means:

  * exact inference (no EM / particle filter), robust on ~50 data points,
  * a single latent fitness feeding *both* paces through `beta_r`, so the app
    works even if you have only ever logged threshold sessions,
  * full predictive uncertainty for free,
  * irregular timestamps and mixed session types handled trivially.

Two noise scales (sigma, tau) are fit by empirical Bayes (maximising the
marginal likelihood); the weather coefficients, the threshold offset and the
fitness level all get informative priors so the model personalises as data
arrives.

NO USER BASELINE
----------------
The user is never asked for a cool-conditions reference pace.  The fitness
anchor is read straight off the logged sessions: each session is
weather-corrected to ideal-conditions vVO2max with the *prior* heat
coefficients (and the prior threshold offset), and their mean seeds the
fitness prior (see `config_from_sessions`).  The anchor is loose (sd ~12%) and
session noise is tight (~1.5%), so a single logged session already pins the
fitness; the anchor only keeps the prior proper and sane for extrapolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
from scipy.optimize import minimize

import physiology as phys


# --------------------------------------------------------------------------
# Configuration / priors  (all in log-velocity units; 0.01 ~ 1%)
# --------------------------------------------------------------------------

# Fallback fitness anchor used only before any session is logged: a typical
# recreational vVO2max pace (~4:00/km).  Once even one session exists the anchor
# is derived from the data (`config_from_sessions`), so this barely matters.
DEFAULT_BASELINE_PACE_SEC = 240.0


@dataclass
class ModelConfig:
    # Fitness anchor: log-velocity of vVO2max in ideal (~optimum) conditions.
    baseline_logv: float
    baseline_sd: float = 0.12          # how far current fitness may sit from it

    # Heat-sensitivity priors (expected to be negative: heat -> slower).
    beta1_mean: float = -0.0045        # per degC above optimum (linear)
    beta1_sd: float = 0.004
    beta2_mean: float = -0.00018       # per degC^2 (quadratic)
    beta2_sd: float = 0.00020

    # Threshold offset prior: log(LT fraction of vVO2max).
    betar_mean: float = float(np.log(phys.DEFAULT_LT_FRACTION))
    betar_sd: float = 0.030

    # Noise scales (starting points; refined by empirical Bayes).
    sigma0: float = 0.015              # session-to-session observation noise
    tau0: float = 0.005               # fitness random-walk sd per sqrt(day)

    # Bounds for the empirical-Bayes search.
    sigma_bounds: tuple = (0.004, 0.06)
    tau_bounds: tuple = (5e-4, 0.03)

    z: float = 1.0                     # interval half-width (1 sd ~ 68%)


# Index of the three regression coefficients within the parameter vector,
# *relative to the end* (the K fitness nodes come first).
_B1, _B2, _BR = -3, -2, -1


def config_from_sessions(sessions: list[dict],
                         lt_fraction: float = phys.DEFAULT_LT_FRACTION,
                         **overrides) -> ModelConfig:
    """Build a ModelConfig whose fitness anchor is read off the sessions.

    There is no user-supplied baseline: instead we weather-correct every logged
    session back to ideal-conditions vVO2max log-velocity using the *prior*
    heat coefficients (and the prior threshold offset for threshold sessions),
    and anchor the fitness prior at their mean.  With no sessions yet we fall
    back to a generic recreational pace so the prior stays proper.

    `lt_fraction` only seeds the threshold-offset prior `betar_mean`; the offset
    itself is re-estimated from the data (the `beta_r` parameter).
    """
    betar_mean = float(np.log(lt_fraction))
    if not sessions:
        baseline_logv = phys.pace_to_logv(DEFAULT_BASELINE_PACE_SEC)
    else:
        # Use the default priors purely to undo weather here; the fit refines them.
        b1, b2 = ModelConfig.beta1_mean, ModelConfig.beta2_mean
        vals = []
        for s in sessions:
            w = phys.Weather.from_row(s)
            p1, p2 = phys.heat_penalty_basis(w)
            ideal = phys.pace_to_logv(s["pace_sec"]) - (b1 * p1 + b2 * p2)
            if s["session_type"] == "threshold":
                ideal -= betar_mean
            vals.append(ideal)
        baseline_logv = float(np.mean(vals))
    return ModelConfig(
        baseline_logv=baseline_logv,
        betar_mean=betar_mean,
        **overrides,
    )


@dataclass
class FitResult:
    nodes: np.ndarray            # sorted unique day-numbers (length K)
    mean: np.ndarray             # posterior mean of [phi_1..phi_K, b1, b2, br]
    cov: np.ndarray              # posterior covariance (M x M)
    sigma: float
    tau: float
    config: ModelConfig
    n_obs: int
    day0: int                    # reference ordinal (nodes are days since day0)


# --------------------------------------------------------------------------
# Building the design + prior
# --------------------------------------------------------------------------

def _phi_prior_precision(nodes: np.ndarray, tau: float, cfg: ModelConfig):
    """Tridiagonal random-walk precision for the fitness nodes, plus a soft
    anchor of the first node to the baseline and a tiny global ridge for
    numerical properness.  Returns (Q, b) in information form so that the
    prior mean (all nodes = baseline) satisfies  Q @ mean = b.
    """
    K = len(nodes)
    Q = np.zeros((K, K))
    b = np.zeros(K)
    mu0 = cfg.baseline_logv

    # Random walk between consecutive nodes (variance scales with the gap).
    for k in range(K - 1):
        gap = max(1.0, float(nodes[k + 1] - nodes[k]))
        w = 1.0 / (tau * tau * gap)
        Q[k, k] += w
        Q[k + 1, k + 1] += w
        Q[k, k + 1] -= w
        Q[k + 1, k] -= w

    # Anchor the first node to the baseline fitness.
    Q[0, 0] += 1.0 / (cfg.baseline_sd ** 2)
    b[0] += mu0 / (cfg.baseline_sd ** 2)

    # Very weak global ridge toward the baseline (keeps far-future / sparse
    # regions proper without meaningfully biasing well-observed nodes).
    ridge = 1.0  # sd = 1.0 in log-velocity -> essentially uninformative
    Q[np.diag_indices(K)] += ridge
    b += ridge * mu0
    return Q, b


def _build(nodes, day_of, P1, P2, thr, y, cfg, sigma, tau):
    """Assemble prior precision Lambda, info-vector b_prior and design X."""
    K = len(nodes)
    M = K + 3
    Lam = np.zeros((M, M))
    b = np.zeros(M)

    Q, bq = _phi_prior_precision(nodes, tau, cfg)
    Lam[:K, :K] = Q
    b[:K] = bq

    Lam[_B1, _B1] = 1.0 / cfg.beta1_sd ** 2
    Lam[_B2, _B2] = 1.0 / cfg.beta2_sd ** 2
    Lam[_BR, _BR] = 1.0 / cfg.betar_sd ** 2
    b[_B1] = cfg.beta1_mean / cfg.beta1_sd ** 2
    b[_B2] = cfg.beta2_mean / cfg.beta2_sd ** 2
    b[_BR] = cfg.betar_mean / cfg.betar_sd ** 2

    N = len(y)
    X = np.zeros((N, M))
    for i in range(N):
        X[i, day_of[i]] = 1.0
    X[:, _B1] = P1
    X[:, _B2] = P2
    X[:, _BR] = thr
    return Lam, b, X


def _prior_mean(K, cfg) -> np.ndarray:
    mu = np.empty(K + 3)
    mu[:K] = cfg.baseline_logv
    mu[_B1] = cfg.beta1_mean
    mu[_B2] = cfg.beta2_mean
    mu[_BR] = cfg.betar_mean
    return mu


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

def _neg_log_marginal(log_params, nodes, day_of, P1, P2, thr, y, cfg):
    sigma = float(np.exp(log_params[0]))
    tau = float(np.exp(log_params[1]))
    Lam, b, X = _build(nodes, day_of, P1, P2, thr, y, cfg, sigma, tau)
    K = len(nodes)
    mu_p = _prior_mean(K, cfg)

    # y ~ N(X mu_p,  X Lam^-1 X^T + sigma^2 I)
    Lam_inv = np.linalg.inv(Lam)
    N = len(y)
    Sigma = X @ Lam_inv @ X.T + sigma * sigma * np.eye(N)
    r = y - X @ mu_p
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        return 1e12
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, r))
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return 0.5 * (r @ alpha + logdet + N * np.log(2 * np.pi))


def fit(sessions: list[dict], cfg: ModelConfig) -> FitResult:
    """Fit the model to a user's sessions.

    `sessions` is a list of dicts with keys:
        date (datetime.date), session_type ('threshold'|'vo2max'),
        pace_sec (float), temp_c, sky, rain, humidity
    """
    if not sessions:
        return FitResult(
            nodes=np.array([], dtype=int),
            mean=_prior_mean(0, cfg),
            cov=np.diag([cfg.beta1_sd ** 2, cfg.beta2_sd ** 2, cfg.betar_sd ** 2]),
            sigma=cfg.sigma0, tau=cfg.tau0, config=cfg, n_obs=0, day0=0,
        )

    sessions = sorted(sessions, key=lambda s: s["date"])
    day0 = sessions[0]["date"].toordinal()
    days = np.array([s["date"].toordinal() - day0 for s in sessions])
    nodes = np.unique(days)
    node_index = {d: k for k, d in enumerate(nodes)}
    day_of = np.array([node_index[d] for d in days])

    P1 = np.empty(len(sessions))
    P2 = np.empty(len(sessions))
    thr = np.empty(len(sessions))
    y = np.empty(len(sessions))
    for i, s in enumerate(sessions):
        w = phys.Weather.from_row(s)
        p1, p2 = phys.heat_penalty_basis(w)
        P1[i], P2[i] = p1, p2
        thr[i] = 1.0 if s["session_type"] == "threshold" else 0.0
        y[i] = phys.pace_to_logv(s["pace_sec"])

    # Empirical Bayes for (sigma, tau) -- only worthwhile with enough data.
    sigma, tau = cfg.sigma0, cfg.tau0
    if len(sessions) >= 4:
        res = minimize(
            _neg_log_marginal,
            x0=np.log([cfg.sigma0, cfg.tau0]),
            args=(nodes, day_of, P1, P2, thr, y, cfg),
            method="L-BFGS-B",
            bounds=[np.log(cfg.sigma_bounds), np.log(cfg.tau_bounds)],
        )
        if res.success or np.isfinite(res.fun):
            sigma, tau = np.exp(res.x)

    # Posterior:  A = Lambda + X^T X / sigma^2 ;  m = A^-1 (b + X^T y / sigma^2)
    Lam, b, X = _build(nodes, day_of, P1, P2, thr, y, cfg, sigma, tau)
    A = Lam + (X.T @ X) / (sigma * sigma)
    rhs = b + (X.T @ y) / (sigma * sigma)
    cov = np.linalg.inv(A)
    mean = cov @ rhs

    return FitResult(
        nodes=nodes, mean=mean, cov=cov, sigma=float(sigma), tau=float(tau),
        config=cfg, n_obs=len(sessions), day0=day0,
    )


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------

def _fitness_functional(fit_res: FitResult, target_day: int):
    """Linear functional `a` (over the parameter vector) and extra independent
    variance giving phi at `target_day` via the random-walk structure.

    Returns (a_phi, extra_var) where a_phi has length K (weights on the
    fitness nodes) and extra_var is the additional latent variance from
    extrapolating / interpolating the random walk to target_day.
    """
    nodes = fit_res.nodes
    K = len(nodes)
    a = np.zeros(K)
    tau2 = fit_res.tau ** 2
    if K == 0:
        return a, fit_res.config.baseline_sd ** 2  # handled by caller anyway

    if target_day <= nodes[0]:
        a[0] = 1.0
        extra = tau2 * (nodes[0] - target_day)
    elif target_day >= nodes[-1]:
        a[-1] = 1.0
        extra = tau2 * (target_day - nodes[-1])
    else:
        k = int(np.searchsorted(nodes, target_day))  # nodes[k-1] < target <= nodes[k]
        d_lo, d_hi = nodes[k - 1], nodes[k]
        span = float(d_hi - d_lo)
        w_hi = (target_day - d_lo) / span
        a[k - 1] = 1.0 - w_hi
        a[k] = w_hi
        # Brownian-bridge variance at an interior point.
        extra = tau2 * (target_day - d_lo) * (d_hi - target_day) / span
    return a, float(extra)


def _predict_logv(fit_res: FitResult, target_day: int, P1, P2, threshold: bool):
    """Posterior mean and variance of log-velocity (the *latent ability*,
    excluding session noise) for one pace type at one day/weather."""
    cfg = fit_res.config
    K = len(fit_res.nodes)
    M = K + 3

    a = np.zeros(M)
    if K == 0:
        # No sessions yet: fall back to the baseline prior for fitness.
        phi_mean = cfg.baseline_logv
        phi_var = cfg.baseline_sd ** 2
        a[_B1], a[_B2] = P1, P2
        if threshold:
            a[_BR] = 1.0
        mean = phi_mean + a @ fit_res.mean
        var = phi_var + a @ fit_res.cov @ a
        return mean, var

    a_phi, extra = _fitness_functional(fit_res, target_day)
    a[:K] = a_phi
    a[_B1], a[_B2] = P1, P2
    if threshold:
        a[_BR] = 1.0

    mean = a @ fit_res.mean
    var = a @ fit_res.cov @ a + extra
    return mean, var


def predict(fit_res: FitResult, target_date: date, weather: phys.Weather):
    """Predict both paces for a given day & weather.

    Returns a dict with, for each of 'vo2max' and 'threshold':
        pace_sec, pace_lo (faster bound), pace_hi (slower bound)
    plus the effective temperature and the implied heat slowdown (%).
    """
    cfg = fit_res.config
    target_day = target_date.toordinal() - fit_res.day0
    if weather.date is None:          # the solar term needs a date
        weather.date = target_date
    P1, P2 = phys.heat_penalty_basis(weather)
    z = cfg.z

    out = {
        "effective_temp": phys.effective_temperature(weather),
        "target_date": target_date,
    }
    for label, is_thr in (("vo2max", False), ("threshold", True)):
        mean, var = _predict_logv(fit_res, target_day, P1, P2, is_thr)
        sd = float(np.sqrt(max(var, 0.0)))
        out[label] = {
            "pace_sec": phys.logv_to_pace(mean),
            # Higher log-v = faster = lower pace, hence the swap.
            "pace_lo": phys.logv_to_pace(mean + z * sd),
            "pace_hi": phys.logv_to_pace(mean - z * sd),
            "logv": mean,
            "logv_sd": sd,
        }

    # Heat slowdown relative to ideal conditions, at current fitness.
    ideal = phys.Weather(temp_c=phys.THERMAL_OPTIMUM_C, sky="overcast",
                         rain="none", humidity=phys.HUMIDITY_REF)
    m_now, _ = _predict_logv(fit_res, target_day, *phys.heat_penalty_basis(weather), False)
    m_ideal, _ = _predict_logv(fit_res, target_day, *phys.heat_penalty_basis(ideal), False)
    out["heat_slowdown_pct"] = 100.0 * (1.0 - np.exp(m_now - m_ideal))
    return out


# --------------------------------------------------------------------------
# Diagnostics for plotting
# --------------------------------------------------------------------------

def fitness_trajectory(fit_res: FitResult):
    """Per-node fitness expressed as ideal-conditions vVO2max pace (sec/km),
    with a 1-sd band.  Returns (days_since_day0, pace, pace_lo, pace_hi)."""
    K = len(fit_res.nodes)
    if K == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    mean = fit_res.mean[:K]
    sd = np.sqrt(np.clip(np.diag(fit_res.cov)[:K], 0, None))
    pace = np.array([phys.logv_to_pace(m) for m in mean])
    lo = np.array([phys.logv_to_pace(m + s) for m, s in zip(mean, sd)])  # faster
    hi = np.array([phys.logv_to_pace(m - s) for m, s in zip(mean, sd)])  # slower
    return fit_res.nodes.astype(float), pace, lo, hi


def heat_curve(fit_res: FitResult, target_date: date,
               temps, sky="overcast", rain="none", humidity=phys.HUMIDITY_REF,
               tod=None):
    """Predicted vVO2max & threshold pace across a range of temperatures at the
    fitness level of `target_date` (and, if given, time of day `tod` for the
    solar term).  Returns dict of arrays for plotting."""
    vo2, thr = [], []
    for t in temps:
        w = phys.Weather(temp_c=float(t), sky=sky, rain=rain, humidity=humidity,
                         date=target_date, time=tod)
        p = predict(fit_res, target_date, w)
        vo2.append(p["vo2max"]["pace_sec"])
        thr.append(p["threshold"]["pace_sec"])
    return {"temp": np.asarray(temps, float),
            "vo2max": np.asarray(vo2), "threshold": np.asarray(thr)}


def learned_lt_fraction(fit_res: FitResult) -> float:
    """The current estimate of vLT / vVO2max (exp of the threshold offset)."""
    return float(np.exp(fit_res.mean[_BR]))


def heat_sensitivity_at(fit_res: FitResult, temp_c: float) -> float:
    """Estimated % slowdown at `temp_c` (overcast, 50% RH) vs the optimum."""
    w = phys.Weather(temp_c=temp_c, sky="overcast", rain="none",
                     humidity=phys.HUMIDITY_REF)
    p1, p2 = phys.heat_penalty_basis(w)
    drop = fit_res.mean[_B1] * p1 + fit_res.mean[_B2] * p2
    return 100.0 * (1.0 - float(np.exp(drop)))


def weather_normalized_paces(fit_res: FitResult, sessions: list[dict]):
    """For each session, remove the estimated weather penalty (and threshold
    offset) to express it as an *ideal-conditions vVO2max* pace.  Lets us plot
    the raw sessions on the same axis as the latent fitness line.

    Returns (days_since_day0, ideal_pace_sec).
    """
    b1, b2, br = fit_res.mean[_B1], fit_res.mean[_B2], fit_res.mean[_BR]
    days, paces = [], []
    for s in sessions:
        w = phys.Weather.from_row(s)
        p1, p2 = phys.heat_penalty_basis(w)
        logv = phys.pace_to_logv(s["pace_sec"])
        ideal = logv - (b1 * p1 + b2 * p2)
        if s["session_type"] == "threshold":
            ideal -= br
        days.append(s["date"].toordinal() - fit_res.day0)
        paces.append(phys.logv_to_pace(ideal))
    return np.asarray(days, float), np.asarray(paces, float)
