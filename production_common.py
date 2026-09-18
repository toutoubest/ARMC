# Shared config and per-method fit/score interface for the production run.
#
# Methods:
#   standard  : standard_scores_batch
#   huber     : huber_scores_batch
#   armc_v1   : armc_v1_scores_batch (naive max(dense, sparse), no null calibration)
#   armc_v2   : fit_profile_v3 / combined_scores, channels=(dense, sparse, scale)
#   armc_v3   : fit_profile_v3_hier / combined_scores_hier,
#               channels=(dense, sparse, scale, dependence)
#   armc_v3_1 : fit_profile_v31_hier / combined_scores_v31_hier
#
# Hyperparameters: DEFAULT_HP_V3 / DEFAULT_HP_V31 in methods_v3.py / methods_v3_1.py.
#
# Seeds are offset by 13 * scenario index to avoid collisions:
#   profile: 10000, calibration: 20000, FA-eval: 90000, power-eval: 95000.
import numpy as np
from core_v3 import generate_batch
from methods_v3 import (
    DEFAULT_HP_V3, fit_profile_v3, combined_scores,
    fit_profile_v3_hier, combined_scores_hier,
    standard_scores_batch, huber_scores_batch, armc_v1_scores_batch,
    get_detection_times_batch, summarize_run, se_prop,
)
from methods_v3_1 import DEFAULT_HP_V31, fit_profile_v31_hier, combined_scores_v31_hier

T, TAU, N_TRAIN, RHO = 500, 250, 100, 0.3

SCENARIOS = {
    "gaussian_dense":     dict(p=20),
    "gaussian_sparse":    dict(p=60, sparse_s=3),
    "t3_dense":           dict(p=20),
    "contaminated_dense": dict(p=20),
    "variance_shift":     dict(p=20, var_gamma=1.5),
    "correlation_shift":  dict(p=20, rho=0.2, corr_rho_after=0.6),
}
SCEN_ORDER = list(SCENARIOS.keys())

DELTA_MAIN = 1.0     # default signal magnitude
HP = DEFAULT_HP_V3


def scen_kwargs(cfg):
    return {k: v for k, v in cfg.items() if k not in ("p", "rho")}


def seeds_for(scenario):
    i = SCEN_ORDER.index(scenario)
    return dict(profile=10000 + 13 * i, cal=20000 + 13 * i,
                fa_eval=90000 + 13 * i, power_eval=95000 + 13 * i)


# ---- per-method scoring interface: fit(profile) -> prof ; score(X, prof) -> (B,T) scores
def fit_standard(scenario, cfg, B_profile, seed):
    return None  # no profiling needed


def score_standard(X, prof):
    return standard_scores_batch(X, N_TRAIN)


def fit_huber(scenario, cfg, B_profile, seed):
    return None


def score_huber(X, prof):
    return huber_scores_batch(X, N_TRAIN)


def fit_armc_v1(scenario, cfg, B_profile, seed):
    return None


def score_armc_v1(X, prof):
    return armc_v1_scores_batch(X, N_TRAIN)


def fit_armc_v2(scenario, cfg, B_profile, seed):
    p = cfg["p"]; rho = cfg.get("rho", RHO)
    return fit_profile_v3(scenario, B_profile, T, p, TAU, N_TRAIN, rho, seed=seed,
                           hp=HP, channels=("dense", "sparse", "scale"), **scen_kwargs(cfg))


def score_armc_v2(X, prof):
    s, _ = combined_scores(X, N_TRAIN, prof, HP)
    return s


def fit_armc_v3(scenario, cfg, B_profile, seed):
    p = cfg["p"]; rho = cfg.get("rho", RHO)
    return fit_profile_v3_hier(scenario, B_profile, T, p, TAU, N_TRAIN, rho, seed=seed,
                                hp=HP, channels=("dense", "sparse", "scale", "dependence"),
                                **scen_kwargs(cfg))


def score_armc_v3(X, prof):
    s, _ = combined_scores_hier(X, N_TRAIN, prof, HP)
    return s


def fit_armc_v3_1(scenario, cfg, B_profile, seed):
    p = cfg["p"]; rho = cfg.get("rho", RHO)
    return fit_profile_v31_hier(scenario, B_profile, T, p, TAU, N_TRAIN, rho, seed=seed,
                                 hp=DEFAULT_HP_V31, channels=("dense", "sparse", "scale", "dependence"),
                                 **scen_kwargs(cfg))


def score_armc_v3_1(X, prof):
    s, _ = combined_scores_v31_hier(X, N_TRAIN, prof, DEFAULT_HP_V31)
    return s


METHODS = {
    "standard": (fit_standard, score_standard),
    "huber":    (fit_huber, score_huber),
    "armc_v1":  (fit_armc_v1, score_armc_v1),
    "armc_v2":  (fit_armc_v2, score_armc_v2),
    "armc_v3":  (fit_armc_v3, score_armc_v3),
    "armc_v3_1": (fit_armc_v3_1, score_armc_v3_1),
}


def gen(scenario, cfg, B, seed, no_change, delta=None, corr_rho_after=None, var_gamma=None, sparse_s=None):
    p = cfg["p"]; rho = cfg.get("rho", RHO)
    kw = dict(scen_kwargs(cfg))
    if corr_rho_after is not None:
        kw["corr_rho_after"] = corr_rho_after
    if var_gamma is not None:
        kw["var_gamma"] = var_gamma
    if sparse_s is not None:
        kw["sparse_s"] = sparse_s
    rng = np.random.default_rng(seed)
    kwargs = dict(kw)
    if no_change:
        kwargs.pop("delta", None)
        X, mu1 = generate_batch(B, T, p, TAU, scenario, rng, rho=rho, no_change=True, **kwargs)
    else:
        X, mu1 = generate_batch(B, T, p, TAU, scenario, rng, rho=rho, no_change=False,
                                 delta=(delta if delta is not None else DELTA_MAIN), **kwargs)
    return X, mu1


def run_experiment_A(scenario, cfg, B_profile, B_cal, B_eval, delta=None):
    """Calibrate each method to alpha=0.05 under its own H0, then evaluate
    power/delay on a fresh H1 batch. Returns dict method -> fa/power/threshold/prof."""
    sd = seeds_for(scenario)
    out = {}
    for name, (fit_fn, score_fn) in METHODS.items():
        prof = fit_fn(scenario, cfg, B_profile, sd["profile"])
        Xc, _ = gen(scenario, cfg, B_cal, sd["cal"], no_change=True)
        sc = score_fn(Xc, prof)
        thr = float(np.quantile(sc[:, N_TRAIN:].max(axis=1), 0.95))

        X0, _ = gen(scenario, cfg, B_eval, sd["fa_eval"], no_change=True)
        s0 = score_fn(X0, prof)
        fa = summarize_run(get_detection_times_batch(s0, thr, N_TRAIN), TAU, no_change=True)

        X1, _ = gen(scenario, cfg, B_eval, sd["power_eval"], no_change=False, delta=delta)
        s1 = score_fn(X1, prof)
        pw = summarize_run(get_detection_times_batch(s1, thr, N_TRAIN), TAU, no_change=False)

        out[name] = dict(fa=fa, power=pw, threshold=thr, prof=prof)
    return out
