"""Auswertung: Welten, Kennzahlen der Instanz, die vier Experimente über feste Ströme und die Verdikte. Nur numpy, kein Streamlit."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import spt_constants as C
import spt_model as model
from inverse_generator import SplitMix64, blurred_exact, grid, normal_cdf, operator, phantom

_SUPER = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def sci(x: float, digits: int = 1) -> str:
    """Zahl in Zehnerpotenz-Schreibweise, z. B. 2,5·10⁹."""
    if x == 0.0 or not math.isfinite(x):
        return de(x, digits) if math.isfinite(x) else "unendlich"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10.0 ** e
    if round(abs(m), digits) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{de(m, digits)}·10{str(e).translate(_SUPER)}"


def lam_text(lam: float) -> str:
    """Strafparameter: ab 0,01 mit Dezimalstellen, darunter als Zehnerpotenz."""
    if lam == 0.0:
        return "0"
    return de(lam, 3 if lam >= 0.1 else 2) if lam >= 0.01 else sci(lam)


def err_pct(err: float) -> str:
    """Fehler als Prozent der Signalnorm: ab 100 Prozent ohne, ab 1 Prozent mit einer, ab 0,01 Prozent mit zwei Stellen, darunter als Zehnerpotenz;
    ab dem Zehnfachen der Signalnorm als Vielfaches."""
    if not math.isfinite(err):
        return "unendlich"
    if err >= 10.0:
        return f"{sci(err)}-fach"
    x = 100.0 * err
    if x >= 99.5:
        return f"{de(x, 0)} %"
    if x >= 0.95:
        return f"{de(x, 1)} %"
    if x >= 0.01:
        return f"{de(x, 2)} %"
    return f"{sci(x)} %"


def int_text(k: float) -> str:
    """Ganze Zahl mit Punkt als Tausendertrenner; ab einer Million als Zehnerpotenz."""
    k = int(round(k))
    return f"{k:,}".replace(",", ".") if k < 10 ** 6 else sci(float(k))


def noise_label(v: float) -> str:
    """Rauschstufe als Prozent des Effektivwerts, z. B. 0,1 %."""
    return de(100.0 * v, 4).rstrip("0").rstrip(",") + " %" if v > 0.0 else "0 %"


def share(x: float) -> str:
    """Anteil (0 bis 1) als ganze Prozent."""
    return f"{de(100.0 * x, 0)} %"


def iter_steps(slider: int) -> int:
    """Schrittzahl von ISTA und FISTA zum Reglerwert: 10^(Regler/10), also 1 bis 10^5."""
    return max(1, int(round(10.0 ** (slider / C.STEP_DECADE))))


# ----------------------------------------------------------------------------- Welten

def world_truth(world: str, n: int, w: float):
    """(x, g): Wahrheit am Gitter und ihre analytische unscharfe Antwort (vom Gitter unabhängig)."""
    s = grid(n)
    if world == "phantom":
        return phantom(s), blurred_exact(s, w)
    x = np.zeros(n)
    g = np.zeros(n)
    if world == "spitzen":
        for pos, mass in C.SPIKES:
            j = int(round(pos * n - 0.5))
            x[j] += mass * n
            g += mass * np.exp(-((s - s[j]) ** 2) / (2.0 * w * w)) / (w * math.sqrt(2.0 * math.pi))
        return x, g
    if world == "bloecke":
        for left, right, height in C.BLOCKS:
            x += height * ((s >= left) & (s < right))
            g += height * (normal_cdf((s - left) / w) - normal_cdf((s - right) / w))
        return x, g
    raise ValueError(world)


class Problem:
    """Eine Instanz: Matrix, SVD, Wahrheit, analytische Daten, Rauschen; Zerlegung und TV-Entwurf werden einmal gerechnet."""

    def __init__(self, world: str, n: int, w: float, noise: float, truth: tuple | None = None):
        self.world, self.n, self.w, self.noise = world, n, w, noise
        self.a = operator(n, w)
        self.u, self.sv, self.vt = model.decompose(self.a)
        self.x, self.g = world_truth(world, n, w) if truth is None else truth
        self.sigma = float(noise) * float(np.sqrt(np.mean(self.g ** 2)))
        self.x_norm = float(np.linalg.norm(self.x))
        self.cond = float(self.sv[0] / self.sv[-1]) if self.sv[-1] > 0 else math.inf
        self.l2 = float(self.sv[0] ** 2)
        self.design = model.tv_design(self.a)
        s = grid(n)
        self.hill = (s[:-1] > C.HILL_RANGE[0]) & (s[:-1] < C.HILL_RANGE[1])

    def data(self, honest: bool, reps: int, seed: int) -> np.ndarray:
        base = self.g if honest else self.a @ self.x
        e = SplitMix64(seed).normal(self.n * reps).reshape(reps, self.n) * self.sigma
        return base[None, :] + e

    def err(self, xs: np.ndarray) -> np.ndarray:
        return np.linalg.norm(np.atleast_2d(xs) - self.x[None, :], axis=-1) / self.x_norm

    def res2(self, xs: np.ndarray, b: np.ndarray) -> np.ndarray:
        return ((np.atleast_2d(xs) @ self.a.T - b) ** 2).sum(axis=-1)

    def path(self, prior: str, b: np.ndarray, lams) -> np.ndarray:
        """Lösungen der gewählten Strafe über lams; Form (len(lams), reps, n)."""
        lams = np.asarray(lams, dtype=float)
        if prior == "tikhonov":
            return np.array([model.tikhonov(self.u, self.sv, self.vt, b, float(l)) for l in lams])
        if prior == "l1":
            return model.l1_path(self.a, b, lams)
        return model.tv_path(self.a, b, lams, self.design)


@lru_cache(maxsize=64)
def problem(world: str, n: int, w: float, noise: float) -> Problem:
    return Problem(world, n, w, noise)


def lam_grid(prior: str) -> np.ndarray:
    lo, hi, pts = C.TIK_GRID if prior == "tikhonov" else C.LAM_GRID
    return np.logspace(lo, hi, pts)


def rms(e: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sqrt(np.mean(np.asarray(e) ** 2, axis=axis))


# ----------------------------------------------------------------------------- Instanz

@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    w: float = C.DEFAULT_W
    noise: float = C.DEFAULT_NOISE
    data: str = "ehrlich"
    world: str = "phantom"
    prior: str = "tikhonov"
    lam: float = C.DEFAULT_LAM
    solver: str = "exakt"
    steps: int = C.DEFAULT_STEPS
    seed: int = C.DEFAULT_SEED


@dataclass
class Analysis:
    settings: Settings
    s: np.ndarray
    x: np.ndarray
    b: np.ndarray
    x_hat: np.ndarray         # gewählte Rekonstruktion
    x_exact: np.ndarray       # exakter Minimierer bei demselben λ (bei ISTA/FISTA verschieden von x_hat)
    x_naive: np.ndarray
    err: float
    err_exact: float
    err_naive: float
    steps: int                # Iterationen (0 bei exakt)
    matvec: int
    lams: np.ndarray          # Gitter der λ-Kurve
    err_curve: dict           # Prior -> Fehler dieser Stichprobe über lams
    best_lam: dict            # Prior -> λ mit dem kleinsten Fehler dieser Stichprobe
    best_err: dict
    best_x: dict
    jumps_curve: dict         # Prior -> Zahl der Sprünge über lams
    nnz_curve: dict           # Prior -> Zahl der Werte != 0 über lams
    jumps: int
    jumps_hill: int
    hill_steps: int
    nnz: int
    cond: float
    sigma: float
    singular: bool
    peak_hill: float          # Wert der Lösung an der Spitze von Hügel 1 (nur Phantom), Wahrheit in peak_truth
    peak_truth: float


def analyse(settings: Settings) -> Analysis:
    st = settings
    p = problem(st.world, st.n, st.w, st.noise)
    b = p.data(st.data != "verbrechen", 1, st.seed)
    lams = np.array(C.LAM_CHOICES)
    err_curve, jumps_curve, nnz_curve, best_lam, best_err, best_x = {}, {}, {}, {}, {}, {}
    for prior in C.PRIORS:
        xs = p.path(prior, b, lams)[:, 0, :]
        e = p.err(xs)
        err_curve[prior] = e
        jumps_curve[prior] = model.n_jumps(xs, C.JUMP_TOL)
        nnz_curve[prior] = (np.abs(xs) > 1e-9).sum(axis=1)
        i = int(np.argmin(e))
        best_lam[prior], best_err[prior], best_x[prior] = float(lams[i]), float(e[i]), xs[i]
    lam = float(st.lam)
    x_exact = p.path(st.prior, b, [lam])[0, 0]
    steps = 0
    x_hat = x_exact
    if st.prior == "l1" and st.solver != "exakt":
        steps = iter_steps(st.steps)
        x_hat = model.iterate_l1(p.a, b, lam, [steps], accel=(st.solver == "fista"), l2=p.l2)[0, 0]
    x_naive = (b[0] @ p.u / p.sv) @ p.vt
    s = grid(st.n)
    h1 = int(np.argmin(np.abs(s - 0.30)))
    return Analysis(
        settings=st, s=s, x=p.x, b=b[0], x_hat=x_hat, x_exact=x_exact, x_naive=x_naive, err=float(p.err(x_hat)[0]), err_exact=float(p.err(x_exact)[0]),
        err_naive=float(p.err(x_naive)[0]), steps=steps, matvec=2 * steps, lams=lams, err_curve=err_curve, best_lam=best_lam, best_err=best_err, best_x=best_x,
        jumps_curve=jumps_curve, nnz_curve=nnz_curve, jumps=int(model.n_jumps(x_hat, C.JUMP_TOL)[0]), jumps_hill=int(model.n_jumps(x_hat, C.JUMP_TOL, p.hill)[0]),
        hill_steps=int(p.hill.sum()), nnz=int((np.abs(x_hat) > 1e-9).sum()), cond=p.cond, sigma=p.sigma, singular=bool(p.cond >= C.SINGULAR_COND),
        peak_hill=float(x_hat[h1]), peak_truth=float(p.x[h1]),
    )


# ----------------------------------------------------------------------------- Experiment 1: welcher Prior in welcher Welt

_PATHS: dict = {}


def _paths(world: str, n: int, w: float, noise: float, honest: bool, reps: int, seed: int, prior: str):
    """Pfade (zwischengespeichert, weil mehrere Experimente sie teilen) und die Daten dazu."""
    key = (world, n, w, noise, honest, reps, seed, prior)
    if key not in _PATHS:
        p = problem(world, n, w, noise)
        b = p.data(honest, reps, seed)
        _PATHS[key] = (p.path(prior, b, lam_grid(prior)), b)
    return _PATHS[key]


def best_cell(world: str, prior: str, honest: bool = True, noise: float = C.DEFAULT_NOISE, reps: int = C.EXP_REPS, n: int = C.DEFAULT_N, w: float = C.DEFAULT_W,
              seed: int = C.SEED_BASE + 1) -> dict:
    """Bester Fehler über das λ-Gitter (RMS über die Wiederholungen), mit Median, 90. Perzentil und dem Diskrepanzprinzip als Wahl aus den Daten."""
    p = problem(world, n, w, noise)
    xs, b = _paths(world, n, w, noise, honest, reps, seed, prior)
    lams = lam_grid(prior)
    errs = np.array([p.err(x) for x in xs])                       # (λ, reps)
    table = rms(errs, axis=1)
    i = int(np.argmin(table))
    res2 = np.array([p.res2(x, b) for x in xs])
    target = p.sigma ** 2 * n
    ok = res2 <= target
    pick = np.array([int(np.flatnonzero(ok[:, r]).max()) if ok[:, r].any() else 0 for r in range(reps)])
    e_dp = errs[pick, np.arange(reps)]
    opt = float(table[i])
    return {"lam": float(lams[i]), "rms": opt, "median": float(np.median(errs[i])), "p90": float(np.quantile(errs[i], 0.9)),
            "dp_median": float(np.median(e_dp)), "dp_rms": float(rms(e_dp)), "dp_fail": float(np.mean(e_dp > C.FAIL_FACTOR * opt)),
            "dp_lam": float(np.median(lams[pick])), "index": i}


def prior_world_experiment(reps: int = C.EXP_REPS, n: int = C.DEFAULT_N, w: float = C.DEFAULT_W, reps_small: int = C.EXP_REPS_SMALL) -> dict:
    out = {"worlds": {}, "noise": [], "crime": {}, "naive": {}, "reps": reps, "reps_small": reps_small, "n": n}
    for world in C.WORLDS:
        out["worlds"][world] = {prior: best_cell(world, prior, reps=reps, n=n, w=w) for prior in C.PRIORS}
        p = problem(world, n, w, C.DEFAULT_NOISE)
        _, b = _paths(world, n, w, C.DEFAULT_NOISE, True, reps, C.SEED_BASE + 1, "tikhonov")
        out["naive"][world] = float(rms(p.err((b @ p.u / p.sv) @ p.vt)))
    for nz in C.EXP_NOISE_LEVELS:
        out["noise"].append({"noise": nz, **{prior: best_cell("phantom", prior, noise=nz, reps=reps_small, n=n, w=w, seed=C.SEED_BASE + 2) for prior in C.PRIORS}})
    out["crime"] = {prior: best_cell("phantom", prior, honest=False, reps=reps_small, n=n, w=w, seed=C.SEED_BASE + 3) for prior in C.PRIORS}
    return out


# ----------------------------------------------------------------------------- Experiment 2: Terrassenbildung

def terrace_experiment(reps: int = C.EXP_REPS, n: int = C.DEFAULT_N, w: float = C.DEFAULT_W) -> dict:
    p = problem("phantom", n, w, C.DEFAULT_NOISE)
    b = p.data(True, reps, C.SEED_BASE + 1)
    lams = np.array(C.EXP_TERRACE_LAMS)
    tv = model.tv_path(p.a, b, lams, p.design)
    tik = np.array([model.tikhonov(p.u, p.sv, p.vt, b, float(l)) for l in lams])
    h1 = int(np.argmin(np.abs(grid(n) - 0.30)))
    rows = []
    for i, lam in enumerate(lams):
        dx = np.abs(np.diff(tv[i], axis=1))
        rows.append({"lam": float(lam), "tv_rms": float(rms(p.err(tv[i]))), "tv_jumps": float(model.n_jumps(tv[i], C.JUMP_TOL, p.hill).mean()),
                     "tv_peak": float(tv[i][:, h1].mean()), "tv_step": float(dx[:, p.hill].max(axis=1).mean()),
                     "tik_rms": float(rms(p.err(tik[i]))), "tik_peak": float(tik[i][:, h1].mean())})
    # rauschfrei: die Terrassen selbst
    p0 = problem("phantom", n, w, 0.0)
    b0 = p0.data(True, 1, C.SEED_BASE + 1)
    clean = {float(l): model.tv_path(p0.a, b0, [l], p0.design)[0, 0] for l in (1e-4, 1e-2)}
    return {"rows": rows, "hill_steps": int(p.hill.sum()), "peak_truth": float(p.x[h1]), "s": grid(n), "x": p.x, "clean": clean, "reps": reps, "n": n}


# ----------------------------------------------------------------------------- Experiment 3: das Verfahren regularisiert mit

def iteration_experiment(reps: int = C.EXP_REPS, n: int = C.DEFAULT_N, w: float = C.DEFAULT_W) -> dict:
    cell = best_cell("phantom", "l1", reps=reps, n=n, w=w)
    lam = cell["lam"]
    p = problem("phantom", n, w, C.DEFAULT_NOISE)
    b = p.data(True, reps, C.SEED_BASE + 1)
    exact = model.l1_path(p.a, b, [lam])[0]
    f_ex = model.objective_l1(p.a, b, exact, lam)
    rows = {}
    for key, accel in (("ista", False), ("fista", True)):
        its = model.iterate_l1(p.a, b, lam, C.EXP_ITER_STEPS, accel=accel, l2=p.l2)
        rows[key] = [{"steps": int(k), "rms": float(rms(p.err(x))), "gap": float(np.max(model.objective_l1(p.a, b, x, lam) - f_ex)),
                      "dist": float(np.abs(x - exact).max())} for k, x in zip(C.EXP_ITER_STEPS, its)]
    return {"lam": lam, "exact_rms": float(rms(p.err(exact))), "rows": rows, "reps": reps, "n": n,
            "tik_rms": best_cell("phantom", "tikhonov", reps=reps, n=n, w=w)["rms"]}


# ----------------------------------------------------------------------------- Experiment 4: Auflösung zweier Spitzen

def resolution_experiment(reps: int = C.EXP_REPS_SMALL, n: int = 48, w: float = C.DEFAULT_W) -> dict:
    s = grid(n)
    j1 = n // 2 - 4
    mass = 0.04
    rows = []
    for noise in C.EXP_RES_NOISE:
        for d in C.EXP_RES_DISTANCES:
            j2 = j1 + d
            x = np.zeros(n)
            x[j1] = x[j2] = mass * n
            g = mass * (np.exp(-((s - s[j1]) ** 2) / (2 * w * w)) + np.exp(-((s - s[j2]) ** 2) / (2 * w * w))) / (w * math.sqrt(2 * math.pi))
            p = Problem("spitzen", n, w, noise, truth=(x, g))
            b = p.data(True, reps, C.SEED_BASE + 4)
            lo, hi, pts = C.EXP_RES_LAMS
            lams = np.logspace(lo, hi, pts)
            xs = model.l1_path(p.a, b, lams)
            table = np.array([float(rms(p.err(v))) for v in xs])
            i = int(np.argmin(table))
            sup = np.abs(xs[i]) > C.SUPPORT_TOL * x.max()
            hit = float(np.mean([set(np.flatnonzero(sup[r])) == {j1, j2} for r in range(reps)]))
            tik = min(float(rms(p.err(model.tikhonov(p.u, p.sv, p.vt, b, float(l))))) for l in lam_grid("tikhonov")[::2])
            rows.append({"noise": noise, "d": d, "l1_rms": float(table[i]), "hit": hit, "tik_rms": tik, "lam": float(lams[i])})
    return {"rows": rows, "reps": reps, "n": n, "width_cells": w * n}


# ----------------------------------------------------------------------------- Verdikte: (Art, Text)

def verdict_instance(a: Analysis) -> tuple[str, str]:
    st = a.settings
    ref = a.best_err["tikhonov"]
    name = C.PRIOR_SHORT[st.prior]
    how = "" if st.solver == "exakt" or st.prior != "l1" else f" ({C.SOLVER_SHORT[st.solver]} nach {int_text(a.steps)} Schritten, der exakte Minimierer läge bei {err_pct(a.err_exact)})"
    lead = f"{name} bei λ = {lam_text(st.lam)}: Fehler {err_pct(a.err)}{how}."
    tail = (f" Bester λ dieser Stichprobe je Prior: Tikhonov {err_pct(a.best_err['tikhonov'])}, L1 {err_pct(a.best_err['l1'])}, TV {err_pct(a.best_err['tv'])}.")
    best_other = a.best_err[st.prior]
    if st.prior == "tikhonov":
        if a.best_err["tv"] < 0.8 * ref or a.best_err["l1"] < 0.8 * ref:
            return "info", f"{lead} In dieser Welt schlägt ein anderer Prior Tikhonov deutlich (bester Fehler unter dem 0,8-Fachen).{tail}"
        return "success", f"{lead} Kein anderer Prior unterbietet Tikhonov deutlich (kein bester Fehler unter dem 0,8-Fachen).{tail}"
    if best_other < 0.8 * ref:
        return "success", f"{lead} Der Prior passt: sein bester Fehler {err_pct(best_other)} liegt unter dem 0,8-Fachen des Fehlers von Tikhonov ({err_pct(ref)}).{tail}"
    if best_other > 1.2 * ref:
        return "warning", f"{lead} Der Prior passt nicht: sein bester Fehler {err_pct(best_other)} liegt über dem 1,2-Fachen des Fehlers von Tikhonov ({err_pct(ref)}).{tail}"
    return "info", f"{lead} Der Prior ist hier etwa so gut wie Tikhonov (bester Fehler {err_pct(best_other)} gegen {err_pct(ref)}).{tail}"


def verdict_prior_world(res: dict) -> tuple[str, str]:
    w = res["worlds"]
    parts = []
    for world in C.WORLDS:
        r = w[world]
        parts.append(f"{C.WORLD_SHORT[world]}: Tikhonov {err_pct(r['tikhonov']['rms'])}, L1 {err_pct(r['l1']['rms'])}, TV {err_pct(r['tv']['rms'])}")
    return "warning", ("Bester Fehler je Welt (RMS über feste Wiederholungen): " + "; ".join(parts) + ". Jeder Prior gewinnt nur in der Welt, zu der er passt; "
                       "auf dem glatten Phantom schlägt Tikhonov beide.")


def verdict_terrace(res: dict) -> tuple[str, str]:
    by = {r["lam"]: r for r in res["rows"]}
    lo, hi = by[1e-6], by[1e-2]
    return "info", (f"Im Hügelbereich ({res['hill_steps']} Gitterschritte) hat TV bei λ = {lam_text(1e-6)} noch {de(lo['tv_jumps'], 1)} Sprünge, bei λ = {lam_text(1e-2)} nur {de(hi['tv_jumps'], 1)}; "
                    f"der Gipfel von Hügel 1 sinkt von {de(res['peak_truth'], 2)} auf {de(lo['tv_peak'], 2)} und {de(hi['tv_peak'], 2)}, bei Tikhonov bleibt er bei {de(hi['tik_peak'], 2)}. "
                    f"Der Fehler steigt von {err_pct(lo['tv_rms'])} auf {err_pct(hi['tv_rms'])}.")


def verdict_iteration(res: dict) -> tuple[str, str]:
    f = res["rows"]["fista"]
    best = min(f, key=lambda r: r["rms"])
    last = f[-1]
    return "warning", (f"Bei λ = {lam_text(res['lam'])} liegt der exakte L1-Minimierer bei {err_pct(res['exact_rms'])}. FISTA erreicht nach {int_text(best['steps'])} Schritten "
                       f"{err_pct(best['rms'])} und nach {int_text(last['steps'])} Schritten {err_pct(last['rms'])}, obwohl die Zielfunktionslücke dabei auf {sci(last['gap'])} fällt: "
                       "das Verfahren ist hier der bessere Regularisierer als das Problem.")


def verdict_resolution(res: dict) -> tuple[str, str]:
    rows = {(r["noise"], r["d"]): r for r in res["rows"]}
    hard = rows[(0.1, 2)]
    easy = rows[(0.001, 2)]
    return "info", (f"Bei {noise_label(0.001)} Rauschen findet L1 beide Spitzen in {share(easy['hit'])} der Wiederholungen schon im Abstand 2 (Fehler: Tikhonov {err_pct(easy['tik_rms'])}, "
                    f"L1 {err_pct(easy['l1_rms'])}); bei {noise_label(0.1)} sind es im Abstand 2 nur {share(hard['hit'])}, im Abstand 6 {share(rows[(0.1, 6)]['hit'])}.")
