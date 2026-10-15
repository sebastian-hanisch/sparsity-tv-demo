"""Kern: Tikhonov, exakte Lösungen von L1 und TV (Aktivmenge), ISTA/FISTA, KKT-Prüfung. Nur numpy, kein Streamlit.

Jede Funktion ist eine kleine, einzeln prüfbare Einheit (DEMO-PLAYBOOK §1b): `tests/test_model.py` rechnet sie an der 3 x 3-Mini-Instanz von Hand und
`tests/test_oracle_spt.py` gegen die KKT-Bedingungen, scipy (L-BFGS-B auf der aufgespaltenen Form), eine unabhängige Koordinatenabstiegs-Fassung und eine
Gegenprobe der TV-Transformation.

Aufgaben (A die Vorwärtsmatrix, b die Daten, lam >= 0):
    L1:   min 1/2 |A x - b|^2 + lam |x|_1
    TV:   min 1/2 |A x - b|^2 + lam |D x|_1            D = Vorwärtsdifferenz (n - 1 Zeilen)
    Tikhonov (wie Stück 2): x = V diag(sigma / (sigma^2 + lam^2)) U^T b

TV wird auf Lasso zurückgeführt: x = x0 * 1 + L w mit w = D x (Sprünge) und der unteren Dreiecksmatrix L; der Mittelwert x0 ist nicht bestraft und wird
herausprojiziert. Beide Aufgaben löst ein Aktivmengenverfahren exakt (bis auf Rundung); ISTA und FISTA sind die Iterationen, die die Linie zeigt.
"""

from __future__ import annotations

import math

import numpy as np

_TINY = 1e-300


def soft(v, t):
    """Weiche Schwelle sign(v) max(|v| - t, 0): die Prox-Abbildung von t |.|_1."""
    v = np.asarray(v, dtype=float)
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


def decompose(a: np.ndarray):
    """Volle SVD A = U diag(sv) V^T (sv absteigend)."""
    return np.linalg.svd(a)


def tikhonov(u: np.ndarray, sv: np.ndarray, vt: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
    """Tikhonov-Lösung für jede Zeile von b: Filter sv^2 / (sv^2 + lam^2) wie in Stück 2."""
    f = sv ** 2 / (sv ** 2 + lam ** 2)
    return ((np.atleast_2d(b) @ u) * f / np.maximum(sv, _TINY)) @ vt


def lasso_active_set(m: np.ndarray, y: np.ndarray, lam: float, w0: np.ndarray | None = None, maxit: int = 5000, tol: float = 1e-10,
                     mtm: np.ndarray | None = None) -> np.ndarray:
    """Exakte Lösung von min 1/2 |M w - y|^2 + lam |w|_1 per Aktivmenge (Warmstart w0 erlaubt, z. B. die Lösung des nächstgrößeren lam).

    Je Runde: auf der aktiven Menge mit festen Vorzeichen das lineare System lösen; wechselt dabei ein Vorzeichen, die Koordinate mit dem frühesten
    Wechsel entfernen und neu lösen; sonst die am stärksten verletzte Optimalitätsbedingung |g_j| <= lam der inaktiven Koordinaten aufnehmen (Vorzeichen -sign g_j)."""
    p = m.shape[1]
    mtm = m.T @ m if mtm is None else mtm
    mty = m.T @ y
    w = np.zeros(p) if w0 is None else np.array(w0, dtype=float)
    active = w != 0.0
    sgn = np.sign(w)
    for _ in range(maxit):
        idx = np.flatnonzero(active)
        if idx.size:
            s = sgn[idx]
            try:
                sol = np.linalg.solve(mtm[idx][:, idx], mty[idx] - lam * s)
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(mtm[idx][:, idx], mty[idx] - lam * s, rcond=None)[0]
            bad = sol * s < 0.0
            if bad.any():
                w_old = w[idx]
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = np.where(bad, w_old / (w_old - sol), np.inf)
                j = int(np.argmin(t))
                w[idx[j]] = 0.0                              # die Koordinate mit dem frühesten Vorzeichenwechsel verlässt die aktive Menge
                active[idx[j]] = False
                continue
            w[idx] = sol
        g = mtm @ w - mty
        viol = np.where(active, 0.0, np.abs(g) - lam)
        j = int(np.argmax(viol))
        if viol[j] <= tol * max(lam, _TINY):
            return w
        active[j] = True
        sgn[j] = -np.sign(g[j])
        w[j] = 0.0
    return w


def tv_design(a: np.ndarray):
    """Transformation der TV-Aufgabe auf Lasso: (L, c, M, proj) mit x = x0 * 1 + L w, c = A 1, proj = I - c c^T / |c|^2, M = proj A L."""
    n = a.shape[0]
    lmat = np.tril(np.ones((n, n - 1)), -1)
    c = a @ np.ones(n)
    proj = np.eye(n) - np.outer(c, c) / float(c @ c)
    return lmat, c, proj @ a @ lmat, proj


def l1_path(a: np.ndarray, b: np.ndarray, lams) -> np.ndarray:
    """Exakte L1-Lösungen für jede Zeile von b und jedes lam (beliebige Reihenfolge, intern absteigend mit Warmstart); Form (len(lams), reps, n)."""
    lams = np.asarray(lams, dtype=float)
    b = np.atleast_2d(b)
    order = np.argsort(lams)[::-1]
    out = np.zeros((len(lams), b.shape[0], a.shape[1]))
    ata = a.T @ a
    for r in range(b.shape[0]):
        w = None
        for i in order:
            w = lasso_active_set(a, b[r], float(lams[i]), w, mtm=ata)
            out[i, r] = w
    return out


def tv_path(a: np.ndarray, b: np.ndarray, lams, design=None) -> np.ndarray:
    """Exakte TV-Lösungen wie l1_path; Form (len(lams), reps, n)."""
    lams = np.asarray(lams, dtype=float)
    b = np.atleast_2d(b)
    lmat, c, m, proj = tv_design(a) if design is None else design
    order = np.argsort(lams)[::-1]
    out = np.zeros((len(lams), b.shape[0], a.shape[1]))
    cc = float(c @ c)
    mtm = m.T @ m
    for r in range(b.shape[0]):
        y = proj @ b[r]
        w = None
        for i in order:
            w = lasso_active_set(m, y, float(lams[i]), w, mtm=mtm)
            x0 = float(c @ (b[r] - a @ (lmat @ w))) / cc
            out[i, r] = x0 + lmat @ w
    return out


def iterate_l1(a: np.ndarray, b: np.ndarray, lam: float, checkpoints, accel: bool, l2: float | None = None) -> np.ndarray:
    """ISTA (accel=False) oder FISTA (accel=True) für die L1-Aufgabe mit Schrittweite 1/sigma_1^2, vektorisiert über die Zeilen von b; gibt die Iterierten an den
    Schrittzahlen `checkpoints` zurück, Form (len(checkpoints), reps, n). Schritt 0 ist die Null."""
    b = np.atleast_2d(b)
    l2 = float(np.linalg.norm(a, 2) ** 2) if l2 is None else l2
    step = 1.0 / l2
    cps = sorted(int(c) for c in checkpoints)
    x = np.zeros((b.shape[0], a.shape[1]))
    y = x.copy()
    t = 1.0
    out = {}
    k = 0
    for target in cps:
        while k < target:
            xn = soft(y - step * ((y @ a.T - b) @ a), step * lam)
            if accel:
                tn = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
                y = xn + ((t - 1.0) / tn) * (xn - x)
                t = tn
            else:
                y = xn
            x = xn
            k += 1
        out[target] = x.copy()
    return np.array([out[int(c)] for c in checkpoints])


def objective_l1(a: np.ndarray, b: np.ndarray, x: np.ndarray, lam: float) -> np.ndarray:
    """1/2 |A x - b|^2 + lam |x|_1 je Zeile."""
    r = np.atleast_2d(x) @ a.T - np.atleast_2d(b)
    return 0.5 * (r ** 2).sum(axis=1) + lam * np.abs(np.atleast_2d(x)).sum(axis=1)


def objective_tv(a: np.ndarray, b: np.ndarray, x: np.ndarray, lam: float) -> np.ndarray:
    """1/2 |A x - b|^2 + lam |D x|_1 je Zeile."""
    x = np.atleast_2d(x)
    r = x @ a.T - np.atleast_2d(b)
    return 0.5 * (r ** 2).sum(axis=1) + lam * np.abs(np.diff(x, axis=1)).sum(axis=1)


def kkt_l1(a: np.ndarray, b: np.ndarray, x: np.ndarray, lam: float) -> np.ndarray:
    """Größte Verletzung der Optimalität je Zeile: g = A^T (A x - b); auf x_i != 0 gilt g_i = -lam sign(x_i), sonst |g_i| <= lam."""
    x = np.atleast_2d(x)
    g = (x @ a.T - np.atleast_2d(b)) @ a
    nz = np.abs(x) > 1e-12
    v = np.where(nz, np.abs(g + lam * np.sign(x)), np.maximum(np.abs(g) - lam, 0.0))
    return v.max(axis=1)


def kkt_tv(a: np.ndarray, b: np.ndarray, x: np.ndarray, lam: float, jump_tol: float = 1e-9) -> np.ndarray:
    """Verletzung der TV-Optimalität je Zeile. Die Bedingung A^T (A x - b) + lam D^T s = 0 mit s in sign(D x) gibt s = cumsum(q) für q = A^T (A x - b) / lam
    (und sum q = 0); s muss auf den Sprüngen gleich sign(D x) sein und sonst |s| <= 1."""
    x = np.atleast_2d(x)
    q = ((x @ a.T - np.atleast_2d(b)) @ a) / lam
    s = np.cumsum(q, axis=1)[:, :-1]
    dx = np.diff(x, axis=1)
    jump = np.abs(dx) > jump_tol
    v = np.where(jump, np.abs(s - np.sign(dx)), np.maximum(np.abs(s) - 1.0, 0.0))
    return np.maximum(v.max(axis=1), np.abs(q.sum(axis=1)))


def n_jumps(x: np.ndarray, tol: float = 1e-9, mask: np.ndarray | None = None) -> np.ndarray:
    """Zahl der Sprünge |x_{i+1} - x_i| > tol je Zeile, optional nur an den Stellen von mask (Länge n - 1)."""
    d = np.abs(np.diff(np.atleast_2d(x), axis=1)) > tol
    if mask is not None:
        d = d[:, mask]
    return d.sum(axis=1)
