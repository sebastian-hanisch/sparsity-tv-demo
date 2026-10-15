"""Orakel (DEMO-PLAYBOOK §12): Kernroutinen gegen einen anderen Rechenweg (KKT-Bedingungen, scipy L-BFGS-B auf der aufgespaltenen Form, Koordinatenabstieg,
duales projiziertes Gradientenverfahren für TV-Entrauschen, die bekannten Konvergenzschranken von ISTA und FISTA, zufällige Störungen). Jeder Test unter 10 s.

Fehlt scipy im Test-venv, überspringen sich die scipy-Tests (pytest.importorskip), sie raten nie."""

import math

import numpy as np
import pytest

import spt_evaluation as ev
import spt_model as m
from inverse_generator import SplitMix64, blurred_exact, grid, operator, phantom


def _inst(n=12, w=0.03, noise=1e-3, seed=5, world="phantom"):
    a = operator(n, w)
    x, g = ev.world_truth(world, n, w)
    sigma = noise * float(np.sqrt(np.mean(g ** 2)))
    b = g + SplitMix64(seed).normal(n) * sigma
    return a, x, b


@pytest.mark.parametrize("n,world", [(12, "phantom"), (24, "phantom"), (48, "phantom"), (48, "spitzen"), (48, "bloecke")])
def test_exact_solutions_satisfy_the_kkt_conditions(n, world):
    a, x, b = _inst(n, world=world)
    lams = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    p1 = m.l1_path(a, b, lams)
    p2 = m.tv_path(a, b, lams)
    for i, lam in enumerate(lams):
        assert m.kkt_l1(a, b, p1[i], lam).max() < 1e-8
        assert m.kkt_tv(a, b, p2[i], lam).max() < 1e-5


def test_l1_matches_a_box_constrained_quasi_newton_on_the_split_form():
    scipy = pytest.importorskip("scipy")
    from scipy import optimize

    a, x, b = _inst(12)
    n = 12
    for lam in (1e-4, 1e-3, 1e-2):
        def f(v):
            r = a @ (v[:n] - v[n:]) - b
            return 0.5 * float(r @ r) + lam * float(v.sum())

        def grad(v):
            g = a.T @ (a @ (v[:n] - v[n:]) - b)
            return np.concatenate([g + lam, -g + lam])

        res = optimize.minimize(f, np.zeros(2 * n), jac=grad, method="L-BFGS-B", bounds=[(0, None)] * (2 * n), options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 5000})
        ref = res.x[:n] - res.x[n:]
        ours = m.lasso_active_set(a, b, lam)
        assert m.objective_l1(a, b, ours, lam)[0] <= m.objective_l1(a, b, ref, lam)[0] + 1e-12
        assert ours == pytest.approx(ref, abs=1e-4)


def test_l1_matches_an_independent_coordinate_descent():
    a, x, b = _inst(12)
    lam = 3e-3
    w = np.zeros(12)
    col2 = (a ** 2).sum(axis=0)
    for _ in range(4000):
        for j in range(12):
            r = b - a @ w + a[:, j] * w[j]
            w[j] = float(np.sign(a[:, j] @ r) * max(abs(a[:, j] @ r) - lam, 0.0) / col2[j])
    assert m.lasso_active_set(a, b, lam) == pytest.approx(w, abs=1e-7)


def test_tv_matches_a_smoothed_objective_minimizer_in_the_objective_value():
    scipy = pytest.importorskip("scipy")
    from scipy import optimize

    a, x, b = _inst(12)
    d = np.diff(np.eye(12), axis=0)
    for lam in (1e-4, 1e-3):
        eps = 1e-9

        def f(v):
            r = a @ v - b
            return 0.5 * float(r @ r) + lam * float(np.sqrt((d @ v) ** 2 + eps).sum())

        res = optimize.minimize(f, np.full(12, b.mean()), method="L-BFGS-B", options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 20000, "maxfun": 100000})
        ours = m.tv_path(a, b, [lam])[0, 0]
        assert m.objective_tv(a, b, ours, lam)[0] <= m.objective_tv(a, b, res.x, lam)[0] + 1e-10
        assert m.objective_tv(a, b, ours, lam)[0] == pytest.approx(m.objective_tv(a, b, res.x, lam)[0], rel=1e-6)


def test_tv_denoising_matches_a_dual_projected_gradient_solver():
    """A = I: x = b - D^T p mit p = argmin 1/2 |D^T p - b|^2 unter |p|_inf <= lam (Chambolle), unabhängig vom Aktivmengenverfahren."""
    n = 16
    b = SplitMix64(9).normal(n)
    d = np.diff(np.eye(n), axis=0)
    for lam in (0.1, 0.5, 1.0):
        p = np.zeros(n - 1)
        for _ in range(60000):
            p = np.clip(p - 0.25 * (d @ (d.T @ p - b)), -lam, lam)
        ref = b - d.T @ p
        assert m.tv_path(np.eye(n), b, [lam])[0, 0] == pytest.approx(ref, abs=1e-6)


def test_exact_solutions_have_no_better_neighbour():
    a, x, b = _inst(24)
    rng = np.random.default_rng(3)
    lam = 1e-3
    for fn, obj in ((lambda: m.l1_path(a, b, [lam])[0, 0], m.objective_l1), (lambda: m.tv_path(a, b, [lam])[0, 0], m.objective_tv)):
        best = fn()
        f0 = obj(a, b, best, lam)[0]
        for _ in range(300):
            e = rng.normal(size=24) * 10.0 ** rng.uniform(-6, -1)
            assert obj(a, b, best + e, lam)[0] >= f0 - 1e-12


def test_ista_and_fista_converge_to_the_exact_solution_with_the_known_rates():
    a, x, b = _inst(12)
    lam = 3e-3
    ex = m.lasso_active_set(a, b, lam)
    f_star = m.objective_l1(a, b, ex, lam)[0]
    l2 = float(np.linalg.norm(a, 2) ** 2)
    e0 = float(ex @ ex)                                                           # |x_0 - x*|^2 mit x_0 = 0
    ks = [10, 50, 200, 1000]
    ista = m.iterate_l1(a, b, lam, ks, accel=False)
    fista = m.iterate_l1(a, b, lam, ks, accel=True)
    for k, xi, xf in zip(ks, ista, fista):
        assert m.objective_l1(a, b, xi, lam)[0] - f_star <= l2 * e0 / (2.0 * k) + 1e-12                 # ISTA: O(1/k)
        assert m.objective_l1(a, b, xf, lam)[0] - f_star <= 2.0 * l2 * e0 / (k + 1) ** 2 + 1e-12       # FISTA: O(1/k^2)
    long = m.iterate_l1(a, b, lam, [20000], accel=True)[0, 0]
    assert long == pytest.approx(ex, abs=1e-6)
    objs = [m.objective_l1(a, b, v, lam)[0] for v in m.iterate_l1(a, b, lam, list(range(0, 60)), accel=False)]
    assert all(x2 <= x1 + 1e-15 for x1, x2 in zip(objs, objs[1:]))                                    # ISTA fällt monoton


def test_limits_lambda_to_zero_and_lambda_above_max():
    a, x, b = _inst(12)
    direct = np.linalg.solve(a, b)
    assert m.lasso_active_set(a, b, 1e-13) == pytest.approx(direct, abs=1e-6)
    assert m.tv_path(a, b, [1e-13])[0, 0] == pytest.approx(direct, abs=1e-6)
    lam_max = float(np.max(np.abs(a.T @ b)))
    assert not m.lasso_active_set(a, b, lam_max * 1.0001).any()
    assert m.lasso_active_set(a, b, lam_max * 0.99).any()
    const = m.tv_path(a, b, [1e6])[0, 0]
    assert np.allclose(const, const[0]) and const[0] == pytest.approx(float((a @ np.ones(12)) @ b) / float((a @ np.ones(12)) @ (a @ np.ones(12))))


def test_tv_solution_has_at_most_as_many_jumps_as_the_data_allow_and_fewer_with_larger_lambda():
    a, x, b = _inst(48)
    lams = [1e-6, 1e-4, 1e-2, 1e-1]
    xs = m.tv_path(a, b, lams)[:, 0]
    jumps = [int(m.n_jumps(v)[0]) for v in xs]
    assert jumps == sorted(jumps, reverse=True) and jumps[0] > jumps[-1] and jumps[0] <= 47


def test_worlds_match_a_numerical_convolution():
    n, w = 48, 0.03
    fine = 4000
    s = (np.arange(fine) + 0.5) / fine
    t = grid(n)
    for world in ("spitzen", "bloecke"):
        x, g = ev.world_truth(world, n, w)
        if world == "bloecke":
            xs = np.zeros(fine)
            for left, right, height in ev.C.BLOCKS:
                xs += height * ((s >= left) & (s < right))
            conv = np.array([np.sum(np.exp(-((ti - s) ** 2) / (2 * w * w)) / (w * math.sqrt(2 * math.pi)) * xs) / fine for ti in t])
            assert g == pytest.approx(conv, abs=2e-4)
        else:
            assert float(g.sum()) / n == pytest.approx(sum(mass for _, mass in ev.C.SPIKES), rel=0.02)     # Masse bleibt erhalten (ohne Rand)
            assert float(x.sum()) / n == pytest.approx(sum(mass for _, mass in ev.C.SPIKES), rel=1e-12)
    x, g = ev.world_truth("phantom", n, w)
    assert x == pytest.approx(phantom(grid(n))) and g == pytest.approx(blurred_exact(grid(n), w))


def test_the_noise_streams_are_reproducible_and_honest_data_differ_from_crime_data():
    p = ev.problem("phantom", 48, 0.03, 1e-3)
    assert np.array_equal(p.data(True, 5, 7), p.data(True, 5, 7)) and not np.array_equal(p.data(True, 5, 7), p.data(False, 5, 7))
    assert p.sigma == pytest.approx(1e-3 * float(np.sqrt(np.mean(p.g ** 2))))
