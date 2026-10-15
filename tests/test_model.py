"""Kernroutinen an der 3 x 3-Mini-Instanz von Hand (DEMO-PLAYBOOK §1b); jede Einheit einzeln."""

import numpy as np
import pytest

import spt_model as m


def test_soft_threshold_by_hand():
    out = m.soft(np.array([-2.0, -0.5, 0.0, 0.5, 2.0]), 1.0)
    assert list(out) == [-1.0, 0.0, 0.0, 0.0, 1.0]
    assert m.soft(np.array([3.0, -3.0]), 0.0).tolist() == [3.0, -3.0]


def test_tikhonov_at_lambda_zero_is_the_direct_inversion_and_matches_the_normal_equations(mini):
    a, x, b = mini
    u, sv, vt = m.decompose(a)
    assert m.tikhonov(u, sv, vt, b, 0.0)[0] == pytest.approx(x, abs=1e-9)
    lam = 0.3
    ref = np.linalg.solve(a.T @ a + lam ** 2 * np.eye(3), a.T @ b)
    assert m.tikhonov(u, sv, vt, b, lam)[0] == pytest.approx(ref, abs=1e-12)


def test_l1_is_zero_above_lambda_max_and_has_one_active_coordinate_just_below(mini):
    a, x, b = mini
    assert np.allclose(m.lasso_active_set(a, b, 1.34 + 1e-9), 0.0)
    w = m.lasso_active_set(a, b, 1.3)
    assert w[0] == 0.0 and w[2] == 0.0 and w[1] == pytest.approx(0.04 / 0.34)
    g = a.T @ (a @ w - b)
    assert abs(g[0]) == pytest.approx(1.2912, abs=1e-4) and abs(g[0]) <= 1.3


def test_l1_with_vanishing_lambda_is_the_direct_inversion(mini):
    a, x, b = mini
    assert m.lasso_active_set(a, b, 1e-12) == pytest.approx(x, abs=1e-6)


def test_l1_denoising_is_the_soft_threshold(mini):
    _, _, b = mini
    assert m.lasso_active_set(np.eye(3), b, 0.5) == pytest.approx([0.8, 0.9, 0.8])
    assert m.lasso_active_set(np.eye(3), b, 0.2) == pytest.approx([1.1, 1.2, 1.1])
    assert not m.lasso_active_set(np.eye(3), b, 1.4).any()


def test_l1_warm_start_does_not_change_the_solution(mini):
    a, x, b = mini
    cold = m.lasso_active_set(a, b, 0.05)
    warm = m.lasso_active_set(a, b, 0.05, w0=m.lasso_active_set(a, b, 0.5))
    assert warm == pytest.approx(cold, abs=1e-12)


def test_tv_design_by_hand(mini):
    a, x, b = mini
    lmat, c, mm, proj = m.tv_design(a)
    assert lmat.shape == (3, 2) and lmat.tolist() == [[0, 0], [1, 0], [1, 1]]
    assert c == pytest.approx([1.0, 1.0, 1.0])                                   # Zeilensumme 1
    assert proj == pytest.approx(np.eye(3) - np.ones((3, 3)) / 3.0)
    assert mm == pytest.approx(proj @ a @ lmat)
    assert (c @ proj) == pytest.approx(np.zeros(3), abs=1e-15)                   # der Mittelwert ist herausprojiziert


def test_tv_is_constant_above_one_three_hundredth_and_has_the_hand_solution_below(mini):
    a, x, b = mini
    assert m.tv_path(a, b, [1.0 / 300.0 + 1e-6])[0, 0] == pytest.approx([4.0 / 3.0] * 3, abs=1e-9)
    assert m.tv_path(a, b, [1.0])[0, 0] == pytest.approx([4.0 / 3.0] * 3, abs=1e-9)
    got = m.tv_path(a, b, [1.0 / 600.0])[0, 0]
    assert got == pytest.approx([7.0 / 6.0, 5.0 / 3.0, 7.0 / 6.0], abs=1e-9)
    assert m.tv_path(a, b, [1e-12])[0, 0] == pytest.approx(x, abs=1e-6)


def test_tv_denoising_by_hand(mini):
    _, _, b = mini
    eye = np.eye(3)
    assert m.tv_path(eye, b, [0.01])[0, 0] == pytest.approx([1.31, 1.38, 1.31], abs=1e-9)
    assert m.tv_path(eye, b, [1.0 / 30.0 + 1e-6])[0, 0] == pytest.approx([4.0 / 3.0] * 3, abs=1e-9)
    assert m.tv_path(eye, b, [0.02])[0, 0] == pytest.approx([1.32, 1.36, 1.32], abs=1e-9)


def test_paths_return_the_requested_order_and_shape(mini):
    a, x, b = mini
    lams = [0.5, 0.01, 0.1]
    p1 = m.l1_path(a, np.vstack([b, b + 0.01]), lams)
    assert p1.shape == (3, 2, 3)
    for i, lam in enumerate(lams):
        assert p1[i, 0] == pytest.approx(m.lasso_active_set(a, b, lam), abs=1e-12)
    p2 = m.tv_path(a, b, lams)
    assert p2.shape == (3, 1, 3)


def test_ista_first_and_second_step_by_hand(mini):
    a, x, b = mini
    out = m.iterate_l1(a, b, 0.3, [0, 1, 2], accel=False)
    assert out.shape == (3, 1, 3) and not out[0].any()
    assert out[1, 0] == pytest.approx([1.03, 1.04, 1.03])
    assert out[2, 0] == pytest.approx([1.0267, 1.0466, 1.0267], abs=1e-9)


def test_fista_first_two_steps_equal_ista_and_the_third_has_the_momentum_by_hand(mini):
    """t_0 = 1, t_1 = (1 + sqrt 5)/2 = 1,618, t_2 = (1 + sqrt(1 + 4 t_1^2))/2 = 2,1894: das Trägheitsglied vor Schritt 3 ist (t_1 - 1)/t_2 = 0,28216."""
    a, x, b = mini
    f = m.iterate_l1(a, b, 0.3, [1, 2, 3], accel=True)
    i = m.iterate_l1(a, b, 0.3, [1, 2, 3], accel=False)
    assert f[0] == pytest.approx(i[0]) and f[1] == pytest.approx(i[1])
    x1, x2 = f[0, 0], f[1, 0]
    y2 = x2 + 0.28216 * (x2 - x1)
    x3 = m.soft(y2 - a.T @ (a @ y2 - b), 0.3)
    assert f[2, 0] == pytest.approx(x3, abs=2e-5)
    assert not np.allclose(f[2], i[2], atol=1e-6)


def test_objectives_and_jump_counts_by_hand(mini):
    a, x, b = mini
    assert m.objective_l1(a, b, x, 0.5)[0] == pytest.approx(0.5 * 4.0)           # Residuum 0, |x|_1 = 4
    assert m.objective_tv(a, b, x, 0.5)[0] == pytest.approx(0.5 * 2.0)           # |Dx|_1 = 1 + 1
    zero = np.zeros(3)
    assert m.objective_l1(a, b, zero, 0.5)[0] == pytest.approx(0.5 * (1.69 + 1.96 + 1.69))
    arr = np.array([[0.0, 0.0, 1.0, 1.0, 3.0]])
    assert m.n_jumps(arr)[0] == 2
    assert m.n_jumps(arr, mask=np.array([True, False, False, False]))[0] == 0
    assert m.n_jumps(arr, tol=1.5)[0] == 1


def test_kkt_detects_violations_by_hand(mini):
    a, x, b = mini
    assert m.kkt_l1(a, b, np.zeros(3), 1.0)[0] == pytest.approx(0.34)            # |g|_max - lambda = 1,34 - 1,0
    assert m.kkt_l1(a, b, np.zeros(3), 1.4)[0] == 0.0
    const = np.full(3, 4.0 / 3.0)
    assert m.kkt_tv(a, b, const, 1.0 / 600.0)[0] == pytest.approx(1.0)           # s = (2, -2): |s| - 1 = 1
    assert m.kkt_tv(a, b, const, 1.0 / 300.0)[0] == pytest.approx(0.0, abs=1e-9)


def test_ista_step_size_is_the_inverse_squared_spectral_norm():
    """A = 2 I: sigma_1^2 = 4, Schrittweite 1/4; ISTA trifft die Lösung 1/2 min |2x - b|^2 + lam |x|_1 = soft(b/2, lam/4) in einem Schritt: bei b = (2, 4) und lam = 0,4
    ist das (0,9, 1,9). Mit 1/sigma_1 (Schrittweite 1/2) käme (1,8, 3,8) heraus."""
    a = 2.0 * np.eye(2)
    b = np.array([2.0, 4.0])
    out = m.iterate_l1(a, b, 0.4, [0, 1], accel=False)
    assert out[1, 0] == pytest.approx([0.9, 1.9])
    assert not out[0].any()


def test_active_set_survives_a_singular_block_by_least_squares():
    """Warmstart mit einer Nullspalte in der aktiven Menge: das System ist singulär, die Ersatzlösung (kleinste Quadrate) gibt (1 - lam, 0)."""
    mm = np.array([[1.0, 0.0], [0.0, 0.0]])
    y = np.array([1.0, 0.0])
    w = m.lasso_active_set(mm, y, 0.1, w0=np.array([0.5, 0.5]))
    assert w == pytest.approx([0.9, 0.0])
