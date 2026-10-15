"""Auswertung: Formatierung, Welten, Analyse der Instanz, Verdikte (Art und Text), Form der Experimente und ihre Reproduzierbarkeit."""

import numpy as np
import pytest

import spt_constants as C
import spt_evaluation as ev


# ----------------------------------------------------------------------------- Formatierung

def test_de_sci_and_share():
    assert ev.de(-1.234, 2) == "−1,23" and ev.sci(2.5e9) == "2,5·10⁹" and ev.sci(3.0e-12) == "3,0·10⁻¹²" and ev.share(0.1525) == "15 %"


def test_err_pct_digits_by_magnitude():
    assert [ev.err_pct(v) for v in (2.58, 0.1188, 0.0399, 0.0087, 8.7e-4, 4471.0, 0.57)] == ["258 %", "11,9 %", "4,0 %", "0,87 %", "0,09 %", "4,5·10³-fach", "57,0 %"]


def test_lam_text_noise_label_and_int_text():
    assert [ev.lam_text(v) for v in (0.5, 0.01, 1e-5, 0.0)] == ["0,500", "0,01", "1,0·10⁻⁵", "0"]
    assert [ev.noise_label(v) for v in C.NOISE_CHOICES] == ["0 %", "0,0001 %", "0,001 %", "0,01 %", "0,1 %", "1 %", "10 %"]
    assert [ev.int_text(v) for v in (7, 8913, 12345678)] == ["7", "8.913", "1,2·10⁷"]


def test_iter_steps_are_powers_of_ten_of_the_slider():
    assert [ev.iter_steps(v) for v in (0, 10, 25, 50)] == [1, 10, 316, 100000]


def test_lambda_choices_are_exact_and_cover_the_presets():
    assert len(C.LAM_CHOICES) == 17 and C.LAM_CHOICES[0] == 1e-8 and C.LAM_CHOICES[-1] == 1.0 and 1e-5 in C.LAM_CHOICES and 1e-2 in C.LAM_CHOICES
    assert C.lam_of(-7) == pytest.approx(10 ** -3.5, rel=1e-5)


# ----------------------------------------------------------------------------- Welten

def test_world_shapes_and_mass():
    for world in C.WORLDS:
        x, g = ev.world_truth(world, 48, 0.03)
        assert x.shape == g.shape == (48,) and np.isfinite(x).all() and np.isfinite(g).all()
    x, g = ev.world_truth("spitzen", 48, 0.03)
    assert (x > 0).sum() == 3 and x.sum() / 48 == pytest.approx(0.12)
    x, g = ev.world_truth("bloecke", 48, 0.03)
    assert set(np.unique(x)) <= {0.0, 0.6, 0.9, 1.0}


# ----------------------------------------------------------------------------- Analyse der Instanz

def test_the_default_settings_are_the_standard_preset():
    assert ev.Settings(**C.PRESETS["Standard"]) == ev.Settings()


def test_the_chosen_solution_matches_prior_and_solver():
    a = ev.analyse(ev.Settings(prior="tv", lam=1e-2))
    assert a.steps == 0 and a.err == pytest.approx(a.err_exact) and a.jumps <= 47
    fi = ev.analyse(ev.Settings(prior="l1", lam=1e-5, solver="fista", steps=25))
    assert fi.steps == 316 and fi.matvec == 632 and fi.err != pytest.approx(fi.err_exact) and not np.allclose(fi.x_hat, fi.x_exact)
    ex = ev.analyse(ev.Settings(prior="l1", lam=1e-5, solver="exakt", steps=25))
    assert ex.steps == 0 and ex.x_hat == pytest.approx(ex.x_exact)
    t = ev.analyse(ev.Settings(prior="tikhonov", solver="fista"))
    assert t.steps == 0                                                             # der Löser wirkt nur bei L1


def test_best_lambda_per_prior_is_the_minimum_of_the_curve():
    a = ev.analyse(ev.Settings())
    for prior in C.PRIORS:
        i = int(np.argmin(a.err_curve[prior]))
        assert a.best_lam[prior] == pytest.approx(a.lams[i]) and a.best_err[prior] == pytest.approx(a.err_curve[prior][i])
        assert a.err_curve[prior].shape == a.lams.shape == a.jumps_curve[prior].shape == a.nnz_curve[prior].shape


def test_tv_jumps_fall_and_l1_support_falls_with_lambda():
    a = ev.analyse(ev.Settings())
    assert a.jumps_curve["tv"][0] > a.jumps_curve["tv"][-1] and a.nnz_curve["l1"][0] > a.nnz_curve["l1"][-1]
    assert a.jumps_curve["tv"][-1] <= 5 and a.nnz_curve["l1"][-1] == 0                  # lambda = 1 liegt über der L1-Schwelle, nahe der TV-Schwelle
    assert a.jumps_curve["tikhonov"].min() >= 40                                       # Tikhonov hat fast alle Differenzen ungleich null


def test_analysis_is_deterministic_and_the_seed_matters():
    a, b, c = ev.analyse(ev.Settings()), ev.analyse(ev.Settings()), ev.analyse(ev.Settings(seed=12))
    assert np.array_equal(a.x_hat, b.x_hat) and a.best_lam == b.best_lam and not np.array_equal(a.b, c.b)


def test_data_kinds_differ_and_crime_is_easier():
    h, c = ev.analyse(ev.Settings(data="ehrlich")), ev.analyse(ev.Settings(data="verbrechen"))
    assert c.best_err["tikhonov"] < h.best_err["tikhonov"]


def test_the_singular_flag_follows_the_condition_number():
    assert ev.analyse(ev.Settings(n=96, w=0.05)).singular and not ev.analyse(ev.Settings()).singular


def test_small_grids_and_all_corners_are_finite():
    for n in (12, 24, 96):
        for w in (0.01, 0.05):
            for noise in (0.0, 1e-1):
                for world in C.WORLDS:
                    for prior in C.PRIORS:
                        a = ev.analyse(ev.Settings(n=n, w=w, noise=noise, world=world, prior=prior, lam=1e-3))
                        assert np.isfinite(a.x_hat).all() and np.isfinite(a.err) and all(np.isfinite(v).all() for v in a.err_curve.values())


# ----------------------------------------------------------------------------- Verdikte

def test_verdict_kinds_and_texts_of_the_presets():
    out = {name: ev.verdict_instance(ev.analyse(ev.Settings(**p))) for name, p in C.PRESETS.items()}
    assert out["Standard"][0] == "success" and out["Spitzen und L1"][0] == "success" and out["Blöcke und TV"][0] == "success"
    assert out["L1 verfehlt"][0] == "warning" and out["FISTA früh"][0] == "warning" and out["Terrassen"][0] == "info"
    assert "passt nicht" in out["L1 verfehlt"][1] and "passt:" in out["Spitzen und L1"][1] and "FISTA nach 316 Schritten" in out["FISTA früh"][1]


# ----------------------------------------------------------------------------- Experimente: Form und Reproduzierbarkeit

def test_prior_world_structure_and_reproducibility():
    r1, r2 = ev.prior_world_experiment(reps=30, reps_small=20), ev.prior_world_experiment(reps=30, reps_small=20)
    assert set(r1["worlds"]) == set(C.WORLDS) and set(r1["worlds"]["phantom"]) == set(C.PRIORS)
    assert [r["noise"] for r in r1["noise"]] == list(C.EXP_NOISE_LEVELS) and set(r1["crime"]) == set(C.PRIORS)
    assert r1["worlds"]["spitzen"]["l1"]["rms"] == r2["worlds"]["spitzen"]["l1"]["rms"]
    assert r1["worlds"]["spitzen"]["l1"]["rms"] < r1["worlds"]["spitzen"]["tikhonov"]["rms"]


def test_terrace_structure():
    r = ev.terrace_experiment(reps=20)
    assert [x["lam"] for x in r["rows"]] == list(C.EXP_TERRACE_LAMS) and set(r["clean"]) == {1e-4, 1e-2} and r["hill_steps"] == 17
    jumps = [x["tv_jumps"] for x in r["rows"]]
    assert jumps[0] > jumps[-1]


def test_iteration_structure():
    r = ev.iteration_experiment(reps=20)
    assert [x["steps"] for x in r["rows"]["fista"]] == list(C.EXP_ITER_STEPS) and set(r["rows"]) == {"ista", "fista"}
    gaps = [x["gap"] for x in r["rows"]["fista"]]
    assert gaps[-1] < gaps[0]


def test_resolution_structure():
    r = ev.resolution_experiment(reps=10)
    assert len(r["rows"]) == len(C.EXP_RES_NOISE) * len(C.EXP_RES_DISTANCES) and all(0.0 <= x["hit"] <= 1.0 for x in r["rows"])


# ----------------------------------------------------------------------------- Schwellen der Verdikte an handgesetzten Werten

def _with_errors(prior, tik, other):
    import dataclasses

    a = ev.analyse(ev.Settings(prior=prior, lam=1e-2))
    return dataclasses.replace(a, best_err={"tikhonov": tik, "l1": other, "tv": other})


def test_int_text_switches_to_powers_of_ten_at_one_million():
    assert ev.int_text(999999) == "999.999" and ev.int_text(10 ** 6) == "1,0·10⁶"


def test_verdict_thresholds_at_the_exact_boundaries():
    # Referenz 1,0: der Prior "passt" nur unter dem 0,8-Fachen, "passt nicht" nur über dem 1,2-Fachen; genau auf der Schwelle gilt "etwa so gut".
    assert ev.verdict_instance(_with_errors("l1", 1.0, 0.79))[0] == "success"
    assert ev.verdict_instance(_with_errors("l1", 1.0, 0.8))[0] == "info"
    assert ev.verdict_instance(_with_errors("l1", 1.0, 1.2))[0] == "info"
    assert ev.verdict_instance(_with_errors("l1", 1.0, 1.21))[0] == "warning"
    # Tikhonov als gewählter Prior: "kein anderer unterbietet deutlich" bis genau 0,8
    assert ev.verdict_instance(_with_errors("tikhonov", 1.0, 0.8))[0] == "success"
    assert ev.verdict_instance(_with_errors("tikhonov", 1.0, 0.79))[0] == "info"


# ----------------------------------------------------------------------------- Grenzen der Formatierung und weitere Verzweigungen

def test_format_boundaries():
    assert ev.err_pct(10.0) == "1,0·10¹-fach" and ev.err_pct(9.99) == "999 %"
    assert ev.err_pct(0.995) == "100 %" and ev.err_pct(0.994) == "99,4 %"
    assert ev.err_pct(0.0095) == "0,9 %" and ev.err_pct(0.0094) == "0,94 %"
    assert ev.err_pct(0.0001) == "0,01 %" and ev.err_pct(0.00009) == "9,0·10⁻³ %"
    assert ev.lam_text(0.1) == "0,100" and ev.lam_text(0.01) == "0,01" and ev.lam_text(0.0099) == "9,9·10⁻³"
    assert ev.sci(0.0) == "0,0" and ev.sci(float("inf")) == "unendlich" and ev.sci(9.96) == "1,0·10¹" and ev.sci(1e5) == "1,0·10⁵"


def test_problem_l2_is_the_squared_spectral_norm():
    p = ev.problem("phantom", 48, 0.03, 1e-3)
    assert p.l2 == pytest.approx(float(np.linalg.norm(p.a, 2)) ** 2) and p.l2 != pytest.approx(p.sv[0], rel=1e-6)


def test_solver_text_only_for_l1_with_an_iterative_solver():
    assert "FISTA" not in ev.verdict_instance(ev.analyse(ev.Settings(prior="tv", lam=1e-2, solver="fista", steps=10)))[1]
    assert "FISTA" in ev.verdict_instance(ev.analyse(ev.Settings(prior="l1", lam=1e-5, solver="fista", steps=10)))[1]


def test_tikhonov_verdict_needs_either_other_prior_to_be_clearly_better():
    import dataclasses

    a = ev.analyse(ev.Settings())
    only_l1 = dataclasses.replace(a, best_err={"tikhonov": 1.0, "l1": 0.5, "tv": 1.0})
    only_tv = dataclasses.replace(a, best_err={"tikhonov": 1.0, "l1": 1.0, "tv": 0.5})
    neither = dataclasses.replace(a, best_err={"tikhonov": 1.0, "l1": 1.0, "tv": 1.0})
    assert ev.verdict_instance(only_l1)[0] == "info" and ev.verdict_instance(only_tv)[0] == "info" and ev.verdict_instance(neither)[0] == "success"


def test_resolution_hits_are_whole_shares_of_the_repetitions():
    r = ev.resolution_experiment(reps=10)
    assert all(abs(x["hit"] * 10 - round(x["hit"] * 10)) < 1e-9 for x in r["rows"]) and any(0.0 < x["hit"] < 1.0 for x in r["rows"])
    assert all(x["tik_rms"] < 1.0 for x in r["rows"]) and all(x["l1_rms"] <= x["tik_rms"] + 1e-12 for x in r["rows"] if x["noise"] <= 0.01)
