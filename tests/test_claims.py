"""Aussagen aus README, Verdikten und Hilfetexten, einzeln an den festen Strömen gemessen (Seeds ab SEED_BASE). Ändert sich eine Zahl, muss der Text mit."""

import numpy as np
import pytest

import spt_constants as C
import spt_evaluation as ev


@pytest.fixture(scope="module")
def prior_world():
    return ev.prior_world_experiment()


@pytest.fixture(scope="module")
def terrace():
    return ev.terrace_experiment()


@pytest.fixture(scope="module")
def iteration():
    return ev.iteration_experiment()


@pytest.fixture(scope="module")
def resolution():
    return ev.resolution_experiment()


# ----------------------------------------------------------------------------- Instanz und Presets

def test_standard_instance_numbers():
    a = ev.analyse(ev.Settings())
    assert 1.2e4 < a.cond < 1.3e4 and a.sigma == pytest.approx(3.4e-4, rel=0.02)
    assert a.err == pytest.approx(0.118, abs=0.003) and a.best_lam["tikhonov"] == pytest.approx(0.01) and a.err_naive > 2.0
    assert a.jumps == 47 and a.jumps_hill == 17


def test_preset_help_numbers():
    res = {name: ev.analyse(ev.Settings(**p)) for name, p in C.PRESETS.items()}
    assert round(100 * res["Standard"].err) == 12
    assert res["Spitzen und L1"].err == pytest.approx(0.0013, abs=0.0006) and res["Spitzen und L1"].best_err["tikhonov"] > 0.3
    assert res["Blöcke und TV"].err == pytest.approx(0.160, abs=0.004) and res["Blöcke und TV"].err < res["Blöcke und TV"].best_err["tikhonov"]
    assert res["Terrassen"].err == pytest.approx(0.195, abs=0.01) and res["Terrassen"].jumps_hill == 7
    assert res["L1 verfehlt"].err == pytest.approx(0.309, abs=0.01)
    assert res["FISTA früh"].err == pytest.approx(0.101, abs=0.01) and res["FISTA früh"].err_exact == pytest.approx(0.309, abs=0.01) and res["FISTA früh"].steps == 316


# ----------------------------------------------------------------------------- Experiment 1

def test_prior_world_numbers(prior_world):
    w = prior_world["worlds"]
    assert w["phantom"]["tikhonov"]["rms"] == pytest.approx(0.117, abs=0.003) and w["phantom"]["tikhonov"]["lam"] == pytest.approx(0.01, rel=0.01)
    assert w["phantom"]["l1"]["rms"] == pytest.approx(0.349, abs=0.01) and w["phantom"]["tv"]["rms"] == pytest.approx(0.128, abs=0.004)
    assert w["spitzen"]["tikhonov"]["rms"] == pytest.approx(0.357, abs=0.01) and w["spitzen"]["l1"]["rms"] < 0.002 and w["spitzen"]["tv"]["rms"] == pytest.approx(0.020, abs=0.004)
    assert w["bloecke"]["tikhonov"]["rms"] == pytest.approx(0.204, abs=0.005) and w["bloecke"]["l1"]["rms"] == pytest.approx(0.213, abs=0.006)
    assert w["bloecke"]["tv"]["rms"] == pytest.approx(0.160, abs=0.005)
    for world, naive in (("phantom", 2.48), ("spitzen", 1.21), ("bloecke", 2.41)):
        assert prior_world["naive"][world] == pytest.approx(naive, abs=0.05)


def test_each_prior_wins_only_in_its_world(prior_world):
    w = prior_world["worlds"]
    best = {world: min(C.PRIORS, key=lambda p: w[world][p]["rms"]) for world in C.WORLDS}
    assert best == {"phantom": "tikhonov", "spitzen": "l1", "bloecke": "tv"}


def test_noise_and_crime_numbers(prior_world):
    rows = {r["noise"]: r for r in prior_world["noise"]}
    assert rows[0.1]["l1"]["rms"] == pytest.approx(0.745, abs=0.04) and rows[0.1]["tikhonov"]["rms"] == pytest.approx(0.193, abs=0.01)
    assert rows[0.01]["l1"]["rms"] > 0.5 and rows[1e-4]["l1"]["rms"] == pytest.approx(0.130, abs=0.01)
    assert rows[1e-4]["tv"]["rms"] == pytest.approx(rows[1e-4]["tikhonov"]["rms"], abs=0.004)
    assert all(r["tikhonov"]["rms"] < r["tv"]["rms"] + 0.003 for r in prior_world["noise"][:3])
    crime = prior_world["crime"]
    assert crime["tv"]["rms"] < crime["tikhonov"]["rms"] < crime["l1"]["rms"] and crime["tv"]["rms"] == pytest.approx(0.065, abs=0.005)


def test_discrepancy_principle_numbers(prior_world):
    w = prior_world["worlds"]
    assert w["phantom"]["tikhonov"]["dp_median"] == pytest.approx(0.118, abs=0.004) and w["phantom"]["tikhonov"]["dp_fail"] == 0.0
    assert w["phantom"]["tv"]["dp_median"] == pytest.approx(0.132, abs=0.006)
    assert 0.1 < w["spitzen"]["tv"]["dp_fail"] < 0.35


# ----------------------------------------------------------------------------- Experiment 2

def test_terrace_numbers(terrace):
    by = {r["lam"]: r for r in terrace["rows"]}
    assert terrace["hill_steps"] == 17 and terrace["peak_truth"] == pytest.approx(0.999, abs=0.002)
    assert by[1e-6]["tv_jumps"] == pytest.approx(13.0, abs=0.5) and by[1e-2]["tv_jumps"] == pytest.approx(7.0, abs=0.3) and by[1e-1]["tv_jumps"] == pytest.approx(5.0, abs=0.3)
    assert by[1e-6]["tv_peak"] == pytest.approx(0.966, abs=0.01) and by[1e-2]["tv_peak"] == pytest.approx(0.880, abs=0.01) and by[1e-1]["tv_peak"] == pytest.approx(0.768, abs=0.02)
    assert by[1e-2]["tik_peak"] == pytest.approx(0.998, abs=0.003) and by[1e-2]["tik_rms"] == pytest.approx(0.117, abs=0.003)
    assert by[1e-6]["tv_rms"] == pytest.approx(0.130, abs=0.004) and by[1e-2]["tv_rms"] == pytest.approx(0.195, abs=0.006)


def test_noise_free_terraces(terrace):
    x = terrace["clean"][1e-2]
    steps = np.abs(np.diff(x))
    assert int((steps > 1e-9).sum()) == 14 and x.max() == pytest.approx(0.88, abs=0.01)


# ----------------------------------------------------------------------------- Experiment 3

def test_iteration_numbers(iteration):
    assert iteration["lam"] == pytest.approx(1e-5, rel=0.01) and iteration["exact_rms"] == pytest.approx(0.349, abs=0.01)
    fista = {r["steps"]: r for r in iteration["rows"]["fista"]}
    ista = {r["steps"]: r for r in iteration["rows"]["ista"]}
    assert fista[300]["rms"] == pytest.approx(0.097, abs=0.006) and fista[100]["rms"] == pytest.approx(0.116, abs=0.006)
    assert fista[30000]["rms"] == pytest.approx(iteration["exact_rms"], abs=0.01) and fista[3000]["rms"] == pytest.approx(0.213, abs=0.02)
    assert ista[300]["rms"] == pytest.approx(0.127, abs=0.006) and ista[30000]["rms"] == pytest.approx(0.097, abs=0.006)
    gaps = [r["gap"] for r in iteration["rows"]["fista"]]
    assert gaps[0] > 1e-5 and gaps[-1] < 1e-8 and fista[300]["rms"] < iteration["tik_rms"]


# ----------------------------------------------------------------------------- Experiment 4

def test_resolution_numbers(resolution):
    rows = {(r["noise"], r["d"]): r for r in resolution["rows"]}
    assert all(rows[(1e-3, d)]["hit"] == 1.0 and rows[(1e-3, d)]["l1_rms"] < 0.005 for d in C.EXP_RES_DISTANCES)
    assert rows[(1e-3, 2)]["tik_rms"] > 0.4
    assert rows[(0.03, 2)]["hit"] == pytest.approx(0.72, abs=0.1) and rows[(0.03, 3)]["hit"] > 0.95
    assert rows[(0.1, 1)]["hit"] > 0.95 and rows[(0.1, 2)]["hit"] == pytest.approx(0.24, abs=0.1) and rows[(0.1, 3)]["hit"] == pytest.approx(0.60, abs=0.1)
    assert rows[(0.1, 6)]["hit"] > 0.9 and rows[(0.1, 2)]["hit"] < rows[(0.1, 1)]["hit"]


def test_verdict_texts_carry_the_numbers(prior_world, terrace, iteration, resolution):
    assert "Phantom: Tikhonov 11,7 %, L1 34,9 %, TV 12,8 %" in ev.verdict_prior_world(prior_world)[1]
    assert "Spitzen: Tikhonov 35,7 %, L1 0,09 %, TV 2,0 %" in ev.verdict_prior_world(prior_world)[1]
    t = ev.verdict_terrace(terrace)[1]
    assert "17 Gitterschritte" in t and "7,0" in t and "0,88" in t
    i = ev.verdict_iteration(iteration)[1]
    assert "34,9 %" in i and "nach 300 Schritten 9,7 %" in i
    assert "nur 24 %" in ev.verdict_resolution(resolution)[1] or "nur 25 %" in ev.verdict_resolution(resolution)[1]


def test_iteration_distance_to_the_exact_minimizer(iteration):
    fista = {r["steps"]: r for r in iteration["rows"]["fista"]}
    assert fista[10]["dist"] > 0.3 and fista[30000]["dist"] < 0.1 and fista[3000]["dist"] > fista[30000]["dist"]
