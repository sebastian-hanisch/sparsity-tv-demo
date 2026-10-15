"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Reglergrenzen."""

import pytest

import spt_constants as C
import spt_evaluation as ev
import spt_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_widget_bounds_and_on_the_step_grid():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and (p["n"] - C.N_MIN) % C.N_STEP == 0, name
        assert C.W_MIN <= p["w"] <= C.W_MAX and (p["w"] - C.W_MIN) / C.W_STEP == pytest.approx(round((p["w"] - C.W_MIN) / C.W_STEP)), name
        assert p["noise"] in C.NOISE_CHOICES and p["data"] in C.DATA_CHOICES and p["world"] in C.WORLDS and p["prior"] in C.PRIORS, name
        assert p["lam"] in C.LAM_CHOICES and p["solver"] in C.SOLVERS and C.STEP_MIN <= p["steps"] <= C.STEP_MAX and 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standard"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_every_default_is_a_valid_widget_value():
    for key, spec in P.SETTING_SPECS.items():
        if spec.choices is not None:
            assert spec.default in spec.choices, key
        else:
            assert spec.lo <= spec.default <= spec.hi, key


def test_the_presets_show_the_documented_outcomes():
    res = {name: ev.analyse(ev.Settings(**p)) for name, p in C.PRESETS.items()}
    assert 0.10 < res["Standard"].err < 0.13
    assert res["Spitzen und L1"].err < 0.01 and res["Spitzen und L1"].best_err["tikhonov"] > 0.3
    assert res["Blöcke und TV"].err < res["Blöcke und TV"].best_err["tikhonov"]
    assert res["Terrassen"].jumps_hill < 10 and res["Terrassen"].peak_hill < 0.92 < res["Terrassen"].peak_truth
    assert 0.28 < res["L1 verfehlt"].err < 0.34
    assert res["FISTA früh"].err < 0.5 * res["FISTA früh"].err_exact and res["FISTA früh"].steps == 316


def test_experiment_constants_are_consistent():
    assert C.EXP_REPS >= 200 and C.EXP_REPS_SMALL >= 100 and C.FAIL_FACTOR == 2.0 and C.DEFAULT_NOISE in C.EXP_NOISE_LEVELS
    assert 1e-2 in C.EXP_TERRACE_LAMS and max(C.EXP_ITER_STEPS) == 30000 and 2 in C.EXP_RES_DISTANCES
    assert C.STEP_DECADE * 5 == C.STEP_MAX
