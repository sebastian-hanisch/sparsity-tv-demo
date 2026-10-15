"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Alle Regler werden immer gezeichnet; der Löser und der Schrittregler wirken nur bei L1 (steht im Hilfetext), es gibt deshalb kein KEPT-Muster und keine
berechneten Grenzen. λ und die Schrittzahl laufen in festen Stufen (halbe Zehnerpotenzen).
"""

import random
from dataclasses import dataclass

import streamlit as st

import spt_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


SETTING_SPECS = {
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "w_slider": SettingSpec("w", float, C.DEFAULT_W, C.W_MIN, C.W_MAX),
    "noise_select": SettingSpec("r", float, C.DEFAULT_NOISE, choices=C.NOISE_CHOICES),
    "data_select": SettingSpec("d", str, "ehrlich", choices=C.DATA_CHOICES),
    "world_select": SettingSpec("world", str, "phantom", choices=C.WORLDS),
    "prior_select": SettingSpec("prior", str, "tikhonov", choices=C.PRIORS),
    "lam_select": SettingSpec("lam", float, C.DEFAULT_LAM, choices=C.LAM_CHOICES),
    "solver_select": SettingSpec("solver", str, "exakt", choices=C.SOLVERS),
    "steps_slider": SettingSpec("steps", int, C.DEFAULT_STEPS, C.STEP_MIN, C.STEP_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {"n": "n_slider", "w": "w_slider", "noise": "noise_select", "data": "data_select", "world": "world_select", "prior": "prior_select",
               "lam": "lam_select", "solver": "solver_select", "steps": "steps_slider", "seed": "seed_input"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf gültige Auswahlwerte geprüft."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = spec.caster(qp[spec.url_param])
        except (ValueError, TypeError):
            continue
        if spec.choices is not None and value not in spec.choices:
            continue
        if spec.lo is not None:
            value = max(spec.lo, value)
        if spec.hi is not None:
            value = min(spec.hi, value)
        st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
