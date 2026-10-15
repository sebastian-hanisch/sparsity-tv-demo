"""Konstanten: Regler-Grenzen, feste Annahmen, Experimente, Farben und Presets."""

# Regler. Rechenaufwand am Maximum: exakter L1- und TV-Löser (Aktivmenge, höchstens n Koordinaten, je Schritt ein lineares System der Größe <= n) und FISTA bis
# 10^5 Schritte auf einer Zeile (1 s); die Experimente nutzen feste Ströme mit 200 Wiederholungen.
N_MIN, N_MAX, N_STEP, DEFAULT_N = 12, 96, 4, 48
W_MIN, W_MAX, W_STEP, DEFAULT_W = 0.01, 0.05, 0.005, 0.03        # Breite der Unschärfe (Anteil der Länge 1)
NOISE_CHOICES = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)        # Rauschen relativ zum Effektivwert der exakten Daten
DEFAULT_NOISE = 1e-3
DATA_CHOICES = ("ehrlich", "verbrechen")
DATA_LABELS = {"ehrlich": "Ehrliche Daten (analytisch, vom Gitter unabhängig)", "verbrechen": "Daten aus demselben Gitter (inverse crime)"}
WORLDS = ("phantom", "spitzen", "bloecke")
WORLD_LABELS = {"phantom": "Phantom (zwei glatte Hügel und ein Kasten)", "spitzen": "Spitzen (drei Impulse)", "bloecke": "Blöcke (drei Kästen)"}
WORLD_SHORT = {"phantom": "Phantom", "spitzen": "Spitzen", "bloecke": "Blöcke"}
PRIORS = ("tikhonov", "l1", "tv")
PRIOR_LABELS = {"tikhonov": "Tikhonov (quadratische Strafe, Stück 2)", "l1": "L1 (Summe der Beträge der Werte)", "tv": "Total Variation (Summe der Beträge der Sprünge)"}
PRIOR_SHORT = {"tikhonov": "Tikhonov", "l1": "L1", "tv": "TV"}
SOLVERS = ("exakt", "ista", "fista")
SOLVER_LABELS = {"exakt": "exakt (Aktivmenge)", "ista": "ISTA (Schwellenwert-Iteration)", "fista": "FISTA (beschleunigt)"}
SOLVER_SHORT = {"exakt": "exakt", "ista": "ISTA", "fista": "FISTA"}
# λ in halben Zehnerpotenzen von 1e-8 bis 1 (Tikhonov nutzt λ wie in Stück 2: Filter σ²/(σ² + λ²); L1 und TV: ½‖Ax − b‖² + λ·Strafe)


def lam_of(half_decades: int) -> float:
    """λ = 10^(half_decades / 2), auf sechs Stellen gerundet (damit Permalink und Voreinstellungen den Auswahlwert exakt treffen)."""
    return float(f"{10.0 ** (half_decades / 2.0):.6g}")


LAM_CHOICES = tuple(lam_of(k) for k in range(-16, 1))
DEFAULT_LAM = 1e-2
STEP_MIN, STEP_MAX, STEP_STEP, DEFAULT_STEPS = 0, 50, 1, 25       # Regler: ISTA und FISTA 10^(Regler/10) Schritte (1 bis 10^5)
STEP_DECADE = 10
SEED_MAX, DEFAULT_SEED = 999999, 11

SINGULAR_COND = 1e15                                              # ab hier ist die Matrix in doppelter Genauigkeit numerisch singulär
SHOW_RANGE = (-1.5, 2.0)                                          # Wertebereich der Rekonstruktion im Diagramm
FAIL_FACTOR = 2.0                                                 # „gescheitert“: Fehler über dem Doppelten des besten Fehlers der Reihe
JUMP_TOL = 1e-9                                                   # ab hier zählt eine Differenz als Sprung
SUPPORT_TOL = 0.1                                                 # Träger: Werte über 10 % der Spitzenhöhe

# Welten: Spitzen (Mitte, Masse) und Blöcke (links, rechts, Höhe); das Phantom kommt aus dem gemeinsamen Generator.
SPIKES = ((0.25, 0.04), (0.50, 0.03), (0.70, 0.05))
BLOCKS = ((0.15, 0.30, 1.0), (0.42, 0.52, 0.6), (0.70, 0.85, 0.9))
HILL_RANGE = (0.10, 0.45)                                         # Hügel 1 des Phantoms: hier zählen die Sprünge für die Terrassen

# Experimente: feste Ströme (Seeds aus SEED_BASE), nie ein einzelner Seed als Befund.
SEED_BASE = 990000
EXP_REPS = 200
EXP_REPS_SMALL = 100                                              # Rauschstufen, Daten aus demselben Gitter und Auflösung: je Zelle 100 feste Wiederholungen
LAM_GRID = (-8.0, -1.0, 15)                                       # Zehnerlogarithmen von 1e-8 bis 1e-1 in halben Zehnerpotenzen (L1 und TV)
TIK_GRID = (-6.0, 0.0, 61)                                        # Tikhonov
EXP_NOISE_LEVELS = (1e-1, 1e-2, 1e-3, 1e-4)                       # Experiment 1 (Phantom)
EXP_TERRACE_LAMS = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)    # Experiment 2
EXP_ITER_STEPS = (10, 30, 100, 300, 1000, 3000, 10000, 30000)     # Experiment 3
EXP_RES_DISTANCES = (1, 2, 3, 4, 6)                               # Experiment 4: Abstand der Spitzen in Gitterpunkten
EXP_RES_NOISE = (1e-3, 1e-2, 3e-2, 1e-1)
EXP_RES_LAMS = (-5.0, -1.0, 9)

COLOR_TRUTH = "#2F6B65"
COLOR_NAIVE = "#c0392b"
COLOR_TIK = "#555555"
COLOR_L1 = "#D68A2E"
COLOR_TV = "#4c78a8"
COLOR_GREY = "#8a8a8a"
PRIOR_COLORS = {"tikhonov": COLOR_TIK, "l1": COLOR_L1, "tv": COLOR_TV}

PRESETS = {
    "Standard": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "data": "ehrlich", "world": "phantom", "prior": "tikhonov", "lam": 1e-2, "solver": "exakt",
                 "steps": DEFAULT_STEPS, "seed": DEFAULT_SEED},
    "Spitzen und L1": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "data": "ehrlich", "world": "spitzen", "prior": "l1", "lam": lam_of(-7), "solver": "exakt",
                       "steps": DEFAULT_STEPS, "seed": DEFAULT_SEED},
    "Blöcke und TV": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "data": "ehrlich", "world": "bloecke", "prior": "tv", "lam": 1e-2, "solver": "exakt",
                      "steps": DEFAULT_STEPS, "seed": DEFAULT_SEED},
    "Terrassen": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "data": "ehrlich", "world": "phantom", "prior": "tv", "lam": 1e-2, "solver": "exakt",
                  "steps": DEFAULT_STEPS, "seed": DEFAULT_SEED},
    "L1 verfehlt": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "data": "ehrlich", "world": "phantom", "prior": "l1", "lam": 1e-5, "solver": "exakt",
                    "steps": DEFAULT_STEPS, "seed": DEFAULT_SEED},
    "FISTA früh": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "data": "ehrlich", "world": "phantom", "prior": "l1", "lam": 1e-5, "solver": "fista",
                   "steps": 25, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standard": "Phantom mit Tikhonov und λ = 0,01: der Fehler liegt bei rund 12 %; das ist der Maßstab der Linie.",
    "Spitzen und L1": "Drei Spitzen, L1: die Spitzen werden fast exakt wiedergefunden, Tikhonov bleibt bei rund einem Drittel Fehler.",
    "Blöcke und TV": "Drei Kästen, TV: die Kanten bleiben scharf, der Fehler liegt unter dem von Tikhonov.",
    "Terrassen": "Phantom mit TV und λ = 0,01: die glatten Hügel zerfallen in Stufen und werden an der Spitze abgeschnitten.",
    "L1 verfehlt": "Phantom mit L1: der Prior passt nicht, der exakte Minimierer liegt bei rund einem Drittel Fehler.",
    "FISTA früh": "Dieselbe L1-Aufgabe, FISTA nach 316 Schritten: der Fehler ist deutlich kleiner als beim exakten Minimierer, weil die Schrittzahl mitregularisiert.",
}
