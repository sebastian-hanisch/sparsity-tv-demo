"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini():
    """3 x 3-Matrix aus Stück 1 bis 5 (symmetrisch, Zeilensumme 1, Singulärwerte 1, 0,5, 0,1), Wahrheit x = (1, 2, 1), rauschfreie Daten b = A x = (1,3, 1,4, 1,3).
    Handrechnung: A^T b = (1,33, 1,34, 1,33), (A^T A)_22 = 0,34, (A^T A)_12 = 0,33.
    L1: lambda_max = 1,34; bei lambda = 1,3 ist nur x_2 aktiv, x_2 = (1,34 - 1,3)/0,34 = 0,117647, und |g_1| = |0,33 x_2 - 1,33| = 1,2912 <= 1,3.
    TV: konstante Lösung 4/3 für lambda >= 1/300; darunter ist x = (u, v, u) mit d = v - u = 1 - 300 lambda und u = (4 - d)/3, z. B. lambda = 1/600:
    d = 0,5, x = (7/6, 5/3, 7/6); lambda -> 0 gibt (1, 2, 1).
    Entrauschen (A = I) mit b: L1 weiche Schwelle (lambda = 0,5 gibt 0,8, 0,9, 0,8); TV mit lambda = 0,01 gibt (1,31, 1,38, 1,31), ab lambda = 1/30 alles 4/3.
    ISTA (Schrittweite 1, weil sigma_1 = 1) mit lambda = 0,3: x_1 = soft(A^T b, 0,3) = (1,03, 1,04, 1,03), x_2 = (1,0267, 1,0466, 1,0267)."""
    a = np.array([[0.6, 0.3, 0.1], [0.3, 0.4, 0.3], [0.1, 0.3, 0.6]])
    x = np.array([1.0, 2.0, 1.0])
    b = np.array([1.3, 1.4, 1.3])
    return a, x, b
