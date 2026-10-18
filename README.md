# Sparsity und Total Variation: passt der Prior zur Welt? – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-sparsity-tv-demo.streamlit.app/)**

Sechstes Stück der Linie **Inverse Probleme und Regularisierung** in der „Konzepte“-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning). Es baut auf [schlecht-gestellt-demo](https://github.com/sebastian-hanisch/schlecht-gestellt-demo), [tikhonov-demo](https://github.com/sebastian-hanisch/tikhonov-demo), [tsvd-demo](https://github.com/sebastian-hanisch/tsvd-demo), [parameterwahl-demo](https://github.com/sebastian-hanisch/parameterwahl-demo) und [iterativ-demo](https://github.com/sebastian-hanisch/iterativ-demo) auf und nutzt dieselbe Welt: ein Phantom aus zwei Hügeln und einem Kasten, gemessen mit einer Gauß-Unschärfe und weißem Rauschen. Die Wahrheit ist bekannt, weil das Phantom selbst die Wahrheit ist; sie dient hier nur als Maßstab, nie als Eingabe eines Verfahrens.

## Kernfrage

Bisher bestrafte die Regularisierung die Größe der Lösung (Tikhonov), ihren Rang (TSVD) oder die Schrittzahl. Hier steht Vorwissen über die **Gestalt** der Lösung: wenige von null verschiedene Werte (**L1**) oder wenige Sprünge (**Total Variation**). Wann zahlt sich das aus, und was kostet es, wenn der Prior nicht zur Welt passt?

## Modell

Generator `inverse_generator.py`, wortgleich in allen Stücken der Linie (Phantom, analytische Unschärfe, Gitter, Rauschen, zwei Datenarten):

- **Wahrheit** x(s) auf [0, 1]; **Messung** mit dem Gauß-Kern der Breite w, analytisch bekannt. Standardinstanz n = 48, w = 0,03, Rauschen 0,1 % des Effektivwerts (absolut 3,4·10⁻⁴), Kondition 1,26·10⁴; die direkte Inversion liegt bei 248 %.
- **Drei Welten:** das *Phantom* (zwei glatte Hügel und ein Kasten, die Welt der Linie), *Spitzen* (drei Impulse) und *Blöcke* (drei Kästen). Die beiden Zusatzwelten stehen in diesem Stück, nicht im gemeinsamen Generator; ihre unscharfe Antwort ist ebenfalls analytisch.
- **Daten:** *ehrlich* (analytisch, vom Gitter unabhängig) oder *aus demselben Gitter* (b = A x + e, „inverse crime“).
- **Aufgaben:** L1 ½‖Ax − b‖² + λ‖x‖₁, TV ½‖Ax − b‖² + λ‖Dx‖₁ mit der Vorwärtsdifferenz D; Tikhonov wie in Stück 2.
- **Löser:** L1 und TV **exakt** per Aktivmengenverfahren (TV über die Sprungkoordinaten, der Mittelwert ist herausprojiziert); dazu ISTA und FISTA als die Iterationen, die die Linie zeigt.

Alles ist erfunden; die Wahrheit ist das simulierte Phantom.

## Methodik

- **Maßstab:** der beste λ je Prior auf einem Gitter (RMS über feste Wiederholungen) und der beste λ der einzelnen Stichprobe; Tikhonov ist die Referenz. Das Diskrepanzprinzip (größtes λ mit Residuum² ≤ σ²n) steht als Wahl aus den Daten daneben.
- **Experimente** laufen über feste Zufallsströme (SplitMix64): 200 Wiederholungen je Welt und Prior, 100 je Zelle bei Rauschstufen, Daten aus demselben Gitter und Auflösung; nie ein einzelner Seed. Die Primäransicht ist eine einzelne Stichprobe und sagt nichts über die Häufigkeit.
- **Optimalität:** die KKT-Bedingungen werden an jeder exakten Lösung geprüft (Verletzung unter 10⁻⁸ bei L1 und TV).

## Befunde (gemessen, keine Behauptungen)

Standardinstanz (n = 48, w = 0,03, Rauschen 0,1 %, ehrliche Daten), wenn nicht anders genannt. Jede Zahl steht in `tests/test_claims.py`.

| Frage | Befund | Test |
|---|---|---|
| Wer gewinnt in welcher Welt (Experiment 1, bester λ, RMS)? | **Phantom:** Tikhonov **11,7 %**, TV 12,8 %, L1 34,9 %. **Spitzen:** Tikhonov 35,7 %, TV 2,0 %, L1 **0,09 %**. **Blöcke:** Tikhonov 20,4 %, L1 21,3 %, TV **16,0 %**. Direkte Inversion 248 / 121 / 241 %. Jeder Prior gewinnt nur in der Welt, zu der er passt. | `test_prior_world_numbers`, `test_each_prior_wins_only_in_its_world` |
| Rauschstufen (Phantom, 100 Wdh.)? | Rauschen 10 / 1 / 0,1 / 0,01 %: Tikhonov 19,3 / 13,3 / 11,7 / 11,5 %; **L1 74,5 / 61,1 / 32,2 / 13,0 %**; TV 22,8 / 16,3 / 12,8 / 11,5 %. Daten aus demselben Gitter: Tikhonov 7,1 %, **TV 6,5 %**, L1 29,8 %. | `test_noise_and_crime_numbers` |
| Diskrepanzprinzip? | Phantom: Tikhonov Median 11,8 %, TV 13,2 %, 0 % gescheitert. In der Welt der Spitzen scheitert es bei TV in 22 % der Wiederholungen (Fehler über dem Doppelten des Optimums). | `test_discrepancy_principle_numbers` |
| Terrassen (Experiment 2)? | Sprünge im Hügelbereich (17 Gitterschritte) und Gipfel von Hügel 1 (Wahrheit 1,00): λ = 10⁻⁶: 13,0 Sprünge, Gipfel 0,97, Fehler 13,0 %; 10⁻²: **7,0 Sprünge, Gipfel 0,88**, Fehler 19,5 %; 10⁻¹: 5,0 Sprünge, 0,77, 33 %. Tikhonov bei 10⁻²: Gipfel 1,00, Fehler 11,7 %. Rauschfrei bleiben bei λ = 10⁻² 14 Sprünge, der Gipfel 0,88. | `test_terrace_numbers`, `test_noise_free_terraces` |
| Verfahren gegen Problem (Experiment 3, L1, λ = 10⁻⁵)? | Exakter Minimierer: **34,9 %**. **FISTA** nach 10 / 100 / 300 / 1.000 / 3.000 / 30.000 Schritten: 14,6 / 11,6 / **9,7** / 11,6 / 21,3 / 34,9 %; **ISTA** 14,9 / 13,5 / 12,7 / 11,8 / 11,0 / 9,7 %. Die Zielfunktionslücke fällt dabei von 2·10⁻⁴ auf unter 10⁻⁹. Das Verfahren ist hier der bessere Regularisierer als das Problem. | `test_iteration_numbers` |
| Auflösung zweier Spitzen (Experiment 4, beide gefunden in 100 Wdh.)? | Rauschen 0,1 %: **100 %** in jedem Abstand von 1 bis 6 Gitterpunkten (Tikhonov 12 bis 49 % Fehler). 3 %: Abstand 2 **72 %**, sonst 100 %. 10 %: Abstand 1 100 %, **2: 24 %, 3: 60 %, 4: 87 %, 6: 96 %**. Abstand 2 ist am schwersten (mittlere Gitterzelle), nicht Abstand 1. | `test_resolution_numbers` |
| Einzelstichproben (Presets) | Standard (Seed 11): 11,8 %. Spitzen und L1: 0,13 %. Blöcke und TV: 16,0 %. Terrassen (Phantom, TV, 10⁻²): 19,5 %. L1 verfehlt: 30,9 %. FISTA früh (316 Schritte): 10,1 % statt 30,9 % beim exakten Minimierer. | `test_preset_help_numbers` |
| Mini-Instanz von Hand | 3 × 3-Matrix: L1 ist null ab λ_max = 1,34, bei λ = 1,3 nur x₂ = 0,1176 aktiv. TV ist konstant 4/3 ab λ = 1/300 und hat darunter d = 1 − 300λ (bei 1/600: (7/6, 5/3, 7/6)). Entrauschen: TV (λ = 0,01) gibt (1,31, 1,38, 1,31), L1 (λ = 0,5) gibt (0,8, 0,9, 0,8). | `tests/test_model.py` |

## Befunde und Korrekturen gegenüber dem Plan

- **Auf dem glatten Phantom verlieren beide Priors.** Die Gliederung nannte „Gaußsche Lösung als Gegenprobe“; die Messung zeigt, dass die Gaußsche Lösung hier die beste ist. Die Dünnbesetztheit zahlt sich nur in den Zusatzwelten aus; ohne sie wäre das Stück eine Reihe von Niederlagen.
- **Die ersten 18 % für L1 kamen von der Abbruchzahl.** Die Vormessung mit festen 3.000 FISTA-Schritten gab für L1 18 %; die exakten Lösungen zeigen 35 %. Der Fehler eines Verfahrens ist nicht der des Problems; das Stück trennt beides (Experiment 3).
- **Terrassen sind bei dem besten λ mild.** Dort hat TV 12,6 von 17 möglichen Sprüngen im Hügelbereich; deutlich werden sie ab λ ≥ 10⁻³ und rauschfrei.
- **Auflösung nicht monoton im Abstand.** Der Abstand 2 ist am schwersten (Gitterartefakt der mittleren Zelle); er steht unverändert in der Tabelle.
- **Ein Vorzeichenfehler in der TV-KKT-Prüfung** zeigte in der Vormessung überall „2,0“: Die Prüfung, nicht der Löser, war falsch (s = −cumsum statt +cumsum). Die von Hand gerechnete Mini-Instanz (Verletzung 1,0 bei λ = 1/600 an der konstanten Lösung) legt die Konvention nun fest.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Prior passt zur Welt** | Auf dem Phantom liegt L1 bei 34,9 % und TV bei 12,8 % gegen 11,7 % bei Tikhonov. | Bayes-Sicht (Stück 7) |
| **Die Daten sind rauscharm genug** | Bei 10 % Rauschen liegt L1 bei 74,5 %. | – |
| **Das Problem wird exakt gelöst** | ISTA und FISTA liefern je nach Schrittzahl andere Lösungen; der Fehler hängt stärker an der Schrittzahl als an der Zielfunktion. | Stück 5, Stück 14 |
| **Eindimensional, Pixelbasis, anisotrop** | Kein zweidimensionales TV, keine Wavelet-Basis (n ist keine Zweierpotenz; Wavelets gehören in die Spektral-Linie). | – |
| **Ein Phantom, zwei Zusatzwelten, eine Matrix** | Die Zahlen gelten für diese erfundenen Beispiele; die Zusatzwelten sind so gebaut, dass der Prior passt. | – |
| **σ ist bekannt (Diskrepanzprinzip)** | GCV und L-Kurve sind für nichtquadratische Strafen nicht einfach übertragbar (kein df in geschlossener Form) und nicht gemessen. | Stück 4 |
| **Das Orakel kennt die Wahrheit** | Der beste λ und die Auswahl je Zelle sind in der Praxis nicht verfügbar. | – |

Die Zahlen gelten für diese erfundene Welt und zeigen nichts über echte Messungen.

## Bezug zu den Nachbarn im Portfolio

- [iterativ-demo](https://github.com/sebastian-hanisch/iterativ-demo): Stück 5; ISTA ist Landweber mit weicher Schwelle, und die Schrittzahl regularisiert dort wie hier.
- [parameterwahl-demo](https://github.com/sebastian-hanisch/parameterwahl-demo): Stück 4; das Diskrepanzprinzip kehrt hier für L1 und TV zurück.
- [tikhonov-demo](https://github.com/sebastian-hanisch/tikhonov-demo): Stück 2; der Maßstab dieses Stücks.

## Tests

Mini-Instanz von Hand (3 × 3: weiche Schwelle, λ_max, Aktivmenge, TV-Entwurf, Entrauschen, ISTA und FISTA in den ersten Schritten, KKT-Verletzung), KKT-Bedingungen der exakten Lösungen auf n = 12 bis 48 in drei Welten, L1 gegen `scipy.optimize.minimize` (L-BFGS-B, aufgespaltene Form) und einen unabhängigen Koordinatenabstieg, TV gegen ein geglättetes Ersatzproblem und gegen ein duales projiziertes Gradientenverfahren (Entrauschen), zufällige Störungen ohne Verbesserung, die Konvergenzschranken O(1/k) und O(1/k²), die Grenzfälle λ → 0 und λ über der Schwelle, die Zusatzwelten gegen eine numerische Faltung, Generator, Auswertung, jede README-Zahl in `test_claims.py`, AppTest (Presets, Regler-Grenzen, jede Welt, jeder Prior und Löser, jede Rauschstufe, PDF, Permalink), Footer, Presets und Laufzeit-Importe.

```bash
python -m pytest tests -q
```

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `inverse_generator.py` | gemeinsamer Generator der Linie (SplitMix64, Phantom, analytische Unschärfe, Operator, Daten) |
| `spt_model.py` | weiche Schwelle, Aktivmengenlöser für L1 und TV, ISTA/FISTA, KKT-Prüfung |
| `spt_evaluation.py` | Welten, Kennzahlen, Experimente, Verdikte |
| `spt_visualization.py`, `spt_pdf_export.py` | Figuren, PDF |
| `spt_constants.py`, `spt_presets.py` | Regler, Presets, Permalink |
| `tests/` | Mini-Instanz, Orakel, Claims, App |

## Bewusst nicht umgesetzt

- **Zweidimensionales TV und Bildverarbeitung:** nicht gebaut (eindimensional).
- **Wavelet- und andere Dünnbesetztheitsbasen:** gehören in die Spektral-Linie (n ist hier keine Zweierpotenz).
- **Isotropes TV, Elastic Net, nichtkonvexe Strafen (Lp, SCAD):** ändern die Aussage nicht.
- **GCV und L-Kurve für nichtquadratische Strafen:** nicht übertragen, nicht gemessen.
- **Praxisdaten:** nicht gebaut; alle Zahlen sind erzeugt.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python 3.12.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Inverse Probleme: vom Rauschen zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-inverse-probleme.html).
