"""Sparsity und Total Variation: L1, TV und die Frage, ob der Prior zur Welt passt - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Sechstes Stück der Linie Inverse Probleme und Regularisierung. Auf der Welt aus Stück 1 bis 5 (Phantom, Gauß-Unschärfe, Rauschen) und zwei Zusatzwelten (Spitzen,
Blöcke) stehen drei Strafen gegeneinander: die quadratische von Tikhonov, L1 (Dünnbesetztheit) und Total Variation (stückweise konstant). Kern: spt_model.py,
Generator: inverse_generator.py (wortgleich in allen Repos der Linie).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import spt_constants as C
from spt_evaluation import (
    Settings,
    analyse,
    de,
    err_pct,
    int_text,
    iter_steps,
    iteration_experiment,
    lam_text,
    noise_label,
    prior_world_experiment,
    resolution_experiment,
    sci,
    share,
    terrace_experiment,
    verdict_instance,
    verdict_iteration,
    verdict_prior_world,
    verdict_resolution,
    verdict_terrace,
)
from spt_pdf_export import build_pdf
from spt_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from spt_visualization import (
    build_error_chart,
    build_iteration_chart,
    build_noise_chart,
    build_prior_world_chart,
    build_recon_chart,
    build_resolution_chart,
    build_structure_chart,
    build_table,
    build_terrace_chart,
    build_terrace_shape_chart,
    style_table,
)

st.set_page_config(page_title="Sparsity und Total Variation – Sebastian Hanisch", layout="wide")

SPINNER = "Experiment rechnet über feste Zufallsströme …"


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=SPINNER)
def _prior_world():
    return prior_world_experiment()


@st.cache_data(show_spinner=SPINNER)
def _terrace():
    return terrace_experiment()


@st.cache_data(show_spinner=SPINNER)
def _iteration():
    return iteration_experiment()


@st.cache_data(show_spinner=SPINNER)
def _resolution():
    return resolution_experiment()


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


st.title("🪚 Sparsity und Total Variation: passt der Prior zur Welt?")
st.markdown(
    """
**Ein Prior hilft nur, wo er stimmt.** Dies ist das sechste Stück der Linie **Inverse Probleme und Regularisierung** und baut auf **Tikhonov**, **TSVD**,
der **Parameterwahl** und den **iterativen Verfahren** auf: dasselbe Phantom, dieselbe Gauß-Unschärfe, dasselbe Rauschen. Drei Strafen stehen gegeneinander:
die **quadratische** (glatte Lösungen), **L1** (wenige von null verschiedene Werte) und **Total Variation** (wenige Sprünge, sonst konstant). Dazu kommen zwei
Zusatzwelten, in denen diese Annahmen stimmen: **Spitzen** und **Blöcke**. Die Welt ist simuliert; die Wahrheit ist bekannt und dient nur als Maßstab.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("Was die drei Strafen bevorzugen", expanded=True):
    st.markdown(
        r"""
1. **Tikhonov** $\lambda^2\|x\|_2^2$: bevorzugt kleine, glatte Lösungen; die Lösung ist ein Filter der Singulärwerte (Stück 2). Kein Sprung wird belohnt.
2. **L1** $\lambda\|x\|_1$: bevorzugt **wenige von null verschiedene Werte**; kleine Komponenten werden auf null gesetzt, große nur um $\lambda$ verkleinert (weiche Schwelle).
3. **Total Variation** $\lambda\|Dx\|_1$ mit der Differenz $D$: bevorzugt **wenige Sprünge**; dazwischen ist die Lösung konstant. Glatte Hügel werden zu Stufen (*Terrassen*).
4. **Maßstab.** Für jeden Prior steht der beste λ **dieser Stichprobe** neben dem gewählten; Tikhonov ist die Referenz aus Stück 2.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Welt**")
    world = st.selectbox("Welche Wahrheit?", C.WORLDS, key="world_select", format_func=lambda v: C.WORLD_LABELS[v],
                         help="Das Phantom ist die Welt der Linie. Spitzen sind dünnbesetzt (L1 passt), Blöcke stückweise konstant (TV passt).")
    st.markdown("**Messung**")
    n = st.slider("Gitterpunkte n", *bounds("n_slider"), step=C.N_STEP, key="n_slider", help="Wie viele Messpunkte und Unbekannte.")
    w = st.slider("Breite der Unschärfe w", *bounds("w_slider"), step=C.W_STEP, key="w_slider", format="%.3f",
                  help="Breite des Gauß-Kerns als Anteil der Länge 1. Je breiter, desto stärker fallen die Singulärwerte.")
    noise = st.select_slider("Rauschen (relativ)", options=C.NOISE_CHOICES, key="noise_select", format_func=noise_label,
                             help="Standardabweichung des weißen Rauschens, relativ zum Effektivwert der exakten Daten.")
    data = st.selectbox("Woher kommen die Daten?", C.DATA_CHOICES, key="data_select", format_func=lambda v: C.DATA_LABELS[v],
                        help="Ehrliche Daten kommen aus der analytischen Unschärfe und sind vom Gitter unabhängig. Daten aus demselben Gitter entstehen aus "
                             "der Matrix, die danach invertiert wird (inverse crime).")
    st.markdown("**Prior**")
    prior = st.selectbox("Welche Strafe?", C.PRIORS, key="prior_select", format_func=lambda v: C.PRIOR_LABELS[v])
    lam = st.select_slider("Strafparameter λ", options=C.LAM_CHOICES, key="lam_select", format_func=lam_text,
                           help="Bei Tikhonov wie in Stück 2 (Filter σ²/(σ² + λ²)); bei L1 und TV die Gewichtung der Strafe in ½‖Ax − b‖² + λ·Strafe.")
    solver = st.selectbox("Löser (nur L1)", C.SOLVERS, key="solver_select", format_func=lambda v: C.SOLVER_LABELS[v],
                          help="Exakt liefert den Minimierer. ISTA und FISTA brechen nach der gewählten Schrittzahl ab; das wirkt wie in Stück 5 als zusätzliche "
                               "Regularisierung. Bei Tikhonov und TV ohne Wirkung.")
    steps = st.slider("Schritte (ISTA und FISTA)", *bounds("steps_slider"), step=C.STEP_STEP, key="steps_slider",
                      help="Schritte = 10^(Regler/10), also 1 bis 100.000. Nur bei L1 mit ISTA oder FISTA.")
    seed = st.number_input("Zufalls-Seed des Rauschens", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neues Rauschen würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "w_slider": round(float(w), 3), "noise_select": float(noise), "data_select": data, "world_select": world, "prior_select": prior,
                   "lam_select": float(lam), "solver_select": solver, "steps_slider": int(steps), "seed_input": int(seed)})

settings = Settings(int(n), round(float(w), 3), float(noise), data, world, prior, float(lam), solver, int(steps), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
if a.singular:
    st.warning(f"Bei dieser Wahl von n und w ist die Matrix in doppelter Genauigkeit numerisch singulär (Kondition {sci(a.cond)}): Die kleinen Singulärwerte sind "
               "Rundungsrauschen, und die Zahlen unten sind nur noch Zahlen.")
c1, c2, c3, c4 = st.columns(4)
how = f"{C.SOLVER_SHORT[solver]}, {int_text(a.steps)} Schritte" if a.steps else "exakter Minimierer"
c1.metric(f"{C.PRIOR_SHORT[prior]}, λ = {lam_text(settings.lam)}: Fehler", err_pct(a.err), delta=how, delta_color="off", help="‖x̂ − x‖ / ‖x‖ mit der Wahrheit dieser Welt.")
c2.metric("Bester λ dieses Priors (diese Stichprobe)", err_pct(a.best_err[prior]), delta=f"λ = {lam_text(a.best_lam[prior])}", delta_color="off",
          help="Der Wert auf dem Gitter der halben Zehnerpotenzen mit dem kleinsten tatsächlichen Fehler dieser Stichprobe; braucht die Wahrheit.")
c3.metric("Tikhonov, bester λ", err_pct(a.best_err["tikhonov"]), delta=f"λ = {lam_text(a.best_lam['tikhonov'])}", delta_color="off", help="Der Maßstab aus Stück 2 in dieser Welt.")
c4.metric("Direkte Inversion", err_pct(a.err_naive), delta="λ = 0 (Stück 1)", delta_color="off", help="Fehler dieser Stichprobe ohne Strafe.")
st.caption(f"Struktur der gewählten Lösung: {a.jumps} Sprünge (im Hügelbereich {a.jumps_hill} von {a.hill_steps}), {a.nnz} von {settings.n} Werten ungleich null.")

tab_rec, tab_err, tab_str, tab_tab = st.tabs(["Rekonstruktion", "Fehler über λ", "Struktur über λ", "Tabelle"])
with tab_rec:
    st.plotly_chart(build_recon_chart(a), width="stretch", key="recon_chart")
    st.caption("Grün die Wahrheit, grau gestrichelt der beste λ dieses Priors, punktiert Tikhonov, farbig die gewählte Lösung, blass rot die direkte Inversion. Die Achse ist fest.")
with tab_err:
    st.plotly_chart(build_error_chart(a), width="stretch", key="error_chart")
    st.caption("Der Fehler dieser Stichprobe über λ für alle drei Strafen (logarithmische Achsen). Die Rauten markieren das Minimum jedes Priors, der Kreis den gewählten Punkt.")
with tab_str:
    st.plotly_chart(build_structure_chart(a), width="stretch", key="structure_chart")
    st.caption("Mit wachsendem λ verliert TV Sprünge (die Lösung wird stückweise konstant) und L1 verliert Werte (die Lösung wird dünn besetzt). Tikhonov bleibt voll besetzt.")
with tab_tab:
    st.dataframe(style_table(build_table(a)), hide_index=True, width="stretch", height=min(40 + 35 * min(settings.n, 40), 560))
    st.caption("Die ersten Punkte der Instanz.")

if st.button("📄 PDF der aktuellen Instanz erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="sparsity-tv.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Passt der Prior zur Welt?")
r1, r2, r3 = st.columns(3)
for col, key in zip((r1, r2, r3), C.PRIORS):
    col.metric(f"{C.PRIOR_SHORT[key]}: bester Fehler dieser Stichprobe", err_pct(a.best_err[key]), delta=f"λ = {lam_text(a.best_lam[key])}", delta_color="off")
verdict(verdict_instance(a))
st.caption("Eine einzelne Stichprobe zeigt nicht, wie oft ein Prior gewinnt: dafür sind die Experimente mit festen Wiederholungen da.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Welcher Prior in welcher Welt?")
res1 = _prior_world()
st.caption(f"Standardinstanz (n = {res1['n']}), ehrliche Daten, {res1['reps']} feste Wiederholungen je Welt; bester λ je Prior auf dem Gitter, Fehler als RMS über die Wiederholungen.")
st.plotly_chart(build_prior_world_chart(res1), width="stretch", key="prior_world_chart")
verdict(verdict_prior_world(res1))
st.dataframe(
    [{"Welt": C.WORLD_SHORT[wd], "Prior": C.PRIOR_SHORT[pr], "bester λ": lam_text(c["lam"]), "RMS": err_pct(c["rms"]), "Median": err_pct(c["median"]),
      "90. Perzentil": err_pct(c["p90"]), "Diskrepanzprinzip (Median)": err_pct(c["dp_median"]), "Diskrepanzprinzip gescheitert": share(c["dp_fail"])}
     for wd in C.WORLDS for pr, c in res1["worlds"][wd].items()],
    hide_index=True, width="stretch", height=360)
st.caption("Das Diskrepanzprinzip wählt das größte λ, dessen Residuum noch unter σ²n liegt. Es trifft das Optimum jedes Priors, wo der Prior passt; in der Welt der Spitzen "
           f"scheitert es bei TV in {share(res1['worlds']['spitzen']['tv']['dp_fail'])} der Wiederholungen. Direkte Inversion (RMS): "
           + ", ".join(f"{C.WORLD_SHORT[wd]} {err_pct(res1['naive'][wd])}" for wd in C.WORLDS) + ".")
st.markdown(f"**Rauschstufen** (Phantom, je Zelle {res1['reps_small']} feste Wiederholungen)")
st.plotly_chart(build_noise_chart(res1), width="stretch", key="noise_chart")
st.dataframe(
    [{"Rauschen": noise_label(r["noise"]), **{f"{C.PRIOR_SHORT[pr]} (RMS)": err_pct(r[pr]["rms"]) for pr in C.PRIORS}} for r in res1["noise"]]
    + [{"Rauschen": "0,1 %, Daten aus demselben Gitter", **{f"{C.PRIOR_SHORT[pr]} (RMS)": err_pct(res1["crime"][pr]["rms"]) for pr in C.PRIORS}}],
    hide_index=True, width="stretch")
st.caption("Je mehr Rauschen, desto stärker die Glättung, die das Phantom braucht: L1 bricht ein, TV bleibt nahe an Tikhonov, ohne es zu schlagen. Bei sehr kleinem "
           "Rauschen liegen TV und Tikhonov gleichauf, bei Daten aus demselben Gitter gewinnt TV knapp.")

st.subheader("🔬 Experiment 2: Terrassenbildung")
res2 = _terrace()
st.caption(f"Phantom, n = {res2['n']}, Rauschen 0,1 %, ehrliche Daten, {res2['reps']} feste Wiederholungen: TV über λ, Gipfel von Hügel 1 und Sprünge im Hügelbereich.")
st.plotly_chart(build_terrace_chart(res2), width="stretch", key="terrace_chart")
verdict(verdict_terrace(res2))
st.dataframe(
    [{"λ": lam_text(r["lam"]), "TV RMS": err_pct(r["tv_rms"]), "Sprünge im Hügelbereich": f"{de(r['tv_jumps'], 1)} von {res2['hill_steps']}", "TV Gipfel": de(r["tv_peak"], 3),
      "größte Stufe": de(r["tv_step"], 2), "Tikhonov RMS": err_pct(r["tik_rms"]), "Tikhonov Gipfel": de(r["tik_peak"], 3)} for r in res2["rows"]],
    hide_index=True, width="stretch")
st.markdown("**Die Terrassen selbst** (TV auf rauschfreien Daten)")
st.plotly_chart(build_terrace_shape_chart(res2), width="stretch", key="terrace_shape_chart")
st.caption("Selbst ohne Rauschen wird der glatte Hügel zu Stufen und an der Spitze abgeschnitten: der Prior verlangt Konstanz und bekommt sie.")

st.subheader("🔬 Experiment 3: Das Verfahren regularisiert mit")
res3 = _iteration()
st.caption(f"Phantom, n = {res3['n']}, {res3['reps']} feste Wiederholungen, L1 bei λ = {lam_text(res3['lam'])} (bester exakter Wert): ISTA und FISTA gegen den exakten Minimierer.")
st.plotly_chart(build_iteration_chart(res3), width="stretch", key="iteration_chart")
verdict(verdict_iteration(res3))
st.dataframe(
    [{"Schritte": int_text(r["steps"]), "ISTA RMS": err_pct(ri["rms"]), "FISTA RMS": err_pct(r["rms"]), "FISTA Zielfunktionslücke": f"{r['gap']:.1e}".replace(".", ","),
      "FISTA größter Abstand zum Minimierer": de(r["dist"], 3)} for ri, r in zip(res3["rows"]["ista"], res3["rows"]["fista"])],
    hide_index=True, width="stretch")
st.caption("Die Zielfunktion fällt mit jedem Schritt, der Fehler nicht: das Problem ist in den kleinen Singulärrichtungen flach, und der exakte Minimierer sammelt dort das Rauschen. "
           "Wie in Stück 5 ist die Schrittzahl ein Regularisierungsparameter; er wurde hier mit der Wahrheit gewählt.")

st.subheader("🔬 Experiment 4: Wie nah dürfen zwei Spitzen liegen?")
res4 = _resolution()
st.caption(f"n = {res4['n']}, zwei gleich hohe Spitzen, Breite der Unschärfe {de(res4['width_cells'], 1)} Gitterpunkte, {res4['reps']} feste Wiederholungen je Zelle; "
           "bester λ je Zelle, „gefunden“ heißt: genau die zwei Gitterpunkte tragen Werte über 10 % der Höhe.")
st.plotly_chart(build_resolution_chart(res4), width="stretch", key="resolution_chart")
verdict(verdict_resolution(res4))
st.dataframe(
    [{"Rauschen": noise_label(r["noise"]), "Abstand": r["d"], "L1 RMS": err_pct(r["l1_rms"]), "L1 beide gefunden": share(r["hit"]), "Tikhonov RMS": err_pct(r["tik_rms"])}
     for r in res4["rows"]],
    hide_index=True, width="stretch", height=420)
st.caption("Der Abstand 2 ist schwerer als 1 und 3: die mittlere Gitterzelle ist mit beiden Spitzen stark korreliert, und L1 setzt gern einen Wert dorthin. Das ist ein "
           "Gitterartefakt, kein Naturgesetz; deshalb steht die Zelle in der Tabelle, nicht geglättet.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
phantom = res1["worlds"]["phantom"]
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Prior passt zur Welt** | Auf dem glatten Phantom liegt L1 bei {err_pct(phantom['l1']['rms'])} und TV bei {err_pct(phantom['tv']['rms'])} gegen {err_pct(phantom['tikhonov']['rms'])} bei Tikhonov. | Bayes-Sicht (Stück 7: der Prior als Annahme) |
| **Die Daten sind rauscharm genug** | Bei 10 % Rauschen liegt L1 bei {err_pct(res1['noise'][0]['l1']['rms'])}. | – |
| **Das Problem wird exakt gelöst** | ISTA und FISTA liefern je nach Schrittzahl andere Lösungen; der Fehler hängt stärker an der Schrittzahl als an der Zielfunktion. | Stück 5 (Abbruch), Stück 14 (feste Schrittzahl als Netz) |
| **Eindimensional, Pixelbasis, anisotrop** | Kein zweidimensionales TV, keine Wavelet-Basis (n ist keine Zweierpotenz; Wavelets gehören in die Spektral-Linie). | – |
| **Ein Phantom, zwei Zusatzwelten, eine Matrix** | Die Zahlen gelten für diese erfundenen Beispiele; die Zusatzwelten sind so gebaut, dass der Prior passt. | – |
| **σ ist bekannt (Diskrepanzprinzip)** | GCV und L-Kurve sind für nichtquadratische Strafen nicht einfach übertragbar (kein df in geschlossener Form) und hier nicht gemessen. | Stück 4 |
| **Das Orakel kennt die Wahrheit** | Der beste λ dieser Stichprobe und der beste λ je Zelle der Experimente sind in der Praxis nicht verfügbar. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Aufgaben.** Mit $A = h K$ (Stück 1) und Daten $b$:
$$\text{L1:}\ \min_x \tfrac12\|Ax - b\|^2 + \lambda\|x\|_1, \qquad \text{TV:}\ \min_x \tfrac12\|Ax - b\|^2 + \lambda\|Dx\|_1,$$
$D$ die Vorwärtsdifferenz ($n - 1$ Zeilen). Tikhonov: $x_\lambda = V\,\mathrm{diag}\big(\sigma_i/(\sigma_i^2 + \lambda^2)\big)U^\top b$.

**Optimalität.** Für L1 gilt $g = A^\top(Ax - b)$ mit $g_i = -\lambda\,\mathrm{sign}(x_i)$ auf $x_i \ne 0$ und $|g_i| \le \lambda$ sonst. Für TV gilt
$A^\top(Ax - b) + \lambda D^\top s = 0$ mit $s_i = \mathrm{sign}((Dx)_i)$ auf den Sprüngen und $|s_i| \le 1$ sonst; $s = \mathrm{cumsum}(q)$, $q = A^\top(Ax - b)/\lambda$.
Beide Bedingungen werden an den Lösungen geprüft (KKT-Verletzung unter $10^{-8}$).

**Exakte Lösung.** TV wird mit $x = x_0\mathbf 1 + Lw$, $w = Dx$, auf ein Lasso-Problem in $w$ zurückgeführt (der Mittelwert $x_0$ ist nicht bestraft und wird
herausprojiziert). Ein Aktivmengenverfahren löst es exakt: auf der aktiven Menge mit festen Vorzeichen ein lineares System, bei Vorzeichenwechsel die Koordinate entfernen,
sonst die am stärksten verletzte Bedingung aufnehmen.

**ISTA und FISTA.** ISTA: $x_{k+1} = \mathrm{soft}\big(x_k - \tfrac1{\sigma_1^2}A^\top(Ax_k - b),\ \tfrac{\lambda}{\sigma_1^2}\big)$ mit $\mathrm{soft}(v, t) = \mathrm{sign}(v)\max(|v| - t, 0)$
(ein Landweber-Schritt mit anschließender weicher Schwelle). FISTA setzt zusätzlich ein Trägheitsglied $y_{k+1} = x_{k+1} + \tfrac{t_k - 1}{t_{k+1}}(x_{k+1} - x_k)$.
Die Zielfunktionslücke fällt wie $1/k$ (ISTA) bzw. $1/k^2$ (FISTA).

**Literatur.** Das Beispiel ist erfunden; ISTA geht auf I. Daubechies, M. Defrise und C. De Mol (2004) zurück, FISTA auf A. Beck und M. Teboulle (2009), Lasso auf
R. Tibshirani (1996), Total Variation auf L. Rudin, S. Osher und E. Fatemi (1992), das Stufenverhalten (Staircasing) auf M. Nikolova (2000)
(alles aus dem Gedächtnis, nicht geprüft; die Zahlen hier sind gemessen, nicht der Literatur entnommen).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
