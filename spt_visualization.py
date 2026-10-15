"""Plotly-Figuren und die Tabelle der Instanz. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import spt_constants as C
from spt_evaluation import Analysis, de, err_pct, int_text, lam_text, noise_label, sci

ERR_TICKS = dict(tickvals=[0.001, 0.01, 0.1, 0.3, 1, 3], ticktext=["0,1 %", "1 %", "10 %", "30 %", "100 %", "300 %"])


def _fixed(fig: go.Figure, height: int = 380, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_table(a: Analysis, rows: int = 40) -> pd.DataFrame:
    k = min(rows, a.settings.n)
    return pd.DataFrame({
        "Punkt": [f"{i + 1}" for i in range(k)],
        "Ort s": a.s[:k],
        "Wahrheit x": a.x[:k],
        "Messwert b": a.b[:k],
        "Gewählte Lösung": a.x_hat[:k],
        "Bester λ dieser Prior": a.best_x[a.settings.prior][:k],
        "Tikhonov (bester λ)": a.best_x["tikhonov"][:k],
    })


def style_table(df: pd.DataFrame):
    return df.style.format({c: (lambda v: de(v, 4)) for c in df.columns if c != "Punkt"})


def build_recon_chart(a: Analysis) -> go.Figure:
    """Wahrheit, gewählte Lösung, beste Lösung des Priors, Tikhonov und (blass) die direkte Inversion; die Achse ist fest."""
    lo, hi = C.SHOW_RANGE
    st = a.settings
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.s, y=a.x_naive, mode="lines", name="direkte Inversion", line=dict(color=C.COLOR_NAIVE, width=1, dash="dot"), opacity=0.5, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.x, mode="lines", name="Wahrheit x", line=dict(color=C.COLOR_TRUTH, width=3), hoverinfo="skip"))
    if st.prior != "tikhonov":
        fig.add_trace(go.Scatter(x=a.s, y=a.best_x["tikhonov"], mode="lines", name=f"Tikhonov, bester λ dieser Stichprobe ({lam_text(a.best_lam['tikhonov'])})",
                                 line=dict(color=C.COLOR_TIK, width=1.5, dash="dot"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.best_x[st.prior], mode="lines", name=f"{C.PRIOR_SHORT[st.prior]}, bester λ dieser Stichprobe ({lam_text(a.best_lam[st.prior])})",
                             line=dict(color=C.COLOR_GREY, width=2, dash="dash"), hoverinfo="skip"))
    if st.prior == "l1" and st.solver != "exakt":
        fig.add_trace(go.Scatter(x=a.s, y=a.x_exact, mode="lines", name=f"exakter Minimierer bei λ = {lam_text(st.lam)}", line=dict(color="#222222", width=1.5, dash="longdash"),
                                 hoverinfo="skip"))
    label = f"{C.PRIOR_SHORT[st.prior]}, λ = {lam_text(st.lam)}" + (f", {C.SOLVER_SHORT[st.solver]} {int_text(a.steps)} Schritte" if a.steps else "")
    fig.add_trace(go.Scatter(x=a.s, y=a.x_hat, mode="lines+markers", name=label, line=dict(color=C.PRIOR_COLORS[st.prior], width=2.5), marker=dict(size=5),
                             hovertemplate="s %{x:.3f}<br>x̂ %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title="Ort s")
    fig.update_yaxes(title="Wert", range=[lo, hi])
    peak = float(np.max(np.abs(a.x_hat))) if np.all(np.isfinite(a.x_hat)) else float("inf")
    note = ""
    if peak > max(abs(lo), abs(hi)):
        pt = "unendlich" if not np.isfinite(peak) else (de(peak, 1) if peak < 1000.0 else sci(peak))
        note = f"Die gewählte Lösung reicht bis ±{pt} und wird bei {de(lo, 1)} bis {de(hi, 1)} abgeschnitten."
    return _fixed(fig, 380, title=dict(text=note, font=dict(size=12, color="#9aa0a6"), x=0.0))


def _lam_axis(fig: go.Figure) -> None:
    fig.update_xaxes(title="Strafparameter λ", type="log", tickformat=".0e")


def build_error_chart(a: Analysis) -> go.Figure:
    """Fehler dieser Stichprobe über λ für alle drei Priors; markiert den gewählten Punkt und das Minimum jedes Priors."""
    st = a.settings
    fig = go.Figure()
    for prior in C.PRIORS:
        fig.add_trace(go.Scatter(x=a.lams, y=a.err_curve[prior], mode="lines", name=C.PRIOR_SHORT[prior], line=dict(color=C.PRIOR_COLORS[prior], width=3 if prior == st.prior else 1.5)))
        fig.add_trace(go.Scatter(x=[a.best_lam[prior]], y=[a.best_err[prior]], mode="markers", name=f"{C.PRIOR_SHORT[prior]}: bester {err_pct(a.best_err[prior])}",
                                 marker=dict(size=11, color=C.PRIOR_COLORS[prior], symbol="diamond", line=dict(width=2, color="#ffffff"))))
    fig.add_trace(go.Scatter(x=[st.lam], y=[a.err_exact], mode="markers", name=f"gewählt: λ = {lam_text(st.lam)}",
                             marker=dict(size=15, color=C.PRIOR_COLORS[st.prior], line=dict(width=2, color="#ffffff"))))
    _lam_axis(fig)
    fig.update_yaxes(title="Fehler relativ zur Signalnorm", type="log", range=[-3.2, 0.8], **ERR_TICKS)
    return _fixed(fig, 400)


def build_structure_chart(a: Analysis) -> go.Figure:
    """Zahl der Sprünge (TV) und der Werte ungleich null (L1) über λ."""
    st = a.settings
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.lams, y=a.jumps_curve["tv"], mode="lines+markers", name="TV: Sprünge", line=dict(color=C.COLOR_TV, width=3), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=a.lams, y=a.nnz_curve["l1"], mode="lines+markers", name="L1: Werte ungleich null", line=dict(color=C.COLOR_L1, width=3), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=a.lams, y=a.jumps_curve["tikhonov"], mode="lines", name="Tikhonov: Sprünge (alle Differenzen)", line=dict(color=C.COLOR_TIK, width=1.5, dash="dot")))
    fig.add_vline(x=st.lam, line=dict(color=C.COLOR_GREY, width=2, dash="dash"), annotation_text="gewählt", annotation_position="top left")
    _lam_axis(fig)
    fig.update_yaxes(title=f"Anzahl (von {st.n})", range=[0, st.n + 2])
    return _fixed(fig, 360)


def build_prior_world_chart(res: dict) -> go.Figure:
    fig = go.Figure()
    xs = [C.WORLD_SHORT[w] for w in C.WORLDS]
    for prior in C.PRIORS:
        fig.add_trace(go.Bar(x=xs, y=[100.0 * res["worlds"][w][prior]["rms"] for w in C.WORLDS], name=C.PRIOR_SHORT[prior], marker_color=C.PRIOR_COLORS[prior],
                             text=[err_pct(res["worlds"][w][prior]["rms"]) for w in C.WORLDS], textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Welt", type="category")
    fig.update_yaxes(title="Bester RMS-Fehler (% der Signalnorm)", range=[0, 45])
    return _fixed(fig, 360)


def build_noise_chart(res: dict) -> go.Figure:
    xs = [noise_label(r["noise"]) for r in res["noise"]]
    fig = go.Figure()
    for prior in C.PRIORS:
        fig.add_trace(go.Scatter(x=xs, y=[100.0 * r[prior]["rms"] for r in res["noise"]], mode="lines+markers", name=C.PRIOR_SHORT[prior],
                                 line=dict(color=C.PRIOR_COLORS[prior], width=3), marker=dict(size=7)))
    fig.update_xaxes(title="Rauschen (% des Effektivwerts der Daten)", type="category")
    fig.update_yaxes(title="Bester RMS-Fehler (% der Signalnorm)", type="log", range=[0.9, 2.1], tickvals=[10, 20, 50, 100], ticktext=["10 %", "20 %", "50 %", "100 %"])
    return _fixed(fig, 340)


def build_terrace_chart(res: dict) -> go.Figure:
    lams = [r["lam"] for r in res["rows"]]
    fig = go.Figure()
    fig.add_hline(y=res["peak_truth"], line=dict(color=C.COLOR_TRUTH, dash="dash"), annotation_text="Wahrheit", annotation_position="top left")
    fig.add_trace(go.Scatter(x=lams, y=[r["tv_peak"] for r in res["rows"]], mode="lines+markers", name="TV: Gipfel von Hügel 1", line=dict(color=C.COLOR_TV, width=3), marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=lams, y=[r["tik_peak"] for r in res["rows"]], mode="lines+markers", name="Tikhonov: Gipfel", line=dict(color=C.COLOR_TIK, width=2, dash="dot"), marker=dict(size=6)))
    _lam_axis(fig)
    fig.update_yaxes(title="Mittlerer Gipfelwert (Mittel über Wiederholungen)", range=[0.7, 1.05])
    return _fixed(fig, 340)


def build_terrace_shape_chart(res: dict) -> go.Figure:
    """Rauschfreie TV-Lösungen bei zwei λ gegen die Wahrheit: die Terrassen selbst."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res["s"], y=res["x"], mode="lines", name="Wahrheit", line=dict(color=C.COLOR_TRUTH, width=3), hoverinfo="skip"))
    for (lam, x), dash in zip(sorted(res["clean"].items()), ("dot", "solid")):
        fig.add_trace(go.Scatter(x=res["s"], y=x, mode="lines+markers", name=f"TV, λ = {lam_text(lam)} (ohne Rauschen)", line=dict(color=C.COLOR_TV, width=2, dash=dash),
                                 marker=dict(size=5), line_shape="hv"))
    fig.update_xaxes(title="Ort s")
    fig.update_yaxes(title="Wert", range=[-0.2, 1.3])
    return _fixed(fig, 340)


def build_iteration_chart(res: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_hline(y=100.0 * res["exact_rms"], line=dict(color="#222222", dash="dash"), annotation_text="exakter L1-Minimierer", annotation_position="top left")
    fig.add_hline(y=100.0 * res["tik_rms"], line=dict(color=C.COLOR_TIK, dash="dot"), annotation_text="Tikhonov, bester λ", annotation_position="bottom left")
    for key, color in (("ista", C.COLOR_L1), ("fista", C.COLOR_TV)):
        rows = res["rows"][key]
        fig.add_trace(go.Scatter(x=[r["steps"] for r in rows], y=[100.0 * r["rms"] for r in rows], mode="lines+markers", name=key.upper(), line=dict(color=color, width=3),
                                 marker=dict(size=7)))
    fig.update_xaxes(title="Schritte", type="log", tickvals=[10, 100, 1000, 10000], ticktext=["10", "100", "1.000", "10.000"])
    fig.update_yaxes(title="RMS-Fehler (% der Signalnorm)", range=[0, 42])
    return _fixed(fig, 360)


def build_resolution_chart(res: dict) -> go.Figure:
    fig = go.Figure()
    for nz in sorted({r["noise"] for r in res["rows"]}):
        rows = [r for r in res["rows"] if r["noise"] == nz]
        fig.add_trace(go.Scatter(x=[r["d"] for r in rows], y=[100.0 * r["hit"] for r in rows], mode="lines+markers", name=f"Rauschen {noise_label(nz)}", line=dict(width=3),
                                 marker=dict(size=7)))
    fig.update_xaxes(title="Abstand der Spitzen (Gitterpunkte)", tickvals=sorted({r["d"] for r in res["rows"]}))
    fig.update_yaxes(title="Beide Spitzen und nur sie gefunden (%)", range=[0, 105])
    return _fixed(fig, 340)
