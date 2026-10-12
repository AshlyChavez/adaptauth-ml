"""Figuras del experimento base (paleta de referencia: azul y naranja, marcas finas)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def risk_distribution(R, y, t1, t2, path):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(0, 1, 41)
    ax.hist(R[y == 0], bins=bins, density=True, histtype="step", linewidth=2, color=BLUE,
            label="Accesos legítimos")
    ax.hist(R[y == 1], bins=bins, density=True, histtype="step", linewidth=2, color=ORANGE,
            label="Accesos anómalos")
    ax.set_yscale("log")
    top = ax.get_ylim()[1]
    for t, name, f, ha, dx in ((t1, "θ1", 0.5, "right", -0.01), (t2, "θ2", 0.12, "left", 0.01)):
        ax.axvline(t, color=INK2, linewidth=1, linestyle="--")
        ax.text(t + dx, top * f, f"{name} = {t:.2f}", color=INK2, fontsize=10, ha=ha)
    ax.set_xlabel("Score de riesgo consolidado R_i")
    ax.set_ylabel("Densidad (escala log)")
    _grid(ax)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def paired_blocks(res, path, etiquetas=("Baseline (solo reglas)", "AdaptAuth-ML")):
    a, b = np.array(res["f1_baseline"]), np.array(res["f1_adaptauth"])
    x = np.arange(1, len(a) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(x, a, color=BLUE, linewidth=2, marker="o", markersize=7, label=etiquetas[0])
    ax.plot(x, b, color=ORANGE, linewidth=2, marker="o", markersize=7, label=etiquetas[1])
    ax.set_xticks(x)
    ax.set_xlabel("Bloque temporal del conjunto de prueba")
    ax.set_ylabel("F1 (análisis θ1)")
    ax.set_ylim(0, 1)
    _grid(ax)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def shap_bars(mean_abs, path, xlabel="Importancia media |SHAP| sobre el score de anomalía"):
    s = mean_abs.sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(s) + 1.4))
    ax.barh(s.index, s.values, color=BLUE, height=0.6)
    for i, v in enumerate(s.values):
        ax.text(v, i, f" {v:.3f}", va="center", color=INK2, fontsize=9)
    ax.set_xlabel(xlabel)
    _grid(ax, "x")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def model_comparison(tabla, path):
    """Barras horizontales agrupadas: AUC-ROC y F1 (θ1) por modelo, misma escala 0 a 1."""
    t = tabla.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(t))
    h = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 0.6 * len(t) + 1.6))
    ax.barh(y + h / 2 + 0.01, t["auc_roc"], height=h, color=BLUE, label="AUC-ROC")
    ax.barh(y - h / 2 - 0.01, t["f1_theta1"], height=h, color=ORANGE, label="F1 (análisis θ1)")
    for i, (a, f) in enumerate(zip(t["auc_roc"], t["f1_theta1"])):
        ax.text(a + 0.01, i + h / 2, f"{a:.2f}", va="center", color=INK2, fontsize=9)
        ax.text(f + 0.01, i - h / 2, f"{f:.2f}", va="center", color=INK2, fontsize=9)
    ax.axvline(0.5, color=INK2, linewidth=1, linestyle="--")
    ax.text(0.505, -0.75, "azar (AUC 0.5)", color=INK2, fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(t["modelo"].str.replace("_", " "))
    ax.set_xlim(0, 1.08)
    ax.set_ylim(-0.9, len(t) - 0.4)
    _grid(ax, "x")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def operating_points(po, path, titulo=""):
    """Recall y precisión en prueba según el FPR fijado en validación (puntos de operación)."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = po["fpr_prueba"] * 100
    ax.plot(x, po["recall"], color=BLUE, linewidth=2, marker="o", markersize=7, label="Recall (ataques detectados)")
    ax.plot(x, po["precision"], color=ORANGE, linewidth=2, marker="o", markersize=7, label="Precisión")
    for xi, f in zip(x, po["f1"]):
        ax.annotate(f"F1 {f:.2f}", (xi, 0.02), color=INK2, fontsize=8, ha="center")
    ax.set_xscale("log")
    ax.set_xticks(x); ax.set_xticklabels([f"{v:.1f}%" for v in x], fontsize=9)
    ax.minorticks_off()
    ax.set_xlabel("FPR en prueba (legítimos a los que se les pide OTP)")
    ax.set_ylim(0, 1.02)
    _grid(ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    if titulo:
        ax.set_title(f"Puntos de operación: {titulo.replace('_', ' ')}", color=INK2, fontsize=11, loc="left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
