"""Figures and robustness checks for the report
"How Does Language Organize Emotions?" (Nathan Tessot, 2026).

Inputs : data.json        VAD coordinates of the 48 linked emotion words and the
                           89 Word2Vec links (cosine >= 0.50) exported from the notebook
         emotions_vad.csv  VAD ratings of the 53 emotion words (Warriner et al., 2013)
Outputs: ../figures/*.pdf, ../figures/fig2_network_vd.png and stats.json

Run from this folder:  python make_figures.py
"""
import json
import os
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from adjustText import adjust_text
import matplotlib.patheffects as pe
import statsmodels.api as sm
from scipy.stats import ttest_ind

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
FIG = os.path.join("..", "figures")
os.makedirs(FIG, exist_ok=True)

# ---------- style -----------------------------------------------------------
# Latin Modern (LaTeX font) if installed, otherwise matplotlib falls back to a serif font
for f in font_manager.findSystemFonts(["/usr/share/texmf/fonts/opentype/public/lm"]):
    font_manager.fontManager.addfont(f)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#a8a7a1"
plt.rcParams.update({
    "font.family": ["Latin Modern Roman", "DejaVu Serif"], "mathtext.fontset": "cm",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "pdf.fonttype": 42,
})

# ---------- data ------------------------------------------------------------
d = json.load(open("data.json"))
vad = {w: np.array(v) for w, v in d["vad"].items()}
edges = {frozenset(e) for e in d["edges"]}
emotions = pd.read_csv("emotions_vad.csv", index_col=0)  # 53 words

words = sorted(vad)
rows = []
for a, b in combinations(words, 2):
    dv = np.abs(vad[a] - vad[b])
    rows.append(dict(a=a, b=b, dV=dv[0], dA=dv[1], dD=dv[2],
                     dist=np.linalg.norm(dv), linked=int(frozenset((a, b)) in edges)))
P = pd.DataFrame(rows)
lk = P.linked == 1

# ---------- statistics ------------------------------------------------------
t, p = ttest_ind(P.dist[lk], P.dist[~lk], equal_var=False)
X = sm.add_constant(P[["dV", "dA", "dD"]])
raw = sm.Logit(P.linked, X).fit(disp=0)
Z = (P[["dV", "dA", "dD"]] - P[["dV", "dA", "dD"]].mean()) / P[["dV", "dA", "dD"]].std()
std = sm.Logit(P.linked, sm.add_constant(Z)).fit(disp=0)

# QAP-style permutation test: shuffle VAD profiles across words, keep the network.
rng = np.random.default_rng(0)
M = np.array([vad[w] for w in words]); idx = {w: i for i, w in enumerate(words)}
ia = P.a.map(idx).values; ib = P.b.map(idx).values
obs = P.dist[~lk].mean() - P.dist[lk].mean()
N, hits = 10000, 0
for _ in range(N):
    pm = M[rng.permutation(len(words))]
    dd = np.linalg.norm(pm[ia] - pm[ib], axis=1)
    hits += (dd[~lk.values].mean() - dd[lk.values].mean()) >= obs

stats = dict(
    n_words=len(emotions), n_linked_words=len(words), n_pairs=len(P), n_links=int(lk.sum()),
    mean_linked=P.dist[lk].mean(), sd_linked=P.dist[lk].std(),
    mean_unlinked=P.dist[~lk].mean(), sd_unlinked=P.dist[~lk].std(),
    t=t, p=p, df_welch=None,
    logit={k: dict(coef=raw.params[k], se=raw.bse[k], z=raw.tvalues[k], p=raw.pvalues[k],
                   ci=list(raw.conf_int().loc[k]), odds=float(np.exp(raw.params[k])))
           for k in raw.params.index},
    logit_std={k: dict(coef=std.params[k], se=std.bse[k], ci=list(std.conf_int().loc[k]))
               for k in std.params.index},
    pseudo_r2=raw.prsquared, llr_p=raw.llr_pvalue,
    corr_dV_dD=P.dV.corr(P.dD), corr_dV_dA=P.dV.corr(P.dA), corr_dA_dD=P.dA.corr(P.dD),
    qap_p=(hits + 1) / (N + 1), qap_obs=obs,
)
json.dump(stats, open("stats.json", "w"), indent=2, default=float)

# ---------- Figure 1: VAD distributions (53 words) ---------------------------
fig, axes = plt.subplots(1, 3, figsize=(6.3, 1.9), sharey=True)
for ax, col, name in zip(axes, ["V", "A", "D"], ["Valence", "Arousal", "Dominance"]):
    ax.hist(emotions[col], bins=np.arange(1, 9.01, 0.5), color=BLUE,
            edgecolor="white", linewidth=1)
    ax.set_xlim(1, 9); ax.set_xlabel(name); ax.set_xticks([1, 3, 5, 7, 9])
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Number of words")
axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
fig.tight_layout(w_pad=1.2)
fig.savefig(os.path.join(FIG, "fig1_vad_distributions.pdf"), bbox_inches="tight")
plt.close(fig)

# ---------- Figure 2: network projected on valence x dominance --------------
fig, ax = plt.subplots(figsize=(6.3, 4.6))
for e in edges:
    a, b = sorted(e)
    ax.plot([vad[a][0], vad[b][0]], [vad[a][2], vad[b][2]], color=BLUE,
            alpha=0.35, linewidth=0.7, zorder=1)
xs = np.array([vad[w][0] for w in words]); ys = np.array([vad[w][2] for w in words])
ax.scatter(xs, ys, s=16, color=INK, edgecolor="white", linewidth=0.8, zorder=2)
texts = [ax.text(x, y, w, fontsize=6.5, color=INK2, zorder=3,
                 path_effects=[pe.withStroke(linewidth=2, foreground="white")]) for w, x, y in zip(words, xs, ys)]
ax.set_xlim(1.5, 9); ax.set_ylim(2.3, 7.6)
adjust_text(texts, x=xs, y=ys, ax=ax, expand=(1.15, 1.35),
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.4))
ax.set_xlabel("Valence (1 = negative, 9 = positive)")
ax.set_ylabel("Dominance (1 = overwhelmed, 9 = in control)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_network_vd.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(FIG, "fig2_network_vd.png"), dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------- Figure 3: distance by link status --------------------------------
fig, ax = plt.subplots(figsize=(3.4, 2.6))
groups = [P.dist[~lk].values, P.dist[lk].values]
bp = ax.boxplot(groups, widths=0.5, patch_artist=True, showfliers=False,
                medianprops=dict(color=INK, linewidth=1.2),
                whiskerprops=dict(color=INK2, linewidth=0.8),
                capprops=dict(color=INK2, linewidth=0.8))
for patch, c in zip(bp["boxes"], [MUTED, BLUE]):
    patch.set(facecolor=c, alpha=0.35, edgecolor=c, linewidth=1)
jit = np.random.default_rng(1)
for i, (g, c) in enumerate(zip(groups, [MUTED, BLUE]), start=1):
    ax.scatter(i + jit.uniform(-0.16, 0.16, len(g)), g, s=3.5, color=c,
               alpha=0.55 if i == 2 else 0.25, linewidth=0, zorder=3)
    ax.scatter([i], [g.mean()], marker="D", s=18, color=INK, zorder=4)
ax.set_xticks([1, 2], [f"Not linked\n($n$ = {len(groups[0])})", f"Linked\n($n$ = {len(groups[1])})"])
ax.set_ylabel("Euclidean distance in VAD space")
ax.set_ylim(bottom=0)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_distance_boxplot.pdf"), bbox_inches="tight")
plt.close(fig)

# ---------- Figure 4: standardized logit coefficients ------------------------
fig, ax = plt.subplots(figsize=(3.0, 2.2))
labels = {"dV": r"$\Delta$ Valence", "dA": r"$\Delta$ Arousal", "dD": r"$\Delta$ Dominance"}
for i, k in enumerate(["dV", "dA", "dD"][::-1]):
    c = stats["logit_std"][k]
    col = MUTED if stats["logit"][k]["p"] > 0.05 else BLUE
    ax.plot(c["ci"], [i, i], color=col, linewidth=2, solid_capstyle="round")
    ax.scatter([c["coef"]], [i], s=30, color=col, edgecolor="white", linewidth=1, zorder=3)
ax.axvline(0, color=INK2, linewidth=0.7)
ax.set_ylim(-0.6, 2.6)
ax.set_yticks(range(3), [labels[k] for k in ["dV", "dA", "dD"][::-1]])
ax.set_xlabel("Log-odds per 1 SD (95% CI)")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig4_coefficients.pdf"), bbox_inches="tight")
plt.close(fig)

print(json.dumps({k: v for k, v in stats.items() if k not in ("logit", "logit_std")},
                 indent=1, default=float))
