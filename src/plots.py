"""Figures for the report"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#1a1a1a"
MUTED = "#b8b8b8"
ACCENT = "#c0392b"

plt.rcParams.update({
    "font.family": "serif", "font.size": 7, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
})


def corr_grid(rhos, key, path, title_prefix="byte", xlim=None):
    """4x4 grid: correlation vs sample, all 256 guesses grey, winner highlighted."""
    fig, axes = plt.subplots(4, 4, figsize=(7.16, 5.6), sharex=True, sharey=True)
    ymax = max(np.abs(r).max() for r in rhos) * 1.1
    for i, ax in enumerate(axes.flat):
        rho = rhos[i]
        for g in range(256):
            if g != key[i]:
                ax.plot(rho[g], color=MUTED, lw=0.3, alpha=0.5)
        ax.plot(rho[key[i]], color=ACCENT, lw=1.0)
        t = int(np.abs(rho[key[i]]).argmax())
        ax.annotate(f"0x{key[i]:02x}", xy=(t, rho[key[i], t]), xytext=(4, -2),
                    textcoords="offset points", fontsize=6, color=INK)
        ax.set_title(f"{title_prefix} {i}", fontsize=7, pad=2)
        ax.set_ylim(-ymax, ymax)
        if xlim:
            ax.set_xlim(*xlim)
        ax.grid(axis="y", color="#e6e6e6", lw=0.4)
    for ax in axes[-1]:
        ax.set_xlabel("sample index")
    for ax in axes[:, 0]:
        ax.set_ylabel("correlation")
    fig.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.4)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def rank_plot(ns, ranks, path, n_min=None):
    """ranks: (len(ns), 16). Each byte grey; worst byte in ink."""
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ranks = np.asarray(ranks)
    for b in range(ranks.shape[1]):
        ax.plot(ns, ranks[:, b], color=MUTED, lw=0.6)
    ax.plot(ns, ranks.max(1), color=INK, lw=1.4, label="worst byte")
    ax.set_xscale("log")
    ax.set_xlabel("number of traces")
    ax.set_ylabel("rank of converged key byte")
    if n_min:
        ax.axvline(n_min, color=ACCENT, lw=0.8, ls="--")
        ax.annotate(f"N = {n_min}", xy=(n_min, ranks.max() * 0.8), xytext=(4, 0),
                    textcoords="offset points", fontsize=6.5, color=INK)
    ax.legend(frameon=False, fontsize=6.5)
    ax.grid(axis="y", color="#e6e6e6", lw=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def corr_vs_n(ns, best, others, path):
    """Peak |rho| of correct guess vs best wrong guess, averaged over bytes."""
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.plot(ns, best, color=ACCENT, lw=1.2, label="converged key byte")
    ax.plot(ns, others, color=INK, lw=0.8, ls="--", label="best wrong guess")
    ax.set_xscale("log")
    ax.set_xlabel("number of traces")
    ax.set_ylabel("peak |correlation| (mean over bytes)")
    ax.legend(frameon=False, fontsize=6.5)
    ax.grid(axis="y", color="#e6e6e6", lw=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def mean_trace(traces, path, marks=()):
    fig, ax = plt.subplots(figsize=(3.5, 1.8))
    ax.plot(traces.mean(0), color=INK, lw=0.7)
    for m, lab in marks:
        ax.axvline(m, color=ACCENT, lw=0.6, ls="--")
        ax.annotate(lab, xy=(m, ax.get_ylim()[0]), xytext=(2, 2),
                    textcoords="offset points", fontsize=6)
    ax.set_xlabel("sample index")
    ax.set_ylabel("mean power (a.u.)")
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=300)
    plt.close(fig)
