"""plot_decoding.py -- when, along the word-onset response, and where does information
about later recall appear?

Encoding contrast only (recalled vs not-recalled words; the lab's RAM classifiers,
ram_utils, also train on retrieval events). Per trial and electrode: the high-gamma
response (plot_latency.response: envelope minus its word-off mean, smoothed) averaged in
100 ms windows, 0-1000 ms; z-scored within session. Per subject (>= MIN_TRIALS recalled and
forgotten words), Burke lobe (hemispheres pooled; >= MIN_ELECTRODES electrodes) and window:
L2 logistic regression on the lobe's electrodes (balanced class weights), as DeHaan et al. 2025:
leave-one-session-out (>= MIN_SESSIONS sessions per montage), C chosen per held-out session by an inner
leave-one-session-out over CS (best mean inner AUC), refit on the training sessions; AUC within
each held-out session, averaged per montage, montages averaged per subject (a montage = one
electrode set; a changed set would zero-fill the missing electrodes). Within one session the AUC is unbiased (chance 0.5); the
pooled held-out AUC is not (fold intercepts track the training recall rate: chance ~0.49).
Across subjects: one-sided t of AUC > 0.5, BH-FDR over lobe x window; stars q < .05; dashed: 0.5.
-> figures/decoding/recall_auc_time_word_on.png (+ _per_subject.csv, _stats.csv)

    python plot_decoding.py [--from-csv] [--simulation-tag TAG]
"""
from __future__ import annotations

import os
from os.path import join

from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy import special
from scipy.stats import rankdata, ttest_1samp
from sklearn.model_selection import LeaveOneGroupOut
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

import fc_comparison_functions as fc
from compute import power_dir
from plot_latency import response

BEH = "word_on"
WIN_MS = 100
WINDOWS = range(0, 1000, WIN_MS)
MIN_TRIALS = 10
MIN_ELECTRODES = 5   # per subject and lobe: features per classifier
CS = np.logspace(-6, 11, 52)   # L2 strengths searched (DeHaan et al. 2025)
MIN_SESSIONS = 3   # as DeHaan et al. 2025
CHANCE = 0.5


def session_features(p: dict, lobe_of: dict) -> pd.DataFrame:
    """(trials x '<lobe>|<electrode>|<window start>') z-scored within the session, +
    recalled, session, list (NaN when trials don't match the events)."""
    roi = fc.roi_of_reg_full(p["reg_full"], lobe_of)
    x, t = response(p)
    f = pd.DataFrame({f"{r.split('-', 1)[1]}|{lab}|{a}": x[:, c, (t >= a) & (t < a + WIN_MS)].mean(1)
                      for c, (r, lab) in enumerate(zip(roi, p["labels"])) if r is not None for a in WINDOWS})
    f = (f - f.mean()) / f.std()
    w = fc.load_events(fc.dfrow_from_sid(p["sid"]), BEH)
    w = w[np.asarray(w.attrs["mask"], bool)]   # word-on events, in the stored trial order
    f["recalled"], f["session"] = p["recalled_hi"], str(p["sid"])
    f["list"] = w["trial"].to_numpy(float) if len(w) == len(f) else np.nan
    return f


def fit_path(X: np.ndarray, y: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """(features + intercept, len(CS)) weights of sklearn's LogisticRegression(C, class_weight="balanced")
    for each C in CS: damped Newton, warm-started along the path (strong -> weak penalty)."""
    n, n1 = len(y), y.sum(); S = np.where(y, n / (2 * n1), n / (2 * (n - n1)))
    X1 = np.c_[X, np.ones(n)]
    def loss(th, r):
        z = X1 @ th
        return (S * (np.logaddexp(0, z) - y * z)).sum() + 0.5 * (r * th ** 2).sum()
    th, out = np.zeros(X1.shape[1]), []
    for c in CS:
        r = np.r_[np.full(X.shape[1], 1 / c), 0.0]   # L2 on the weights, not the intercept
        for it in range(100):
            q = special.expit(X1 @ th)
            step = solve((X1.T * (S * q * (1 - q))) @ X1 + np.diag(r), X1.T @ (S * (q - y)) + r * th, assume_a="pos")
            f0, k = loss(th, r), 1.0
            while loss(th - k * step, r) > f0 and k > 1e-4:   # halve the step if the loss rises
                k /= 2
            th = th - k * step
            if np.abs(k * step).max() < tol:
                break
        out.append(th)
        if it == 0:   # unchanged from the previous C: the penalty no longer matters, nor will it
            break
    return np.column_stack(out + [th] * (len(CS) - len(out)))


def session_auc(score: np.ndarray, y: np.ndarray) -> np.ndarray:
    """AUC of each column of score (one held-out session)."""
    rk, n1 = rankdata(score, axis=0), y.sum()
    return ((rk * y[:, None]).sum(0) - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def loso(X: np.ndarray, y: np.ndarray, sess: np.ndarray) -> list:
    """[(test index, (n test, len(CS)) scores)] per held-out session with both classes."""
    out = []
    for tr, te in LeaveOneGroupOut().split(X, groups=sess):
        if 0 < y[te].sum() < len(te) and 0 < y[tr].sum() < len(tr):
            out.append((te, np.c_[X[te], np.ones(len(te))] @ fit_path(X[tr], y[tr])))
    return out


def auc(d: pd.DataFrame, cols: list[str], y: np.ndarray, sess: np.ndarray) -> float:
    """Nested leave-one-session-out AUC: per held-out session, C maximising the mean inner
    held-out-session AUC, scores from the refit on all training sessions; mean over sessions."""
    X, out = d[cols].fillna(0).to_numpy(), []
    for te, score in loso(X, y, sess):
        tr = np.setdiff1d(np.arange(len(y)), te)
        inner = [session_auc(s, y[tr][i]) for i, s in loso(X[tr], y[tr], sess[tr])]
        out.append(session_auc(score[:, [np.argmax(np.mean(inner, 0))]], y[te])[0])
    return float(np.mean(out))


def decode_subject(sub: str, fs: list) -> list:
    """Per lobe x window AUCs for one subject (mean over montages)."""
    mont: dict[frozenset, list] = {}
    for f in fs:
        mont.setdefault(frozenset(f.columns), []).append(f)
    tres = []
    for m in mont.values():
        d = pd.concat(m, ignore_index=True).dropna(subset=["recalled"])
        y = d["recalled"].to_numpy(int)
        sess = d["session"].to_numpy()
        if min(y.sum(), len(y) - y.sum()) < MIN_TRIALS or len(set(sess)) < MIN_SESSIONS:
            continue
        for lobe in fc.LOBES:
            for a in WINDOWS:
                cols = [c for c in d.columns if c.startswith(f"{lobe}|") and c.endswith(f"|{a}")]
                if len(cols) >= MIN_ELECTRODES:
                    tres.append({"sub": sub, "lobe": lobe, "window_ms": a, "n_electrodes": len(cols),
                                 "auc": auc(d, cols, y, sess)})
    if not tres:
        return []
    return pd.DataFrame(tres).groupby(["sub", "lobe", "window_ms"], as_index=False).mean().to_dict("records")


def main() -> None:
    args, save_root = fc.plot_args(__doc__, join("figures", "decoding"), **{
        "--from-csv": dict(action="store_true", help="redraw from the saved per-subject CSV (no decoding)")})
    stem = join(args.out_dir, f"recall_auc_time_{BEH}")
    if args.from_csv:
        tres = pd.read_csv(fc.table_path(f"{stem}_per_subject.csv"))
    else:
        lobe_of = fc.load_burke_maps()
        subs: dict[str, list] = {}
        for f in tqdm(fc.session_files(power_dir(save_root, BEH, "high_gamma"), "_power.pkl", args.n_sessions),
                      desc="sessions"):
            p = fc.load_pickle(str(f))
            subs.setdefault(str(p["sid"][0]), []).append(session_features(p, lobe_of))
        out = Parallel(n_jobs=max(1, os.cpu_count() // 2))(
            delayed(decode_subject)(sub, fs) for sub, fs in tqdm(subs.items(), desc="subjects"))
        tres = pd.DataFrame([t for r in out for t in r])
        os.makedirs(args.out_dir, exist_ok=True)
        tres.to_csv(fc.table_path(f"{stem}_per_subject.csv"), index=False)

    g = tres.groupby(["lobe", "window_ms"])
    st = g["auc"].agg(["mean", "sem", "size"]).reset_index()
    st["null"] = CHANCE   # kept for plot_sim_summary.py
    st["p"] = [ttest_1samp(x["auc"], CHANCE, alternative="greater").pvalue for _, x in g]
    st["q"] = multipletests(st["p"], method="fdr_bh")[1]
    st.to_csv(fc.table_path(f"{stem}_stats.csv"), index=False)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
    for lobe, ax in zip(fc.LOBES, axes.flat):
        s = st[st["lobe"] == lobe]
        x, col = s["window_ms"] + WIN_MS / 2, fc.LOBE_COLORS[lobe]
        ax.fill_between(x, s["mean"] - s["sem"], s["mean"] + s["sem"], color=col, alpha=0.3, lw=0)
        ax.plot(x, s["mean"], "o-", color=col, lw=2, ms=4)
        sig = s["q"] < 0.05
        ax.plot(x[sig], (s["mean"] + s["sem"])[sig] + 0.004, "*", color=col, ms=8)
        ax.plot(x, s["null"], color="black", ls="--", lw=1)   # chance level
        ax.set_title(f"{lobe.capitalize()} (n={int(s['size'].max())})", fontsize=12, color=col)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("Time after word onset (ms)", fontsize=13)
    fig.supylabel("Recall AUC", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=300)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)
    print(f"[decoding] {tres['sub'].nunique()} subjects; "
          f"{int((st['q'] < 0.05).sum())} of {len(st)} lobe x window cells AUC > 0.5 at q < .05")
    print(f"[saved] {stem}.png")


if __name__ == "__main__":
    main()
