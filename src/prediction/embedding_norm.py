"""
Normalisation of embedding matrices along five reference axes with seven transformations (E189).

Every transformation maps a value x against a REFERENCE set R of the same coordinate:

    Z    (x - mean R) / sd R                       (ddof 0)
    ROB  (x - median R) / (q75 R - q25 R)          (linear-interpolated quantiles)
    MM   (x - min R) / (max R - min R), clipped to [0, 1]
    PCT  p = (L + 0.5 E) / n                       L = #{r < x}, E = #{r = x}, n = |R|
    RNK  Phi^-1(clip(p, 0.5/n, 1 - 0.5/n))
    CZ   clip(Z, -3, 3)
    BIN  Numerai's five bins of q = clip((L + 0.5 E - 0.5)/(n - 1), 0, 1), cuts inclusive at
         .05/.25/.75/.95 in integer arithmetic. For x inside R this is exactly
         src.data_mani.history_features.bin_ties_preserved; outside R it evaluates x against the
         frozen reference distribution.

A constant reference (sd, IQR or range below EPS, or n == 1) yields the neutral value NEUTRAL[tf].
Mid-ranks throughout, so ties share one value. Inputs must be finite: missing rows are excluded by
the caller, never imputed here.

The axis decides what R is (see `snapshot` and `fit_reference`):

    CS    covered rows of the same date                        snapshot, per coordinate
    GRP   covered rows of the same date and group              snapshot, per coordinate (CS fallback)
    DIM   the row's own coordinates                            snapshot, per row
    TS    the same asset's rows on the fold's training dates   train-referenced (POOL fallback)
    POOL  all rows on the fold's training dates                train-referenced
"""
import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

TRANSFORMS = ("Z", "ROB", "MM", "PCT", "RNK", "CZ", "BIN")
AXES = ("CS", "TS", "POOL", "DIM", "GRP")
SNAPSHOT_AXES = ("CS", "DIM", "GRP")
TRAIN_AXES = ("TS", "POOL")
EPS = 1e-6
CLIP = 3.0
MIN_CS = 20          # covered assets a date needs (else the caller excludes the date)
MIN_GROUP = 20       # covered assets a (date, group) cell needs, else it takes the CS reference
MIN_TS = 30          # training rows an asset needs, else it takes the POOL reference
NEUTRAL = {"Z": 0.0, "ROB": 0.0, "MM": 0.5, "PCT": 0.5, "RNK": 0.0, "CZ": 0.0, "BIN": 0.5}
BIN_LABELS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
RANK_TF = ("PCT", "RNK", "BIN")


def _bins_from_twice_r(r2, n):
    """Bins from 2r = 2L + E - 1 (an integer) against n - 1, as bin_ties_preserved does with r."""
    m = n - 1
    r2 = np.clip(r2, 0, 2 * m)
    idx = (10 * r2 > m).astype(np.int8) + (2 * r2 > m) + (2 * r2 > 3 * m) + (10 * r2 > 19 * m)
    return BIN_LABELS[idx]


def _from_counts(tf, L, E, n):
    """PCT / RNK / BIN from per-element counts below (L) and equal (E) in a reference of size n."""
    if tf == "PCT":
        return (L + 0.5 * E) / n
    if tf == "RNK":
        p = (L + 0.5 * E) / n
        return ndtri(np.clip(p, 0.5 / n, 1.0 - 0.5 / n))
    return _bins_from_twice_r(2 * L + E - 1, n)


class Reference:
    """
    Per-column reference statistics of a matrix R (n, p), frozen once fitted.

    `apply(X)` transforms the columns of X (m, p) against them. Rank transforms keep each column's
    sorted reference, so a later row is placed in the TRAINING distribution, never re-ranked among
    later rows.
    """

    def __init__(self, tf, R):
        assert tf in TRANSFORMS, tf
        R = np.asarray(R, np.float64)
        assert R.ndim == 2 and np.isfinite(R).all(), "reference must be a finite 2-d matrix"
        self.tf, self.n, self.p = tf, R.shape[0], R.shape[1]
        if self.n == 0:
            raise ValueError("empty reference")
        lo, hi = R.min(0), R.max(0)
        self.const = (hi - lo < EPS) | (self.n == 1)
        if tf in ("Z", "CZ"):
            self.loc, self.scale = R.mean(0), R.std(0)
            self.const |= self.scale < EPS
        elif tf == "ROB":
            q25, med, q75 = np.quantile(R, [0.25, 0.5, 0.75], axis=0)
            self.loc, self.scale = med, q75 - q25
            self.const |= self.scale < EPS
        elif tf == "MM":
            self.loc, self.scale = lo, hi - lo
        else:
            self.sorted = np.sort(R, axis=0)

    def apply(self, X):
        X = np.asarray(X, np.float64)
        assert X.ndim == 2 and X.shape[1] == self.p and np.isfinite(X).all()
        if self.tf in RANK_TF:
            out = np.empty_like(X)
            for j in range(self.p):
                s = self.sorted[:, j]
                L = np.searchsorted(s, X[:, j], "left")
                E = np.searchsorted(s, X[:, j], "right") - L
                out[:, j] = _from_counts(self.tf, L, E, self.n)
        else:
            scale = np.where(self.const, 1.0, self.scale)
            out = (X - self.loc) / scale
            if self.tf == "CZ":
                out = np.clip(out, -CLIP, CLIP)
            elif self.tf == "MM":
                out = np.clip(out, 0.0, 1.0)
        out[:, self.const] = NEUTRAL[self.tf]
        return out

    def summary(self):
        """Compact, storable description of the fitted parameters."""
        d = {"tf": self.tf, "n": np.int64(self.n), "const": self.const}
        if self.tf in RANK_TF:
            d["quantiles"] = np.quantile(self.sorted, np.linspace(0, 1, 21), axis=0).astype(np.float32)
        else:
            d["loc"], d["scale"] = self.loc.astype(np.float32), self.scale.astype(np.float32)
        return d


def in_sample(tf, X):
    """
    Transform every column of X against itself (the snapshot case). Identical to
    Reference(tf, X).apply(X), computed with one rankdata call for the rank transforms.
    """
    X = np.asarray(X, np.float64)
    if tf not in RANK_TF:
        return Reference(tf, X).apply(X)
    n = X.shape[0]
    const = (X.max(0) - X.min(0) < EPS) | (n == 1)
    avg = rankdata(X, method="average", axis=0)           # = L + (E + 1) / 2
    if tf == "BIN":
        out = _bins_from_twice_r(np.rint(2 * avg - 2).astype(np.int64), n)
    else:
        p = (avg - 0.5) / n                               # = (L + 0.5 E) / n
        out = p if tf == "PCT" else ndtri(np.clip(p, 0.5 / n, 1.0 - 0.5 / n))
    out[:, const] = NEUTRAL[tf]
    return out


def snapshot(tf, axis, X, dates, groups=None):
    """
    CS / GRP / DIM transforms of the covered rows X (m, p). Each row uses only its own date's
    snapshot (CS, GRP) or itself (DIM), so no other date can move it.

    :param dates: (m,) date of each row
    :param groups: (m,) group label of each row (GRP only)
    :return: (m, p) float64, and the number of GRP rows that fell back to CS
    """
    X = np.asarray(X, np.float64)
    out = np.empty_like(X)
    fallback = 0
    if axis == "DIM":
        return in_sample(tf, X.T).T, 0
    dates = np.asarray(dates)
    order = np.argsort(dates, kind="stable")
    bounds = np.flatnonzero(np.r_[True, dates[order][1:] != dates[order][:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        if axis == "CS":
            out[idx] = in_sample(tf, X[idx])
            continue
        assert axis == "GRP"
        g = np.asarray(groups)[idx]
        cs = None
        for k in np.unique(g):
            sub = idx[g == k]
            if len(sub) >= MIN_GROUP:
                out[sub] = in_sample(tf, X[sub])
            else:
                if cs is None:
                    cs = in_sample(tf, X[idx])
                out[sub] = cs[g == k]
                fallback += len(sub)
    return out, fallback


def train_referenced(tf, axis, X, assets, fit_mask, apply_mask):
    """
    TS / POOL transforms. References are fitted on the rows `fit_mask` (the fold's training rows)
    and frozen; rows `apply_mask` are transformed against them. Rows outside fit_mask never enter a
    reference, whatever their values.

    :return: (out (m, p) with NaN outside apply_mask, info dict with fallbacks and fitted references)
    """
    X = np.asarray(X, np.float64)
    fit_mask, apply_mask = np.asarray(fit_mask, bool), np.asarray(apply_mask, bool)
    out = np.full_like(X, np.nan)
    pool = Reference(tf, X[fit_mask])
    info = {"pool": pool, "assets": {}, "fallback_assets": 0, "fallback_rows": 0, "n_fit": int(fit_mask.sum())}
    if axis == "POOL":
        out[apply_mask] = pool.apply(X[apply_mask])
        return out, info
    assert axis == "TS"
    assets = np.asarray(assets)
    for a in np.unique(assets[apply_mask]):
        rows = assets == a
        tgt = rows & apply_mask
        ref = rows & fit_mask
        if ref.sum() >= MIN_TS:
            r = Reference(tf, X[ref])
            info["assets"][a] = r
        else:
            r = pool
            info["fallback_assets"] += 1
            info["fallback_rows"] += int(tgt.sum())
        out[tgt] = r.apply(X[tgt])
    return out, info
