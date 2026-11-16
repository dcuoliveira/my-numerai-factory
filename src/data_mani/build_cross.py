"""
Cross-block INTERACTIONS across the engineered blocks of a consolidated crypto
dataset. Two families, two modes each:

    python3 -m src.data_mani.build_cross --family crbmc    --mode curated|full
    python3 -m src.data_mani.build_cross --family crbmacro --mode curated|full

Operators:
  * per-symbol x per-symbol  ->  PRODUCT (feature_x2_mul_) + SPREAD (feature_x2_sprd_)
  * DATE-LEVEL (macro) x per-symbol  ->  centered GATE only (feature_x2_gate_):
       (a-0.5)*(b-0.5), so the macro's SIGN flips the per-symbol signal's
       cross-sectional ranking by regime. A macro SPREAD/RATIO is degenerate
       (date-constant shift leaves the per-date ranking unchanged), so it is NOT
       formed. macro x macro is skipped (both date-level -> inert).

Cross terms are built on the block columns BEFORE the neutral fill (read from the
per-block source datasets, so a missing leg -> NaN -> fillmask, not a fake 0.5),
then rank-binned per date and fillmasked. Output = the family's spine PLUS the cross
columns. Only CROSS-block pairs. Built in column chunks (resumable via chunk files).
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from src.data_mani.utils import (download_if_missing, get_api, get_spec,
                                 local_path)

_COINT = ("coint", "crypto/v2.0-coint", ("feature_coint_",))
_REV = ("rev", "crypto/v2.0-rev", ("feature_rev_",))
_BASIS = ("basis", "crypto/v2.0-basis", ("feature_basis_", "feature_funding_"))
_MOM = ("mom", "crypto/v2.0-litmom", ("feature_mom_",))
_TF = ("tf", "crypto/v2.0-momswap-macro-carver", ("feature_tf_",))
_MACRO = ("macro", "crypto/v2.0-momswap-macro", ("feature_macro_",))

FAMILIES = {
    "crbmc": {
        "spine": "crypto/v2.0-coint-rev-basis-mom-carver-fillmask",
        "blocks": [_COINT, _REV, _BASIS, _MOM, _TF],
        "date_level": set(),
        "curated": [  # (op, a, b)
            ("mul", "feature_funding_z_30d", "feature_mom_ret_28d"),
            ("mul", "feature_funding_z_30d", "feature_tf_ewmac_16_64"),
            ("mul", "feature_basis_z_60d", "feature_tf_ewmac_8_32"),
            ("mul", "feature_basis_chg_20d", "feature_mom_ret_21d"),
            ("mul", "feature_coint_adf_120d", "feature_rev_str_7d"),
            ("mul", "feature_coint_halflife_120d", "feature_rev_str_7d"),
            ("sprd", "feature_coint_resid_z_120d", "feature_rev_bbz_20d"),
            ("mul", "feature_mom_trend_robust", "feature_tf_clenow_90"),
            ("mul", "feature_mom_ret_28d", "feature_tf_ewmac_8_32"),
            ("sprd", "feature_tf_ewmac_8_32", "feature_rev_rsi_14d"),
            ("sprd", "feature_mom_ret_21d", "feature_rev_str_7d"),
            ("sprd", "feature_basis_pct", "feature_coint_resid_z_120d"),
            ("mul", "feature_rev_amihud_28d", "feature_rev_str_7d"),
            ("sprd", "feature_mom_ret_28d", "feature_rev_ivol_28d"),
            ("sprd", "feature_tf_clenow_90", "feature_rev_ivol_28d"),
            ("mul", "feature_funding_z_30d", "feature_rev_str_7d"),
        ],
    },
    "crbmacro": {
        "spine": "crypto/v2.0-coint-rev-basis-macro-fillmask",
        "blocks": [_COINT, _REV, _BASIS, _MACRO],
        "date_level": {"macro"},
        "curated": [  # macro regime-gates (centered) + per-symbol crb cross
            ("gate", "feature_macro_dxy_ret_20d", "feature_rev_str_7d"),
            ("gate", "feature_macro_dxy_ret_20d", "feature_basis_z_60d"),
            ("gate", "feature_macro_snp_ret_20d", "feature_rev_rsi_14d"),
            ("gate", "feature_macro_snp_ret_60d", "feature_basis_pct"),
            ("gate", "feature_macro_y10", "feature_funding_rate"),
            ("gate", "feature_macro_y10", "feature_coint_resid_z_120d"),
            ("gate", "feature_macro_slope_2s10s", "feature_funding_ma_30d"),
            ("gate", "feature_macro_gold_ret_60d", "feature_rev_str_7d"),
            ("mul", "feature_coint_adf_120d", "feature_rev_str_7d"),
            ("sprd", "feature_coint_resid_z_120d", "feature_rev_bbz_20d"),
            ("sprd", "feature_basis_pct", "feature_coint_resid_z_120d"),
            ("mul", "feature_funding_z_30d", "feature_rev_str_7d"),
            ("mul", "feature_basis_z_60d", "feature_rev_dist_ma_20d"),
            ("mul", "feature_coint_halflife_120d", "feature_rev_str_7d"),
        ],
    },
}


def _short(f):
    return f[len("feature_"):]


def _apply(op, a, b):
    if op == "mul":
        return a * b
    if op == "sprd":
        return a - b
    if op == "gate":
        return (a - 0.5) * (b - 0.5)
    raise ValueError(op)


def fast_bin(frame, cols, period):
    pct = frame.groupby(period)[cols].rank(method="first", pct=True).to_numpy()
    b = np.select([pct <= 0.05, pct <= 0.25, pct <= 0.75, pct <= 0.95],
                  [0.0, 0.25, 0.5, 0.75], default=1.0).astype("float32")
    b[np.isnan(pct)] = np.nan
    return pd.DataFrame(b, index=frame.index, columns=cols)


def parse_args():
    p = argparse.ArgumentParser(description="Cross-block interactions (family-driven).")
    p.add_argument("--family", choices=list(FAMILIES), required=True)
    p.add_argument("--mode", choices=["curated", "full"], required=True)
    p.add_argument("--out-version", default=None)
    p.add_argument("--chunk", type=int, default=500)
    p.add_argument("--fill-value", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    fam = FAMILIES[args.family]
    spec = get_spec("crypto"); api = get_api("crypto")
    idc, per = spec["id_col"], spec["period_col"]
    out_version = args.out_version or ("crypto/v2.0-%s-x%s-fillmask" % (args.family, args.mode))

    base = pd.read_parquet(download_if_missing(api, "crypto", fam["spine"], "train.parquet")).reset_index()
    base[per] = pd.to_datetime(base[per])
    keys = base[[idc, per]].copy()
    B = keys.copy(); block_of = {}
    for blk, ver, prefixes in fam["blocks"]:
        d = pd.read_parquet(download_if_missing(api, "crypto", ver, "train.parquet")).reset_index()
        d[per] = pd.to_datetime(d[per])
        cols = [c for c in d.columns if c.startswith(prefixes)]
        for c in cols:
            block_of[c] = blk
        B = B.merge(d[[idc, per] + cols].drop_duplicates([idc, per]), on=[idc, per], how="left")
    feats = list(block_of)
    dlev = fam["date_level"]
    print("family=%s: assembled %d block columns on %d rows" % (args.family, len(feats), len(B)))

    defs = []  # (name, op, a, b)
    def add(op, a, b):
        defs.append(("feature_x2_%s_%s__%s" % (op, _short(a), _short(b)), op, a, b))
    if args.mode == "curated":
        for op, a, b in fam["curated"]:
            if a not in block_of or b not in block_of:
                raise SystemExit("curated references missing column: %s / %s" % (a, b))
            add(op, a, b)
    else:
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                a, b = feats[i], feats[j]
                ba, bb = block_of[a], block_of[b]
                if ba == bb:
                    continue
                a_d, b_d = ba in dlev, bb in dlev
                if a_d and b_d:
                    continue                      # macro x macro -> inert
                if a_d or b_d:
                    add("gate", a, b)             # date-level x per-symbol -> centered gate
                else:
                    add("mul", a, b); add("sprd", a, b)
    print("mode=%s -> %d cross features" % (args.mode, len(defs)))

    ckpt = local_path("crypto", out_version, "train.parquet").rsplit("/", 1)[0] + "/_chunks"
    os.makedirs(ckpt, exist_ok=True)
    n_chunks = (len(defs) + args.chunk - 1) // args.chunk
    for ci in range(n_chunks):
        cpath = os.path.join(ckpt, "chunk_%03d.parquet" % ci)
        if os.path.exists(cpath):
            print("  chunk %d/%d cached" % (ci + 1, n_chunks)); continue
        sub = defs[ci * args.chunk:(ci + 1) * args.chunk]
        raw = {name: _apply(op, B[a], B[b]) for name, op, a, b in sub}
        names = [d[0] for d in sub]
        frame = pd.concat([keys, pd.DataFrame(raw, index=B.index)], axis=1)
        binned = fast_bin(frame, names, per).fillna(args.fill_value)
        pd.concat([keys, binned], axis=1).to_parquet(cpath, index=False)
        print("  chunk %d/%d: %d cols -> %s" % (ci + 1, n_chunks, len(names), cpath))

    print("assembling final dataset ...")
    out = base.copy()
    for ci in range(n_chunks):
        out = out.merge(pd.read_parquet(os.path.join(ckpt, "chunk_%03d.parquet" % ci)), on=[idc, per], how="left")
    n_feat = sum(c.startswith("feature_") for c in out.columns)
    dest = local_path("crypto", out_version, "train.parquet")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    out.set_index(idc).to_parquet(dest)
    print("features: %d total (%d spine + %d cross) -> %s (%.0f MB)"
          % (n_feat, len(base.filter(like="feature_").columns), len(defs), dest, os.path.getsize(dest) / 1e6))
    for filename in ("historical_meta_models.parquet",):
        source = download_if_missing(api, "crypto", fam["spine"], filename)
        target = local_path("crypto", out_version, filename)
        if not os.path.exists(target):
            shutil.copy(source, target)
    shutil.rmtree(ckpt, ignore_errors=True)
    print("\nrun it with:  --data-version %s" % out_version)


if __name__ == "__main__":
    main()
