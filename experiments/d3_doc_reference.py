"""
Freeze the Decision 3 (Default Donation Rate) reference from the methodology document.

Source of truth: "Donation_Rate_Decision_Methodology-2-1 250925.docx" (no Stata .dta of its
own exists; its Stata commands are in the document). This script re-implements those
commands on the 280-participant raw data the professor shipped with Decision 1,
"Stata_File_Decision 1_Updated.dta" (the doc's dependent variable `twtsospesoaw2ax2periods12`
is stored there as `twtsospesoaw2ax2periods1`, Stata's 32-character name limit).

Doc section 3 (final models), for each income mode:

    encode group, gen(groupcat)                               (base = HighSub, the first level)
    encode studyprogramcategory, gen(studyprogramcategorycat) (base = G 2-year Program)
    categorical: regress twtsospesoaw2ax2periods12 i.groupcat i.totalallowance i.studyprogramcategorycat honesty_humility
    continuous:  regress twtsospesoaw2ax2periods12 i.groupcat totalallowance   i.studyprogramcategorycat honesty_humility
    predict predprosocial

Doc section 5 (anchor):

    summarize predprosocial
    gen minpredprosocial = r(min) ; gen maxpredprosocial = r(max)
    summarize twtsospesoaw2ax2periods12
    gen mintwt... = r(min) ; gen maxtwt... = r(max)
    gen s100predprosocial = (predprosocial-minpredprosocial)/(maxpredprosocial-minpredprosocial)
    gen s100twt... = (twt...-mintwt...)/(maxtwt...-mintwt...)
    gen s100donationanchor = 0.75*s100twt... + 0.25*s100predprosocial

Doc section 6 step 4 with sigma = 0 (draw_k = anchor_k):

    score_k = 100 * max(draw_k, 0) / max_j max(draw_j, 0)

Output: data/d3_doc_reference.csv, one row per (participant, income mode):
    participantid, income_mode, doc_pred, doc_anchor, doc_final
doc_anchor is the doc's s100donationanchor (a 0-1 quantity despite its name); doc_final is
score_k / 100, i.e. the donation rate as a proportion (the app's donation_default scale).

Usage:
    python experiments/d3_doc_reference.py [path/to/Stata_File_Decision 1_Updated.dta] [out.csv]
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DTA = sys.argv[1] if len(sys.argv) > 1 else "/Users/suedagul/Downloads/Stata_File_Decision 1_Updated.dta"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "data", "d3_doc_reference.csv")

Y = "twtsospesoaw2ax2periods1"


def encode_dummies(series: pd.Series, prefix: str) -> pd.DataFrame:
    """Stata `encode` + `i.`: levels sorted, the first (lowest code) is the omitted base."""
    levels = sorted(series.unique())
    return pd.DataFrame({f"{prefix}={lv}": (series == lv).astype(float) for lv in levels[1:]},
                        index=series.index)


def regress_predict(y: np.ndarray, X: pd.DataFrame):
    """OLS with constant (Stata `regress`), returns (coefficients, fitted values = `predict`)."""
    A = np.column_stack([X.values, np.ones(len(y))])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    coef = dict(zip(list(X.columns) + ["_cons"], beta))
    return coef, A @ beta


def build(dta_path: str):
    d = pd.read_stata(dta_path, convert_categoricals=False)
    assert len(d) == 280, len(d)
    y = d[Y].astype(float).values

    G = encode_dummies(d["group"], "groupcat")
    S = encode_dummies(d["studyprogramcategory"], "studyprogramcategorycat")
    TA = encode_dummies(d["totalallowance"].astype(int), "totalallowance")
    hh = d["honesty_humility"].astype(float).rename("honesty_humility")
    ta = d["totalallowance"].astype(float).rename("totalallowance")

    specs = {
        "categorical": pd.concat([G, TA, S, hh], axis=1),
        "continuous": pd.concat([G, ta, S, hh], axis=1),
    }

    s100_obs = (y - y.min()) / (y.max() - y.min())
    rows, coefs = [], {}
    for mode, X in specs.items():
        coef, pred = regress_predict(y, X)
        coefs[mode] = coef
        s100_pred = (pred - pred.min()) / (pred.max() - pred.min())
        anchor = 0.75 * s100_obs + 0.25 * s100_pred
        floored = np.maximum(anchor, 0.0)
        final = floored / floored.max()
        rows.append(pd.DataFrame({
            "participantid": d["participantid"].astype(int).values,
            "income_mode": mode,
            "doc_pred": pred,
            "doc_anchor": anchor,
            "doc_final": final,
        }))
    return pd.concat(rows, ignore_index=True), coefs


if __name__ == "__main__":
    ref, coefs = build(DTA)
    for mode, coef in coefs.items():
        p = ref.loc[ref.income_mode == mode, "doc_pred"]
        print(f"[{mode}] pred min {p.min():.10f} max {p.max():.10f}")
        for k, v in coef.items():
            print(f"    {k:45s} {v: .7g}")
    ref.to_csv(OUT, index=False, float_format="%.17g")
    print(f"wrote {OUT} ({len(ref)} rows)")
