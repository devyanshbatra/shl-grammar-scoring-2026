"""Train/test distribution shift: diagnose it and choose models that survive it.

1. Adversarial validation: a classifier separates train from test clips (AUC >> 0.5
   means the test set was recorded/collected differently).
2. "Test-like" holdout: the 25% of training clips the classifier thinks look most
   like test clips. Models are compared there instead of on random CV folds.
3. Variants: which inputs (text features / WavLM / Whisper encoder) and whether audio
   embeddings are mean-normalised per split (removes a constant recording offset,
   like cepstral mean normalisation).

Usage:  python src/shift_eval.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).parent))
from train import HI, LO, linear, load, load_audio, rmse  # noqa: E402


def center(a, b):
    """Per-split mean normalisation: each split minus its own mean."""
    return a - a.mean(), b - b.mean()


def main():
    X, Xte, E, Ete, y, te = load(True, "whisper_mpnet")
    A, Ate = load_audio(11, X.index, te)
    W, Wte = load_audio(27, X.index, te, "whisperenc")

    # 1-2. adversarial probability that each train clip "looks like test"
    both = pd.concat([pd.concat([X, A, W], axis=1), pd.concat([Xte, Ate, Wte], axis=1)])
    is_test = np.r_[np.zeros(len(X)), np.ones(len(te))]
    adv = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        LogisticRegression(C=0.05, max_iter=5000))
    p = cross_val_predict(adv, both, is_test, method="predict_proba",
                          cv=StratifiedKFold(5, shuffle=True, random_state=0))[:len(X), 1]
    hold = p >= np.quantile(p, 0.75)
    print(f"test-like holdout: {hold.sum()} clips (label mean {y[hold].mean():.2f} "
          f"vs rest {y[~hold].mean():.2f})")

    # 3. variants, trained on the rest and scored on the test-like holdout
    def score(name, parts, centered=()):
        tr_parts, ho_parts = [], []
        for nm, M in parts:
            a, b = M[~hold], M[hold]
            if nm in centered:
                a, b = center(a, b)
            tr_parts.append(a)
            ho_parts.append(b)
        m = linear().fit(pd.concat(tr_parts, axis=1), y[~hold])
        pr = np.clip(m.predict(pd.concat(ho_parts, axis=1)), LO, HI)
        print(f"  {name:<40} RMSE {rmse(y[hold], pr):.4f}  r {pearsonr(y[hold], pr)[0]:.4f}")
        return pr

    F, Ae, We = ("feat", X), ("wavlm", A), ("wenc", W)
    print("Ridge models on the test-like holdout:")
    score("text features only", [F])
    score("features + WavLM", [F, Ae])
    score("features + WavLM (centered)", [F, Ae], {"wavlm"})
    score("features + Whisper-enc", [F, We])
    score("features + Whisper-enc (centered)", [F, We], {"wenc"})
    score("features + both audio", [F, Ae, We])
    score("features + both audio (centered)", [F, Ae, We], {"wavlm", "wenc"})


if __name__ == "__main__":
    main()
