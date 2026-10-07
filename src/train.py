"""Train grammar-score regressors with repeated K-fold CV and write a submission.

Models (all scored on out-of-fold predictions, i.e. each clip predicted by a
model that never saw it):
  ridge_feat   Ridge on the hand-crafted features
  ridge_emb    Ridge on sentence embedding + features
  svr_feat     RBF-kernel SVR on the hand-crafted features
  lgbm_feat    LightGBM on the hand-crafted features
  ridge_audio  Ridge on features + WavLM acoustic embedding (with --audio-layer)
  blend        non-negative weighted mix of the above, weights fit on OOF preds

Usage:  python src/train.py [--emb whisper_mpnet] [--no-crisper] [--tag NAME]
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

DATA = Path("data/Dataset_Final")
OUT = Path("outputs")
LO, HI = 1.0, 5.0  # real score range once the unscored 0.0 clips are removed
N_SPLITS, N_REPEATS = 5, 3


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def report(name, y, p):
    r = pearsonr(y, p)[0]
    print(f"  {name:<12} RMSE {rmse(y, p):.4f}   Pearson {r:.4f}")
    return {"rmse": rmse(y, p), "pearson": float(r)}


def load(use_crisper, emb_name):
    feats = pd.read_parquet(OUT / "features.parquet")
    if not use_crisper:
        feats = feats[[c for c in feats.columns if not c.startswith("cw_")]]
    train = pd.read_csv(DATA / "train.csv").set_index("filename")
    train = train[train.index.str.extract(r"(\d+)")[0].astype(int).values < 5000]
    test = pd.read_csv(DATA / "test.csv").set_index("filename")

    emb_idx = pd.read_csv(OUT / f"emb_{emb_name}_index.csv").iloc[:, 0]
    emb = pd.DataFrame(np.load(OUT / f"emb_{emb_name}.npy"), index=emb_idx.values)
    emb.columns = [f"e{i}" for i in emb.columns]

    # Features are keyed 'split/filename' because 212 test files reuse train
    # filenames for different recordings.
    tr_keys, te_keys = "train/" + train.index, "test/" + test.index
    cols = [c for c in feats.columns if c != "split"]
    Xtr = feats.loc[tr_keys, cols].set_axis(train.index)
    Xte = feats.loc[te_keys, cols].set_axis(test.index)
    Etr = emb.loc[tr_keys].set_axis(train.index)
    Ete = emb.loc[te_keys].set_axis(test.index)
    return Xtr, Xte, Etr, Ete, train.label.values, test.index


def load_audio(layer, train_ids, test_ids):
    """Mean-pooled WavLM hidden layer (src/audio_embed.py) for train and test clips."""
    idx = pd.read_csv(OUT / "wavlm_index.csv").iloc[:, 0].values
    A = pd.DataFrame(np.load(OUT / "wavlm_layers.npy")[:, layer, :], index=idx)
    A.columns = [f"a{i}" for i in A.columns]
    return (A.loc["train/" + train_ids].set_axis(train_ids),
            A.loc["test/" + test_ids].set_axis(test_ids))


def linear():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         RidgeCV(alphas=np.logspace(-2, 5, 29)))


def scan_audio_layers(y, train_ids, test_ids):
    """Which WavLM layer predicts the score best on its own (Ridge, 5-fold CV)."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    folds = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=0)
                 .split(np.zeros(len(y)), np.round(y).astype(int)))
    n_layers = np.load(OUT / "wavlm_layers.npy", mmap_mode="r").shape[1]
    for layer in range(n_layers):
        A, _ = load_audio(layer, train_ids, test_ids)
        p = cross_val_predict(linear(), A, y, cv=folds)
        print(f"  layer {layer:2d}  RMSE {rmse(y, np.clip(p, LO, HI)):.4f}  "
              f"Pearson {pearsonr(y, p)[0]:.4f}")


def make_models(with_audio=False):
    svr = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                SVR(C=3.0, epsilon=0.1, gamma="scale"))
    gbm = lambda: lgb.LGBMRegressor(n_estimators=600, learning_rate=0.02, num_leaves=15,
                                    min_child_samples=15, subsample=0.8, subsample_freq=1,
                                    colsample_bytree=0.7, reg_lambda=1.0, verbose=-1)
    models = {"ridge_feat": ("feat", linear), "ridge_emb": ("both", linear),
              "svr_feat": ("feat", svr), "lgbm_feat": ("feat", gbm)}
    if with_audio:
        models["ridge_audio"] = ("feat_audio", linear)  # features + WavLM embedding
    return models


def main(args):
    Xtr, Xte, Etr, Ete, y, test_ids = load(not args.no_crisper, args.emb)
    print(f"train {Xtr.shape}  test {Xte.shape}  emb {Etr.shape[1]}d")
    if args.scan_audio:
        return scan_audio_layers(y, Xtr.index, test_ids)
    inputs = {"feat": (Xtr, Xte),
              "both": (pd.concat([Xtr, Etr], axis=1), pd.concat([Xte, Ete], axis=1))}
    with_audio = args.audio_layer is not None
    if with_audio:
        Atr, Ate = load_audio(args.audio_layer, Xtr.index, test_ids)
        inputs["feat_audio"] = (pd.concat([Xtr, Atr], axis=1), pd.concat([Xte, Ate], axis=1))

    # Stratify folds on rounded score so each fold sees the full label range.
    strata = np.round(y).astype(int)
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)
    oof, test_pred = {}, {}
    for name, (inp, factory) in make_models(with_audio).items():
        Xa, Xb = inputs[inp]
        o = np.zeros((N_REPEATS, len(y)))
        t = np.zeros(len(Xb))
        for k, (tr, va) in enumerate(cv.split(Xa, strata)):
            m = factory().fit(Xa.iloc[tr], y[tr])
            o[k // N_SPLITS, va] = m.predict(Xa.iloc[va])
            t += m.predict(Xb) / (N_SPLITS * N_REPEATS)
        oof[name], test_pred[name] = o.mean(0), t

    print("\nOut-of-fold CV (honest estimate of test performance):")
    metrics = {n: report(n, y, np.clip(p, LO, HI)) for n, p in oof.items()}

    # Blend: non-negative weights on OOF predictions, normalised to sum to 1.
    names = list(oof)
    w, _ = nnls(np.column_stack([oof[n] for n in names]), y)
    w = w / w.sum()
    blend_oof = sum(wi * oof[n] for wi, n in zip(w, names))
    blend_test = sum(wi * test_pred[n] for wi, n in zip(w, names))
    print("  blend weights:", dict(zip(names, np.round(w, 3))))
    metrics["blend"] = report("blend", y, np.clip(blend_oof, LO, HI))

    # Training-data RMSE (required by the challenge): refit every model on all
    # training clips and score on those same clips. Optimistic by construction,
    # so it is reported next to the out-of-fold numbers above, never instead.
    print("\nTraining-data fit (same clips used for fitting -> optimistic):")
    train_fit, importances = {}, None
    for name, (inp, factory) in make_models(with_audio).items():
        Xa, _ = inputs[inp]
        m = factory().fit(Xa, y)
        train_fit[name] = m.predict(Xa)
        if name == "lgbm_feat":
            importances = pd.Series(m.booster_.feature_importance("gain"), index=Xa.columns)
    blend_train = sum(wi * train_fit[n] for wi, n in zip(w, names))
    train_metrics = report("blend", y, np.clip(blend_train, LO, HI))

    tag = args.tag or f"{args.emb}{'' if args.no_crisper else '_cw'}"
    sub = pd.DataFrame({"filename": test_ids, "label": np.clip(blend_test, LO, HI)})
    sub.to_csv(OUT / f"submission_{tag}.csv", index=False)
    oof_df = pd.DataFrame({"filename": Xtr.index, "label": y, "oof": blend_oof,
                           "train_fit": blend_train})
    for n in names:
        oof_df[f"oof_{n}"] = oof[n]
    oof_df.to_csv(OUT / f"oof_{tag}.csv", index=False)
    importances.sort_values(ascending=False).to_csv(OUT / f"importance_{tag}.csv")
    (OUT / f"metrics_{tag}.json").write_text(json.dumps(
        {"cv": metrics, "train_fit": train_metrics,
         "weights": dict(zip(names, w.tolist())), "args": vars(args)}, indent=2))
    print(f"\nwrote outputs/submission_{tag}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="whisper_mpnet",
                    help="embedding cache name: <whisper|crisper>_<mpnet|bge|qwen>")
    ap.add_argument("--no-crisper", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--audio-layer", type=int, default=None,
                    help="add a Ridge on features + this WavLM layer (src/audio_embed.py)")
    ap.add_argument("--scan-audio", action="store_true",
                    help="only report CV of each WavLM layer on its own")
    main(ap.parse_args())
