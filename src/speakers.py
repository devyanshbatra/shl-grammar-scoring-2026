"""Group training clips by speaker so cross-validation never splits one voice.

Finding: training clips have near-identical voices among other training clips
(median nearest-neighbour cosine 0.87 on WavLM layer 2; 265/732 clips > 0.9),
while test clips are far from all training clips (median 0.62; 1/216 > 0.9).
So the train set repeats speakers and the test set has new ones. With random
folds, audio embeddings let a model recognise a voice it has already seen, which
inflates CV and does not transfer to the leaderboard.

Speaker groups = connected components of the "same voice" graph (cosine similarity
above THRESH on mean-pooled WavLM layer 2, a speaker-identity layer).

Usage:  python src/speakers.py  -> outputs/speaker_groups.csv (key, group)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

OUT = Path("outputs")
THRESH = 0.85  # test->train similarities sit below ~0.72 (p90), same-speaker pairs above


def main():
    L = np.load(OUT / "wavlm60_layers.npy")[:, 2]
    idx = pd.read_csv(OUT / "wavlm60_index.csv").iloc[:, 0].values
    tr = np.array([k.startswith("train/") for k in idx])
    X = L[tr] - L[tr].mean(0)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    sim = X @ X.T
    n, labels = connected_components(sim > THRESH, directed=False)
    sizes = pd.Series(labels).value_counts()
    print(f"{tr.sum()} train clips -> {n} speaker groups; "
          f"singletons {int((sizes == 1).sum())}, largest group {sizes.max()} clips, "
          f"clips in multi-clip groups {int(sizes[sizes > 1].sum())}")
    pd.DataFrame({"key": idx[tr], "group": labels}).to_csv(OUT / "speaker_groups.csv", index=False)


if __name__ == "__main__":
    main()
