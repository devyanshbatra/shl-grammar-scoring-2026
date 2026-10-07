"""Acoustic embeddings straight from the waveform (no ASR involved).

WavLM-base-plus is a self-supervised speech model; its hidden states encode
pronunciation, prosody and fluency, which human raters hear but transcripts lose.
Only the first 30 s of each clip is used, in 15 s windows, to fit a CPU budget
(the GPU is busy with CrisperWhisper).
For every clip we mean-pool each of the 13 hidden layers over time and save a
(clips, 13, 768) array; train.py picks the layer by cross-validation.

Usage:  python src/audio_embed.py      -> outputs/wavlm_layers.npy, outputs/wavlm_index.csv
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from transcribe import DATA, clip_list  # noqa: E402

OUT = Path("outputs")
MODEL = os.environ.get("EMB_MODEL", "microsoft/wavlm-base-plus")
MAX_SECONDS = int(os.environ.get("EMB_SECONDS", "30"))  # CPU budget; 60 on GPU
NAME = os.environ.get("EMB_NAME", "wavlm")
# Leave CPU cores for the CrisperWhisper job running on the GPU.
torch.set_num_threads(int(os.environ.get("EMB_THREADS", "4")))


def main():
    from transformers import AutoFeatureExtractor, WavLMModel

    device = "cuda" if os.environ.get("EMB_DEVICE") == "cuda" else "cpu"
    fe = AutoFeatureExtractor.from_pretrained(MODEL)
    model = WavLMModel.from_pretrained(MODEL).eval().to(device)

    clips = clip_list()
    out = np.zeros((len(clips), model.config.num_hidden_layers + 1,
                    model.config.hidden_size), dtype=np.float32)
    chunk = 15 * 16000  # 15 s windows: attention cost grows with length squared
    with torch.no_grad():
        for i, (split, f) in enumerate(clips):
            wav, sr = sf.read(DATA / split / f, dtype="float32")
            wav = wav[:MAX_SECONDS * sr]  # first 30 s: enough to hear proficiency, half the CPU time
            pieces = [wav[s:s + chunk] for s in range(0, len(wav), chunk)]
            pieces = [p for p in pieces if len(p) >= sr] or [wav]  # drop <1 s tails
            acc, n_frames = 0, 0
            for p in pieces:
                inp = fe(p, sampling_rate=sr, return_tensors="pt").input_values.to(device)
                hs = model(inp, output_hidden_states=True).hidden_states
                frames = hs[0].shape[1]
                acc = acc + torch.stack([h[0].sum(0) for h in hs])
                n_frames += frames
            out[i] = (acc / n_frames).cpu().numpy()  # frame-weighted mean over the clip
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(clips)}", flush=True)

    np.save(OUT / f"{NAME}_layers.npy", out)
    pd.Series([f"{s}/{f}" for s, f in clips]).to_csv(OUT / f"{NAME}_index.csv", index=False)
    print("saved", out.shape)


if __name__ == "__main__":
    main()

