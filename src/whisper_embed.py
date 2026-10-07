"""Acoustic embeddings from a Whisper encoder (second audio view next to WavLM).

Uses the encoder of CrisperWhisper 2.0 turbo (a Whisper large-v3-turbo encoder
fine-tuned for verbatim speech; weights already downloaded for transcription).
Whisper's encoder was trained on ~5M hours of labelled, often accented speech,
so its states carry pronunciation/fluency cues complementary to WavLM.

Each clip is split into 30 s windows (Whisper's input size); every encoder layer
is mean-pooled over the frames that contain real audio.

Usage:  python src/whisper_embed.py  -> outputs/whisperenc_layers.npy, outputs/whisperenc_index.csv
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
MODEL = "nyralabs/CrisperWhisper2.0_turbo"
WINDOW = 30 * 16000
FRAMES_PER_SEC = 50  # encoder output: 1500 frames per 30 s


def main():
    from transformers import WhisperFeatureExtractor, WhisperModel

    fe = WhisperFeatureExtractor.from_pretrained(MODEL)
    enc = WhisperModel.from_pretrained(MODEL, torch_dtype=torch.float16).encoder.eval().cuda()

    clips = clip_list()
    n_layers = enc.config.encoder_layers + 1
    out = np.zeros((len(clips), n_layers, enc.config.d_model), dtype=np.float32)
    with torch.no_grad():
        for i, (split, f) in enumerate(clips):
            wav, sr = sf.read(DATA / split / f, dtype="float32")
            acc, n_frames = 0, 0
            for s in range(0, len(wav), WINDOW):
                piece = wav[s:s + WINDOW]
                if len(piece) < sr:  # skip <1 s tails
                    continue
                feats = fe(piece, sampling_rate=sr, return_tensors="pt").input_features
                hs = enc(feats.cuda().half(), output_hidden_states=True).hidden_states
                valid = min(1500, int(len(piece) / sr * FRAMES_PER_SEC))  # drop padding frames
                acc = acc + torch.stack([h[0, :valid].float().sum(0) for h in hs])
                n_frames += valid
            out[i] = (acc / n_frames).cpu().numpy()
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(clips)}", flush=True)

    np.save(OUT / "whisperenc_layers.npy", out)
    pd.Series([f"{s}/{f}" for s, f in clips]).to_csv(OUT / "whisperenc_index.csv", index=False)
    print("saved", out.shape, flush=True)


if __name__ == "__main__":
    main()
