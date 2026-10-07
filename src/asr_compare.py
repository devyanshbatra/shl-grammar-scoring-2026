"""Compare two ASR models on a small, label-stratified sample of training clips.

Models:
  - faster-whisper large-v3-turbo (standard Whisper, MIT)
  - CrisperWhisper 2.0 turbo (verbatim + intended transcripts in one pass)

Output: outputs/asr_compare.json with transcripts, timings and labels so the
transcripts can be read side by side before transcribing the full dataset.
"""
import json
import os
import time
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch

# CTranslate2 (used by both ASR models) needs the CUDA 12 cuBLAS/cuDNN DLLs.
# The CUDA build of PyTorch already ships them, so reuse that folder on Windows.
if os.name == "nt":
    os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))

DATA = Path("data/Dataset_Final")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# Verbatim-style prompt: nudges Whisper to keep fillers and repetitions.
VERBATIM_PROMPT = "Umm, so I I was, uh, going to the... the market, and I goes there."


def pick_sample(n_zero=4, n_per_bin=4, seed=0):
    """4 clips labelled 0.0 plus 4 from each score band 2, 3, 4, 5."""
    df = pd.read_csv(DATA / "train.csv")
    parts = [df[df.label == 0].sample(n_zero, random_state=seed)]
    for lo, hi in [(1, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, 5.1)]:
        band = df[(df.label >= lo) & (df.label < hi)]
        parts.append(band.sample(n_per_bin, random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


def load_audio(f):
    """Clips are 16 kHz mono WAV; load directly (avoids PyAV decode issues)."""
    wav, sr = sf.read(DATA / "train" / f, dtype="float32")
    assert sr == 16000 and wav.ndim == 1, (f, sr, wav.shape)
    return wav


def run_faster_whisper(files):
    from faster_whisper import WhisperModel

    model = WhisperModel("deepdml/faster-whisper-large-v3-turbo-ct2",
                         device="cuda", compute_type="int8_float16")
    out = {}
    for f in files:
        t0 = time.time()
        segments, info = model.transcribe(
            load_audio(f), language="en", beam_size=5,
            condition_on_previous_text=False, initial_prompt=VERBATIM_PROMPT,
            word_timestamps=True, vad_filter=True)
        segments = list(segments)
        out[f] = {
            "text": " ".join(s.text.strip() for s in segments),
            "avg_logprob": [round(s.avg_logprob, 3) for s in segments],
            "no_speech_prob": [round(s.no_speech_prob, 3) for s in segments],
            "secs": round(time.time() - t0, 1),
        }
        print(f"[whisper] {f} {out[f]['secs']}s")
    del model
    return out


def run_crisper(files):
    from crisperwhisper import CrisperWhisperModel

    # The ct2 backend needs a patched CTranslate2 fork with no Windows wheel,
    # so use the pure-PyTorch backend (same weights, no speculative decoding).
    model = CrisperWhisperModel("nyralabs/CrisperWhisper2.0_turbo",
                                backend="transformers", device="cuda")
    out = {}
    for f in files:
        t0 = time.time()
        # transcribe_dual() is ct2-only, so run the two modes separately.
        wav = load_audio(f)
        verb = model.transcribe(wav, sr=16000, language="en",
                                mode="verbatim", word_timestamps=True)
        intd = model.transcribe(wav, sr=16000, language="en", mode="intended")
        out[f] = {
            "verbatim": verb.text,
            "intended": intd.text,
            "n_words": len(verb.words or []),
            "secs": round(time.time() - t0, 1),
        }
        print(f"[crisper] {f} {out[f]['secs']}s")
    del model
    return out


if __name__ == "__main__":
    sample = pick_sample()
    files = sample.filename.tolist()
    # Cache each model's output so a crash in one doesn't force a rerun of the other.
    fw_cache = OUT / "asr_compare_whisper.json"
    if fw_cache.exists():
        fw = json.loads(fw_cache.read_text(encoding="utf-8"))
    else:
        fw = run_faster_whisper(files)
        fw_cache.write_text(json.dumps(fw, indent=2), encoding="utf-8")
    cw = run_crisper(files)
    rows = [{"filename": f, "label": float(l), "whisper": fw[f], "crisper": cw[f]}
            for f, l in zip(files, sample.label)]
    (OUT / "asr_compare.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("saved outputs/asr_compare.json")
