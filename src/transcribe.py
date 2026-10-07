"""Transcribe every train/test clip with one ASR model and cache results.

Usage:
    python src/transcribe.py whisper   # faster-whisper large-v3-turbo
    python src/transcribe.py crisper   # CrisperWhisper 2.0 turbo (verbatim + intended)

Results are appended to outputs/transcripts_<model>.jsonl, one JSON object per
clip, so the job can be stopped and resumed without redoing finished clips.
"""
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch

# CTranslate2 needs the CUDA 12 cuBLAS/cuDNN DLLs; the CUDA build of PyTorch ships them.
if os.name == "nt":
    os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))

DATA = Path("data/Dataset_Final")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# Verbatim-style prompt: nudges Whisper to keep fillers and repetitions.
VERBATIM_PROMPT = "Umm, so I I was, uh, going to the... the market, and I goes there."


def clip_list():
    """(split, filename) for all test clips, then train clips.

    Train files numbered >= 5000 are skipped: all 37 are labelled 0.0, none
    appear in test, and their speech is normal, so the label means "unscored".
    """
    test = pd.read_csv(DATA / "test.csv").filename
    train = pd.read_csv(DATA / "train.csv").filename
    train = train[train.str.extract(r"(\d+)")[0].astype(int) < 5000]
    return [("test", f) for f in test] + [("train", f) for f in train]


def load_audio(split, f):
    wav, sr = sf.read(DATA / split / f, dtype="float32")
    assert sr == 16000 and wav.ndim == 1, (f, sr, wav.shape)
    return wav


def whisper_runner():
    from faster_whisper import WhisperModel

    model = WhisperModel("deepdml/faster-whisper-large-v3-turbo-ct2",
                         device="cuda", compute_type="int8_float16")

    def run(wav):
        segments, _ = model.transcribe(
            wav, language="en", beam_size=5,
            condition_on_previous_text=False, initial_prompt=VERBATIM_PROMPT,
            word_timestamps=True, vad_filter=True)
        segments = list(segments)
        return {
            "text": " ".join(s.text.strip() for s in segments),
            "segments": [{"start": s.start, "end": s.end,
                          "avg_logprob": s.avg_logprob,
                          "no_speech_prob": s.no_speech_prob,
                          "compression_ratio": s.compression_ratio} for s in segments],
            "words": [{"w": w.word, "s": round(w.start, 3), "e": round(w.end, 3),
                       "p": round(w.probability, 4)}
                      for s in segments for w in (s.words or [])],
        }
    return run


def crisper_runner():
    from crisperwhisper import CrisperWhisperModel

    # ct2 backend needs a CTranslate2 fork with no Windows wheel -> PyTorch backend.
    model = CrisperWhisperModel("nyralabs/CrisperWhisper2.0_turbo",
                                backend="transformers", device="cuda")

    def run(wav):
        # Verbatim pass only: the intended pass doubled the runtime (~20 s/clip on a
        # 4 GB RTX 3050), and Whisper's transcript already serves as a "cleaned"
        # reference. Word timings come from the Whisper pass.
        verb = model.transcribe(wav, sr=16000, language="en", mode="verbatim")
        return {"verbatim": verb.text}
    return run


def main(name):
    out_path = OUT / f"transcripts_{name}.jsonl"
    done = set()
    if out_path.exists():
        # Key on (split, filename): 212 test files reuse train filenames.
        done = {(r["split"], r["filename"])
                for r in map(json.loads, out_path.open(encoding="utf-8"))}
    todo = [(s, f) for s, f in clip_list() if (s, f) not in done]
    print(f"[{name}] {len(done)} cached, {len(todo)} to do", flush=True)

    run = {"whisper": whisper_runner, "crisper": crisper_runner}[name]()
    t_start = time.time()
    with out_path.open("a", encoding="utf-8") as fh:
        for i, (split, f) in enumerate(todo, 1):
            t0 = time.time()
            try:
                rec = run(load_audio(split, f))
            except Exception as e:  # keep going; failed clips are retried on resume
                print(f"[{name}] ERROR {f}: {e!r}", flush=True)
                continue
            rec.update(filename=f, split=split, secs=round(time.time() - t0, 2))
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 25 == 0 or i == len(todo):
                rate = (time.time() - t_start) / i
                print(f"[{name}] {i}/{len(todo)}  {rate:.1f}s/clip  "
                      f"ETA {rate * (len(todo) - i) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
