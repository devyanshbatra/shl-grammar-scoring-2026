"""Unattended end-of-pipeline runner.

Waits for the long jobs (WavLM embeddings, CrisperWhisper transcripts), then:
  1. picks the best WavLM layer by CV
  2. rebuilds features (incl. CrisperWhisper features)
  3. runs the few-shot Claude judge on CrisperWhisper verbatim text (skipped on failure)
  4. trains every candidate configuration and keeps the lowest CV RMSE
  5. submits that one to Kaggle and builds/executes the report notebook

Log: outputs/auto_finish.log
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path("outputs")
LOG = OUT / "auto_finish.log"
PY = sys.executable
N_CLIPS = 948


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(args, timeout=None, env_extra=None):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "TRANSFORMERS_VERBOSITY": "error",
           "HF_HUB_DISABLE_SYMLINKS_WARNING": "1", **(env_extra or {})}
    log("RUN " + " ".join(map(str, args)))
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout, env=env)
    tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-25:])
    log(f"exit {r.returncode}\n{tail}")
    return r


def n_lines(path):
    return sum(1 for _ in path.open(encoding="utf-8")) if path.exists() else 0


def wait_for(cond, what, poll=120):
    while not cond():
        log(f"waiting for {what} ...")
        time.sleep(poll)
    log(f"ready: {what}")


def cv_rmse(tag):
    m = json.loads((OUT / f"metrics_{tag}.json").read_text())
    return m["cv"]["blend"]["rmse"]


def main():
    cpu = {"CUDA_VISIBLE_DEVICES": ""}

    # 1. WavLM layer choice -------------------------------------------------
    wait_for(lambda: (OUT / "wavlm_layers.npy").exists(), "WavLM embeddings", 60)
    r = run([PY, "src/train.py", "--no-crisper", "--scan-audio"], env_extra=cpu)
    scores = {int(m.group(1)): float(m.group(2))
              for m in re.finditer(r"layer\s+(\d+)\s+RMSE ([\d.]+)", r.stdout)}
    best_layer = min(scores, key=scores.get) if scores else None
    log(f"best WavLM layer: {best_layer} ({scores.get(best_layer)})")

    # Quick win while CrisperWhisper still runs: text features + audio.
    if best_layer is not None:
        run([PY, "src/train.py", "--no-crisper", "--audio-layer", str(best_layer),
             "--tag", "v5_audio"], env_extra=cpu)

    # 2. CrisperWhisper finished -> full feature rebuild --------------------
    # If the CrisperWhisper job dies (e.g. its parent session ends), restart it;
    # transcribe.py resumes from its cache.
    path = OUT / "transcripts_crisper.jsonl"
    last, stalled = n_lines(path), 0
    while n_lines(path) < N_CLIPS:
        time.sleep(300)
        now = n_lines(path)
        stalled = stalled + 1 if now == last else 0
        last = now
        log(f"CrisperWhisper {now}/{N_CLIPS}")
        if stalled >= 3:  # no progress for 15 min
            log("CrisperWhisper stalled -> restarting")
            subprocess.Popen([PY, "-u", "src/transcribe.py", "crisper"],
                             stdout=open(OUT / "crisper_restart.log", "a"),
                             stderr=subprocess.STDOUT)
            stalled = 0
    log("ready: CrisperWhisper transcripts")
    run([PY, "src/features.py"], timeout=4 * 3600, env_extra=cpu)

    # 3. Claude few-shot judge on verbatim text (needs Claude Code quota) ----
    try:
        run([PY, "src/llm_judge.py", "crisper", "--fewshot"], timeout=2 * 3600)
        run([PY, "src/features.py", "--no-emb"], timeout=3600, env_extra=cpu)
    except Exception as e:  # quota / CLI problems must not block the submission
        log(f"crisper judge skipped: {e!r}")

    # 4. Candidate configurations, chosen by CV RMSE ------------------------
    cands = {"v6_cw": ["--emb", "whisper_mpnet"]}
    if best_layer is not None:
        cands["v6_cw_audio"] = ["--emb", "whisper_mpnet", "--audio-layer", str(best_layer)]
    for tag, extra in cands.items():
        run([PY, "src/train.py", *extra, "--tag", tag], env_extra=cpu)
    pool = [t for t in ["v4_fs", "v5_audio", *cands] if (OUT / f"metrics_{t}.json").exists()]
    ranked = sorted(pool, key=cv_rmse)
    log("CV ranking: " + ", ".join(f"{t}={cv_rmse(t):.4f}" for t in ranked))
    best = ranked[0]

    # 5. Submit + notebook ---------------------------------------------------
    if best != "v4_fs":
        kaggle = str(Path(PY).parent / "kaggle.exe")
        run([kaggle, "competitions", "submit", "shl-hiring-assessment-2026",
             "-f", str(OUT / f"submission_{best}.csv"),
             "-m", f"{best}: auto-selected by CV RMSE {cv_rmse(best):.4f}"])
    (OUT / "best_tag.txt").write_text(best)
    run([PY, "src/build_notebook.py", best])
    run([PY, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
         "--ExecutePreprocessor.timeout=1800", "notebooks/grammar_scoring.ipynb"],
        env_extra=cpu)
    log(f"DONE. best = {best}")


if __name__ == "__main__":
    main()
