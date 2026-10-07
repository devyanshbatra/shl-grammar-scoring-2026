# Grammar Scoring Engine for Spoken English — SHL Hiring Assessment 2026

Predict a **1–5 grammar score** (averaged human MOS ratings) from a 45–60 s English speech clip.
Kaggle: `shl-hiring-assessment-2026` · Kaggle username: **devyanshbatra** · leaderboard metric: RMSE (lower is better); Pearson also reported.

**Approach in one line:** *listen → write down what was said → measure grammar and fluency evidence, from both the text and the sound → learn the mapping to the human score.*

```
audio ─┬─► Whisper large-v3-turbo ──► transcript, word timings, confidence ─┐
       ├─► CrisperWhisper 2.0 turbo ─► verbatim transcript (errors kept) ────┼─► text features + Claude rubric judge ─┐
       ├─► WavLM-base-plus ──────────► acoustic embedding (layer 11) ─────────┤                                        ├─► Ridge / SVR / LightGBM ─► NNLS blend ─► score
       └─► Whisper encoder ──────────► acoustic embedding (layer 27, v7) ─────┘                                        │
```

## Final results

| Final pick | Model | CV RMSE | Speaker-grouped CV | Public LB | Training-data RMSE* |
|---|---|---|---|---|---|
| **#1 — v5** | Whisper text features + Claude rubric judge + WavLM-base audio | 0.542 (r 0.848) | 0.558 | **0.3376** (≈ rank 15 / 247) | 0.387 |
| **#2 — v7** | v6 (+ CrisperWhisper verbatim text, Claude on verbatim) + Whisper-encoder audio | **0.504** (r 0.871) | **0.523** | 0.3730 | 0.327 |

`submission.csv` = v5, reproduced by `notebooks/grammar_scoring.ipynb`.

\* *Training-data RMSE (required by the challenge): models refit on all 732 scored training clips and evaluated on those same clips — optimistic by construction; the CV columns are the honest estimates.*
*CV = repeated stratified 5-fold ×3 out-of-fold. Speaker-grouped CV = same, but a voice never appears in both the fitting and validation part (see below).*

**Why two different picks:** v5 has the best public score; v7 is best on both validation schemes. The public leaderboard scores only a subset of the 216 test clips (noise ≈ ±0.02–0.03 RMSE), so the picks hedge both views.

## Experiment log — everything tried

| Ver. | Change | CV | Grouped CV | Public | Outcome |
|---|---|---|---|---|---|
| v1 | Whisper text features + mpnet embedding; Ridge/LightGBM blend | 0.647 | – | 0.4963 | baseline |
| v2 | + Claude rubric judge (zero-shot), + SVR | 0.6025 | – | 0.4079 | **kept** — biggest text gain |
| v3 | + second Claude run averaged, + RoBERTa-CoLA | 0.6055 | – | – | neutral → not submitted |
| v4 | + few-shot Claude (11 rater-scored anchors, their own scores blanked) | 0.6022 | 0.607 | 0.4033 | **kept** — better-calibrated judge |
| **v5** | + **WavLM-base-plus audio** (first 30 s, layer 11) | 0.542 | 0.558 | **0.3376** | **final pick #1** |
| v5b | + RBF-SVR on the audio embedding | 0.536 | – | 0.3501 | better CV, worse public |
| v6 | + CrisperWhisper verbatim features; Claude on verbatim (r 0.717 → 0.738) | 0.530 | 0.551 | 0.3476 | better CV; public within noise |
| **v7** | + Whisper-turbo encoder embedding (layer 27) | **0.504** | **0.523** | 0.3730 | **final pick #2** |
| v8 | + Ridge on features + both audio embeddings | 0.503 | – | 0.3730 | no gain over v7 |
| v9 | 50/50 average of v5 and v7 predictions | – | – | 0.3473 | between the two |
| v10a | v5 recipe, WavLM-base on full 60 s (layer 7) | 0.527 | 0.553 | 0.3521 | better CV, worse public |
| v10b | v6 + WavLM 60 s | 0.509 | – | 0.3659 | better CV, worse public |
| v11 | v5 recipe, **WavLM-large** (layer 21 of 25, 60 s) | 0.516 | 0.541 | 0.3671 | better CV, worse public |

Also tried and dropped: bge-large / Qwen3 text embeddings (redundant once the Claude judge was in; CPU-bound), per-split mean normalisation of audio embeddings (worse on a test-like holdout), and one probe submission predicting the train mean (public 1.023) to confirm the test score spread.

## Key findings

1. **The 0.0 labels are an unscored batch.** All 37 clips labelled 0.0 are exactly the files numbered ≥ 5000, every such file is 0.0, none are in test, and their speech is ordinary. Removed; predictions clipped to 1–5.
2. **212 test filenames collide with train filenames** (`audio_0.wav` exists in both folders) but are different recordings. Everything is keyed by `split/filename`.
3. **Standard ASR hides grammar errors.** Whisper inserted missing articles (*"has goal"* → *"has a goal"*), dropped repetitions and cut-offs, and once hallucinated text. CrisperWhisper's verbatim mode keeps them; Claude judging verbatim text correlates better with the labels (0.738 vs 0.717).
4. **An LLM judge is the strongest text signal — as a feature.** Claude with the official rubric and 11 rater-scored anchors (few-shot) correlates r ≈ 0.72 with the labels; the regressors calibrate it and combine it with evidence it cannot see.
5. **The text model could not separate the extremes** (true 2.0 predicted ≈ 2.7, true 5.0 ≈ 4.4). Raters *listen*, so self-supervised speech embeddings were added: WavLM layer 11 alone reaches r 0.79; with it public RMSE fell 0.403 → 0.338. Upper layers (word-level / prosodic structure) beat lower acoustic layers; the last layer dips (pre-training-specific).
6. **CV vs leaderboard.** From v5 on, better CV came with worse public scores. Investigated:
   * *Adversarial validation:* train vs test separable (AUC 0.71 on text features → 0.83 on Whisper-encoder audio).
   * *Speaker overlap:* nearest-voice cosine (WavLM layer 2) train→train median **0.87** (265/732 clips > 0.9) vs test→train **0.62** (1/216). Training speakers repeat; test speakers are new.
   * *Speaker-grouped CV* (489 voice groups): audio models lose 0.015–0.025 (text-only only 0.005) — leakage confirmed — but the ranking holds. Remaining public differences are treated as subset noise.

## Pipeline

| Step | Script | What / why |
|---|---|---|
| ASR comparison | `src/asr_compare.py` | Whisper vs CrisperWhisper on 20 stratified clips → choose ASR on evidence |
| Transcription | `src/transcribe.py whisper` | faster-whisper large-v3-turbo (int8, verbatim prompt, no cross-segment conditioning, VAD, word timestamps); ~2 s/clip on an RTX 3050 4 GB |
| | `src/transcribe.py crisper` | CrisperWhisper 2.0 turbo, verbatim mode; PyTorch backend on Windows (~15–25 s/clip) |
| LLM judge | `src/llm_judge.py <whisper\|crisper> [RUN] [--fewshot]` | Claude (Sonnet) via Claude Code headless, batches of 20, rubric (+ anchors) in the prompt, JSON out |
| Text features | `src/features.py` | fluency, ASR confidence, lexical, LanguageTool, GPT-2, CoLA, Claude, CrisperWhisper; cached per clip |
| Audio embeddings | `src/audio_embed.py` | WavLM (base-plus / large), mean-pooled per layer; env vars pick model, seconds, device |
| | `src/whisper_embed.py` | Whisper-turbo encoder (CrisperWhisper weights) on the GPU, mean-pooled per layer |
| Models + blend | `src/train.py` | Ridge / SVR / LightGBM on features, text embedding and audio embeddings; NNLS blend on OOF; `--scan-audio`, `--group-cv`, `--models` |
| Shift analysis | `src/shift_eval.py`, `src/speakers.py` | adversarial validation, test-like holdout, voice groups for grouped CV |
| Orchestration | `src/auto_finish.py` | unattended end-of-pipeline runner used while long jobs ran |
| Report | `notebooks/grammar_scoring.ipynb` | built by `src/build_notebook.py`; reproduces v5 with charts, CV + training RMSE, experiment log |

### Features

| Group | Features | Rubric link |
|---|---|---|
| Fluency (Whisper word timings) | words/sec, articulation rate, pause ratio, pauses > 0.3 s / > 1 s per 100 words, longest pause | hesitation while building sentences |
| ASR confidence | segment log-prob, word-probability mean / 10th pct, share of low-confidence words | broken structures are hard to decode |
| Lexical | word count, type-token ratio, word length, sentence length | control of complex structures |
| LanguageTool | grammar-category errors per 100 words (casing/punctuation/style ignored) | direct error count |
| GPT-2 | mean and 90th-pct token negative log-likelihood | ungrammatical sequences are improbable |
| RoBERTa-CoLA | per-sentence grammatical-acceptability probability | grammar-specific classifier |
| Claude judge | rubric score 1–5, error count, errors per 100 words (zero-shot ×2 and few-shot) | closest to the raters' task |
| CrisperWhisper verbatim | filler, cut-off and repeat rates; similarity of verbatim to Whisper's cleaner text | self-corrections, incomplete sentences |
| Text embedding | all-mpnet-base-v2 (768-d) | overall language quality |
| Audio embedding | WavLM layer / Whisper-encoder layer, mean over time | how the speech sounds: pronunciation, rhythm, fluency |

## Reproduce

```bash
python -m venv .venv && .venv\Scripts\activate            # Windows
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# Kaggle data in data/Dataset_Final/{train,test,train.csv,test.csv}
# LanguageTool needs Java 17+ (a portable JRE in tools/ works; path set in src/features.py)

python src/transcribe.py whisper                       # ~30 min on RTX 3050
python src/transcribe.py crisper                       # several hours (PyTorch backend)
python src/llm_judge.py whisper && python src/llm_judge.py whisper 1
python src/llm_judge.py whisper --fewshot && python src/llm_judge.py crisper --fewshot
python src/features.py
python src/audio_embed.py                              # WavLM-base-plus, first 30 s (CPU)
python src/whisper_embed.py                            # Whisper encoder (GPU)
python src/speakers.py                                 # voice groups for --group-cv

# v5 (final pick #1, = submission.csv)
python src/train.py --no-crisper --emb whisper_mpnet --audio-layer 11 \
    --models ridge_feat,ridge_emb,svr_feat,lgbm_feat,ridge_audio --tag v5_exact
# v7 (final pick #2)
python src/train.py --emb whisper_mpnet --audio-layer 11 --whisper-layer 27 --tag v7_wenc

python src/build_notebook.py v5_exact
jupyter nbconvert --to notebook --execute --inplace notebooks/grammar_scoring.ipynb
```

Cached Claude scores (`outputs/llm_scores_*.csv`), metrics, feature importances and every submission are committed, so modelling re-runs without Claude access. Transcripts, embeddings and anything containing training labels are not committed (competition data must not be redistributed).

## Notes and limitations

* **Data is not included** — the competition rules forbid redistribution.
* **Licences:** Whisper (MIT); CrisperWhisper 2.0 weights *and outputs* are under a non-commercial research licence — fine for this assessment, but production use would need a licence or the Whisper-only path (v5 does not depend on CrisperWhisper). WavLM (MIT), LanguageTool (LGPL), GPT-2 (MIT), all-mpnet-base-v2 (Apache-2.0).
* **Claude judge:** transcripts were sent to Claude (the rules allow external LLMs/APIs); prompts are in `src/llm_judge.py`.
* **Limitations:** ~730 noisy averaged labels from repeated speakers; ASR errors on accented speech can look like grammar errors; LanguageTool/CoLA target written text; the public leaderboard is too small to separate models within ~0.03 RMSE.
* **Next steps:** speaker-grouped CV from the start; fine-tune a small encoder on verbatim transcripts; domain-robust audio features (e.g. speaker-normalised embeddings) for unseen speakers.
