# Grammar Scoring Engine for Spoken English — SHL Hiring Assessment 2026

Predict a **1–5 grammar score** (averaged human MOS ratings) from a 45–60 s English speech clip.
Kaggle: `shl-hiring-assessment-2026` · metric: RMSE (leaderboard) and Pearson correlation.

**Approach in one line:** *listen → write down exactly what was said → measure grammar errors and fluency → learn the mapping to the human score.*

```
audio ─┬─► Whisper large-v3-turbo ──► text, word timings, confidence ─┐
       │                                                                ├─► ~45 features ─► Ridge / SVR / LightGBM ─► NNLS blend ─► score
       └─► CrisperWhisper 2.0 turbo ─► verbatim text (errors kept) ─────┘   + Claude rubric judge
                                                                            + sentence embedding
```

## Results

<!-- RESULTS_TABLE -->

*CV = repeated stratified 5-fold (×3) out-of-fold predictions on the 732 scored training clips.
"Training fit" = model refit on all training clips and scored on them (required by the challenge; optimistic by construction).*

## Key findings (what mattered most)

1. **The 0.0 labels are an unscored batch.** All 37 clips labelled 0.0 are exactly the files numbered ≥ 5000, every such file is 0.0, none appear in test, and their speech is ordinary (some fluent). They were removed and predictions clipped to 1–5.
2. **212 test filenames collide with train filenames** (`audio_0.wav` exists in both folders) but are different recordings. All caches are keyed by `split/filename`; keying by filename silently feeds train audio features to test clips.
3. **Standard ASR hides grammar errors.** On a 20-clip side-by-side, Whisper inserted missing articles (*"has goal"* → *"has a goal"*), dropped repetitions and cut-off words, and once hallucinated a sentence. CrisperWhisper's verbatim mode keeps them, so it is used for the disfluency/grammar features.
4. **An LLM judge is the strongest single signal**, but only as a feature. Claude scoring transcripts against the official rubric correlates r ≈ 0.69 with the labels on its own; the regressors calibrate it and combine it with acoustic fluency evidence it cannot see.

## Pipeline

| Step | Script | What / why |
|---|---|---|
| 0. ASR comparison | `src/asr_compare.py` | Whisper vs CrisperWhisper on 20 stratified clips → choose ASR on evidence |
| 1. Transcription | `src/transcribe.py whisper` | faster-whisper large-v3-turbo (int8, verbatim prompt, no cross-segment conditioning, VAD, word timestamps). ~2 s/clip on an RTX 3050 4 GB |
| | `src/transcribe.py crisper` | CrisperWhisper 2.0 turbo, *verbatim* mode (fillers `[UM]`/`[UH]`, repetitions, cut-offs kept). PyTorch backend on Windows |
| 2. LLM judge | `src/llm_judge.py whisper [RUN]` | Claude (Sonnet) via Claude Code headless, batches of 20, rubric in the prompt, JSON out; 2 runs with different batching averaged |
| 3. Features | `src/features.py` | see table below; expensive features cached per clip |
| 4–5. Models + blend | `src/train.py` | Ridge, Ridge+embedding, RBF-SVR, LightGBM; non-negative least-squares blend on OOF predictions; writes submission |
| Report | `notebooks/grammar_scoring.ipynb` | built by `src/build_notebook.py`; charts, CV + training RMSE, error analysis |

### Features

| Group | Features | Rubric link |
|---|---|---|
| Fluency (Whisper word timings) | words/sec, articulation rate, pause ratio, pauses > 0.3 s / > 1 s per 100 words, longest pause | hesitation while building sentences |
| ASR confidence | segment log-prob, word-probability mean / 10th pct, share of low-confidence words | broken structures are hard to decode |
| Lexical | word count, type-token ratio, word length, sentence length | control of complex structures |
| LanguageTool | grammar-category errors per 100 words (casing/punctuation/style ignored) | direct error count |
| GPT-2 | mean and 90th-pct token negative log-likelihood | ungrammatical sequences are improbable |
| RoBERTa-CoLA | per-sentence grammatical-acceptability probability | grammar-specific classifier |
| Claude judge | rubric score 1–5, error count, errors per 100 words | closest to the human raters' task |
| CrisperWhisper verbatim | filler, cut-off and repeat rates; similarity of verbatim to Whisper's smoother text | self-corrections, incomplete sentences |
| Embedding | all-mpnet-base-v2 sentence embedding (768-d) | overall language quality |

## Reproduce

```bash
python -m venv .venv && .venv\Scripts\activate            # Windows
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# put the Kaggle data in data/Dataset_Final/{train,test,train.csv,test.csv}
# LanguageTool needs Java 17+ (a portable JRE in tools/ works; path set in src/features.py)

python src/transcribe.py whisper        # ~30 min on RTX 3050
python src/transcribe.py crisper        # several hours on RTX 3050 (PyTorch backend)
python src/llm_judge.py whisper         # needs the Claude Code CLI (claude -p)
python src/llm_judge.py whisper 1
python src/features.py
python src/train.py --emb whisper_mpnet --tag final
python src/build_notebook.py final && jupyter nbconvert --to notebook --execute --inplace notebooks/grammar_scoring.ipynb
```

Cached Claude scores are committed in `outputs/llm_scores_*.csv`, so the modelling steps re-run without Claude access.

## Notes and limitations

* **Data is not included** — the competition rules forbid redistribution.
* **Licences:** Whisper (MIT); CrisperWhisper 2.0 weights *and outputs* are under a non-commercial research licence — fine for this assessment, but a production system would need a commercial licence or the Whisper-only path. LanguageTool (LGPL), GPT-2 (MIT), RoBERTa-CoLA, all-mpnet-base-v2 (Apache-2.0).
* **Claude judge:** transcripts were sent to Claude (the rules allow external LLMs/APIs). Scores are averaged over runs with different batching to reduce batch-relative noise.
* **Limitations:** ASR errors on heavily accented speech can look like grammar errors; LanguageTool and CoLA are built for written text; ~730 noisy averaged labels cap what any model can learn.
* **Next steps:** fine-tune a small encoder on verbatim transcripts with 5-fold CV; acoustic embeddings (WavLM) for fluency cues independent of ASR.
