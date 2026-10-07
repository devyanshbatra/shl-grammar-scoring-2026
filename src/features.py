"""Turn cached transcripts into a per-clip feature table.

Feature groups
  fluency   speaking rate, pauses, fillers, repetitions (Whisper word timings)
  asr       Whisper confidence: segment log-prob, word probabilities
  lexical   word count, vocabulary variety, sentence length
  grammar   LanguageTool error rate (grammar-type rules only)
  lm        GPT-2 perplexity of the transcript (ungrammatical text is less likely)
  cola      RoBERTa-CoLA grammatical-acceptability probability per sentence
  llm       Claude rubric-judge score and error count (src/llm_judge.py)
  crisper   CrisperWhisper verbatim: fillers, cut-offs, repeats, gap to Whisper's cleaner text
  emb       sentence embedding of the transcript (kept as a separate matrix)

Usage:  python src/features.py   ->  outputs/features.parquet, outputs/emb_<name>.npy
"""
import difflib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("outputs")
JAVA_HOME = Path("tools/jdk-21.0.12.1+1-jre").resolve()
FILLERS = {"um", "umm", "uh", "uhh", "hmm", "erm", "er", "ah", "mm"}
WORD_RE = re.compile(r"[a-z']+")


def load_jsonl(name):
    """Index is 'split/filename': 212 test files reuse train filenames
    (e.g. audio_0.wav) but are different recordings."""
    path = OUT / f"transcripts_{name}.jsonl"
    if not path.exists():
        return pd.DataFrame()
    df = pd.DataFrame([json.loads(l) for l in path.open(encoding="utf-8")])
    df["key"] = df["split"] + "/" + df["filename"]
    return df.drop_duplicates("key", keep="last").set_index("key")


def tokens(text):
    return WORD_RE.findall(text.lower())


def repeat_count(toks):
    """Immediate word repeats ('the the', 'I I')."""
    return sum(a == b for a, b in zip(toks, toks[1:]))


# ---------------------------------------------------------------- Whisper based
def whisper_features(rec):
    words, segs, text = rec["words"], rec["segments"], rec["text"]
    toks = tokens(text)
    n = max(len(toks), 1)
    f = {}
    if words:
        starts = np.array([w["s"] for w in words])
        ends = np.array([w["e"] for w in words])
        probs = np.array([w["p"] for w in words])
        gaps = np.clip(starts[1:] - ends[:-1], 0, None) if len(words) > 1 else np.zeros(1)
        speech_span = ends[-1] - starts[0]
        voiced = np.sum(ends - starts)
        f.update(
            speech_span=speech_span,
            words_per_sec=len(words) / max(speech_span, 1e-3),
            articulation_rate=len(words) / max(voiced, 1e-3),  # rate excluding pauses
            pause_ratio=1 - voiced / max(speech_span, 1e-3),
            pauses_03=np.sum(gaps > 0.3) / n * 100,  # per 100 words
            pauses_1s=np.sum(gaps > 1.0) / n * 100,
            mean_gap=gaps.mean(),
            max_gap=gaps.max(),
            word_prob_mean=probs.mean(),
            word_prob_p10=np.percentile(probs, 10),
            low_prob_frac=np.mean(probs < 0.5),
        )
    if segs:
        lp = np.array([s["avg_logprob"] for s in segs])
        f.update(
            seg_logprob_mean=lp.mean(),
            seg_logprob_min=lp.min(),
            no_speech_max=max(s["no_speech_prob"] for s in segs),
            compression_ratio_mean=np.mean([s["compression_ratio"] for s in segs]),
        )
    sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    content = [t for t in toks if t not in FILLERS]
    f.update(
        n_words=len(toks),
        n_content_words=len(content),
        filler_rate=sum(t in FILLERS for t in toks) / n * 100,
        repeat_rate=repeat_count(content) / n * 100,
        ttr=len(set(content)) / max(len(content), 1),  # vocabulary variety
        mean_word_len=np.mean([len(t) for t in content]) if content else 0,
        mean_sent_len=len(content) / max(len(sents), 1),
    )
    return f


# --------------------------------------------------------------- Crisper based
def clean_crisper(text):
    """Drop [UM]/[UH]/[laughter] tags for text-level comparisons."""
    return re.sub(r"\[[^\]]*\]", " ", text)


def crisper_features(verb, reference):
    """Verbatim transcript vs a cleaner reference transcript.

    The reference is Whisper's text for every clip: Whisper smooths speech, and on
    115 clips transcribed in both CrisperWhisper modes the verbatim-vs-Whisper gap
    tracked the verbatim-vs-intended gap with r = 0.93, so the slower intended
    pass was dropped.
    """
    vt = tokens(clean_crisper(verb))
    it = tokens(reference)
    n = max(len(vt), 1)
    sm = difflib.SequenceMatcher(a=vt, b=it, autojunk=False)
    return {
        "cw_n_words": len(vt),
        "cw_filler_rate": len(re.findall(r"\[(UM|UH)\]", verb, re.I)) / n * 100,
        "cw_cutoff_rate": len(re.findall(r"\b\w+-(?=\s|$)", verb)) / n * 100,  # 'th-'
        "cw_repeat_rate": repeat_count(vt) / n * 100,
        "cw_event_count": len(re.findall(r"\[(?!UM|UH)[^\]]+\]", verb, re.I)),
        "cw_clean_ratio": len(it) / n,     # how much the clean version shrinks
        "cw_edit_sim": sm.ratio(),         # 1.0 = verbatim already clean
        "cw_edit_ops": sum(1 for op in sm.get_opcodes() if op[0] != "equal") / n * 100,
    }


def cached(name, texts, fn):
    """Run an expensive per-clip feature function only on clips not yet cached."""
    path = OUT / f"cache_{name}.parquet"
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    todo = texts[~texts.index.isin(old.index)]
    if len(todo):
        new = fn(todo)
        old = pd.concat([old, new]) if len(old) else new
        old.to_parquet(path)
    return old.loc[texts.index]


# ------------------------------------------------------------------- Grammar
def languagetool_features(texts):
    os.environ["JAVA_HOME"] = str(JAVA_HOME)
    os.environ["PATH"] = str(JAVA_HOME / "bin") + os.pathsep + os.environ["PATH"]
    import language_tool_python as lt

    # Spoken transcripts have unreliable casing/punctuation, so ignore those rules.
    skip = {"CASING", "PUNCTUATION", "TYPOGRAPHY", "TYPOS", "STYLE", "REDUNDANCY"}
    tool = lt.LanguageTool("en-US")
    out = {}
    for fn, text in texts.items():
        matches = [m for m in tool.check(text) if m.category not in skip]
        n = max(len(tokens(text)), 1)
        out[fn] = {
            "lt_err_rate": len(matches) / n * 100,
            "lt_grammar_rate": sum(m.category == "GRAMMAR" for m in matches) / n * 100,
        }
    tool.close()
    return pd.DataFrame.from_dict(out, orient="index")


def perplexity_features(texts, model_name="gpt2"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModelForCausalLM.from_pretrained(model_name).eval()
    out = {}
    with torch.no_grad():
        for fn, text in texts.items():
            ids = tok(text, return_tensors="pt", truncation=True, max_length=512).input_ids
            if ids.shape[1] < 2:
                out[fn] = {"lm_nll": np.nan, "lm_nll_p90": np.nan}
                continue
            logits = lm(ids).logits[0, :-1]
            nll = torch.nn.functional.cross_entropy(logits, ids[0, 1:], reduction="none")
            out[fn] = {"lm_nll": nll.mean().item(),  # log-perplexity
                       "lm_nll_p90": np.percentile(nll.numpy(), 90)}
    return pd.DataFrame.from_dict(out, orient="index")


def cola_features(texts, model_name="textattack/roberta-base-CoLA"):
    """Grammatical acceptability per sentence from a RoBERTa classifier trained on
    CoLA (sentences labelled grammatical / ungrammatical). Unlike general sentence
    embeddings, which mostly encode meaning, this model targets grammar directly."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    clf = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    out = {}
    with torch.no_grad():
        for fn, text in texts.items():
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 3]
            if not sents:
                out[fn] = {"cola_mean": np.nan, "cola_min": np.nan, "cola_bad_frac": np.nan}
                continue
            enc = tok(sents, return_tensors="pt", padding=True, truncation=True, max_length=128)
            p = clf(**enc).logits.softmax(-1)[:, 1].numpy()  # label 1 = acceptable
            out[fn] = {"cola_mean": p.mean(), "cola_min": p.min(), "cola_bad_frac": np.mean(p < 0.5)}
    return pd.DataFrame.from_dict(out, orient="index")


# bge-large / Qwen3-Embedding-0.6B were considered but not used: once the Claude
# judge was added, the mpnet embedding was already largely redundant.
EMB_MODELS = {
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}


def embed(texts, model_name):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    return model.encode(list(texts.values), batch_size=8, show_progress_bar=True,
                        normalize_embeddings=True)


def save_embeddings(texts, source):
    """One matrix per (transcript source, embedding model); cached on disk."""
    for short, model_name in EMB_MODELS.items():
        name = f"{source}_{short}"
        if (OUT / f"emb_{name}.npy").exists():
            continue
        print(f"embedding {name} ...", flush=True)
        np.save(OUT / f"emb_{name}.npy", embed(texts, model_name))
        pd.Series(texts.index).to_csv(OUT / f"emb_{name}_index.csv", index=False)


def llm_features(source, prefix="", fewshot=False):
    """Claude rubric-judge scores from src/llm_judge.py, averaged over all
    completed runs (each run batches the transcripts differently).

    Few-shot runs (prompt includes labelled anchor clips) are kept as separate
    features, and the anchors' own few-shot scores are blanked so their labels
    cannot leak into cross-validation."""
    n_clips = len(load_jsonl(source))
    paths = [p for p in sorted(OUT.glob(f"llm_scores_{source}*.csv"))
             if ("_fs" in p.name) == fewshot]
    runs = []
    for path in paths:
        df = pd.read_csv(path).drop_duplicates("key", keep="last").set_index("key")
        if len(df) == n_clips:  # skip runs still in progress
            runs.append(df[["llm_score", "llm_errors"]])
    if not runs:
        return None
    print(f"llm[{source}{'_fs' if fewshot else ''}]: averaging {len(runs)} run(s)")
    out = sum(runs) / len(runs)
    if fewshot:
        anchors = json.loads((OUT / f"llm_anchors_{source}.json").read_text())
        out.loc[out.index.isin(anchors)] = np.nan
        prefix += "fs_"
    return out.add_prefix(prefix)


def main(with_embeddings=True):
    wh = load_jsonl("whisper")
    cw = load_jsonl("crisper")
    print(f"whisper transcripts: {len(wh)}  crisper transcripts: {len(cw)}")

    feats = pd.DataFrame.from_dict(
        {fn: whisper_features(r) for fn, r in wh.iterrows()}, orient="index")
    feats["split"] = wh["split"]
    feats = feats.join(cached("lt_whisper", wh["text"], languagetool_features))
    feats = feats.join(cached("lm_whisper", wh["text"], perplexity_features))
    feats = feats.join(cached("cola_whisper", wh["text"], cola_features))
    if len(cw):
        cwf = pd.DataFrame.from_dict(
            {k: crisper_features(r["verbatim"], wh.loc[k, "text"]) for k, r in cw.iterrows()},
            orient="index")
        feats = feats.join(cwf)
        cw_lt = cached("lt_crisper", cw["verbatim"].map(clean_crisper),
                       languagetool_features).add_prefix("cw_")
        feats = feats.join(cw_lt)
    for src, prefix in [("whisper", ""), ("crisper", "cw_")]:
        for fewshot in (False, True):
            llm = llm_features(src, prefix, fewshot)
            if llm is None:
                continue
            feats = feats.join(llm)
            p = prefix + ("fs_" if fewshot else "")
            n = feats[f"{prefix}n_words"].clip(lower=1)
            feats[f"{p}llm_err_rate"] = feats[f"{p}llm_errors"] / n * 100
    feats.index.name = "key"
    feats.to_parquet(OUT / "features.parquet")
    print("features:", feats.shape)
    if not with_embeddings:
        return

    save_embeddings(wh["text"], "whisper")
    if len(cw) == len(wh):  # only once CrisperWhisper has covered every clip
        save_embeddings(cw["verbatim"].map(clean_crisper), "crisper")


if __name__ == "__main__":
    import sys
    main(with_embeddings="--no-emb" not in sys.argv)
