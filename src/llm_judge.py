"""Score transcripts against the SHL grammar rubric with Claude (rubric-aware LLM judge).

Calls the Claude Code CLI in headless mode (`claude -p`), batching 20 transcripts
per call. The LLM score is used as ONE FEATURE in the regressors, which calibrate
it against the human labels; it is not used as the prediction on its own.

Usage:  python src/llm_judge.py whisper [RUN]   # judge Whisper transcripts
        python src/llm_judge.py crisper [RUN]   # judge CrisperWhisper verbatim transcripts
        python src/llm_judge.py whisper --fewshot   # + 11 labelled anchor examples
RUN > 0 reshuffles the batches (seed = RUN); features.py averages all runs.
Output: outputs/llm_scores_<source>[_fs][_r<RUN>].csv  (key, llm_score, llm_errors), resumable.
"""
import json
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import clean_crisper, load_jsonl  # noqa: E402

OUT = Path("outputs")
MODEL = "claude-sonnet-5-5"
BATCH, WORKERS = 20, 3

PROMPT = """You are an expert English examiner rating SPOKEN grammar.
Each item is an automatic speech-recognition transcript of a 45-60 second spoken answer.
Punctuation and capitalisation come from the ASR and are unreliable - ignore them.
Fillers like um/uh are normal in speech; only penalise them when they break sentence structure.

Rubric (Grammar Score):
1 = struggles with sentence structure and syntax; limited control of simple structures, relies on memorised patterns.
2 = limited understanding of sentence structure; uses simple structures but consistently makes basic grammatical mistakes; may leave sentences incomplete.
3 = decent grasp of sentence structure but errors in grammatical structure, or decent grammar but errors in syntax/structure.
4 = strong understanding of sentence structure and syntax; good control of grammar; occasional minor errors that do not cause misunderstanding; corrects most of them.
5 = high grammatical accuracy and adept control of complex grammar; seldom makes noticeable mistakes; handles complex structures well and self-corrects.

For every item return a score from 1.0 to 5.0 (one decimal, use the full range) and the number of
distinct grammatical errors you found. Judge each item independently.

Reply with ONLY a JSON object, no prose:
{"results": [{"id": <id>, "score": <float>, "errors": <int>}, ...]}

Items:
"""


ANCHOR_LEVELS = [2.0, 2.0, 2.5, 3.0, 3.0, 3.5, 4.0, 4.0, 4.5, 5.0, 5.0]


def pick_anchors(texts, seed=7):
    """One or two real training clips per score level, shown to the judge with their
    official scores so it learns how the human raters apply the rubric."""
    tr = pd.read_csv("data/Dataset_Final/train.csv")
    tr["key"] = "train/" + tr.filename
    tr = tr[tr.key.isin(texts.index)]
    picks = []
    for lvl in sorted(set(ANCHOR_LEVELS)):
        n = ANCHOR_LEVELS.count(lvl)
        picks += tr[tr.label == lvl].sample(n, random_state=seed).key.tolist()
    return picks


def fewshot_block(texts, anchors, labels):
    lines = ["\nReference examples scored by the official human raters "
             "(use them to calibrate your scale):"]
    lines += [f"[score {labels[k]:.1f}] {texts[k]}" for k in anchors]
    return "\n".join(lines) + "\n\n"


def judge_batch(batch, prefix=""):
    """batch: list of (id, text). Returns {id: (score, errors)}."""
    items = "\n".join(f"[id {i}] {t}" for i, t in batch)
    cmd = [shutil.which("claude"), "-p", "--model", MODEL, "--output-format", "text"]
    for attempt in range(3):
        r = subprocess.run(cmd, input=PROMPT.replace("Items:\n", prefix + "Items:\n") + items,
                           capture_output=True, text=True, encoding="utf-8", timeout=600)
        m = re.search(r"\{.*\}", r.stdout, re.S)
        try:
            res = json.loads(m.group(0))["results"]
            got = {int(x["id"]): (float(x["score"]), int(x["errors"])) for x in res}
            if set(got) == {i for i, _ in batch}:
                return got
        except Exception:
            pass
        print(f"  retry {attempt + 1}: {r.stdout[:200]!r} {r.stderr[:200]!r}", flush=True)
    return {}


def main(source, run=0, fewshot=False):
    df = load_jsonl(source)
    texts = df["text"] if source == "whisper" else df["verbatim"].map(clean_crisper)
    prefix = ""
    if fewshot:
        # Anchor clips are listed in llm_anchors_<source>.json; features.py blanks
        # their few-shot scores so their labels cannot leak into cross-validation.
        labels = pd.read_csv("data/Dataset_Final/train.csv").set_index("filename").label
        labels.index = "train/" + labels.index
        anchors = pick_anchors(texts)
        (OUT / f"llm_anchors_{source}.json").write_text(json.dumps(anchors))
        prefix = fewshot_block(texts, anchors, labels)
    # Each run uses a different random batching; averaging runs reduces the noise
    # from a score being judged relative to the other items in its batch.
    suffix = ("_fs" if fewshot else "") + ("" if run == 0 else f"_r{run}")
    out_path = OUT / f"llm_scores_{source}{suffix}.csv"
    done = set(pd.read_csv(out_path).key) if out_path.exists() else set()
    keys = [k for k in texts.index if k not in done]
    if run:
        keys = list(pd.Series(keys).sample(frac=1, random_state=run))
    print(f"[{source}] {len(done)} cached, {len(keys)} to judge", flush=True)

    # Ids are batch positions, so the model never sees filenames or splits.
    batches = [keys[i:i + BATCH] for i in range(0, len(keys), BATCH)]

    def work(bkeys):
        got = judge_batch([(j, texts[k]) for j, k in enumerate(bkeys)], prefix)
        return [(k, *got[j]) for j, k in enumerate(bkeys) if j in got]

    with ThreadPoolExecutor(WORKERS) as ex:
        for n, rows in enumerate(ex.map(work, batches), 1):
            pd.DataFrame(rows, columns=["key", "llm_score", "llm_errors"]).to_csv(
                out_path, mode="a", header=not out_path.exists(), index=False)
            print(f"[{source}] batch {n}/{len(batches)}  (+{len(rows)})", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--fewshot"]
    main(args[0], int(args[1]) if len(args) > 1 else 0, fewshot="--fewshot" in sys.argv)
