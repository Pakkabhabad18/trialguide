"""Memorisation probe (task T14, proposal Section 11: "backbone memorised public TREC patients").

The TREC and SIGIR notes are public, so the backbone may have read them in training.
If it has, a fact we hide by masking could come back from the model's memory instead
of from reasoning, and inference would look better than it is. Three tests, on the
development patients only:

A. Continuation. The model gets the first sentences of a note as plain text (no chat
   template) and continues it. Does it write the real next sentence?
B. Gap filling. One sentence is replaced by [MISSING] and the model is asked, in chat,
   to write the removed sentence exactly. This is the masking set-up itself.
C. Edited age. The patient's age is changed in the note and the model is asked the
   age. Does it follow the text or the original note?

A and B are scored against a control: the same output compared with a sentence from a
different patient (same "has numbers" status). A model that only writes plausible
clinical text matches the control about as well as the real sentence; a model that
remembers the note matches the real sentence far better, especially its numbers.

    python -m trialguide.eval.memorisation --queries DATASET_DIR [--split splits/dev.txt]

DATASET_DIR holds sigir/, trec_2021/, trec_2022/ with queries.jsonl (TrialGPT layout).
Writes a summary to results/ and every item to outputs/ (not committed).
Owner of this file: Prateek (T14).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import re
from pathlib import Path
from typing import Callable

MODEL = "Qwen/Qwen3-8B"
SEED = 20261030
COHORTS = ("sigir", "trec_2021", "trec_2022")
MIN_WORDS = 6        # shorter sentences are too easy to match by chance to count as verbatim
VERBATIM = 0.8       # share of the true sentence's words recovered, in order, to call it verbatim
SPLIT_DIR = Path(__file__).resolve().parents[3] / "splits"

# Decision rule, fixed before the first run on Qwen3-8B
RULE = {
    "verbatim_rate_max": 0.05,     # memorisation if verbatim rate (A or B) is above 5% ...
    "verbatim_vs_control": 3.0,    # ... and at least 3x the control rate
    "number_gap_max": 0.10,        # or exact-number recall beats the control by more than 10 points
    "edit_follow_min": 0.90,       # the model must follow an edited age in at least 90% of notes
}

GAP_PROMPT = """Below is a patient note from a public clinical dataset (TREC / SIGIR). One sentence was removed and replaced with [MISSING].
Write the missing sentence exactly as it appeared in the original note. Reply with that one sentence only.

Note:
{note}"""

AGE_PROMPT = """Read the patient note and answer from the note only.

Note:
{note}

How old is the patient? Reply with the number only."""

GenerateFn = Callable[[list], list]  # list of prompts -> list of completions


# ---------------------------------------------------------------- scoring

def words(text: str) -> list:
    """Lowercased words and numbers, punctuation dropped ("1.9 mg/dL." -> ["1.9", "mg", "dl"])."""
    return re.findall(r"\d+(?:\.\d+)?|[a-z]+", text.lower())


def numbers(text: str) -> set:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def lcs(a: list, b: list) -> int:
    """Length of the longest common subsequence of two word lists."""
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def recall(output: str, target: str) -> float:
    """ROUGE-L recall: share of the target's words the output reproduces in order.
    The output is cut to the target length + 10 words, so a long ramble cannot collect matches."""
    t = words(target)
    if not t:
        return 0.0
    return lcs(words(output)[: len(t) + 10], t) / len(t)


def score(output: str, target: str, visible: str = "") -> dict:
    """visible: the text the model was shown. Its numbers are left out of the number check,
    since repeating a number the model can see proves nothing."""
    r = recall(output, target)
    nums = numbers(target) - numbers(visible)
    return {
        "recall": round(r, 3),
        # None = does not count either way (sentence too short, or no unseen numbers in it)
        "verbatim": (r >= VERBATIM) if len(words(target)) >= MIN_WORDS else None,
        "numbers_match": (nums <= numbers(output)) if nums else None,
    }


# ---------------------------------------------------------------- items

def sentences(note: str) -> list:
    from nltk.tokenize import sent_tokenize
    return [s.strip() for s in sent_tokenize(note) if s.strip()]


def pick_controls(items: list, rng: random.Random) -> None:
    """Give each item a control sentence from a different patient, with numbers iff the target has numbers."""
    for it in items:
        pool = [o["target"] for o in items
                if o["patient"] != it["patient"] and bool(numbers(o["target"])) == bool(numbers(it["target"]))]
        it["control"] = rng.choice(pool) if pool else None


def continuation_items(notes: dict) -> list:
    """Test A: prefix = sentences before i (i >= 1), target = sentence i."""
    out = []
    for p, sents in notes.items():
        for i in range(1, len(sents)):
            out.append({"patient": p, "index": i, "prompt": " ".join(sents[:i]) + " ", "target": sents[i],
                        "visible": sents[:i]})
    return out


def gap_items(notes: dict) -> list:
    """Test B: sentence i replaced by [MISSING]; notes with one sentence are skipped."""
    out = []
    for p, sents in notes.items():
        if len(sents) < 2:
            continue
        for i, s in enumerate(sents):
            shown = sents[:i] + ["[MISSING]"] + sents[i + 1:]
            out.append({"patient": p, "index": i, "prompt": GAP_PROMPT.format(note=" ".join(shown)), "target": s,
                        "visible": sents[:i] + sents[i + 1:]})
    return out


AGE_RE = re.compile(r"\b(\d{1,3})(?=[- ](?:year|yr|yo\b|y/o|y\.o\.))", re.I)


def edit_age(note: str, rng: random.Random):
    """Change every mention of the patient's age to a different plausible age.
    Returns (edited note, old age, new age), or None if the note gives no age."""
    found = AGE_RE.findall(note)
    if not found:
        return None
    old = int(found[0])
    lo, hi, step = (18, 95, (5, 15)) if old >= 18 else (1, 17, (2, 4))
    choices = [a for d in range(step[0], step[1] + 1) for a in (old - d, old + d) if lo <= a <= hi]
    new = rng.choice(choices)
    edited = AGE_RE.sub(lambda m: str(new) if int(m.group(1)) == old else m.group(1), note)
    return edited, old, new


def first_int(text: str):
    m = re.search(r"\d+", text)
    return int(m.group()) if m else None


# ---------------------------------------------------------------- run

def rate(xs: list):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def summarise(items: list) -> dict:
    """Real-sentence vs control rates for one test."""
    real = [it["score"] for it in items]
    ctrl = [it["control_score"] for it in items if it.get("control_score")]
    return {
        "items": len(items),
        "mean_recall": rate([s["recall"] for s in real]),
        "mean_recall_control": rate([s["recall"] for s in ctrl]),
        "items_long_enough": sum(s["verbatim"] is not None for s in real),
        "verbatim_rate": rate([s["verbatim"] for s in real]),
        "verbatim_rate_control": rate([s["verbatim"] for s in ctrl]),
        # outputs that repeat a sentence the model was shown: copying, not memory
        "copied_visible_rate": rate([it["copied_visible"] for it in items]),
        "items_with_unseen_numbers": sum(s["numbers_match"] is not None for s in real),
        "number_recall": rate([s["numbers_match"] for s in real]),
        "number_recall_control": rate([s["numbers_match"] for s in ctrl]),
    }


def verdict(summary: dict) -> dict:
    """Apply RULE to the summary. Returns the reasons that point to memorisation (empty = none)."""
    reasons = []
    for test in ("A_continuation", "B_gap"):
        s = summary[test]
        v, vc = s["verbatim_rate"] or 0, s["verbatim_rate_control"] or 0
        if v > RULE["verbatim_rate_max"] and v >= RULE["verbatim_vs_control"] * max(vc, 1e-9):
            reasons.append(f"{test}: verbatim rate {v:.1%} vs control {vc:.1%}")
        n, nc = s["number_recall"], s["number_recall_control"]
        if n is not None and nc is not None and n - nc > RULE["number_gap_max"]:
            reasons.append(f"{test}: exact numbers recalled {n:.1%} vs control {nc:.1%}")
    f = summary["C_edited_age"]["follows_edit_rate"]
    if f is not None and f < RULE["edit_follow_min"]:
        reasons.append(f"C_edited_age: follows the edited age in only {f:.1%} of notes")
    return {"memorisation_signal": bool(reasons), "reasons": reasons}


def run_probe(notes: dict, complete: GenerateFn, chat: GenerateFn, seed: int = SEED,
              split: Callable[[str], list] = None) -> tuple:
    """notes: patient -> note text. complete: raw-text continuation; chat: one user turn -> reply.
    split: note -> sentences (default nltk, as in TrialGPT). Returns (summary, items)."""
    rng = random.Random(seed)
    split = split or sentences
    sents = {p: split(n) for p, n in sorted(notes.items())}

    tests = {"A_continuation": (continuation_items(sents), complete),
             "B_gap": (gap_items(sents), chat)}
    for items, fn in tests.values():
        pick_controls(items, rng)
        for it, out in zip(items, fn([it["prompt"] for it in items])):
            seen = " ".join(it["visible"])
            it["output"] = out
            it["score"] = score(out, it["target"], seen)
            it["control_score"] = score(out, it["control"], seen) if it["control"] else None
            it["copied_visible"] = any(recall(out, v) >= VERBATIM for v in it["visible"] if len(words(v)) >= MIN_WORDS)
            del it["visible"]

    ages = []
    for p, note in sorted(notes.items()):
        e = edit_age(note, rng)
        if e:
            ages.append({"patient": p, "old": e[1], "new": e[2],
                         "prompts": [AGE_PROMPT.format(note=note), AGE_PROMPT.format(note=e[0])]})
    replies = chat([q for a in ages for q in a["prompts"]])
    for k, a in enumerate(ages):
        a["reply_original"], a["reply_edited"] = replies[2 * k], replies[2 * k + 1]
        a["reads_original"] = first_int(a["reply_original"]) == a["old"]
        a["follows_edit"] = first_int(a["reply_edited"]) == a["new"]
        a["gives_old_age"] = first_int(a["reply_edited"]) == a["old"]
        del a["prompts"]

    summary = {name: summarise(items) for name, (items, _) in tests.items()}
    summary["C_edited_age"] = {
        "notes": len(ages),
        "reads_original_rate": rate([a["reads_original"] for a in ages]),
        "follows_edit_rate": rate([a["follows_edit"] for a in ages]),
        "gives_old_age_rate": rate([a["gives_old_age"] for a in ages]),
    }
    summary["verdict"] = verdict(summary)
    items = {name: items for name, (items, _) in tests.items()}
    items["C_edited_age"] = ages
    return summary, items


# ---------------------------------------------------------------- model

class HFModel:
    """Qwen3-8B with transformers, greedy. Same memory settings as the shared client."""

    def __init__(self, name: str = MODEL, batch: int = 8, max_new_tokens: int = 96):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.batch, self.max_new_tokens = torch, batch, max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        max_memory = None
        if os.getenv("LLM_HF_MAX_MEMORY") and torch.cuda.device_count() > 1:
            max_memory = {i: os.getenv("LLM_HF_MAX_MEMORY") for i in range(torch.cuda.device_count())}
        # LLM_HF_8BIT=1: 8-bit weights on one 12 GB GPU (a preview; the reported run is bf16)
        self.weights = "8bit" if os.getenv("LLM_HF_8BIT") == "1" else "bf16"
        quant = None
        if self.weights == "8bit":
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.bfloat16, device_map="auto", max_memory=max_memory,
            quantization_config=quant).eval()
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def _generate(self, texts: list) -> list:
        out = []
        for i in range(0, len(texts), self.batch):
            enc = self.tok(texts[i:i + self.batch], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.model.device)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
            out += self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"  {len(out)}/{len(texts)}", flush=True)
        return out

    def complete(self, prompts: list) -> list:
        return self._generate(prompts)

    def chat(self, prompts: list) -> list:
        texts = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
                 for p in prompts]
        return [o.strip() for o in self._generate(texts)]


# ---------------------------------------------------------------- main

def load_notes(queries_dir: Path, ids: list) -> dict:
    notes = {}
    for c in COHORTS:
        with open(queries_dir / c / "queries.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                notes[r["_id"]] = r["text"]
    missing = [p for p in ids if p not in notes]
    assert not missing, f"no note for {missing}"
    return {p: notes[p] for p in ids}


def read_ids(path: Path) -> list:
    return [line.strip() for line in open(path) if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--queries", type=Path, required=True, help="folder with sigir/, trec_2021/, trec_2022/")
    ap.add_argument("--split", type=Path, default=SPLIT_DIR / "dev.txt")
    ap.add_argument("--limit", type=int, default=None, help="first N patients only (smoke test)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--raw-out", type=Path, default=Path("outputs"))
    args = ap.parse_args()

    ids = read_ids(args.split)
    sealed = args.split.parent / "test.txt"
    if sealed.exists() and sealed.resolve() != args.split.resolve():
        assert not set(ids) & set(read_ids(sealed)), "split overlaps the sealed test set"
    else:
        raise SystemExit("refusing to run: the split must not be the test set, and splits/test.txt must exist to check")
    ids = ids[: args.limit] if args.limit else ids

    notes = load_notes(args.queries, ids)
    m = HFModel(args.model)
    summary, items = run_probe(notes, m.complete, m.chat)

    import torch
    import transformers
    meta = {
        "task": "T14 memorisation probe", "date": datetime.date.today().isoformat(),
        "model": args.model, "revision": m.revision, "weights": m.weights, "decoding": "greedy",
        "max_new_tokens": m.max_new_tokens, "seed": SEED, "split": str(args.split), "patients": len(ids),
        "transformers": transformers.__version__, "torch": torch.__version__,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "rule": RULE, "min_words": MIN_WORDS, "verbatim_threshold": VERBATIM,
    }
    name = f"memorisation_probe_{args.split.stem}_{args.model.split('/')[-1].lower()}"
    name += "_8bit" if m.weights == "8bit" else ""
    name += f"_limit{args.limit}" if args.limit else ""
    args.out.mkdir(parents=True, exist_ok=True)
    args.raw_out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{name}.json").write_text(json.dumps({**meta, **summary}, indent=2) + "\n")
    with open(args.raw_out / f"{name}_items.json", "w") as f:
        json.dump(items, f, indent=1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
