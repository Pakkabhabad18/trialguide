"""Memorisation probe (T14), with scripted fake models instead of Qwen."""

import random
import re

from trialguide.eval.memorisation import (edit_age, gap_items, continuation_items, recall, run_probe,
                                          score, verdict, words)

NOTES = {
    "p1": "A 58-year-old woman has chest pain. Her creatinine is 1.9 mg/dL. She takes no medications.",
    "p2": "A 45-year-old man has a cough for 3 weeks. He smoked 20 pack-years. Chest X-ray shows a mass.",
}


def test_words_keep_decimals_and_drop_punctuation():
    assert words("Creatinine 1.9 mg/dL.") == ["creatinine", "1.9", "mg", "dl"]


def test_recall_is_in_order_share_of_target():
    assert recall("her creatinine is 1.9 mg/dL", "Her creatinine is 1.9 mg/dL.") == 1.0
    assert recall("mg/dL 1.9 is creatinine her", "Her creatinine is 1.9 mg/dL.") == 2 / 6  # only "mg dl" stays in order
    assert recall("anything", "") == 0.0


def test_recall_ignores_words_past_the_target_length():
    target = "she takes no medications at all today"            # 7 words
    ramble = " ".join(["x"] * 17) + " " + target                # match starts after 7 + 10 words
    assert recall(ramble, target) == 0.0


def test_score_verbatim_needs_long_sentence_and_numbers_need_all():
    s = score("Her creatinine is 1.9 mg/dL today", "Her creatinine is 1.9 mg/dL.")
    assert s["verbatim"] and s["numbers_match"] is True
    assert score("Chest pain.", "Chest pain.")["verbatim"] is None           # too short to count
    assert score("BP 120/80", "BP 120/80, HR 90")["numbers_match"] is False
    assert score("No numbers", "She takes no medications.")["numbers_match"] is None
    # a number the model could see does not count
    assert score("aged 58", "She is 58.", visible="A 58-year-old woman.")["numbers_match"] is None


def test_items_cover_every_sentence():
    sents = {"p": ["S1.", "S2.", "S3."]}
    assert [it["target"] for it in continuation_items(sents)] == ["S2.", "S3."]
    assert continuation_items(sents)[1]["prompt"] == "S1. S2. "
    gaps = gap_items(sents)
    assert len(gaps) == 3 and "S1. [MISSING] S3." in gaps[1]["prompt"]
    assert gap_items({"q": ["only one."]}) == []


def test_edit_age_changes_every_mention_and_stays_plausible():
    edited, old, new = edit_age("A 58-year-old woman. At 58 years old she retired.", random.Random(0))
    assert old == 58 and new != 58 and 18 <= new <= 95 and abs(new - 58) >= 5
    assert "58" not in edited and edited.count(str(new)) == 2
    _, old, new = edit_age("A 4-year-old girl with fever.", random.Random(0))
    assert old == 4 and 1 <= new <= 17
    assert edit_age("Patient with fever.", random.Random(0)) is None
    assert edit_age("He had a stroke 10-15 years ago.", random.Random(0)) is None
    assert edit_age("Fernandez is a 41 year man.", random.Random(0))[1] == 41


def sentences(text):
    """Stand-in for nltk (CI has no punkt data): split after a full stop that ends a word."""
    return re.split(r"(?<=[A-Za-z]\.) ", text.strip())


def _probe(remembers: bool, follows_text: bool):
    """Fake model: either recalls the note's true sentences or writes generic text."""
    truth = {p: sentences(n) for p, n in NOTES.items()}

    def complete(prompts):
        out = []
        for pr in prompts:
            p = next(p for p in NOTES if NOTES[p].startswith(pr.strip()))
            i = len(sentences(pr))
            out.append(truth[p][i] if remembers else "The patient was seen in clinic.")
        return out

    def chat(prompts):
        out = []
        for pr in prompts:
            if pr.startswith("Read the patient note"):
                age = int(pr.split("-year-old")[0].split()[-1])
                orig = {"woman": 58, "man": 45}["woman" if "woman" in pr else "man"]
                out.append(str(age if follows_text else orig))
            else:
                p = "p1" if ("creatinine" in pr or "woman" in pr) else "p2"
                i = pr.split("Note:\n")[1].split("[MISSING]")[0].count(".")
                i -= pr.split("Note:\n")[1].split("[MISSING]")[0].count("1.9")    # the decimal point
                out.append(truth[p][i] if remembers else "The patient was seen in clinic.")
        return out

    return run_probe(NOTES, complete, chat, split=sentences)


def test_probe_flags_a_model_that_remembers_the_notes():
    summary, items = _probe(remembers=True, follows_text=False)
    assert summary["A_continuation"]["verbatim_rate"] == 1.0
    assert summary["B_gap"]["number_recall"] == 1.0
    assert summary["C_edited_age"]["follows_edit_rate"] == 0.0
    assert summary["C_edited_age"]["reads_original_rate"] == 1.0
    assert summary["verdict"]["memorisation_signal"]
    assert len(items["B_gap"]) == 6 and all(it["control"] for it in items["B_gap"])
    assert summary["B_gap"]["copied_visible_rate"] == 0.0


def test_copying_a_visible_sentence_is_reported_not_counted_as_memory():
    def copy_first(prompts):   # always repeats the note's first sentence
        return ["A 58-year-old woman has chest pain." if "woman" in p else "A 45-year-old man has a cough for 3 weeks."
                for p in prompts]
    summary, _ = run_probe(NOTES, copy_first, copy_first, split=sentences)
    assert summary["A_continuation"]["copied_visible_rate"] == 1.0
    assert summary["A_continuation"]["verbatim_rate"] == 0.0


def test_probe_passes_a_model_that_only_reads_the_text():
    summary, _ = _probe(remembers=False, follows_text=True)
    assert summary["A_continuation"]["verbatim_rate"] == 0.0
    assert summary["C_edited_age"]["follows_edit_rate"] == 1.0
    assert summary["verdict"] == {"memorisation_signal": False, "reasons": []}


def test_verdict_needs_both_high_rate_and_gap_over_control():
    base = {"verbatim_rate": 0.0, "verbatim_rate_control": 0.0, "number_recall": None, "number_recall_control": None}
    s = {"A_continuation": dict(base, verbatim_rate=0.30, verbatim_rate_control=0.20),
         "B_gap": dict(base), "C_edited_age": {"follows_edit_rate": 1.0}}
    assert not verdict(s)["memorisation_signal"]           # 30% but only 1.5x the control
    s["B_gap"].update(number_recall=0.40, number_recall_control=0.10)
    assert verdict(s)["reasons"] == ["B_gap: exact numbers recalled 40.0% vs control 10.0%"]
