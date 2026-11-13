"""The interactive demo's session logic and recorded bundles (T40), with a scripted checker."""

import pytest

from trialguide.demo.build_bundle import explore
from trialguide.demo.record import (
    MissingRecording, RecordingChecker, ReplayChecker, load_bundle, patient_note, patient_rules, save_bundle,
)
from trialguide.demo.session import DemoSession, answer_sentence
from trialguide.schemas import Certainty, InfoCategory, RuleCheck, RuleKind, RuleStatus, Split, TrialRule, VisibleNote

INC, EXC = RuleKind.INCLUSION, RuleKind.EXCLUSION
RULES = [TrialRule("t1", "inc_0", INC, "Adults over 18"),
         TrialRule("t1", "inc_1", INC, "No prior chemotherapy"),
         TrialRule("t2", "inc_0", INC, "Adults over 18"),
         TrialRule("t2", "exc_0", EXC, "Pregnancy")]


def note():
    return VisibleNote("p1", Split.DEV, {0: "A 45-year-old woman with breast cancer.", 1: "She is not pregnant."})


def scripted(n, r):
    """Age and pregnancy are stated; chemotherapy is missing until the doctor answers."""
    def out(status, category, ev=(), key=None):
        return RuleCheck(n.patient_id, r.trial_id, r.rule_id, r.kind, status, category,
                         evidence_ids=list(ev), confidence=0.95, fact_key=key)
    if r.text == "Adults over 18":
        return out(RuleStatus.MET, InfoCategory.STATED, (0,))
    if r.text == "Pregnancy":
        return out(RuleStatus.NOT_MET, InfoCategory.STATED, (1,))
    answered = [i for i, s in n.sentences.items() if "prior chemotherapy:" in s.lower()]
    if answered:
        said_no = n.sentences[answered[0]].lower().endswith("no.")
        return out(RuleStatus.MET if said_no else RuleStatus.NOT_MET, InfoCategory.STATED, (answered[0],))
    return out(RuleStatus.NOT_ENOUGH_INFO, InfoCategory.MISSING, key="prior_chemotherapy")


def session(check=scripted):
    s = DemoSession(note(), RULES, check, threshold=0.9)
    s.start()
    return s


def test_first_question_is_about_the_missing_fact():
    s = session()
    assert s.question is not None and "chemotherapy" in s.question.text
    assert s.counts("t1") == {"stated": 1, "inferred": 0, "answered": 0, "open": 1}


def test_a_sure_no_settles_the_rule_and_raises_the_trial():
    s = session()
    before = s.scores()["t1"]
    s.answer("no", Certainty.SURE)
    assert s.turns[0].settled == [("t1", "inc_1", "not_enough_info", "met")]
    assert s.scores()["t1"] > before and s.counts("t1")["answered"] == 1
    assert s.finished                                   # nothing left to ask


def test_not_documented_adds_nothing_and_ends_the_questions_on_that_fact():
    s = session()
    s.answer("", Certainty.NOT_DOCUMENTED)
    assert s.turns[0].settled == [] and s.finished
    assert s.counts("t1")["open"] == 1                   # still open: reported as an abstention


def test_unsure_allows_one_follow_up_on_the_same_fact():
    s = session()
    s.answer("no", Certainty.UNSURE)
    assert s.question is not None and s.question.is_followup and "chemotherapy" in s.question.text
    assert s.counts("t1")["open"] == 1                   # an unsure answer is not used as evidence


def test_b6_ignores_answers_and_never_infers():
    s = session()
    b6_before = s.b6_scores()
    s.answer("no", Certainty.SURE)
    assert s.b6_scores() == b6_before


def test_answer_sentence_reads_like_a_record_line():
    s = session()
    assert answer_sentence(s.question, "no") == "Prior chemotherapy: no."


def test_fork_is_independent():
    s = session()
    twin = s.fork()
    twin.answer("no", Certainty.SURE)
    assert s.turns == [] and len(twin.turns) == 1 and twin.check is s.check


def test_recorded_bundle_replays_every_answer_path(tmp_path):
    rec = RecordingChecker(scripted)
    s = DemoSession(note(), RULES, rec, threshold=0.9)
    s.start()
    assert explore(s) > 1
    p = {"cohort": "x", "split": "dev", "visible": {str(i): t for i, t in note().sentences.items()},
         "hidden": {}, "trials": [{"trial_id": t, "title": t, "gold": 2,
                                   "rules": [{"rule_id": r.rule_id, "kind": r.kind.value, "text": r.text}
                                             for r in RULES if r.trial_id == t]} for t in ("t1", "t2")]}
    save_bundle(tmp_path / "b.json", {"p1": p}, rec.store, {"threshold": 0.9})
    patients, store, meta = load_bundle(tmp_path / "b.json")
    replay = DemoSession(patient_note(patients["p1"], "p1"), patient_rules(patients["p1"]), ReplayChecker(store))
    replay.start()
    replay.answer("yes", Certainty.PROBABLY)             # any offered answer was recorded
    assert replay.turns[0].answer == "yes"


def test_replay_refuses_an_unrecorded_path():
    with pytest.raises(MissingRecording):
        DemoSession(note(), RULES, ReplayChecker({})).start()


def test_the_app_builds(tmp_path):
    pytest.importorskip("gradio")
    from trialguide.demo.app import build
    rec = RecordingChecker(scripted)
    s = DemoSession(note(), RULES, rec)
    s.start()
    p = {"cohort": "x", "split": "dev", "visible": {"0": "a", "1": "b"}, "hidden": {"2": "c"},
         "trials": [{"trial_id": "t1", "title": "T", "gold": 2,
                     "rules": [{"rule_id": "inc_0", "kind": "inclusion", "text": "Adults over 18"}]}]}
    save_bundle(tmp_path / "b.json", {"p1": p}, rec.store, {"threshold": 0.9})
    assert build(tmp_path / "b.json") is not None


def test_session_record_and_page_pieces():
    from trialguide.demo import render
    s = session()
    s.answer("no", Certainty.SURE)
    gold = {"t1": 2, "t2": 2}
    rec = s.to_record(gold)
    assert rec["turns"][0]["rules_changed"] == [["t1", "inc_1", "not_enough_info", "met"]]
    assert rec["ranking"] and len(rec["rules"]) == len(RULES) and rec["ndcg@10"] is not None
    p = {"visible": {"0": "a", "1": "b"}, "hidden": {"5": "secret"},
         "trials": [{"trial_id": "t1", "title": "One", "gold": 2}, {"trial_id": "t2", "title": "Two", "gold": 2}]}
    assert "secret" not in render.note_html(p, s, reveal=False)      # hidden text stays hidden
    assert "secret" in render.note_html(p, s, reveal=True)
    assert "doctor" in render.note_html(p, s, reveal=False)          # the answer shows as a doctor line
    page = render.trials_html(s, p) + render.kpis(s, gold) + render.timeline_html(s) + render.rules_html(s, "t1")
    assert "One" in page and "inc_1" in page and "NDCG@10" in page
    assert "finished" in render.question_html(s)
