"""Live demo: the two-model session, with word-matching stand-ins instead of the models."""

from trialguide.demo.live import TwoModelSession, keyword_checker, note_from_facts, note_from_text
from trialguide.pipeline import ACCEPTED
from trialguide.schemas import Certainty, RuleKind, TrialRule

RULES = [TrialRule("T1", "inc_0", RuleKind.INCLUSION, "Histologically confirmed breast carcinoma"),
         TrialRule("T1", "inc_1", RuleKind.INCLUSION, "Signed informed consent"),
         TrialRule("T1", "exc_0", RuleKind.EXCLUSION, "Serum creatinine above normal range"),
         TrialRule("T2", "inc_0", RuleKind.INCLUSION, "Prior chemotherapy for metastatic disease"),
         TrialRule("T2", "exc_0", RuleKind.EXCLUSION, "Pregnant or nursing women")]
NOTE = "A 58-year-old woman has histologically confirmed breast carcinoma. She takes metformin."


def session(budget=3):
    s = TwoModelSession(note_from_text(NOTE), RULES, keyword_checker(), keyword_checker(settle_open=True),
                        threshold=0.766, budget=budget)
    s.start()
    return s


def test_second_model_settles_only_rules_the_first_left_open():
    s = session()
    acts = s.action_of()
    assert ("T1", "inc_1") in s.by_second                     # consent: only the second model settles it
    assert acts[("T1", "inc_1")].action in ACCEPTED
    assert "second model" in acts[("T1", "inc_1")].reason
    assert ("T1", "inc_0") not in s.by_second                 # stated in the note: the first model's
    assert s.counts("T1")["second"] == 1


def test_doctor_is_asked_only_about_rules_neither_model_settles():
    s = session()
    if s.question is not None:                     # the selector may also stop at once (no question can help)
        assert not set(s.question.rule_refs) & s.by_second
    open_refs = {r for r, a in s.action_of().items() if a.action not in ACCEPTED}
    assert not open_refs & s.by_second


def test_answer_and_added_information_update_the_note():
    s = session()
    n = len(s.note.sentences)
    t = s.fork()
    if t.question is not None:
        t.answer("no", Certainty.SURE)
    m = len(t.note.sentences)
    t.add_info("Serum creatinine is within the normal range. She is not pregnant.")
    assert len(t.note.sentences) == m + 2 and len(s.note.sentences) == n    # the fork left s alone
    assert t.turns[-1].question == "You added information"


def test_note_from_facts():
    note = note_from_facts("54", "Female", "type 2 diabetes")
    assert list(note.sentences.values()) == ["The patient is a 54-year-old female.", "Main diagnosis: type 2 diabetes."]


def test_stand_in_reads_age_from_the_note():
    from trialguide.demo.live import _age_check
    note = note_from_text("A 64-year-old man with lung cancer.")
    older = TrialRule("NCT1", "inc_0", RuleKind.INCLUSION, "Age 18 years or older")
    younger = TrialRule("NCT1", "inc_1", RuleKind.INCLUSION, "Aged 18 to 60 years")
    assert _age_check(note, older)[0].value == "met"
    assert _age_check(note, younger)[0].value == "not_met"


def test_stand_in_question_names_are_short_and_readable():
    from trialguide.demo.live import _fact_label
    assert _fact_label("Patients with a history of congestive heart failure") == "congestive_heart_failure"
    assert _fact_label(">40 years of age") == "age"
    assert _fact_label("For COPD patients:") is None


def test_live_page_panels_render_for_a_session():
    from trialguide.demo import live_view
    s = session()
    record = {"trials": [{"trial_id": "T1", "title": "Breast trial"}, {"trial_id": "T2", "title": "Other trial"}]}
    assert "Breast trial" in live_view.trials(s, record)
    assert "trials checked" in live_view.tiles(s)
    assert "Patient:" in live_view.patient_bar(s)
    assert ("Question" in live_view.question(s)) or ("All done" in live_view.question(s))
    assert "Recorded demo" in live_view.header(True, "https://example.org")


def test_rule_checks_survive_the_trip_to_the_gpu_and_back():
    import json
    from trialguide.demo.live import check_to_json, note_to_json, rule_to_json, serve_checks
    from trialguide.policy.calibrate import check_from_json
    note = note_from_text(NOTE)
    payload = json.dumps({"model": "m", "note": note_to_json(note), "rules": [rule_to_json(r) for r in RULES]})
    back = [check_from_json(d) for d in json.loads(serve_checks(payload, {"m": keyword_checker()}))]
    direct = [keyword_checker()(note, r) for r in RULES]
    assert [c.status for c in back] == [c.status for c in direct]
    assert check_to_json(back[0])["kind"] == "inclusion"


def test_website_falls_back_to_the_stand_in_when_the_gpu_is_gone():
    from trialguide.demo.live import RemoteChecker
    remote = RemoteChecker("http://127.0.0.1:9", "m", "key", fallback=keyword_checker())
    s = TwoModelSession(note_from_text(NOTE), RULES, remote, keyword_checker(settle_open=True),
                        threshold=0.766, budget=3)
    s.start()                                       # no server on port 9: every check uses the stand-in
    assert remote.failed
    assert s.checks[("T1", "inc_0")].status.value == "met"


def test_demo_safeguard_reopens_conclusions_that_cite_an_unrelated_sentence():
    # cases seen with the real models on a prostate cancer note (Kaggle run)
    from trialguide.demo.live import Guarded
    from trialguide.schemas import InfoCategory, RuleCheck, RuleStatus
    note = note_from_text("A 67-year-old man has metastatic prostate cancer with bone metastases. "
                          "His PSA is rising on androgen deprivation therapy. He has not received docetaxel. "
                          "ECOG performance status is 1.")
    def rule(rid, kind, text):
        return TrialRule("T", rid, kind, text)
    cases = [  # (rule, status, category, cited sentence, open afterwards for the first / second model)
        (rule("i11", RuleKind.INCLUSION, "Written informed consent"), "not_met", "inferred", 0, True, True),
        (rule("e8", RuleKind.EXCLUSION, "Active uncontrolled bacterial, viral or fungal infection"),
         "not_met", "stated", 2, False, True),
        (rule("i3", RuleKind.INCLUSION, "WHO performance status of 2 or less"), "met", "stated", 3, False, False),
        (rule("i2", RuleKind.INCLUSION, "Bone metastases (6 lesions or more) on bone scintigraphy"),
         "not_met", "inferred", 0, False, False),
    ]
    for r, status, cat, sid, open_first, open_second in cases:
        c = RuleCheck(note.patient_id, "T", r.rule_id, r.kind, RuleStatus(status), InfoCategory(cat),
                      evidence_ids=[sid], confidence=0.9, n_samples=1)
        model = lambda n, x, c=c: c
        assert Guarded(model)(note, r).unresolved == open_first, r.text
        assert Guarded(model, strict=True)(note, r).unresolved == open_second, r.text
