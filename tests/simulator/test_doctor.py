"""Simulated doctor (T11), with a scripted fake model instead of Llama."""

import json

from trialguide.schemas import Certainty, Question
from trialguide.simulator.doctor import SimulatedDoctor, find_quote, mentions_topic

FACTS = {"p1": ["She denies smoking, diabetes, hypercholesterolemia, or a family history of heart disease.",
                "She currently takes no medications."]}


def question(text="Is the patient on any medications?"):
    return Question("p1-q1", "p1", "medications", text, 1)


def doctor(reply, tmp_path, facts=FACTS):
    """A doctor whose model always gives `reply`; also records the prompts it was sent."""
    sent = []

    def fake_chat(messages, model):
        sent.append(messages)
        return reply if isinstance(reply, str) else json.dumps(reply)

    return SimulatedDoctor(facts, chat_fn=fake_chat, log_path=str(tmp_path / "log.jsonl")), sent


def test_sure_answer_with_exact_quote(tmp_path):
    d, _ = doctor({"answer": "no", "certainty": "sure", "quote": "She currently takes no medications."}, tmp_path)
    a = d.answer("p1", question())
    assert a.certainty == Certainty.SURE and a.value == "no"
    assert a.quote == "She currently takes no medications."


def test_near_copy_quote_is_mapped_to_the_record_sentence(tmp_path):
    # the model dropped "currently" and the full stop; the record's own sentence is used as the quote
    d, _ = doctor({"answer": "no", "certainty": "sure", "quote": "she takes no medications"}, tmp_path)
    assert d.answer("p1", question()).quote == "She currently takes no medications."


def test_invented_quote_becomes_not_documented(tmp_path):
    # D10: the answer is not supported by the hidden facts, so it must not reach the system
    d, _ = doctor({"answer": "yes", "certainty": "sure", "quote": "She takes warfarin daily."}, tmp_path)
    a = d.answer("p1", question())
    assert a.certainty == Certainty.NOT_DOCUMENTED and a.quote is None and a.value == ""


def test_missing_quote_becomes_not_documented(tmp_path):
    d, _ = doctor({"answer": "no", "certainty": "probably", "quote": None}, tmp_path)
    assert d.answer("p1", question()).certainty == Certainty.NOT_DOCUMENTED


def test_not_documented_reply(tmp_path):
    d, _ = doctor({"answer": "", "certainty": "not_documented", "quote": None}, tmp_path)
    assert d.answer("p1", question("Hepatitis B status?")).certainty == Certainty.NOT_DOCUMENTED


def test_unparsable_or_unknown_certainty_is_not_documented(tmp_path):
    for reply in ["I think she is fine.", {"answer": "no", "certainty": "very sure", "quote": "x"}]:
        d, _ = doctor(reply, tmp_path)
        assert d.answer("p1", question()).certainty == Certainty.NOT_DOCUMENTED


def test_json_in_code_fences_is_read(tmp_path):
    reply = '```json\n{"answer": "no", "certainty": "unsure", "quote": "She denies smoking"}\n```'
    d, _ = doctor(reply, tmp_path)
    a = d.answer("p1", question("Does she smoke?"))
    assert a.certainty == Certainty.UNSURE and a.quote.startswith("She denies smoking")


def test_no_hidden_facts_needs_no_model_call(tmp_path):
    d, sent = doctor({"answer": "yes", "certainty": "sure", "quote": "x"}, tmp_path, facts={})
    assert d.answer("p1", question()).certainty == Certainty.NOT_DOCUMENTED
    assert sent == []


def test_prompt_holds_only_hidden_facts_and_the_question(tmp_path):
    # leakage control: no patient id, no trial or rule text reaches the doctor model
    d, sent = doctor({"answer": "", "certainty": "not_documented", "quote": None}, tmp_path)
    d.answer("p1", question())
    prompt = json.dumps(sent[0])
    assert "p1" not in prompt.replace("p1-", "")  # not even the patient id
    assert "She currently takes no medications." in prompt
    assert "Is the patient on any medications?" in prompt


def test_every_exchange_is_logged_with_the_reason(tmp_path):
    d, _ = doctor({"answer": "yes", "certainty": "sure", "quote": "She takes warfarin daily."}, tmp_path)
    d.answer("p1", question())
    entry = json.loads((tmp_path / "log.jsonl").read_text().splitlines()[0])
    assert entry["certainty"] == "not_documented"
    assert entry["reason"] == "quote not found in the record"
    assert entry["fact_key"] == "medications" and entry["model"].startswith("meta-llama/")


def test_find_quote_rejects_unrelated_text():
    assert find_quote("No known drug allergies.", FACTS["p1"]) is None
    assert find_quote("", FACTS["p1"]) is None


def test_sure_answer_whose_quote_is_off_topic_is_capped_at_probably(tmp_path):
    # Seen on Llama: "pregnant? no" quoting the smoking sentence, i.e. a guess with a real quote
    reply = {"answer": "no", "certainty": "sure",
             "quote": "She denies smoking, diabetes, hypercholesterolemia, or a family history of heart disease."}
    d, _ = doctor(reply, tmp_path)
    a = d.answer("p1", question("Is the patient pregnant?"))
    assert a.certainty == Certainty.PROBABLY and a.value == "no"
    assert json.loads((tmp_path / "log.jsonl").read_text())["reason"] == "quote does not mention the topic"


def test_mentions_topic_matches_word_forms_and_ignores_question_words():
    assert mentions_topic("Does the patient smoke?", "She denies smoking.")
    assert mentions_topic("Is the patient obese?", "She has hypertension and obesity.")
    assert mentions_topic("How old is the patient?", "A 58-year-old woman.")
    assert not mentions_topic("Has the patient had a myocardial infarction before?", "The pain started while walking.")
    assert not mentions_topic("Is the patient taking anticoagulants?", "She currently takes no medications.")
