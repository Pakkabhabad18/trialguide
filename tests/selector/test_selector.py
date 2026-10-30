"""Question selector (T20). Expected Eq. 1 values are worked out by hand in the comments."""

import math

import pytest

from trialguide.schemas import (
    Certainty, ConversationState, DoctorAnswer, InfoCategory, Question, RuleCheck, RuleKind,
    RuleStatus, VisibleNote,
)
from trialguide.selector.selector import (
    Candidate, can_change_top_k, canonical_fact_key, follow_up_question, merge_answer,
    merge_candidates, score_candidate, select_question, trial_score,
)

INC, EXC = RuleKind.INCLUSION, RuleKind.EXCLUSION
MET, NOT_MET, NEI, NA = RuleStatus.MET, RuleStatus.NOT_MET, RuleStatus.NOT_ENOUGH_INFO, RuleStatus.NOT_APPLICABLE


def check(trial, rule, kind, status, confidence=1.0, fact_key=None):
    # decided rules are "stated" with one evidence sentence; unresolved ones are "missing"
    decided = status in (MET, NOT_MET)
    return RuleCheck(
        patient_id="p1", trial_id=trial, rule_id=rule, kind=kind, status=status,
        category=InfoCategory.STATED if decided else InfoCategory.MISSING,
        evidence_ids=[0] if decided else [], confidence=confidence, fact_key=fact_key,
    )


def half_met(trial, fact):
    # one inclusion met, one unresolved: score 1/2 = 0.5
    return [check(trial, "inc_1", INC, MET), check(trial, "inc_2", INC, NEI, fact_key=fact)]


def all_met(trial):
    return [check(trial, "inc_1", INC, MET), check(trial, "inc_2", INC, MET)]


# --- trial score (TrialGPT linear matching score) ---

def test_trial_score_matches_trialgpt_formula():
    assert trial_score(all_met("t")) == 1.0
    assert trial_score(half_met("t", "f")) == 0.5
    # one met, one not met: 1/2 - 1 = -0.5
    assert trial_score([check("t", "a", INC, MET), check("t", "b", INC, NOT_MET)]) == -0.5
    # an exclusion that is met costs 1; not-applicable rules are left out
    assert trial_score(all_met("t") + [check("t", "e", EXC, MET), check("t", "n", INC, NA)]) == 0.0


# --- Eq. 1 ---

def test_eq1_single_trial_by_hand():
    # trial ranked 1, score 0.5; answer met -> 1.0 (change 0.5), not met -> 0.5 - 1 = -0.5 (change 1.0)
    # Score = 0.5 * 0.5 + 0.5 * 1.0 = 0.75, divided by log2(1 + 1) = 1
    checks = half_met("t1", "smoking")
    cand = Candidate("smoking", [("t1", "inc_2")])
    assert math.isclose(score_candidate(cand, checks), 0.75)


def test_lower_ranked_trials_count_less():
    # same unresolved rule, but in the trial ranked 2 (behind a fully met trial): 0.75 / log2(3)
    checks = all_met("a") + half_met("b", "smoking")
    cand = Candidate("smoking", [("b", "inc_2")])
    assert math.isclose(score_candidate(cand, checks), 0.75 / math.log2(3))


def test_a_fact_shared_by_two_trials_scores_higher():
    checks = half_met("t1", "smoking") + half_met("t2", "smoking")
    one = Candidate("smoking", [("t1", "inc_2")])
    both = Candidate("smoking", [("t1", "inc_2"), ("t2", "inc_2")])
    assert score_candidate(both, checks) > score_candidate(one, checks)


def test_answer_probabilities_are_used():
    # if the rule is surely met, only the 0.5 change counts
    checks = half_met("t1", "smoking")
    cand = Candidate("smoking", [("t1", "inc_2")])
    assert math.isclose(score_candidate(cand, checks, p_met=lambda t, r: 1.0), 0.5)


# --- stopping rule ---

def test_no_answer_can_move_a_trial_into_the_top_k():
    # with k = 1: "b" can at best tie "a" (1.0), and ties go to the earlier id, so the top 1 cannot change
    checks = all_met("a") + half_met("b", "smoking")
    cand = Candidate("smoking", [("b", "inc_2")])
    assert not can_change_top_k(cand, checks, k=1)
    assert select_question([cand], checks, ConversationState("p1"), k=1) is None


def test_a_change_in_order_inside_the_top_k_counts():
    # "a" scores 2/3 (two of three met); "b" scores 0.5 but 1.0 if its rule is met, overtaking "a"
    checks = all_met("a") + [check("a", "inc_3", INC, NEI, fact_key="other")] + half_met("b", "smoking")
    cand = Candidate("smoking", [("b", "inc_2")])
    assert can_change_top_k(cand, checks, k=10)


def test_score_changes_that_keep_the_order_do_not_count():
    # "b" can only tie "a" (ties keep id order) or fall further behind: the ranking cannot change
    checks = all_met("a") + half_met("b", "smoking")
    cand = Candidate("smoking", [("b", "inc_2")])
    assert not can_change_top_k(cand, checks, k=10)


# --- choosing questions ---

def scenario():
    # trial "a" (rank 1) has an unresolved exclusion: if met, it drops from 1.0 to 0.0
    # trial "b" (rank 2) is fully met; trial "c" (rank 3) has an unresolved inclusion
    checks = (all_met("a") + [check("a", "exc_1", EXC, NEI, confidence=0.9, fact_key="hepatitis_b")]
              + all_met("b")
              + [check("c", "inc_1", INC, NEI, confidence=0.2, fact_key="smoking")])
    candidates = merge_candidates({"hepatitis_b": [("a", "exc_1")], "smoking": [("c", "inc_1")]})
    return checks, candidates


def test_ranking_strategy_picks_the_bigger_expected_change_near_the_top():
    # hepatitis_b: trial "a" rank 1, change 1.0 with prob 0.5 -> 0.5 / log2(2) = 0.5
    # smoking: trial "c" rank 3, 0 -> 1.0 or -1.0, change 1.0 either way -> 1.0 / log2(4) = 0.5
    # a tie, so the alphabetically first fact wins; make "a" matter more by sharing the fact
    checks, _ = scenario()
    checks += [check("b", "exc_1", EXC, NEI, fact_key="Hepatitis B")]
    candidates = merge_candidates({"hepatitis_b": [("a", "exc_1")], "Hepatitis B": [("b", "exc_1")],
                                   "smoking": [("c", "inc_1")]})
    q = select_question(candidates, checks, ConversationState("p1"))
    assert q.fact_key == canonical_fact_key("hepatitis_b")
    assert sorted(q.rule_refs) == [("a", "exc_1"), ("b", "exc_1")]
    assert q.turn == 1 and q.score > 0.5


def test_uncertainty_strategy_picks_the_least_confident_fact():
    checks, candidates = scenario()
    q = select_question(candidates, checks, ConversationState("p1"), strategy="uncertainty")
    assert q.fact_key == "smoking"  # confidence 0.2 < 0.9


def test_fixed_strategy_follows_trial_rank():
    checks, candidates = scenario()
    q = select_question(candidates, checks, ConversationState("p1"), strategy="fixed")
    assert q.fact_key == canonical_fact_key("hepatitis_b")  # its trial is ranked 1


def test_budget_and_no_repeats():
    checks, candidates = scenario()
    state = ConversationState("p1", budget=1)
    q = select_question(candidates, checks, state, strategy="fixed")
    state.add_turn(q, DoctorAnswer(q.question_id, "", Certainty.NOT_DOCUMENTED))
    assert select_question(candidates, checks, state, strategy="fixed") is None  # budget used

    state = ConversationState("p1", budget=3)
    state.add_turn(q, DoctorAnswer(q.question_id, "", Certainty.NOT_DOCUMENTED))
    q2 = select_question(candidates, checks, state, strategy="fixed")
    assert q2.fact_key == "smoking" and q2.turn == 2  # hepatitis b is never asked again


def test_unknown_strategy_fails_loudly():
    checks, candidates = scenario()
    with pytest.raises(ValueError):
        select_question(candidates, checks, ConversationState("p1"), strategy="random")


def test_questions_are_not_leading():
    checks, candidates = scenario()
    q = select_question(candidates, checks, ConversationState("p1"), strategy="fixed")
    assert q.text == "What does the patient's record say about hepatitis b?"


# --- fact merging ---

def test_spelling_variants_merge_into_one_fact():
    keys = ["prior_chemotherapy", "history_of_chemo", "Chemotherapy history", "previous chemotherapy"]
    assert len({canonical_fact_key(k) for k in keys}) == 1
    assert canonical_fact_key("hepatitis_b") != canonical_fact_key("hepatitis_c")


def test_merge_candidates_unions_rules_without_duplicates():
    merged = merge_candidates({"prior_chemotherapy": [("t1", "inc_2")],
                               "history_of_chemo": [("t2", "exc_1"), ("t1", "inc_2")]})
    assert len(merged) == 1
    assert merged[0].rule_refs == [("t1", "inc_2"), ("t2", "exc_1")]
    assert merged[0].label == "prior chemotherapy"


# --- answers and follow-ups ---

def note():
    return VisibleNote(patient_id="p1", split="dev", sentences={0: "A 58-year-old woman.", 3: "No medications."})


def question(followup=False):
    return Question("p1-q1", "p1", "hepatitis_b", "What does the patient's record say about hepatitis b?",
                    1, [("a", "exc_1")], is_followup=followup)


def test_documented_answer_becomes_a_new_sentence():
    answer = DoctorAnswer("p1-q1", "no", Certainty.SURE, quote="HBsAg negative (2024).")
    new_note, sid = merge_answer(note(), question(), answer)
    assert sid == 4 and new_note.sentences[4].endswith("HBsAg negative (2024).")
    assert 4 not in note().sentences  # the original note is not modified


def test_not_documented_adds_nothing():
    answer = DoctorAnswer("p1-q1", "", Certainty.NOT_DOCUMENTED)
    new_note, sid = merge_answer(note(), question(), answer)
    assert sid is None and new_note.sentences == note().sentences


def test_only_one_follow_up():
    state = ConversationState("p1")
    first = question()
    state.add_turn(first, DoctorAnswer("p1-q1", "maybe", Certainty.UNSURE, quote="Hepatitis panel pending."))
    follow = follow_up_question(first, state)
    assert follow.is_followup and follow.turn == 2 and follow.fact_key == first.fact_key
    assert follow_up_question(follow, state) is None
