
from trialguide.schemas import RuleKind, RuleStatus, favours_eligibility, is_harmful


def test_inclusion_met_favours_eligibility():
    assert favours_eligibility(RuleKind.INCLUSION, RuleStatus.MET)


def test_exclusion_not_met_favours_eligibility():
    assert favours_eligibility(RuleKind.EXCLUSION, RuleStatus.NOT_MET)


def test_wrong_inclusion_met_is_harmful():
    assert is_harmful(
        RuleKind.INCLUSION,
        RuleStatus.MET,
        RuleStatus.NOT_MET,
    )


def test_wrong_exclusion_not_met_is_harmful():
    assert is_harmful(
        RuleKind.EXCLUSION,
        RuleStatus.NOT_MET,
        RuleStatus.MET,
    )


def test_wrong_prediction_toward_exclusion_is_not_harmful():
    assert not is_harmful(
        RuleKind.INCLUSION,
        RuleStatus.NOT_MET,
        RuleStatus.MET,
    )


def test_matching_prediction_is_not_harmful():
    assert not is_harmful(
        RuleKind.EXCLUSION,
        RuleStatus.MET,
        RuleStatus.MET,
    )
