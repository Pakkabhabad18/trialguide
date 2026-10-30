"""BMI tool."""

from trialguide.tools import call_tool
from trialguide.tools.body_size import bmi


def test_metric():
	# 68 / 1.60^2 = 26.5625
	result = bmi(weight=68, height=160)
	assert result.ok and result.value == 26.56 and result.unit == "kg/m2"


def test_proposal_example():
	# proposal Table 2: height 160 cm, weight 82 kg -> BMI 32.0, so BMI >= 30 is met
	assert bmi(weight=82, height=160).value == 32.03


def test_imperial_matches_703_rule():
	# 703 * 150 / 65^2 = 24.96
	assert bmi(weight=150, weight_unit="lb", height=65, height_unit="in").value == 24.96


def test_height_in_metres():
	assert bmi(weight=68, height=1.6, height_unit="m").value == bmi(weight=68, height=160).value


def test_missing_input_is_missing():
	result = bmi(weight=82)
	assert not result.ok and "height" in result.missing


def test_non_positive_is_missing():
	assert not bmi(weight=0, height=160).ok


def test_bad_unit_through_call_tool_is_missing_not_an_exception():
	result = call_tool("bmi", {"weight": 70, "height": 170, "height_unit": "furlong"})
	assert not result.ok and "bad arguments" in result.missing
