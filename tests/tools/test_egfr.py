"""eGFR tool. Expected values are worked out by hand in the comments."""

import math

from trialguide.tools.renal import ckd_epi_2009, ckd_epi_2021


def test_knee_point_male():
	# Scr == k (0.9) makes both min/max terms 1: 142 * 0.9938^40 = 110.7
	assert ckd_epi_2021(creatinine=0.9, age=40, sex="male").value == 110.7


def test_knee_point_female():
	# Scr == k (0.7): 142 * 0.9938^40 * 1.012 = 112.1
	assert ckd_epi_2021(creatinine=0.7, age=40, sex="female").value == 112.1


def test_below_knee_uses_alpha():
	# Scr 0.5 < 0.7: 142 * (0.5/0.7)^-0.241 * 0.9938^50 * 1.012 = 114.2
	assert ckd_epi_2021(creatinine=0.5, age=50, sex="F").value == 114.2


def test_proposal_example():
	# proposal Table 2: 62-year-old woman, creatinine 1.9 -> "about 29", so eGFR >= 60 is not met
	result = ckd_epi_2021(creatinine=1.9, age=62, sex="female")
	assert result.ok and result.value == 29.5 and result.unit == "mL/min/1.73 m2"


def test_umol_per_litre_gives_the_same_value():
	mg = ckd_epi_2021(creatinine=1.9, age=62, sex="female")
	umol = ckd_epi_2021(creatinine=167.96, age=62, sex="female", creatinine_unit="umol/L")
	assert mg.value == umol.value


def test_higher_creatinine_and_age_lower_egfr():
	assert ckd_epi_2021(creatinine=2.0, age=50, sex="male").value < ckd_epi_2021(creatinine=1.0, age=50, sex="male").value
	assert ckd_epi_2021(creatinine=1.0, age=80, sex="male").value < ckd_epi_2021(creatinine=1.0, age=40, sex="male").value


def test_missing_input_is_missing_not_guessed():
	result = ckd_epi_2021(creatinine=None, age=62, sex="female")
	assert not result.ok and result.value is None and "creatinine" in result.missing
	assert not ckd_epi_2021(creatinine=1.0, age=62).ok


def test_children_and_unknown_sex_are_missing():
	assert "adults" in ckd_epi_2021(creatinine=0.6, age=12, sex="male").missing
	assert "sex" in ckd_epi_2021(creatinine=1.0, age=40, sex="unknown").missing


def test_result_cites_inputs_and_formula():
	result = ckd_epi_2021(creatinine=1.9, age=62, sex="female")
	assert result.inputs == {"creatinine_mg_dl": 1.9, "age": 62, "sex": "female"}
	assert "CKD-EPI 2021" in result.formula


def test_2009_reference_matches_eka_item():
	# Eka urology_nephrology_001: 62-year-old non-Black man, SCr 1.6 -> eGFR1 45.5 (tolerance 0.1)
	assert math.isclose(ckd_epi_2009(creatinine=1.6, age=62, sex="male").value, 45.5, abs_tol=0.1)
