"""Estimated GFR from serum creatinine.

The tool is CKD-EPI 2021 (Inker et al., NEJM 2021), the race-free equation the
proposal specifies:

	eGFR = 142 * min(Scr/k, 1)^a * max(Scr/k, 1)^-1.200 * 0.9938^age * 1.012 [female]
	k = 0.7 (female), 0.9 (male);  a = -0.241 (female), -0.302 (male)

with Scr in mg/dL, in mL/min/1.73 m2, for adults only.

ckd_epi_2009 is the older race-based equation. The system never uses it; it is
here because the Eka calculator set grades eGFR with it, so matching Eka checks
our arithmetic and unit handling.
"""

from trialguide.tools.base import ToolResult, absent, missing

UNIT = "mL/min/1.73 m2"
UMOL_PER_MG_DL = 88.4


def _sex(sex):
	s = str(sex).strip().lower()
	if s in ("female", "f", "woman"):
		return "female"
	if s in ("male", "m", "man"):
		return "male"
	return None


def _creatinine_mg_dl(creatinine, unit):
	u = unit.replace(" ", "").lower()
	if u in ("mg/dl", "mg/dl."):
		return creatinine
	if u in ("umol/l", "µmol/l", "μmol/l", "micromol/l"):
		return creatinine / UMOL_PER_MG_DL
	raise ValueError(f"unknown creatinine unit: {unit}")


def _checked_inputs(tool, creatinine, age, sex, creatinine_unit):
	"""Returns (scr_mg_dl, age, sex, None) or (.., .., .., ToolResult) if missing."""
	inputs = {"creatinine": creatinine, "creatinine_unit": creatinine_unit, "age": age, "sex": sex}
	gaps = absent(creatinine=creatinine, age=age, sex=sex)
	if gaps:
		return None, None, None, missing(tool, f"input not given: {', '.join(gaps)}", **inputs)
	sex_norm = _sex(sex)
	if sex_norm is None:
		return None, None, None, missing(tool, f"sex not recognised: {sex!r}", **inputs)
	if age < 18:
		return None, None, None, missing(tool, "CKD-EPI is validated for adults (age >= 18) only", **inputs)
	if creatinine <= 0:
		return None, None, None, missing(tool, "creatinine must be positive", **inputs)
	return _creatinine_mg_dl(creatinine, creatinine_unit), age, sex_norm, None


def ckd_epi_2021(creatinine=None, age=None, sex=None, creatinine_unit="mg/dL"):
	tool = "egfr_ckd_epi_2021"
	scr, age, sex, gap = _checked_inputs(tool, creatinine, age, sex, creatinine_unit)
	if gap:
		return gap

	female = sex == "female"
	k = 0.7 if female else 0.9
	a = -0.241 if female else -0.302
	value = 142 * min(scr / k, 1) ** a * max(scr / k, 1) ** -1.200 * 0.9938 ** age
	if female:
		value *= 1.012

	return ToolResult(
		tool=tool,
		value=round(value, 1),
		unit=UNIT,
		inputs={"creatinine_mg_dl": round(scr, 3), "age": age, "sex": sex},
		formula="CKD-EPI 2021 creatinine (Inker et al., NEJM 2021)",
	)


def ckd_epi_2009(creatinine=None, age=None, sex=None, black=False, creatinine_unit="mg/dL"):
	"""Reference only (not used by the system); see the module docstring."""
	tool = "egfr_ckd_epi_2009"
	scr, age, sex, gap = _checked_inputs(tool, creatinine, age, sex, creatinine_unit)
	if gap:
		return gap

	female = sex == "female"
	k = 0.7 if female else 0.9
	a = -0.329 if female else -0.411
	value = 141 * min(scr / k, 1) ** a * max(scr / k, 1) ** -1.209 * 0.993 ** age
	if female:
		value *= 1.018
	if black:
		value *= 1.159

	return ToolResult(
		tool=tool,
		value=round(value, 1),
		unit=UNIT,
		inputs={"creatinine_mg_dl": round(scr, 3), "age": age, "sex": sex, "black": black},
		formula="CKD-EPI 2009 creatinine (Levey et al., Ann Intern Med 2009)",
	)
