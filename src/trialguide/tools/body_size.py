"""Body mass index: weight (kg) / height (m)^2."""

from trialguide.tools.base import ToolResult, absent, missing

KG_PER = {"kg": 1.0, "g": 0.001, "lb": 0.45359237, "lbs": 0.45359237}
M_PER = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254, "inch": 0.0254, "ft": 0.3048}


def bmi(weight=None, height=None, weight_unit="kg", height_unit="cm"):
	tool = "bmi"
	inputs = {"weight": weight, "weight_unit": weight_unit, "height": height, "height_unit": height_unit}
	gaps = absent(weight=weight, height=height)
	if gaps:
		return missing(tool, f"input not given: {', '.join(gaps)}", **inputs)

	w_unit, h_unit = weight_unit.strip().lower(), height_unit.strip().lower()
	if w_unit not in KG_PER:
		raise ValueError(f"unknown weight unit: {weight_unit}")
	if h_unit not in M_PER:
		raise ValueError(f"unknown height unit: {height_unit}")

	kg = weight * KG_PER[w_unit]
	m = height * M_PER[h_unit]
	if kg <= 0 or m <= 0:
		return missing(tool, "weight and height must be positive", **inputs)

	return ToolResult(
		tool=tool,
		value=round(kg / m ** 2, 2),
		unit="kg/m2",
		inputs={"weight_kg": round(kg, 3), "height_m": round(m, 4)},
		formula="weight (kg) / height (m)^2",
	)
