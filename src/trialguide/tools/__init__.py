"""Calculators the model should not guess: eGFR (CKD-EPI 2021), BMI, drug class via RxClass (T13).

TOOLS maps each tool name to its function and an OpenAI-style JSON schema, so
the same definitions serve direct calls and LLM tool calling. call_tool never
raises: bad arguments come back as a missing result with the reason.

Owner: Prateek.
"""

from trialguide.tools.base import ToolResult, missing
from trialguide.tools.body_size import bmi
from trialguide.tools.drugs import drug_class
from trialguide.tools.renal import ckd_epi_2021

TOOLS = {
	"egfr_ckd_epi_2021": {
		"function": ckd_epi_2021,
		"schema": {
			"name": "egfr_ckd_epi_2021",
			"description": "Estimated GFR (mL/min/1.73 m2) from serum creatinine, age and sex, CKD-EPI 2021 race-free equation. Adults only. Use only values stated in the note.",
			"parameters": {
				"type": "object",
				"properties": {
					"creatinine": {"type": "number", "description": "serum creatinine"},
					"creatinine_unit": {"type": "string", "enum": ["mg/dL", "umol/L"]},
					"age": {"type": "number", "description": "age in years"},
					"sex": {"type": "string", "enum": ["female", "male"]},
				},
				"required": ["creatinine", "age", "sex"],
			},
		},
	},
	"bmi": {
		"function": bmi,
		"schema": {
			"name": "bmi",
			"description": "Body mass index (kg/m2) from weight and height. Use only values stated in the note.",
			"parameters": {
				"type": "object",
				"properties": {
					"weight": {"type": "number"},
					"weight_unit": {"type": "string", "enum": ["kg", "lb"]},
					"height": {"type": "number"},
					"height_unit": {"type": "string", "enum": ["cm", "m", "in", "ft"]},
				},
				"required": ["weight", "height"],
			},
		},
	},
	"drug_class_atc": {
		"function": drug_class,
		"schema": {
			"name": "drug_class_atc",
			"description": "ATC drug classes of a medication (generic or brand name), from NLM RxClass. Returns every class of the drug.",
			"parameters": {
				"type": "object",
				"properties": {"drug": {"type": "string"}},
				"required": ["drug"],
			},
		},
	},
}


def call_tool(name, arguments):
	"""Runs a tool by name; returns a ToolResult (missing on any bad input)."""
	if name not in TOOLS:
		return missing(name, f"unknown tool: {name}", **(arguments or {}))
	try:
		return TOOLS[name]["function"](**(arguments or {}))
	except (TypeError, ValueError) as e:
		return missing(name, f"bad arguments: {e}", **(arguments or {}))


__all__ = ["TOOLS", "ToolResult", "bmi", "call_tool", "ckd_epi_2021", "drug_class"]
