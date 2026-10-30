"""Shared result type for the clinical tools (T13).

A tool never guesses: if an input is absent or out of the formula's range, the
result is missing and says why (proposal Section 3: "If any input is absent,
the value is missing, not inferred"). A computed result carries its inputs and
formula so an inference can cite them.
"""

from dataclasses import dataclass, field


@dataclass
class ToolResult:
	tool: str
	value: object = None             # number, list of classes, ... or None if missing
	unit: str = ""
	inputs: dict = field(default_factory=dict)
	formula: str = ""
	missing: str = ""                # why there is no value; empty when computed

	@property
	def ok(self):
		return not self.missing

	def to_dict(self):
		return {
			"tool": self.tool,
			"value": self.value,
			"unit": self.unit,
			"inputs": self.inputs,
			"formula": self.formula,
			"missing": self.missing,
		}


def missing(tool, reason, **inputs):
	return ToolResult(tool=tool, inputs=inputs, missing=reason)


def absent(**inputs):
	"""Names of inputs that were not given."""
	return [name for name, value in inputs.items() if value is None or value == ""]
