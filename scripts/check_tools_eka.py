"""Check the T13 tools on the Eka medical calculator set (ekacare/medical_calculator_eval).

Runs each tool on the Eka items it can answer, with the inputs in
scripts/eka_tool_items.json, and compares with Eka's answer within Eka's
per-item tolerance. Also lists the Eka calculators our tools do not cover.

	PYTHONPATH=src python scripts/check_tools_eka.py [--out results/tools_eka_check.json]
"""

import argparse
import json
import os
import re

import pandas as pd
from huggingface_hub import hf_hub_download

from trialguide.tools.body_size import bmi
from trialguide.tools.renal import ckd_epi_2009

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = {"bmi": bmi, "egfr_ckd_epi_2009": ckd_epi_2009}
RELATED = re.compile(r"gfr|bmi|creatinine|body_mass|drug", re.I)


def numbers_in(text):
	return {float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)}


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--out", default="results/tools_eka_check.json")
	args = parser.parse_args()

	path = hf_hub_download("ekacare/medical_calculator_eval", "data/test-00000-of-00001.parquet", repo_type="dataset")
	eka = pd.read_parquet(path).set_index("id")
	items = json.load(open(os.path.join(HERE, "eka_tool_items.json")))

	rows = []
	for tool_name, tool in TOOLS.items():
		for item in items[tool_name]:
			row = eka.loc[item["id"]]
			kwargs = {k: v for k, v in item.items() if k != "id"}
			# guard against transcription mistakes: every numeric input must be in the question
			stated = numbers_in(row.question_text)
			for key, value in kwargs.items():
				if isinstance(value, (int, float)):
					assert float(value) in stated, f"{item['id']}: {key}={value} not in question text"

			result = tool(**kwargs)
			expected = json.loads(row.expected_output)[row.primary_field]
			error = abs(result.value - expected)
			rows.append({
				"id": item["id"], "calculator": row.expected_calculator, "tool": tool_name,
				"inputs": kwargs, "ours": result.value, "eka": expected,
				"tolerance": row.tolerance, "correct": bool(error <= row.tolerance),
			})

	for r in rows:
		print(f"{'OK ' if r['correct'] else 'BAD'} {r['id']:<24} {r['tool']:<18} ours {r['ours']:>7}  eka {r['eka']:>8}  tol {r['tolerance']}")

	checked = {r["id"] for r in rows}
	related = eka[eka.expected_calculator.str.contains(RELATED) & ~eka.index.isin(checked)]
	not_covered = related.groupby("expected_calculator").size().to_dict()

	summary = {
		"eka_items_total": len(eka),
		"checked": len(rows),
		"correct": sum(r["correct"] for r in rows),
		"by_tool": {t: f"{sum(r['correct'] for r in rows if r['tool'] == t)}/{sum(r['tool'] == t for r in rows)}" for t in TOOLS},
		"related_calculators_not_covered": not_covered,
		"note": "Eka grades eGFR with CKD-EPI 2009, so those items check the 2009 reference function. The CKD-EPI 2021 tool is checked by unit tests. Eka has no drug-class items.",
		"items": rows,
	}
	os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
	with open(args.out, "w") as f:
		json.dump(summary, f, indent=2)
	print(f"\n{summary['correct']}/{summary['checked']} correct {summary['by_tool']}")
	print("related Eka calculators not covered:", not_covered)
