"""Drug class lookup via the NLM RxClass API (ATC classes).

Answers rules such as "no prior platinum-based chemotherapy" or "no current
anticoagulant": cisplatin -> L01XA Platinum compounds, Coumadin -> warfarin ->
B01AA Vitamin K antagonists. Brand names resolve to ingredients. A drug can have
several classes (aspirin: antiplatelet B01AC and analgesic N02BA); all are
returned and choosing the one a rule means is left to the model.

Successful lookups are cached on disk (TOOLS_CACHE), so reruns are
reproducible and work offline. Failed lookups are not cached.
"""

import json
import os
import threading
import urllib.parse
import urllib.request

from trialguide.tools.base import ToolResult, missing

URL = "https://rxnav.nlm.nih.gov/REST/rxclass/class/byDrugName.json"
CACHE_PATH = os.getenv("TOOLS_CACHE", "outputs/rxclass_cache.json")
TIMEOUT = 30

_lock = threading.Lock()


def _fetch(name):
	query = urllib.parse.urlencode({"drugName": name, "relaSource": "ATC"})
	with urllib.request.urlopen(f"{URL}?{query}", timeout=TIMEOUT) as response:
		return json.load(response)


def _cached_fetch(name):
	key = name.strip().lower()
	with _lock:
		cache = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}
	if key in cache:
		return cache[key]

	data = _fetch(key)
	with _lock:
		cache = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}
		cache[key] = data
		os.makedirs(os.path.dirname(CACHE_PATH) or ".", exist_ok=True)
		with open(CACHE_PATH, "w") as f:
			json.dump(cache, f, indent=1, sort_keys=True)
	return data


def drug_class(drug=None):
	tool = "drug_class_atc"
	if drug is None or not str(drug).strip():
		return missing(tool, "input not given: drug", drug=drug)

	try:
		data = _cached_fetch(str(drug))
	except Exception as e:
		return missing(tool, f"RxClass lookup failed: {type(e).__name__}: {e}", drug=drug)

	entries = data.get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", [])
	# drop combination products (e.g. "linagliptin / metformin") unless asked for one
	if "/" not in str(drug):
		single = [e for e in entries if "/" not in e["minConcept"]["name"]]
		entries = single or entries
	if not entries:
		return missing(tool, "no ATC class found for this drug name", drug=drug)

	classes = {}
	for e in entries:
		item = e["rxclassMinConceptItem"]
		classes[item["classId"]] = item["className"]
	ingredients = sorted({e["minConcept"]["name"] for e in entries})

	return ToolResult(
		tool=tool,
		value=[{"atc": atc, "name": name} for atc, name in sorted(classes.items())],
		inputs={"drug": drug, "resolved_ingredients": ingredients},
		formula="NLM RxClass, ATC level 4 (rxnav.nlm.nih.gov)",
	)
