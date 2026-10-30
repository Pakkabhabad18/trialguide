"""Drug class tool, offline: RxClass responses recorded on 2026-10-25 are replayed."""

import json
import os
import tempfile

from trialguide.tools import drugs as module

RESPONSES = json.load(open(os.path.join(os.path.dirname(__file__), "rxclass_responses.json")))


def _offline(fail=False):
	"""Replays recorded responses and uses a fresh cache file."""
	def fetch(name):
		if fail:
			raise OSError("network down")
		return RESPONSES[name]
	module._fetch = fetch
	module.CACHE_PATH = os.path.join(tempfile.mkdtemp(), "rxclass_cache.json")


def test_brand_name_resolves_to_ingredient_class():
	_offline()
	result = module.drug_class("Coumadin")
	assert result.ok
	assert result.inputs["resolved_ingredients"] == ["warfarin"]
	assert result.value == [{"atc": "B01AA", "name": "Vitamin K antagonists"}]


def test_combination_products_are_dropped():
	_offline()
	result = module.drug_class("metformin")
	assert result.value == [{"atc": "A10BA", "name": "Biguanides"}]


def test_all_classes_of_a_drug_are_returned():
	_offline()
	atcs = [c["atc"] for c in module.drug_class("aspirin").value]
	assert "B01AC" in atcs and "N02BA" in atcs  # antiplatelet and analgesic


def test_platinum_compound():
	_offline()
	assert module.drug_class("cisplatin").value[0]["name"] == "Platinum compounds"


def test_unknown_drug_is_missing():
	_offline()
	result = module.drug_class("notadrug")
	assert not result.ok and "no ATC class" in result.missing


def test_no_drug_given_is_missing():
	_offline()
	assert not module.drug_class(None).ok


def test_network_failure_is_missing_and_not_cached():
	_offline(fail=True)
	result = module.drug_class("cisplatin")
	assert not result.ok and "lookup failed" in result.missing
	assert not os.path.exists(module.CACHE_PATH)


def test_lookups_are_cached():
	_offline()
	module.drug_class("cisplatin")
	module._fetch = None  # a second lookup must not touch the network
	assert module.drug_class("Cisplatin").ok
