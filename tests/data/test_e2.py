"""T24: E2 data. Pure logic on hand-made inputs; the frozen lists are checked when present."""

import json
from pathlib import Path

import pandas as pd
import pytest

from trialguide.data import e2
from trialguide.data import masking as mk
from trialguide.schemas import Split

PUBMED_XML = b"""<?xml version="1.0"?>
<PubmedArticleSet>
 <PubmedArticle><MedlineCitation><PMID>111</PMID><Article><AuthorList>
  <Author><AffiliationInfo><Affiliation>Dept of Oncology, AIIMS, New Delhi, India.</Affiliation></AffiliationInfo></Author>
  <Author><AffiliationInfo><Affiliation>Mayo Clinic, Rochester, USA.</Affiliation></AffiliationInfo></Author>
 </AuthorList></Article></MedlineCitation></PubmedArticle>
 <PubmedArticle><MedlineCitation><PMID>222</PMID><Article><AuthorList>
  <Author><AffiliationInfo><Affiliation>Indian Ocean Research Institute, Mauritius.</Affiliation></AffiliationInfo></Author>
 </AuthorList></Article></MedlineCitation></PubmedArticle>
</PubmedArticleSet>"""


def test_parse_affiliations_and_india_check():
    affs = e2.parse_affiliations(PUBMED_XML)
    assert affs["111"][0].startswith("Dept of Oncology") and len(affs["111"]) == 2
    assert e2.is_indian(affs["111"])
    assert not e2.is_indian(affs["222"])          # "Indian Ocean" is not an Indian institution
    assert not e2.is_indian(["Indiana University, USA"])
    assert not e2.is_indian([])


def _study(overall="RECRUITING", sites=(("India", "RECRUITING"),), criteria=""):
    return {"protocolSection": {
        "identificationModule": {"nctId": "NCT00000001", "briefTitle": "A trial", "officialTitle": "A trial, long"},
        "statusModule": {"overallStatus": overall},
        "contactsLocationsModule": {"locations": [{"country": c, "status": s} for c, s in sites]},
        "eligibilityModule": {"eligibilityCriteria": criteria, "minimumAge": "18 Years", "sex": "ALL"},
        "conditionsModule": {"conditions": ["Breast Cancer"]},
        "designModule": {"phases": ["PHASE3"]},
    }}


def test_recruits_in_india_needs_a_recruiting_indian_site():
    assert e2.recruits_in_india(_study())
    assert not e2.recruits_in_india(_study(sites=(("India", "NOT_YET_RECRUITING"), ("USA", "RECRUITING"))))
    assert not e2.recruits_in_india(_study(overall="COMPLETED"))
    assert e2.recruits_in_india(_study(sites=(("India", "NOT_YET_RECRUITING"), ("India", "RECRUITING"))))


CRITERIA = ("Inclusion Criteria: • Age ≥ 18years\n\n* History of seizure\n* KPS \\>= 50\n\n"
            "Exclusion Criteria:\n\n* KPS \\< 50\n1. Brain metastasis\n")


def test_split_eligibility_handles_bullets_numbers_and_escapes():
    inc, exc = e2.split_eligibility(CRITERIA)
    assert inc == ["Age ≥ 18years", "History of seizure", "KPS >= 50"]
    assert exc == ["KPS < 50", "Brain metastasis"]
    assert e2.split_eligibility("") == ([], [])
    assert e2.split_eligibility("Adults with lung cancer") == (["Adults with lung cancer"], [])


def test_corpus_entry_round_trips_through_the_masking_criteria_splitter():
    entry = e2.to_corpus_entry(_study(criteria=CRITERIA))
    assert entry["_id"] == "NCT00000001" and entry["metadata"]["diseases_list"] == ["Breast Cancer"]
    # the same splitter the masking and checker side use must recover the items
    assert mk.split_criteria(entry["metadata"]["inclusion_criteria"]) == ["Age ≥ 18years", "History of seizure", "KPS >= 50"]
    assert mk.split_criteria(entry["metadata"]["exclusion_criteria"]) == ["KPS < 50", "Brain metastasis"]


def test_age_years():
    assert e2.age_years("[[60.0, 'year']]") == 60
    assert e2.age_years("[[1.0, 'year'], [6.0, 'month']]") == 1.5
    assert e2.age_years("[]") is None and e2.age_years(None) is None and e2.age_years("garbage") is None


def test_pool_keeps_one_adult_per_article():
    adult, child = "[[40.0, 'year']]", "[[4.0, 'year']]"
    pmc = pd.DataFrame({"patient_uid": ["9-2", "9-1", "8-1", "7-1", "6-1", "6-2"],
                        "PMID": [111, 111, 333, 444, 555, 555],
                        "age": [adult, adult, adult, adult, child, adult]})
    pool = e2.build_pool(pmc, {"111", "333", "555"})
    # 444 not Indian; of 111 only the first patient; of 555 the child is skipped, its adult kept
    assert list(pool.patient_uid) == ["6-2", "8-1", "9-1"]
    assert len(e2.build_pool(pmc, {"555"}, min_age=-1)) == 1


def test_sample_is_seeded_disjoint_and_halved():
    ids = [f"p{i}" for i in range(300)]
    a, b = e2.draw_sample(ids), e2.draw_sample(list(reversed(ids)))
    assert a == b                                          # independent of input order
    assert len(a[Split.CAL]) == len(a[Split.TEST]) == 50
    assert not set(a[Split.CAL]) & set(a[Split.TEST])
    with pytest.raises(ValueError):
        e2.draw_sample(ids[:10])


def test_e2_test_half_is_sealed():
    with pytest.raises(PermissionError):
        e2.load_e2_patients(Path("data/raw"), Split.TEST)
    with pytest.raises(ValueError):
        e2.load_e2_patients(Path("data/raw"), Split.DEV)


# ---- the frozen lists, when they exist ----

frozen = pytest.mark.skipif(not (e2.E2_DIR / "e2_info.json").exists(), reason="E2 lists not built yet")


@frozen
def test_frozen_lists_unchanged_and_consistent():
    info = json.loads((e2.E2_DIR / "e2_info.json").read_text())
    for name, digest in info["sha256"].items():
        assert e2._hash(e2.E2_DIR / name) == digest, f"{name} was edited; revert it"
    cal, test = e2.read_list("e2_cal.txt"), e2.read_list("e2_test.txt")
    assert len(cal) == len(test) == e2.SAMPLE_SIZE // 2 and not set(cal) & set(test)
    assert all(t.startswith("NCT") for t in e2.read_list("e2_trials.txt"))


PMC_CSV = Path("data/raw/pmc_patients/PMC-Patients.csv")


@frozen
@pytest.mark.skipif(not PMC_CSV.exists(), reason="run the downloader with --with-pmc-patients")
def test_every_sampled_article_is_excluded_from_e1():
    pmc = pd.read_csv(PMC_CSV, usecols=["patient_uid", "PMID"])
    sampled = set(e2.read_list("e2_cal.txt") + e2.read_list("e2_test.txt"))
    pmids = set(pmc[pmc.patient_uid.isin(sampled)].PMID.astype(str))
    assert len(pmids) == e2.SAMPLE_SIZE                    # one patient per article
    assert pmids <= set(e2.read_list("e1_exclude_pmids.txt"))
