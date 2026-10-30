"""Extension E2 data (T24): Indian-affiliated oncology case reports and trials recruiting in India.

    python -m trialguide.data.e2 --build    # query PubMed + ClinicalTrials.gov, write the frozen lists (once)
    python -m trialguide.data.e2            # check the committed lists against the cached raw files

Patients: PMC-Patients V1 summaries whose source article (a) has at least one author affiliation
naming India and (b) is an oncology case report (PubMed: case reports, Neoplasms as a MAJOR MeSH
topic). Adults only (18 or older in the report), because almost all trials recruiting in India
are adult-only. One patient per article. A seeded sample of 100 is split 50 calibration / 50 test, so
E2 can be recalibrated on its own calibration half (proposal, E2).

Trials: ClinicalTrials.gov studies that are recruiting overall AND at an Indian site, for a
cancer condition, converted to the TrialGPT corpus format (so retrieval and the checker run
unchanged).

PubMed and ClinicalTrials.gov change daily, so the committed ID lists in splits/e2/ ARE the
dataset; raw API answers are cached in data/raw/e2/ (git-ignored). Limits: docs/e2_limits.md.
"""

import argparse
import ast
import hashlib
import json
import random
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import pandas as pd

from trialguide.data.patients import Patient, split_sentences
from trialguide.schemas import Split

PUBMED_TERM = "India[ad] AND case reports[pt] AND neoplasms[majr]"
CTGOV_COND = "cancer OR neoplasm OR tumor OR carcinoma OR lymphoma OR leukemia OR myeloma OR sarcoma"
SAMPLE_SIZE = 100
MIN_AGE_YEARS = 18
SEED = 20261030                         # fixed once, chosen before building (not searched)
E2_DIR = Path(__file__).resolve().parents[3] / "splits" / "e2"

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_CTGOV = "https://clinicaltrials.gov/api/v2/studies"
_INDIA = re.compile(r"\bIndia\b")       # "Indian" alone (e.g. "Indian Ocean") does not match


# ---------------------------------------------------------------------------
# Network (only used by --build)
# ---------------------------------------------------------------------------

def _get(url: str, params: dict, data: bool = False, retries: int = 4) -> bytes:
    """GET (or POST when data=True) with retries; NCBI allows 3 requests a second without a key."""
    body = urllib.parse.urlencode(params).encode() if data else None
    full = url if data else f"{url}?{urllib.parse.urlencode(params)}"
    for attempt in range(1, retries + 1):
        try:
            time.sleep(0.4)
            with urllib.request.urlopen(urllib.request.Request(full, data=body), timeout=90) as r:
                return r.read()
        except Exception:
            if attempt == retries:
                raise
            time.sleep(3 * attempt)


def _pubmed_count(term: str, lo: str, hi: str) -> int:
    q = {"db": "pubmed", "term": term, "retmode": "json", "retmax": 0, "datetype": "pdat",
         "mindate": lo, "maxdate": hi, "tool": "trialguide"}
    return int(json.loads(_get(f"{_EUTILS}/esearch.fcgi", q))["esearchresult"]["count"])


def pubmed_ids(term: str = PUBMED_TERM, first_year: int = 1950, last_year: int = None) -> list:
    """All PMIDs for a query. PubMed returns at most 9,999 ids per query, so publication-date
    ranges are halved until each fits."""
    last_year = last_year or date.today().year
    out, stack = set(), [(first_year, last_year)]
    while stack:
        a, b = stack.pop()
        lo, hi = f"{a}/01/01", f"{b}/12/31"
        n = _pubmed_count(term, lo, hi)
        if n == 0:
            continue
        if n > 9999:
            if a == b:
                raise RuntimeError(f"more than 9,999 hits in one year ({a}); split by month")
            mid = (a + b) // 2
            stack += [(a, mid), (mid + 1, b)]
            continue
        q = {"db": "pubmed", "term": term, "retmode": "json", "retmax": 9999, "datetype": "pdat",
             "mindate": lo, "maxdate": hi, "tool": "trialguide"}
        out.update(json.loads(_get(f"{_EUTILS}/esearch.fcgi", q))["esearchresult"]["idlist"])
    return sorted(out, key=int)


def parse_affiliations(xml_bytes: bytes) -> dict:
    """{pmid: [affiliation strings]} from a PubMed efetch XML answer."""
    out = {}
    for art in ET.fromstring(xml_bytes).iter("PubmedArticle"):
        pmid = art.findtext("MedlineCitation/PMID")
        out[pmid] = [a.text.strip() for a in art.iter("Affiliation") if a.text]
    return out


def pubmed_affiliations(pmids: list, batch: int = 200) -> dict:
    out = {}
    for i in range(0, len(pmids), batch):
        q = {"db": "pubmed", "id": ",".join(pmids[i:i + batch]), "retmode": "xml", "tool": "trialguide"}
        out.update(parse_affiliations(_get(f"{_EUTILS}/efetch.fcgi", q, data=True)))
    return out


def india_trials() -> list:
    """Full ClinicalTrials.gov records: recruiting, a site in India, a cancer condition."""
    params = {"query.locn": "India", "filter.overallStatus": "RECRUITING", "query.cond": CTGOV_COND,
              "pageSize": 100, "format": "json"}
    studies, token = [], None
    while True:
        page = json.loads(_get(_CTGOV, {**params, **({"pageToken": token} if token else {})}))
        studies += page.get("studies", [])
        token = page.get("nextPageToken")
        if not token:
            return studies


# ---------------------------------------------------------------------------
# Pure logic (unit-tested)
# ---------------------------------------------------------------------------

def is_indian(affiliations: list) -> bool:
    return any(_INDIA.search(a) for a in affiliations)


def recruits_in_india(study: dict) -> bool:
    """Overall status recruiting AND at least one Indian site recruiting. A trial whose Indian
    sites are all 'not yet recruiting' or closed cannot enrol an Indian patient now."""
    p = study["protocolSection"]
    if p.get("statusModule", {}).get("overallStatus") != "RECRUITING":
        return False
    sites = p.get("contactsLocationsModule", {}).get("locations", [])
    return any(s.get("country") == "India" and s.get("status") == "RECRUITING" for s in sites)


def _criteria_lines(block: str) -> list:
    """Bullet or line items of one criteria section, without markdown escapes."""
    items = []
    for line in block.splitlines():
        line = re.sub(r"\\([<>*_#\[\]()-])", r"\1", line).strip()
        line = re.sub(r"^([*•\-]|\d+[.)])\s*", "", line).strip()
        if line:
            items.append(line)
    return items


def split_eligibility(text: str) -> tuple:
    """(inclusion items, exclusion items) from ClinicalTrials.gov eligibility text."""
    text = text or ""
    m = re.search(r"exclusion criteria\s*:?", text, flags=re.I)
    inc, exc = (text[:m.start()], text[m.end():]) if m else (text, "")
    inc = re.sub(r"^\s*inclusion criteria\s*:?", "", inc, flags=re.I)
    return _criteria_lines(inc), _criteria_lines(exc)


def to_corpus_entry(study: dict) -> dict:
    """One trial in the TrialGPT corpus format: criteria as blank-line separated items under
    'inclusion criteria:' / 'exclusion criteria:' headers, as the masking and checker code expect."""
    p = study["protocolSection"]
    ident, elig = p["identificationModule"], p.get("eligibilityModule", {})
    inc, exc = split_eligibility(elig.get("eligibilityCriteria", ""))
    return {
        "_id": ident["nctId"],
        "title": ident.get("officialTitle") or ident.get("briefTitle", ""),
        "text": p.get("descriptionModule", {}).get("briefSummary", ""),
        "metadata": {
            "brief_title": ident.get("briefTitle", ""),
            "phase": ", ".join(p.get("designModule", {}).get("phases", [])),
            "diseases_list": p.get("conditionsModule", {}).get("conditions", []),
            "inclusion_criteria": "inclusion criteria: \n\n " + " \n\n ".join(inc),
            "exclusion_criteria": "exclusion criteria: \n\n " + " \n\n ".join(exc),
            "minimum_age": elig.get("minimumAge", ""),
            "maximum_age": elig.get("maximumAge", ""),
            "sex": elig.get("sex", ""),
        },
    }


_UNIT_YEARS = {"year": 1, "month": 1 / 12, "week": 1 / 52, "day": 1 / 365, "hour": 1 / 8760}


def age_years(age_field) -> float:
    """PMC-Patients age, e.g. "[[1.0, 'year'], [2.0, 'month']]", in years; None if missing."""
    try:
        parts = ast.literal_eval(age_field) if isinstance(age_field, str) else age_field
        return sum(v * _UNIT_YEARS[u] for v, u in parts) if parts else None
    except (ValueError, SyntaxError, KeyError, TypeError):
        return None


def build_pool(pmc: pd.DataFrame, indian_pmids: set, min_age: float = MIN_AGE_YEARS) -> pd.DataFrame:
    """PMC-Patients rows from verified Indian-affiliated oncology articles, adults only, one patient
    per article (the lowest-numbered adult), so two patients of one article never land in
    different halves."""
    rows = pmc[pmc.PMID.astype(str).isin(indian_pmids)].copy()
    rows = rows[rows.age.map(age_years).fillna(-1) >= min_age]
    rows["_n"] = rows.patient_uid.str.rsplit("-", n=1).str[1].astype(int)
    rows = rows.sort_values(["PMID", "_n"]).drop_duplicates("PMID")
    return rows.drop(columns="_n").sort_values("patient_uid").reset_index(drop=True)


def draw_sample(pool_ids: list, n: int = SAMPLE_SIZE, seed: int = SEED) -> dict:
    """Seeded sample of n patients, split in half: {Split.CAL: [...], Split.TEST: [...]}."""
    if len(pool_ids) < n:
        raise ValueError(f"pool has {len(pool_ids)} patients, fewer than the {n} needed")
    chosen = random.Random(seed).sample(sorted(pool_ids), n)
    return {Split.CAL: sorted(chosen[: n // 2]), Split.TEST: sorted(chosen[n // 2:])}


# ---------------------------------------------------------------------------
# Frozen files and loading
# ---------------------------------------------------------------------------

def _write_list(path: Path, ids) -> None:
    path.write_text("\n".join(str(i) for i in ids) + "\n", encoding="utf-8", newline="\n")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


LISTS = ("e2_cal.txt", "e2_test.txt", "e2_trials.txt", "e1_exclude_pmids.txt")


def read_list(name: str, directory: Path = E2_DIR) -> list:
    return (directory / name).read_text(encoding="utf-8").split()


def load_e2_patients(raw: Path, split: Split, allow_test: bool = False) -> dict:
    """{patient_uid: Patient} for the E2 calibration or test half. The E2 test half is sealed in
    the same way as the main test split."""
    if split == Split.TEST and not allow_test:
        raise PermissionError("The E2 test half is sealed until the E2 evaluation (T38).")
    if split not in (Split.CAL, Split.TEST):
        raise ValueError("E2 has a calibration and a test half only")
    ids = set(read_list(f"e2_{split.value}.txt"))
    pmc = pd.read_csv(Path(raw) / "pmc_patients" / "PMC-Patients.csv", usecols=["patient_uid", "patient"])
    pmc = pmc[pmc.patient_uid.isin(ids)]
    return {u: Patient(u, "pmc_patients", split, split_sentences(t, "pmc_patients"))
            for u, t in zip(pmc.patient_uid, pmc.patient)}


def load_e2_trials(raw: Path) -> list:
    """The frozen E2 trials, in corpus format, from the cached ClinicalTrials.gov answer."""
    keep = set(read_list("e2_trials.txt"))
    with open(Path(raw) / "e2" / "corpus.jsonl", encoding="utf-8") as f:
        return [t for t in map(json.loads, f) if t["_id"] in keep]


# ---------------------------------------------------------------------------
# Build once, check afterwards
# ---------------------------------------------------------------------------

def build(raw: Path) -> dict:
    if (E2_DIR / "e2_cal.txt").exists():
        raise SystemExit("splits/e2/ exists: the E2 lists are frozen and are never rewritten")
    cache = Path(raw) / "e2"
    cache.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()

    pmids = pubmed_ids()
    affs = pubmed_affiliations(pmids)
    (cache / "pubmed_affiliations.json").write_text(json.dumps(affs))
    verified = {p for p, a in affs.items() if is_indian(a)}

    pmc = pd.read_csv(Path(raw) / "pmc_patients" / "PMC-Patients.csv",
                      usecols=["patient_uid", "PMID", "title", "age", "gender"])
    in_pmc = set(pmc.PMID.astype(str))
    any_age = build_pool(pmc, verified, min_age=-1)
    pool = build_pool(pmc, verified)
    sample = draw_sample(list(pool.patient_uid))

    studies = india_trials()
    (cache / "ctgov_studies.json").write_text(json.dumps(studies))
    trials = sorted(to_corpus_entry(s)["_id"] for s in studies if recruits_in_india(s))
    with open(cache / "corpus.jsonl", "w", encoding="utf-8") as f:
        for s in studies:
            f.write(json.dumps(to_corpus_entry(s)) + "\n")

    E2_DIR.mkdir(parents=True, exist_ok=True)
    _write_list(E2_DIR / "e2_cal.txt", sample[Split.CAL])
    _write_list(E2_DIR / "e2_test.txt", sample[Split.TEST])
    _write_list(E2_DIR / "e2_trials.txt", trials)
    # E1 must never train on E2 patients: exclude every article of the pool, not only the sample
    _write_list(E2_DIR / "e1_exclude_pmids.txt", sorted(set(pool.PMID.astype(str)), key=int))
    info = {
        "built": today, "seed": SEED, "sample_size": SAMPLE_SIZE,
        "pubmed_term": PUBMED_TERM, "ctgov_condition_query": CTGOV_COND,
        "counts": {
            "pubmed_hits": len(pmids),
            "pubmed_affiliation_verified": len(verified),
            "verified_and_in_pmc_patients": len(verified & in_pmc),
            "articles_any_age": len(any_age),
            "pool_adults_one_per_article": len(pool),
            "ctgov_studies_returned": len(studies),
            "trials_recruiting_at_an_indian_site": len(trials),
        },
        "pool_gender": pool.gender.value_counts().to_dict(),
        "sha256": {n: _hash(E2_DIR / n) for n in LISTS},
    }
    (E2_DIR / "e2_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8", newline="\n")
    return info


def check(raw: Path) -> bool:
    """Committed lists unchanged, and (if raw files are cached) the sample re-derives exactly."""
    info = json.loads((E2_DIR / "e2_info.json").read_text())
    ok = all(_hash(E2_DIR / n) == h for n, h in info["sha256"].items())
    cache = Path(raw) / "e2"
    if ok and (cache / "pubmed_affiliations.json").exists():
        affs = json.loads((cache / "pubmed_affiliations.json").read_text())
        pmc = pd.read_csv(Path(raw) / "pmc_patients" / "PMC-Patients.csv", usecols=["patient_uid", "PMID", "age"])
        pool = build_pool(pmc, {p for p, a in affs.items() if is_indian(a)})
        sample = draw_sample(list(pool.patient_uid))
        ok = sample[Split.CAL] == read_list("e2_cal.txt") and sample[Split.TEST] == read_list("e2_test.txt")
    return ok


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default="data")
    p.add_argument("--build", action="store_true", help="query the APIs and write the frozen lists (once)")
    a = p.parse_args(argv)
    raw = Path(a.root) / "raw"
    if a.build:
        print(json.dumps(build(raw), indent=2))
        return 0
    ok = check(raw)
    print("E2 lists unchanged and reproducible:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
