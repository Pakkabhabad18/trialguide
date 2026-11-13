"""Run the tools (T13) on a visible note, so the rule checker (T12) can use their results.

The checker gets the results as an extra block in its prompt; it never computes eGFR or BMI,
or guesses a drug's class, itself (proposal Section 3). Ablation A5 turns this off.

- eGFR (CKD-EPI 2021) when the note states serum creatinine, age and sex;
- BMI when it states weight and height (and no BMI);
- ATC classes (RxClass) of every medication the note names. The model lists the medications in
  one call per note (cached); the classes come from RxClass, never from the model.

Only the visible note is read, so a hidden value is never used. A tool whose inputs are absent
gives no line: the block lists computed values only.

Owner: Prateek.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

from trialguide.tools.body_size import bmi
from trialguide.tools.drugs import drug_class
from trialguide.tools.renal import ckd_epi_2021

NUM = r"(\d+(?:\.\d+)?)"
AGE_RE = re.compile(r"\b(\d{1,3})[- ](?:year|yr)s?[- ]old\b", re.I)
FEMALE_RE = re.compile(r"\b(woman|female|girl|lady|she|her|mrs)\b", re.I)
MALE_RE = re.compile(r"\b(man|male|boy|gentleman|he|his|him|mr)\b", re.I)
CREATININE_RE = re.compile(r"creatinine[^0-9.;]{0,25}?" + NUM + r"\s*(mg/dl|[uµμ]mol/l)?", re.I)
WEIGHT_RE = re.compile(r"\bweigh(?:s|t|ing)?\b[^0-9.;]{0,25}?" + NUM + r"\s*(kg|lbs?|pounds)\b", re.I)
HEIGHT_RE = re.compile(r"\b(?:height|tall|measures)\b[^0-9.;]{0,25}?" + NUM + r"\s*(cm|m|in|inches)\b"
                       r"|" + NUM + r"\s*(cm|m)\s+tall\b", re.I)
BMI_RE = re.compile(r"\b(bmi|body mass index)\b", re.I)

DRUGS_PROMPT = """List every medication or drug the patient note names (generic or brand names), including past ones.
Do not list drug classes, procedures, or substances that are not medicines (alcohol, tobacco).
Reply with a JSON object only: {"drugs": ["name", ...]}"""


def _sex(text: str) -> Optional[str]:
    f, m = len(FEMALE_RE.findall(text)), len(MALE_RE.findall(text))
    if f == m:
        return None
    return "female" if f > m else "male"


def egfr_line(text: str) -> Optional[str]:
    age, cr = AGE_RE.search(text), CREATININE_RE.search(text)
    sex = _sex(text)
    if not (age and cr and sex):
        return None
    value = float(cr.group(1))
    unit = cr.group(2) or ("mg/dL" if value < 20 else "umol/L")   # mg/dL values are small
    unit = "mg/dL" if unit.lower() == "mg/dl" else "umol/L"
    r = ckd_epi_2021(creatinine=value, age=int(age.group(1)), sex=sex, creatinine_unit=unit)
    if not r.ok:
        return None
    return (f"eGFR {r.value} {r.unit} (CKD-EPI 2021, from creatinine {value} {unit}, "
            f"age {age.group(1)}, {sex})")


def bmi_line(text: str) -> Optional[str]:
    if BMI_RE.search(text):
        return None                                       # the note states it already
    w, h = WEIGHT_RE.search(text), HEIGHT_RE.search(text)
    if not (w and h):
        return None
    h_value, h_unit = (h.group(1), h.group(2)) if h.group(1) else (h.group(3), h.group(4))
    unit = {"lb": "lb", "lbs": "lb", "pounds": "lb"}.get(w.group(2).lower(), "kg")
    r = bmi(weight=float(w.group(1)), height=float(h_value), weight_unit=unit,
            height_unit={"inches": "in"}.get(h_unit.lower(), h_unit.lower()))
    if not r.ok:
        return None
    return f"BMI {r.value} {r.unit} (from weight {w.group(1)} {w.group(2)}, height {h_value} {h_unit})"


def note_drugs(text: str, model: str) -> list[str]:
    """Medication names in the note, listed by the model (one cached call per note)."""
    from trialguide.llm.client import chat, parse_json
    raw = chat([{"role": "system", "content": DRUGS_PROMPT},
                {"role": "user", "content": f"Patient note:\n{text}\n\nJSON reply:"}],
               model, temperature=0)
    try:
        drugs = parse_json(raw).get("drugs", [])
    except (ValueError, AttributeError, TypeError):
        return []
    seen, names = set(), []
    for d in drugs if isinstance(drugs, list) else []:
        name = str(d).strip()
        # a listed name must appear in the note: the model may not add medicines
        if name and name.lower() not in seen and name.lower() in text.lower():
            seen.add(name.lower())
            names.append(name)
    return names


def drug_lines(text: str, model: str) -> list[str]:
    lines = []
    for name in note_drugs(text, model):
        r = drug_class(name)
        if r.ok:
            classes = "; ".join(f"{c['atc']} {c['name']}" for c in r.value)
            lines.append(f"{name}: {classes}")
    return lines


@lru_cache(maxsize=4096)           # one block per note, reused for each of its rules
def tool_block(text: str, model: str) -> str:
    """The calculator results for a visible note, as a prompt block ('' if there are none)."""
    lines = [line for line in (egfr_line(text), bmi_line(text)) if line]
    drugs = drug_lines(text, model)
    if drugs:
        lines.append("ATC drug classes (NLM RxClass) of medications named in the note: " + " | ".join(drugs))
    if not lines:
        return ""
    return ("Calculator results (computed by tools from values in the visible note; cite the "
            "sentences that give the inputs):\n" + "\n".join(f"- {line}" for line in lines))
