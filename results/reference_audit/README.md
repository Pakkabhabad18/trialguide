# Full-note checker against the experts (automatic checks for T45 and T44)

The plan has two human checks: a 200-rule audit of the model-derived full-note reference (T45) and a hand
check of 50 cited inferences (T44). **Neither human check was done.** This is the automatic part that can
be done from saved outputs: it compares the checker's judgement on the FULL note (the model-derived
reference) and the sentences it cites with the experts' labels and evidence sentences.

Source: `results/test_run/checks_full.jsonl` (full-note checker outputs of the frozen system) against the
1,015 expert rule labels, linked to our rule ids with `eval/expert_reference.match_rule` (exact or clearly
closest text, never guessed). Reproduce: `python -m trialguide.eval.reference_audit --checks
results/test_run/checks_full.jsonl --out results/reference_audit`. Post-hoc and descriptive: nothing here
changed a prompt, threshold or model, and it is not a substitute for the human checks.

## Coverage (limits the numbers)

Only **266 of the 496** expert-labelled test rules (19 of 26 patients) could be matched to a rule the test
run checked: the run covered a sample of trials per patient (4 per label) and not every expert-labelled
trial is in it. Intervals are patient-level bootstrap 95% CIs (5,000 resamples).

## 1. Is the model-derived reference close to the experts? (T45)

| | Result |
|---|---|
| Rules compared | 266 (161 the experts settled as met / not met, 86 they call "not enough information") |
| Same label as the experts (all rules) | 54.5% [49.5, 60.3] |
| Of the 161 expert-settled rules, the checker also settles | 73.9% [66.5, 81.6] |
| ... when it settles one, it is right | **90.8% [83.7, 95.6]** |
| Wrong toward eligibility, among expert-settled rules | 1.2% [0.0, 3.0] |
| Of the 86 rules the experts say the note does not settle, the checker still settles | **57.0% [41.5, 72.0]** |

Reading: where the checker commits on the full note it agrees with the experts about nine times in ten,
and it rarely errs toward eligibility. But it is not a neutral reference: it misses about a quarter of
what the experts settle, and it settles more than half of the rules the experts say the note cannot
settle (the experts' "not enough information" is stricter than the checker's). Any result that uses the
full-note checker as a reference (calibration on model-derived references, E2) should say so, and the
expert-only analysis is the safer one.

## 2. Do the cited sentences match the experts' evidence? (T44)

On the 65 rules the experts settled from cited sentences, where the checker also decided and cited something:

| | Result |
|---|---|
| Precision (cited sentences that are the experts') | 84.0% [69.6, 96.3] |
| Recall (expert sentences that the checker cites) | 73.9% [63.5, 85.5] |
| Exactly the same sentences | 76.9% [66.2, 87.0] |
| At least one sentence in common | 95.4% [90.8, 100] |
| Cited ids that are not sentences of the note | 0 |

Reading: when the checker decides, it points at the same evidence as the experts in nearly every case,
and never cites a sentence that does not exist. Evidence exists only for the SIGIR patients with rule
labels, so this says nothing about TREC notes. Still open: the human check that each cited sentence
actually supports the status (verbatim and sufficient), the 50-inference hand check of the plan.
