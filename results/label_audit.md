# Expert-label audit (T7)

Source: `ncbi/TrialGPT-Criterion-Annotations`, 1015 rule-level rows, 53 patients, 105 patient-trial pairs.

## Coverage by cohort

| Cohort | Patients (queries) | With trial labels | With rule labels | Rule-label rows |
|---|---|---|---|---|
| sigir | 59 | 58 | 52 | 997 |
| trec_2021 | 75 | 75 | 0 | 0 |
| trec_2022 | 50 | 50 | 0 | 0 |

Annotated patients missing from every cohort file: {'sigir-201426': 18}

Patients without trial labels: sigir: ['sigir-201428']

## Expert label counts

- exclusion: {'not excluded': 484, 'not enough information': 119, 'not applicable': 46, 'excluded': 10}
- inclusion: {'not enough information': 174, 'included': 150, 'not included': 23, 'not applicable': 9}

## Not enough information (NEI)

| | Rows |
|---|---|
| Expert says NEI | 293 (28.9%) |
| GPT-4 says NEI | 289 |
| Both say NEI | 255 |
| **Recoverable** (GPT-4 NEI, expert gives a definite label) | 34 |
| ... of which the expert cites note sentences | 21 |
| GPT-4 decided, expert says NEI | 21 |
| GPT-4 NEI, expert says not applicable | 0 |

Recoverable rows are 26.4% of the 129 GPT-4/expert disagreements and 11.8% of GPT-4's NEI answers (TrialGPT paper: 30.7% of 26 errors). By type: {'exclusion': 17, 'inclusion': 17}.

