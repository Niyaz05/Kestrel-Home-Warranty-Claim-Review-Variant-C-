# Decisions and ambiguities

| # | Question | Decision | Why |
|---|---|---|---|
| 1 | Undecided labels | Drop the 202 blank rows from training/evaluation | They are not negatives |
| 2 | Legacy Zoho 0s may be undecided | Keep, cannot identify; state the risk | Dropping all legacy rows loses 4,606 of 11,146 claims |
| 3 | Duplicate claim_ids | Keep the last submission; test has none | A bounced claim re-submitted is one claim |
| 4 | UTC vs IST | Nothing to convert | The UTC issue applies to resolution events, absent from these files; `submitted_at` is IST |
| 5 | Rs 260 contact cost | Reported both ways | Policy text is ambiguous (per review vs blended overhead) |
| 6 | Label delay | 30 days for training rows; frozen snapshot for scoring | An outcome is not known on filing day. The first version said 14 in the docs and used 30 in code; docs are now generated |
| 7 | Inspection fields | Excluded | Exist only for inspected claims; inspection of small claims stopped on 1 May |
| 8 | Free text | Never a feature, never sent to an LLM; whitelist check; 5 injected rows in train, 0 in test | The 12 + 7 stock phrases carry no signal; injection rows ask for a misleading evaluation |
| 9 | Scoring-time partner history | One frozen snapshot (labels up to the last labelled claim) | Same code path for the API, predictions.csv and validation |
| 10 | Class weighting | None | The first version used scale_pos_weight ~80, which squeezed every score into 0.44-0.56 with 41 distinct values |
| 11 | Final model | 0.7 CatBoost (depth 3) + 0.3 logistic regression, probabilities | Rank-averaging and the rules score were tried; this blend was not worse than either part and breaks ties. Hand-set, see EVIDENCE section 4 |
| 12 | KPI | Pushed back: accuracy replaced by rupees stopped per claim checked | Accuracy is 97.5% for doing nothing |
| 13 | Public vs private repo | Private; `model/reference.json` holds per-outlet fraud counts | Client data must not be published |
| 14 | Reasons text | Deterministic templates filled from the claim and outlet history. No LLM in the product | Must run with no key; nothing to fail |
