# Kestrel Home: warranty claim review (Variant C)

Ranks warranty claims by fraud risk so the investigation desk (40 reviews a month) looks at the right ones.
A FastAPI service plus one screen. Needs no data folder and no API key to run.

**Read first:** `memo_to_ritu.md` (decision, number, rupees, next week), `EVIDENCE.md` (how we know, and how often it fails), `COSTS.md` (the rupee arithmetic) and `REFRESH.md` (how it is kept current).

## Run it (clean machine)

Python 3.9 to 3.12.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn src.app:app --port 8000
```

Open <http://localhost:8000> for the screen (buttons load a risky claim and a clean claim, and a CSV can be dropped for batch scoring). <http://localhost:8000/docs> has the API docs.

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/score -H "Content-Type: application/json" -d '{
  "submitted_at": "2026-08-15 10:30", "partner_id": "SP3160", "sku": "KH-AF-03",
  "product_serial": "KH123456789", "days_since_purchase": 180, "claim_amount_inr": 1975,
  "photo_attached": "N", "partner_inspected": "N", "claim_description": "motor not running",
  "customer_prior_claims": 2, "source": "crm"}'
```

The response has `score` (estimated fraud probability), `risk_band`, `investigate` (true if the claim would fall in the monthly top-40 list),
`reasons` (plain English, built from templates: no LLM, nothing to fail without a key) and `warnings` (unknown outlet, odd serial, ignored free text).
If the model files are missing the service still starts; `/health` says `degraded` and `/score` returns 503 with the fix.

## Test it

```bash
python -m pytest tests -q
```

Passes without the client data (one test that compares ids with `sample_submission.csv` skips itself when `data/` is absent).
It checks: predictions file shape, API vs the offline pipeline on fixed claims, bad input, unknown outlet/SKU, injected text, batch upload, leakage guards on the outlet-history features, and the refresh tool's month windows and live-model scoring.

## Monthly refresh (needs the client's `data/` folder)

```bash
python -m src.refresh              # dry run: backtest, check the live model, staged retrain, gates -> outputs/refresh_report.md
python -m src.refresh --promote    # same, then install the new model and predictions.csv if no hard gate failed
```

Run it on the first working day of each month. Procedure, gates, rollback and off-cycle triggers: `REFRESH.md`.

## Retrain and rebuild the documents (needs the client's `data/` folder)

```bash
# data/ must contain train.csv, test_unlabelled.csv, partners.csv, products.csv, sample_submission.csv
python -m src.train        # about a minute; writes model/, predictions.csv, outputs/evaluation_results.json
python -m src.make_docs    # rewrites EVIDENCE.md, memo, DECISIONS, recording script, form from the results file
```

Retrain on the first working day of every month. A model trained before the 1 May 2026 rule change found no fraud in May (see EVIDENCE section 5.4).

## What is in the box

| Path | What |
|---|---|
| `predictions.csv` | One score per test claim (higher = more likely fraud) |
| `src/core.py` | Cleaning and features: the single code path used by training, `predictions.csv` and the API |
| `src/train.py` | Validation, final model, snapshot, predictions, results JSON |
| `src/app.py`, `templates/index.html` | Service and screen |
| `model/` | CatBoost + logistic regression, `reference.json` (outlet/product tables and history snapshot), `golden.json` (parity test) |
| `src/refresh.py`, `REFRESH.md`, `docs/` | Monthly refresh tool, its runbook, and three real dry-run reports (stale model caught, June reproduced, bad batch blocked) |
| `COSTS.md` | Cost arithmetic: worked June example, where reviews stop paying, break-even precision, run costs |
| `src/bench.py` | Measures per-prediction time (`outputs/latency.json`) |
| `EVIDENCE.md`, `memo_to_ritu.md`, `DECISIONS.md`, `recording_script.md`, `submission-form.md` | Deliverables (generated from `outputs/evaluation_results.json`) |

## Honest limits

* 141 frauds in total, 22 in the one regime-consistent test month (June). Intervals are wide; see EVIDENCE.
* The history snapshot in `model/reference.json` stops at 30 Jun 2026. Newer outcomes only count after retraining.
* New outlets have no history, so they are scored on the claim alone (the API warns).
* Free text is never used for scoring. Five rows in the training file contain instructions addressed to an AI reviewer; they are ignored and listed in EVIDENCE.
* **Keep this repository private.** `model/reference.json` holds per-outlet fraud counts derived from client data. Never commit `data/`.
