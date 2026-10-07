# Monthly refresh: how the model is kept current

**Short version.** On the first working day of every month: update `data/`, run `python -m src.refresh`, read `outputs/refresh_report.md`, and if the hard gates pass run it again with `--promote` and restart the service. About 10 seconds of compute; most of the time is reading the report.

## 1. Why monthly

* **The May 2026 failure.** A model trained before the 1 May rule change found 0 frauds in its 40 picks for May (ROC-AUC 0.24). The pattern it had never seen (small claims, no inspection, a few outlets) was the whole story. See EVIDENCE section 5.4.
* **Outcomes arrive about 30 days late.** On 1 July you can judge June's list, not July's. That lag is why the cadence is monthly and why the refresh report starts with a check of the *live* model on the month that just ended.
* **The service scores with a frozen snapshot** of each outlet's fraud history (`model/reference.json`, dated in `/health`). New outcomes count only after a refresh.

## 2. Monthly procedure

| # | Who | Step |
|---|---|---|
| 1 | Investigation desk | Record every outcome as fraud / not fraud within 30 days. **Leave open cases blank, never 0** (the Zoho 0-for-blank problem hides fraud from the model). |
| 2 | Data and IT | Export ALL labelled claims to date (not just last month) as `data/train.csv`; export the claims to be scored as `data/test_unlabelled.csv`; refresh `partners.csv` and `products.csv` (new outlets and SKUs must be in them). Same columns as the original files. |
| 3 | Analyst | `python -m src.refresh` (dry run). Read `outputs/refresh_report.md`. |
| 4 | Analyst | If no HARD FAIL: `python -m src.refresh --promote`. This keeps the old model in `model_prev/`, installs the new one in `model/` and writes the new `predictions.csv`. |
| 5 | Analyst | Restart the service (it loads the model at start). Check `curl /health`: `history_snapshot` must show the new date. Run `python -m pytest tests -q`. |
| 6 | Ops head | Tell the desk this month's list, and any outlet named in the warnings (see section 4). |

## 3. What the refresh does and does not change

| Changes every refresh | Does not change (needs a human decision and a re-run of `src.train` and `src.make_docs`) |
|---|---|
| CatBoost and logistic-regression weights, refit on all labelled claims | The feature list, the 0.7/0.3 blend, hyper-parameters |
| Each outlet's fraud history snapshot | The 30-day label delay, the 40 a month cap |
| The queue threshold and risk bands (top about 5.3% of scored claims, i.e. 40 of ~750) | The Rs 380 / Rs 260 cost inputs (they live in `src/core.py`) |
| `predictions.csv`, `model/golden.json` | |

## 4. The checks (`outputs/refresh_report.md`)

1. **Backtest of the last 3 complete months**: train before each month, frozen snapshot at the month start, top-40 queue, precision, lift over random, ROC-AUC, net rupees. The window rolls forward automatically.
2. **Incumbent check**: the model currently in `model/` scored on the months since it was trained, exactly as the live service scores. This is the check that would have caught May.
3. **Gates**

| Gate | Type | Fires when | What to do |
|---|---|---|---|
| Distinct scores | HARD | fewer than 90% of claims have a distinct score | Do not promote. Ties would decide the queue. Check the export. |
| Score spread | HARD | scores span less than 0.30 | Do not promote. Model is not separating claims. |
| Queue size | HARD | flags outside 2.7%-10.7% of claims | Do not promote. Wrong data or a broken threshold. |
| Incumbent went stale | WARNING | live model lift under 2x random or AUC under 0.60 on the last month | Promote the new model; ask what changed (rule, outlet, product). |
| Newest backtest month weak | WARNING | AUC under 0.60 or lift under 2x | The recipe is struggling; look at the month before changing anything. |
| Outlets with no history | WARNING | over 5% of claims | Scored on claim facts only; review by hand. |
| Undecided share | WARNING | over 15% of the last 60 days still blank | Recent outlet history is understated; chase the desk for outcomes. |
| Mean score moved | WARNING | new mean is outside 0.5x-2x of the old on the same claims | Check the export; a real fraud wave also does this. |

A WARNING never blocks promotion by itself; a person reads it. A HARD FAIL leaves the live model and `predictions.csv` untouched.

## 5. Rollback

```bash
rm -rf model && mv model_prev model     # back to last month's model
# restore last month's predictions.csv from your own archive, then restart the service
```

Archive each month's `outputs/refresh_report.md` and `predictions.csv`; the tool keeps only one previous model.

## 6. Off-cycle triggers (refresh before the month ends)

* A change to the claim rules (the Rs 2,000 limit, inspection rules): the pattern will move. Refresh once about 30 days of outcomes exist under the new rule, and expect a weak period before that.
* A batch of new outlets (more than about 5% of claims from outlets with no history).
* A sudden change in how many claims the queue flags per week, or in the share of small claims.
* A warning from the incumbent check in the middle of a month (if the desk has outcomes early).

## 7. Evidence that the procedure works (dry runs on the real export)

These are real outputs of `src.refresh`, kept in `docs/`:

* `docs/refresh_dryrun_stale_incumbent_may.md`: pretend it is 1 June with a live model trained to 30 April. The report shows the live model on May: 0 of 40, ROC-AUC 0.24, and raises `INCUMBENT WENT STALE`.
* `docs/refresh_dryrun_june.md`: pretend it is 1 July with a live model trained to 31 May. The live check on June reproduces the offline backtest (15 of 40, ROC-AUC 0.92), so the live scoring path and the backtest agree.
* `docs/refresh_dryrun_gate_failure.md`: a deliberately broken batch (300 identical claims) fails three hard gates; `--promote` leaves the live model untouched and writes no predictions.

## 8. Limits

* Staleness is only visible after outcomes arrive (about 30 days). The warnings about unseen outlets, score movement and flag counts are early hints, not proof.
* The incumbent check needs at least 100 decided claims and both outcomes in the month; otherwise it says so.
* With 141 frauds in total, a single month's numbers move a lot. Do not react to one month's precision alone: the gates are deliberately coarse.
* The refresh never changes features or weights of the blend by itself. If the recipe stays weak for two refreshes in a row, that is a modelling task, not an operations task.
