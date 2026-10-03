# Submission form: Kestrel Home, Warranty Claim Review (Variant C)

**1. What did you build, and what business outcome does it move? State the number and the money.**
A review-queue ranking (CatBoost + logistic regression on claim facts and each outlet's as-of fraud history) with an API and a screen. It picks the 40 claims a month the investigation desk should review.
In June 2026 its 40 picks held 15 real frauds worth Rs 21,272, about Rs 532 stopped per claim checked (38% precision vs 3.1% random).
Net of Rs 380 goodwill it is Rs 11,772 for June; with Rs 260 per review also charged, Rs 1,372. In May a stale model found 0 of 40, so the number is conditional on monthly retraining.

**2. What score do you expect predictions.csv to get on the hidden outcomes, on which metric, and why that metric? How did you estimate it?**
Precision in the top 40 per month: about 25% (plausible 8-45%), i.e. about 10 frauds per 40 reviews; ROC-AUC about 0.85 (plausible 0.65-0.95).
Why: the fraud rate is 1-3% and the desk can review 40 a month, so the top of the list is what matters; accuracy would reward approving everything.
How: rolling monthly backtests with a frozen outlet-history snapshot (the way the test file is scored). June (regime-aware, 22 frauds): precision 37.5%, AUC 0.92, interval [8%, 68%].
May (stale model): 0%. Pooled May-Jun: 18.8%. I stated 25%, between the pooled and June figures, because the test months are longer, the snapshot ages up to 3 months, and the fraud pattern may drift.

**3. How do you know it works? Sample size, how you checked, error rate, what it gets wrong.**
11,146 decided claims, 141 frauds; monthly rolling-origin validation Feb-Jun 2026 (no random splits, no leakage: outlet history is as-of with a 30-day label delay); partner-clustered bootstrap intervals.
June: 15 of 40 correct, 25 genuine claims held. May: 0 of 40. It gets wrong: outlets with no history, any month after the fraud pattern changes, and genuine claims at flagged outlets.
Rules baseline in June: 12 of 40; plain logistic regression 6; CatBoost alone 17. The final blend (15) is not clearly better than CatBoost alone; the differences are within noise.

**4. Did you change, narrow or push back on the client's ask?**
Yes. (a) Replaced accuracy with rupees stopped per claim checked: "approve everything" is 97.5% accurate. (b) Tested "it's the newer partners": new outlets show 7% fraud on claims vs 1%, but most new outlets are clean and five outlets hold 83% of post-May small-claim fraud. (c) Said the queue is near break-even and recommended outlet-level action and monthly retraining.

**5. What is wrong with what you are handing us, or with the data?**
Data: 681 duplicate rows; 202 blank labels; legacy Zoho undecided cases stored as 0 (cannot be identified); 5 description rows containing instructions to an AI reviewer (train; none in test); messy serials; the policy's UTC note concerns fields not in these files; 14 test outlets never seen in training.
Mine: only 141 frauds and 22 in the June fold, so intervals are wide; blend weights and hyper-parameters were set by hand on the same folds (mildly optimistic); one regime-consistent fold; the first version of this project (class-weighted model, hand-typed documents) had wrong numbers, wrong outlet IDs, a 41-value score range and an API that did not match the CSV. All repaired; documents are now generated from the results file.
Two June-onboarded outlets (SP3118, SP3286) rank high on two claims each: thin evidence.

**6. What did you deliberately leave out, and why?**
Free text and inspection fields (no signal / regime-dependent), serial repeat counts, claim-volume counts, hour of day (noise with 141 positives), deep models, an LLM in the product (nothing to gain over templates; must run without a key), expected-rupee ranking (tested, no better).

**7. Anything you built or found that nobody asked for?**
The injected instructions in the free text; the May-fold failure showing monthly retraining is required; the outlet-level rule (no model: outlets with 3+ confirmed frauds at month start) that covered 45% of June's fraud but nothing in May; a parity test that the API and predictions.csv agree; documents generated from results.

**8. What did you use AI for?**
I have used Claude Sonnet for building this project and ChatGPT Go to clear up any doubts I had. IDE used : Google Antigravity. 

**9. Public Google Drive link:** [link]

**10. Someone picks this up on Monday and you are unreachable. The three things they need.**
(1) `pip install -r requirements.txt`, then `python -m uvicorn src.app:app`; tests: `python -m pytest tests`. Retrain with the client's `data/` folder: `python -m src.train && python -m src.make_docs`.
(2) Retrain on the first working day of each month with newly decided outcomes; a model older than a month missed all fraud in May.
(3) The list is a ranking for 40 human reviews a month, not an auto-reject; check thinly evidenced outlets by hand before acting.

**11. Honest hours spent:** 5

**12. GitHub repo link:** https://github.com/Niyaz05/Kestrel-Home-Warranty-Claim-Review-Variant-C-

**13. What does one prediction cost, and what would a month cost (about 750 claims)?**
No paid calls. A prediction is a CatBoost and a logistic regression evaluation on one row, a few milliseconds of CPU: Rs 0 per prediction. 750 x Rs 0 = Rs 0 a month, excluding the cost of running the server. If an LLM were added later for wording, the cost would be tokens per call x price per token x 750; it is not in this product.
