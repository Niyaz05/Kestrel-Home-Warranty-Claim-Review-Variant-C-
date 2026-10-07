# Cost arithmetic

All inputs are the policy's own figures (ops-policy v4.1 section 4); every result is generated from `outputs/evaluation_results.json`.

## 1. The three costs and how they are used

| Line | Policy text | How it enters the arithmetic |
|---|---|---|
| Genuine claim held | "goodwill worth Rs 380 on average" | Rs 380 for every genuine claim in the review queue |
| Fraud paid | "the full claim amount" | The claim amount of a fraud *not* stopped. A fraud in the queue saves its amount |
| Service contact | "Blended cost of a service contact: Rs 260" | **Ambiguous.** Reading A: not charged to reviews (it is overhead that exists anyway). Reading B: charged to every review. We show both and do not pick silently |

Net for a queue = (frauds caught x their amounts) - (genuine claims held x Rs 380) - (reviews x Rs 260, reading B only).

## 2. Worked example: the June 2026 queue (the regime-aware backtest)

The model trained on everything before 1 June picked the top 40 of June's 720 decided claims.

| Step | Arithmetic | Result |
|---|---|---|
| Frauds in the 40 | counted | 15 |
| Genuine claims held | 40 - 15 | 25 |
| Fraud rupees stopped | sum of the 15 claim amounts (average Rs 1,418) | Rs 21,272 |
| Goodwill paid | 25 x Rs 380 | Rs 9,500 |
| **Net, reading A** | Rs 21,272 - Rs 9,500 | **Rs 11,772** |
| Contact cost, reading B | 40 x Rs 260 | Rs 10,400 |
| **Net, reading B** | Rs 11,772 - Rs 10,400 | **Rs 1,372** |
| Per claim checked (gross / A / B) | / 40 | Rs 532 / Rs 294 / Rs 34 |

For scale: June had 22 decided frauds worth Rs 45,673 in total; the queue stopped 47% of those rupees. The other 7 frauds (Rs 24,401) were paid.

## 3. Where the 40 reviews stop paying (the marginal arithmetic)

Same June queue, cut at different depths. **Caveat:** one month, 22 frauds, and the cut-off is read off the same month; treat the shape as a hint, not a fitted rule.

| Queue depth | Frauds | Precision | Fraud Rs stopped | Goodwill | Contact (B) | Net A | Net B |
|---|---|---|---|---|---|---|---|
| top 5 a month | 3 of 5 | 60% | Rs 5,499 | Rs 760 | Rs 1,300 | Rs 4,739 | Rs 3,439 |
| top 10 a month | 7 of 10 | 70% | Rs 10,990 | Rs 1,140 | Rs 2,600 | Rs 9,850 | Rs 7,250 |
| top 20 a month | 12 of 20 | 60% | Rs 17,313 | Rs 3,040 | Rs 5,200 | Rs 14,273 | Rs 9,073 |
| top 30 a month | 14 of 30 | 47% | Rs 20,081 | Rs 6,080 | Rs 7,800 | Rs 14,001 | Rs 6,201 |
| top 40 a month | 15 of 40 | 38% | Rs 21,272 | Rs 9,500 | Rs 10,400 | Rs 11,772 | Rs 1,372 |

What each extra block of reviews added in June:

| Block | Frauds found | Fraud Rs stopped | Marginal net A | Marginal net B |
|---|---|---|---|---|
| reviews 6-10 | 4 of 5 | Rs 5,491 | Rs 5,111 | Rs 3,811 |
| reviews 11-20 | 5 of 10 | Rs 6,323 | Rs 4,423 | Rs 1,823 |
| reviews 21-30 | 2 of 10 | Rs 2,768 | -Rs 272 | -Rs 2,872 |
| reviews 31-40 | 1 of 10 | Rs 1,191 | -Rs 2,229 | -Rs 4,829 |

A block of 10 more reviews pays for itself only if it finds at least 10 x (380 + c) / (A + 380) frauds, with c = 0 (A) or 260 (B) and A the average fraud claim (Rs 1,418):
**2.1 frauds per 10 (reading A) or 3.6 per 10 (reading B).**
In June the first 20 reviews cleared that bar and the last 10 did not.

The same cut on May and June pooled (80 reviews; May is the stale-model month with 0 frauds in 40):

| Queue depth | Frauds | Precision | Fraud Rs stopped | Goodwill | Contact (B) | Net A | Net B |
|---|---|---|---|---|---|---|---|
| top 5 a month | 3 of 10 | 30% | Rs 5,499 | Rs 2,660 | Rs 2,600 | Rs 2,839 | Rs 239 |
| top 10 a month | 7 of 20 | 35% | Rs 10,990 | Rs 4,940 | Rs 5,200 | Rs 6,050 | Rs 850 |
| top 20 a month | 12 of 40 | 30% | Rs 17,313 | Rs 10,640 | Rs 10,400 | Rs 6,673 | -Rs 3,727 |
| top 30 a month | 14 of 60 | 23% | Rs 20,081 | Rs 17,480 | Rs 15,600 | Rs 2,601 | -Rs 12,999 |
| top 40 a month | 15 of 80 | 19% | Rs 21,272 | Rs 24,700 | Rs 20,800 | -Rs 3,428 | -Rs 24,228 |

**Working rule proposed:** work down the list and stop when the last block of 10 finds fewer than about 4 frauds. With a stale model that stops almost immediately, which is the point.

## 4. Break-even precision

Per review: p x A - (1 - p) x 380 - c >= 0, so **p >= (380 + c) / (A + 380)**.

| Average fraud claim in the queue | Needed precision, reading A (c = 0) | Needed precision, reading B (c = 260) |
|---|---|---|
| Rs 1,000 | 28% | 46% |
| June queue average (Rs 1,418) | 21% | 36% |
| post-May average fraud claim (Rs 1,865) | 17% | 29% |
| all-period average fraud claim (Rs 4,522) | 8% | 13% |

Bigger fraud claims lower the bar a lot. The model ranks by probability and ignores amount; ranking by expected rupees was tested and did not help in these folds (EVIDENCE section 5).

## 5. What a test month could look like (40 reviews, Jul-Sep)

Each row applies a *measured* precision from a past window to 40 reviews, with that window's own average fraud claim.

| Precision taken from | Precision | Frauds in 40 | Avg fraud claim | Fraud Rs stopped | Goodwill | Net A | Net B |
|---|---|---|---|---|---|---|---|
| February-April pooled (before the change; larger claims) | 12% | 5.0 | Rs 7,891 | Rs 39,453 | Rs 13,300 | Rs 26,153 | Rs 15,753 |
| May-June pooled (includes the stale-model May) | 19% | 7.5 | Rs 1,418 | Rs 10,636 | Rs 12,350 | -Rs 1,714 | -Rs 12,114 |
| June only (regime-aware) | 38% | 15.0 | Rs 1,418 | Rs 21,272 | Rs 9,500 | Rs 11,772 | Rs 1,372 |
| Stated expectation for Jul-Sep (25%) | 25% | 10.0 | Rs 1,418 | Rs 14,181 | Rs 11,400 | Rs 2,781 | -Rs 7,619 |

Across these four cases a month's net ranges from -Rs 1,714 to Rs 26,153 under reading A and from -Rs 12,114 to Rs 15,753 under reading B. Fraud paid in May-June averaged Rs 33,562 a month, so even the best case stops well under all of it.

## 6. The outlet rule (no model) priced the same way

Rule: send small claims (under Rs 2,000) from outlets with 3 or more confirmed frauds at the start of the month back to inspection.

| Month | Claims inspected | Frauds inside | Fraud Rs stopped | Genuine held x Rs 380 | Net A | Net B (x Rs 260 each) |
|---|---|---|---|---|---|---|
| May | 8 | 0 | Rs 0 | 8 x 380 = Rs 3,040 | -Rs 3,040 | -Rs 5,120 |
| June | 32 | 10 | Rs 13,784 | 22 x 380 = Rs 8,360 | Rs 5,424 | -Rs 2,896 |

It is not free money either: under reading B it loses in June too. Partner inspection may cost less than the desk's Rs 260 and has no 40-a-month cap, but we have no figure for it. The rule's value is that it needs no model and no desk time. It also depends on the wave having started (May: nothing).

## 7. What one prediction and one month cost to run

| Item | Arithmetic | Cost |
|---|---|---|
| Paid API calls | the product calls no external model or API | Rs 0 |
| One prediction | CatBoost + logistic regression on one row, CPU only; measured median 16 ms through the API (95th percentile 19 ms) | Rs 0 in fees |
| One month at Kestrel volume | 750 claims x Rs 0; a 750-claim batch took 9 s | Rs 0 in fees |
| Monthly refresh compute | `python -m src.refresh`, about 10 seconds on the build machine, then a restart | Rs 0 in fees |
| Hosting | one small CPU server, or run on an existing machine | **not known; not included** |
| Analyst time for the refresh | about 1 hour a month (my estimate, not measured) x Kestrel's hourly rate | **rate not known; not included** |
| Outcome recording | the desk already records investigations; the change is to leave open cases blank, never 0 | no new cost |

If an LLM were added later for wording, add tokens per call x price per token x 750 per month. It is not in this product.

## 8. What the arithmetic leaves out

* Customer annoyance beyond the Rs 380 average, and the effect on outlets that are held in error (Meenal's concern for new outlets).
* Fraud that never gets investigated: blanks are dropped, and Zoho blanks stored as 0 count as genuine here.
* Desk hours beyond the Rs 260 and the cost of re-inspection under the outlet rule.
* Deterrence: if fraudsters learn that outlets are watched, fraud may move; the model would then need the monthly refresh even more.
