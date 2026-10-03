# Memo to Ritu: warranty fraud model

**Decision.** Use the model to choose which 40 claims a month the investigation desk reviews. Do not auto-reject anything. Stop using accuracy as the KPI.

**Why not accuracy.** Approving every claim is 97.5% accurate and stops no fraud. Even a perfect 40-a-month list would score about the same, and a realistic one scores lower, so accuracy cannot tell a useful model from none. The number to track is **rupees of fraud stopped per claim checked**.

**The number.** In June 2026 the 40 claims the model picked included 15 real frauds (38%) worth Rs 21,272: **about Rs 532 stopped per claim checked**. Picking at random would find about 1 fraud.
For the July-September claims I expect about 10 frauds in each 40 (a plausible range is 3 to 18), roughly Rs 14,181 stopped a month, against about Rs 33,562 a month of fraud paid in May-June.

**The rupees, honestly.** Each genuine claim held costs Rs 380 goodwill. To pay for itself, a review must find fraud about 21% of the time (36% if the Rs 260 contact cost is charged per review). My expectation, 25%, is near break-even:
about Rs 2,781 a month after goodwill, -Rs 7,619 if Rs 260 per review is counted. In May the model, trained before the rule change, found nothing. **It must be retrained every month.**

**New partners.** Your instinct has some support: new outlets show 7% fraud on claims against 1% for older ones. But 38 of the 46 new outlets with claims have none, as Meenal says. The problem is a handful of outlets:
SP3160 (10 of 12), SP3232 (7 of 10), SP3318 (5 of 10), SP3129 (4 of 9), SP3319 (3 of 7). They hold 83% of post-May small-claim fraud.

**Next week.**
1. Have Meenal's team review the outlets that fill the top of the list: SP3118, SP3286, SP3160, SP3129, SP3319, SP3318, SP3232, SP3300. SP3118 and SP3286 are June-onboarded with two claims each, and SP3300 has no history, so check by hand before suspending anyone.
2. Put small claims (under Rs 2,000) from outlets with 3 or more confirmed frauds back on inspection. In June that rule would have covered 32 claims from 10 outlets and 45% of the month's fraud, with no model (in May it would have caught nothing: the wave had not started).
3. Ask the investigation desk to record every outcome within 30 days; open cases are being stored as 0 and that hides fraud from the model.
4. Re-run the training script on the first working day of each month.

**What it will miss.** First-time fraud from outlets with no history (about 81 test claims come from 14 such outlets). Evidence rests on 141 frauds in total, so treat every figure here as an estimate.
