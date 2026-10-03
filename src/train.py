"""
Train + evaluate + score. Run:  python -m src.train        (needs data/ with the client files)

Outputs
  model/catboost_final.cbm, model/logreg.pkl, model/model_meta.pkl  (scoring artefacts)
  model/reference.json     partner/product tables + frozen partner-history snapshot, so the API
                           starts from a clean checkout WITHOUT data/
  model/golden.json        synthetic claims + expected scores (parity test)
  predictions.csv          one row per test claim_id
  outputs/evaluation_results.json   every number quoted in EVIDENCE.md / memo / form

Final score = 0.7 * CatBoost probability + 0.3 * logistic-regression probability (see DECISIONS.md).
"""
from __future__ import annotations

import json
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from . import core
from .core import (ALL_FEATURES, CATEGORICAL_FEATURES, CONTACT_COST, GOODWILL_COST,
                   MONTHLY_CAPACITY, ROOT)

warnings.filterwarnings("ignore")
SEED = 7
DATA = ROOT / "data"
MODEL_DIR = ROOT / "model"
OUT_DIR = ROOT / "outputs"
W_CB, W_LR = 0.7, 0.3
CB_PARAMS = dict(iterations=300, depth=3, learning_rate=0.05, l2_leaf_reg=10,
                 random_seed=SEED, verbose=0, thread_count=4)


# ── model wrappers ──────────────────────────────────────────────────────

def fit_models(trn: pd.DataFrame):
    X = core.feature_matrix(trn)
    cb = CatBoostClassifier(**CB_PARAMS).fit(
        Pool(X, trn["is_fraud"], cat_features=CATEGORICAL_FEATURES))
    Xl = pd.get_dummies(X, columns=CATEGORICAL_FEATURES)
    sc = StandardScaler().fit(Xl)
    lr = LogisticRegression(C=0.05, max_iter=3000).fit(sc.transform(Xl), trn["is_fraud"])
    return {"cb": cb, "lr": lr, "scaler": sc, "lr_columns": list(Xl.columns)}


def predict_parts(models, df: pd.DataFrame):
    X = core.feature_matrix(df)
    p_cb = models["cb"].predict_proba(Pool(X, cat_features=CATEGORICAL_FEATURES))[:, 1]
    Xl = pd.get_dummies(X, columns=CATEGORICAL_FEATURES).reindex(
        columns=models["lr_columns"], fill_value=0)
    p_lr = models["lr"].predict_proba(models["scaler"].transform(Xl))[:, 1]
    return p_cb, p_lr


def blend(p_cb, p_lr):
    return W_CB * p_cb + W_LR * p_lr


# ── evaluation helpers ──────────────────────────────────────────────────

def top_k(frame: pd.DataFrame, col: str, k=MONTHLY_CAPACITY, seed=0):
    o = frame.sample(frac=1, random_state=seed)          # random tie-break
    return o.nlargest(k, col)


def queue_metrics(frames: list, col: str) -> dict:
    """frames = one DataFrame per month. Reviews the top-40 of each month."""
    tp = caught = fp = 0
    for f in frames:
        t = top_k(f, col)
        hit = t["is_fraud"] == 1
        tp += int(hit.sum()); fp += int((~hit).sum())
        caught += float(t.loc[hit, "claim_amount_inr"].sum())
    n = tp + fp
    allf = pd.concat(frames)
    net = caught - GOODWILL_COST * fp - CONTACT_COST * n
    return {
        "frauds_in_queue": tp, "reviews": n, "precision": tp / n if n else 0.0,
        "fraud_rupees_caught": caught, "false_holds": fp,
        "net_rs_goodwill_only": caught - GOODWILL_COST * fp,
        "net_rs_goodwill_and_contact": net,
        "net_rs_per_review_goodwill_and_contact": net / n if n else 0.0,
        "roc_auc": float(roc_auc_score(allf["is_fraud"], allf[col])),
        "pr_auc": float(average_precision_score(allf["is_fraud"], allf[col])),
        "positives": int(allf["is_fraud"].sum()), "claims": int(len(allf)),
    }


def bootstrap(frames: list, col: str, n_boot=1000, seed=1) -> dict:
    """Resample PARTNERS (claims from one outlet are not independent) and recompute metrics."""
    allf = pd.concat([f.assign(_m=i) for i, f in enumerate(frames)])
    partners = allf["partner_id"].unique()
    groups = {p: g for p, g in allf.groupby("partner_id")}
    rng = np.random.default_rng(seed)
    prec, caught, auc = [], [], []
    for _ in range(n_boot):
        pick = rng.choice(partners, size=len(partners), replace=True)
        b = pd.concat([groups[p] for p in pick])
        if b["is_fraud"].nunique() < 2:
            continue
        auc.append(roc_auc_score(b["is_fraud"], b[col]))
        tp = cr = n = 0
        for _, g in b.groupby("_m"):
            t = g.nlargest(min(MONTHLY_CAPACITY, len(g)), col)
            tp += t["is_fraud"].sum(); n += len(t)
            cr += t.loc[t["is_fraud"] == 1, "claim_amount_inr"].sum()
        prec.append(tp / n); caught.append(cr)
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"precision_ci": q(prec), "rupees_caught_ci": q(caught), "roc_auc_ci": q(auc)}


def accuracy_trap(frames: list, col: str) -> dict:
    a = pd.concat(frames).reset_index(drop=True)
    never = 1 - a["is_fraud"].mean()
    flagged_ids = set(pd.concat([top_k(f, col) for f in frames])["claim_id"])
    pred = a["claim_id"].isin(flagged_ids).astype(int)
    return {"accuracy_predict_nobody": float(never),
            "accuracy_top40_flagged": float((pred.values == a["is_fraud"].values).mean()),
            "fraud_rate": float(a["is_fraud"].mean())}


# ── main ────────────────────────────────────────────────────────────────

def main():
    MODEL_DIR.mkdir(exist_ok=True); OUT_DIR.mkdir(exist_ok=True)
    partners, products = core.load_reference(DATA)
    raw_train = pd.read_csv(DATA / "train.csv")
    raw_test = pd.read_csv(DATA / "test_unlabelled.csv")
    train = core.basic_features(core.clean_claims(raw_train, partners, products))
    test = core.basic_features(core.clean_claims(raw_test, partners, products))
    R: dict = {"seed": SEED, "weights": {"catboost": W_CB, "logreg": W_LR}}

    # ---- data quality ---------------------------------------------------
    dq = {"train_rows_raw": len(raw_train), "train_rows_unique_claims": len(train),
          "duplicate_rows_removed": len(raw_train) - len(train),
          "test_rows_raw": len(raw_test), "test_rows_unique_claims": len(test)}
    for name, df, col, allowed in [("train_description", raw_train, "claim_description", core.STOCK_DESCRIPTIONS),
                                   ("train_note", raw_train, "inspector_note", core.STOCK_NOTES),
                                   ("test_description", raw_test, "claim_description", core.STOCK_DESCRIPTIONS),
                                   ("test_note", raw_test, "inspector_note", core.STOCK_NOTES)]:
        flags = df[col].map(lambda t: core.sanitize_text(t, allowed)[1])
        dq[f"injected_or_unrecognised_{name}_rows"] = int(flags.sum())
        if flags.sum():
            dq[f"injected_ids_{name}"] = sorted(df.loc[flags, "claim_id"].unique().tolist())
    dec = train[train["is_fraud"].notna()].copy()
    dec["is_fraud"] = dec["is_fraud"].astype(int)
    dq.update({"undecided_blank_labels": int(train["is_fraud"].isna().sum()),
               "decided_claims": len(dec), "frauds": int(dec["is_fraud"].sum()),
               "fraud_rate": float(dec["is_fraud"].mean()),
               "legacy_rows_decided": int((dec["source"] == "legacy_zoho").sum()),
               "claims_before_partner_onboarding_train": int((train["submitted_at"] < train["onboarded_date"]).sum()),
               "claims_before_partner_onboarding_test": int((test["submitted_at"] < test["onboarded_date"]).sum()),
               "test_partners_not_in_train": int(len(set(test["partner_id"]) - set(train["partner_id"]))),
               "test_rows_unseen_partner": int((~test["partner_id"].isin(train["partner_id"])).sum()),
               "test_date_range": [str(test["submitted_at"].min()), str(test["submitted_at"].max())]})
    R["data_quality"] = dq
    print("data quality:", {k: v for k, v in dq.items() if not k.startswith("injected_ids")})

    # ---- regime facts ---------------------------------------------------
    d = dec.copy()
    pre, post = d[d.post_may == 0], d[d.post_may == 1]
    R["regime"] = {
        "pre_small": [int((pre.small_claim == 1).sum()), int(pre[pre.small_claim == 1].is_fraud.sum())],
        "pre_large": [int((pre.small_claim == 0).sum()), int(pre[pre.small_claim == 0].is_fraud.sum())],
        "post_small": [int((post.small_claim == 1).sum()), int(post[post.small_claim == 1].is_fraud.sum())],
        "post_large": [int((post.small_claim == 0).sum()), int(post[post.small_claim == 0].is_fraud.sum())],
        "near_2000": [int(d.near_2000.sum()), int(d[d.near_2000 == 1].is_fraud.sum())],
        "small_post_inspected_share": float((train[(train.post_may == 1) & (train.small_claim == 1)].partner_inspected == "Y").mean()),
        "small_pre_inspected_share": float((train[(train.post_may == 0) & (train.small_claim == 1)].partner_inspected == "Y").mean()),
        "post_may_fraud_amount_mean": float(post[post.is_fraud == 1].claim_amount_inr.mean()),
        "post_may_fraud_amount_median": float(post[post.is_fraud == 1].claim_amount_inr.median()),
        "post_may_fraud_total_rs": float(post[post.is_fraud == 1].claim_amount_inr.sum()),
        "post_may_months": 2,
    }
    pf = d.groupby("partner_id").agg(claims=("is_fraud", "size"), frauds=("is_fraud", "sum")).reset_index()
    pf = pf.merge(partners[["partner_id", "onboarded_date"]], on="partner_id")
    pf["new"] = pf["onboarded_date"] >= pd.Timestamp("2025-07-01")
    ps = post[post.small_claim == 1].groupby("partner_id").is_fraud.agg(["size", "sum"]).sort_values("sum", ascending=False)
    top5 = ps.head(5)
    dn = d.merge(pf[["partner_id", "new"]], on="partner_id")
    new_pf = pf[pf["new"]]
    R["partners"] = {
        "partners_total": len(partners),
        "new_partners_listed": int((partners["onboarded_date"] >= pd.Timestamp("2025-07-01")).sum()),
        "new_partners_onboarded_after_train_end": int((partners["onboarded_date"] > pd.Timestamp("2026-06-30")).sum()),
        "new_with_labelled_claims": len(new_pf),
        "new_with_any_fraud": int((new_pf.frauds > 0).sum()),
        "old_with_labelled_claims": int((~pf["new"]).sum()),
        "old_with_any_fraud": int((pf[~pf["new"]].frauds > 0).sum()),
        "fraud_rate_new": float(dn[dn["new"]].is_fraud.mean()),
        "fraud_rate_old": float(dn[~dn["new"]].is_fraud.mean()),
        "post_may_small_top5": [{"partner_id": i, "claims": int(r["size"]), "frauds": int(r["sum"])} for i, r in top5.iterrows()],
        "post_may_small_top5_share_of_fraud": float(top5["sum"].sum() / ps["sum"].sum()),
        "top12_partner_fraud_share": float(pf.sort_values("frauds", ascending=False).head(12).frauds.sum() / pf.frauds.sum()),
    }
    top5_ids = list(top5.index)
    pidx = partners.set_index("partner_id")
    R["partners"]["top5_onboarded"] = {p: str(pidx.loc[p, "onboarded_date"].date()) for p in top5_ids}
    print("top5 outlets:", R["partners"]["post_may_small_top5"])

    # ---- rolling-origin evaluation (monthly; snapshot at month start) ---
    month_starts = [pd.Timestamp(f"2026-{m:02d}-01") for m in range(2, 8)]
    labels = {"rand": "Random", "rules": "Rules baseline", "lr": "Logistic regression",
              "cb": "CatBoost (unweighted)", "final": "FINAL: 0.7 CatBoost + 0.3 LogReg"}
    frames_by_month = {}
    for i in range(len(month_starts) - 1):
        s, e = month_starts[i], month_starts[i + 1]
        trn = dec[dec.submitted_at < s].copy()
        val = dec[(dec.submitted_at >= s) & (dec.submitted_at < e)].copy()
        trn = core.add_asof_partner_features(trn, trn)
        val = core.add_snapshot_partner_features(val, trn, s)
        m = fit_models(trn)
        p_cb, p_lr = predict_parts(m, val)
        f = val[["claim_id", "partner_id", "is_fraud", "claim_amount_inr", "submitted_at"]].copy()
        f["rules"] = core.rules_score(val); f["lr"] = p_lr; f["cb"] = p_cb
        f["final"] = blend(p_cb, p_lr)
        f["rand"] = np.random.default_rng(SEED + i).random(len(f))
        f["ev"] = f["final"] * f["claim_amount_inr"] - (1 - f["final"]) * GOODWILL_COST
        frames_by_month[s.strftime("%b")] = f
        print(f"  fold {s:%b}: n={len(f)} frauds={int(f.is_fraud.sum())}")

    def block(months):
        fr = [frames_by_month[m] for m in months]
        out = {k: queue_metrics(fr, k) for k in labels}
        out["ev_ranked"] = queue_metrics(fr, "ev")
        out["bootstrap_final"] = bootstrap(fr, "final")
        out["accuracy_trap_final"] = accuracy_trap(fr, "final")
        base = float(pd.concat(fr)["is_fraud"].mean())
        out["random_expected_precision"] = base
        out["lift_vs_expected_random"] = out["final"]["precision"] / base
        return out
    R["eval_pre_regime_Feb_Apr"] = block(["Feb", "Mar", "Apr"])
    R["eval_May_Jun"] = block(["May", "Jun"])
    R["eval_Jun"] = block(["Jun"])
    R["eval_by_month"] = {m: queue_metrics([frames_by_month[m]], "final") for m in frames_by_month}
    R["model_labels"] = labels

    mj = pd.concat([frames_by_month["May"], frames_by_month["Jun"]])
    seen_before = set(dec[dec.submitted_at < pd.Timestamp("2026-05-01")].partner_id)
    uns = mj[~mj.partner_id.isin(seen_before)]
    R["unseen_partners_May_Jun"] = {"claims": int(len(uns)), "frauds": int(uns.is_fraud.sum()),
                                    "partners": int(uns.partner_id.nunique())}

    # ---- small ablation on the Jun fold (~22 positives: differences of 1-3 frauds are noise)
    s, e = month_starts[4], month_starts[5]
    trn = core.add_asof_partner_features(dec[dec.submitted_at < s].copy(), dec[dec.submitted_at < s])
    val = core.add_snapshot_partner_features(dec[(dec.submitted_at >= s) & (dec.submitted_at < e)].copy(), trn, s)
    abl = {}
    groups = {"full model": [],
              "no partner history": ["partner_n", "partner_fraud_count", "partner_rate", "partner_small_n", "partner_small_rate"],
              "no amount features": ["claim_amount_inr", "log_amount", "amount_ratio", "near_2000", "dist_below_2000", "amount_le_500"],
              "no customer history": ["customer_prior_claims"]}
    for name, drop in groups.items():
        t2, v2 = trn.copy(), val.copy()
        for c in drop:
            t2[c] = 0.0; v2[c] = 0.0
        m = fit_models(t2)
        p_cb, p_lr = predict_parts(m, v2)
        f = v2[["claim_id", "partner_id", "is_fraud", "claim_amount_inr"]].copy(); f["s"] = blend(p_cb, p_lr)
        q = queue_metrics([f], "s")
        abl[name] = {"frauds_in_top40": q["frauds_in_queue"], "roc_auc": q["roc_auc"]}
    R["ablation_Jun"] = abl

    # ---- staleness check: the test set is scored with a snapshot that ages up to ~3 months.
    # Same June fold, but the partner-history snapshot is frozen 0 / 30 / 60 days before June.
    m_stale = fit_models(trn)
    stale = {}
    for days in (0, 30, 60):
        v2 = core.add_snapshot_partner_features(
            dec[(dec.submitted_at >= s) & (dec.submitted_at < e)].copy(), trn, s - pd.Timedelta(days=days))
        p_cb, p_lr = predict_parts(m_stale, v2)
        f = v2[["claim_id", "partner_id", "is_fraud", "claim_amount_inr"]].copy(); f["s"] = blend(p_cb, p_lr)
        q = queue_metrics([f], "s")
        stale[f"snapshot_{days}d_old"] = {"frauds_in_top40": q["frauds_in_queue"], "roc_auc": q["roc_auc"],
                                          "fraud_rupees_caught": q["fraud_rupees_caught"]}
    R["staleness_Jun"] = stale

    # ---- outlet-level alternative: re-inspect SMALL claims from outlets with >= 3 confirmed
    # frauds as of the start of the month (no model needed). Out-of-time, snapshot at month start.
    orule = {}
    for mname, i0 in (("May", 3), ("Jun", 4)):
        s_, e_ = month_starts[i0], month_starts[i0 + 1]
        v = core.add_snapshot_partner_features(
            dec[(dec.submitted_at >= s_) & (dec.submitted_at < e_)].copy(), dec, s_)
        fl = v[(v.partner_fraud_count >= 3) & (v.small_claim == 1)]
        orule[mname] = {"claims_to_inspect": int(len(fl)), "frauds_inside": int(fl.is_fraud.sum()),
                        "outlets": int(fl.partner_id.nunique()),
                        "precision": float(fl.is_fraud.mean()) if len(fl) else 0.0,
                        "share_of_month_frauds": float(fl.is_fraud.sum() / max(v.is_fraud.sum(), 1)),
                        "fraud_rupees_inside": float(fl.loc[fl.is_fraud == 1, "claim_amount_inr"].sum())}
    R["outlet_rule"] = orule

    # ---- FINAL model: all decided claims, as-of features ---------------
    final_train = core.add_asof_partner_features(dec.copy(), dec)
    models = fit_models(final_train)
    snapshot_time = dec["submitted_at"].max()
    test_f = core.add_snapshot_partner_features(test.copy(), dec, snapshot_time)
    p_cb, p_lr = predict_parts(models, test_f)
    test_f["score"] = blend(p_cb, p_lr)
    sample = pd.read_csv(DATA / "sample_submission.csv")
    preds = sample[["claim_id"]].merge(test_f[["claim_id", "score"]], on="claim_id", how="left")
    assert preds["score"].notna().all() and len(preds) == len(sample) and preds["claim_id"].is_unique
    preds.to_csv(ROOT / "predictions.csv", index=False)
    R["predictions"] = {"rows": len(preds), "distinct_scores": int(preds["score"].nunique()),
                        "min": float(preds["score"].min()), "max": float(preds["score"].max()),
                        "mean": float(preds["score"].mean())}
    print("predictions:", R["predictions"])

    n_month = 750
    thr = float(np.quantile(preds["score"], 1 - MONTHLY_CAPACITY / n_month))
    bands = {"critical": thr,
             "high": float(np.quantile(preds["score"], 0.85)),
             "medium": float(np.quantile(preds["score"], 0.70))}
    test_f["in_queue"] = test_f["score"] >= thr
    tm = test_f.groupby(test_f["submitted_at"].dt.to_period("M"))["in_queue"].sum()
    R["queue"] = {"threshold": thr, "bands": bands, "test_flags_per_month": {str(k): int(v) for k, v in tm.items()}}
    R["test_profile"] = {"unlabelled_claims": len(test),
                         "small_claim_share": float(test.small_claim.mean()),
                         "inspected_share": float((test.partner_inspected == "Y").mean())}
    # concentration of the queue on the known outlets (out-of-sample sanity)
    R["queue"]["test_top_queue_partners"] = test_f[test_f.in_queue].partner_id.value_counts().head(8).to_dict()

    # ---- artefacts ------------------------------------------------------
    models["cb"].save_model(str(MODEL_DIR / "catboost_final.cbm"))
    joblib.dump({"lr": models["lr"], "scaler": models["scaler"], "lr_columns": models["lr_columns"]},
                MODEL_DIR / "logreg.pkl")
    ids = sorted(set(partners["partner_id"]))
    stats = core.partner_history(pd.DataFrame({"partner_id": ids}),
                                 np.full(len(ids), np.datetime64(snapshot_time), dtype="datetime64[ns]"), dec)
    stats.insert(0, "partner_id", ids)
    reference = {
        "snapshot_time": str(snapshot_time),
        "partners": {r.partner_id: {"city": r.city, "onboarded_date": str(r.onboarded_date.date()),
                                    "partner_type": r.partner_type} for r in partners.itertuples()},
        "products": {r.sku: {"family": r.family, "list_price_inr": float(r.list_price_inr),
                             "warranty_months": int(r.warranty_months)} for r in products.itertuples()},
        "partner_history": {r.partner_id: {"n": float(r.partner_n), "frauds": float(r.partner_fraud_count),
                                           "rate": float(r.partner_rate), "small_n": float(r.partner_small_n),
                                           "small_rate": float(r.partner_small_rate)} for r in stats.itertuples()},
        "overall_fraud_rate": float(dec["is_fraud"].mean()),
        "queue": R["queue"], "weights": {"catboost": W_CB, "logreg": W_LR}, "features": ALL_FEATURES,
    }
    (MODEL_DIR / "reference.json").write_text(json.dumps(reference, indent=1))
    joblib.dump({"features": ALL_FEATURES, "weights": [W_CB, W_LR], "snapshot_time": str(snapshot_time),
                 "queue": R["queue"], "seed": SEED}, MODEL_DIR / "model_meta.pkl")

    base = {"submitted_at": "2026-08-15 10:30", "sku": "KH-AF-03", "product_serial": "KH123456789",
            "days_since_purchase": 90, "claim_amount_inr": 5200, "photo_attached": "Y", "partner_inspected": "Y",
            "claim_description": "motor not running", "customer_prior_claims": 0, "source": "crm"}
    hot = top5_ids[0]
    cases = [dict(base, partner_id=partners["partner_id"].iloc[0]),
             dict(base, partner_id=hot, claim_amount_inr=1975, customer_prior_claims=2, photo_attached="N",
                  partner_inspected="N", days_since_purchase=180),
             dict(base, partner_id=hot, claim_amount_inr=1450, customer_prior_claims=1, photo_attached="Y",
                  partner_inspected="N", days_since_purchase=60),
             dict(base, partner_id=partners["partner_id"].iloc[5], claim_amount_inr=320, days_since_purchase=45)]
    gdf = core.basic_features(core.clean_claims(
        pd.DataFrame([dict(c, claim_id=f"G{i}") for i, c in enumerate(cases)]), partners, products))
    gdf = core.add_snapshot_partner_features(gdf, dec, snapshot_time)
    g_cb, g_lr = predict_parts(models, gdf)
    order = {f"G{i}": i for i in range(len(cases))}
    gscores = blend(g_cb, g_lr)
    gmap = {cid: float(s) for cid, s in zip(gdf["claim_id"], gscores)}
    (MODEL_DIR / "golden.json").write_text(json.dumps(
        [{"claim": c, "expected_score": gmap[f"G{i}"]} for i, c in enumerate(cases)], indent=1))

    (OUT_DIR / "evaluation_results.json").write_text(json.dumps(R, indent=1, default=str))
    print("done ->", OUT_DIR / "evaluation_results.json")


if __name__ == "__main__":
    sys.exit(main())
