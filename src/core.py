"""
Shared data cleaning + feature engineering for Kestrel warranty-claim scoring.

ONE implementation is used by training, by predictions.csv and by the API, so the
three can never drift apart (the first version had three separate copies).

Design rules
------------
* Free text is never a model feature. It is only sanitised for display.
* Partner history is "as-of": for training rows it uses only labelled claims that are
  at least LABEL_DELAY_DAYS older than the claim; for scoring it uses a frozen snapshot.
* No inspection features (partner_inspected / inspector_note): they exist only for
  inspected claims, and inspection stopped for small claims on 1 May 2026.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

POLICY_CHANGE = pd.Timestamp("2026-05-01")
SMALL_CLAIM_LIMIT = 2000
LABEL_DELAY_DAYS = 30          # a fraud outcome is not known the day the claim is filed
SMOOTHING = 10.0               # Bayesian shrinkage strength for partner rates
PRIOR_RATE = 0.0127            # overall fraud rate, fallback only
MONTHLY_CAPACITY = 40          # investigation desk limit (ops-policy s5)
GOODWILL_COST = 380.0          # per genuine claim held (ops-policy s4)
CONTACT_COST = 260.0           # blended service-contact cost (see DECISIONS.md)

STOCK_DESCRIPTIONS = {
    "blade jammed", "burning smell", "display not working", "filter indicator stuck",
    "loud noise while running", "motor not running", "not charging",
    "power button not working", "remote not working", "tripping mcb",
    "unit not heating", "water leaking",
}
STOCK_NOTES = {
    "customer has bill, serial verified", "pcb replaced under warranty",
    "photos match fault, approved", "heating element open circuit",
    "minor fault, part swapped", "motor winding failure confirmed",
    "unit inspected, fault confirmed",
}

NUMERIC_FEATURES = [
    "claim_amount_inr", "log_amount", "amount_ratio", "near_2000", "dist_below_2000",
    "small_claim", "amount_le_500",
    "days_since_purchase", "warranty_fraction", "outside_warranty", "days_365_540",
    "customer_prior_claims", "photo_flag",
    "partner_tenure_days", "serial_valid",
    "post_may", "post_may_small",
    "partner_n", "partner_fraud_count", "partner_rate", "partner_small_n", "partner_small_rate",
]
CATEGORICAL_FEATURES = ["partner_type", "family"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ── text handling ───────────────────────────────────────────────────────

def normalize_serial(s) -> str:
    """Upper-case, strip spaces/dashes/punctuation: ' kh-556901419 ' -> 'KH556901419'."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def sanitize_text(text, allowed: set) -> tuple[str, bool]:
    """
    Whitelist approach (replaces the first keyword detector, which flagged every
    'Motor winding fAIlure confirmed' note because 'ai' is a substring of 'failure').

    Returns (clean_text, was_modified). Text is kept only if it is an exact stock phrase,
    or starts with one (anything appended after the stock phrase is dropped and flagged).
    Anything else is blanked and flagged. Nothing here ever feeds the model.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return "", False
    raw = str(text).strip()
    if raw == "":
        return "", False
    low = raw.lower()
    if low in allowed:
        return raw, False
    head = re.split(r"[.;\[\n]", low, maxsplit=1)[0].strip()
    if head in allowed:
        return raw[: len(head)], True
    return "", True


# ── loading / cleaning ──────────────────────────────────────────────────

def load_reference(data_dir: Path):
    partners = pd.read_csv(data_dir / "partners.csv", parse_dates=["onboarded_date"])
    products = pd.read_csv(data_dir / "products.csv")
    return partners, products


def clean_claims(df: pd.DataFrame, partners: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Dedupe (keep the last submission per claim_id), parse, merge reference data."""
    df = df.copy()
    df["submitted_at"] = pd.to_datetime(df["submitted_at"])
    df = df.sort_values("submitted_at", kind="mergesort").drop_duplicates("claim_id", keep="last")
    df = df.reset_index(drop=True)
    df["serial_clean"] = df["product_serial"].map(normalize_serial)
    df = df.merge(partners, on="partner_id", how="left").merge(products, on="sku", how="left")
    return df


def basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features that need nothing but the claim itself plus partner/product reference data."""
    d = df
    amt = d["claim_amount_inr"].astype(float)
    warranty_days = d["warranty_months"].fillna(12) * 30.44
    d["log_amount"] = np.log1p(amt)
    d["amount_ratio"] = amt / d["list_price_inr"].fillna(amt * 2).clip(lower=1)
    d["near_2000"] = ((amt >= 1900) & (amt < SMALL_CLAIM_LIMIT)).astype(int)
    d["dist_below_2000"] = (SMALL_CLAIM_LIMIT - amt).clip(0, SMALL_CLAIM_LIMIT)
    d["small_claim"] = (amt < SMALL_CLAIM_LIMIT).astype(int)
    d["amount_le_500"] = (amt <= 500).astype(int)
    d["warranty_fraction"] = d["days_since_purchase"] / warranty_days.clip(lower=1)
    d["outside_warranty"] = (d["days_since_purchase"] > warranty_days).astype(int)
    d["days_365_540"] = d["days_since_purchase"].between(365, 540).astype(int)
    d["photo_flag"] = (d["photo_attached"] == "Y").astype(int)
    onboarded = d["onboarded_date"] if "onboarded_date" in d else pd.NaT
    d["partner_tenure_days"] = (d["submitted_at"] - onboarded).dt.days.fillna(0).astype(float)
    d["serial_valid"] = (d["serial_clean"].str.len() == 11).astype(int)
    d["post_may"] = (d["submitted_at"] >= POLICY_CHANGE).astype(int)
    d["post_may_small"] = d["post_may"] * d["small_claim"]
    d["partner_type"] = d["partner_type"].fillna("unknown").astype(str)
    d["family"] = d["family"].fillna("unknown").astype(str)
    return d


# ── partner history (as-of / snapshot) ──────────────────────────────────

def _prefix_counts(times: np.ndarray, labels: np.ndarray, cuts: np.ndarray):
    """For each cut: (#labelled claims with time <= cut, #frauds among them). `times` sorted."""
    idx = np.searchsorted(times, cuts, side="right")
    cum = np.concatenate([[0], np.cumsum(labels)])
    return idx, cum[idx]


def partner_history(query: pd.DataFrame, cuts: np.ndarray, decided: pd.DataFrame) -> pd.DataFrame:
    """
    For every row of `query` return the partner's labelled history up to its own cut time.

    cuts : array of np.datetime64, same length as query. Only labelled claims submitted at or
           before the cut are visible. Training rows use claim_time - LABEL_DELAY; scoring uses
           one frozen snapshot time.
    """
    dec = decided.sort_values("submitted_at")
    t_all = dec["submitted_at"].values.astype("datetime64[ns]")
    y_all = dec["is_fraud"].values.astype(int)
    small = dec["small_claim"].values == 1
    t_s, y_s = t_all[small], y_all[small]

    cuts = np.asarray(cuts, dtype="datetime64[ns]")
    gn, gf = _prefix_counts(t_all, y_all, cuts)
    gsn, gsf = _prefix_counts(t_s, y_s, cuts)
    g_rate = np.where(gn >= 200, gf / np.maximum(gn, 1), PRIOR_RATE)
    gs_rate = np.where(gsn >= 200, gsf / np.maximum(gsn, 1), PRIOR_RATE)

    n = np.zeros(len(query)); f = np.zeros(len(query))
    ns = np.zeros(len(query)); fs = np.zeros(len(query))
    pid = query["partner_id"].values
    by_partner = {p: g for p, g in dec.groupby("partner_id")}
    for p in pd.unique(pid):
        rows = np.where(pid == p)[0]
        g = by_partner.get(p)
        if g is None:
            continue
        tp = g["submitted_at"].values.astype("datetime64[ns]")
        yp = g["is_fraud"].values.astype(int)
        n[rows], f[rows] = _prefix_counts(tp, yp, cuts[rows])
        sm = g["small_claim"].values == 1
        ns[rows], fs[rows] = _prefix_counts(tp[sm], yp[sm], cuts[rows])

    out = pd.DataFrame(index=query.index)
    out["partner_n"] = n
    out["partner_fraud_count"] = f
    out["partner_rate"] = (f + SMOOTHING * g_rate) / (n + SMOOTHING)
    out["partner_small_n"] = ns
    out["partner_small_rate"] = (fs + SMOOTHING * gs_rate) / (ns + SMOOTHING)
    return out


def add_asof_partner_features(df: pd.DataFrame, decided: pd.DataFrame) -> pd.DataFrame:
    """Training-style features: each claim sees labels only from claims >= LABEL_DELAY_DAYS older."""
    cuts = (df["submitted_at"] - pd.Timedelta(days=LABEL_DELAY_DAYS)).values
    return df.join(partner_history(df, cuts, decided))


def add_snapshot_partner_features(df: pd.DataFrame, decided: pd.DataFrame, snapshot_time) -> pd.DataFrame:
    """Scoring-style features: every claim sees all labels up to one frozen snapshot time."""
    cuts = np.full(len(df), np.datetime64(pd.Timestamp(snapshot_time)), dtype="datetime64[ns]")
    return df.join(partner_history(df, cuts, decided))


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[ALL_FEATURES].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].fillna("unknown").astype(str)
    for c in NUMERIC_FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    return X


# ── simple rules score (also the transparent baseline) ──────────────────

def rules_score(df: pd.DataFrame) -> np.ndarray:
    """Hand-written risk score from the findings in EVIDENCE.md. Higher = riskier."""
    pc = df["customer_prior_claims"].values
    s = np.select([pc >= 3, pc == 2, pc == 1], [0.30, 0.15, 0.05], default=0.0)
    s = s + 0.15 * df["near_2000"].values + 0.10 * df["post_may_small"].values
    s = s + 2.0 * df["partner_small_rate"].values
    s = s - 0.10 * df["amount_le_500"].values - 0.05 * df["days_365_540"].values
    return s.astype(float)
