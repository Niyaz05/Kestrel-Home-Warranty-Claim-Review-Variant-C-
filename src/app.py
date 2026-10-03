"""
Kestrel warranty-claim scoring service.

  POST /score   one claim as JSON -> score, risk band, queue flag, plain-English reasons, warnings
  POST /batch   CSV upload -> list of the same objects
  GET  /health  model/reference status
  GET  /        single-page screen

Starts from a clean checkout: everything it needs is in model/ (no data/ folder, no API key).
If model files are missing it still starts, /health says "degraded" and /score returns 503.
No LLM is used: reasons are deterministic templates filled with facts from the claim and the
partner-history snapshot, so there is nothing to fail without a key.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from . import core

ROOT = core.ROOT
MODEL_DIR = ROOT / "model"
TEMPLATES_DIR = ROOT / "templates"
MODEL_VERSION = "v2.0 (0.7 CatBoost + 0.3 LogReg, snapshot partner history)"

app = FastAPI(title="Kestrel Home - Warranty Claim Review", version="2.0.0")
STATE: dict = {"ready": False, "error": None}


def load_artifacts():
    try:
        STATE["ref"] = json.loads((MODEL_DIR / "reference.json").read_text())
        STATE["cb"] = CatBoostClassifier().load_model(str(MODEL_DIR / "catboost_final.cbm"))
        STATE["lr"] = joblib.load(MODEL_DIR / "logreg.pkl")
        STATE["ready"], STATE["error"] = True, None
    except Exception as exc:  # fail politely, keep serving /health and the screen
        STATE["ready"], STATE["error"] = False, f"{type(exc).__name__}: {exc}"


load_artifacts()


class ClaimInput(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), json_schema_extra={"example": {
        "submitted_at": "2026-08-15 10:30", "partner_id": "SP3160", "sku": "KH-AF-03",
        "product_serial": "KH123456789", "days_since_purchase": 180, "claim_amount_inr": 1975,
        "photo_attached": "N", "partner_inspected": "N", "claim_description": "motor not running",
        "customer_prior_claims": 2, "source": "crm"}})
    claim_id: Optional[str] = "WC000000"
    submitted_at: str = Field(..., description="IST timestamp, e.g. 2026-08-15 10:30")
    partner_id: str
    sku: str
    product_serial: Optional[str] = ""
    days_since_purchase: int = Field(..., ge=0)
    claim_amount_inr: float = Field(..., gt=0)
    photo_attached: str = Field(default="N", pattern="^[YN]$")
    partner_inspected: str = Field(default="N", pattern="^[YN]$")
    claim_description: Optional[str] = ""
    inspector_note: Optional[str] = ""
    customer_prior_claims: int = Field(default=0, ge=0)
    source: str = "crm"


class ScoreResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    claim_id: str
    score: float = Field(..., ge=0, le=1, description="Estimated fraud probability")
    risk_band: str
    investigate: bool = Field(..., description="In the monthly top-40 review queue")
    reasons: List[str]
    model_version: str
    warnings: List[str] = []


def _frame(claim: ClaimInput):
    """Single claim -> feature frame via the SAME core code as training. Returns (df, warnings)."""
    ref, warns = STATE["ref"], []
    p = ref["partners"].get(claim.partner_id)
    if p is None:
        warns.append(f"Unknown partner {claim.partner_id}: no history available, scored on claim facts only.")
        p = {"city": "unknown", "onboarded_date": None, "partner_type": "unknown"}
    pr = ref["products"].get(claim.sku)
    if pr is None:
        warns.append(f"Unknown SKU {claim.sku}: assumed 12-month warranty, list price unknown.")
        pr = {"family": "unknown", "list_price_inr": np.nan, "warranty_months": 12}
    try:
        ts = pd.Timestamp(claim.submitted_at)
        if ts is pd.NaT:
            raise ValueError
    except Exception:
        raise HTTPException(status_code=422, detail="submitted_at is not a valid timestamp (use e.g. 2026-08-15 10:30).")
    row = {"claim_id": claim.claim_id or "WC000000", "submitted_at": ts, "partner_id": claim.partner_id,
           "sku": claim.sku, "product_serial": claim.product_serial or "",
           "days_since_purchase": claim.days_since_purchase, "claim_amount_inr": claim.claim_amount_inr,
           "photo_attached": claim.photo_attached, "partner_inspected": claim.partner_inspected,
           "customer_prior_claims": claim.customer_prior_claims, "source": claim.source,
           "onboarded_date": pd.Timestamp(p["onboarded_date"]) if p["onboarded_date"] else pd.NaT,
           "partner_type": p["partner_type"], "city": p["city"], "family": pr["family"],
           "list_price_inr": pr["list_price_inr"], "warranty_months": pr["warranty_months"]}
    df = pd.DataFrame([row])
    df["serial_clean"] = df["product_serial"].map(core.normalize_serial)
    if df["serial_clean"].iloc[0] and len(df["serial_clean"].iloc[0]) != 11:
        warns.append(f"Serial '{claim.product_serial}' normalises to {len(df['serial_clean'].iloc[0])} characters (expected 11); check it.")
    if ts < pd.Timestamp(ref["snapshot_time"]) - pd.Timedelta(days=1):
        warns.append("Claim date is before the model's history snapshot; partner history reflects the snapshot, not that date.")
    if row["onboarded_date"] is not pd.NaT and ts < row["onboarded_date"]:
        warns.append("Claim is dated before the partner's onboarding date; check the data.")
    for label, text, allowed in [("claim_description", claim.claim_description, core.STOCK_DESCRIPTIONS),
                                 ("inspector_note", claim.inspector_note, core.STOCK_NOTES)]:
        if core.sanitize_text(text, allowed)[1]:
            warns.append(f"{label} contains unrecognised or extra text; it was ignored (free text is never used for scoring).")
    df = core.basic_features(df)
    h = ref["partner_history"].get(claim.partner_id, {"n": 0, "frauds": 0, "rate": ref["overall_fraud_rate"],
                                                       "small_n": 0, "small_rate": ref["overall_fraud_rate"]})
    df["partner_n"], df["partner_fraud_count"], df["partner_rate"] = h["n"], h["frauds"], h["rate"]
    df["partner_small_n"], df["partner_small_rate"] = h["small_n"], h["small_rate"]
    return df, warns, h


def _score(df: pd.DataFrame) -> float:
    X = core.feature_matrix(df)
    p_cb = STATE["cb"].predict_proba(Pool(X, cat_features=core.CATEGORICAL_FEATURES))[:, 1]
    lr = STATE["lr"]
    Xl = pd.get_dummies(X, columns=core.CATEGORICAL_FEATURES).reindex(columns=lr["lr_columns"], fill_value=0)
    p_lr = lr["lr"].predict_proba(lr["scaler"].transform(Xl))[:, 1]
    w = STATE["ref"]["weights"]
    return float(w["catboost"] * p_cb[0] + w["logreg"] * p_lr[0])


def _band(score: float) -> str:
    b = STATE["ref"]["queue"]["bands"]
    return ("CRITICAL" if score >= b["critical"] else "HIGH" if score >= b["high"]
            else "MEDIUM" if score >= b["medium"] else "LOW")


def _reasons(claim: ClaimInput, df: pd.DataFrame, h: dict) -> List[str]:
    r, ref, out = df.iloc[0], STATE["ref"], []
    base = ref["overall_fraud_rate"]
    if h["n"] >= 5 and h["frauds"] >= 1 and h["rate"] > 2 * base:
        out.append(f"Outlet {claim.partner_id}: {int(h['frauds'])} of {int(h['n'])} past decided claims were fraud "
                   f"(company average {base:.1%}).")
    elif h["n"] == 0:
        out.append(f"Outlet {claim.partner_id} has no decided claims on record, so there is no outlet history to judge it on.")
    if r["near_2000"]:
        out.append(f"Claim is Rs {claim.claim_amount_inr:,.0f}, just under the Rs 2,000 limit above which inspection is required.")
    if r["post_may_small"] and claim.partner_inspected == "N":
        out.append("Small claim with no inspection (auto-approved since 1 May 2026): fraud is about 3% in this group vs about 0.3% for large claims.")
    if claim.customer_prior_claims >= 2:
        out.append(f"Customer has {claim.customer_prior_claims} earlier warranty claims (fraud rate rises sharply from 2).")
    if r["photo_flag"] == 0:
        out.append("No photo attached.")
    if r["outside_warranty"]:
        out.append("Product is older than its standard warranty period.")
    if r["amount_le_500"]:
        out.append("Very small claim (Rs 500 or less): historically almost never fraud.")
    if not out:
        out.append("No strong risk signal: amount, outlet history and customer history all look ordinary.")
    return out[:5]


def score_one(claim: ClaimInput) -> ScoreResponse:
    if not STATE["ready"]:
        raise HTTPException(status_code=503, detail=f"Model not loaded ({STATE['error']}). Run: python -m src.train")
    df, warns, h = _frame(claim)
    s = _score(df)
    return ScoreResponse(claim_id=claim.claim_id or "WC000000", score=round(s, 5), risk_band=_band(s),
                         investigate=bool(s >= STATE["ref"]["queue"]["threshold"]),
                         reasons=_reasons(claim, df, h), model_version=MODEL_VERSION, warnings=warns)


@app.get("/health")
def health():
    return {"status": "healthy" if STATE["ready"] else "degraded", "model_loaded": STATE["ready"],
            "error": STATE["error"], "version": MODEL_VERSION,
            "history_snapshot": STATE.get("ref", {}).get("snapshot_time")}


@app.post("/score", response_model=ScoreResponse)
def score(claim: ClaimInput):
    return score_one(claim)


@app.post("/batch")
async def batch(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}")
    need = ["submitted_at", "partner_id", "sku", "claim_amount_inr", "days_since_purchase"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")
    if len(df) > 5000:
        raise HTTPException(status_code=400, detail="Batch limited to 5,000 rows")
    out = []
    for _, row in df.iterrows():
        try:
            clean = {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in row.items()}
            clean = {k: v for k, v in clean.items() if k in ClaimInput.model_fields and v is not None}
            if "claim_id" in clean:
                clean["claim_id"] = str(clean["claim_id"])
            out.append(score_one(ClaimInput(**clean)).model_dump())
        except HTTPException as exc:
            out.append({"claim_id": str(row.get("claim_id", "unknown")), "error": str(exc.detail)})
        except Exception as exc:
            out.append({"claim_id": str(row.get("claim_id", "unknown")), "error": str(exc)[:200]})
    return JSONResponse(content=out)


@app.get("/", response_class=HTMLResponse)
def index():
    p = TEMPLATES_DIR / "index.html"
    if p.exists():
        return HTMLResponse(p.read_text())
    return HTMLResponse("<h1>Kestrel Fraud Scorer</h1><p>Screen template missing. POST /score still works.</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="127.0.0.1", port=8000)
