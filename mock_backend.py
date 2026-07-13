"""
mock_backend.py — Simple FastAPI backend for the Multi-Agent Customer Support System.

Purpose: Give agent tools something real to act on.
Design: In-memory dicts only (~130 lines). No auth, no migrations, no complexity.
Intentional failure modes: locked accounts, already-refunded orders, unknown users —
so the Action Agent has real errors to reason about in its ReAct loop.

Run: uvicorn mock_backend:app --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
import uuid

app = FastAPI(title="Support Mock Backend", version="1.0.0")

# ---------------------------------------------------------------------------
# In-memory "database"
# ---------------------------------------------------------------------------

ACCOUNTS = {
    "u001": {"name": "Alice Johnson",  "email": "alice@example.com",  "status": "active",  "plan": "premium"},
    "u002": {"name": "Bob Smith",      "email": "bob@example.com",    "status": "locked",  "plan": "basic"},
    "u003": {"name": "Carol White",    "email": "carol@example.com",  "status": "active",  "plan": "basic"},
    "u004": {"name": "Dan Brown",      "email": "dan@example.com",    "status": "active",  "plan": "premium"},
    "u_repeat": {"name": "Eve Repeat", "email": "eve@example.com",   "status": "active",  "plan": "basic"},
}

ORDERS = {
    "1001": {"user_id": "u001", "status": "delivered", "amount": 89.99,  "delivery_date": "2024-01-10", "refund_eligible": True,  "refunded": False},
    "1002": {"user_id": "u003", "status": "delivered", "amount": 34.50,  "delivery_date": "2024-01-08", "refund_eligible": True,  "refunded": False},
    "1003": {"user_id": "u002", "status": "delivered", "amount": 149.00, "delivery_date": "2024-01-05", "refund_eligible": True,  "refunded": False},
    "1004": {"user_id": "u001", "status": "in_transit","amount": 22.00,  "delivery_date": None,         "refund_eligible": False, "refunded": False},
    "1005": {"user_id": "u004", "status": "delivered", "amount": 59.99,  "delivery_date": "2023-11-01", "refund_eligible": False, "refunded": True},
}

REFUNDS: dict[str, dict] = {}          # refund_id → refund record
PASSWORD_RESETS: dict[str, dict] = {}  # user_id  → reset record

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class RefundRequest(BaseModel):
    order_id: str
    user_id: str
    amount: float
    reason: str = "customer_request"

class PasswordResetRequest(BaseModel):
    user_id: str
    method: str = "email"   # email | sms

class AccountUpdateRequest(BaseModel):
    field: str   # email | plan | name
    value: str

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/account/{user_id}")
def get_account(user_id: str):
    if user_id not in ACCOUNTS:
        raise HTTPException(status_code=404, detail=f"Account '{user_id}' not found")
    return {"user_id": user_id, **ACCOUNTS[user_id]}


@app.patch("/account/{user_id}")
def update_account(user_id: str, body: AccountUpdateRequest):
    if user_id not in ACCOUNTS:
        raise HTTPException(status_code=404, detail=f"Account '{user_id}' not found")
    if ACCOUNTS[user_id]["status"] == "locked":
        raise HTTPException(status_code=403, detail="Account is locked. Cannot update.")
    if body.field not in ("email", "plan", "name"):
        raise HTTPException(status_code=400, detail=f"Field '{body.field}' is not updatable")
    ACCOUNTS[user_id][body.field] = body.value
    return {"success": True, "user_id": user_id, "updated": {body.field: body.value}}


@app.get("/order/{order_id}")
def get_order(order_id: str):
    if order_id not in ORDERS:
        raise HTTPException(status_code=404, detail=f"Order '{order_id}' not found")
    order = ORDERS[order_id]
    return {"order_id": order_id, **order}


@app.post("/refund")
def process_refund(body: RefundRequest):
    # Intentional failure: order not found
    if body.order_id not in ORDERS:
        raise HTTPException(status_code=404, detail=f"Order '{body.order_id}' not found")

    order = ORDERS[body.order_id]

    # Intentional failure: account locked
    user = ACCOUNTS.get(body.user_id)
    if user and user["status"] == "locked":
        raise HTTPException(status_code=403, detail="Account is locked. Refund cannot be processed until account is unlocked.")

    # Intentional failure: order belongs to different user
    if order["user_id"] != body.user_id:
        raise HTTPException(status_code=403, detail="Order does not belong to this user")

    # Intentional failure: already refunded
    if order["refunded"]:
        raise HTTPException(status_code=409, detail="Refund already processed for this order")

    # Intentional failure: not eligible (in transit / outside window)
    if not order["refund_eligible"]:
        raise HTTPException(status_code=422, detail="Order is not eligible for refund (in transit or outside refund window)")

    # Success path
    refund_id = f"R-{uuid.uuid4().hex[:8].upper()}"
    ORDERS[body.order_id]["refunded"] = True
    REFUNDS[refund_id] = {
        "refund_id":  refund_id,
        "order_id":   body.order_id,
        "user_id":    body.user_id,
        "amount":     body.amount,
        "reason":     body.reason,
        "status":     "approved",
        "created_at": datetime.utcnow().isoformat(),
        "eta_days":   3,
    }
    return {"success": True, **REFUNDS[refund_id]}


@app.post("/reset-password")
def reset_password(body: PasswordResetRequest):
    if body.user_id not in ACCOUNTS:
        raise HTTPException(status_code=404, detail=f"User '{body.user_id}' not found")

    if ACCOUNTS[body.user_id]["status"] == "locked":
        # Password reset IS allowed for locked accounts — this is intentional
        # so the Action Agent can unlock a locked user via reset
        pass

    reset_token = uuid.uuid4().hex
    PASSWORD_RESETS[body.user_id] = {
        "user_id":    body.user_id,
        "method":     body.method,
        "token":      reset_token,
        "sent_to":    ACCOUNTS[body.user_id].get("email", "unknown"),
        "expires_at": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
    }
    return {
        "success":  True,
        "message":  f"Password reset link sent via {body.method}",
        "sent_to":  PASSWORD_RESETS[body.user_id]["sent_to"],
        "expires_in_minutes": 60,
    }


@app.get("/refund/{refund_id}")
def get_refund(refund_id: str):
    if refund_id not in REFUNDS:
        raise HTTPException(status_code=404, detail=f"Refund '{refund_id}' not found")
    return REFUNDS[refund_id]
