from fastapi.testclient import TestClient

import mock_backend

client = TestClient(mock_backend.app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_get_account_active():
    r = client.get("/account/u001")
    assert r.status_code == 200
    assert r.json()["status"] == "active"


def test_get_account_not_found():
    assert client.get("/account/u999").status_code == 404


def test_update_locked_account_forbidden():
    r = client.patch("/account/u002", json={"field": "email", "value": "x@example.com"})
    assert r.status_code == 403


def test_get_order():
    r = client.get("/order/1001")
    assert r.status_code == 200
    assert r.json()["user_id"] == "u001"


def test_get_order_not_found():
    assert client.get("/order/9999").status_code == 404


def test_refund_happy_path():
    r = client.post("/refund", json={"order_id": "1001", "user_id": "u001", "amount": 89.99})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["refund_id"].startswith("R-")


def test_refund_locked_account_forbidden():
    r = client.post("/refund", json={"order_id": "1003", "user_id": "u002", "amount": 149.0})
    assert r.status_code == 403


def test_refund_already_refunded_conflict():
    r = client.post("/refund", json={"order_id": "1005", "user_id": "u004", "amount": 59.99})
    assert r.status_code == 409


def test_refund_not_eligible():
    r = client.post("/refund", json={"order_id": "1004", "user_id": "u001", "amount": 22.0})
    assert r.status_code == 422


def test_refund_wrong_user_forbidden():
    r = client.post("/refund", json={"order_id": "1001", "user_id": "u003", "amount": 89.99})
    assert r.status_code == 403


def test_reset_password_works_even_when_locked():
    r = client.post("/reset-password", json={"user_id": "u002", "method": "email"})
    assert r.status_code == 200
    assert r.json()["success"] is True
