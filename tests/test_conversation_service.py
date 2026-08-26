import pytest

import conversation_service
import db.crud as crud
from db.session import SessionLocal, init_db


@pytest.fixture
def db_session():
    init_db()
    session = SessionLocal()
    yield session
    session.close()


def test_compose_ticket_text_passthrough_when_not_awaiting(db_session):
    conv = crud.create_conversation(db_session, user_id="u001")
    assert conv.awaiting_customer_input is False
    result = conversation_service._compose_ticket_text(db_session, conv, "fresh new issue")
    assert result == "fresh new issue"


def test_compose_ticket_text_chains_across_request_info_round_trip(db_session):
    conv = crud.create_conversation(db_session, user_id="u004")
    crud.set_conversation_awaiting(db_session, conv.id, True)
    db_session.refresh(conv)

    crud.create_message(db_session, conv.id, "user", "I want a refund but forgot my order id")
    crud.create_ticket_run(
        db_session, conversation_id=conv.id, user_id="u004",
        ticket_text="I want a refund but forgot my order id",
        status="completed", state_json={},
    )
    crud.create_message(
        db_session, conv.id, "assistant", "Sure — what's your order ID?",
    )

    composed = conversation_service._compose_ticket_text(db_session, conv, "1005")
    assert "I want a refund but forgot my order id" in composed
    assert "[Assistant asked]: Sure — what's your order ID?" in composed
    assert "[Customer replied]: 1005" in composed


def test_build_next_turn_state_includes_history_and_flag(db_session):
    conv = crud.create_conversation(db_session, user_id="u001")
    crud.create_message(db_session, conv.id, "user", "hello")
    crud.create_message(db_session, conv.id, "assistant", "hi there")

    state = conversation_service.build_next_turn_state(db_session, conv, "TKT-TEST01", "follow up")
    assert state["conversation_id"] == conv.id
    assert state["ticket_text"] == "follow up"
    assert state["previous_turn_requested_info"] is False
    assert {"role": "user", "content": "hello"} in state["conversation_history"]
    assert {"role": "assistant", "content": "hi there"} in state["conversation_history"]
