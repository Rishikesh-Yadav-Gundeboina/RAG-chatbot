from backend.services import log_service


def _use_temp_db(monkeypatch, tmp_path):
    monkeypatch.setattr(log_service, "DB_PATH", tmp_path / "test.db")


def test_get_recent_messages_returns_chronological_order(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)
    session_id = log_service.create_session()
    log_service.add_message(session_id, "user", "q1")
    log_service.add_message(session_id, "assistant", "a1")
    log_service.add_message(session_id, "user", "q2")
    log_service.add_message(session_id, "assistant", "a2")

    history = log_service.get_recent_messages(session_id)

    assert [m["content"] for m in history] == ["q1", "a1", "q2", "a2"]
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]


def test_get_recent_messages_respects_limit(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)
    session_id = log_service.create_session()
    for i in range(6):
        log_service.add_message(session_id, "user" if i % 2 == 0 else "assistant", f"m{i}")

    history = log_service.get_recent_messages(session_id, limit=4)

    assert [m["content"] for m in history] == ["m2", "m3", "m4", "m5"]


def test_get_recent_messages_empty_for_new_session(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)
    session_id = log_service.create_session()

    assert log_service.get_recent_messages(session_id) == []


def test_get_recent_messages_isolated_per_session(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path)
    session_a = log_service.create_session()
    session_b = log_service.create_session()
    log_service.add_message(session_a, "user", "a-question")
    log_service.add_message(session_b, "user", "b-question")

    history_a = log_service.get_recent_messages(session_a)
    history_b = log_service.get_recent_messages(session_b)

    assert [m["content"] for m in history_a] == ["a-question"]
    assert [m["content"] for m in history_b] == ["b-question"]
