import json
import time

from ML_tradingAlgo.data import token_health as th


def _write_token(path, age_days):
    path.write_text(json.dumps({
        "creation_timestamp": int(time.time() - age_days * 86400),
        "token": {"refresh_token": "x", "access_token": "y"},
    }))
    return str(path)


def test_missing_token(tmp_path):
    msgs = []
    res = th.check_token_freshness(str(tmp_path / "nope.json"), notify=msgs.append)
    assert res["status"] == "missing"
    assert res["age_days"] is None
    assert msgs and "schwab_auth" in msgs[0]


def test_healthy_token_no_notify(tmp_path):
    path = _write_token(tmp_path / "t.json", age_days=2.0)
    msgs = []
    res = th.check_token_freshness(path, notify=msgs.append)
    assert res["status"] == "ok"
    assert 1.9 < res["age_days"] < 2.1
    assert 4.9 < res["days_remaining"] < 5.1
    assert msgs == []  # healthy never notifies


def test_warn_window_notifies(tmp_path):
    path = _write_token(tmp_path / "t.json", age_days=6.2)
    msgs = []
    res = th.check_token_freshness(path, notify=msgs.append)
    assert res["status"] == "warn"
    assert msgs and "expires in" in msgs[0]


def test_expired_notifies(tmp_path):
    path = _write_token(tmp_path / "t.json", age_days=7.5)
    msgs = []
    res = th.check_token_freshness(path, notify=msgs.append)
    assert res["status"] == "expired"
    assert msgs and "EXPIRED" in msgs[0]


def test_age_from_env(tmp_path, monkeypatch):
    path = _write_token(tmp_path / "t.json", age_days=1.0)
    monkeypatch.setenv("SCHWAB_TOKEN_PATH", path)
    assert 0.9 < th.token_age_days() < 1.1


def test_malformed_token_file(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    assert th.token_age_days(str(p)) is None
