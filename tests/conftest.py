import pytest


@pytest.fixture(autouse=True)
def _own_codex_homes(tmp_path, monkeypatch):
    """Tests never touch the real Codex homes: the agent's own home and the login both point into tmp_path."""
    monkeypatch.setenv("PRUNEAGENT_CODEX_HOME", str(tmp_path / "codex-agent-home"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-login-home"))
