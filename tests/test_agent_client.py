"""Agent client with fake backends, a fake `codex` runner and a fake API transport (no real calls)."""

import json
import subprocess
from pathlib import Path

import pytest

from pruneagent import agent_client as C

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["candidates", "rationale"],
          "properties": {"candidates": {"type": "array", "items": {"type": "string"}},
                         "rationale": {"type": "string"}}}
GOOD = '{"candidates": ["a"], "rationale": "r"}'


class FakeBackend:
    """Returns scripted replies or raises scripted errors, in order; counts calls."""

    name = "fake"
    model = "fake-model"

    def __init__(self, *script):
        self.script = list(script)
        self.calls = 0

    def identity(self):
        return {"backend": self.name, "model": self.model}

    def complete(self, prompt, schema, timeout):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return C.BackendResult(text=item, usage=C.Usage(input_tokens=100, output_tokens=10), model=self.model)


def client(tmp_path, backend, **kw):
    return C.AgentClient(backend, cache=C.Cache(tmp_path / "cache"), ledger=C.Ledger(tmp_path / "usage.jsonl"),
                         sleep=lambda s: None, **kw)


# ------------------------------------------------------------------------------------------------- client


def test_propose_parses_json_and_logs_usage(tmp_path):
    cl = client(tmp_path, FakeBackend("Sure:\n```json\n" + GOOD + "\n```"), run_id="r1")
    reply = cl.propose(C.AgentRequest("plan", SCHEMA, tag="round 1"))
    assert reply.data == {"candidates": ["a"], "rationale": "r"} and not reply.cached
    (entry,) = C.Ledger(tmp_path / "usage.jsonl").entries()
    assert entry["ok"] and entry["backend"] == "fake" and entry["total_tokens"] == 110
    assert entry["run_id"] == "r1" and entry["tag"] == "round 1" and "seconds" in entry and "time" in entry


def test_identical_request_is_served_from_cache(tmp_path):
    fake = FakeBackend(GOOD)
    client(tmp_path, fake).propose(C.AgentRequest("plan", SCHEMA))
    again = client(tmp_path, fake).propose(C.AgentRequest("plan", SCHEMA))  # new client, same cache dir
    assert fake.calls == 1 and again.cached and again.data["rationale"] == "r"
    totals = C.Ledger(tmp_path / "usage.jsonl").totals()
    assert totals["calls"] == 1 and totals["cache_hits"] == 1


def test_cache_key_changes_with_prompt_schema_and_backend():
    a, b = FakeBackend(), FakeBackend()
    b.model = "other"
    k = C.cache_key(C.AgentRequest("p", SCHEMA), a)
    assert k != C.cache_key(C.AgentRequest("q", SCHEMA), a)
    assert k != C.cache_key(C.AgentRequest("p", None), a)
    assert k != C.cache_key(C.AgentRequest("p", SCHEMA), b)
    assert k == C.cache_key(C.AgentRequest("p", SCHEMA, tag="other tag"), a)


def test_one_retry_on_transient_failure(tmp_path):
    fake = FakeBackend(C.TransientError("timed out"), GOOD)
    assert client(tmp_path, fake).propose(C.AgentRequest("p", SCHEMA)).data["candidates"] == ["a"]
    assert fake.calls == 2
    entries = C.Ledger(tmp_path / "usage.jsonl").entries()
    assert [e["ok"] for e in entries] == [False, True]


def test_gives_up_after_second_transient_failure(tmp_path):
    fake = FakeBackend(C.TransientError("x"), C.TransientError("y"), GOOD)
    with pytest.raises(C.TransientError):
        client(tmp_path, fake).propose(C.AgentRequest("p", SCHEMA))
    assert fake.calls == 2


@pytest.mark.parametrize("err", [C.UsageLimitReached("limit"), C.AgentError("not logged in")])
def test_limit_and_login_errors_are_not_retried(tmp_path, err):
    fake = FakeBackend(err, GOOD)
    with pytest.raises(type(err)):
        client(tmp_path, fake).propose(C.AgentRequest("p", SCHEMA))
    assert fake.calls == 1


def test_malformed_reply_is_rejected_and_not_cached(tmp_path):
    fake = FakeBackend('{"candidates": ["a"]}', "no json at all", GOOD)
    cl = client(tmp_path, fake)
    with pytest.raises(C.MalformedReply, match="rationale"):
        cl.propose(C.AgentRequest("p", SCHEMA))
    with pytest.raises(C.MalformedReply):
        cl.propose(C.AgentRequest("p", SCHEMA))
    assert cl.propose(C.AgentRequest("p", SCHEMA)).data["rationale"] == "r"  # third try: sent, not cached
    assert fake.calls == 3


def test_call_cap_counts_earlier_calls_of_the_same_run(tmp_path):
    client(tmp_path, FakeBackend(GOOD), run_id="r").propose(C.AgentRequest("p1", SCHEMA))
    cl = client(tmp_path, FakeBackend(GOOD, GOOD), run_id="r", max_calls=2)
    cl.propose(C.AgentRequest("p2", SCHEMA))
    with pytest.raises(C.CapReached):
        cl.propose(C.AgentRequest("p3", SCHEMA))
    cl.propose(C.AgentRequest("p1", SCHEMA))  # a cached answer is still allowed


# ------------------------------------------------------------------------------------------------ parsing


def test_extract_json_variants():
    assert C.extract_json(GOOD) == {"candidates": ["a"], "rationale": "r"}
    assert C.extract_json("text {\"x\": 1} more " + GOOD, ("candidates",))["rationale"] == "r"
    assert C.extract_json("```\n{\"a\": {\"b\": [1, 2]}}\n```") == {"a": {"b": [1, 2]}}
    with pytest.raises(C.MalformedReply, match="found an object"):
        C.extract_json('{"x": 1}', ("candidates",))
    with pytest.raises(C.MalformedReply):
        C.extract_json("{not json")


def test_tokens_used_from_plain_codex_output():
    assert C.tokens_used("codex\nok\ntokens used\n19,426\n") == 19426
    assert C.tokens_used("nothing") is None


def test_classify_failure():
    assert C.classify_failure("You've hit your usage limit. Try again at 3pm") is C.UsageLimitReached
    assert C.classify_failure("Not logged in. Run codex login") is C.AgentError
    assert C.classify_failure("stream disconnected before completion") is C.TransientError


# -------------------------------------------------------------------------------------------------- Codex

EVENTS_OK = "\n".join(json.dumps(e) for e in [
    {"type": "thread.started", "thread_id": "t"},
    {"type": "item.completed", "item": {"id": "item_0", "type": "error", "message": "Code Mode is unavailable"}},
    {"type": "turn.started"},
    {"type": "item.completed", "item": {"id": "item_1", "type": "agent_message", "text": GOOD}},
    {"type": "turn.completed", "usage": {"input_tokens": 3135, "cached_input_tokens": 0, "output_tokens": 9,
                                         "reasoning_output_tokens": 0}},
])


class FakeRunner:
    def __init__(self, stdout="", stderr="", returncode=0, exc=None):
        self.out = (stdout, stderr, returncode)
        self.exc = exc
        self.seen = []

    def __call__(self, cmd, **kw):
        self.seen.append((cmd, kw))
        if kw.get("input") is not None and "--output-schema" in cmd:
            path = cmd[cmd.index("--output-schema") + 1]
            self.schema = json.loads(Path(path).read_text())
        if self.exc:
            raise self.exc
        return subprocess.CompletedProcess(cmd, self.out[2], self.out[0], self.out[1])


def test_codex_command_is_trimmed_pinned_and_read_only(tmp_path):
    run = FakeRunner(EVENTS_OK)
    res = C.CodexBackend(run=run).complete("the prompt", SCHEMA, timeout=60)
    assert res.text == GOOD and res.usage.input_tokens == 3135 and res.usage.total_tokens == 3144
    cmd, kw = run.seen[0]
    assert cmd[:2] == ["codex", "exec"] and cmd[-1] == "-" and kw["input"] == "the prompt"
    for flag in ["--sandbox", "read-only", "--ephemeral", "--ignore-user-config", "--json", "--skip-git-repo-check"]:
        assert flag in cmd
    assert cmd[cmd.index("--model") + 1] == C.CODEX_MODEL
    assert 'model_reasoning_effort="low"' in cmd and "skills.include_instructions=false" in cmd
    assert any(c.startswith("model_instructions_file=") for c in cmd)
    assert run.schema == SCHEMA
    workdir = kw["cwd"]
    assert workdir.name == "empty" and not workdir.exists()  # ran in an empty temp dir, removed afterwards


def test_codex_untrimmed_command_matches_task_1_11():
    cmd = C.CodexBackend(trim=False).command()
    assert "--ignore-user-config" not in cmd and "--disable" not in cmd
    assert cmd[cmd.index("--model") + 1] == C.CODEX_MODEL


def test_codex_failures_are_classified(tmp_path):
    limit = json.dumps({"type": "turn.failed", "error": {"message": "You've hit your usage limit."}})
    with pytest.raises(C.UsageLimitReached):
        C.CodexBackend(run=FakeRunner(limit, returncode=1)).complete("p", None, 60)
    with pytest.raises(C.TransientError, match="timed out"):
        C.CodexBackend(run=FakeRunner(exc=subprocess.TimeoutExpired("codex", 60))).complete("p", None, 60)
    with pytest.raises(C.AgentError):
        C.CodexBackend(run=FakeRunner("", "Error: Not logged in", 1)).complete("p", None, 60)
    with pytest.raises(C.TransientError):  # exit 0 but no answer
        C.CodexBackend(run=FakeRunner(json.dumps({"type": "turn.started"}))).complete("p", None, 60)


def test_codex_plain_output_falls_back_to_tokens_used():
    out = json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": GOOD}})
    res = C.CodexBackend().parse_output(out, "tokens used\n19,426\n", 0)
    assert res.usage.input_tokens == 19426


def test_client_with_fake_codex_end_to_end(tmp_path):
    run = FakeRunner(EVENTS_OK)
    cl = client(tmp_path, C.CodexBackend(run=run))
    assert cl.propose(C.AgentRequest("p", SCHEMA)).data["rationale"] == "r"
    assert cl.propose(C.AgentRequest("p", SCHEMA)).cached
    assert len(run.seen) == 1
    (sent, _) = C.Ledger(tmp_path / "usage.jsonl").entries()
    assert sent["model"] == C.CODEX_MODEL and sent["reasoning_effort"] == "low" and sent["input_tokens"] == 3135


# ---------------------------------------------------------------------------------------------------- API


class FakeTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, url, headers, payload, timeout):
        self.requests.append((url, headers, payload))
        return self.responses.pop(0)


API_OK = (200, {"model": "m-2026", "output": [{"type": "message", "content": [{"type": "output_text", "text": GOOD}]}],
                "usage": {"input_tokens": 1000, "output_tokens": 200, "input_tokens_details": {"cached_tokens": 0},
                          "output_tokens_details": {"reasoning_tokens": 50}}})


def test_api_backend_structured_output_and_cost():
    t = FakeTransport(API_OK)
    api = C.ApiBackend(model="m", api_key="test-key", price_in_per_mtok=1.0, price_out_per_mtok=10.0, transport=t)
    res = api.complete("prompt", SCHEMA, 30)
    assert res.text == GOOD and res.model == "m-2026"
    assert res.usage.cost_usd == pytest.approx(0.003) and res.usage.reasoning_tokens == 50
    _url, headers, payload = t.requests[0]
    assert payload["text"]["format"]["schema"] == SCHEMA and payload["text"]["format"]["strict"]
    assert payload["input"] == "prompt" and payload["model"] == "m"
    assert headers["Authorization"] == "Bearer test-key"


@pytest.mark.parametrize("status,code,kind", [(429, "rate_limit_exceeded", C.TransientError),
                                              (429, "insufficient_quota", C.UsageLimitReached),
                                              (503, None, C.TransientError), (400, "bad", C.AgentError)])
def test_api_errors_are_classified(status, code, kind):
    api = C.ApiBackend(api_key="k", transport=FakeTransport((status, {"error": {"code": code, "message": "m"}})))
    with pytest.raises(kind):
        api.complete("p", None, 30)


def test_api_key_from_env_file_and_missing_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("# secrets\nOTHER=1\nexport OPENAI_API_KEY='sk-test'\n")
    assert C.read_api_key(env_file=env) == "sk-test"
    api = C.ApiBackend(env_file=tmp_path / "missing.env", transport=FakeTransport(API_OK))
    with pytest.raises(C.AgentError, match="no API key"):
        api.complete("p", None, 30)


def test_api_cost_cap_stops_calls(tmp_path):
    t = FakeTransport(API_OK, API_OK)
    api = C.ApiBackend(api_key="sk-secret-test", price_in_per_mtok=1000.0, price_out_per_mtok=1000.0, transport=t)
    cl = client(tmp_path, api, max_cost_usd=1.0)
    cl.propose(C.AgentRequest("p1", SCHEMA))  # costs $1.20
    with pytest.raises(C.CapReached, match="cost cap"):
        cl.propose(C.AgentRequest("p2", SCHEMA))
    assert len(t.requests) == 1
    assert C.Ledger(tmp_path / "usage.jsonl").entries()[0]["cost_usd"] == pytest.approx(1.2)
    assert "sk-secret-test" not in (tmp_path / "usage.jsonl").read_text()
    assert "sk-secret-test" not in "".join(p.read_text() for p in (tmp_path / "cache").rglob("*.json"))
