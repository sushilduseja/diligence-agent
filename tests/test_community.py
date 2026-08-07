import json

import httpx

from dd_agent.nodes.community import community_node


class FakeResponse:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self._text = text

    @property
    def text(self):
        return self._text


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _stack_json(n):
    items = [
        {
            "link": f"https://stackoverflow.com/q/{i}",
            "title": f"question {i}",
        }
        for i in range(n)
    ]
    return json.dumps({"items": items})


def test_200_with_items():
    client = FakeClient([FakeResponse(200, _stack_json(2))])
    result = community_node("stripe connect", client)
    assert len(result) == 2
    assert all(e.source_type == "community" for e in result)


def test_capped_at_five():
    client = FakeClient([FakeResponse(200, _stack_json(20))])
    result = community_node("stripe connect", client)
    assert len(result) == 5


def test_403_returns_empty(mocker):
    client = FakeClient([FakeResponse(403, "denied")])
    mocker.patch("dd_agent.nodes.community.logger.warning")
    assert community_node("stripe connect", client) == []


def test_timeout_returns_empty(mocker):
    mocker.patch("dd_agent.nodes.http_retry.time.sleep")

    def _raise_get(*args, **kwargs):
        raise httpx.TimeoutException("timed out")

    client = FakeClient([])
    client.get = _raise_get
    mocker.patch("dd_agent.nodes.community.logger.warning")
    assert community_node("stripe connect", client) == []


def test_malformed_json_returns_empty(mocker):
    client = FakeClient([FakeResponse(200, "not json")])
    mocker.patch("dd_agent.nodes.community.logger.warning")
    assert community_node("stripe connect", client) == []
