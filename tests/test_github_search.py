import json

import httpx

from dd_agent.nodes.github_search import github_search_node


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
        return self.responses.pop(0)


def _issue_json(n):
    items = [
        {
            "html_url": f"https://github.com/x/repo/issues/{i}",
            "title": f"issue {i}",
            "body": f"body {i}",
        }
        for i in range(n)
    ]
    return json.dumps({"items": items})


def test_200_with_two_issues():
    client = FakeClient([FakeResponse(200, _issue_json(2))])
    result = github_search_node("stripe connect", client)
    assert len(result) == 2
    assert all(e.source_type == "github" for e in result)
    assert result[0].url == "https://github.com/x/repo/issues/0"


def test_403_rate_limit_returns_empty(mocker):
    client = FakeClient([FakeResponse(403, "rate limited")])
    warn = mocker.patch("dd_agent.nodes.retrieval.logger.warning")
    result = github_search_node("stripe connect", client)
    assert result == []
    warn.assert_called_once()


def test_timeout_returns_empty(mocker):
    mocker.patch("dd_agent.nodes.http_retry.time.sleep")

    def _raise_get(*args, **kwargs):
        raise httpx.TimeoutException("timed out")

    client = FakeClient([])
    client.get = _raise_get
    mocker.patch("dd_agent.nodes.retrieval.logger.warning")
    result = github_search_node("stripe connect", client)
    assert result == []


def test_malformed_json_returns_empty(mocker):
    client = FakeClient([FakeResponse(200, "not json {{{")])
    mocker.patch("dd_agent.nodes.retrieval.logger.warning")
    result = github_search_node("stripe connect", client)
    assert result == []


def test_empty_query_does_not_call_client():
    client = FakeClient([])
    result = github_search_node("", client)
    assert result == []
    assert client.calls == []
