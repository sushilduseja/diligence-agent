import json

import httpx

from dd_agent.nodes.github_search import github_search_node

OK = type("R", (), {"status_code": 200, "text": json.dumps({"items": [{"html_url": "https://github.com/x/1", "title": "t", "body": "b"}]})})()


class FlakyClient:
    def __init__(self, failures):
        self.failures = failures
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        if self.failures > 0:
            self.failures -= 1
            raise httpx.TimeoutException("boom")
        return OK


def test_fails_twice_then_succeeds(mocker):
    mocker.patch("dd_agent.nodes.http_retry.time.sleep")
    client = FlakyClient(failures=2)
    result = github_search_node("stripe connect", client)
    assert len(result) == 1
    assert client.calls == 3
