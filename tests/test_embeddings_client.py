import httpx
import pytest

from src.embeddings import EmbeddingClient


class _FakeEmbeddingHttpClient:
    def __init__(self, handler):
        self.handler = handler
        self.headers = []

    def post(self, url, headers=None, json=None):
        self.headers.append(headers or {})
        request = httpx.Request("POST", url)
        status, body = self.handler(json)
        return httpx.Response(status, request=request, json=body)


def test_embedding_400_batch_retry_falls_back_to_single_inputs(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "8")
    calls = []

    def handler(payload):
        texts = payload["input"]
        calls.append(list(texts))
        if len(texts) > 1:
            return 400, {"error": "batch too large"}
        text = texts[0]
        return 200, {"data": [{"index": 0, "embedding": [float(len(text)), 1.0]}]}

    client = EmbeddingClient(url="http://embeddings.test/v1/embeddings", model="embed-test")
    client._client = _FakeEmbeddingHttpClient(handler)

    vecs = client.encode(["a", "bbbb"], normalize_embeddings=False)

    assert calls == [["a", "bbbb"], ["a"], ["bbbb"]]
    assert vecs.tolist() == [[1.0, 1.0], [4.0, 1.0]]


def test_embedding_400_single_input_retries_with_truncated_text(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MAX_CHARS", "200")
    lengths = []

    def handler(payload):
        text = payload["input"][0]
        lengths.append(len(text))
        if len(text) > 200:
            return 400, {"error": "context length exceeded"}
        return 200, {"data": [{"index": 0, "embedding": [2.0, 0.0]}]}

    client = EmbeddingClient(url="http://embeddings.test/v1/embeddings", model="embed-test")
    client._client = _FakeEmbeddingHttpClient(handler)

    vecs = client.encode(["x" * 250], normalize_embeddings=False)

    assert lengths == [250, 200]
    assert vecs.tolist() == [[2.0, 0.0]]


def test_embedding_non_400_errors_are_not_retried_or_swallowed():
    calls = 0

    def handler(payload):
        nonlocal calls
        calls += 1
        return 500, {"error": "server error"}

    client = EmbeddingClient(url="http://embeddings.test/v1/embeddings", model="embed-test")
    client._client = _FakeEmbeddingHttpClient(handler)

    with pytest.raises(httpx.HTTPStatusError):
        client.encode(["a"], normalize_embeddings=False)

    assert calls == 1


def test_embedding_retry_path_preserves_api_key_header():
    seen_headers = []

    def handler(payload):
        return 200, {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}

    client = EmbeddingClient(
        url="http://embeddings.test/v1/embeddings",
        model="embed-test",
        api_key="secret-key",
    )
    fake = _FakeEmbeddingHttpClient(handler)
    client._client = fake

    vecs = client.encode(["a"], normalize_embeddings=False)
    seen_headers.extend(fake.headers)

    assert vecs.tolist() == [[1.0, 0.0]]
    assert seen_headers == [{"Authorization": "Bearer secret-key"}]


def test_embedding_client_close_closes_http_client():
    import unittest.mock as mock
    client = EmbeddingClient(url="http://embed.test", model="m")
    fake_http = mock.MagicMock()
    client._client = fake_http

    client.close()

    fake_http.close.assert_called_once()


def test_embedding_client_context_manager_closes_on_exit():
    import unittest.mock as mock
    fake_http = mock.MagicMock()
    with EmbeddingClient(url="http://embed.test", model="m") as c:
        c._client = fake_http
    fake_http.close.assert_called_once()


def test_factory_closes_probe_client_when_endpoint_down(monkeypatch):
    import src.embeddings as emb_mod
    emb_mod.reset_http_embed_state()

    created = []

    class _FailingProbe:
        def __init__(self, **kwargs):
            self.url = kwargs.get("url", "http://embed.test")
            self.model = kwargs.get("model", "m")
            self._client = None
            created.append(self)
            self._closed = False

        def get_sentence_embedding_dimension(self):
            raise RuntimeError("connection refused")

        def close(self):
            self._closed = True

    def _no_fastembed(*a, **k):
        raise ImportError("fastembed not installed")

    monkeypatch.setattr(emb_mod, "EmbeddingClient", _FailingProbe)
    monkeypatch.setattr(emb_mod, "FastEmbedClient", _no_fastembed)
    monkeypatch.setattr(emb_mod, "_load_persisted_endpoint", lambda: {})

    result = emb_mod.get_embedding_client()

    assert result is None
    assert created and created[0]._closed is True
