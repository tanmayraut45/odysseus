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


def test_encode_raises_clear_error_on_missing_embedding_field():
    client = EmbeddingClient(url="http://embed.test/v1/embeddings", model="all-minilm")
    client._client = _FakeEmbeddingHttpClient(
        lambda _: (200, {"data": [{"index": 0}]})  # missing 'embedding' key
    )

    with pytest.raises(ValueError) as excinfo:
        client.encode(["hello"])

    msg = str(excinfo.value)
    assert "embedding" in msg
    assert "http://embed.test/v1/embeddings" in msg
    assert "all-minilm" in msg


def test_encode_valid_response_returns_vectors():
    client = EmbeddingClient(url="http://embed.test/v1/embeddings", model="all-minilm")
    client._client = _FakeEmbeddingHttpClient(
        lambda _: (200, {"data": [{"embedding": [3.0, 4.0], "index": 0}]})
    )

    out = client.encode(["hello"], normalize_embeddings=False)

    assert out.shape == (1, 2)
    assert out.tolist() == [[3.0, 4.0]]
