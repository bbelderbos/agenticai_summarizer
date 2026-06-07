from unittest.mock import Mock, patch

from summarizer import fetch
from summarizer.fetch import fetch_text


def test_raw_text_passes_through():
    assert fetch_text("  just some text  ") == "just some text"


def test_html_is_stripped_to_text():
    html = (
        "<html><body><h1>Title</h1><script>x=1</script><p>Hello world</p></body></html>"
    )
    with patch.object(fetch.requests, "get") as mock_get:
        mock_get.return_value = Mock(text=html, raise_for_status=Mock())
        result = fetch_text("https://example.com")
    assert "Hello world" in result
    assert "Title" in result
    assert "x=1" not in result


def test_long_text_is_truncated():
    long_text = "a" * (fetch.MAX_CHARS + 500)
    assert len(fetch_text(long_text)) == fetch.MAX_CHARS
