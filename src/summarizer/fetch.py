import requests
from bs4 import BeautifulSoup

MAX_CHARS = 12_000
REQUEST_TIMEOUT = 15


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return " ".join(soup.get_text(" ").split())


def fetch_text(source: str) -> str:
    if source.startswith(("http://", "https://")):
        response = requests.get(source, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        text = _html_to_text(response.text)
    else:
        text = source.strip()
    return text[:MAX_CHARS]
