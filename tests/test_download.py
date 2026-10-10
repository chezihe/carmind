import httpx
import pytest

from carmind.data.download import Source, download, load_sources


def make_client(content_type: str) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": content_type}, content=b"%PDF-1.7 fake")

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_load_sources(tmp_path):
    p = tmp_path / "sources.yaml"
    p.write_text("sources:\n  - id: a\n    url: https://example.com/a.pdf\n")
    assert load_sources(p) == [Source(id="a", url="https://example.com/a.pdf")]


def test_download_writes_pdf_and_skips_existing(tmp_path):
    src = Source(id="a", url="https://example.com/a.pdf")
    path = download(src, tmp_path, make_client("application/pdf"))
    assert path.read_bytes().startswith(b"%PDF")
    # 두 번째 호출은 네트워크 없이 기존 파일 반환
    assert download(src, tmp_path, make_client("text/html")) == path


def test_download_rejects_non_pdf(tmp_path):
    with pytest.raises(ValueError):
        download(Source(id="b", url="https://example.com/b"), tmp_path, make_client("text/html"))
    assert not (tmp_path / "b.pdf").exists()
