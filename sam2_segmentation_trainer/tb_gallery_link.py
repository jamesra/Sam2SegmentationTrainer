"""TensorBoard server with a Gallery link in the header.

TensorBoard 2.21 draws its header inside ``<tb-webapp>`` and has no flag for
an extra link. This process is the normal ``tensorboard`` server with one
WSGI wrapper: the index document gets a script that places
``https://gallery.codepharm.net`` in that header once the toolbar exists.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable

GALLERY_URL = "https://gallery.codepharm.net"


def gallery_link_script(gallery_url: str = GALLERY_URL) -> bytes:
    """Script that inserts the gallery anchor into TensorBoard's toolbar.

    Served as its own file. TensorBoard's Content-Security-Policy allows
    same-origin scripts and blocks an inline ``<script>``.
    """
    href = json.dumps(gallery_url)
    return f"""(function () {{
  var href = {href};
  function place() {{
    var toolbar = document.querySelector("app-header mat-toolbar");
    if (!toolbar || toolbar.querySelector("a.gallery-link")) {{
      return Boolean(toolbar && toolbar.querySelector("a.gallery-link"));
    }}
    var link = document.createElement("a");
    link.className = "gallery-link";
    link.href = href;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = "Gallery";
    link.style.cssText = "color:#fff;flex:0 0 auto;font-size:14px;margin:0 12px;text-decoration:none;";
    var help = toolbar.querySelector("a.readme");
    if (help) {{
      toolbar.insertBefore(link, help);
    }} else {{
      toolbar.appendChild(link);
    }}
    return true;
  }}
  var observer = new MutationObserver(function () {{ place(); }});
  observer.observe(document.documentElement, {{childList: true, subtree: true}});
  place();
}})();
""".encode("utf-8")


def inject_gallery_link(html: bytes, gallery_url: str = GALLERY_URL) -> bytes:
    """Add the gallery script to TensorBoard's index document.

    Other HTML responses are returned unchanged. A document that already
    contains the script is left as-is.
    """
    if b"<tb-webapp>" not in html or b"gallery-link.js" in html:
        return html
    # gallery_url is compiled into gallery-link.js; the index only references it.
    del gallery_url
    tag = b'<script src="gallery-link.js"></script>'
    marker = b"</body>"
    if marker not in html:
        return html + tag
    return html.replace(marker, tag + marker, 1)


class GalleryLinkMiddleware:
    """Inject the gallery link into TensorBoard's HTML shell only."""

    def __init__(self, app):
        self._app = app

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO") or ""
        if path.endswith("gallery-link.js"):
            body = gallery_link_script()
            start_response(
                "200 OK",
                [
                    ("Content-Type", "application/javascript"),
                    ("Content-Length", str(len(body))),
                    ("Cache-Control", "no-cache"),
                    ("X-Content-Type-Options", "nosniff"),
                ],
            )
            return [body]
        captured: dict = {}

        def _start(status, headers, exc_info=None):
            captured["status"] = status
            captured["headers"] = list(headers)
            captured["exc_info"] = exc_info

        iterable = self._app(environ, _start)
        headers = captured.get("headers") or []
        if not _is_html(headers):
            start_response(captured["status"], headers, captured.get("exc_info"))
            return iterable
        try:
            body = b"".join(iterable)
        finally:
            close = getattr(iterable, "close", None)
            if close is not None:
                close()
        rewritten = inject_gallery_link(body)
        if rewritten is not body:
            headers = _replace_length(headers, len(rewritten))
        start_response(captured["status"], headers, captured.get("exc_info"))
        return [rewritten]


def main(argv: list[str] | None = None) -> int:
    """Start TensorBoard with :class:`GalleryLinkMiddleware` installed."""
    from tensorboard.program import TensorBoard, create_port_scanning_werkzeug_server

    def server_class(app, flags):
        return create_port_scanning_werkzeug_server(GalleryLinkMiddleware(app), flags)

    board = TensorBoard(server_class=server_class)
    board.configure(sys.argv if argv is None else argv)
    return board.main()


def _is_html(headers: Iterable[tuple[str, str]]) -> bool:
    html = False
    for key, value in headers:
        if key.lower() == "content-type" and "text/html" in value.lower():
            html = True
        if key.lower() == "content-encoding" and value.lower() not in ("", "identity"):
            return False
    return html


def _replace_length(headers: list[tuple[str, str]], length: int) -> list[tuple[str, str]]:
    updated = [(key, value) for key, value in headers if key.lower() != "content-length"]
    updated.append(("Content-Length", str(length)))
    return updated


if __name__ == "__main__":
    raise SystemExit(main())
