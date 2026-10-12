"""Gallery link injected into TensorBoard's index HTML only."""

from __future__ import annotations

import unittest

from sam2_segmentation_trainer.tb_gallery_link import (
    GALLERY_URL,
    GalleryLinkMiddleware,
    inject_gallery_link,
)


class GalleryLinkTests(unittest.TestCase):
    def test_index_references_script(self) -> None:
        html = b"<!doctype html><tb-webapp></tb-webapp></body>"
        rewritten = inject_gallery_link(html)
        self.assertIn(b'<script src="gallery-link.js"></script></body>', rewritten)
        self.assertNotIn(b"<script>(function", rewritten)

    def test_other_html_is_unchanged(self) -> None:
        html = b"<html><body><p>not tensorboard</p></body></html>"
        self.assertEqual(inject_gallery_link(html), html)

    def test_second_inject_is_stable(self) -> None:
        html = b"<!doctype html><tb-webapp></tb-webapp></body>"
        once = inject_gallery_link(html)
        self.assertEqual(inject_gallery_link(once), once)

    def test_middleware_updates_length_and_skips_json(self) -> None:
        index = b"<tb-webapp></tb-webapp></body>"

        def app(environ, start_response):
            path = environ.get("PATH_INFO", "")
            if path == "/data":
                body = b"{}"
                start_response(
                    "200 OK",
                    [("Content-Type", "application/json"), ("Content-Length", str(len(body)))],
                )
                return [body]
            start_response(
                "200 OK",
                [("Content-Type", "text/html"), ("Content-Length", str(len(index)))],
            )
            return [index]

        middleware = GalleryLinkMiddleware(app)
        html_status: dict = {}

        def start_html(status, headers, exc_info=None):
            html_status["status"] = status
            html_status["headers"] = headers

        html_body = b"".join(middleware({"PATH_INFO": "/"}, start_html))
        self.assertEqual(html_status["status"], "200 OK")
        self.assertIn(b"gallery-link.js", html_body)
        length = dict(html_status["headers"])["Content-Length"]
        self.assertEqual(int(length), len(html_body))

        script_status: dict = {}

        def start_script(status, headers, exc_info=None):
            script_status["status"] = status
            script_status["headers"] = headers

        script_body = b"".join(middleware({"PATH_INFO": "/gallery-link.js"}, start_script))
        self.assertEqual(script_status["status"], "200 OK")
        self.assertIn(GALLERY_URL.encode("utf-8"), script_body)
        self.assertEqual(
            dict(script_status["headers"])["Content-Type"],
            "application/javascript",
        )

        json_status: dict = {}

        def start_json(status, headers, exc_info=None):
            json_status["headers"] = headers

        json_body = b"".join(middleware({"PATH_INFO": "/data"}, start_json))
        self.assertEqual(json_body, b"{}")
        self.assertEqual(dict(json_status["headers"])["Content-Length"], "2")
