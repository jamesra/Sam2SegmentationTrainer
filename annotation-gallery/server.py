"""Slim AnnotationCrops gallery: sqlite read, ignore/restore, no Nornir."""

from __future__ import annotations

import hmac
import json
import os
import posixpath
import re
import secrets
import ssl
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, unquote, urlencode, urlparse
from urllib.request import Request, urlopen

from catalog import (
    approve_window,
    build_repair_pack,
    export_volume_review,
    ignore_location,
    import_review_lists,
    list_catalog_rows,
    load_approved,
    load_held_reviews,
    load_ignore_ids,
    parse_review_import,
    parse_training_review,
    review_lists_are_empty,
    save_held_review,
    load_source_document,
    read_crop_size,
    save_source_settings,
    restore_location,
    unapprove_window,
)
from identity import (
    PERMISSION_READ,
    PERMISSION_REVIEW,
    OidcIdentity,
    Principal,
    StubIdentity,
)
from refresh import (
    RefreshUnavailable,
    drop_stale_thumbnails,
    nornir_buildmanager_root,
    refresh_location,
)
from thumbs import ensure_thumbnail, parse_thumb_request, start_thumbnail_watch, thumb_parts_allowed

VOLUME_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
FILE_PREFIXES = frozenset({"images", "masks", "ignored"})
STATIC_DIR = Path(__file__).resolve().parent / "static"
SESSION_COOKIE = "gallery_session"


def env_map() -> dict[str, str]:
    """Process environment as a plain str→str map."""
    return {str(key): str(value) for key, value in os.environ.items()}


_VERSION_NAME = re.compile(r"v\d+[a-z]*")


def registry_dir(environ: dict[str, str] | None = None) -> Path:
    """Folder of volume-root links. Child name is the Identity volume name."""
    mapping = environ or env_map()
    raw = mapping.get("GALLERY_VOLUME_DIR") or "/gallery-volumes"
    return Path(raw)


def list_training_sets() -> list[str]:
    """Version folders beside ``current``, plus ``current`` itself.

    When the gallery is not pointed at a ``current`` symlink, the only set is
    the configured registry, reported as ``current``.
    """
    registry = registry_dir()
    if registry.name != "current":
        return ["current"]
    root = registry.parent
    if not root.is_dir():
        return ["current"]
    names: list[str] = []
    for child in sorted(root.iterdir(), key=lambda path: path.name.lower()):
        if child.name != "current" and not _VERSION_NAME.fullmatch(child.name):
            continue
        if child.is_dir() or child.is_symlink():
            names.append(child.name)
    if "current" not in names:
        names.insert(0, "current")
    return names


def resolve_training_set(name: str) -> Path:
    """Return the volume-root folder for one training set. Rejects escape paths."""
    if name not in list_training_sets():
        raise FileNotFoundError(name)
    registry = registry_dir()
    if registry.name != "current":
        return registry
    path = registry.parent / name
    if not path.exists():
        raise FileNotFoundError(name)
    path.resolve().relative_to(registry.parent.resolve())
    return path


def identity_mode(environ: dict[str, str] | None = None) -> str:
    mapping = environ or env_map()
    mode = (mapping.get("GALLERY_IDENTITY_MODE") or "stub").strip().lower()
    if mode not in {"stub", "oidc"}:
        return "stub"
    return mode


_VIKING_OPEN = "viking://open?volumeName={volume}&location={id}"


def viking_template(environ: dict[str, str] | None = None) -> str:
    """Link that the OS hands to a running Viking. The old public site is not a launch URL."""
    mapping = environ or env_map()
    template = (mapping.get("GALLERY_VIKING_URL") or "").strip()
    if not template or "connectomes.utah.edu/connectome" in template:
        return _VIKING_OPEN
    return template


def odata_base(volume: str, environ: dict[str, str] | None = None) -> str:
    """OData service root for one Identity volume name."""
    mapping = environ or env_map()
    template = mapping.get("GALLERY_ODATA_TEMPLATE") or "https://websvc.codepharm.net/{volume}/OData"
    return template.replace("{volume}", quote(volume, safe="")).rstrip("/")


def tag_tokens(raw: str | None) -> list[str]:
    """Split a structure or type tag field. XML text nodes, otherwise semicolons."""
    if not raw:
        return []
    text = str(raw).strip()
    if "<" in text:
        found = [item.strip() for item in re.findall(r">([^<]+)<", text) if item.strip()]
        if found:
            return found
    return [item.strip() for item in text.split(";") if item.strip()]


def structure_ids_for_tag_pattern(
    structures: list[dict[str, Any]],
    types: list[dict[str, Any]],
    pattern: str,
) -> list[int]:
    """Structure IDs whose own tags, or their type's tags, match *pattern*.

    *pattern* is a regular expression. A plain word matches that tag as a substring.
    """
    compiled = re.compile(pattern, re.IGNORECASE)
    matched_types = {
        int(item["ID"])
        for item in types
        if any(
            compiled.search(token)
            for token in tag_tokens(item.get("Tags")) + tag_tokens(item.get("StructureTags"))
        )
    }
    ids: list[int] = []
    for item in structures:
        type_id = item.get("TypeID")
        own = any(compiled.search(token) for token in tag_tokens(item.get("Tags")))
        inherited = type_id is not None and int(type_id) in matched_types
        if own or inherited:
            ids.append(int(item["ID"]))
    return ids


def fetch_odata_location(base_url: str, location_id: int) -> dict[str, Any]:
    """Read one location's MosaicShape and LastModified from an OData service root."""
    query = urlencode(
        {
            "$select": "ID,ParentID,TypeCode,Z,OffEdge,MosaicShape,LastModified,Radius",
            "$expand": "Parent($select=ID,TypeID,Label;$expand=Type($select=ID,Name,ParentID))",
        },
        safe="(),$=",
    )
    url = f"{base_url.rstrip('/')}/Locations({int(location_id)})?{query}"
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict) or "ID" not in payload:
        raise ValueError("OData location response has no ID")
    return payload


def _odata_values(url: str, *, pages: int = 40) -> list[dict[str, Any]]:
    """Follow OData pages and return entity objects. Stops after *pages* pages."""
    rows: list[dict[str, Any]] = []
    next_url: str | None = url
    for _ in range(pages):
        if not next_url:
            break
        request = Request(next_url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        rows.extend(item for item in (payload.get("value") or []) if isinstance(item, dict))
        next_url = payload.get("@odata.nextLink")
    return rows


def tag_structure_ids(volume: str, pattern: str, environ: dict[str, str] | None = None) -> list[int]:
    """Ask OData for structure and type tags, then return matching structure IDs.

    Location rows stay in the local catalog. Callers intersect these IDs instead of
    downloading every location.
    """
    base = odata_base(volume, environ)
    structures = _odata_values(f"{base}/Structures?$select=ID,TypeID,Tags")
    types = _odata_values(f"{base}/StructureTypes?$select=ID,Tags,StructureTags")
    return structure_ids_for_tag_pattern(structures, types, pattern)


def session_secret(environ: dict[str, str] | None = None) -> bytes:
    mapping = environ or env_map()
    text = mapping.get("GALLERY_SESSION_SECRET") or "dev-only-not-secret"
    return text.encode("utf-8")


def list_registry_names(registry: Path) -> list[str]:
    """Identity volume names present as registry children. Rejects ``..``."""
    if not registry.is_dir():
        return []
    names: list[str] = []
    for child in sorted(registry.iterdir(), key=lambda path: path.name.lower()):
        if not VOLUME_NAME.fullmatch(child.name):
            continue
        if child.is_dir() or child.is_symlink():
            names.append(child.name)
    return names


def volume_root(registry: Path, name: str) -> Path:
    """Return the volume-root path for a registry child. Name is never a path."""
    if not VOLUME_NAME.fullmatch(name):
        raise ValueError("invalid volume name")
    path = registry / name
    if not path.exists():
        raise FileNotFoundError(name)
    return path


def crops_dir(registry: Path, name: str) -> Path:
    """``{volume-root}/AnnotationCrops`` for a registry child."""
    return volume_root(registry, name) / "AnnotationCrops"


def _allowed_crop_parts(parts: list[str]) -> bool:
    """Crop images, masks, thumbnail JPEGs, and the split-location parts layout JSON."""
    if parts[0] in FILE_PREFIXES:
        return True
    if thumb_parts_allowed(parts):
        return True
    return (
        len(parts) == 3
        and parts[0] == "overlays"
        and parts[1] == "parts"
        and bool(re.fullmatch(r"[0-9]+\.json", parts[2]))
    )


def safe_crop_file(crops: Path, relative: str) -> Path:
    """Resolve a crop/mask path under AnnotationCrops. Rejects traversal."""
    stripped = unquote(relative).replace("\\", "/").lstrip("/")
    if not stripped or stripped.startswith("/"):
        raise ValueError("invalid path")
    parts = posixpath.normpath(stripped).split("/")
    if parts[0] == ".." or ".." in parts or not _allowed_crop_parts(parts):
        raise ValueError("invalid path")
    target = (crops / Path(*parts)).resolve()
    root = crops.resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("invalid path") from exc
    return target


def encode_session(principal: Principal, secret: bytes) -> str:
    payload = json.dumps(
        {
            "sub": principal.subject,
            "name": principal.display_name,
            "grants": principal.grants,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    signature = hmac.new(secret, payload, "sha256").hexdigest()
    return payload.hex() + "." + signature


def decode_session(token: str | None, secret: bytes) -> Principal | None:
    if not token or "." not in token:
        return None
    payload_hex, signature = token.rsplit(".", 1)
    try:
        payload = bytes.fromhex(payload_hex)
    except ValueError:
        return None
    expected = hmac.new(secret, payload, "sha256").hexdigest()
    if not hmac.compare_digest(expected, signature):
        return None
    try:
        data = json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError:
        return None
    grants = data.get("grants") or {}
    if not isinstance(grants, dict):
        return None
    cleaned = {
        str(name): str(role)
        for name, role in grants.items()
        if role in {PERMISSION_READ, PERMISSION_REVIEW}
    }
    return Principal(
        subject=str(data.get("sub") or "user"),
        display_name=str(data.get("name") or "user"),
        grants=cleaned,
    )


def content_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".js":
        return "text/javascript; charset=utf-8"
    if suffix == ".css":
        return "text/css; charset=utf-8"
    if suffix == ".html":
        return "text/html; charset=utf-8"
    if suffix == ".json":
        return "application/json; charset=utf-8"
    return "application/octet-stream"


class GalleryHandler(BaseHTTPRequestHandler):
    """HTTP API + static SPA. Does not spawn nornir-build."""

    server_version = "AnnotationGallery/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

    def _registry(self) -> Path:
        """Training set for this request. Omitted ``set`` keeps the configured registry."""
        parsed = urlparse(self.path)
        name = (parse_qs(parsed.query).get("set") or [""])[0].strip()
        if not name:
            return registry_dir()
        return resolve_training_set(name)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)
        if path == "/health":
            self._send_json({"ok": True})
            return
        if path in {"/", "/index.html"}:
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return
        if path.startswith("/static/"):
            relative = path[len("/static/") :]
            self._send_static(relative)
            return
        if path == "/api/config":
            self._send_json(
                {
                    "identityMode": identity_mode(),
                    "vikingUrl": viking_template(),
                }
            )
            return
        if path == "/login":
            self._handle_login()
            return
        if path == "/callback":
            self._handle_callback(query)
            return
        if path == "/logout":
            self._clear_session()
            return
        if path == "/api/training-sets":
            self._send_json({"sets": list_training_sets(), "default": "current"})
            return
        if path == "/api/review-lists":
            self._handle_training_review_export()
            return
        if path == "/api/me":
            principal = self._principal()
            self._send_json(
                {
                    "subject": principal.subject,
                    "displayName": principal.display_name,
                    "volumes": [
                        {"name": name, "permission": role}
                        for name, role in sorted(principal.grants.items())
                    ],
                }
            )
            return
        match = re.fullmatch(r"/api/volumes/([^/]+)/catalog", path)
        if match:
            self._handle_catalog(match.group(1))
            return
        match = re.fullmatch(r"/api/volumes/([^/]+)/ignore-list", path)
        if match:
            self._handle_ignore_export(match.group(1))
            return
        match = re.fullmatch(r"/api/volumes/([^/]+)/settings", path)
        if match:
            self._handle_settings_get(match.group(1))
            return
        match = re.fullmatch(r"/api/volumes/([^/]+)/repair-pack", path)
        if match:
            self._handle_repair_pack(match.group(1))
            return
        match = re.fullmatch(r"/api/volumes/([^/]+)/tags", path)
        if match:
            self._handle_tags(match.group(1), (query.get("q") or [""])[0])
            return
        match = re.fullmatch(r"/api/volumes/([^/]+)/file", path)
        if match:
            rel = (query.get("path") or [""])[0]
            self._handle_file(match.group(1), rel)
            return
        self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/api/review-lists":
            self._handle_training_review_import()
            return
        match = re.fullmatch(r"/api/volumes/([^/]+)/ignore-list", path)
        if match:
            self._handle_ignore_import(match.group(1))
            return
        match = re.fullmatch(r"/api/volumes/([^/]+)/settings", path)
        if match:
            self._handle_settings_post(match.group(1))
            return
        match = re.fullmatch(r"/api/volumes/([^/]+)/refresh-location", path)
        if match:
            self._handle_refresh_location(match.group(1))
            return
        match = re.fullmatch(r"/api/volumes/([^/]+)/(ignore|restore|approve|unapprove)", path)
        if not match:
            self._send_json({"error": "not found"}, status=404)
            return
        volume = match.group(1)
        action = match.group(2)
        principal = self._principal()
        role = principal.grants.get(volume)
        if role is None:
            self._send_json({"error": "not found"}, status=404)
            return
        if role != PERMISSION_REVIEW:
            self._send_json({"error": "review required"}, status=403)
            return
        try:
            body = self._read_json()
            location_id = int(body.get("location_id"))
            image_key = body.get("image_key") or None
            if image_key is not None:
                image_key = str(image_key)
        except (TypeError, ValueError, json.JSONDecodeError):
            self._send_json({"error": "location_id required"}, status=400)
            return
        try:
            crops = crops_dir(self._registry(), volume)
        except (ValueError, FileNotFoundError):
            self._send_json({"error": "not found"}, status=404)
            return
        if action == "ignore":
            ignore_location(crops, location_id, image_key)
        elif action == "restore":
            restore_location(crops, location_id, image_key)
        elif action == "approve":
            approve_window(crops, location_id, image_key)
        else:
            unapprove_window(crops, location_id, image_key)
        self._send_json({
            "ok": True,
            "location_id": location_id,
            "ignored": action == "ignore",
            "approved": action == "approve",
        })

    def _handle_login(self) -> None:
        if identity_mode() != "oidc":
            self._redirect("/")
            return
        oidc = OidcIdentity(env_map())
        state = secrets.token_urlsafe(16)
        redirect_uri = os.environ.get("GALLERY_OIDC_REDIRECT_URI") or self._callback_uri()
        url = oidc.login_url(redirect_uri, state)
        self.send_response(302)
        self.send_header("Location", url)
        self.send_header("Set-Cookie", f"gallery_oidc_state={state}; Path=/; HttpOnly; SameSite=Lax")
        self.end_headers()

    def _handle_callback(self, query: dict[str, list[str]]) -> None:
        if identity_mode() != "oidc":
            self._redirect("/")
            return
        code = (query.get("code") or [""])[0]
        if not code:
            self._send_json({"error": "missing code"}, status=400)
            return
        oidc = OidcIdentity(env_map())
        redirect_uri = os.environ.get("GALLERY_OIDC_REDIRECT_URI") or self._callback_uri()
        try:
            principal = oidc.complete_login(code, redirect_uri, list_registry_names(registry_dir()))
        except PermissionError as exc:
            self._send_json({"error": str(exc)}, status=401)
            return
        cookie = encode_session(principal, session_secret())
        self.send_response(302)
        self.send_header("Location", "/")
        self.send_header(
            "Set-Cookie",
            f"{SESSION_COOKIE}={cookie}; Path=/; HttpOnly; SameSite=Lax",
        )
        self.end_headers()

    def _handle_ignore_export(self, volume: str) -> None:
        principal = self._principal()
        if volume not in principal.grants:
            self._send_json({"error": "not found"}, status=404)
            return
        try:
            crops = crops_dir(self._registry(), volume)
        except (ValueError, FileNotFoundError):
            self._send_json({"error": "not found"}, status=404)
            return
        ids = sorted(load_ignore_ids(crops))
        approved = [
            {"location_id": location_id, "image_key": image_key}
            for location_id, image_key in sorted(load_approved(crops))
        ]
        self._send_json({
            "volume": volume,
            "rejected": ids,
            "locationIds": ids,
            "approved": approved,
        })

    def _handle_repair_pack(self, volume: str) -> None:
        principal = self._principal()
        if volume not in principal.grants:
            self._send_json({"error": "not found"}, status=404)
            return
        try:
            crops = crops_dir(self._registry(), volume)
        except (ValueError, FileNotFoundError):
            self._send_json({"error": "not found"}, status=404)
            return
        payload = build_repair_pack(crops, volume=volume, viking_template=viking_template())
        if not payload:
            self._send_json({"error": "no ignored locations"}, status=404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Content-Disposition", f'attachment; filename="{volume}-repair.zip"')
        self.end_headers()
        self.wfile.write(payload)

    def _handle_ignore_import(self, volume: str) -> None:
        principal = self._principal()
        role = principal.grants.get(volume)
        if role is None:
            self._send_json({"error": "not found"}, status=404)
            return
        if role != PERMISSION_REVIEW:
            self._send_json({"error": "review required"}, status=403)
            return
        try:
            rejected, approved, replace = parse_review_import(self._read_json())
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc) or "invalid review list"}, status=400)
            return
        try:
            crops = crops_dir(self._registry(), volume)
        except (ValueError, FileNotFoundError):
            self._send_json({"error": "not found"}, status=404)
            return
        result = import_review_lists(crops, rejected, approved, replace=replace)
        self._send_json({"ok": True, "volume": volume, **result})

    def _handle_training_review_export(self) -> None:
        principal = self._principal()
        try:
            registry = self._registry()
        except FileNotFoundError:
            self._send_json({"error": "not found"}, status=404)
            return
        volumes = []
        for name in list_registry_names(registry):
            if name not in principal.grants:
                continue
            crops = registry / name / "AnnotationCrops"
            if not crops.is_dir():
                continue
            entry = export_volume_review(crops, name)
            entry["present"] = True
            volumes.append(entry)
        present = {entry["volume"] for entry in volumes}
        for name, document in load_held_reviews(registry).items():
            if name in present or not VOLUME_NAME.fullmatch(name):
                continue
            rejected, approved, _replace = parse_review_import(document)
            volumes.append({
                "volume": name,
                "rejected": rejected,
                "locationIds": rejected,
                "approved": [
                    {"location_id": location_id, "image_key": image_key}
                    for location_id, image_key in approved
                ],
                "present": False,
            })
        volumes.sort(key=lambda entry: entry["volume"].lower())
        self._send_json({"trainingSet": registry.name, "volumes": volumes})

    def _handle_training_review_import(self) -> None:
        principal = self._principal()
        if PERMISSION_REVIEW not in principal.grants.values():
            self._send_json({"error": "review required"}, status=403)
            return
        try:
            registry = self._registry()
            entries = parse_training_review(self._read_json())
        except FileNotFoundError:
            self._send_json({"error": "not found"}, status=404)
            return
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc) or "invalid review list"}, status=400)
            return
        present_names = set(list_registry_names(registry))
        applied = []
        held = []
        for volume, rejected, approved in entries:
            if not VOLUME_NAME.fullmatch(volume):
                continue
            if volume not in present_names:
                save_held_review(registry, volume, rejected, approved)
                held.append(volume)
                continue
            if principal.grants.get(volume) != PERMISSION_REVIEW:
                continue
            crops = registry / volume / "AnnotationCrops"
            if not crops.is_dir():
                save_held_review(registry, volume, rejected, approved)
                held.append(volume)
                continue
            replace = review_lists_are_empty(crops)
            result = import_review_lists(crops, rejected, approved, replace=replace)
            applied.append({"volume": volume, "replaced": replace, **result})
        self._send_json({"ok": True, "applied": applied, "held": held})

    def _settings_payload(self, volume: str, source: dict[str, Any]) -> dict[str, Any]:
        url = source.get("odata")
        text = url.strip() if isinstance(url, str) else ""
        exporter = str(source.get("exporter") or "tiled").strip().lower()
        if exporter not in {"tiled", "legacy"}:
            exporter = "tiled"
        path = source.get("path")
        return {
            "volume": volume,
            "kind": source.get("kind"),
            "odata": text,
            "path": str(path) if path else "",
            "exporter": exporter,
            "placeholder": odata_base(volume),
        }

    def _handle_settings_get(self, volume: str) -> None:
        principal = self._principal()
        if volume not in principal.grants:
            self._send_json({"error": "not found"}, status=404)
            return
        try:
            crops = crops_dir(self._registry(), volume)
        except (ValueError, FileNotFoundError):
            self._send_json({"error": "not found"}, status=404)
            return
        self._send_json(self._settings_payload(volume, load_source_document(crops)))

    def _handle_settings_post(self, volume: str) -> None:
        principal = self._principal()
        role = principal.grants.get(volume)
        if role is None:
            self._send_json({"error": "not found"}, status=404)
            return
        if role != PERMISSION_REVIEW:
            self._send_json({"error": "review required"}, status=403)
            return
        try:
            body = self._read_json()
            saved = save_source_settings(
                crops_dir(self._registry(), volume),
                odata=str(body.get("odata") or ""),
                exporter=str(body.get("exporter") or "tiled"),
            )
        except (TypeError, ValueError, json.JSONDecodeError, FileNotFoundError) as exc:
            status = 404 if isinstance(exc, FileNotFoundError) else 400
            self._send_json({"error": str(exc) or "invalid settings"}, status=status)
            return
        self._send_json(self._settings_payload(volume, saved))

    def _handle_refresh_location(self, volume: str) -> None:
        principal = self._principal()
        role = principal.grants.get(volume)
        if role is None:
            self._send_json({"error": "not found"}, status=404)
            return
        if role != PERMISSION_REVIEW:
            self._send_json({"error": "review required"}, status=403)
            return
        try:
            body = self._read_json()
            location_id = int(body.get("location_id"))
        except (TypeError, ValueError, json.JSONDecodeError):
            self._send_json({"error": "location_id required"}, status=400)
            return
        try:
            crops = crops_dir(self._registry(), volume)
        except (ValueError, FileNotFoundError):
            self._send_json({"error": "not found"}, status=404)
            return
        source = load_source_document(crops)
        url = source.get("odata")
        if not isinstance(url, str) or not url.strip():
            self._send_json({"error": "Set an OData endpoint for this volume before refreshing"}, status=400)
            return
        if nornir_buildmanager_root() is None:
            self._send_json({
                "error": "Mask refresh needs the nornir-buildmanager mount. No masks were changed.",
            }, status=503)
            return
        exporter = str(source.get("exporter") or "tiled")
        try:
            entity = fetch_odata_location(url.strip(), location_id)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError) as exc:
            self._send_json({"error": f"OData lookup failed: {exc}"}, status=502)
            return
        try:
            keys = refresh_location(crops, entity, exporter)
        except RefreshUnavailable as exc:
            self._send_json({"error": str(exc)}, status=503)
            return
        except (TypeError, ValueError) as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        drop_stale_thumbnails(crops, keys)
        self._send_json({"ok": True, "location_id": location_id, "imageKeys": keys})

    def _handle_catalog(self, volume: str) -> None:
        principal = self._principal()
        if volume not in principal.grants:
            self._send_json({"error": "not found"}, status=404)
            return
        try:
            crops = crops_dir(self._registry(), volume)
        except (ValueError, FileNotFoundError):
            self._send_json({"error": "not found"}, status=404)
            return
        rows = list_catalog_rows(crops)
        self._send_json({
            "volume": volume,
            "permission": principal.grants[volume],
            "cropSize": read_crop_size(crops) or 1024,
            "rows": rows,
        })

    def _handle_tags(self, volume: str, pattern: str) -> None:
        principal = self._principal()
        if volume not in principal.grants:
            self._send_json({"error": "not found"}, status=404)
            return
        text = pattern.strip()
        if not text:
            self._send_json({"structureIds": []})
            return
        try:
            re.compile(text)
        except re.error as exc:
            self._send_json({"error": f"invalid tag pattern: {exc}"}, status=400)
            return
        try:
            ids = tag_structure_ids(volume, text)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            self._send_json({"error": f"tag lookup failed: {exc}"}, status=502)
            return
        self._send_json({"structureIds": ids})

    def _handle_file(self, volume: str, relative: str) -> None:
        principal = self._principal()
        if volume not in principal.grants:
            self._send_json({"error": "not found"}, status=404)
            return
        try:
            crops = crops_dir(self._registry(), volume)
            target = safe_crop_file(crops, relative)
        except (ValueError, FileNotFoundError):
            self._send_json({"error": "invalid path"}, status=400)
            return
        except OSError:
            self._send_json({"error": "invalid path"}, status=400)
            return
        thumb = parse_thumb_request(relative)
        if thumb is not None:
            size, image_key = thumb
            try:
                ensured = ensure_thumbnail(crops, size, image_key)
            except (OSError, ValueError):
                self._send_json({"error": "not found"}, status=404)
                return
            if ensured is None or not ensured.is_file():
                self._send_json({"error": "not found"}, status=404)
                return
            self._send_file(ensured, content_type_for(ensured))
            return
        if not target.is_file():
            self._send_json({"error": "not found"}, status=404)
            return
        self._send_file(target, content_type_for(target))

    def _principal(self) -> Principal:
        try:
            names = list_registry_names(self._registry())
        except (ValueError, FileNotFoundError):
            names = []
        mode = identity_mode()
        header = self.headers.get("Authorization") or ""
        if header.lower().startswith("bearer "):
            token = header[7:].strip()
            if mode == "stub":
                stub = StubIdentity(env_map())
                principal = stub.principal_from_bearer(token, names)
                if principal is not None:
                    return principal
        cookie = self._cookie(SESSION_COOKIE)
        principal = decode_session(cookie, session_secret())
        if principal is not None:
            grants = intersect_keep(principal.grants, names)
            return Principal(principal.subject, principal.display_name, grants)
        if mode == "stub":
            return StubIdentity(env_map()).principal_for_registry(names)
        return Principal(subject="anonymous", display_name="anonymous", grants={})

    def _cookie(self, name: str) -> str | None:
        header = self.headers.get("Cookie") or ""
        for part in header.split(";"):
            piece = part.strip()
            if piece.startswith(name + "="):
                return piece[len(name) + 1 :]
        return None

    def _callback_uri(self) -> str:
        host = self.headers.get("Host") or "127.0.0.1:8443"
        return f"http://{host}/callback"

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        payload = json.loads(raw.decode("utf-8") or "{}")
        if not isinstance(payload, dict):
            raise json.JSONDecodeError("object required", "", 0)
        return payload

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self._send_json({"error": "not found"}, status=404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_static(self, relative: str) -> None:
        parts = posixpath.normpath(relative).split("/")
        if ".." in parts:
            self._send_json({"error": "invalid path"}, status=400)
            return
        path = (STATIC_DIR / Path(*parts)).resolve()
        try:
            path.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self._send_json({"error": "invalid path"}, status=400)
            return
        self._send_file(path, content_type_for(path))

    def _redirect(self, location: str) -> None:
        self.send_response(302)
        self.send_header("Location", location)
        self.end_headers()

    def _clear_session(self) -> None:
        self.send_response(302)
        self.send_header("Location", "/")
        self.send_header("Set-Cookie", f"{SESSION_COOKIE}=; Path=/; Max-Age=0")
        self.end_headers()


def intersect_keep(grants: dict[str, str], names: list[str]) -> dict[str, str]:
    allowed = set(names)
    return {name: role for name, role in grants.items() if name in allowed}


def make_server(
    host: str = "0.0.0.0",
    port: int = 80,
) -> ThreadingHTTPServer:
    """Bind one gallery HTTP server. Callers wrap the socket for TLS."""
    return ThreadingHTTPServer((host, port), GalleryHandler)


def _https_server(host: str, port: int) -> ThreadingHTTPServer | None:
    """Bind HTTPS when both PEM paths exist. Missing certs leave HTTPS off."""
    cert = os.environ.get("SSL_CERT_PATH") or ""
    key = os.environ.get("SSL_KEY_PATH") or ""
    if not cert or not key or not Path(cert).is_file() or not Path(key).is_file():
        print(
            "annotation-gallery HTTPS disabled: SSL_CERT_PATH or SSL_KEY_PATH missing",
            flush=True,
        )
        return None
    server = make_server(host, port)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    try:
        context.load_cert_chain(cert, key)
    except (OSError, ssl.SSLError) as exc:
        print(f"annotation-gallery HTTPS disabled: {exc}", flush=True)
        server.server_close()
        return None
    server.socket = context.wrap_socket(server.socket, server_side=True)
    return server


def main() -> None:
    host = os.environ.get("GALLERY_HOST") or "0.0.0.0"
    http_port = int(os.environ.get("GALLERY_HTTP_PORT") or os.environ.get("GALLERY_PORT") or "80")
    https_port = int(os.environ.get("GALLERY_HTTPS_PORT") or "443")
    http_server = make_server(host, http_port)
    https_server = _https_server(host, https_port)
    start_thumbnail_watch(registry_dir())
    print(f"annotation-gallery HTTP on {host}:{http_port}", flush=True)
    if https_server is not None:
        print(f"annotation-gallery HTTPS on {host}:{https_port}", flush=True)
        threading.Thread(target=https_server.serve_forever, daemon=True).start()
    try:
        http_server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        http_server.server_close()
        if https_server is not None:
            https_server.shutdown()
            https_server.server_close()


if __name__ == "__main__":
    main()
