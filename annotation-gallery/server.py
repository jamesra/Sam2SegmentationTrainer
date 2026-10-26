"""Slim AnnotationCrops gallery: sqlite read, ignore/restore, no Nornir."""

from __future__ import annotations

import gzip
import hashlib
import hmac
import json
import os
import posixpath
import re
import secrets
import ssl
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, unquote, urlencode, urlparse
from urllib.request import Request, urlopen

from catalog import (
    annotation_db_name,
    approve_window,
    build_repair_pack,
    export_volume_review,
    ignore_location,
    import_review_lists,
    list_catalog_rows,
    location_image_keys,
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
    review_locations,
    unapprove_window,
)
from identity import (
    PERMISSION_READ,
    PERMISSION_REVIEW,
    IdentityHttpError,
    OidcIdentity,
    Principal,
    StubIdentity,
    default_http_json,
    intersect_grants,
)
from refresh import (
    RefreshUnavailable,
    drop_stale_thumbnails,
    nornir_buildmanager_root,
    refresh_location,
)
from cardmask import drop_card_masks, ensure_card_mask
from negatives_view import negative_bitmask
import update_all
from converter import jobs as converter_jobs
from converter import odata as odata_module
from converter import web as converter_web
from converter.builder import BuildParams
from converter.tilesource import load_volume as load_viking_volume
import score as score_module
from gold import draw as gold_draw
from gold import store as gold_store
from gold import web as gold_web
from crops_path import crops_dir_for
from thumbs import ensure_thumbnail, parse_thumb_request, start_thumbnail_watch, thumb_parts_allowed

VOLUME_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
# "original" holds the converter's current-outline masks, drawn beside the proposal.
FILE_PREFIXES = frozenset({"images", "masks", "ignored", "original"})
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
    if converter_web.SET_NAME.fullmatch(name):
        return converter_web.resolve_structure_set(converter_web.converter_root(env_map()), name)
    if gold_store.SET_NAME.fullmatch(name):
        return gold_store.resolve_set(gold_store.gold_root(env_map()), name)
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


def odata_endpoint(volume: str, source: dict[str, Any], environ: dict[str, str] | None = None) -> str:
    """Saved volume URL, or the default service root when none is stored."""
    url = source.get("odata")
    if isinstance(url, str) and url.strip():
        return url.strip().rstrip("/")
    return odata_base(volume, environ)


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


def compile_tag_pattern(pattern: str) -> re.Pattern[str]:
    """The tag pattern as a case-insensitive regular expression.

    A pattern that is not a valid regular expression but uses ``*`` or ``?``, such as ``*CLS*``, is read as a
    wildcard pattern instead (``*`` any run of characters, ``?`` one character, anything else literal). Raises
    ``re.error`` for any other invalid pattern.
    """
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        if "*" not in pattern and "?" not in pattern:
            raise
        wildcard = "".join(".*" if ch == "*" else "." if ch == "?" else re.escape(ch) for ch in pattern)
        return re.compile(wildcard, re.IGNORECASE)


def structure_ids_for_tag_pattern(
    structures: list[dict[str, Any]],
    types: list[dict[str, Any]],
    pattern: str,
) -> list[int]:
    """Structure IDs whose own tags, or their type's tags, match *pattern*.

    *pattern* is a regular expression (see ``compile_tag_pattern`` for the wildcard fallback). A plain word matches
    that tag as a substring.
    """
    compiled = compile_tag_pattern(pattern)
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


ODATA_ATTEMPTS = 3
ODATA_RETRY_DELAY_SECONDS = 1.0


def _read_odata_json(url: str) -> Any:
    """GET *url* as JSON, retrying connection-level failures.

    The first DNS lookup from the container after a quiet spell can fail once (Docker Desktop's
    resolver, "No address associated with hostname") and succeed on the next try. HTTP error
    statuses are real answers from the service and are raised at once.
    """
    for attempt in range(1, ODATA_ATTEMPTS + 1):
        request = Request(url, headers={"Accept": "application/json"})
        try:
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError:
            raise
        except URLError as exc:
            print(f"annotation-gallery OData request failed (attempt {attempt}/{ODATA_ATTEMPTS}): {url}: {exc}",
                  file=sys.stderr, flush=True)
            if attempt == ODATA_ATTEMPTS:
                raise
            time.sleep(ODATA_RETRY_DELAY_SECONDS)


def fetch_odata_location(base_url: str, location_id: int) -> dict[str, Any]:
    """Read one location's MosaicShape and LastModified from an OData service root."""
    query = urlencode(
        {
            "$select": "ID,ParentID,TypeCode,Z,X,Y,OffEdge,MosaicShape,LastModified,Radius",
            "$expand": "Parent($select=ID,TypeID,Label;$expand=Type($select=ID,Name,ParentID))",
        },
        safe="(),$=",
    )
    url = f"{base_url.rstrip('/')}/Locations({int(location_id)})?{query}"
    payload = _read_odata_json(url)
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
        payload = _read_odata_json(next_url)
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


def require_login(environ: dict[str, str] | None = None) -> bool:
    """Anonymous visitors are sent to the Identity login. On by default in oidc mode, never in stub mode."""
    mapping = environ or env_map()
    if identity_mode(mapping) != "oidc":
        return False
    return (mapping.get("GALLERY_REQUIRE_LOGIN") or "1").strip().lower() not in {"0", "false", "no", "off"}


def public_https_origin(environ: dict[str, str] | None = None) -> str:
    mapping = environ or env_map()
    return (mapping.get("GALLERY_PUBLIC_HTTPS_ORIGIN") or "").strip().rstrip("/")


def cookie_attributes(environ: dict[str, str] | None = None, max_age: int | None = None) -> str:
    """Attributes for every cookie we set. Secure once the public origin is HTTPS."""
    attrs = "Path=/; HttpOnly; SameSite=Lax"
    if public_https_origin(environ).lower().startswith("https://"):
        attrs += "; Secure"
    if max_age is not None:
        attrs += f"; Max-Age={max_age}"
    return attrs


SESSION_MAX_AGE = 8 * 3600

# The catalog of a big volume is tens of MB of JSON (CPED: 52k rows, 27 MB, ~4.5 s to build). It is rebuilt only when
# the database or the review files change, and sent gzipped to browsers that accept it.
_CATALOG_CACHE: dict[str, tuple[tuple, bytes, bytes]] = {}
_CATALOG_LOCK = threading.Lock()


def catalog_stamp(crops: Path) -> tuple:
    """Changes whenever something the catalog shows changes (rows, review state, rejected masks)."""
    parts = []
    for name in ("annotation_crops.sqlite", "annotation_crops.sqlite-wal", "ignore.json", "approved.json", "ignored"):
        try:
            info = (Path(crops) / name).stat()
            parts.append((name, info.st_mtime_ns, info.st_size))
        except OSError:
            parts.append((name, 0, 0))
    return tuple(parts)


def catalog_body(crops: Path, volume: str, permission: str, crop_size: int) -> tuple[bytes, bytes]:
    """(json, gzip) of the whole catalog answer, cached until the volume's stamp changes."""
    key = f"{crops}|{volume}|{permission}|{crop_size}"
    stamp = catalog_stamp(crops)
    with _CATALOG_LOCK:
        cached = _CATALOG_CACHE.get(key)
        if cached and cached[0] == stamp:
            return cached[1], cached[2]
    body = json.dumps(
        {"volume": volume, "permission": permission, "cropSize": crop_size, "rows": list_catalog_rows(crops)}
    ).encode("utf-8")
    packed = gzip.compress(body, compresslevel=5)
    with _CATALOG_LOCK:
        _CATALOG_CACHE[key] = (stamp, body, packed)
    return body, packed


TOKEN_COOKIE = "gallery_sid"

# Identity access tokens by random session id, in memory only: a restart means the next Viking link asks the
# user to sign in again. The cookie carries the id, never the token.
_TOKENS: dict[str, tuple[str, float]] = {}
_TOKENS_LOCK = threading.Lock()


def remember_token(token: str, expires_in: int) -> str:
    """Store an access token and return the session id that finds it."""
    sid = secrets.token_urlsafe(24)
    lifetime = min(max(expires_in, 60), SESSION_MAX_AGE) if expires_in else SESSION_MAX_AGE
    now = time.monotonic()
    with _TOKENS_LOCK:
        for key in [key for key, (_token, expiry) in _TOKENS.items() if expiry <= now]:
            del _TOKENS[key]
        _TOKENS[sid] = (token, now + lifetime - 30)
    return sid


def recall_token(sid: str | None) -> str:
    """The access token for a session id, or '' when unknown or expired."""
    if not sid:
        return ""
    with _TOKENS_LOCK:
        found = _TOKENS.get(sid)
        if found is None:
            return ""
        token, expiry = found
        if expiry <= time.monotonic():
            del _TOKENS[sid]
            return ""
        return token


def forget_token(sid: str | None) -> None:
    if sid:
        with _TOKENS_LOCK:
            _TOKENS.pop(sid, None)


def viking_link(
    api_base: str,
    token: str,
    identity_name: str,
    location: str,
    http_json: Callable[..., Any] = default_http_json,
) -> str:
    """viking:// link that opens (or jumps) Viking on a location, signed in as the token's user.

    Asks Identity for a one-use launch code. When Identity refuses the volume (403) the plain link is returned:
    Viking then signs in by itself. A 401 is raised as IdentityHttpError so the caller can send the user to sign in.
    """
    plain = "viking://open?volumeName=" + quote(identity_name, safe="") + "&location=" + quote(location, safe="")
    try:
        payload = http_json(
            api_base.rstrip("/") + "/api/viking/launch-code",
            method="POST",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            body=json.dumps({"volume_name": identity_name}).encode("utf-8"),
        )
    except IdentityHttpError as exc:
        if exc.status in {403, 404}:
            return plain
        raise
    url = str((payload or {}).get("viking_url") or "")
    if not url.lower().startswith("viking://"):
        return plain
    return url + "&location=" + quote(location, safe="")


def identity_logout_url(environ: dict[str, str] | None = None) -> str:
    """Where /logout sends the browser after the gallery cookie is cleared.

    In oidc mode that is Identity Server's end-session page, so the single sign-on session ends too; otherwise
    the next visit would sign straight back in. Without an id_token_hint Identity asks the user to confirm.
    """
    mapping = environ or env_map()
    if identity_mode(mapping) != "oidc":
        return "/"
    authority = (mapping.get("GALLERY_OIDC_AUTHORITY") or "https://identity.codepharm.net:5001").rstrip("/")
    return f"{authority}/connect/endsession"

# Reachable without a session: health probe, the login round trip, robots.txt and the static shell assets.
_PUBLIC_PATHS = {"/health", "/login", "/callback", "/logout", "/robots.txt", "/api/config"}


def is_public_path(path: str) -> bool:
    return path in _PUBLIC_PATHS or path.startswith("/static/")


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


def registry_aliases(registry: Path) -> dict[str, str]:
    """Registry folder -> annotation database name (what Identity grants are named after)."""
    aliases: dict[str, str] = {}
    for name in list_registry_names(registry):
        try:
            aliases[name] = annotation_db_name(crops_dir(registry, name), name)
        except (ValueError, FileNotFoundError, OSError):
            aliases[name] = name
    return aliases


def volume_root(registry: Path, name: str) -> Path:
    """Return the volume-root path for a registry child. Name is never a path."""
    if not VOLUME_NAME.fullmatch(name):
        raise ValueError("invalid volume name")
    path = registry / name
    if not path.exists():
        raise FileNotFoundError(name)
    return path


def crops_dir(registry: Path, name: str) -> Path:
    """Crops folder for a registry child: the volume root itself (flat) or ``AnnotationCrops`` under it."""
    return crops_dir_for(volume_root(registry, name))


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


def https_redirect_location(
    host_header: str,
    request_target: str,
    *,
    http_port: int,
    https_port: int,
    public_origin: str | None = None,
) -> str:
    """Build the Location URL for an HTTP→HTTPS redirect.

    When ``public_origin`` is set (e.g. ``https://gallery.codepharm.net``), that
    origin wins. Otherwise the Host header is rewritten to HTTPS using
    ``https_port`` as the browser-facing port (compose may set this to the
    published port, e.g. 8443, while the container still listens on 443).
    ``http_port`` matches Host ports that should be replaced (80 or 8080).
    """
    path = request_target if request_target.startswith("/") else f"/{request_target}"
    origin = (public_origin or "").strip().rstrip("/")
    if origin:
        return f"{origin}{path}"

    host = (host_header or "127.0.0.1").strip() or "127.0.0.1"
    hostname = host
    host_port: int | None = None
    if host.startswith("["):
        end = host.find("]")
        if end != -1:
            hostname = host[: end + 1]
            rest = host[end + 1 :]
            if rest.startswith(":") and rest[1:].isdigit():
                host_port = int(rest[1:])
    elif host.count(":") == 1:
        maybe_host, port_text = host.rsplit(":", 1)
        if port_text.isdigit():
            hostname = maybe_host
            host_port = int(port_text)

    if https_port == 443:
        return f"https://{hostname}{path}"
    if host_port is None or host_port in {80, http_port, https_port}:
        return f"https://{hostname}:{https_port}{path}"
    return f"https://{hostname}:{https_port}{path}"


# Crawlers and AI agents that identify themselves; they get 403 on everything but /robots.txt.
AI_BOT_MARKERS = (
    "gptbot", "chatgpt-user", "oai-searchbot", "claudebot", "claude-web", "claude-user", "claude-searchbot",
    "anthropic-ai", "ccbot", "google-extended", "googleother", "perplexitybot", "perplexity-user",
    "bytespider", "amazonbot", "applebot-extended", "cohere-ai", "meta-externalagent", "facebookbot",
    "diffbot", "omgili", "imagesiftbot", "timpibot", "youbot", "ai2bot", "duckassistbot",
    "mistralai-user",
)

ROBOTS_TXT = "User-agent: *\nDisallow: /\n"


def is_ai_bot(user_agent: str) -> bool:
    lowered = (user_agent or "").lower()
    return any(marker in lowered for marker in AI_BOT_MARKERS)


class GalleryHandler(BaseHTTPRequestHandler):
    """HTTP API + static SPA. Does not spawn nornir-build."""

    server_version = "AnnotationGallery/1.0"

    def end_headers(self) -> None:
        # Keep every response out of search indexes and AI training sets.
        self.send_header("X-Robots-Tag", "noindex, nofollow, noarchive, noimageindex, noai, noimageai")
        super().end_headers()

    def _refuse_bots(self, path: str) -> bool:
        """Serve robots.txt; answer 403 to self-identified AI crawlers."""
        if path == "/robots.txt":
            body = ROBOTS_TXT.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return True
        if is_ai_bot(self.headers.get("User-Agent") or ""):
            self.send_response(403)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return True
        return False

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

    def _registry(self) -> Path:
        """Training set for this request. Omitted ``set`` keeps the configured registry."""
        parsed = urlparse(self.path)
        name = (parse_qs(parsed.query).get("set") or [""])[0].strip()
        if not name:
            return registry_dir()
        return resolve_training_set(name)

    def _maybe_redirect_https(self) -> bool:
        """When this process has HTTPS bound, send 301 from the cleartext listener."""
        if not getattr(self.server, "force_https_redirect", False):
            return False
        location = https_redirect_location(
            self.headers.get("Host") or "",
            self.path,
            http_port=int(getattr(self.server, "gallery_http_port", 80)),
            https_port=int(getattr(self.server, "gallery_https_port", 443)),
            public_origin=getattr(self.server, "gallery_public_https_origin", None),
        )
        self.send_response(301)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()
        return True

    def _require_session(self, path: str) -> bool:
        """True when the request was answered because it has no login. Pages go to /login, the API gets 401."""
        if not require_login() or is_public_path(path):
            return False
        if self._principal().subject != "anonymous":
            return False
        if path.startswith("/api/"):
            self._send_json({"error": "login required"}, status=401)
        else:
            self._redirect("/login")
        return True

    def do_GET(self) -> None:
        if self._refuse_bots(unquote(urlparse(self.path).path)):
            return
        if self._maybe_redirect_https():
            return
        if self._require_session(unquote(urlparse(self.path).path)):
            return
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)
        if path == "/health":
            self._send_json({"ok": True})
            return
        if (
            path in {"/", "/index.html"}
            or (path in {"/converter", "/converter/"} and converter_web.converter_root(env_map()))
            or (path in {"/gold", "/gold/"} and gold_store.gold_root(env_map()))
        ):
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
                    "converter": converter_web.converter_root(env_map()) is not None,
                    "gold": gold_store.gold_root(env_map()) is not None,
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
            forget_token(self._cookie(TOKEN_COOKIE))
            self._clear_session(identity_logout_url())
            return
        if path == "/api/training-sets":
            if (query.get("app") or [""])[0] == "gold":
                sets = gold_store.list_sets(gold_store.gold_root(env_map()))
                self._send_json({"sets": sets, "default": sets[-1] if sets else ""})
                return
            if (query.get("app") or [""])[0] == "converter":
                sets = converter_web.list_structure_sets(converter_web.converter_root(env_map()))
                self._send_json({"sets": sets, "default": sets[-1] if sets else ""})
                return
            self._send_json({"sets": list_training_sets(), "default": "current"})
            return
        if path == "/api/review-lists":
            self._handle_training_review_export()
            return
        if path == "/api/me":
            principal = self._principal()
            try:
                aliases = registry_aliases(self._registry())
            except (ValueError, FileNotFoundError):
                aliases = {}
            self._send_json(
                {
                    "subject": principal.subject,
                    "displayName": principal.display_name,
                    "volumes": [
                        {"name": name, "permission": role, "identityName": aliases.get(name, name)}
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
        if path == "/api/converter/builds":
            self._handle_converter_builds()
            return
        match = re.fullmatch(r"/api/gold/sets/([^/]+)/(progress|export)", path)
        if match:
            self._handle_gold_read(match.group(1), match.group(2), (query.get("volume") or [""])[0])
            return
        match = re.fullmatch(r"/api/volumes/([^/]+)/gold/cards", path)
        if match:
            crops = self._granted_crops(match.group(1))
            if crops is not None:
                self._send_json({"cards": gold_store.cards(crops)})
            return
        match = re.fullmatch(r"/api/volumes/([^/]+)/converter/(cards|batch)", path)
        if match:
            self._handle_converter_read(match.group(1), match.group(2))
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
        match = re.fullmatch(r"/api/volumes/([^/]+)/update-all", path)
        if match:
            self._handle_update_all(match.group(1), "status")
            return
        match = re.fullmatch(r"/api/volumes/([^/]+)/cardmask", path)
        if match:
            self._handle_cardmask(
                match.group(1),
                (query.get("size") or [""])[0],
                (query.get("key") or [""])[0],
                (query.get("loc") or [""])[0],
            )
            return
        match = re.fullmatch(r"/api/volumes/([^/]+)/negatives", path)
        if match:
            self._handle_negatives(
                match.group(1), (query.get("key") or [""])[0], (query.get("loc") or [""])[0]
            )
            return
        self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        if self._maybe_redirect_https():
            return
        if self._require_session(unquote(urlparse(self.path).path)):
            return
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/api/review-lists":
            self._handle_training_review_import()
            return
        if path == "/api/converter/structures":
            self._handle_converter_create()
            return
        if path == "/api/converter/delete":
            self._handle_converter_delete()
            return
        if path == "/api/gold/sets":
            self._handle_gold_create()
            return
        if path == "/api/converter/preview":
            self._handle_converter_preview()
            return
        match = re.fullmatch(r"/api/gold/sets/([^/]+)/(list|replacements)", path)
        if match:
            self._handle_gold_change(match.group(1), match.group(2))
            return
        match = re.fullmatch(r"/api/volumes/([^/]+)/gold/edit", path)
        if match:
            self._handle_gold_edit(match.group(1))
            return
        match = re.fullmatch(r"/api/volumes/([^/]+)/converter/edit", path)
        if match:
            self._handle_converter_edit(match.group(1))
            return
        match = re.fullmatch(r"/api/volumes/([^/]+)/viking-link", path)
        if match:
            self._handle_viking_link(match.group(1))
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
        match = re.fullmatch(r"/api/volumes/([^/]+)/import-location", path)
        if match:
            self._handle_import_location(match.group(1))
            return
        match = re.fullmatch(r"/api/volumes/([^/]+)/review-batch", path)
        if match:
            self._handle_review_batch(match.group(1))
            return
        match = re.fullmatch(r"/api/volumes/([^/]+)/update-all(?:/(cancel))?", path)
        if match:
            self._handle_update_all(match.group(1), "cancel" if match.group(2) else "start")
            return
        match = re.fullmatch(r"/api/volumes/([^/]+)/(ignore|restore|approve|unapprove)", path)
        if not match:
            self._send_json({"error": "not found"}, status=404)
            return
        volume = match.group(1)
        action = match.group(2)
        if not self._review_granted(volume):
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
        crops = self._granted_crops(volume)
        if crops is None:
            return
        in_gold = bool(gold_store.SET_NAME.fullmatch(self._registry().name))
        if action == "ignore":
            ignore_location(crops, location_id, image_key)
        elif action == "restore":
            restore_location(crops, location_id, image_key)
        elif action == "approve":
            approve_window(crops, location_id, image_key)
        else:
            unapprove_window(crops, location_id, image_key)
        if in_gold:
            if action == "approve":
                principal = self._principal()
                gold_web.record_approval(crops, location_id, principal.display_name or principal.subject or "unknown")
            else:
                gold_web.clear_approval(crops, location_id)
        if action in ("ignore", "restore"):
            # A rejected mask stops being a negative for its neighbors, so the whole tile's layers are stale.
            drop_card_masks(crops, location_image_keys(crops, [location_id]))
        self._send_json({
            "ok": True,
            "location_id": location_id,
            "ignored": action == "ignore",
            "approved": action == "approve",
        })

    def _handle_update_all(self, volume: str, action: str) -> None:
        """Start, watch, or cancel the volume's Update All job. Start and cancel need review permission."""
        principal = self._principal()
        role = principal.grants.get(volume)
        if role is None:
            self._send_json({"error": "not found"}, status=404)
            return
        if action != "status" and role != PERMISSION_REVIEW:
            self._send_json({"error": "review required"}, status=403)
            return
        try:
            crops = crops_dir(self._registry(), volume)
        except (ValueError, FileNotFoundError):
            self._send_json({"error": "not found"}, status=404)
            return
        set_name = self._registry().name
        if converter_web.SET_NAME.fullmatch(set_name):
            self._handle_converter_job(volume, action, crops, set_name, int(converter_web.SET_NAME.fullmatch(set_name).group(1)))
            return
        job = update_all.current_job(volume)
        if action == "status":
            payload = update_all.idle_status(crops)
            if job is not None:
                payload = {**payload, **job.snapshot()}
            self._send_json(payload)
            return
        if action == "cancel":
            if job is not None and job.running:
                job.cancel()
            self._send_json({"ok": True, "running": bool(job and job.running)})
            return
        if nornir_buildmanager_root() is None:
            self._send_json({
                "error": "Update all needs the nornir-buildmanager mount. No masks were changed.",
            }, status=503)
            return
        source = load_source_document(crops)
        url = odata_endpoint(volume, source).strip()
        started_job, _new = update_all.start_job(volume, crops, url, str(source.get("exporter") or "tiled"))
        self._send_json({**update_all.idle_status(crops), **started_job.snapshot()})

    def _volume_source(self, volume: str) -> dict[str, Any]:
        """The volume's own source record (OData endpoint, where its tiles are) from the gallery's default training data.

        Volumes are not all served from the same place (the SEM ones are other labs' servers), so nothing is guessed from a
        template when the volume says where it is.
        """
        try:
            return load_source_document(crops_dir(registry_dir(), volume))
        except (ValueError, FileNotFoundError, OSError):
            return {}

    def _build_params(self, volume: str, structure_id: int, body: dict[str, Any]) -> BuildParams:
        def numbers(key: str) -> tuple[int, ...]:
            return tuple(int(item) for item in body.get(key) or [])

        def optional_int(key: str) -> int | None:
            return None if body.get(key) in (None, "") else int(body[key])

        source = self._volume_source(volume)
        tiles = source.get("tiles") if isinstance(source.get("tiles"), dict) else {}
        return BuildParams(
            volume=volume,
            odata_base=odata_endpoint(volume, source),
            viking_xml=(str(tiles.get("vikingXml")).strip() or None) if tiles.get("vikingXml") else None,
            tile_base=(str(tiles.get("base")).strip() or None) if tiles.get("base") else None,
            structure_id=structure_id,
            type_codes=numbers("typeCodes") or (1, 4, 6, 7),
            z_min=optional_int("zMin"),
            z_max=optional_int("zMax"),
            channel=(body.get("channel") or None),
            filter_name=(body.get("filter") or None),
            retry_failed=bool(body.get("retryFailed", True)),
        )

    def _handle_converter_job(self, volume: str, action: str, crops: Path, set_name: str, structure_id: int) -> None:
        """Status, start (build or continue) and cancel of a structure's build, in Update All's status shape."""
        job = converter_jobs.current_job(set_name, volume)
        if action == "status":
            self._send_json(job.snapshot() if job is not None else converter_jobs.idle_status(set_name, volume))
            return
        if action == "cancel":
            if job is not None and job.running:
                job.cancel()
            self._send_json({"ok": True, "running": bool(job and job.running)})
            return
        try:
            body = self._read_json()
            params = self._build_params(volume, structure_id, body if isinstance(body, dict) else {})
        except (TypeError, ValueError, json.JSONDecodeError):
            self._send_json({"error": "invalid build settings"}, status=400)
            return
        started = converter_jobs.start_job(converter_jobs.new_job(set_name, volume, crops, params))
        self._send_json(started.snapshot())

    def _handle_converter_edit(self, volume: str) -> None:
        """POST {location_id, image_key, rings}: a retraced outline replaces the card's proposal (the batch, not the database)."""
        if not self._review_granted(volume):
            return
        try:
            body = self._read_json()
            location_id = int(body["location_id"])
            image_key = str(body["image_key"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self._send_json({"error": "location_id, image_key and rings required"}, status=400)
            return
        crops = self._granted_crops(volume)
        if crops is None:
            return
        try:
            self._send_json(converter_web.apply_edit(crops, location_id, image_key, body.get("rings")))
        except converter_web.EditRejected as rejected:
            self._send_json({"error": str(rejected)}, status=400)

    # ---- gold sets ------------------------------------------------------------------------------------------------

    def _gold_set_dir(self, name: str) -> Path | None:
        try:
            return gold_store.resolve_set(gold_store.gold_root(env_map()), name)
        except (FileNotFoundError, ValueError):
            self._send_json({"error": "not found"}, status=404)
            return None

    def _gold_source_crops(self, meta: dict[str, Any], volume: str) -> Path:
        return crops_dir(resolve_training_set(str(meta.get("sourceSet") or "current")), volume)

    def _principal_for(self, set_dir: Path) -> Principal:
        """The caller's grants against the volumes of one gold set (this request's own ?set= is not that set)."""
        try:
            names = list_registry_names(set_dir)
        except (ValueError, FileNotFoundError):
            names = []
        return self._principal_over(names)

    def _review_granted_in(self, set_dir: Path, volume: str) -> bool:
        role = self._principal_for(set_dir).grants.get(volume)
        if role is None:
            self._send_json({"error": "not found"}, status=404)
        elif role != PERMISSION_REVIEW:
            self._send_json({"error": "review required"}, status=403)
        return role == PERMISSION_REVIEW

    def _handle_gold_read(self, name: str, what: str, volume: str) -> None:
        set_dir = self._gold_set_dir(name)
        if set_dir is None:
            return
        grants = self._principal_for(set_dir).grants
        meta = gold_store.read_meta(set_dir)
        if what == "progress":
            progress = gold_store.progress(set_dir)
            self._send_json({"set": name, "meta": meta, "progress": {v: p for v, p in progress.items() if v in grants}})
            return
        every = list(meta.get("volumes") or {})
        if volume and (volume not in grants or volume not in every):
            self._send_json({"error": "not found"}, status=404)
            return
        if not volume and any(v not in grants for v in every):
            self._send_json({"error": "export one volume at a time (volume=...); you hold no grant on all of them"}, status=403)
            return
        body = gold_web.export_zip(set_dir, volume or None)
        suffix = f"-{volume}" if volume else ""
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition", f'attachment; filename="{name}{suffix}.zip"')
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _handle_gold_create(self) -> None:
        """POST {name, sourceSet?, volumes?, n?, seed?, stratify?, band?}: draw a sample from the source set into a new gold set."""
        root = gold_store.gold_root(env_map())
        if root is None:
            self._send_json({"error": "not found"}, status=404)
            return
        try:
            body = self._read_json()
            name = str(body["name"])
            source_set = str(body.get("sourceSet") or "current")
            seed = int(body["seed"]) if body.get("seed") not in (None, "") else int.from_bytes(os.urandom(4), "big")
            count = int(body["n"]) if body.get("n") not in (None, "") else gold_draw.DEFAULT_COUNT
            stratify = str(body.get("stratify") or "")
            band = int(body.get("band") or gold_draw.DEFAULT_Z_BAND)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self._send_json({"error": "name is required; n, seed and band are numbers"}, status=400)
            return
        if gold_store.SET_NAME.fullmatch(source_set) or converter_web.SET_NAME.fullmatch(source_set):
            self._send_json({"error": "a gold set is drawn from a training set, not from a gold or converter set"}, status=400)
            return
        try:
            source_registry = resolve_training_set(source_set)
            grants = self._principal_over(list_registry_names(source_registry)).grants
        except (FileNotFoundError, ValueError):
            self._send_json({"error": "unknown source set"}, status=404)
            return
        lists = body.get("lists")
        wanted = list(lists) if isinstance(lists, dict) and lists else body.get("volumes")
        volumes = [str(v) for v in wanted] if isinstance(wanted, list) and wanted else sorted(
            v for v, role in grants.items() if role == PERMISSION_REVIEW
        )
        if not volumes or any(grants.get(v) != PERMISSION_REVIEW for v in volumes):
            self._send_json({"error": "review rights are needed on every volume in the sample"}, status=403)
            return
        try:
            sources = {v: crops_dir(source_registry, v) for v in volumes}
            if isinstance(lists, dict) and lists:
                parsed: dict[str, list[int]] = {}
                for volume, text in lists.items():
                    ids, bad = gold_draw.parse_id_list(str(text))
                    if bad:
                        raise gold_store.GoldError(f"{volume}: not location ids: {' '.join(bad[:5])}")
                    parsed[str(volume)] = ids
                meta = gold_store.create_set_from_lists(root, name, source_set, sources, parsed, note=str(body.get("note") or ""))
            else:
                meta = gold_store.create_set(root, name, source_set, sources, n=count, seed=seed, stratify=stratify, band=band)
        except (gold_store.GoldError, ValueError, FileNotFoundError) as error:
            self._send_json({"error": str(error)}, status=400)
            return
        self._send_json({"ok": True, "set": meta["name"], "meta": meta}, status=201)

    def _handle_gold_change(self, name: str, what: str) -> None:
        """POST list {volume, text, mode, force?} or replacements {volume, count?}."""
        set_dir = self._gold_set_dir(name)
        if set_dir is None:
            return
        try:
            body = self._read_json()
            volume = str(body["volume"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self._send_json({"error": "volume required"}, status=400)
            return
        if not self._review_granted_in(set_dir, volume):
            return
        meta = gold_store.read_meta(set_dir)
        try:
            source = self._gold_source_crops(meta, volume)
            if what == "list":
                ids, bad = gold_draw.parse_id_list(str(body.get("text") or ""))
                if not ids:
                    self._send_json({"error": "the list has no location ids", "bad": bad[:20]}, status=400)
                    return
                result = gold_store.apply_id_list(
                    set_dir, volume, source, ids, str(body.get("mode") or "extend"), force=bool(body.get("force"))
                )
                result["notIds"] = bad[:50]
                self._send_json(result)
                return
            added = gold_store.draw_replacement(set_dir, volume, source, max(1, min(50, int(body.get("count") or 1))))
            self._send_json({"ok": True, "added": added})
        except (gold_store.GoldError, ValueError, FileNotFoundError) as error:
            self._send_json({"error": str(error)}, status=400)

    def _handle_gold_edit(self, volume: str) -> None:
        """POST {location_id, image_key, rings | png}: replace a gold card mask (retrace rings or a painted PNG)."""
        if not self._review_granted(volume):
            return
        if not gold_store.SET_NAME.fullmatch(self._registry().name):
            self._send_json({"error": "not a gold set"}, status=400)
            return
        try:
            body = self._read_json()
            location_id = int(body["location_id"])
            image_key = str(body["image_key"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self._send_json({"error": "location_id and image_key required"}, status=400)
            return
        crops = self._granted_crops(volume)
        if crops is None:
            return
        try:
            self._send_json(gold_web.apply_edit(crops, location_id, image_key, rings=body.get("rings"), png=body.get("png")))
        except gold_web.EditRejected as rejected:
            self._send_json({"error": str(rejected)}, status=400)

    def _handle_converter_builds(self) -> None:
        """Every build the caller may see: one entry per (structure set, volume) with its counts and whether it is running."""
        root = converter_web.converter_root(env_map())
        if root is None:
            self._send_json({"builds": []})
            return
        builds = []
        for set_name in converter_web.list_structure_sets(root):
            set_dir = root / set_name
            grants = self._principal_for(set_dir).grants
            structure_id = int(converter_web.SET_NAME.fullmatch(set_name).group(1))
            for volume in sorted(grants):
                crops = set_dir / volume
                if not crops.is_dir():
                    continue
                job = converter_jobs.current_job(set_name, volume)
                builds.append(
                    {
                        "set": set_name, "volume": volume, "structureId": structure_id,
                        "running": bool(job and job.running), **converter_web.build_summary(crops),
                    }
                )
        builds.sort(key=lambda item: item.get("updated") or "", reverse=True)
        self._send_json({"builds": builds})

    def _handle_converter_preview(self) -> None:
        """POST {structureId, typeCodes?, zMin?, zMax?}: how many annotations match in each volume the caller may build in.

        This is what stops a build being started in the wrong volume: the page shows where the structure really is.
        """
        if converter_web.converter_root(env_map()) is None:
            self._send_json({"error": "not found"}, status=404)
            return
        try:
            body = self._read_json()
            structure_id = int(body["structureId"])
            if structure_id <= 0:
                raise ValueError("bad id")
            probe = self._build_params("x", structure_id, body)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self._send_json({"error": "structureId required"}, status=400)
            return
        volumes = sorted(v for v, role in self._principal().grants.items() if role == PERMISSION_REVIEW)
        get_json = converter_jobs.OVERRIDES.get("get_json", odata_module.http_get_json)

        def count(volume: str) -> dict[str, Any]:
            try:
                number = odata_module.count_locations(
                    odata_endpoint(volume, self._volume_source(volume)), structure_id, get_json=get_json,
                    type_codes=probe.type_codes, z_min=probe.z_min, z_max=probe.z_max,
                )
                return {"count": number}
            except Exception as error:  # an unreachable endpoint must not hide the other volumes
                return {"error": str(error)[:200]}

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max(1, min(8, len(volumes)))) as pool:
            counts = dict(zip(volumes, pool.map(count, volumes)))
        self._send_json({"structureId": structure_id, "volumes": counts})

    def _handle_converter_delete(self) -> None:
        """POST {set, volume}: delete one build and everything kept for it. Refused while it is building (409)."""
        root = converter_web.converter_root(env_map())
        if root is None:
            self._send_json({"error": "not found"}, status=404)
            return
        try:
            body = self._read_json()
            set_name = str(body["set"])
            volume = str(body["volume"])
            if not converter_web.SET_NAME.fullmatch(set_name) or not VOLUME_NAME.fullmatch(volume):
                raise ValueError("bad name")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self._send_json({"error": "set and volume required"}, status=400)
            return
        try:
            set_dir = converter_web.resolve_structure_set(root, set_name)
        except (FileNotFoundError, ValueError):
            self._send_json({"error": "not found"}, status=404)
            return
        if not self._review_granted_in(set_dir, volume):
            return
        job = converter_jobs.current_job(set_name, volume)
        if job is not None and job.running:
            self._send_json({"error": "this build is running; cancel it first"}, status=409)
            return
        try:
            converter_web.delete_build(root, set_name, volume)
        except (FileNotFoundError, ValueError):
            self._send_json({"error": "not found"}, status=404)
            return
        converter_jobs.forget_job(set_name, volume)
        self._send_json({"deleted": {"set": set_name, "volume": volume}})

    def _handle_converter_create(self) -> None:
        """POST {structureId, volume, typeCodes?, zMin?, zMax?, channel?, filter?}: make the structure's set and start building."""
        root = converter_web.converter_root(env_map())
        if root is None:
            self._send_json({"error": "not found"}, status=404)
            return
        try:
            body = self._read_json()
            structure_id = int(body["structureId"])
            volume = str(body["volume"])
            if structure_id <= 0 or not VOLUME_NAME.fullmatch(volume):
                raise ValueError("bad id")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self._send_json({"error": "structureId and volume required"}, status=400)
            return
        if not self._review_granted(volume):
            return
        set_name = converter_web.structure_set_name(structure_id)
        crops = root / set_name / volume
        try:
            params = self._build_params(volume, structure_id, body)
        except (TypeError, ValueError):
            self._send_json({"error": "invalid build settings"}, status=400)
            return
        converter_jobs.write_source(crops, params.odata_base, structure_id, params.viking_xml, params.tile_base)
        started = converter_jobs.start_job(converter_jobs.new_job(set_name, volume, crops, params))
        self._send_json({**started.snapshot(), "set": set_name}, status=202)

    def _handle_review_batch(self, volume: str) -> None:
        """Approve or reject a list of whole locations: body ``{"action": "approve"|"ignore", "location_ids": [...]}``."""
        if not self._review_granted(volume):
            return
        try:
            body = self._read_json()
            action = str(body.get("action"))
            raw_ids = body.get("location_ids")
            if action not in ("approve", "ignore") or not isinstance(raw_ids, list):
                raise ValueError("action and location_ids required")
            location_ids = [int(item) for item in raw_ids]
        except (TypeError, ValueError, json.JSONDecodeError):
            self._send_json({"error": "action (approve or ignore) and location_ids required"}, status=400)
            return
        crops = self._granted_crops(volume)
        if crops is None:
            return
        count = review_locations(crops, location_ids, action)
        if action == "ignore":
            drop_card_masks(crops, location_image_keys(crops, location_ids))
        self._send_json({"ok": True, "action": action, "count": count})

    def _handle_viking_link(self, volume: str) -> None:
        """POST {"location": id}: a viking:// link carrying a one-use launch code for the signed-in user."""
        principal = self._principal()
        if volume not in principal.grants:
            self._send_json({"error": "not found"}, status=404)
            return
        try:
            body = self._read_json()
        except (json.JSONDecodeError, ValueError):
            self._send_json({"error": "invalid JSON"}, status=400)
            return
        location = str(body.get("location") or "").strip()
        if not location:
            self._send_json({"error": "location required"}, status=400)
            return
        try:
            identity_name = registry_aliases(self._registry()).get(volume, volume)
        except (ValueError, FileNotFoundError):
            identity_name = volume
        token = recall_token(self._cookie(TOKEN_COOKIE))
        if not token:
            self._send_json({"error": "login required", "login": "/login"}, status=401)
            return
        api = os.environ.get("GALLERY_IDENTITY_API") or "https://identity.codepharm.net:6001"
        try:
            url = viking_link(api, token, identity_name, location)
        except IdentityHttpError as exc:
            status = 401 if exc.status == 401 else 502
            self._send_json({"error": "launch code failed", "login": "/login"}, status=status)
            return
        except PermissionError:
            self._send_json({"error": "identity unavailable"}, status=502)
            return
        self._send_json({"url": url})

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
        self.send_header("Set-Cookie", f"gallery_oidc_state={state}; {cookie_attributes(max_age=600)}")
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
            registry = registry_dir()
            principal = oidc.complete_login(
                code, redirect_uri, list_registry_names(registry), registry_aliases(registry)
            )
        except PermissionError as exc:
            self._send_json({"error": str(exc)}, status=401)
            return
        cookie = encode_session(principal, session_secret())
        sid = remember_token(principal.access_token, principal.token_expires_in) if principal.access_token else ""
        self.send_response(302)
        if sid:
            self.send_header("Set-Cookie", f"{TOKEN_COOKIE}={sid}; {cookie_attributes(max_age=SESSION_MAX_AGE)}")
        self.send_header("Location", "/")
        self.send_header(
            "Set-Cookie",
            f"{SESSION_COOKIE}={cookie}; {cookie_attributes(max_age=SESSION_MAX_AGE)}",
        )
        self.end_headers()

    def _handle_ignore_export(self, volume: str) -> None:
        crops = self._granted_crops(volume)
        if crops is None:
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
        crops = self._granted_crops(volume)
        if crops is None:
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
        if not self._review_granted(volume):
            return
        try:
            rejected, approved, replace = parse_review_import(self._read_json())
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc) or "invalid review list"}, status=400)
            return
        crops = self._granted_crops(volume)
        if crops is None:
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
            crops = crops_dir_for(registry / name)
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
            crops = crops_dir_for(registry / volume)
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
        crops = self._granted_crops(volume)
        if crops is None:
            return
        self._send_json(self._settings_payload(volume, load_source_document(crops)))

    def _handle_settings_post(self, volume: str) -> None:
        if not self._review_granted(volume):
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
        if not self._review_granted(volume):
            return
        try:
            body = self._read_json()
            location_id = int(body.get("location_id"))
        except (TypeError, ValueError, json.JSONDecodeError):
            self._send_json({"error": "location_id required"}, status=400)
            return
        crops = self._granted_crops(volume)
        if crops is None:
            return
        source = load_source_document(crops)
        url = odata_endpoint(volume, source)
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
            if gold_store.SET_NAME.fullmatch(self._registry().name):
                keys = gold_web.refresh_card(crops, location_id, entity)
            else:
                keys = refresh_location(crops, entity, exporter)
        except RefreshUnavailable as exc:
            self._send_json({"error": str(exc)}, status=503)
            return
        except (TypeError, ValueError, gold_web.EditRejected) as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        drop_stale_thumbnails(crops, keys)
        score = self._score_refreshed(volume, crops, entity)
        self._send_json({"ok": True, "location_id": location_id, "imageKeys": keys, "score": score})

    def _score_refreshed(self, volume: str, crops: Path, entity: dict[str, Any]) -> dict[str, Any] | None:
        """Score the refreshed location with the current model (see score.py) and store it as the card's SAM2 badge.

        Uses the converter's segmentation server (``GALLERY_SEGMENTATION_TARGET``) and the volume's tiles. A failure
        is returned as ``{"error": ...}``; the refresh itself has already succeeded. ``GALLERY_SCORE_ON_REFRESH=0``
        turns this off (None).
        """
        if os.environ.get("GALLERY_SCORE_ON_REFRESH", "1").strip().lower() in ("0", "false", "no", "off"):
            return None
        source = self._volume_source(volume)
        tiles = source.get("tiles") if isinstance(source.get("tiles"), dict) else {}
        viking_xml = (str(tiles.get("vikingXml")).strip() or None) if tiles.get("vikingXml") else None
        tile_base = (str(tiles.get("base")).strip() or None) if tiles.get("base") else None
        make_volume = converter_jobs.OVERRIDES.get("make_volume", converter_jobs.default_volume)
        if viking_xml:
            def load_volume() -> Any:
                return load_viking_volume(viking_xml, host=tile_base)
        else:
            def load_volume() -> Any:
                return make_volume(volume)
        return score_module.score_after_refresh(
            crops,
            volume,
            entity,
            make_client=converter_jobs.OVERRIDES.get("make_client", converter_jobs.default_client),
            load_volume=load_volume,
            volume_key=viking_xml or volume,
            make_cache=converter_jobs.OVERRIDES.get("make_cache", converter_jobs.tile_cache),
            semaphore=converter_jobs.streams(),
        )

    def _handle_import_location(self, volume: str) -> None:
        """Look a location id up in OData and, when the volume has the crop images, import it."""
        if not self._review_granted(volume):
            return
        try:
            location_id = int(self._read_json().get("location_id"))
        except (TypeError, ValueError, json.JSONDecodeError):
            self._send_json({"error": "location_id required"}, status=400)
            return
        crops = self._granted_crops(volume)
        if crops is None:
            return
        job = update_all.current_job(volume)
        if job is not None and job.running:
            self._send_json({"status": "busy", "location_id": location_id,
                             "reason": "Update all is running"})
            return
        url = odata_endpoint(volume, load_source_document(crops)).strip()
        try:
            entity = fetch_odata_location(url, location_id)
        except HTTPError as exc:
            if exc.code == 404:
                self._send_json({"status": "not_found", "location_id": location_id})
            else:
                self._send_json({"error": f"OData lookup failed: {exc}"}, status=502)
            return
        except (URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError) as exc:
            self._send_json({"error": f"OData lookup failed: {exc}"}, status=502)
            return
        if nornir_buildmanager_root() is None:
            self._send_json({"error": "Importing needs the nornir-buildmanager mount."}, status=503)
            return
        try:
            result = update_all.import_location(crops, entity)
        except RefreshUnavailable as exc:
            self._send_json({"error": str(exc)}, status=503)
            return
        except (TypeError, ValueError, KeyError) as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        self._send_json(result)

    def _handle_converter_read(self, volume: str, what: str) -> None:
        """``cards``: per-card state, reason and overlap for the page. ``batch``: the approved shapes as a JSON-lines download."""
        crops = self._granted_crops(volume)
        if crops is None:
            return
        if what == "cards":
            self._send_json({"cards": converter_web.card_rows(crops), "links": converter_web.link_rows(crops)})
            return
        body = converter_web.batch_jsonl(crops)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Content-Disposition", f'attachment; filename="{self._registry().name}-{volume}-approved.jsonl"')
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _handle_catalog(self, volume: str) -> None:
        crops = self._granted_crops(volume)
        if crops is None:
            return
        principal = self._principal()
        body_json, body_gzip = catalog_body(crops, volume, principal.grants[volume], read_crop_size(crops) or 1024)
        wants_gzip = "gzip" in (self.headers.get("Accept-Encoding") or "").lower()
        body = body_gzip if wants_gzip else body_json
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Vary", "Accept-Encoding")
        if wants_gzip:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        self.wfile.write(body)

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
            compile_tag_pattern(text)
        except re.error as exc:
            self._send_json({"error": f"invalid tag pattern: {exc}"}, status=400)
            return
        try:
            ids = tag_structure_ids(volume, text)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            self._send_json({"error": f"tag lookup failed: {exc}"}, status=502)
            return
        self._send_json({"structureIds": ids})

    def _handle_negatives(self, volume: str, image_key: str, location_id: str) -> None:
        """PNG bitmask of the other annotations in a tile, with their count in ``X-Negative-Count``.

        These are the areas the trainer marks as background for this location, drawn by the viewer as a
        red overlay. Same grants as the file route; ``no-cache`` so a changed mask or circle file shows up.
        """
        principal = self._principal()
        if volume not in principal.grants:
            self._send_json({"error": "not found"}, status=404)
            return
        try:
            location = int(location_id)
            data, count = negative_bitmask(crops_dir(self._registry(), volume), image_key, location)
        except (ValueError, OSError):
            self._send_json({"error": "not found"}, status=404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Negative-Count", str(count))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _handle_cardmask(self, volume: str, size: str, image_key: str, location_id: str) -> None:
        """Card-size RGB layer (R target mask, G negatives, B overlap) for one card.

        ``no-cache`` plus an ETag, so a changed mask is picked up and an unchanged one costs a 304.
        """
        principal = self._principal()
        if volume not in principal.grants:
            self._send_json({"error": "not found"}, status=404)
            return
        try:
            data = ensure_card_mask(crops_dir(self._registry(), volume), int(size), image_key, int(location_id))
        except (ValueError, OSError):
            self._send_json({"error": "not found"}, status=404)
            return
        etag = '"' + hashlib.sha1(data).hexdigest()[:20] + '"'
        if self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("ETag", etag)
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

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

    def _review_granted(self, volume: str) -> bool:
        """True when the caller may review ``volume``; otherwise sends 404 (no grant) or 403 (read-only)."""
        role = self._principal().grants.get(volume)
        if role is None:
            self._send_json({"error": "not found"}, status=404)
        elif role != PERMISSION_REVIEW:
            self._send_json({"error": "review required"}, status=403)
        return role == PERMISSION_REVIEW

    def _granted_crops(self, volume: str) -> Path | None:
        """Crops folder of a volume the caller holds a grant on; otherwise sends 404 and returns None."""
        try:
            if volume in self._principal().grants:
                return crops_dir(self._registry(), volume)
        except (ValueError, FileNotFoundError):
            pass
        self._send_json({"error": "not found"}, status=404)
        return None

    def _principal(self) -> Principal:
        try:
            names = list_registry_names(self._registry())
        except (ValueError, FileNotFoundError):
            names = []
        return self._principal_over(names)

    def _principal_over(self, names: list[str]) -> Principal:
        """The caller with grants limited to ``names`` (the volumes of whichever set the request is about)."""
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
            grants = intersect_grants(principal.grants, names)
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
        origin = public_https_origin()
        if origin:
            return f"{origin}/callback"
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

    def _clear_session(self, location: str = "/") -> None:
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Set-Cookie", f"{SESSION_COOKIE}=; {cookie_attributes(max_age=0)}")
        self.send_header("Set-Cookie", f"{TOKEN_COOKIE}=; {cookie_attributes(max_age=0)}")
        self.end_headers()


def make_server(
    host: str = "0.0.0.0",
    port: int = 80,
) -> ThreadingHTTPServer:
    """Bind one gallery HTTP server. Callers wrap the socket for TLS."""
    return ThreadingHTTPServer((host, port), GalleryHandler)


def _configure_http_redirect(
    http_server: ThreadingHTTPServer,
    *,
    enabled: bool,
    http_port: int,
    https_port: int,
) -> None:
    """Mark the cleartext server so handlers 301 to HTTPS when TLS is up."""
    http_server.force_https_redirect = enabled
    http_server.gallery_http_port = http_port
    http_server.gallery_https_port = https_port
    origin = (os.environ.get("GALLERY_PUBLIC_HTTPS_ORIGIN") or "").strip()
    http_server.gallery_public_https_origin = origin or None


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
    # HTTPS listener must not redirect to itself.
    _configure_http_redirect(server, enabled=False, http_port=port, https_port=port)
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
    # Browser-facing ports may differ from container listen ports (8080/8443 publish).
    http_public_port = int(os.environ.get("GALLERY_HTTP_PUBLIC_PORT") or http_port)
    https_public_port = int(os.environ.get("GALLERY_HTTPS_PUBLIC_PORT") or https_port)
    http_server = make_server(host, http_port)
    https_server = _https_server(host, https_port)
    _configure_http_redirect(
        http_server,
        enabled=https_server is not None,
        http_port=http_public_port,
        https_port=https_public_port,
    )
    start_thumbnail_watch(registry_dir())
    print(f"annotation-gallery HTTP on {host}:{http_port}", flush=True)
    if https_server is not None:
        print(f"annotation-gallery HTTPS on {host}:{https_port}", flush=True)
        print("annotation-gallery HTTP redirects to HTTPS", flush=True)
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
