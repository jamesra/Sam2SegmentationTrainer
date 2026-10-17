"""Gallery identity: stub fixtures now, live OIDC against identity.codepharm.net later."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

HttpJson = Callable[..., Any]

PERMISSION_READ = "read"
PERMISSION_REVIEW = "review"

DEFAULT_OIDC_AUTHORITY = "https://identity.codepharm.net:5001"
DEFAULT_IDENTITY_API = "https://identity.codepharm.net:6001"


@dataclass(frozen=True)
class Principal:
    """Authenticated gallery user. Grants never include filesystem paths."""

    subject: str
    display_name: str
    grants: dict[str, str]


def map_permission_names(
    names: list[str],
    *,
    read_name: str = "Read",
    review_name: str = "Annotate",
) -> str | None:
    """Map Identity permission names to gallery Read vs Review."""
    lowered = {str(item).strip().lower() for item in names if item}
    if review_name.lower() in lowered:
        return PERMISSION_REVIEW
    if read_name.lower() in lowered:
        return PERMISSION_READ
    return None


def parse_stub_grants(text: str | None) -> dict[str, str]:
    """Parse ``RC2:review,RPC1:read`` into a grant map."""
    grants: dict[str, str] = {}
    if not text:
        return grants
    for chunk in text.split(","):
        piece = chunk.strip()
        if not piece or ":" not in piece:
            continue
        name, role = piece.split(":", 1)
        role = role.strip().lower()
        if role not in {PERMISSION_READ, PERMISSION_REVIEW}:
            continue
        grants[name.strip()] = role
    return grants


def intersect_grants(grants: dict[str, str], registry_names: list[str]) -> dict[str, str]:
    """Keep Identity names that exist as registry children. Drop unknown volumes."""
    allowed = set(registry_names)
    return {name: role for name, role in grants.items() if name in allowed}


def grants_for_registry(
    grants: dict[str, str],
    registry_names: list[str],
    aliases: dict[str, str] | None = None,
) -> dict[str, str]:
    """Grants keyed by registry folder: a folder takes the grant of its annotation database name.

    Names are compared case-insensitively: Identity calls the context ``cped`` and the dataset folder is ``CPED``.
    """
    by_lower: dict[str, str] = {}
    for name, role in grants.items():
        key = name.lower()
        if by_lower.get(key) != PERMISSION_REVIEW:
            by_lower[key] = role
    out: dict[str, str] = {}
    for folder in registry_names:
        role = by_lower.get(((aliases or {}).get(folder) or folder).lower())
        if role:
            out[folder] = role
    return out


class StubIdentity:
    """Fixture identity: every registry volume gets Read or Review from env/Bearer."""

    role: str
    explicit: dict[str, str]
    subject: str
    display_name: str

    def __init__(self, environ: dict[str, str]) -> None:
        self.role = (environ.get("GALLERY_STUB_ROLE") or PERMISSION_READ).strip().lower()
        if self.role not in {PERMISSION_READ, PERMISSION_REVIEW}:
            self.role = PERMISSION_READ
        self.explicit = parse_stub_grants(environ.get("GALLERY_STUB_GRANTS"))
        self.subject = environ.get("GALLERY_STUB_SUBJECT") or "stub-user"
        self.display_name = environ.get("GALLERY_STUB_NAME") or f"stub:{self.role}"

    def principal_from_bearer(self, token: str, registry_names: list[str]) -> Principal | None:
        """Tests send ``Authorization: Bearer read`` or ``Bearer review``."""
        role = token.strip().lower()
        if role not in {PERMISSION_READ, PERMISSION_REVIEW}:
            return None
        grants = {name: role for name in registry_names}
        return Principal(subject=self.subject, display_name=f"stub:{role}", grants=grants)

    def principal_for_registry(self, registry_names: list[str]) -> Principal:
        """Default env role applied to every registry child, plus explicit grants."""
        grants = {name: self.role for name in registry_names}
        grants.update(self.explicit)
        return Principal(
            subject=self.subject,
            display_name=self.display_name,
            grants=intersect_grants(grants, registry_names),
        )


class OidcIdentity:
    """Authorization-code OIDC against IdentityServer + IdentityApi volume grants."""

    authority: str
    api: str
    client_id: str
    client_secret: str
    scope: str
    authorize_path: str
    token_path: str
    userinfo_path: str
    volumes_path: str
    read_name: str
    review_name: str
    _http: HttpJson

    def __init__(
        self,
        environ: dict[str, str],
        *,
        http_json: HttpJson | None = None,
    ) -> None:
        self.authority = (environ.get("GALLERY_OIDC_AUTHORITY") or DEFAULT_OIDC_AUTHORITY).rstrip("/")
        self.api = (environ.get("GALLERY_IDENTITY_API") or DEFAULT_IDENTITY_API).rstrip("/")
        self.client_id = environ.get("GALLERY_OIDC_CLIENT_ID") or ""
        self.client_secret = environ.get("GALLERY_OIDC_CLIENT_SECRET") or ""
        # Viking.Annotation is the Permissions API's scope: without it the API treats the token as inactive (401).
        self.scope = environ.get("GALLERY_OIDC_SCOPE") or "openid profile Viking.Annotation"
        self.authorize_path = environ.get("GALLERY_OIDC_AUTHORIZE_PATH") or "/connect/authorize"
        self.token_path = environ.get("GALLERY_OIDC_TOKEN_PATH") or "/connect/token"
        self.userinfo_path = environ.get("GALLERY_OIDC_USERINFO_PATH") or "/connect/userinfo"
        # IdentityApi Permissions controller: {id: {Name, AnnotationDatabaseName, permissions: [...]}} for the caller.
        self.volumes_path = environ.get("GALLERY_IDENTITY_VOLUMES_PATH") or "/Permissions/AccessibleVolumes"
        self.read_name = environ.get("GALLERY_READ_PERMISSION") or "Read"
        self.review_name = environ.get("GALLERY_REVIEW_PERMISSION") or "Annotate"
        self._http = http_json or default_http_json

    def login_url(self, redirect_uri: str, state: str) -> str:
        """Browser redirect to the IdentityServer authorize endpoint."""
        query = urlencode(
            {
                "client_id": self.client_id,
                "redirect_uri": redirect_uri,
                "response_type": "code",
                "scope": self.scope,
                "state": state,
            }
        )
        return f"{self.authority}{self.authorize_path}?{query}"

    def complete_login(
        self,
        code: str,
        redirect_uri: str,
        registry_names: list[str],
        aliases: dict[str, str] | None = None,
    ) -> Principal:
        """Exchange the authorization code and load per-volume Read/Review grants."""
        token_payload = self._http(
            f"{self.authority}{self.token_path}",
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            body=urlencode(
                {
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                }
            ).encode("utf-8"),
        )
        access_token = str(token_payload.get("access_token") or "")
        if not access_token:
            raise PermissionError("OIDC token response did not include access_token")
        info = self._http(
            f"{self.authority}{self.userinfo_path}",
            method="GET",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        subject = str(info.get("sub") or info.get("name") or "oidc-user")
        display = str(info.get("name") or info.get("preferred_username") or subject)
        grants = self.volume_grants(access_token, registry_names, aliases)
        return Principal(subject=subject, display_name=display, grants=grants)

    def volume_grants(
        self,
        access_token: str,
        registry_names: list[str],
        aliases: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """IdentityApi volume list ∩ registry names, mapped to Read vs Review.

        ``aliases`` maps a registry folder name to the annotation database name Identity knows
        it by (SEM folders are not named after their database); a folder missing from it
        is matched by its own name.
        """
        url = urljoin(self.api + "/", self.volumes_path.lstrip("/"))
        payload = self._http(
            url,
            method="GET",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        grants: dict[str, str] = {}
        for item in _as_volume_items(payload):
            # The annotation database name is what the gallery matches on; the resource name is accepted too.
            labels = [
                str(item.get(key) or "")
                for key in ("AnnotationDatabaseName", "annotationDatabaseName", "name", "Name")
            ]
            names = item.get("permissions") or item.get("Permissions") or []
            if isinstance(names, str):
                names = [names]
            role = map_permission_names(
                [str(entry) for entry in names],
                read_name=self.read_name,
                review_name=self.review_name,
            )
            print(
                f"identity grant: names={[l for l in labels if l]} permissions={[str(n) for n in names]} -> {role}",
                file=sys.stderr,
            )
            if not role:
                continue
            for label in labels:
                if label and grants.get(label) != PERMISSION_REVIEW:
                    grants[label] = role
        result = grants_for_registry(grants, registry_names, aliases)
        print(
            f"identity grants: {len(grants)} named by Identity, registry {registry_names}, aliases {aliases} -> {result}",
            file=sys.stderr,
        )
        return result


def default_http_json(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    timeout: float = 30.0,
) -> Any:
    """JSON HTTP helper used by OIDC. Tests inject a fake."""
    request = Request(url, data=body, method=method, headers=headers or {})
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", "replace")[:200]
        except OSError:
            detail = ""
        raise PermissionError(f"Identity request failed: {exc} {url} {detail}".strip()) from exc
    except URLError as exc:
        raise PermissionError(f"Identity request failed: {exc} {url}") from exc
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _as_volume_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("value", "volumes", "items"):
            nested = payload.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
        # IdentityApi answers {resourceId: {...}}.
        return [item for item in payload.values() if isinstance(item, dict)]
    return []
