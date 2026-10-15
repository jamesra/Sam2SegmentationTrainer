# Annotation gallery docker environment template

Copy this folder’s templates to the host build path and fill in real values.
Do **not** commit filled copies or Cloudflare tokens.

## Copy targets

| Template | Copy to |
|----------|---------|
| `docker-env-template.txt` | `D:\Docker\Builds\AnnotationGallery\.env` |
| Shared Cloudflare DNS token | `D:\Docker\Builds\cloudflare.ini` (one file for all renewers) |

## Let’s Encrypt (certbot)

The gallery uses the shared `Servers/certbot` image with its **own** volume
(`gallery-letsencrypt`). It does not share Identity’s certificate lineage.

1. Copy `docker-env-template.txt` → `D:\Docker\Builds\AnnotationGallery\.env`.
2. Set `LETSENCRYPT_EMAIL`. Keep `LETSENCRYPT_PRIMARY_DOMAIN=gallery.codepharm.net`.
3. For production HTTPS on the public hostname, set the publish and PEM paths in
   that `.env` (see template comments): `80:80` / `443:443`,
   `GALLERY_SSL_*` under `/etc/letsencrypt/live/gallery.codepharm.net/`,
   `GALLERY_PUBLIC_HTTPS_ORIGIN` and OIDC redirect `https://gallery.codepharm.net/callback`.
4. Ensure `D:\Docker\Builds\cloudflare.ini` exists (DNS-01).

From the Sam2Trainer repo root:

```bash
D:\Docker\Builds\AnnotationGallery\rebuild-and-start.ps1   (add -NoLetsEncrypt to skip the renewer)
```

The first successful enroll restarts `annotation-gallery` onto the certificate.
Until PEMs exist, HTTP still serves the app; once HTTPS binds, cleartext :80
responds with **301** to HTTPS.

Skip the renewer (local stub / bind-mounted `/certs` only):

```bash
docker compose --env-file docker/.env -f docker/compose.cursor-dev.yaml up -d annotation-gallery
docker compose -f docker/compose.cursor-dev.yaml stop annotation-gallery-certbot-renewer
```

## Smoke

1. `http://gallery.codepharm.net/` → **301** → `https://gallery.codepharm.net/…`
2. HTTPS serves the SPA (`/health` on HTTPS returns `{"ok": true}`).
3. Renewer logs mention lineage `gallery.codepharm.net`.
4. Identity OIDC redirect allow-list includes `https://gallery.codepharm.net/callback`.
