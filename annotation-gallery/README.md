# Annotation overlay gallery
#
# Slim HTTP + SPA for per-volume ``AnnotationCrops``. This image does **not**
# install Nornir. Export, SAM2 scoring, and weekly refresh run on ``nornir:prod``.
#
# Lives beside the SAM2 trainer Compose service. Operator notes are in the
# trainer README.
#
# ## Ports and TLS
#
# - HTTP ``:80``, HTTPS ``:443`` when ``SSL_CERT_PATH`` / ``SSL_KEY_PATH`` exist.
# - When HTTPS is bound, every cleartext request returns **301** to HTTPS
#   (path and query preserved). Set ``GALLERY_PUBLIC_HTTPS_ORIGIN`` for the
#   canonical public name (``https://gallery.codepharm.net``).
# - Local compose publishes ``8080:80`` and ``127.0.0.1:8443:443``; redirect
#   uses ``GALLERY_HTTP_PUBLIC_PORT`` / ``GALLERY_HTTPS_PUBLIC_PORT``.
#
# ## Let's Encrypt
#
# Use the shared ``Servers/certbot`` renewer (own ``gallery-letsencrypt`` volume;
# not Identity's). Domain: ``gallery.codepharm.net``.
#
# 1. Copy ``config-template/build/docker-env-template.txt`` to
#    ``D:\Docker\Builds\AnnotationGallery\.env`` and set ``LETSENCRYPT_EMAIL``.
# 2. Ensure ``D:\Docker\Builds\cloudflare.ini`` exists for DNS-01.
# 3. From the trainer repo:
#
# ```bash
# docker compose --env-file docker/.env -f docker/compose.cursor-dev.yaml \
#   --profile letsencrypt up -d annotation-gallery annotation-gallery-certbot-renewer
# ```
#
# First enroll restarts ``annotation-gallery`` onto the PEMs. Smoke:
# ``http://gallery.codepharm.net/`` → 301 → ``https://…``; renewer lineage
# ``gallery.codepharm.net``. OIDC callback:
# ``https://gallery.codepharm.net/callback``.
#
# Details: ``config-template/README.md``.
