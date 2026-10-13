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
#
# ## Card masks and Update all
#
# - Grid cards below Huge load ``overlays/cardmask/<size>/<imageKey>_<locationId>.png``: R is the location's mask,
#   G the other annotations in the tile (the trainer's negatives), B their overlap. Built on first request, reused
#   while newer than the sidecar, the circle file and every mask in the tile, and deleted when the gallery rejects,
#   restores, refreshes or updates a mask. Press N to hide the negatives on the grid as well as in the viewer.
#   ``GALLERY_CARDMASK_BUILDS`` (default 4) limits concurrent cold builds.
# - **Update all** (review role, needs the nornir mount) asks OData for every location changed since the volume's
#   watermark in ``AnnotationCrops/update_state.json``, one progress tick per OData page
#   (``GALLERY_UPDATE_PAGE_SIZE``, default 500). A location the volume has is redrawn; a new one is added only when
#   every crop image it needs is already on disk, otherwise it is skipped and counted (``update_all.log``, retried
#   on the next run). The watermark moves only after a run with no failures. First run: the older of the ``masks/``
#   folder's mtime and the oldest mask's creation time. Label and type edits do not change OData ``LastModified``,
#   so they arrive only with a changed shape or a full re-export.
#   Update all also rewrites ``circle_annotations.json``: a changed circle location gets its annotated circle
#   (center and the stored ``Radius``) recomputed, and a redrawn or added polygon gets its inscribed circle
#   recomputed, in every crop window the circle reaches. Only the listed locations change, and the file is
#   written once at the end. A volume without that file is left without one (the dialog says so). Deleted
#   annotations do not appear in an OData "changed since" query, so their circles stay until the next export.
