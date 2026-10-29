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
# 1. Copy ``config-template/docker-env-template.txt`` to
#    ``D:\Docker\Builds\AnnotationGallery\.env`` and set ``LETSENCRYPT_EMAIL``.
# 2. Ensure ``D:\Docker\Builds\cloudflare.ini`` exists for DNS-01.
# 3. Run:
#
# ```powershell
# D:\Docker\Builds\AnnotationGallery\rebuild-and-start.ps1   (add -NoLetsEncrypt to skip the renewer)
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

## Converter (`/converter/`)

A twin of the gallery that turns the annotations of one Viking structure into proposed plain polygons. It is a mode of this
same server, off unless `GALLERY_CONVERTER_ROOT` is set. A user opens `/converter/`, enters a structure ID and presses
**Build structure**; each matching annotation (circle, polygon, curve polygon, closed curve; filter by type and section
range) is sent to the segmentation server, and a card appears with the proposal and, as a thin line, the current outline.
**Approve** adds the card to the structure's batch (`Download approved`, JSON lines); **Reject** leaves that location alone.
Nothing is written to the database. The pen retrace (key `R` in the viewer, also in the gallery) edits the outline; in the
converter the edit replaces the proposal, in the gallery it is only shown.

Each build is a training set `structure-<id>` under `GALLERY_CONVERTER_ROOT`, laid out like any crops folder, so the grid,
viewer, approve and reject are the gallery's own.

| Setting | Meaning |
|---|---|
| `GALLERY_CONVERTER_ROOT` | Folder builds are kept in. Unset turns the converter off. |
| `GALLERY_SEGMENTATION_TARGET` | `host:port` of the segmentation server's TLS gRPC port. |
| `GALLERY_SEGMENTATION_CA` | Optional PEM of trusted roots. |
| `GALLERY_SEGMENTATION_TLS_NAME` | Optional name to verify the certificate against when reached by another name. |
| `GALLERY_CONVERTER_VIKINGXML` | URL template of the volume's VikingXML, with `{volume}`: where the tile pyramids are listed. |
| `GALLERY_CONVERTER_TILE_CACHE` | Optional folder for cached tiles (revalidated by ETag). |
| `GALLERY_CONVERTER_STREAMS` | Segmentation streams at once across all builds (default 1). |

What the approved batch holds (`type_code` 6, plain `POLYGON` control points in mosaic pixels, the source shape's hash, the
model and the overlap with the old outline) is for a later database writer; none exists yet. Neighbouring annotations are
not yet sent to the segmentation server as background masks.

## Gold standard review (`/gold/`)

A third mode of the same server for building an evaluation set: for each volume, a seeded random sample of locations (100 by
default) whose masks a person touches up and approves. Off unless `GALLERY_GOLD_ROOT` is set. It never changes the training set
or the database: the sampled cards are copied into `<GALLERY_GOLD_ROOT>/gold-<name>/<volume>/` (an ordinary crops folder plus
`source_masks/`, the masks as drawn, never edited) and `gold.json` records the seed, stratification and source so the draw can
be audited and re-created.

* **Draw sample**: name, the training set to draw from, locations per volume, optional seed (recorded if left blank) and an
  optional spread over structure types or section ranges. A location that spans several crop windows (about half of RC1)
  contributes one window, chosen by the seed among those holding a substantial part of its mask, so large cells are not
  left out.
* **Open list**: a `.txt` of location ids (one per line or comma-separated, `5-8` an inclusive range, `#` comments) that extends or replaces a volume's
  cards. Ids that cannot be used are reported with the reason.
* **Touch up**: pen retrace (`R`) or the brush and eraser (`B`, hold Shift to erase, Ctrl+Z to undo a stroke, Save mask).
  Edits are exact to the pixel (a pixel is inside when its centre is).
* **Approve** pins the mask's SHA-1 with who and when. Editing an approved card withdraws the approval. Rejecting a card
  leaves it out; **Draw replacement** adds the next location of the seeded order, so replacements are reproducible too.
* **Download gold**: a zip of the approved masks and images with `manifest.json` (ids, hashes, edited or not, approver, seed,
  source set) and a list of any approved card whose mask no longer matches its approval.

Setting: `GALLERY_GOLD_ROOT`, the folder gold sets are kept in.
