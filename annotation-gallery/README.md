# Annotation overlay gallery
#
# Slim HTTP + SPA for per-volume ``AnnotationCrops``. This image does **not**
# install Nornir. Export, SAM2 scoring, and weekly refresh run on ``nornir:prod``.
#
# Lives beside the SAM2 trainer Compose service. Operator notes are in the
# trainer README. HTTP :80, HTTPS :443 when SSL_CERT_PATH and SSL_KEY_PATH exist.
