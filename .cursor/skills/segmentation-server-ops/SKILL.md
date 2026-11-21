---
name: segmentation-server-ops
description: Deploy, verify and test the Viking segmentation server on James's workstation (legacy local deploy from D:\Docker\Builds\SegmentationServer) - rebuild one model's container, wait for health, prove a real segment through the demo API, run the server's pytest suite in Docker, and know where each cache lives. Use when asked to deploy or redeploy a segmentation model locally, check that a server answers, change a model's cache settings, or run the server tests.
---

# Segmentation server: local deploy, verify, test

The server code is `SegmentationServer/` in this repo. Hosts other than the workstation (Build2) deploy through
connectomes-deploy (`deploy/stacks.psd1`, stack `viking-segmentation`); this skill covers the workstation only.

## Models and ports (workstation)
- Config: `D:\Docker\Builds\SegmentationServer\models.json` (not in a repo; no commit). One row per model: `prefix`,
  `model` (under `D:\Docker\Run\segmentation-server`), `port`, optional `embeddingCacheDir`/`embeddingCacheMaxBytes`.
- Compose project `viking`; containers `viking-segmentation-<prefix>-1`.
- TLS port / demo HTTP port: v3 40445/40545, sem-v1 40447/40547, latest 40449/40549.

## Redeploy one model
```powershell
& D:\Docker\Builds\SegmentationServer\rebuild-and-start.ps1 <prefix> -ServerRoot D:\src\git\Sam2Trainer -SkipProxy *> <scratchpad>\deploy_<prefix>.log
```
- About 20 s of downtime for that model only; the others keep running.
- Wait for health with a poll loop (a bare `Start-Sleep` is blocked):
  `while ((docker ps --filter name=viking-segmentation-<prefix> --format '{{.Status}}') -notmatch 'healthy\)') { [Threading.Thread]::Sleep(5000) }`
- Then check the log: `docker logs viking-segmentation-<prefix>-1 2>&1 | Select-String 'Embedding disk cache|Result cache|Tile store|Permission'`.

## Prove a real segment (demo API, plain HTTP on the demo port)
```powershell
$h=@{'X-Demo-Client'='1'}   # without it: 403
$r=Invoke-WebRequest -Method Post -Uri http://127.0.0.1:<demo>/api/images -Headers $h -ContentType image/png -InFile <png>
Invoke-WebRequest -Method Post -Uri "http://127.0.0.1:<demo>/api/images/$($r.Headers['X-Image-Id'])/segment" -Headers $h `
  -ContentType application/json -Body '{"points":[{"x":603,"y":564,"label":1}]}'
```
Use `SegmentationServer\Example\images\RC1_111_D1_X72192-73216_Y63488-64512.png` with point (603, 564); expect 200
and a segment score around 0.9.

## Caches
- Embedding disk store: off unless `SEGMENTATION_EMBEDDING_CACHE_MAX_BYTES` > 0; when off, old stamp folders are purged
  at startup. Uploaded tiles still go to `<SEGMENTATION_EMBEDDING_CACHE>/tiles` (8 GiB cap) - expected, not a leak.
- Result store: `D:\Docker\cache\segmentation-results`, zlib-compressed, 16 GiB cap per volume.
- Host folders default to `D:\Docker\cache\segmentation-<prefix>-*`. Never point a cache into the WSL disk
  (`\\wsl.localhost\Ubuntu\...`): it lives on F:, which filled up on 2026-11-21.
- The container runs as uid 1001; a cache folder you create by hand must be writable by it.

## Server tests
```
docker run --rm --gpus all -v D:/src/git/Sam2Trainer:/repo -w /repo/SegmentationServer/segmentation_server --entrypoint bash viking-segmentation-server:pe-test -c "pip install -q --break-system-packages pytest pytest-asyncio hypothesis cryptography; python3 -m pytest -q -p no:cacheprovider tests"
```
Run it from PowerShell, or from Git Bash with `MSYS_NO_PATHCONV=1`. Without pytest-asyncio the async tests fail.
Baseline: 4 `test_predict_options` failures that predate the move from Viking.
