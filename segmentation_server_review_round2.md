---
name: Segmentation server review, round 2
overview: Second review of the segmentation server after the round-1 implementation (388 tests green). Six reviewers read the new code in chunks; I then re-checked every High claim against the code and, where possible, by running it. Two of the top-ranked reviewer findings did not survive verification and are listed under "Rejected". What remains is a short list of real defects (a stream DoS, a leftover error-text leak, an ignored request flag, a cache race), then operations/doc fixes, then small growth-walk and test-hygiene items. Nothing here needs a design change; one item needs your decision (gRPC authentication).
todos:
  - id: r2-stream-progress
    content: "SegmentTilesStream: make the answer timeout progress-based. Only reset the idle clock when an answer retires at least one outstanding tile; cap total messages per wait; test with an endless stream of empty / wrong-tile TilesAnswers."
    status: pending
  - id: r2-stream-cap
    content: "Cap concurrent tile streams (SEGMENTATION_MAX_CONCURRENT_STREAMS) with a semaphore -> RESOURCE_EXHAUSTED; test."
    status: pending
  - id: r2-error-text
    content: "Stop echoing exception text to clients from UploadImage, UploadTile, DeleteImage, stream INTERNAL and the decode error; log server-side, return fixed messages. Test that a raising predictor/decoder does not leak its message."
    status: pending
  - id: r2-omit-labeled
    content: "Honor omit_labeled_image in SegmentImage and MultiSegmentImage (only the stream and the helper do today); tests for both."
    status: pending
  - id: r2-upload-singleflight
    content: "ImageCache.upload_tile: single-flight per tile key so two concurrent uploads of the same key do not both encode and orphan the first entry; test with gather()."
    status: pending
  - id: r2-decode-pool
    content: "Bounded dedicated executor for decode and response encoding instead of the unbounded default executor; validate cheap request fields before taking cache pins."
    status: pending
  - id: r2-ops-docs
    content: "README --cache-max-images default 8 -> 32; start.sh PyCharm branch drops '$@'; healthcheck honours the port the server actually listens on and dials the bind address; compose only publishes 40444 when the demo is on; remove or fix Example/setup.py; fullchain fallback in healthcheck."
    status: pending
  - id: r2-lifecycle
    content: "create_server cleans up both executors and the grpc server if SegmentationServicer() raises; --generate-grpc failure and bind failure exit non-zero (2 for bind); install signal handlers before model load; single shutdown call on drain timeout; dev_cert key file 0600; TLS minimum version."
    status: pending
  - id: r2-test-hygiene
    content: "test_eager_compile.py must not leave MagicMock torch/sam2 in sys.modules (scope it with a fixture/monkeypatch); make the SegmentImageSets test context raise on abort like grpc.aio does; add pytest markers gpu/integration; fix Example/test_service.py stdout pipe."
    status: pending
  - id: r2-growth-small
    content: "GrowthWalk: mark a margin-owned cell blocked by max_cells so it is not dequeued repeatedly; skip _follow_contacts enqueue when no novel seeds; align _already_seen_by with window_box clipping; make window_box tie-break order-independent; remove unused SeamGraph.drop_cell (or call it when a shared tile is replaced); add the missing walk tests."
    status: pending
  - id: r2-model-small
    content: "Model/caches: run the compiled parity forward under _gpu_lock; validate shapes and orig_hw when loading a disk embedding blob; make flush_stale re-runnable; cap queued predictor builds."
    status: pending
  - id: r2-client
    content: "Viking client (VikingLegacy-Maintenance first, mirror to VikingLegacy): readable messages for DEADLINE_EXCEEDED and RESOURCE_EXHAUSTED from tile streams. Legacy endpoint remap for http://host with no port is a question for you (see decisions)."
    status: pending
isProject: false
---

# Segmentation server review, round 2

## How this was done

Six explore reviewers read the code as it stands after round 1 (`server.py` 1744 non-blank lines, `tile_growth.py` 767, and so on), each with its own chunk. I did not accept their severity ratings. For every claim rated High, and for the ones that would change this ordering, I re-read the code and, for the growth-walk claim, ran a reproduction against the test `World`.

Result: 3 of the top claims are real, 3 are not. The "Rejected" section is as useful as the plan, because it stops these coming back next round.

## Priority order

### P0 - real defects with a cheap fix

1. **Stream wait never times out under a chatty client** (`_receive_tiles`, `server.py` ~1445). Verified by reading: the loop is `while outstanding:` and each `_next_message(..., timeout)` call returns as soon as any message arrives. A client that keeps sending empty or wrong-tile `TilesAnswer`s resets the idle clock every time, so the RPC, its cache pins and its load-tracker slot are held indefinitely. The fix is a progress-based clock: reset only when an answer retires an outstanding tile, plus a cap on messages per wait.
2. **No cap on concurrent tile streams.** Each stream can pin cache entries and queue inference. Add a semaphore (`SEGMENTATION_MAX_CONCURRENT_STREAMS`) that answers RESOURCE_EXHAUSTED when full. Together with item 1 this is the only way I found for one client to starve the server.
3. **Exception text still returned to clients** on the paths I did not touch in round 1: `UploadImage` (~929), `DeleteImage` (~950), `UploadTile` (~1049), stream INTERNAL (~1180) and the decode error (~718, which includes the PIL message). Log with `logger.exception`, return a fixed string.
4. **`omit_labeled_image` is ignored by `SegmentImage` and `MultiSegmentImage`.** Verified by grep: only the helper and the stream pass it. The proto documents it as skipping the full-frame PNG, so unary callers pay the encode and the bytes for nothing.
5. **Concurrent `upload_tile` for the same key.** `_upload_tile_body` drops the lock between "delete the stale entry" and "index the new one". Two simultaneous uploads (two clients, or one client's retry) both miss, both encode, and the first entry loses its index slot but stays in `_cache` until LRU eviction. Single-flight per key fixes it and also saves a duplicate encoder pass.

### P1 - operations, docs, lifecycle

6. Unbounded default executor for decode and response encoding; and `UploadTile`/stream take cache pins before cheap validation, so a malformed request briefly holds a pin.
7. `README.md` line 214 says `--cache-max-images` defaults to 8; the code (`DEFAULT_MAX_ENTRIES`) and line 127 say 32. `start.sh`'s PyCharm branch drops `"$@"`, so flags such as `--tls-port` are silently ignored there. Healthcheck reads only `SEGMENTATION_TLS_PORT`, so a server started with the CLI flag is reported unhealthy; it also dials `localhost` while the server binds `[::]`. Compose publishes 40444 even with the demo off. `Example/setup.py` contradicts the pyproject-only story.
8. `create_server` leaks both executors and the grpc server when `SegmentationServicer()` raises (model load). The process exits anyway, so this is hygiene, but it is a three-line try/except. Also: `--generate-grpc` failure exits 0; a port bind failure should exit 2 like the TLS errors; signal handlers are installed only after the model loads; on a drain timeout the executor gets both `shutdown(wait=True)` and `shutdown(cancel_futures=True)` from different tasks; `dev_cert.py` writes the private key world-readable; no minimum TLS version.
9. Test hygiene: `test_eager_compile.py` puts a MagicMock `torch`/`sam2` into `sys.modules` at import when torch is absent. I saw this round 1: `test_predict_options.py` ran against the mock. Scope it to a fixture. The `SegmentImageSets` test context returns from `abort`, whereas `grpc.aio` raises; make it raise so tests cannot hide control-flow mistakes. Add `gpu`/`integration` markers.
10. Viking client shows a generic failure for `DEADLINE_EXCEEDED` and `RESOURCE_EXHAUSTED` from the stream; give them readable messages.

### P2 - small, measure or batch with other work

11. Growth walk: `max_cells` blocks a margin-owned cell without marking it, so it is dequeued again; `_follow_contacts` enqueues neighbours even when no novel seeds exist; `_already_seen_by` compares against unclipped boxes while seeds use clipped `window_box` (one redundant predict near mosaic edges); `window_box` tie-break depends on input order; `Canvas.core()` copies per probe; `SeamGraph.drop_cell` is unused. None of these has produced a wrong mask. They cost predictions or CPU, and the walk of 29 predictions took about 0.4 s in the earlier profile, so do them as cleanup, not as optimization.
12. Model: compiled parity forward runs outside `_gpu_lock` (benign today because the swap happens before the first use of the new module, but cheap to fix); disk blob load does not validate shapes/`orig_hw`; `flush_stale` is one-shot; no bound on queued predictor builds.
13. Tests the reviewers found missing and I agree with: per-cell prediction cap; `band_depths(rows=...)` at 0 and 511; rejected-then-widened range; `create_server` failure cleanup; SIGTERM-to-drain integration; `install_stop_handlers`; `run()` exit code 2; `SegmentImageSets` with mixed `image_id`.

## Rejected after verification

- **"Identical-bytes `upload_tile` hit bypasses the encoder-generation check, so a stale eager predictor lives forever" (reviewer High).** `upload_image` already checks `_is_stale` under the same lock hold that attaches the predictor and rebuilds once. A predictor that finishes after the swap is rebuilt there, and one attached before it is dropped by `flush_stale`. Both interleavings end compiled. No change.
- **"`RpcCancelled` is caught by `except Exception` and returned as INTERNAL" (reviewer High).** The cancel event is only set inside the handler's own `except asyncio.CancelledError`, so `RpcCancelled` can only be raised after the handler task has already been cancelled. Nothing is left to turn it into INTERNAL. No change.
- **"`SegmentImageSets` yields an empty OK response after `_dispatch_segmentation` aborts" (reviewer High).** In production `context.abort` raises `AbortError`, so the generator ends there. It only happens in the test double, which is why it is in item 9 as a test fix.
- **"Foreground click whose cell was edge-predicted but not covered never gets a user prompt" (reviewer High).** I reproduced the setup (a second click on a separate object in a cell the walk reaches by an edge). The edge attempt already includes every foreground click in the window as a positive, and the selection rule prefers the mask covering most foreground points, so the click ended up in the canvas. The one case left is the model refusing to cover the click, in which case a repeat prompt without the box would be near-identical. I would not change the skip.
- **Peak VRAM of two model copies during the compile swap.** You said resources are not a concern. Left alone.
- **Parity check uses `forward_image` on random CHW instead of the `set_image` path.** `set_image` calls `forward_image` on the same module, so the compiled graph under test is the same one. I will confirm on a real GPU, which I still have not been able to run.
- **Cancel does not interrupt an in-flight SAM2 call in the walk; mid-stream replacement of a shared tile leaves the session on the old bytes/embedding.** Both are bounded (walk budgets; each tile's bytes and embedding stay consistent with each other). Worth a sentence in `README.md`, not code.
- **Hole-fill 4-connectivity, `RETR_CCOMP` two-level nesting, greedy `find_circles`.** Corner cases with no observed bad output; I would not chase them (you said the same about corners earlier).

## Decisions I need from you

1. **gRPC authentication.** Round 1 made the endpoint TLS-only and added a token to the demo site, but the gRPC service itself has no authentication: anyone who can reach port 40443 can upload and segment. Options: (a) leave it, relying on the network boundary; (b) optional shared bearer token in metadata, enforced by a server interceptor when `SEGMENTATION_AUTH_TOKEN` is set, with Viking sending it from settings (small client change in both trees). I recommend (b), off by default.
2. **`http://host` with no port.** The client only remaps the legacy port 40080 to 40443. The old compose mapping was 40080:80, so a user on an implicit port 80 is probably behind a reverse proxy of their own. I would leave this unremapped and let the TLS failure message name the fix. Say if you want it remapped.

## Notes on process

- Nothing from round 1 is committed. The proto change is inside the `gRPC_Protos` submodule and needs its own commit and a pointer bump in the parent, per the submodule convention.
- Still unexercised, and not covered by any test here: real SAM2 on a GPU (parity thresholds, the centre click, box-only prompts), the Docker image build and compose file, and the .NET test host (client was build-verified only).
