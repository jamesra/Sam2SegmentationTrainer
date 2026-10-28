"""``sam2-em-score-cards``: score gallery cards with a checkpoint exactly as a training run's validation does.

A card is one crop window of a location (``(location_id, image_key)``). Every card is scored by the run's own
validation suite (:func:`train.build_validation_suite`), restricted to one family (``circle_annotator`` by default,
the family that selects checkpoints): eval mode, no augmentation, the family's prompt seeded from the run seed and
the card, the run's ``config.yaml`` and the model's prompt-encoding record. So a card's IoU, core recall and spill
are the per-sample numbers behind the run's ``val_full/*`` figures, and pooled over the validation split they
reproduce them. The results go to each volume's catalog (:mod:`annotation_crops.card_scores`).

Three ways to run it::

    sam2-em-score-cards --checkpoint best:sam2_em_v5 --volumes RC1 --locations 1234,5678   # some locations
    sam2-em-score-cards --checkpoint best:sam2_em_v5 --all [--split val] [--skip-scored]    # every card
    sam2-em-score-cards --checkpoint best:sam2_em_v5 --serve                                # the gallery's queue

``--checkpoint`` is a weights file, ``best:RUN`` (the run's ``best_model.json`` epoch, looked up again before each
request, so the service follows the trainer) or ``latest:RUN`` (its newest ``serve_epoch*.pt``).
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from annotation_crops import card_scores as store
from sam2_segmentation_trainer.paths import annotation_crops_root, run_dir_for

_SERVE_RE = re.compile(r"^serve_epoch(\d+)\.pt$")


def resolve_checkpoint(spec: str, *, ckpt_root: Path | None = None) -> Path:
    """A weights file from ``spec``: a path, ``best:RUN`` or ``latest:RUN`` (see the module docstring)."""
    kind, _, run = spec.partition(":")
    if kind not in ("best", "latest") or not run:
        path = Path(spec)
        if not path.is_file():
            raise FileNotFoundError(f"no checkpoint {path}")
        return path
    run_dir = run_dir_for(run, ckpt_root=ckpt_root)
    serve = run_dir / "serve"
    if kind == "best":
        best = json.loads((run_dir / "best_model.json").read_text(encoding="utf-8"))
        path = serve / f"serve_epoch{int(best['epoch'])}.pt"
    else:
        epochs = sorted(
            (int(m.group(1)), p) for p in serve.glob("serve_epoch*.pt") if (m := _SERVE_RE.match(p.name))
        )
        if not epochs:
            raise FileNotFoundError(f"no serve_epoch*.pt in {serve}")
        path = epochs[-1][1]
    if not path.is_file():
        raise FileNotFoundError(f"{spec}: {path} is missing (not exported yet?)")
    return path


def model_label(path: Path) -> str:
    """The model's version from its ``<stem>.model.json`` sidecar (``v5.4``), else the file name."""
    sidecar = path.with_suffix(".model.json")
    try:
        info = json.loads(sidecar.read_text(encoding="utf-8"))
        version = (info.get("model") or {}).get("model_version")
        if version:
            return str(version)
    except (OSError, ValueError):
        pass
    return path.name


def run_dir_of(path: Path) -> Path | None:
    """The run folder of an exported ``<run>/serve/serve_epochK.pt``, when it has the run's ``config.yaml``."""
    run_dir = path.parent.parent if path.parent.name == "serve" else path.parent
    return run_dir if (run_dir / "config.yaml").is_file() else None


def scoring_cfg(path: Path, family: str, overrides: Sequence[str] = ()):
    """The config the card is scored under: the defaults, then the run's ``config.yaml`` (what the run validated
    with), then the model's prompt-encoding record and ``overrides``, with validation narrowed to ``family``."""
    from omegaconf import OmegaConf

    from sam2_segmentation_trainer import train
    from sam2_segmentation_trainer.encoding_record import read_record, resolve_encoding

    cfg = train.load_train_cfg()
    run_dir = run_dir_of(path)
    if run_dir is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(run_dir / "config.yaml"))
    record, source = read_record(path)
    resolved = resolve_encoding(record, source=source)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(resolved.cfg_overrides()) + list(overrides)))
    OmegaConf.update(cfg, "eval.val_prompts", [family], force_add=True)
    OmegaConf.update(cfg, "eval.primary_val_prompt", family, force_add=True)
    return cfg


@dataclass
class _Inventory:
    examples: list
    tiles: Any
    stamp: float


@dataclass
class CardScorer:
    """Loads the checkpoint named by ``spec`` on first use (again when ``best:RUN`` moves to another epoch), keeps
    each volume's inventory and tile context until its manifest changes, and scores cards."""

    spec: str
    family: str = store.DEFAULT_FAMILY
    root: Path | None = None
    num_workers: int = 2
    batch_size: int | None = None
    overrides: tuple[str, ...] = ()
    path: Path | None = None
    label: str = ""
    epoch: int | None = None
    cfg: Any = None
    device_name: str | None = None
    """"cpu" or "cuda"; None: the GPU when there is one."""
    model: Any = None
    parked: Any = None
    device: Any = None
    last_used: float = 0.0
    _inventories: dict[str, _Inventory] = field(default_factory=dict)

    def ensure_config(self) -> None:
        """Resolve ``spec`` and the config it is scored under, without loading the model (a ``best:RUN`` that moved
        to another epoch also drops the loaded model)."""
        from sam2_segmentation_trainer.data.prompts import PromptConfig

        path = resolve_checkpoint(self.spec)
        if self.cfg is not None and path == self.path:
            return
        self.unload()
        cfg = scoring_cfg(path, self.family, self.overrides)
        if self.batch_size is not None:
            cfg.val.batch_size = int(self.batch_size)
        self.prompt_cfg = PromptConfig.from_cfg(cfg)
        if path != self.path:
            self._inventories.clear()  # another run's config may draw circles differently
        self.cfg, self.path, self.label = cfg, path, model_label(path)
        match = _SERVE_RE.match(path.name)
        self.epoch = int(match.group(1)) if match else None
        if self.root is None:
            from sam2_segmentation_trainer.paths import data_root

            self.root = data_root(cfg.data.get("root"))

    def ensure_model(self) -> None:
        """Load (or reload) the model ``spec`` points at now."""
        import torch

        from sam2_segmentation_trainer.export_serve import load_finetuned_model
        from sam2_segmentation_trainer.paths import pretrained_checkpoint

        self.ensure_config()
        wanted = self.device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(wanted)
        if self.model is None and self.parked is not None:
            started = time.perf_counter()
            self.model = self.parked.to(self.device)  # back from CPU memory: no reload from the share
            self.parked = None
            print(f"model back on {self.device} in {time.perf_counter() - started:.1f} s", flush=True)
        if self.model is None:
            print(f"loading {self.path} ({self.label}) on {self.device}", flush=True)
            self.model = load_finetuned_model(self.path, pretrained_checkpoint(), str(self.cfg.model.config),
                                              str(self.device))
        self.last_used = time.monotonic()

    def unload(self, *, park: bool = False) -> None:
        """Free the GPU memory the model holds. ``park`` keeps it in CPU memory (the idle service does this, so the
        next request moves it back in a moment instead of reloading it); otherwise it is dropped."""
        import torch

        if self.model is not None and park and self.device is not None and self.device.type == "cuda":
            self.parked = self.model.to("cpu")
            print("model parked in CPU memory", flush=True)
        elif self.model is not None or self.parked is not None:
            self.parked = None
            print("model unloaded", flush=True)
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def catalog_path(self, volume: str) -> Path:
        return annotation_crops_root(volume, self.root) / "annotation_crops.sqlite"

    def inventory(self, volume: str) -> _Inventory:
        """The volume's usable examples and its tile context (neighbours, circles, adjacent shapes), as training
        builds them; rebuilt when the manifest changes (a gallery refresh re-exports a location)."""
        from sam2_segmentation_trainer import train
        from sam2_segmentation_trainer.data.manifest import index_volumes, take_model_sized
        from sam2_segmentation_trainer.data.tiles import build_tile_context

        manifest = annotation_crops_root(volume, self.root) / "manifest.jsonl"
        stamp = manifest.stat().st_mtime
        cached = self._inventories.get(volume)
        if cached is not None and cached.stamp == stamp:
            return cached
        started = time.perf_counter()
        examples = index_volumes(volumes=[volume], root=self.root, skip_missing=True)
        examples, _wrong_size = take_model_sized(examples, int(self.cfg.data.image_size))
        tiles = build_tile_context(examples, **train._circle_guard(self.cfg, self.prompt_cfg))
        inventory = _Inventory(examples=list(examples), tiles=tiles, stamp=stamp)
        self._inventories[volume] = inventory
        print(f"{volume}: {len(examples)} cards indexed in {time.perf_counter() - started:.1f} s", flush=True)
        return inventory

    def score_examples(self, volume: str, targets: Sequence, keep: list | None = None) -> list[store.CardScore]:
        """Score ``targets`` (examples of ``volume``) with the validation suite; one score per card. ``keep`` collects
        the raw validation records too (for :func:`validation.summarize_mode`)."""
        from sam2_segmentation_trainer import train

        if not targets:
            return []
        self.ensure_model()
        inventory = self.inventory(volume)
        workers = 0 if len(targets) < 16 else self.num_workers
        suite = train.build_validation_suite(
            self.cfg, list(targets), inventory.tiles, approved=None, gold=None, prompt_cfg=self.prompt_cfg,
            num_workers=workers,
        )
        threshold = float(self.cfg.data.mask_threshold)
        outcome = suite.run(self.model, self.device, int(self.cfg.data.image_size), threshold, full=True, keep_records=True)
        scored_at = store.now_iso()
        self.last_used = time.monotonic()
        records = (outcome.records or {}).get(self.family, [])
        if keep is not None:
            keep.extend(records)
        return [
            store.CardScore.from_record(r, family=self.family, model=self.label, scored_at=scored_at, epoch=self.epoch)
            for r in records
        ]

    def targets(self, volume: str, location_ids: Sequence[int] | None = None, *, split: str = "all",
                skip_scored: bool = False) -> list:
        """The examples to score: every card, or those of ``location_ids``; ``split`` "val"/"train" keeps one side of
        the run's ``split.json``, "subset" the run's per-epoch validation subset; ``skip_scored`` drops cards this model
        has already scored."""
        self.ensure_config()
        examples = self.inventory(volume).examples
        if location_ids is not None:
            wanted = {int(i) for i in location_ids}
            examples = [e for e in examples if int(e.location_id) in wanted]
        if split != "all":
            from sam2_segmentation_trainer import train
            from sam2_segmentation_trainer.data.splits import load_split
            from sam2_segmentation_trainer.data.valsets import partition_with_gold
            from sam2_segmentation_trainer.validation import select_validation_subset

            run_dir = run_dir_of(self.path)
            if run_dir is None or not (run_dir / "split.json").is_file():
                raise FileNotFoundError(f"--split {split} needs the run's split.json beside {self.path}")
            gold = train.load_gold_set(self.cfg)
            train_ex, val_ex = partition_with_gold(gold, *load_split(run_dir / "split.json").partition(examples))
            examples = train_ex if split == "train" else val_ex
            if split == "subset":
                approved = train.load_approved_set(self.cfg, self.root)
                examples = select_validation_subset(
                    val_ex, float(self.cfg.val.epoch_subset), int(self.cfg.training.seed), approved, gold
                )
        if skip_scored:
            with sqlite3.connect(str(self.catalog_path(volume))) as connection:
                store.ensure_card_scores_schema(connection)
                done = {
                    (int(r[0]), str(r[1]))
                    for r in connection.execute(
                        "SELECT location_id, image_key FROM card_scores WHERE model = ? AND family = ?",
                        (self.label, self.family),
                    )
                }
            examples = [e for e in examples if (int(e.location_id), str(e.image_key)) not in done]
        return list(examples)

    def write(self, volume: str, scores: Sequence[store.CardScore]) -> int:
        connection = sqlite3.connect(str(self.catalog_path(volume)), timeout=30.0)
        try:
            connection.execute("PRAGMA busy_timeout=30000")
            store.ensure_card_scores_schema(connection)
            written = store.write_card_scores(connection, scores)
            connection.commit()
            return written
        finally:
            connection.close()


def score_bulk(scorer: CardScorer, volumes: Sequence[str], *, location_ids: Sequence[int] | None, split: str,
               skip_scored: bool, chunk: int, limit: int | None, summary: bool = False) -> int:
    """Score the selected cards of each volume in chunks, writing each chunk as it finishes (an interrupted run
    keeps what it scored; ``skip_scored`` resumes it). Returns the number of cards written. ``summary`` prints the
    pooled figures of everything scored, computed as a training run computes its ``val_*`` numbers."""
    total = 0
    kept: list | None = [] if summary else None
    for volume in volumes:
        targets = scorer.targets(volume, location_ids, split=split, skip_scored=skip_scored)
        if limit is not None:
            targets = targets[: int(limit)]
        print(f"{volume}: scoring {len(targets)} cards with {scorer.label} ({scorer.family})", flush=True)
        started = time.perf_counter()
        for begin in range(0, len(targets), max(1, int(chunk))):
            part = targets[begin : begin + max(1, int(chunk))]
            written = scorer.write(volume, scorer.score_examples(volume, part, keep=kept))
            total += written
            done = begin + len(part)
            rate = done / max(time.perf_counter() - started, 1e-6)
            print(f"{volume}: {done}/{len(targets)} cards, {rate:.2f} cards/s", flush=True)
    if kept:
        from sam2_segmentation_trainer.validation import summarize_mode

        result = summarize_mode(kept, target=float(scorer.cfg.val.get("core_recall_target", 0.975)))
        print(f"pooled over {len(kept)} cards ({scorer.family}, {scorer.label}): {result}", flush=True)
    return total


def serve(scorer: CardScorer, volumes: Sequence[str], *, poll_s: float, idle_unload_s: float) -> None:
    """Score the locations the gallery queues, forever. A request is claimed, every card of its location scored
    and written, then marked done (or failed with the error).

    On the CPU (the default for the service) the model stays loaded and the process runs at low priority, so it never
    takes GPU memory from training and yields the CPU to it; its scores match the GPU's within float rounding. On the
    GPU the model is parked in CPU memory ``idle_unload_s`` after the last request."""
    import os

    scorer.ensure_config()
    on_cpu = scorer.device_name == "cpu"
    if on_cpu:
        try:
            os.nice(10)
        except OSError:
            pass
    paths = {v: scorer.catalog_path(v) for v in volumes}
    connections: dict[str, sqlite3.Connection] = {}
    for volume, path in paths.items():
        connection = sqlite3.connect(str(path), timeout=30.0, isolation_level=None)
        connection.execute("PRAGMA busy_timeout=30000")
        store.ensure_card_scores_schema(connection)
        requeued = store.requeue_running(connection)
        if requeued:
            print(f"{volume}: requeued {requeued} interrupted request(s)", flush=True)
        connections[volume] = connection
    for volume in volumes:  # CPU only: so a volume's first request does not wait for its index
        scorer.inventory(volume)
    if on_cpu:
        scorer.ensure_model()  # costs no GPU memory, so the first request does not wait for the load
    print(f"serving re-score requests for {', '.join(volumes)} with {scorer.spec} ({scorer.label})", flush=True)
    last_beat = 0.0
    while True:
        busy = False
        if time.monotonic() - last_beat >= 10.0:
            # The gallery queues a re-score only where a scorer is alive (store.scorer_alive).
            for connection in connections.values():
                try:
                    store.beat(connection, spec=scorer.spec, model=scorer.label or None)
                except sqlite3.Error as exc:
                    print(f"heartbeat not written: {exc}", flush=True)
            last_beat = time.monotonic()
        for volume, connection in connections.items():
            for request_id, location_id in store.claim_requests(connection):
                busy = True
                started = time.perf_counter()
                try:
                    scorer.ensure_model()
                    scores = scorer.score_examples(volume, scorer.targets(volume, [location_id]))
                    scorer.write(volume, scores)
                    store.finish_request(connection, request_id, model=scorer.label, cards=len(scores))
                    print(f"{volume} location {location_id}: {len(scores)} card(s) with {scorer.label} in "
                          f"{time.perf_counter() - started:.1f} s", flush=True)
                except Exception as exc:  # one bad location must not stop the service
                    traceback.print_exc()
                    store.finish_request(connection, request_id, model=scorer.label or None, cards=0,
                                         error=f"{type(exc).__name__}: {exc}"[:500])
        if not busy:
            # A busy pass is followed at once by another (requests queued meanwhile reuse the model on the GPU);
            # the first pass that finds nothing parks it, so training gets the GPU back right away.
            if not on_cpu and scorer.model is not None and time.monotonic() - scorer.last_used >= idle_unload_s:
                scorer.unload(park=True)
            time.sleep(poll_s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score gallery cards as validation does (IoU, core recall, spill)")
    parser.add_argument("--checkpoint", required=True, help="Weights file, best:RUN or latest:RUN")
    parser.add_argument("--volumes", default=None, help="Comma-separated volumes (default: the run's data.volumes)")
    parser.add_argument("--root", type=Path, default=None, help="Data root (default: the run's data.root)")
    parser.add_argument("--family", default=store.DEFAULT_FAMILY, help="Validation family to score with")
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--locations", default=None, help="Comma-separated location ids (every card of each)")
    what.add_argument("--all", action="store_true", help="Every card of the volumes")
    what.add_argument("--serve", action="store_true", help="Score the gallery's re-score queue until stopped")
    parser.add_argument("--split", choices=("all", "val", "train", "subset"), default="all",
                        help="Only one side of the run's split.json (val reproduces the run's val_full figures, "
                             "subset its per-epoch val_subset figures)")
    parser.add_argument("--summary", action="store_true", help="Print the pooled figures of the cards scored")
    parser.add_argument("--skip-scored", action="store_true", help="Skip cards this model already scored")
    parser.add_argument("--limit", type=int, default=None, help="At most this many cards per volume (a sample)")
    parser.add_argument("--chunk", type=int, default=2000, help="Cards written per chunk")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1, help="Images per batch (val.batch_size); 1 fits beside training")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Config override key=value")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--idle-unload-seconds", type=float, default=0.0,
                        help="Free the model's GPU memory this long after the last request (--serve; 0: as soon as "
                             "no request waits). The model waits in CPU memory for the next one")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="auto: the CPU for --serve (never takes GPU memory from training), else the GPU")
    args = parser.parse_args(argv)

    device = args.device if args.device != "auto" else ("cpu" if args.serve else None)
    scorer = CardScorer(
        spec=args.checkpoint, family=args.family, root=args.root, num_workers=args.num_workers,
        batch_size=args.batch_size, overrides=tuple(args.overrides), device_name=device,
    )
    scorer.ensure_config()
    volumes = [v.strip() for v in args.volumes.split(",") if v.strip()] if args.volumes else \
        [str(v) for v in scorer.cfg.data.volumes]
    if args.serve:
        serve(scorer, volumes, poll_s=args.poll_seconds, idle_unload_s=args.idle_unload_seconds)
        return 0
    location_ids = None if args.all else [int(i) for i in args.locations.split(",") if i.strip()]
    written = score_bulk(scorer, volumes, location_ids=location_ids, split=args.split, skip_scored=args.skip_scored,
                         chunk=args.chunk, limit=args.limit, summary=args.summary)
    print(f"wrote {written} card score(s)", flush=True)
    return 0


# ---- written by a training run ----------------------------------------------------------------------------------


def run_model_label(run_name: str, epoch: int) -> str:
    """``v5.6`` for epoch 6 of ``sam2_em_v5`` (the name its serve export gets); ``<run>.<epoch>`` otherwise."""
    from sam2_segmentation_trainer.model_manifest import model_id_for_run, model_version_name

    model_id = model_id_for_run(str(run_name))
    return model_version_name(model_id, epoch) if model_id else f"{run_name}.{int(epoch)}"


def write_scores_by_volume(data_dir: Path, by_volume: dict[str, list[store.CardScore]]) -> int:
    """Append each volume's scores to its catalog; a volume without a catalog, or a locked one, is skipped (scores
    are best-effort and must never stop training). Returns the number written."""
    written = 0
    for volume, scores in by_volume.items():
        if not scores:
            continue
        path = annotation_crops_root(volume, data_dir) / "annotation_crops.sqlite"
        if not path.is_file():
            continue
        try:
            connection = sqlite3.connect(str(path), timeout=10.0)
            try:
                connection.execute("PRAGMA busy_timeout=10000")
                store.ensure_card_scores_schema(connection)
                written += store.write_card_scores(connection, scores)
                connection.commit()
            finally:
                connection.close()
        except sqlite3.Error as exc:
            print(f"card_scores write skipped for {volume}: {exc}", flush=True)
    return written


def record_validation_card_scores(
    data_dir: Path, records: dict[str, list[dict]] | None, *, family: str, model: str, epoch: int
) -> int:
    """Write the ``family`` records of an epoch's validation pass as each card's validation score."""
    rows = (records or {}).get(family) or []
    scored_at = store.now_iso()
    by_volume: dict[str, list[store.CardScore]] = {}
    for record in rows:
        volume = str(record["key"][0])
        by_volume.setdefault(volume, []).append(
            store.CardScore.from_record(record, family=family, model=model, scored_at=scored_at, epoch=epoch)
        )
    return write_scores_by_volume(data_dir, by_volume)


class TrainPassScores:
    """Per-card scores measured on the training pass, as each card is trained (``family == "train"``).

    Each step hands over its prediction (``submit``): the thresholded masks are copied to pinned host memory without
    waiting for the GPU. At each loss flush (``flush``, after the loss window's own transfer has synchronised the
    GPU) the queued masks are scored in a thread pool with the validation code (:func:`validation.score_sample`),
    the prompt variants of one card pooled into one score, and the scores finished so far are written. Scoring is
    best-effort: when the pool falls behind, steps are skipped (``skipped``) rather than slowing training.
    """

    def __init__(self, data_dir: Path, *, run_name: str, threshold: float, band: float, threads: int = 2,
                 max_backlog: int = 64) -> None:
        import math
        from concurrent.futures import ThreadPoolExecutor

        self.data_dir = Path(data_dir)
        self.run_name = str(run_name)
        threshold = min(max(float(threshold), 1e-6), 1.0 - 1e-6)
        self.logit_threshold = math.log(threshold / (1.0 - threshold))
        self.band = float(band)
        self.max_backlog = int(max_backlog)
        self.pool = ThreadPoolExecutor(max_workers=max(1, int(threads)), thread_name_prefix="card-score")
        self.queued: list[tuple] = []
        self.running: list = []
        self.skipped = 0
        self.written = 0

    def submit(self, pred, batch: dict, epoch: int) -> None:
        """Queue one step's prediction (``pred``: logits ``[V, H, W]``, one row per prompt variant)."""
        import torch

        if len(self.queued) + len(self.running) >= self.max_backlog:
            self.skipped += 1
            return
        meta = batch.get("meta") or []
        if not meta:
            return
        masks = (pred.detach().reshape(pred.shape[0], *pred.shape[-2:]) > self.logit_threshold)
        host = torch.empty(masks.shape, dtype=torch.bool, pin_memory=torch.cuda.is_available())
        host.copy_(masks, non_blocking=True)
        index = batch["variant_index"]
        self.queued.append((
            host,
            batch["mask"],
            index.tolist() if hasattr(index, "tolist") else list(index),
            list(meta),
            list(batch.get("prompt_type") or []),
            batch.get("edge_weight"),
            batch.get("neighbor_core"),
            int(epoch),
        ))

    def _score_step(self, item: tuple) -> list[tuple[str, store.CardScore]]:
        from sam2_segmentation_trainer.validation import score_sample

        host, gt, index, meta, prompt_types, edge, cores, epoch = item
        preds = host.numpy()
        by_card: dict[tuple[str, int, str], tuple[list[dict], list[str]]] = {}
        for row, image in enumerate(index):
            if row >= len(meta) or row >= preds.shape[0]:
                break
            item_meta = meta[row] or {}
            volume = str(item_meta.get("volume", "")).strip()
            location_id = item_meta.get("location_id")
            image_key = str(item_meta.get("image_key") or "")
            if not volume or location_id is None or not image_key:
                continue
            prompt = prompt_types[row] if row < len(prompt_types) else ""
            record = score_sample(
                preds[row],
                gt[image].numpy() > 0,
                item_meta,
                weight_np=None if edge is None else edge[image].float().numpy(),
                neighbor_core=None if cores is None else cores[image].numpy(),
                input_np=None,
                mode=prompt or None,
                band=self.band,
            )
            records, prompts = by_card.setdefault((volume, int(location_id), image_key), ([], []))
            records.append(record)
            if prompt:
                prompts.append(str(prompt))
        scored_at = store.now_iso()
        model = run_model_label(self.run_name, epoch)
        return [
            (volume, store.CardScore.pooled(records, family=store.TRAIN_FAMILY, model=model, scored_at=scored_at,
                                            epoch=epoch, prompt=",".join(sorted(set(prompts))) or None))
            for (volume, _location_id, _image_key), (records, prompts) in by_card.items()
        ]

    def flush(self, *, wait: bool = False) -> int:
        """Start scoring the queued steps (their copies are complete once the GPU has been synchronised, as the loss
        flush does) and write the scores that are finished; ``wait`` finishes everything (end of the run).
        Returns the number of scores written."""
        for item in self.queued:
            self.running.append(self.pool.submit(self._score_step, item))
        self.queued.clear()
        done = [f for f in self.running if wait or f.done()]
        self.running = [f for f in self.running if f not in done]
        by_volume: dict[str, list[store.CardScore]] = {}
        for future in done:
            try:
                for volume, score in future.result():
                    by_volume.setdefault(volume, []).append(score)
            except Exception as exc:  # a bad sample must never stop training
                print(f"training-pass card score failed: {type(exc).__name__}: {exc}", flush=True)
        written = write_scores_by_volume(self.data_dir, by_volume)
        self.written += written
        return written

    def close(self) -> None:
        self.flush(wait=True)
        self.pool.shutdown(wait=True)


if __name__ == "__main__":
    sys.exit(main())
