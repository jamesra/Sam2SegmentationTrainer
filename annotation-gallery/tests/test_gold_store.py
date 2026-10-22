"""Gold sets: seeded sampling, the single-window rule, copying cards, id lists, replacements and progress."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import catalog  # noqa: E402
from gold import draw, store  # noqa: E402


def make_source(folder: Path, locations: int = 40, split: tuple = (), rejected: tuple = (), types: int = 3) -> Path:
    """A source crops folder: one window per location (two for ``split``), types cycling, z = location id % 100."""
    folder.mkdir(parents=True, exist_ok=True)
    connection = catalog.connect(folder)
    # Real catalogs key a row by (location, window); the fresh-catalog schema keys it by location alone.
    connection.execute("DROP TABLE locations")
    connection.execute(
        "CREATE TABLE locations (location_id INTEGER NOT NULL, z INTEGER NOT NULL, structure_id INTEGER, structure_label TEXT, "
        "type_id INTEGER, type_name TEXT, radius REAL, image_key TEXT NOT NULL, image_relpath TEXT, mask_relpath TEXT, "
        "ignored INTEGER NOT NULL DEFAULT 0, sam2PredIou REAL, sam2ObjectScore REAL, sam2Stability REAL, sam2GtIou REAL, "
        "sam2Checkpoint TEXT, sam2ScoredAt TEXT, PRIMARY KEY (location_id, image_key))"
    )
    for location_id in range(1, locations + 1):
        windows = 2 if location_id in split else 1
        for index in range(windows):
            key = f"V_{location_id % 100}_D1_X{index}-1024_Y0-1024"
            image = folder / "images" / f"{key}.png"
            mask = folder / "masks" / f"{key}_{location_id}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            mask.parent.mkdir(parents=True, exist_ok=True)
            if not image.exists():
                Image.fromarray(np.full((16, 16), 90, dtype=np.uint8)).save(image)
            pixels = np.zeros((16, 16), dtype=np.uint8)
            # Window 0 holds the whole mask; a second window holds only a sliver of it.
            pixels[2:2 + (location_id % 10 + 3 if index == 0 else 1), 2:10] = 255
            Image.fromarray(pixels).save(mask)
            connection.execute(
                "INSERT INTO locations (location_id, z, structure_id, structure_label, type_id, type_name, radius, image_key, "
                "image_relpath, mask_relpath, ignored) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [location_id, location_id % 100, location_id, "s", location_id % types, "T", 3.0, key,
                 f"images/{key}.png", f"masks/{key}_{location_id}.png", 1 if location_id in rejected else 0],
            )
    connection.commit()
    connection.close()
    return folder


# ---- drawing ----------------------------------------------------------------------------------------------------

def pool_of(folder: Path) -> draw.Pool:
    return draw.build_pool(catalog.list_catalog_rows(folder))


def test_pool_excludes_rejected_but_keeps_multi_window_locations(tmp_path: Path) -> None:
    pool = pool_of(make_source(tmp_path / "s", locations=20, split=(3, 4), rejected=(5,)))
    assert 5 not in pool.ids and pool.rejected == 1 and pool.multi == [3, 4]
    assert len(pool.ids) == 19 and len(pool.windows[3]) == 2 and len(pool.windows[6]) == 1


def test_a_location_with_several_windows_contributes_one_window_that_holds_its_mask(tmp_path: Path) -> None:
    source = make_source(tmp_path / "s", locations=20, split=(3,))
    pool = pool_of(source)
    from gold.store import _mask_pixels

    chosen = draw.choose_window(1, 3, pool.windows[3], _mask_pixels(source))
    assert chosen["image_key"].endswith("X0-1024_Y0-1024"), "the sliver window is not offered"
    assert draw.choose_window(1, 3, pool.windows[3], _mask_pixels(source)) == chosen
    only = pool.windows[6][0]
    assert draw.choose_window(1, 6, [only], _mask_pixels(source)) is only


def test_comparable_windows_are_chosen_between_by_the_seed() -> None:
    windows = [{"image_key": "a"}, {"image_key": "b"}, {"image_key": "c"}]
    picks = {draw.choose_window(seed, 9, windows, lambda row: 100)["image_key"] for seed in range(30)}
    assert picks == {"a", "b", "c"}
    assert draw.choose_window(4, 9, windows, lambda row: 100) == draw.choose_window(4, 9, windows, lambda row: 100)


def test_the_same_seed_gives_the_same_order_and_other_seeds_differ(tmp_path: Path) -> None:
    pool = pool_of(make_source(tmp_path / "s", locations=60))
    a, b = draw.order(pool, 7), draw.order(pool, 7)
    assert a == b and sorted(a) == sorted(pool.ids)
    assert draw.order(pool, 8) != a
    assert draw.sample(pool, 10, 7) == a[:10]


def test_stratified_order_keeps_each_strata_in_proportion(tmp_path: Path) -> None:
    pool = pool_of(make_source(tmp_path / "s", locations=90, types=3))
    sampled = draw.sample(pool, 30, 1, stratify="type")
    counts: Dict[int, int] = {}
    for location_id in sampled:
        type_id = pool.rows[location_id]["type_id"]
        counts[type_id] = counts.get(type_id, 0) + 1
    assert sorted(counts.values()) == [10, 10, 10]
    assert draw.sample(pool, 30, 1, stratify="type") == sampled


def test_z_bands_are_stratified(tmp_path: Path) -> None:
    pool = pool_of(make_source(tmp_path / "s", locations=80))
    sampled = draw.sample(pool, 8, 3, stratify="z", band=25)
    assert {pool.rows[i]["z"] // 25 for i in sampled} == {0, 1, 2, 3}


def test_bad_stratify_is_rejected() -> None:
    with pytest.raises(ValueError):
        draw.order(draw.Pool(), 1, "nonsense")


def test_next_unused_continues_the_same_order() -> None:
    ordered = [5, 3, 9, 1, 7]
    assert draw.next_unused(ordered, [5, 3], 2) == [9, 1]
    assert draw.next_unused(ordered, [5, 3, 9, 1, 7], 2) == []


def test_fingerprint_ignores_order_and_changes_with_content() -> None:
    assert draw.fingerprint([3, 1, 2]) == draw.fingerprint([1, 2, 3]) != draw.fingerprint([1, 2, 4])


def test_id_lists_accept_commas_lines_comments_and_report_junk() -> None:
    ids, bad = draw.parse_id_list("# my list\n12, 7 7\n\n99;5 # trailing\nabc 0 -4 3.5\n")
    assert ids == [12, 7, 99, 5] and bad == ["abc", "0", "-4", "3.5"]
    with pytest.raises(ValueError):
        draw.parse_id_list("1 2 3", limit=2)


# ---- the set ----------------------------------------------------------------------------------------------------

def test_creating_a_set_copies_cards_and_records_the_draw(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src" / "RC1", locations=40, split=(3,), rejected=(9,))
    root = tmp_path / "gold"
    meta = store.create_set(root, "first", "v3", {"RC1": source}, n=10, seed=42)

    assert meta["name"] == "gold-first" and meta["volumes"]["RC1"]["drawn"] == 10
    assert meta["volumes"]["RC1"]["multiWindow"] == 1 and meta["volumes"]["RC1"]["rejectedExcluded"] == 1
    crops = root / "gold-first" / "RC1"
    rows = catalog.list_catalog_rows(crops)
    assert len(rows) == 10 and sorted(r["location_id"] for r in rows) == sorted(meta["volumes"]["RC1"]["used"])
    for row in rows:
        assert (crops / row["mask_relpath"]).is_file() and (crops / row["image_relpath"]).is_file()
        source_mask = crops / "source_masks" / f"{row['image_key']}_{row['location_id']}.png"
        assert source_mask.read_bytes() == (crops / row["mask_relpath"]).read_bytes()
    assert store.read_meta(root / "gold-first")["seed"] == 42
    assert all(c["mask_sha1"] == c["source_mask_sha1"] and not c["edited"] for c in store.cards(crops))


def test_the_same_seed_reproduces_the_same_sample_in_a_new_set(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src" / "RC1", locations=50)
    one = store.create_set(tmp_path / "gold", "a", "v3", {"RC1": source}, n=12, seed=5)
    two = store.create_set(tmp_path / "gold", "b", "v3", {"RC1": source}, n=12, seed=5)
    assert one["volumes"]["RC1"]["used"] == two["volumes"]["RC1"]["used"]


def test_a_set_is_never_overwritten_and_a_failure_leaves_nothing(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src" / "RC1", locations=10)
    root = tmp_path / "gold"
    store.create_set(root, "x", "v3", {"RC1": source}, n=3, seed=1)
    with pytest.raises(store.GoldError, match="already exists"):
        store.create_set(root, "x", "v3", {"RC1": source}, n=3, seed=1)
    for mask in (source / "masks").glob("*.png"):
        mask.unlink()
    with pytest.raises(store.GoldError, match="missing"):
        store.create_set(root, "y", "v3", {"RC1": source}, n=3, seed=1)
    assert not (root / "gold-y").exists()


def test_bad_names_and_counts_are_refused(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src" / "RC1", locations=5)
    for name in ("../x", "a b", "", "x" * 60):
        with pytest.raises(store.GoldError):
            store.create_set(tmp_path / "gold", name, "v3", {"RC1": source}, n=2, seed=1)
    with pytest.raises(store.GoldError):
        store.create_set(tmp_path / "gold", "ok", "v3", {"RC1": source}, n=0, seed=1)


def test_a_small_volume_gives_what_it_has(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src" / "RC1", locations=7)
    meta = store.create_set(tmp_path / "gold", "small", "v3", {"RC1": source}, n=100, seed=1)
    assert meta["volumes"]["RC1"]["drawn"] == 7 and meta["volumes"]["RC1"]["short"] == 93


def test_two_volumes_are_drawn_independently(tmp_path: Path) -> None:
    sources = {"RC1": make_source(tmp_path / "src" / "RC1", locations=30), "RC2": make_source(tmp_path / "src" / "RC2", locations=30)}
    meta = store.create_set(tmp_path / "gold", "two", "v3", sources, n=5, seed=9)
    assert {v: info["drawn"] for v, info in meta["volumes"].items()} == {"RC1": 5, "RC2": 5}


def test_uploaded_list_extends_and_reports_what_it_could_not_use(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src" / "RC1", locations=30, split=(3,), rejected=(9,))
    root = tmp_path / "gold"
    store.create_set(root, "g", "v3", {"RC1": source}, n=3, seed=2)
    present = store.cards(root / "gold-g" / "RC1")[0]["location_id"]
    wanted = [c for c in (11, 12, 13, 14) if c != present][:2]
    result = store.apply_id_list(root / "gold-g", "RC1", source, [*wanted, present, 3, 9, 777], "extend")
    assert sorted(set(result["added"]) - {3}) == sorted(wanted) and result["cards"] == 6
    assert result["skipped"] == {"alreadyInSet": [present], "rejected": [9], "notInSource": [777]}
    assert 3 in result["added"], "a multi-window location is added through its chosen window"


def test_replacing_a_list_swaps_the_cards_and_refuses_to_discard_approved_ones(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src" / "RC1", locations=30)
    root = tmp_path / "gold"
    store.create_set(root, "g", "v3", {"RC1": source}, n=3, seed=2)
    crops = root / "gold-g" / "RC1"
    keep = store.cards(crops)[0]
    catalog.approve_window(crops, keep["location_id"], keep["image_key"])
    with pytest.raises(store.GoldError, match="approved"):
        store.apply_id_list(root / "gold-g", "RC1", source, [20, 21], "replace")

    result = store.apply_id_list(root / "gold-g", "RC1", source, [20, 21], "replace", force=True)
    assert sorted(c["location_id"] for c in store.cards(crops)) == [20, 21] and result["cards"] == 2
    assert not catalog.load_approved(crops)
    assert not (crops / "masks" / f"{keep['image_key']}_{keep['location_id']}.png").exists()


def test_replacements_continue_the_seeded_order_and_never_reuse_a_rejected_id(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src" / "RC1", locations=40)
    root = tmp_path / "gold"
    meta = store.create_set(root, "g", "v3", {"RC1": source}, n=5, seed=11)
    set_dir, crops = root / "gold-g", root / "gold-g" / "RC1"
    first = meta["volumes"]["RC1"]["used"]
    victim = first[0]
    key = next(c["image_key"] for c in store.cards(crops) if c["location_id"] == victim)
    catalog.ignore_location(crops, victim, key)

    added = store.draw_replacement(set_dir, "RC1", source, 1)
    expected = draw.order(pool_of(source), 11)[5]
    assert added == [expected] and victim not in added
    again = store.draw_replacement(set_dir, "RC1", source, 2)
    assert set(again).isdisjoint(set(first) | set(added))
    assert again == draw.order(pool_of(source), 11)[6:8]


def test_replacement_refuses_when_the_source_changed(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src" / "RC1", locations=20)
    root = tmp_path / "gold"
    store.create_set(root, "g", "v3", {"RC1": source}, n=3, seed=1)
    connection = catalog.connect(source)
    connection.execute("DELETE FROM locations WHERE location_id = 20")
    connection.commit()
    connection.close()
    with pytest.raises(store.GoldError, match="changed"):
        store.draw_replacement(root / "gold-g", "RC1", source, 1)


def test_progress_counts(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src" / "RC1", locations=20)
    root = tmp_path / "gold"
    store.create_set(root, "g", "v3", {"RC1": source}, n=6, seed=3)
    crops = root / "gold-g" / "RC1"
    cards = store.cards(crops)
    catalog.approve_window(crops, cards[0]["location_id"], cards[0]["image_key"])
    catalog.ignore_location(crops, cards[1]["location_id"], cards[1]["image_key"])
    assert store.progress(root / "gold-g")["RC1"] == {"target": 6, "cards": 5, "approved": 1, "rejected": 1, "edited": 0, "remaining": 4}


def test_listing_and_resolving_sets(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src" / "RC1", locations=10)
    root = tmp_path / "gold"
    store.create_set(root, "b", "v3", {"RC1": source}, n=2, seed=1)
    store.create_set(root, "a", "v3", {"RC1": source}, n=2, seed=1)
    assert store.list_sets(root) == ["gold-a", "gold-b"] and store.list_sets(None) == []
    assert store.resolve_set(root, "gold-a") == root / "gold-a"
    for bad in ("gold-zzz", "gold-../a", "current"):
        with pytest.raises(FileNotFoundError):
            store.resolve_set(root, bad)


def test_a_set_can_be_made_from_given_lists_and_reports_unusable_ids(tmp_path: Path) -> None:
    sources = {"RC1": make_source(tmp_path / "src" / "RC1", locations=30, split=(3,), rejected=(9,)), "RC2": make_source(tmp_path / "src" / "RC2", locations=20)}
    meta = store.create_set_from_lists(tmp_path / "gold", "gold_v1", "v3", sources, {"RC1": [5, 3, 9, 777, 5], "RC2": [1, 2]}, note="valsets")
    assert meta["fromLists"] is True and meta["note"] == "valsets" and meta["name"] == "gold-gold_v1"
    assert meta["volumes"]["RC1"]["used"] == [5, 3] and meta["volumes"]["RC1"]["multiWindow"] == 1
    assert meta["volumes"]["RC1"]["skipped"] == {"notInSource": [777], "rejected": [9]}
    assert meta["volumes"]["RC2"]["used"] == [1, 2] and "skipped" not in meta["volumes"]["RC2"]
    rows = catalog.list_catalog_rows(tmp_path / "gold" / "gold-gold_v1" / "RC1")
    assert sorted(r["location_id"] for r in rows) == [3, 5]


def test_lists_need_ids_known_volumes_and_a_new_name(tmp_path: Path) -> None:
    sources = {"RC1": make_source(tmp_path / "src" / "RC1", locations=10)}
    with pytest.raises(store.GoldError, match="no location ids"):
        store.create_set_from_lists(tmp_path / "gold", "a", "v3", sources, {"RC1": []})
    with pytest.raises(store.GoldError, match="no source volume"):
        store.create_set_from_lists(tmp_path / "gold", "a", "v3", sources, {"XX": [1]})
    store.create_set_from_lists(tmp_path / "gold", "a", "v3", sources, {"RC1": [1]})
    with pytest.raises(store.GoldError, match="already exists"):
        store.create_set_from_lists(tmp_path / "gold", "a", "v3", sources, {"RC1": [1]})


def test_a_list_made_set_has_no_replacement_order(tmp_path: Path) -> None:
    source = make_source(tmp_path / "src" / "RC1", locations=10)
    store.create_set_from_lists(tmp_path / "gold", "a", "v3", {"RC1": source}, {"RC1": [1, 2]})
    with pytest.raises(store.GoldError, match="made from a list"):
        store.draw_replacement(tmp_path / "gold" / "gold-a", "RC1", source, 1)
