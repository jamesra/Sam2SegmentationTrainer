"""List Viking circle annotations that are probably larger than their cell, for review in Viking.

Reads one volume's Locations and LocationLinks from Viking OData and compares only sizes, so the
misregistration between sections does not matter. A circle is flagged when:

* ``oversize_vs_polygon``: its radius exceeds the equal-area radius (Viking ``Radius``) of a polygon of the same
  structure that it is linked to within two sections. An inscribed circle can never do that.
* ``2x_jump``: it is linked to a circle of the same structure whose radius is about half of its own
  (|log2 ratio| within 0.85..1.15). The larger circle is the suspect.
* ``copied_from_flagged``: it carries exactly the radius of a flagged circle it is chained to through
  same-structure circle links. Viking copies a circle's radius to the next section, so one bad radius often
  runs down a whole chain.

Usage: python circle_review.py VOLUME [VOLUME ...] [--out DIR] [--base https://websvc.codepharm.net]
Writes ``{out}/{VOLUME}_circle_review.csv`` and caches the OData pull in ``{out}/_cache``.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import urllib.parse
import urllib.request
from pathlib import Path

LINK_MAX_DZ = 2
JUMP_LOG2 = (0.85, 1.15)
SAME_RADIUS = 0.01


def _pages(url: str):
    while url:
        with urllib.request.urlopen(url, timeout=300) as response:
            payload = json.load(response)
        yield from payload["value"]
        url = payload.get("@odata.nextLink")


def pull(base: str, volume: str, cache: Path) -> dict:
    path = cache / f"{volume}_odata.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    root = f"{base.rstrip('/')}/{volume}/OData"
    flt = urllib.parse.quote("(TypeCode eq 1 or TypeCode eq 4 or TypeCode eq 6)")
    data = {
        "locs": list(_pages(f"{root}/Locations?$filter={flt}&$select=ID,ParentID,Z,TypeCode,Radius,Created,Username")),
        "links": list(_pages(f"{root}/LocationLinks?$select=A,B")),
    }
    cache.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def review(volume: str, data: dict) -> list[dict]:
    locs = {r["ID"]: r for r in data["locs"] if r.get("Radius")}
    is_circle = lambda r: r["TypeCode"] == 1
    neighbors: dict[int, list[int]] = collections.defaultdict(list)
    for link in data["links"]:
        a, b = locs.get(link["A"]), locs.get(link["B"])
        if a and b and a["ParentID"] == b["ParentID"] and abs(a["Z"] - b["Z"]) <= LINK_MAX_DZ:
            neighbors[a["ID"]].append(b["ID"])
            neighbors[b["ID"]].append(a["ID"])

    flags: dict[int, dict] = {}

    def flag(loc_id: int, reason: str, ratio: float, evidence: int, root: int) -> None:
        if loc_id not in flags:  # first (strongest) reason wins
            flags[loc_id] = {"reason": reason, "ratio": ratio, "evidence": evidence, "root": root}

    circles = [r for r in locs.values() if is_circle(r)]
    for c in circles:  # pass 1: direct evidence against a linked polygon
        for other_id in neighbors[c["ID"]]:
            other = locs[other_id]
            if not is_circle(other) and c["Radius"] > other["Radius"]:
                flag(c["ID"], "oversize_vs_polygon", c["Radius"] / other["Radius"], other_id, c["ID"])
                break
    for c in circles:  # pass 2: ~2x jump to a linked circle; the larger side is suspect
        for other_id in neighbors[c["ID"]]:
            other = locs[other_id]
            if is_circle(other) and c["Radius"] > other["Radius"]:
                if JUMP_LOG2[0] <= math.log2(c["Radius"] / other["Radius"]) <= JUMP_LOG2[1]:
                    flag(c["ID"], "2x_jump", c["Radius"] / other["Radius"], other_id, c["ID"])
                    break
    for root_id in [i for i, f in flags.items() if f["reason"] != "copied_from_flagged"]:  # pass 3: copies
        radius = locs[root_id]["Radius"]
        stack, seen = [root_id], {root_id}
        while stack:
            for other_id in neighbors[stack.pop()]:
                other = locs[other_id]
                if other_id in seen or not is_circle(other) or abs(other["Radius"] - radius) > SAME_RADIUS:
                    continue
                seen.add(other_id)
                stack.append(other_id)
                flag(other_id, "copied_from_flagged", flags[root_id]["ratio"], flags[root_id]["evidence"], root_id)

    rows = []
    for loc_id, f in flags.items():
        r = locs[loc_id]
        rows.append({
            "volume": volume, "location_id": loc_id, "structure_id": r["ParentID"], "z": r["Z"],
            "radius_px": round(r["Radius"], 1), "username": (r.get("Username") or "").strip(),
            "created": (r.get("Created") or "")[:10], "reason": f["reason"], "ratio": round(f["ratio"], 2),
            "evidence_location_id": f["evidence"], "chain_root_id": f["root"],
        })
    rows.sort(key=lambda row: (row["structure_id"], row["chain_root_id"], row["z"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("volumes", nargs="+")
    parser.add_argument("--out", type=Path, default=Path("circle_review"))
    parser.add_argument("--base", default="https://websvc.codepharm.net")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for volume in args.volumes:
        data = pull(args.base, volume, args.out / "_cache")
        rows = review(volume, data)
        n_circles = sum(1 for r in data["locs"] if r["TypeCode"] == 1)
        path = args.out / f"{volume}_circle_review.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["volume"])
            writer.writeheader()
            writer.writerows(rows)
        counts = collections.Counter(row["reason"] for row in rows)
        print(f"{volume}: {n_circles} circles, {len(rows)} flagged ({100 * len(rows) / max(1, n_circles):.2f}%) "
              f"{dict(counts)} -> {path}", flush=True)


if __name__ == "__main__":
    main()
