"""Quantized ("crushed") tilesets are shown through Viking's default window and gamma.

The expected numbers are worked out by hand from the rules in Viking's ``CrushInfo.cs`` and ``TileLayout.fx``.
"""

from __future__ import annotations

import io
from xml.etree import ElementTree

import numpy as np
from PIL import Image

from annotation_crops.crush_display import CrushInfo, display_lut
from converter.tilesource import SectionTiles, TileCache, parse_viking_xml

TILE = 4

# Codes 1..4 hold 10, 20, 40 and 30 pixels (bin 0 is the empty edge and is ignored).
HISTOGRAM = "1000 10 20 40 30"


def quantized_xml(attributes: str = 'CrushVersion="1" CrushStep="40" CrushOffset="0" CrushBits="6"',
                  body: str = f"<CodeHistogram>{HISTOGRAM}</CodeHistogram>") -> str:
    return f"""<Volume Name="V" path="http://t.test/V"><Sections><Section Number="1" Path="0001">
      <Tileset name="TEM.Quantized" path="TEM/Quantized" TileXDim="{TILE}" TileYDim="{TILE}" FilePrefix="" FilePostfix=".png" {attributes}>
        {body}
        <Level Downsample="1" path="001" GridDimX="1" GridDimY="1"/>
      </Tileset></Section></Sections></Volume>"""


def codes_http(codes: np.ndarray):
    out = io.BytesIO()
    Image.fromarray(codes, mode="L").save(out, format="PNG")
    data = out.getvalue()
    return lambda url, headers=None: (200, {"etag": '"x"'}, data) if url.endswith("X000_Y000.png") else (404, {}, b"")


def section_for(xml: str, codes: np.ndarray) -> SectionTiles:
    volume = parse_viking_xml(xml, source="http://t.test/V/V.VikingXML")
    return SectionTiles(volume, volume.select(1), 1, TileCache(None, codes_http(codes)))


def test_crush_attributes_and_histogram_are_read() -> None:
    tileset = parse_viking_xml(quantized_xml(), source="s").select(1)
    assert tileset.crush == CrushInfo(40, 0, 6, None, (1000, 10, 20, 40, 30))


def test_a_tileset_without_crush_attributes_is_not_quantized() -> None:
    assert parse_viking_xml(quantized_xml(attributes="", body=""), source="s").select(1).crush is None


def test_auto_window_clips_the_tails_and_skips_bin_zero() -> None:
    crush = parse_viking_xml(quantized_xml(), source="s").select(1).crush
    # 100 pixels in bins 1..4: 0.1% is reached at code 1, 99.9% at code 4.
    assert crush.auto_range() == (1.0, 4.0)
    # Median (the 50th pixel) is code 3: t = (3 - 1) / 3, gamma = log(t) / log(0.5).
    assert abs(crush.default_gamma() - np.log(2 / 3) / np.log(0.5)) < 1e-9


def test_without_a_histogram_the_display_range_then_every_code_is_used() -> None:
    ranged = CrushInfo.from_element(ElementTree.fromstring(
        '<Tileset CrushVersion="1" CrushStep="40" CrushBits="6" CrushDisplayMin="5" CrushDisplayMax="30"/>'))
    assert ranged.auto_range() == (5.0, 30.0) and ranged.default_gamma() == 1.0
    bare = CrushInfo(40, 0, 6)
    assert bare.auto_range() == (0.0, 63.0)


def test_lut_is_the_window_then_gamma() -> None:
    crush = CrushInfo(40, 0, 6, (10.0, 30.0), None)
    lut = display_lut(crush)
    assert lut[10] == 0 and lut[30] == 255 and lut[0] == 0 and lut[200] == 255
    assert lut[20] == 128  # linear: halfway through the window
    assert display_lut(crush, gamma=2.0)[20] == round(255 * 0.5 ** 0.5)


def test_region_and_protocol_tile_show_display_gray_not_codes() -> None:
    codes = np.array([[0, 1, 2, 3]] * TILE, dtype=np.uint8)
    section = section_for(quantized_xml(), codes)
    shown = section.region(0, 0, TILE, TILE)
    gamma = section.tileset.crush.default_gamma()
    expected = np.rint(np.clip((np.arange(4) / 255 - 1 / 255) / (3 / 255), 0, 1) ** (1 / gamma) * 255).astype(np.uint8)
    assert (shown[0] == expected).all() and shown.max() > 100


def test_dim_tissue_that_windows_to_black_is_not_an_empty_tile(monkeypatch) -> None:
    import converter.tilesource as tilesource
    monkeypatch.setattr(tilesource, "PROTOCOL_TILE", TILE)
    codes = np.zeros((TILE, TILE), dtype=np.uint8)
    codes[0, 0] = 1  # below the window's black point once the display range starts above 1
    xml = quantized_xml('CrushVersion="1" CrushStep="40" CrushBits="6" CrushDisplayMin="5" CrushDisplayMax="30"', "")
    assert section_for(xml, codes).protocol_tile(0, 0) is not None


def test_wider_than_eight_bit_codes_and_plain_tilesets_are_left_as_stored() -> None:
    codes = np.array([[0, 1, 2, 3]] * TILE, dtype=np.uint8)
    wide = section_for(quantized_xml('CrushVersion="1" CrushStep="1" CrushBits="12"', ""), codes)
    assert (wide.region(0, 0, TILE, TILE) == codes).all()
    plain = section_for(quantized_xml(attributes="", body=""), codes)
    assert (plain.region(0, 0, TILE, TILE) == codes).all()
