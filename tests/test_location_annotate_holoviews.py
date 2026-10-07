"""``legend="annotate"`` for the interactive ``locations`` overlay.

Each labelled selection group's name is an ``hv.Labels`` beside its shape; its glyphs
carry no legend label, and only the unlabelled groups stay keyed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_skill.plot.locations import HOVER_FIELDS, _seam_split


def _hover(name):
    return {f: (name if f == "name" else "") for f in HOVER_FIELDS}


def _map(lon):
    lat = np.linspace(-40.0, 40.0, 20)
    z = np.add.outer(np.linspace(0.0, 1.0, lat.size), np.linspace(0.0, 1.0, len(lon)))
    return xr.DataArray(
        z, dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, attrs={"units": "m"}
    )


PLAIN = _map(np.linspace(-98.0, -80.0, 30))
STRADDLER = _map(
    np.unique(
        np.concatenate([np.linspace(80.0, 180.0, 40), np.linspace(-180.0, -44.0, 30)])
    )
)


def _sel(name, kind, label=True, **geom):
    return {
        **_hover(name),
        "kind": kind,
        "featureType": "selection",
        "legend_label": name if label else None,
        **geom,
    }


PLAIN_ITEMS = [
    {
        **_hover("stn"),
        "kind": "point",
        "featureType": "timeSeries",
        "lon": -90.0,
        "lat": 5.0,
    },
    _sel("SG", "point", lon=-92.0, lat=-3.0),
    _sel("BOX", "extent", bboxes=[(-96.0, -5.0, -88.0, 5.0)]),
    _sel("Eq", "line", paths=[np.array([[-95.0, 0.0], [-85.0, 0.0]])]),
]

STRADDLE_ITEMS = [
    _sel("Pt", "point", lon=155.0, lat=-15.0),
    _sel(
        "Band",
        "extent",
        bboxes=[(170.0, 5.0, 180.0, 15.0), (-180.0, 5.0, -160.0, 15.0)],
    ),
    _sel(
        "Eq", "line", paths=_seam_split(np.array([160.0, 210.0]), np.array([0.0, 0.0]))
    ),
]


def _item(field):
    return {
        "field": field,
        "facet_dim": None,
        "row_dim": None,
        "units": "m",
        "standard_name": "sea_floor_depth_below_geoid",
    }


def _hv(field, items, **opts):
    from ocean_skill.plot.holoviews_renderer import _field_facet

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _field_facet(_item(field), tiles=False, location_items=items, **opts)


def _of(overlay, name):
    return [e for e in overlay if type(e).__name__ == name]


def _label_xy(overlay):
    return {
        e.dimension_values(2)[0]: (e.dimension_values(0)[0], e.dimension_values(1)[0])
        for e in _of(overlay, "Labels")
    }


def _legend_labels(overlay):
    import holoviews as hv

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = hv.render(overlay, backend="bokeh")
    return [item.label["value"] for leg in fig.legend for item in leg.items], fig


def test_annotate_draws_labels_and_keys_only_unlabelled_groups():
    overlay = _hv(PLAIN, PLAIN_ITEMS, legend="annotate")
    assert sorted(_label_xy(overlay)) == ["BOX", "Eq", "SG"]
    labels, _ = _legend_labels(overlay)
    assert labels == ["timeSeries"]


def test_annotate_label_style_and_offset():
    overlay = _hv(PLAIN, PLAIN_ITEMS, legend="annotate")
    by_text = {e.dimension_values(2)[0]: e for e in _of(overlay, "Labels")}
    kw = by_text["SG"].opts.get().kwargs
    assert kw["apply_ranges"] is False
    assert kw["text_font_style"] == "bold"
    assert kw["text_align"] == "left" and kw["text_baseline"] == "middle"
    kw = by_text["BOX"].opts.get().kwargs
    assert kw["text_align"] == "center" and kw["text_baseline"] == "bottom"
    _, fig = _legend_labels(overlay)
    # bokeh's Text glyph: x_offset/y_offset are screen pixels (positive y is up)
    got = sorted(
        (r.glyph.x_offset, r.glyph.y_offset)
        for r in fig.renderers
        if type(r.glyph).__name__ in ("Text", "LabelSet")
    )
    assert got == [(0, 5), (0, 5), (5, 0)]


def test_all_labelled_means_no_legend():
    items = [i for i in PLAIN_ITEMS if i["featureType"] == "selection"]
    overlay = _hv(PLAIN, items, legend="annotate")
    labels, _ = _legend_labels(overlay)
    assert labels == []


def test_true_and_false_unchanged_and_bad_value_raises():
    overlay = _hv(PLAIN, PLAIN_ITEMS, legend=True)
    assert not _of(overlay, "Labels")
    labels, _ = _legend_labels(overlay)
    assert sorted(labels) == ["BOX", "Eq", "SG", "timeSeries"]
    assert not _of(_hv(PLAIN, PLAIN_ITEMS, legend=False), "Labels")
    with pytest.raises(ValueError):
        _hv(PLAIN, PLAIN_ITEMS, legend="bogus")


def test_annotate_labels_on_every_facet_panel_legend_on_first():
    time = pd.date_range("2012-01-01", periods=2, freq="MS")
    field = xr.concat([PLAIN, PLAIN + 1.0], dim="time").assign_coords(time=time)
    item = {**_item(field), "facet_dim": "time"}
    from ocean_skill.plot.holoviews_renderer import _field_facet

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        layout = _field_facet(
            item, tiles=False, location_items=PLAIN_ITEMS, legend="annotate"
        )
    for panel in layout:
        assert sorted(_label_xy(panel)) == ["BOX", "Eq", "SG"]
    shows = [
        _of(panel, "Points")[0].opts.get().kwargs["show_legend"] for panel in layout
    ]
    assert shows == [True, False]


def test_labels_sit_on_their_shapes_plain_frame():
    xy = _label_xy(_hv(PLAIN, PLAIN_ITEMS, legend="annotate"))
    assert xy["SG"] == (-92.0, -3.0)
    assert xy["BOX"] == (-92.0, 5.0)
    assert xy["Eq"] == (-85.0, 0.0)


def test_labels_sit_on_their_shapes_180_centred_frame():
    overlay = _hv(STRADDLER, STRADDLE_ITEMS, legend="annotate")
    xy = _label_xy(overlay)
    # x = lon % 360 - 180: 155E -> -25
    assert xy["Pt"] == (-25.0, -15.0)
    assert xy["Band"][1] == 15.0
    (rects,) = _of(overlay, "Rectangles")
    x0 = rects.dimension_values(0)
    x1 = rects.dimension_values(2)
    assert min(x0) <= xy["Band"][0] <= max(x1)  # on the band, not across the map
    # the line ends at the last vertex of its last piece, 210E -> 30
    (path,) = _of(overlay, "Path")
    last = path.split()[-1]
    assert xy["Eq"] == (last.dimension_values(0)[-1], last.dimension_values(1)[-1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import holoviews as hv

        hv.render(overlay, backend="bokeh")


def test_standalone_locations_map_annotates():
    from ocean_skill.plot.holoviews_renderer import _locations

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        overlay = _locations(PLAIN_ITEMS, tiles=None, land=False, legend="annotate")
    assert sorted(_label_xy(overlay)) == ["BOX", "Eq", "SG"]
    labels, _ = _legend_labels(overlay)
    assert labels == ["timeSeries"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        every = _locations(
            [i for i in PLAIN_ITEMS if i["featureType"] == "selection"],
            tiles=None,
            land=False,
            legend="annotate",
        )
    labels, _ = _legend_labels(every)
    assert labels == []
