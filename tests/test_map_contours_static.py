"""Static map figures: contour lines of an overlay variable drawn over the fill.

One test, on the stress case: a dateline-straddling comparison row stored in +-180, so
the seam falls inside the array and a line joined across it would span the globe. The
geometry itself (:mod:`ocean_skill.plot.map_contours`) and the pairing
(``tests/test_contour_pairing.py``) are tested elsewhere; this checks the static
renderer acts on them. The item is built directly and handed to ``render(PlotSpec())``,
with the land off -- the slow part of a map, and not what is checked here.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.contour import ContourSet

from ocean_skill.plot.registry import render
from ocean_skill.plot.spec import PlotSpec


def _straddling_180(*, scale: float = 1.0, shift: float = 0.0) -> xr.DataArray:
    """Return a 140E-290E map stored in +-180 (seam inside the array), ramp 0..40."""
    lon = np.unique(
        np.concatenate([np.linspace(140.0, 180.0, 9), np.linspace(-180.0, -70.0, 22)])
    )
    lat = np.linspace(-20.0, 20.0, 13)
    lon2, lat2 = np.meshgrid(lon % 360.0, lat)
    values = 20 * (lon2 - 140.0) / 150.0 + 20 * (lat2 + 20.0) / 40.0
    return xr.DataArray(
        scale * values + shift,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        attrs={"units": "mmol m-3"},
    )


def _lines(ax) -> list[ContourSet]:
    return [c for c in ax.collections if isinstance(c, ContourSet) and not c.filled]


def test_a_straddling_row_draws_its_lines_on_test_and_reference_inside_the_frame():
    test, ref = _straddling_180(shift=1.0), _straddling_180()
    item = {
        "aligned": xr.Dataset(
            {"test": test, "reference": ref, "difference": test - ref}
        ),
        "units": "mmol m-3",
        "standard_name": "sea_water_temperature",
        "labels": ("model", "obs"),
        "label": "row",
        "contour": {"test": _straddling_180(), "reference": _straddling_180(scale=2.0)},
    }
    fig = render(
        PlotSpec(family="field_row", items=[item], options={"land": False}),
        renderer="matplotlib",
    )
    test_ax, ref_ax, diff_ax = [ax for ax in fig.axes if isinstance(ax, GeoAxes)]
    assert _lines(diff_ax) == []  # a difference has no isolines of its own
    (t_lines,), (r_lines,) = _lines(test_ax), _lines(ref_ax)
    # one pooled rule for the figure: the same levels on both panels, reaching past the
    # test's own 0..40 because the reference spans twice that
    assert list(t_lines.levels) == list(r_lines.levels)
    assert max(t_lines.levels) > 40.0
    for ax, lines in ((test_ax, t_lines), (ref_ax, r_lines)):
        # cartopy hands the vertices back in the axes' own (180-centred) frame: no
        # piece may jump across the globe, and none may land off the panel (3 degrees
        # of slack: a line reaches the edge cell's centre, the fill's view a bit short)
        segments = [np.asarray(s) for level in lines.allsegs for s in level if len(s)]
        assert segments
        (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
        for seg in segments:
            assert np.abs(np.diff(seg[:, 0])).max() < 30.0, "a line spans the globe"
            assert x0 - 3 <= seg[:, 0].min() and seg[:, 0].max() <= x1 + 3
            assert y0 - 3 <= seg[:, 1].min() and seg[:, 1].max() <= y1 + 3
        assert lines.labelTexts  # labelled once the layout settled
