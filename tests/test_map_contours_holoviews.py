"""Interactive maps with contour-line overlays (``contours=`` over a map).

The holoviews twin of ``tests/test_map_contours_static.py``, on the same stress case: a
dateline-straddling comparison row stored in +-180, the seam inside its own array. Bokeh
has no contour primitive, so the lines are ``hv.Contours`` from
:func:`ocean_skill.plot.section.contour_paths`, their vertices moved into the panel's
180-centred frame by the transform the domain ring takes. The object is rendered through
bokeh: holoviews' failures here show only at render.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from ocean_skill.plot.registry import render
from ocean_skill.plot.spec import PlotSpec

hv = pytest.importorskip("holoviews")
pytest.importorskip("hvplot")
pytest.importorskip("geoviews")
pytest.importorskip("cartopy")


def _ramp(lon, lat):
    """Return a field linear in true (0-360) longitude and latitude."""
    return 0.1 * (np.asarray(lon) % 360.0) + 0.2 * np.asarray(lat)


def _straddling_180(offset: float = 0.0) -> xr.DataArray:
    """Return a 140E-290E ramp stored in +-180, the seam inside the array."""
    lon = np.unique(
        np.concatenate([np.linspace(140.0, 180.0, 8), np.linspace(-180.0, -70.0, 10)])
    )
    lat = np.linspace(-20.0, 20.0, 12)
    lon2, lat2 = np.meshgrid(lon, lat)
    return xr.DataArray(
        _ramp(lon2, lat2) + offset,
        dims=("lat", "lon"),
        coords={"lon": lon, "lat": lat},
        attrs={"units": "mmol m-3"},
    )


def _exact(obj, kind) -> list:
    """Return every element of exactly ``kind`` (a ``Path`` is not a ``Contours``)."""
    return [e for e in obj.traverse(lambda x: x, [kind]) if type(e) is kind]


def test_a_straddling_row_draws_its_lines_on_test_and_reference_in_the_180_frame():
    test, reference = _straddling_180(1.0), _straddling_180()
    item = {
        "aligned": xr.Dataset(
            {"test": test, "reference": reference, "difference": test - reference}
        ),
        "units": "mmol m-3",
        "standard_name": "mole_concentration_of_nitrate_in_sea_water",
        "labels": ("roms_run", "woa23"),
        "contour": {"test": _straddling_180(), "reference": _straddling_180(1.0)},
        "contour_standard_name": "sea_water_temperature",
        "contour_units": "degC",
    }
    levels = [float(v) for v in range(12, 30, 2)]
    obj = render(
        PlotSpec(
            family="field_row",
            items=[item],
            options={"tiles": False, "land": False, "contour_levels": levels},
        ),
        renderer="holoviews",
    )
    hv.render(obj, backend="bokeh")  # holoviews fails at bokeh render, not before
    test_panel, ref_panel, diff_panel = list(obj.values())
    assert _exact(diff_panel, hv.Contours) == []  # no isolines of its own
    for panel, offset in ((test_panel, 0.0), (ref_panel, 1.0)):
        lines = [line for c in _exact(panel, hv.Contours) for line in c.data]
        assert lines and len(_exact(panel, hv.Labels)) == 1
        for line in lines:
            x, y = np.asarray(line["x"]), np.asarray(line["y"])
            # in the 180-centred frame (140..290 -> -40..110), never across the globe
            assert x.min() >= -41.0 and x.max() <= 111.0, (x.min(), x.max())
            assert np.abs(np.diff(x)).max(initial=0.0) < 30.0
            # and on the overlay's own isoline, once the frame shift is undone
            np.testing.assert_allclose(
                _ramp(x + 180.0, y) + offset, float(line["level"]), atol=1e-6
            )
        hover = panel.traverse(lambda e: e, [hv.Contours])[0].opts.get().kwargs
        assert hover["hover_tooltips"] == [("temperature", "@text")]
