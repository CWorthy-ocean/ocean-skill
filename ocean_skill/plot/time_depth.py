"""Shared geometry for depth-against-time plots: the ``time_depth`` family.

One place both renderers read, so a ``time_depth`` panel cannot look different
statically than interactively — the same reason :mod:`ocean_skill.plot.section` exists
for the vertical-slice family, and :mod:`ocean_skill.plot.series` for the line family.
:func:`prepare_time_depth` decides every axis convention this family needs once: which
coordinate is depth and which sign it reads positive, what the title calls the place
and the period, and (via :func:`default_mark`) whether the record is ragged enough to
draw as scattered points rather than a mesh. A renderer's own drawing function calls
this first and then only draws — it makes no convention decisions of its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

__all__ = [
    "SCATTER_NAN_FRACTION",
    "TimeDepthGeometry",
    "default_mark",
    "prepare_time_depth",
]

#: Fraction of the (time, depth) rectangle that must be NaN, once all-NaN rows and
#: columns are dropped, before :func:`default_mark` calls a record ragged enough to
#: draw as scattered points rather than a mesh. A repeat-visit station's own build
#: (:func:`ocean_skill.tabular._timeseriesprofile_dataset`) NaN-fills every
#: (visit, level) combination no cast actually sampled -- a bottle station that visits
#: a handful of standard depths on each cast still has real holes even after the
#: all-NaN trim, where a continuously logging mooring at a fixed set of levels has
#: none. The threshold is a heuristic, not a measurement of raggedness itself, and
#: ``mark=`` overrides it either way.
SCATTER_NAN_FRACTION = 0.25


@dataclass
class TimeDepthGeometry:
    """What a ``time_depth`` panel's axes mean, decided once and read by both renderers.

    ``x_name``/``y_name`` name the coordinates :func:`prepare_time_depth` attaches to
    its returned field -- ``"time"``/``"depth"`` -- each 1-D, on their own single
    dimension: unlike :class:`ocean_skill.plot.section.SectionGeometry`'s
    ``distance``/``depth`` (which broadcasts, because a section's vertical axis can
    genuinely vary *along* its horizontal one), a ``time_depth`` panel is always one
    place (see :attr:`ocean_skill.field.Field.is_time_depth`), so its vertical
    coordinate -- an observational level, or a native ``z_rho`` -- has only the
    vertical dimension left to vary along. One rectilinear mesh, the ordinary shape
    both :func:`~ocean_skill.plot.matplotlib_renderer.time_depth` and
    :func:`~ocean_skill.plot.holoviews_renderer._time_depth`'s mesh branch draw
    without needing 2-D coordinates at all.
    """

    x_name: str
    y_name: str
    x_label: str
    y_label: str
    place_note: str
    period_note: str


def prepare_time_depth(da: xr.DataArray) -> tuple[xr.DataArray, TimeDepthGeometry]:
    """Return ``(field, geometry)``: ``da``, reordered and carrying ``depth``.

    ``da`` must be exactly two-dimensional: a time axis (see
    :func:`ocean_skill.operators.resolve_dim`, axis ``"T"``) and one vertical axis,
    already carrying a real coordinate to label it with -- a bare, coordinate-less
    native ``s_rho``/``s_w`` axis is refused before reaching here (see
    :func:`ocean_skill.field._labelless_vertical`,
    :meth:`ocean_skill.field.Field._time_depth_item`), the same guard
    :meth:`~ocean_skill.field.Field._series_items` applies to a fanned line.

    The vertical coordinate is read the same way
    :func:`ocean_skill.plot.profile.vertical_values` does: whichever of the axis's own
    coordinate or a same-dim ``z_rho`` is present, taken as ``abs()`` -- a no-op for an
    already positive-down observational ``depth``, and what turns ROMS's negative-down
    ``z_rho`` into the positive-down metres every other depth label in this package
    uses. Unlike :func:`ocean_skill.plot.section.prepare_section`, this never negates
    the raw coordinate first: a tabular station's ``depth`` is already positive-down,
    so :func:`~ocean_skill.plot.section.prepare_section`'s "must be negative-down"
    convention does not apply here.

    The returned field is transposed to ``(vertical, time)`` -- the row/column order
    ``pcolormesh``/``hvplot.quadmesh`` both expect for two 1-D coordinates (``C.shape
    == (len(y), len(x))``), so a renderer draws ``x=geometry.x_name,
    y=geometry.y_name`` against it with no further reshaping.
    """
    from ocean_skill.operators import resolve_dim
    from ocean_skill.plot.series import _period_of, _place_of, time_values

    tdim = resolve_dim(da, "T")
    if tdim is None or tdim not in da.dims:
        raise ValueError(
            f"prepare_time_depth expects a time axis -- got dims {sorted(da.dims)}."
        )
    extra = [str(d) for d in da.dims if d != tdim]
    if len(extra) != 1:
        raise ValueError(
            "prepare_time_depth expects exactly one vertical axis beside time -- "
            f"got dims {sorted(da.dims)}."
        )
    zdim = extra[0]

    z_rho = da.coords.get("z_rho")
    native_z_rho = z_rho is not None and zdim in z_rho.dims and zdim not in da.coords
    depth_source = z_rho if native_z_rho else da[zdim]

    depth = np.abs(depth_source).rename("depth")
    depth.attrs["units"] = depth_source.attrs.get("units", "m")

    result = da.assign_coords(
        depth=(zdim, np.asarray(depth)), time=(tdim, time_values(da[tdim]))
    ).transpose(zdim, tdim)

    geometry = TimeDepthGeometry(
        x_name="time",
        y_name="depth",
        x_label="time",
        y_label=f"Depth [{depth.attrs.get('units') or 'm'}]",
        place_note=_place_of(da),
        period_note=_period_of(da[tdim]),
    )
    return result, geometry


def default_mark(values: xr.DataArray) -> str:
    """``"scatter"`` for a ragged record, ``"pcolormesh"`` for a dense one.

    Dropping all-NaN rows and columns first means a repeat-visit station's own union
    of every visit's distinct depths (:func:`ocean_skill.tabular
    ._timeseriesprofile_dataset`) is judged on the holes *within* that trimmed
    rectangle, not on how much of the full union any single cast happened to reach --
    a station with a few dense casts at genuinely different depths still reads as
    ragged, which is the case this threshold exists for. See
    :data:`SCATTER_NAN_FRACTION`; ``mark=`` on the family functions overrides this
    outright.
    """
    dim0, dim1 = values.dims
    trimmed = values.dropna(dim0, how="all").dropna(dim1, how="all")
    total = trimmed.size
    if total == 0:
        return "scatter"
    nan_fraction = float(trimmed.isnull().sum()) / total
    return "scatter" if nan_fraction >= SCATTER_NAN_FRACTION else "pcolormesh"
