"""Movies: the same ``test | reference | difference`` row, over frames.

Two renderings of one idea, so the tests come in pairs where they can: the static
renderer encodes the frames into an mp4 or gif, the interactive one puts them on a
slider. What both must agree on is the thing that makes an animation readable at all —
**one colour scale for the whole movie**. A scale re-derived per frame makes a changing
ruler look like a changing field, which is worse than no movie, and it is the kind of
regression that looks fine in any single frame.

The gif is written for real (Pillow ships with matplotlib), and so is the mp4 when
ffmpeg is present; the "no ffmpeg" path is faked, since the interesting case is the one
this machine may not be in.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from ocean_skill.plot.registry import render
from ocean_skill.plot.spec import PlotSpec

NITRATE = "mole_concentration_of_nitrate_in_sea_water"


@pytest.fixture(autouse=True)
def agg_backend():
    """Draw headless: every test here builds a figure, none of them wants a window."""
    import matplotlib

    matplotlib.use("Agg")
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _field(offset: float, *, shape: tuple[int, int] = (8, 10)) -> xr.DataArray:
    ny, nx = shape
    da = xr.DataArray(
        5.0 + offset + np.linspace(0, 1, ny * nx).reshape(ny, nx),
        dims=("lat", "lon"),
        coords={
            "lat": np.linspace(18, 26, ny),
            "lon": np.linspace(-100, -90, nx),
        },
    )
    da.attrs["units"] = "mmol m-3"
    return da


def _frame(index: int, *, offset: float | None = None) -> dict:
    """One frame, shaped exactly as ``ComparisonSet.movie()`` builds it."""
    test = _field(1.0 if offset is None else offset)
    reference = _field(0.0)
    return {
        "aligned": {
            "test": test,
            "reference": reference,
            "difference": test - reference,
        },
        "units": "mmol m-3",
        "standard_name": NITRATE,
        "metrics": {"bias": 0.1 * index, "rmse": 0.5, "corr": 0.98},
        "labels": ("GOM_bgc", "woa23_nitrate"),
        "frame_label": f"2012-01-{index + 1:02d}",
    }


@pytest.fixture
def frames() -> list[dict]:
    """Four frames whose test field climbs, so a fixed scale is observably fixed."""
    return [_frame(i, offset=1.0 + 0.5 * i) for i in range(4)]


def _movie(frames, **options):
    return render(
        PlotSpec(family="field_movie", items=frames, options=options),
        renderer="matplotlib",
    )


def _interactive(frames, **options):
    return render(
        PlotSpec(family="field_movie", items=frames, options=options),
        renderer="holoviews",
    )


# --- writing files ------------------------------------------------------------------


def test_a_gif_is_written(frames, tmp_path):
    out = tmp_path / "movie.gif"
    _movie(frames, save=out)
    assert out.exists()
    # a real GIF, not an empty file with the right name
    assert out.read_bytes()[:4] == b"GIF8"


@pytest.mark.skipif(
    not __import__("matplotlib.animation", fromlist=["x"]).FFMpegWriter.isAvailable(),
    reason="ffmpeg not installed; the missing-ffmpeg path is tested separately",
)
def test_an_mp4_is_written(frames, tmp_path):
    out = tmp_path / "movie.mp4"
    _movie(frames, save=out)
    assert out.exists()
    # ISO base media: 'ftyp' at bytes 4-8. Also guards the yuv420p/even-dimension
    # pad filter, which ffmpeg would otherwise fail on outright.
    assert out.read_bytes()[4:8] == b"ftyp"


def test_a_missing_ffmpeg_says_how_to_get_one(frames, tmp_path, monkeypatch):
    """An mp4 is the one format with an external dependency, so say so usefully."""
    from matplotlib import animation

    monkeypatch.setattr(animation.FFMpegWriter, "isAvailable", staticmethod(lambda: 0))
    with pytest.raises(RuntimeError) as excinfo:
        _movie(frames, save=tmp_path / "movie.mp4")
    message = str(excinfo.value)
    assert "ffmpeg" in message
    assert "conda" in message
    assert ".gif" in message, "the fallback that needs nothing extra has to be named"


def test_an_unknown_extension_is_refused(frames, tmp_path):
    with pytest.raises(ValueError, match="unknown extension"):
        _movie(frames, save=tmp_path / "movie.avi")


def test_subdirectories_are_created(frames, tmp_path):
    out = tmp_path / "figures" / "nested" / "movie.gif"
    _movie(frames, save=out)
    assert out.exists()


# --- what makes an animation readable -----------------------------------------------


def _norms(ani):
    """Each panel's (vmin, vmax), which must not move as the movie plays."""
    return [
        (im.norm.vmin, im.norm.vmax)
        for im in ani._fig.axes[0].collections + ani._fig.axes[2].collections
    ]


def test_the_colour_scale_never_moves_between_frames(frames):
    ani = _movie(frames)
    before = _norms(ani)
    for index in range(len(frames)):
        ani._func(index)
    assert _norms(ani) == before


def test_the_shared_scale_covers_every_frame_not_just_the_first(frames):
    """The point of shared_limits: the last frame's values are on the bar too.

    Frame 3's field is 1.5 above frame 0's, so a scale taken from frame 0 alone would
    put most of the last frame off the top of its own colorbar.
    """
    shared = _movie(frames)
    first_only = _movie(frames, shared_limits=False)
    assert _norms(shared)[0][1] > _norms(first_only)[0][1]
    last = float(np.asarray(frames[-1]["aligned"]["test"]).max())
    assert _norms(shared)[0][1] >= _norms(first_only)[0][1]
    assert last > _norms(first_only)[0][1], "fixture no longer exercises the difference"


def test_every_frame_redraws_its_own_values(frames):
    ani = _movie(frames)
    mesh = ani._fig.axes[0].collections[0]
    ani._func(3)
    drawn = np.asarray(mesh.get_array()).ravel()
    expected = np.asarray(frames[3]["aligned"]["test"]).ravel()
    assert np.allclose(drawn, expected)


def test_the_frame_label_and_the_metrics_box_follow_the_frame(frames):
    ani = _movie(frames)
    ani._func(2)
    texts = [t.get_text() for ax in ani._fig.axes for t in ax.texts]
    assert "2012-01-03" in texts
    assert any("bias=0.2" in t for t in texts), texts


def test_the_frame_label_can_be_omitted(frames):
    ani = _movie(frames, frame_label=False)
    texts = [t.get_text() for ax in ani._fig.axes for t in ax.texts]
    assert not any("2012-01" in t for t in texts)


def test_the_frame_label_is_stylable(frames):
    ani = _movie(frames, frame_label_kwargs={"fontsize": 14, "color": "darkred"})
    label = next(
        t
        for ax in ani._fig.axes
        for t in ax.texts
        if t.get_text().startswith("2012-01")
    )
    assert label.get_fontsize() == 14
    assert label.get_color() == "darkred"


def test_contourf_frames_are_redrawn_rather_than_refilled(frames):
    """A filled contour set has no array to swap, so it must be replaced instead.

    Replaced, not added to: the old set has to go, or every frame leaves its
    predecessor underneath and by the end the panel is drawing the whole movie at once.
    (``ax.collections`` also holds cartopy's land and coastline artists, hence the
    filter rather than a count of everything.)
    """
    from matplotlib.contour import ContourSet

    ani = _movie(frames, mark="contourf")
    contours = lambda ax: [c for c in ax.collections if isinstance(c, ContourSet)]  # noqa: E731
    assert len(contours(ani._fig.axes[0])) == 1
    ani._func(2)
    ani._func(3)
    assert len(contours(ani._fig.axes[0])) == 1


# --- frame selection ----------------------------------------------------------------


def test_every_thins_the_frames(frames):
    assert _movie(frames, every=2)._save_count == 2


def test_every_must_be_positive(frames):
    with pytest.raises(ValueError, match="every must be 1 or more"):
        _movie(frames, every=0)


def test_a_long_movie_warns_before_spending_the_time():
    many = [_frame(i) for i in range(205)]
    with pytest.warns(UserWarning, match="long movie"):
        _movie(many)


def test_a_long_movie_is_not_capped():
    many = [_frame(i) for i in range(205)]
    with pytest.warns(UserWarning):
        assert _movie(many)._save_count == 205


def test_no_frames_at_all_is_refused():
    with pytest.raises(ValueError, match="at least one frame"):
        _movie([])


def _tall_frame(index: int) -> dict:
    """Build a frame whose domain is taller than wide (a coastal strip, aspect ~0.2)."""
    frame = _frame(index)
    lon = np.linspace(-98, -95.5, 10)  # 2.5 deg across against 8 deg of latitude
    frame["aligned"] = {
        key: da.assign_coords(lon=lon) for key, da in frame["aligned"].items()
    }
    return frame


def _bar_orientation(fig) -> str:
    bars = [ax for ax in fig.axes if getattr(ax, "_osk_cbar_parents", None)]
    assert bars, "no colorbars were recorded"
    return "horizontal" if bars[0]._osk_cbar_horizontal else "vertical"


def test_a_movie_frame_is_laid_out_exactly_like_the_still_row():
    """This function's docstring promises it; for a tall domain it used not to hold.

    ``field_row`` picks its colorbar orientation from the map's aspect ratio, but
    ``field_movie`` hardcoded horizontal. Below about 0.8 the two disagreed: the movie
    put the bars under the maps and reshaped the figure to fit them, where the still put
    them beside. Since a frame is documented as being the row that ``field_row`` draws,
    drawn by the same code, that is a broken promise rather than a matter of taste.

    Checked on a tall domain because the default fixture is wide (aspect 1.25) and would
    have passed throughout.
    """
    from ocean_skill.plot.matplotlib_renderer import field_row

    tall = [_tall_frame(i) for i in range(2)]
    still = field_row(tall[0]["aligned"], units="mmol m-3", title="t")
    movie_fig = _movie(tall)._fig

    assert _bar_orientation(movie_fig) == _bar_orientation(still) == "vertical"
    # and the figure the frames are drawn into matches the still's shape
    assert tuple(movie_fig.get_size_inches()) == pytest.approx(
        tuple(still.get_size_inches()), rel=0.02
    )


def test_a_movie_still_honours_an_explicit_colorbar_orientation():
    """The caller's own choice wins, and sizes the figure — as it does for a row."""
    tall = [_tall_frame(i) for i in range(2)]
    beside = _movie(tall)._fig
    below = _movie(tall, colorbar_kwargs={"orientation": "horizontal"})._fig
    assert _bar_orientation(below) == "horizontal"
    assert below.get_size_inches()[1] > beside.get_size_inches()[1]


def test_frames_on_different_grids_are_refused(frames):
    """Frames share one figure, so a grid that changes mid-movie has nowhere to go."""
    coarse = _field(0.0, shape=(6, 8))
    odd = {
        **_frame(9),
        "aligned": {
            "test": coarse,
            "reference": coarse,
            "difference": coarse - coarse,
        },
    }
    with pytest.raises(ValueError, match="same grid"):
        _movie([*frames, odd])


# --- the interactive counterpart ----------------------------------------------------


def _hv(obj):
    """Return the holoviews object, whether or not panel wraps it for its widget.

    The movie families return a ``pn.pane.HoloViews`` by default, because holoviews'
    own control for a string-valued dimension is a dropdown; ``widget="dropdown"``
    returns the bare object. Both carry the same plot underneath.
    """
    return getattr(obj, "object", obj)


def _holomaps(obj):
    import holoviews as hv

    return [el for el in _hv(obj).traverse() if isinstance(el, hv.HoloMap)]


def test_the_interactive_movie_puts_every_frame_on_one_slider(frames):
    maps = _holomaps(_interactive(frames))
    assert len(maps) == 3, "test, reference and difference each need a HoloMap"
    for holomap in maps:
        assert [d.name for d in holomap.kdims] == ["frame"]
        assert len(holomap.keys()) == len(frames)


def test_the_interactive_slider_keeps_the_frames_in_order(frames):
    """A HoloMap sorts its keys, so author order has to be pinned on the dimension.

    Without that, '2012-01-10' would sort before '2012-01-02' — and a movie of depths
    or runs would be reordered alphabetically into nonsense.
    """
    holomap = _holomaps(_interactive(frames))[0]
    assert list(holomap.keys()) == [f["frame_label"] for f in frames]


def _clim(element) -> tuple[float, float]:
    """Return the colour limits one interactive panel will actually be drawn with.

    Read off the value dimension's range, which is where hvplot resolves ``clim`` to —
    and which the bokeh ``LinearColorMapper``'s ``low``/``high`` come from, so this is
    what a viewer sees rather than what was asked for. A geo panel is an ``Overlay`` of
    the mesh and its coastline, so the mesh has to be found inside it.
    """
    for node in element.traverse():
        if getattr(node, "vdims", None):
            return node.vdims[0].range
    raise AssertionError(f"no value dimension found in {element!r}")


def test_the_interactive_movie_fixes_one_colour_scale_too(frames):
    """Parity with the static side: dragging the slider must not move the ruler."""
    holomap = _holomaps(_interactive(frames))[0]
    clims = {_clim(el) for el in holomap.values()}
    assert len(clims) == 1, f"the scale moves between frames: {clims}"


def test_the_interactive_and_static_scales_agree(frames):
    """The two renderers must put the same numbers on the bar, not merely fix one."""
    holomap = _holomaps(_interactive(frames))[0]
    interactive = _clim(next(iter(holomap.values())))
    assert interactive == pytest.approx(_norms(_movie(frames))[0])


def test_the_interactive_titles_carry_the_frame_and_the_metrics(frames):
    import holoviews as hv
    from bokeh.plotting import figure

    obj = _hv(_interactive(frames))
    titles = [
        f.title.text for f in hv.render(obj, backend="bokeh").select({"type": figure})
    ]
    assert any("2012-01-01" in t for t in titles), titles
    assert any("bias=" in t for t in titles), titles


def test_the_interactive_movie_writes_a_standalone_page(frames, tmp_path):
    out = tmp_path / "movie.html"
    _interactive(frames, save=out)
    assert out.exists()
    assert "html" in out.read_text(errors="ignore")[:2000].lower()


def test_the_interactive_movie_refuses_a_video_extension(frames, tmp_path):
    """It has no encoder, so it should name the renderer that does."""
    with pytest.raises(ValueError, match="renderer='matplotlib'"):
        _interactive(frames, save=tmp_path / "movie.mp4")


def test_the_frames_get_a_slider_not_a_dropdown(frames):
    """An ordered sequence wants a control you can drag, not one you search.

    Holoviews picks a dropdown for a string-valued dimension, which makes stepping to
    the next frame two clicks and makes dragging through the movie impossible.
    """
    import panel as pn

    pane = _interactive(frames)
    assert isinstance(pane, pn.pane.HoloViews)
    assert [type(w).__name__ for w in pane.widget_box] == ["DiscreteSlider"]


def test_a_player_is_available_for_when_it_should_just_run(frames):
    pane = _interactive(frames, widget="player", fps=4)
    players = [w for w in pane.widget_box if hasattr(w, "interval")]
    assert players, "widget='player' has to produce a widget that plays"
    assert players[0].interval == 250, "and it plays at the fps the static side encodes"


def test_the_holoviews_default_control_is_still_reachable(frames):
    """``dropdown`` returns the bare holoviews object, as every other family does."""
    import holoviews as hv

    assert isinstance(_interactive(frames, widget="dropdown"), hv.Layout)


def test_an_unknown_widget_is_refused(frames):
    with pytest.raises(ValueError, match="unknown widget"):
        _interactive(frames, widget="knob")


def test_static_only_styling_still_warns_here(frames):
    with pytest.warns(UserWarning, match="only affect the static"):
        _interactive(frames, frame_label_kwargs={"fontsize": 14})


# --- the model-only movie: one field, its facet axis played --------------------------
#
# The counterpart of field_facet rather than of field_grid: one panel, no reference, no
# difference, no metrics. What has to hold is that a movie and a facet grid of the same
# field agree — same labels, same one colour scale — since they are one field read two
# ways and a reader will compare them.


def _run(days: int = 6, *, depths=None) -> xr.DataArray:
    """``days`` of daily output, optionally with a depth axis left standing too."""
    import pandas as pd

    rng = np.random.default_rng(0)
    coords = {
        "time": pd.date_range("2012-01-01", periods=days, freq="D"),
        "lat": np.linspace(18, 31, 12),
        "lon": np.linspace(-98, -80, 20),
    }
    dims = ("time", "lat", "lon")
    shape = (days, 12, 20)
    if depths is not None:
        coords["depth"] = list(depths)
        dims = ("time", "depth", "lat", "lon")
        shape = (days, len(depths), 12, 20)
    return xr.DataArray(
        rng.normal(5.0, 1.0, shape),
        dims=dims,
        coords=coords,
        attrs={"units": "mmol m-3"},
    )


def _facet_item(field, facet_dim="time") -> dict:
    """One spec item shaped as ``Field.as_item()`` builds it."""
    return {
        "field": field,
        "facet_dim": facet_dim,
        "row_dim": None,
        "units": "mmol m-3",
        "standard_name": NITRATE,
        "label": "GOM_bgc",
    }


def _facet_film(field, *, renderer="matplotlib", facet_dim="time", **options):
    return render(
        PlotSpec(
            family="facet_movie", items=[_facet_item(field, facet_dim)], options=options
        ),
        renderer=renderer,
    )


def test_a_field_movie_is_one_panel_per_frame(tmp_path):
    out = tmp_path / "run.gif"
    ani = _facet_film(_run(4), save=out, domain=None)
    assert out.read_bytes()[:4] == b"GIF8"
    assert ani._save_count == 4
    # one map, one colorbar — no reference and no difference panel
    assert len(ani._fig.axes) == 2


def test_a_field_movie_labels_its_frames_like_the_facet_grid_titles(tmp_path):
    """Same field, same labels — a movie and a grid of it must not disagree."""
    from ocean_skill.plot.matplotlib_renderer import facet_labels, frame_labels

    monthly = _run(3).resample(time="1MS").mean()
    assert frame_labels(monthly["time"]) == facet_labels(monthly["time"])
    ani = _facet_film(monthly, domain=None)
    ani._func(0)
    texts = [t.get_text() for ax in ani._fig.axes for t in ax.texts]
    assert "Jan 2012" in texts


def test_frame_labels_get_finer_when_a_month_is_not_enough():
    """31 frames all called 'Jan 2012' would be a slider that loses 30 of them.

    ``"%b %Y"`` is right for the reduction it was written for (monthly means), and a
    movie is as often over the raw axis, so the label has to refine itself — the same
    refinement a facet grid of the same axis gets, which is why the two agree here.
    """
    from ocean_skill.plot.matplotlib_renderer import facet_labels, frame_labels

    daily = _run(31)
    labels = frame_labels(daily["time"])
    assert labels == facet_labels(daily["time"])
    assert len(set(labels)) == 31
    assert labels[0] == "2012-01-01"


def test_a_field_movie_fixes_one_colour_scale(tmp_path):
    ani = _facet_film(_run(4), domain=None)
    before = ani._fig.axes[0].collections[0].norm.vmin
    for index in range(4):
        ani._func(index)
    assert ani._fig.axes[0].collections[0].norm.vmin == before


def test_a_field_movie_pins_its_title_position(tmp_path):
    """Automatic title placement must be off, or matplotlib 3.11 crops the map away.

    A movie's panel has no title text — the frame label is a box inside the panel — and
    the obvious way to write that, skipping ``set_title`` altogether, leaves automatic
    title placement switched on. Over a cartopy GeoAxes carrying gridline labels that
    computes an infinite y on 3.11, giving the title a NaN extent, the axes a NaN tight
    bbox, and ``bbox_inches="tight"`` a figure with the map dropped out of it: Jupyter
    renders the colorbar alone. Asserted on ``_autotitlepos`` rather than on the crop
    because the crop only reproduces on 3.11 while the cause is visible on any version —
    which is exactly how this shipped.
    """
    ani = _facet_film(_run(4), domain=None)
    maps = [ax for ax in ani._fig.axes if not getattr(ax, "_osk_cbar_parents", None)]
    assert maps and all(ax._autotitlepos is False for ax in maps)


def test_a_field_movie_survives_a_tight_bbox_crop(tmp_path):
    """The crop itself, for the versions that reproduce it: the map must still be there.

    A colorbar-only strip is far wider than it is tall, so the aspect ratio tells the
    two apart without needing to inspect pixels.
    """
    from PIL import Image

    out = tmp_path / "frame.png"
    ani = _facet_film(_run(4), domain=None)
    ani._fig.savefig(out, bbox_inches="tight", dpi=80)
    width, height = Image.open(out).size
    assert width / height < 3, "the map was cropped out; only the colorbar survived"


def test_a_single_map_has_nothing_to_play():
    flat = _run(4).mean("time")
    with pytest.raises(ValueError, match="needs an axis to play"):
        _facet_film(flat, facet_dim=None, domain=None)


def test_a_second_facet_axis_is_refused_rather_than_animated():
    """A depth axis left standing beside time would make a frame 3-D, not a map."""
    with pytest.raises(ValueError, match="still stands beyond"):
        _facet_film(_run(4, depths=(0.0, 50.0)), domain=None)


def test_the_interactive_field_movie_puts_the_facet_axis_on_a_slider():
    movie = _hv(_facet_film(_run(5), renderer="holoviews", domain=None))
    assert [d.name for d in movie.kdims] == ["frame"]
    assert list(movie.keys()) == [f"2012-01-0{i}" for i in range(1, 6)]


def test_a_field_movie_names_its_variable_in_both_renderers():
    """The frames say when; the variable name says what — as on the facet grid.

    Where it sits differs by necessity, not by choice: matplotlib has a suptitle above
    the map, bokeh's only title is the panel's own, so there the name joins the frame
    label rather than replacing it.
    """
    ani = _facet_film(_run(3), domain=None)
    assert ani._fig._suptitle.get_text() == "nitrate"

    import holoviews as hv
    from bokeh.plotting import figure

    movie = _hv(_facet_film(_run(3), renderer="holoviews", domain=None))
    titles = [
        f.title.text
        for el in movie.values()
        for f in hv.render(el, backend="bokeh").select({"type": figure})
    ]
    assert titles[0] == "nitrate — 2012-01-01"
    assert len(set(titles)) == 3, "the name must join the frame labels, not replace them"


def test_an_explicit_movie_title_still_wins():
    ani = _facet_film(_run(3), domain=None, title="GOM run")
    assert ani._fig._suptitle.get_text() == "GOM run"
    assert _facet_film(_run(3), domain=None, title="")._fig._suptitle is None


def test_the_interactive_field_movie_fixes_one_colour_scale():
    movie = _hv(_facet_film(_run(5), renderer="holoviews", domain=None))
    assert len({_clim(el) for el in movie.values()}) == 1


def test_both_renderers_agree_on_the_field_movie_scale():
    field = _run(5)
    static = _facet_film(field, domain=None)._fig.axes[0].collections[0].norm
    movie = _hv(_facet_film(field, renderer="holoviews"))
    interactive = _clim(next(iter(movie.values())))
    assert interactive == pytest.approx((static.vmin, static.vmax))


def test_every_thins_a_field_movie_too():
    assert _facet_film(_run(10), every=3, domain=None)._save_count == 4


def test_field_movie_routes_through_the_facet_movie_family(monkeypatch):
    """``Field.movie()`` is ``Field.plot()``'s axis played instead of laid out."""
    from ocean_skill import comparison
    from ocean_skill.field import field as make_field

    monkeypatch.setattr(comparison, "prepare_source", lambda *a, **k: (_run(4), None))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ani = make_field("stub", NITRATE).movie(domain=None)
    assert ani._save_count == 4


def test_nothing_is_reduced_unless_asked(monkeypatch):
    """A month selected is a month of frames: no reduction happens behind your back.

    This used to default to ``{"time": "mean"}``, so selecting January silently returned
    its mean and a movie had nothing to play. Pinned at the ``_prepare`` level because
    that is where the default lived, and a movie is only the first thing to notice it.
    """
    from ocean_skill.comparison import NO_AGGREGATION, _prepare

    assert NO_AGGREGATION == {}
    ds = _run(31).rename("salt").to_dataset()
    ds["salt"].attrs["standard_name"] = NITRATE
    select = {"time": "2012-01"}

    for label, agg in (("unset", None), ("explicit", {})):
        kept, _ = _prepare(ds, {}, NITRATE, select, agg)
        assert kept.sizes["time"] == 31, f"aggregate {label} reduced something"
    collapsed, _ = _prepare(ds, {}, NITRATE, select, {"time": "mean"})
    assert "time" not in collapsed.dims, "an explicit mean must still collapse"


def test_the_collapsed_field_error_names_the_way_out():
    """The error has to name a reduction that keeps the axis, not just refuse."""
    with pytest.raises(ValueError) as excinfo:
        _facet_film(_run(4).mean("time"), facet_dim=None, domain=None)
    message = str(excinfo.value)
    assert "resample" in message
    assert "every step" in message


def test_a_field_with_no_axis_left_says_so_from_movie(monkeypatch):
    from ocean_skill import comparison
    from ocean_skill.field import field as make_field

    monkeypatch.setattr(
        comparison, "prepare_source", lambda *a, **k: (_run(4).mean("time"), None)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="needs an axis to play"):
            make_field("stub", NITRATE).movie(domain=None)
