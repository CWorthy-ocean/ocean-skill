"""Suite report layout: figures packed onto PDF pages, ``max_rows``, captions, defaults.

Three parts. ``PdfReport`` packs consecutive figures (and a divider's header) onto one
Letter page and draws a caption under a figure. ``pages.build`` cuts a compare page's
long row grid into figures of at most ``plot: {max_rows: N}`` rows. And the page keys
that ride along: ``caption:``, ``defaults.plot`` (kept off pages whose draw method does
not take it) and ``compare: {over: ...}``.
"""

from __future__ import annotations

import json
import types
import warnings

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib.figure import Figure

from ocean_skill.config import SuiteConfig
from ocean_skill.suites import pages as P
from ocean_skill.suites import report as _report
from ocean_skill.suites.report import CAPTION_WRAP, PdfReport
from ocean_skill.suites.run import run_suite
from tests.test_suites_report import (
    _big,
    _figure,
    _full_bleed_figure,
    _log_pdf_writes,
    _mediaboxes,
    _small,
)
from tests.test_suites_run import (  # noqa: F401  (stub_model is a fixture)
    _model_only_suite,
    _write_suite,
    stub_model,
)

INDEX = pd.date_range("2010-01-05", periods=10, freq="7D")  # Jan 5 .. Mar 9, 2010


# ======================================================================================
# PdfReport: packing and captions
# ======================================================================================


def _pages_in(tmp_path) -> int:
    return len(_mediaboxes(tmp_path / "report.pdf"))


def test_figures_that_fit_share_a_page_and_the_next_one_starts_another(tmp_path):
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        for i in range(5):  # four fit (about 2.2in each), the fifth does not
            report.emit(_small(i), f"f{i}")

    assert _pages_in(tmp_path) == 2
    for box in _mediaboxes(tmp_path / "report.pdf"):
        assert box == pytest.approx((0.0, 0.0, 612.0, 792.0), abs=0.5)
    # the PNGs are one per figure, as ever
    assert [p.name for p in report.png_paths] == [
        f"0{i}_f{i - 1}.png" for i in range(1, 6)
    ]


def test_packing_is_greedy_and_keeps_the_order(tmp_path):
    """Big+small fill page 1 (a second small would overflow it); small+big page 2."""
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        for name, fig in (
            ("a", _big()),
            ("b", _small()),
            ("c", _small()),
            ("d", _big()),
        ):
            report.emit(fig, name)
    assert _pages_in(tmp_path) == 2


def test_a_page_with_no_room_to_grow_is_still_a_vector_page(tmp_path, monkeypatch):
    kinds = []
    real_alone, real_packed = PdfReport._write_alone, PdfReport._write_packed
    monkeypatch.setattr(
        PdfReport,
        "_write_alone",
        lambda self, *a: (kinds.append("vector"), real_alone(self, *a))[1],
    )
    monkeypatch.setattr(
        PdfReport,
        "_write_packed",
        lambda self, items, *a: (
            kinds.append((len(items), round(a[-1], 2))),
            real_packed(self, items, *a),
        )[1],
    )
    full = (7.7, 9.5)  # 7.9 x 9.7in with its pad: under 10% from filling either way
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        report.emit(_full_bleed_figure(full), "a")  # nothing fits beside it
        report.emit(_full_bleed_figure(full), "b")  # also alone: not enlarged
        report.emit(_small(), "c")  # c, d and e share a page, filled to its height
        report.emit(_small(), "d")
        report.emit(_small(), "e")
        report.emit(_full_bleed_figure(full), "f")
    assert kinds[0] == "vector" and kinds[1] == "vector" and kinds[-1] == "vector"
    ((n, fill),) = kinds[2:-1]
    assert n == 3 and 1.0 < fill <= _report.MAX_FILL_SCALE


def test_a_figure_wider_than_the_page_is_scaled_to_the_page_width(
    tmp_path, monkeypatch
):
    seen = []
    real = PdfReport._write_packed
    monkeypatch.setattr(
        PdfReport,
        "_write_packed",
        lambda self, items, *a: (seen.extend(items), real(self, items, *a)),
    )
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        report.emit(_full_bleed_figure((8.4, 2.0)), "a")  # PNG 8.6in wide with its pad
        report.emit(_full_bleed_figure((8.4, 2.0)), "b")

    assert _pages_in(tmp_path) == 1 and len(seen) == 2
    for item in seen:
        assert item.w <= 8.5 - 2 * _report.SIDE_MARGIN + 1e-9
        assert item.w / item.h == pytest.approx(8.6 / 2.2, rel=0.01)  # aspect kept


def test_a_figure_too_tall_for_a_page_even_scaled_is_never_packed(tmp_path):
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        report.emit(_small(), "before")
        with pytest.warns(UserWarning, match="does not fit"):
            report.emit(_full_bleed_figure((12.0, 20.0)), "wide")
        report.emit(_small(), "after")
    assert _pages_in(tmp_path) == 3


def _capture_packed(monkeypatch):
    """Record each packed page's items as ``(item.w, item.h, item.scaled)`` tuples."""
    pages = []
    real = PdfReport._write_packed
    monkeypatch.setattr(
        PdfReport,
        "_write_packed",
        lambda self, items, *a: (
            pages.append([(i.w, i.h, i.nat_h, i.scaled) for i in items]),
            real(self, items, *a),
        )[1],
    )
    return pages


def test_figures_a_hair_wider_than_the_page_are_scaled_and_packed(
    tmp_path, monkeypatch
):
    """Two 3.5in-tall maps 8.58in wide (page 8.5) share a page, not one apiece."""
    pages = _capture_packed(monkeypatch)
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        for i in range(2):
            report.emit(_full_bleed_figure((8.58, 3.5)), f"m{i}")

    assert _pages_in(tmp_path) == 1
    (page,) = pages
    assert len(page) == 2
    for w, h, nat_h, scaled in page:
        assert scaled and w <= 8.5 - 2 * _report.SIDE_MARGIN + 1e-9
        assert h < nat_h  # shrunk with the width, aspect kept
        assert w / h == pytest.approx((8.58 + 0.2) / (3.5 + 0.2), rel=0.01)


def test_a_lone_figure_a_hair_too_wide_is_scaled_onto_a_letter_page(tmp_path):
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        report.emit(_full_bleed_figure((8.58, 3.5)), "wide")
    (box,) = _mediaboxes(tmp_path / "report.pdf")
    assert box == pytest.approx((0.0, 0.0, 612.0, 792.0), abs=0.5)  # not grown


def test_a_wide_figure_still_too_tall_after_scaling_gets_a_page_to_itself(
    tmp_path, monkeypatch
):
    pages = _capture_packed(monkeypatch)
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        report.emit(_small(), "before")
        with pytest.warns(UserWarning, match="does not fit"):
            report.emit(_full_bleed_figure((10.0, 14.0)), "tall")  # 8 x 11.2 scaled
        report.emit(_small(), "after")
    assert _pages_in(tmp_path) == 3
    # all three alone: the small ones filled (packed pages), the tall one vector
    assert len(pages) == 2 and all(len(page) == 1 for page in pages)


def test_a_figure_is_scaled_to_fit_under_a_divider_when_it_barely_must(
    tmp_path, monkeypatch
):
    pages = _capture_packed(monkeypatch)
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        report.section(1, "Part one", "Three lines\nof notes\nhere.")
        report.emit(_full_bleed_figure((6.0, 9.2)), "tall")  # 9.4in with its pad

    assert _pages_in(tmp_path) == 1  # not a divider page and a figure page
    assert report.written_sections == {1}
    ((w, h, nat_h, scaled),) = pages[0]
    assert scaled and _report.MIN_FIT_SCALE <= h / nat_h < 1
    assert w / h == pytest.approx(6.2 / 9.4, rel=0.01)  # aspect kept
    assert (8.5 - w) / 2 > 0  # centred by construction: equal side margins


def test_a_figure_that_would_shrink_below_the_minimum_leaves_the_divider_alone(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(_report, "MIN_FIT_SCALE", 0.7)
    pages = _capture_packed(monkeypatch)
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        report.section(1, "Part one", "\n".join(["A line of notes."] * 12))
        report.emit(_full_bleed_figure((6.0, 9.8)), "tall")  # needs about 0.65
    assert _pages_in(tmp_path) == 2 and pages == []
    assert report.written_sections == {1}


def test_the_caption_counts_when_a_figure_is_fitted_under_a_divider(
    tmp_path, monkeypatch
):
    pages = _capture_packed(monkeypatch)
    caption = "A long caption that wraps onto a few lines. " * 8
    with PdfReport(tmp_path / "plain.pdf", tmp_path / "f1") as report:
        report.section(1, "Part", "Notes")
        report.emit(_full_bleed_figure((6.0, 9.0)), "tall")
    with PdfReport(tmp_path / "caps.pdf", tmp_path / "f2") as report:
        report.section(1, "Part", "Notes")
        report.emit(_full_bleed_figure((6.0, 9.0)), "tall", caption=caption)
        # the caption is under the figure inside the page, not past its bottom edge
        assert report._used <= 11.0 - _report._top_margin() - _report.PAGE_BOTTOM + 1e-9
    (plain,), (captioned,) = pages
    assert captioned[1] < plain[1]  # the figure gave way to its caption


def test_a_caption_that_would_not_fit_moves_the_next_figure_not_over_it(
    tmp_path, monkeypatch
):
    pages = _capture_packed(monkeypatch)
    caption = "Caption line.\n" * 10  # ten lines, about 1.8in
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        report.emit(_full_bleed_figure((6.0, 4.0)), "a", caption=caption)
        report.emit(_full_bleed_figure((6.0, 4.0)), "b", caption=caption)
        # 2 x (4.2 + 1.8) + gap would overflow a page: b must not share it with a
    assert _pages_in(tmp_path) == 2
    assert all(len(page) == 1 for page in pages)


def test_writing_a_page_later_does_not_repeat_its_warnings_on_a_later_page(tmp_path):
    """Layout warnings belong to the page that drew them (``--strict`` goes by time)."""
    import matplotlib.artist

    class Noisy(matplotlib.artist.Artist):
        def draw(self, renderer):
            warnings.warn("noisy layout")

    noisy = _big()
    noisy.add_artist(Noisy())
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        with pytest.warns(UserWarning, match="noisy layout"):
            report.emit(noisy, "noisy")
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            report.emit(_big(), "clean")  # closes the first page, drawing it again
    assert not [w for w in seen if "noisy layout" in str(w.message)]


def test_a_caption_is_drawn_under_a_lone_figure_and_wrapped(tmp_path):
    caption = "Mean nitrate against WOA23, surface. " * 8
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        log = _log_pdf_writes(report)
        report.emit(_big(), "a", caption=caption)

    (texts,) = log.texts
    assert len(texts) == 1 and "\n" in texts[0]
    assert all(len(line) <= CAPTION_WRAP for line in texts[0].splitlines())
    assert " ".join(texts[0].split()) == caption.strip()
    assert _pages_in(tmp_path) == 1


def test_a_caption_goes_under_its_own_figure_on_a_packed_page(tmp_path):
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        log = _log_pdf_writes(report)
        report.emit(_small(), "a", caption="first")
        report.emit(_small(), "b")
        report.emit(_small(), "c", caption="third")

    assert len(log) == 1  # all three on one page
    assert sorted(t for t in log.texts[0]) == ["first", "third"]


def test_captions_take_room_so_fewer_figures_fit(tmp_path):
    caption = "A caption that runs to several lines. " * 12  # about six lines
    with PdfReport(tmp_path / "plain.pdf", tmp_path / "f1") as report:
        for i in range(4):
            report.emit(_small(), f"f{i}")
    with PdfReport(tmp_path / "captioned.pdf", tmp_path / "f2") as report:
        for i in range(4):
            report.emit(_small(), f"f{i}", caption=caption)

    assert len(_mediaboxes(tmp_path / "plain.pdf")) == 1
    assert len(_mediaboxes(tmp_path / "captioned.pdf")) == 2


def test_the_png_never_carries_the_caption(tmp_path):
    with PdfReport(tmp_path / "a.pdf", tmp_path / "plain") as report:
        report.emit(_figure(), "f")
    with PdfReport(tmp_path / "b.pdf", tmp_path / "caps") as report:
        report.emit(_figure(), "f", caption="a caption")
    plain = mpimg.imread(tmp_path / "plain" / "01_f.png")
    caps = mpimg.imread(tmp_path / "caps" / "01_f.png")
    assert plain.shape == caps.shape and np.array_equal(plain, caps)


def test_a_caption_is_a_no_op_without_a_pdf(tmp_path):
    with PdfReport(None, tmp_path / "figures") as report:
        report.emit(_figure(), "f", caption="ignored")
    assert [p.name for p in report.png_paths] == ["01_f.png"]


def test_a_divider_heads_the_page_its_figures_share(tmp_path):
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        log = _log_pdf_writes(report)
        report.section(1, "Part one", "Notes for the part.")
        report.emit(_small(), "a", caption="cap")
        report.emit(_small(), "b")

    assert log == ["divider:Part one"] and _pages_in(tmp_path) == 1
    assert "Part one" in log.texts[0] and "cap" in log.texts[0]


def test_a_packed_page_survives_the_jpeg_pass(tmp_path):
    import shutil

    if shutil.which("gs") is None:
        pytest.skip("Ghostscript (gs) not installed")
    from tests.test_suites_report import _map_like_figure

    pdf = tmp_path / "report.pdf"
    with PdfReport(pdf, tmp_path / "figures") as report:
        report.emit(_map_like_figure(0), "a")
        report.emit(_map_like_figure(1), "b")
    assert _pages_in(tmp_path) == 1
    sizes = _report.compress_pdf_images(pdf)
    assert sizes is not None and sizes[1] < sizes[0]
    assert _pages_in(tmp_path) == 1


# -- filling the page ----------------------------------------------------------------

#: The page's usable height, the room a stack fills (see ``_fill_scale``).
_ROOM = 11.0 - _report._top_margin() - _report.PAGE_BOTTOM


def _written_pages(report):
    """Collect every page figure handed to ``PdfPages.savefig``, in order."""
    pages = []
    real = report._pdf.savefig

    def savefig(fig, *args, **kwargs):
        pages.append(fig)
        return real(fig, *args, **kwargs)

    report._pdf.savefig = savefig
    return pages


def _boxes(page):
    """Each image's ``(x0, y0, w, h)`` in inches on a packed page, top to bottom."""
    out = []
    for ax in page.axes:
        pos = ax.get_position()
        out.append((pos.x0 * 8.5, pos.y0 * 11.0, pos.width * 8.5, pos.height * 11.0))
    return sorted(out, key=lambda b: -b[1])


def _fill_one(tmp_path, figsize, **kw):
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        pages = _written_pages(report)
        report.emit(_full_bleed_figure(figsize), "a", **kw)
    (page,) = pages
    (box,) = _boxes(page)
    return page, box


def test_a_lone_small_figure_is_scaled_up_to_the_cap_and_kept_in_proportion(tmp_path):
    _, (x0, y0, w, h) = _fill_one(tmp_path, (2.8, 1.8))  # 3.0 x 2.0in with its pad
    assert h == pytest.approx(2.0 * _report.MAX_FILL_SCALE)  # room for more: capped
    assert w / h == pytest.approx(3.0 / 2.0, rel=0.01)  # aspect kept
    assert x0 + w / 2 == pytest.approx(8.5 / 2)  # centred
    assert y0 + h == pytest.approx(11.0 - _report._top_margin())  # still at the top


def test_a_lone_tall_figure_is_scaled_up_until_the_page_height_is_used(tmp_path):
    _, (_, y0, w, h) = _fill_one(tmp_path, (4.8, 7.8))  # 5.0 x 8.0: 1.26 to fill
    assert h == pytest.approx(_ROOM) and y0 == pytest.approx(_report.PAGE_BOTTOM)
    assert w / h == pytest.approx(5.0 / 8.0, rel=0.01)
    assert w < 8.5 - 2 * _report.SIDE_MARGIN


def test_a_lone_wide_figure_is_scaled_up_until_the_page_width_is_used(tmp_path):
    _, (_, _, w, h) = _fill_one(tmp_path, (7.0, 3.0))  # 7.2 x 3.2: 1.11 to fill
    assert w == pytest.approx(8.5 - 2 * _report.SIDE_MARGIN)
    assert h == pytest.approx(3.2 * w / 7.2, rel=0.01)  # not height-limited: room left
    assert h < _ROOM


def test_a_scaled_up_figure_is_drawn_from_enough_pixels_to_stay_sharp(tmp_path):
    page, (_, _, w, _) = _fill_one(tmp_path, (2.8, 1.8))  # scaled by the cap
    (image,) = page.axes[0].images
    pixels_wide = image.get_array().shape[1]
    assert pixels_wide / w >= _report.PNG_DPI - 1  # on paper, as sharp as its PNG


def test_two_packed_figures_scale_by_the_same_factor_and_stay_on_the_page(tmp_path):
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        pages = _written_pages(report)
        report.emit(_full_bleed_figure((4.8, 3.8)), "a")  # 5.0 x 4.0in each
        report.emit(_full_bleed_figure((3.8, 2.8)), "b")  # 4.0 x 3.0in
    (page,) = pages
    top, bottom = _boxes(page)
    scale = top[3] / 4.0
    assert scale > 1.0
    for box, (nat_w, nat_h) in ((top, (5.0, 4.0)), (bottom, (4.0, 3.0))):
        # the PNG's pad is rounded to a pixel, hence the tolerance
        assert box[3] / nat_h == pytest.approx(scale, rel=0.01)
        assert box[2] / nat_w == pytest.approx(scale, rel=0.01)  # proportions kept
    # the stack uses the whole room: the gap between is the natural one, the last
    # image ends at the bottom margin, and nothing is wider than the usable width
    assert top[1] - (bottom[1] + bottom[3]) == pytest.approx(_report.PACK_GAP)
    assert bottom[1] == pytest.approx(_report.PAGE_BOTTOM)
    assert all(b[2] <= 8.5 - 2 * _report.SIDE_MARGIN + 1e-9 for b in (top, bottom))
    assert all(b[0] + b[2] / 2 == pytest.approx(8.5 / 2) for b in (top, bottom))


def test_a_page_is_limited_by_its_widest_figure(tmp_path):
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        pages = _written_pages(report)
        report.emit(_full_bleed_figure((7.0, 1.8)), "wide")  # 7.2in: 1.11 to fill
        report.emit(_full_bleed_figure((2.8, 1.8)), "narrow")
    (page,) = pages
    wide, narrow = _boxes(page)
    assert wide[2] == pytest.approx(8.0)
    assert narrow[3] / 2.0 == pytest.approx(8.0 / 7.2, rel=0.01)  # the same factor


def test_captions_and_header_keep_their_size_and_never_overlap_the_figures(tmp_path):
    caption = "A caption that wraps onto a couple of lines. " * 4
    pdfs = {}
    for name, cap in (("fill", _report.MAX_FILL_SCALE), ("plain", 1.0)):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_report, "MAX_FILL_SCALE", cap)
            with PdfReport(tmp_path / f"{name}.pdf", tmp_path / name) as report:
                pages = _written_pages(report)
                report.section(1, "Part one", "Some notes.")
                report.emit(_full_bleed_figure((4.8, 2.8)), "a", caption=caption)
                report.emit(_full_bleed_figure((4.8, 2.8)), "b", caption=caption)
        (pdfs[name],) = pages
    filled, plain = pdfs["fill"], pdfs["plain"]
    sizes = lambda page: [(t.get_text(), t.get_fontsize()) for t in page.texts]  # noqa: E731
    assert sizes(filled) == sizes(plain)  # same words, same type size

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    renderer = FigureCanvasAgg(filled).get_renderer()
    boxes = _boxes(filled)
    caps = [t for t in filled.texts if "caption" in t.get_text()]
    caps.sort(key=lambda t: -t.get_window_extent(renderer).y1)
    inch = lambda px: px / filled.dpi  # noqa: E731
    title = max(filled.texts, key=lambda t: t.get_fontsize())
    # top to bottom: title, image, caption, image, caption, bottom margin
    assert inch(title.get_window_extent(renderer).y0) >= boxes[0][1] + boxes[0][3]
    for (x0, y0, w, h), cap in zip(boxes, caps, strict=True):
        ext = cap.get_window_extent(renderer)
        assert inch(ext.y1) <= y0 + 1e-6  # the caption hangs under its own image
    assert inch(caps[0].get_window_extent(renderer).y0) >= boxes[1][1] + boxes[1][3]
    assert inch(caps[1].get_window_extent(renderer).y0) >= _report.PAGE_BOTTOM - 1e-6
    assert boxes[0][3] > 2.8  # and the figures did grow


def test_filling_never_moves_a_figure_to_another_page_or_touches_the_png(tmp_path):
    def run(name, cap):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_report, "MAX_FILL_SCALE", cap)
            with PdfReport(tmp_path / f"{name}.pdf", tmp_path / name) as report:
                for i in range(7):
                    report.emit(_small(i), f"f{i}")
        return report

    filled, plain = run("fill", _report.MAX_FILL_SCALE), run("plain", 1.0)
    assert _pages_in_file(tmp_path / "fill.pdf") == _pages_in_file(
        tmp_path / "plain.pdf"
    )
    for a, b in zip(filled.png_paths, plain.png_paths, strict=True):
        assert a.read_bytes() == b.read_bytes()


def _pages_in_file(path) -> int:
    return len(_mediaboxes(path))


def test_an_oversize_figure_is_not_filled(tmp_path):
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        with pytest.warns(UserWarning, match="does not fit"):
            report.emit(_full_bleed_figure((10.0, 14.0)), "tall")
        assert _report._fill_scale(report._items, None) == 1.0


def test_enlarging_a_figure_does_not_repeat_its_layout_warnings(tmp_path):
    import matplotlib.artist

    class Noisy(matplotlib.artist.Artist):
        def draw(self, renderer):
            warnings.warn("noisy layout")

    counts = {}
    for name, cap in (("fill", _report.MAX_FILL_SCALE), ("plain", 1.0)):
        fig = _full_bleed_figure((2.8, 1.8))  # enlarged, so rendered a second time
        fig.add_artist(Noisy())
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_report, "MAX_FILL_SCALE", cap)
            with (
                PdfReport(tmp_path / f"{name}.pdf", tmp_path / name) as report,
                warnings.catch_warnings(record=True) as seen,
            ):
                warnings.simplefilter("always")
                report.emit(fig, "noisy")
        counts[name] = len([w for w in seen if "noisy layout" in str(w.message)])
    assert counts["fill"] == counts["plain"] > 0


# ======================================================================================
# pages: max_rows
# ======================================================================================


@pytest.fixture(autouse=True)
def _time_index(monkeypatch):
    monkeypatch.setattr("ocean_skill.extrema._native_time_index", lambda source: INDEX)


def _suite(pages, **top):
    return SuiteConfig.model_validate({"name": "t", "pages": pages, **top})


def _expand(page, **top):
    top.setdefault("defaults", {"test": "stub"})
    return P.expand(_suite([page], **top))


class _Comp:
    def __init__(self, family, index, tag="test"):
        self.family, self.index, self.tag = family, index, tag

    def metrics(self):
        return {"index": self.index}


def _fake_osk(monkeypatch, families, *, suptitle="surface", contour_families=None):
    """Stub osk.compare/ComparisonSet with real figures; returns the recorded calls."""
    import ocean_skill as osk

    calls = types.SimpleNamespace(compare=[], plot=[], nan_checks=0)

    class FakeSet(list):
        def plot(self, **opts):
            calls.plot.append((list(self), opts))
            fig = Figure()
            if "title" in opts:
                fig.suptitle(opts["title"])
            elif suptitle:
                fig.suptitle(suptitle)
            return fig

    def fake_compare(**kwargs):
        calls.compare.append(kwargs)
        fams = families if len(calls.compare) == 1 else contour_families
        tag = "test" if len(calls.compare) == 1 else "contour"
        return FakeSet(_Comp(f, i, tag) for i, f in enumerate(fams))

    def no_nan(c):
        calls.nan_checks += 1
        return None

    monkeypatch.setattr(osk, "compare", fake_compare)
    monkeypatch.setattr(osk, "ComparisonSet", FakeSet)
    monkeypatch.setattr(P, "_all_nan_reason", no_nan)
    return calls


def _compare_page(**extra):
    return {
        "title": "x",
        "compare": {"reference": ["woa"], "variables": ["temperature"]},
        **extra,
    }


def _suptitles(results):
    return [fig.get_suptitle() for _, fig in results]


def test_a_row_family_longer_than_max_rows_is_cut_into_figures(monkeypatch):
    calls = _fake_osk(monkeypatch, ["field_row"] * 6)
    (page,) = _expand(_compare_page())
    results = P.build(page)

    assert [s for s, _ in results] == ["", "_part2"]
    assert [len(members) for members, _ in calls.plot] == [4, 2]  # the default is 4
    # the order of the set is kept, so a variable's statistics stay together
    assert [c.index for c in calls.plot[0][0] + calls.plot[1][0]] == list(range(6))
    assert _suptitles(results) == ["surface", "surface (cont.)"]
    assert all("max_rows" not in opts for _, opts in calls.plot)


@pytest.mark.parametrize(
    "max_rows, sizes", [(2, [2, 2, 2]), (6, [6]), (0, [6]), (None, [6])]
)
def test_max_rows_is_the_pages_own_and_zero_or_null_means_no_limit(
    monkeypatch, max_rows, sizes
):
    calls = _fake_osk(monkeypatch, ["field_row"] * 6)
    (page,) = _expand(_compare_page(plot={"max_rows": max_rows}))
    results = P.build(page)
    assert [len(m) for m, _ in calls.plot] == sizes
    assert [s for s, _ in results] == ["", "_part2", "_part3"][: len(sizes)]


@pytest.mark.parametrize(
    "family", ["field_row", "time_depth", "section_row", "skill_map"]
)
def test_every_row_per_comparison_family_is_cut(monkeypatch, family):
    calls = _fake_osk(monkeypatch, [family] * 5)
    (page,) = _expand(_compare_page(plot={"max_rows": 2}))
    P.build(page)
    assert [len(m) for m, _ in calls.plot] == [2, 2, 1]


@pytest.mark.parametrize("family", ["series", "profile"])
def test_families_that_share_one_panel_are_never_cut(monkeypatch, family):
    calls = _fake_osk(monkeypatch, [family] * 9)
    (page,) = _expand(_compare_page(plot={"max_rows": 2}))
    results = P.build(page)
    assert [len(m) for m, _ in calls.plot] == [9] and len(results) == 1


def test_each_family_is_cut_on_its_own_and_the_suffixes_say_which(monkeypatch):
    _fake_osk(monkeypatch, ["field_row"] * 5 + ["section_row"] * 2)
    (page,) = _expand(_compare_page(plot={"max_rows": 4}))
    results = P.build(page)
    assert [s for s, _ in results] == [
        "_field_row",
        "_field_row_part2",
        "_section_row",
    ]


def test_the_whole_set_is_checked_and_recorded_once_not_once_per_figure(monkeypatch):
    calls = _fake_osk(monkeypatch, ["field_row"] * 9)
    (page,) = _expand(_compare_page(plot={"max_rows": 2}))
    results = P.build(page)
    assert len(results) == 5
    assert calls.nan_checks == 9  # one look at each comparison
    assert len(calls.compare) == 1
    assert [r["index"] for r in page.metrics_records] == list(range(9))


def test_the_all_nan_drop_is_done_before_the_cut(monkeypatch):
    """Two of six comparisons are all NaN: the four left are one figure, not two."""
    calls = _fake_osk(monkeypatch, ["field_row"] * 6)
    monkeypatch.setattr(
        P,
        "_all_nan_reason",
        lambda c: "all-NaN: reference has no data" if c.index in (1, 4) else None,
    )
    (page,) = _expand(_compare_page())
    with pytest.warns(UserWarning, match="dropped 2 of 6"):
        results = P.build(page)
    assert len(results) == 1 and len(calls.plot[0][0]) == 4
    assert len(page.dropped) == 2 and len(page.metrics_records) == 4


def test_a_title_the_page_gave_is_extended_before_drawing(monkeypatch):
    calls = _fake_osk(monkeypatch, ["field_row"] * 5)
    (page,) = _expand(_compare_page(plot={"title": "Nitrate", "max_rows": 4}))
    results = P.build(page)
    assert _suptitles(results) == ["Nitrate", "Nitrate (cont.)"]
    assert calls.plot[1][1]["title"] == "Nitrate (cont.)"


def test_an_explicit_empty_title_stays_empty(monkeypatch):
    _fake_osk(monkeypatch, ["field_row"] * 5)
    (page,) = _expand(_compare_page(plot={"title": "", "max_rows": 4}))
    assert _suptitles(P.build(page)) == ["", ""]


def test_a_figure_the_renderer_gave_no_title_is_drawn_once_and_left_untitled(
    monkeypatch,
):
    calls = _fake_osk(monkeypatch, ["field_row"] * 5, suptitle=None)
    (page,) = _expand(_compare_page(plot={"max_rows": 4}))
    results = P.build(page)
    assert _suptitles(results) == ["", ""]
    # one draw per chunk: no second draw of the continuation to give it a title
    assert [len(m) for m, _ in calls.plot] == [4, 1]
    assert all("title" not in o for _, o in calls.plot)


def test_a_continuation_chunk_is_not_drawn_again_when_the_method_refuses_a_title(
    monkeypatch,
):
    """A plot method that takes no ``title=`` still pages: the chunk is drawn once."""
    import ocean_skill as osk

    _fake_osk(monkeypatch, ["field_row"] * 5, suptitle=None)
    drawn = []

    class NoTitleSet(osk.ComparisonSet):
        def plot(self, **opts):
            if "title" in opts:
                raise TypeError("plot() got an unexpected keyword argument 'title'")
            drawn.append(len(self))
            return Figure()

    monkeypatch.setattr(osk, "ComparisonSet", NoTitleSet)
    (page,) = _expand(_compare_page(plot={"max_rows": 4}))
    assert len(P.build(page)) == 2
    assert drawn == [4, 1]


def _pooled_plot(kind, **extra):
    return {
        "shared": {"shared_limits": True},
        "difference": {"panels": "difference"},
    }[kind] | extra


@pytest.mark.parametrize("kind", ["shared", "difference"])
@pytest.mark.parametrize(
    "family", ["field_row", "time_depth", "section_row", "skill_map"]
)
def test_rows_on_one_scale_are_not_cut_so_the_scale_is_not_split(
    monkeypatch, family, kind
):
    """One scale over every row cannot be shared by separate figures: no cut, a warning.

    ``shared_limits=True`` pools the data panels and the differences over the whole set;
    ``panels="difference"`` pools the differences. Cut at ``max_rows``, each figure
    would find its own limits and the same variable would change scale at the break.
    """
    calls = _fake_osk(monkeypatch, [family] * 6)
    (page,) = _expand(_compare_page(plot=_pooled_plot(kind, max_rows=4)))
    with pytest.warns(UserWarning, match=r"max_rows=4 was not applied to 6"):
        results = P.build(page)
    assert [s for s, _ in results] == [""]
    assert [len(m) for m, _ in calls.plot] == [6]


def test_the_scale_is_only_pooled_when_asked(monkeypatch):
    calls = _fake_osk(monkeypatch, ["field_row"] * 6)
    (page,) = _expand(_compare_page(plot={"shared_limits": False, "panels": "all"}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        results = P.build(page)
    assert len(results) == 2 and [len(m) for m, _ in calls.plot] == [4, 2]


def test_a_pooled_scale_with_nothing_to_cut_or_no_limit_is_silent(monkeypatch):
    for rows, limit in ((3, 4), (6, 0)):
        calls = _fake_osk(monkeypatch, ["field_row"] * rows)
        (page,) = _expand(
            _compare_page(plot={"shared_limits": True, "max_rows": limit})
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            results = P.build(page)
        assert len(results) == 1 and [len(m) for m, _ in calls.plot] == [rows]


def test_contour_overlays_are_cut_with_the_fill(monkeypatch):
    calls = _fake_osk(
        monkeypatch, ["section_row"] * 5, contour_families=["section_row"] * 5
    )
    page = _compare_page(contours={"variables": ["salinity"]}, plot={"max_rows": 2})
    (p,) = _expand(page)
    P.build(p)
    for fill, opts in calls.plot:
        overlay = opts["contours"]
        assert [c.index for c in overlay] == [c.index for c in fill]
        assert all(c.tag == "contour" for c in overlay)


def test_contours_true_overlays_each_figures_own_rows(monkeypatch):
    calls = _fake_osk(monkeypatch, ["section_row"] * 3)
    (page,) = _expand(_compare_page(contours=True, plot={"max_rows": 2}))
    P.build(page)
    for fill, opts in calls.plot:
        assert opts["contours"] is not None and list(opts["contours"]) == fill


@pytest.mark.parametrize("bad", ["many", -1, 2.5, True, [3]])
def test_a_bad_max_rows_is_refused_naming_the_page(bad):
    with pytest.raises(ValueError, match=r"page 'x': plot: max_rows:"):
        _expand(_compare_page(plot={"max_rows": bad}))


@pytest.mark.parametrize(
    "page",
    [
        {"title": "x", "field": {"variables": ["temperature"]}},
        _compare_page(draw="target"),
    ],
)
def test_a_pages_own_max_rows_where_it_means_nothing_is_refused(page):
    with pytest.raises(ValueError, match=r"page 'x': plot: max_rows: only applies"):
        _expand({**page, "plot": {"max_rows": 3}})


def test_a_suite_wide_max_rows_reaches_only_the_pages_it_suits(monkeypatch):
    suite = _suite(
        [
            _compare_page(title="grid"),
            _compare_page(title="target", draw="target"),
            {"title": "field", "field": {"variables": ["temperature"]}},
            {"title": "s", "summary": {}},
        ],
        defaults={"test": "stub", "plot": {"max_rows": 3}},
    )
    grid, target, field, summary = P.expand(suite)
    assert grid.plot["max_rows"] == 3
    for page in (target, field, summary):
        assert "max_rows" not in page.plot


# ======================================================================================
# pages: a real ComparisonSet (nothing faked below the comparisons themselves)
# ======================================================================================


def _lane(seed, *, nan=False, time=True):
    lat, lon = np.linspace(18, 30, 9), np.linspace(-97, -82, 12)
    rng = np.random.default_rng(seed)
    if time:
        t = pd.date_range("2012-01-01", periods=30, freq="D")
        values = rng.normal(5.0, 0.5, (len(t), len(lat), len(lon)))
        dims, coords = ("time", "lat", "lon"), {"time": t, "lat": lat, "lon": lon}
    else:
        values = rng.normal(5.0, 0.5, (len(lat), len(lon)))
        dims, coords = ("lat", "lon"), {"lat": lat, "lon": lon}
    if nan:
        values[:] = np.nan
    return xr.DataArray(values, dims=dims, coords=coords, attrs={"units": "mmol m-3"})


@pytest.fixture
def lanes(monkeypatch):
    """Serve two lanes by name through the real ``compare()``; ``lanes[name] = da``."""
    import ocean_skill.catalog as catalog
    import ocean_skill.comparison as comparison

    pytest.importorskip("xesmf")
    served = {"model": _lane(0), "sat": _lane(1)}
    monkeypatch.setattr(
        comparison, "prepare_source", lambda source, *a, **k: (served[source], None)
    )
    monkeypatch.setattr(comparison, "_domain_of", lambda name: None)
    monkeypatch.setattr(
        catalog, "resolve", lambda name, *a, **k: types.SimpleNamespace(metadata={})
    )
    return served


_VARIABLES = ["nitrate", "phosphate", "silicate", "oxygen", "salinity"]


def test_a_real_grid_is_drawn_as_figures_of_max_rows_rows(lanes):
    lanes["model"], lanes["sat"] = _lane(0, time=False), _lane(1, time=False)
    page = _compare_page(plot={"max_rows": 2})
    page["compare"] = {
        "test": "model",
        "reference": ["sat"],
        "variables": _VARIABLES,
        "select": {"test": {}, "reference": {}},
    }
    (p,) = _expand(page)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = P.build(p)

    assert [s for s, _ in results] == ["", "_part2", "_part3"]
    first, second, last = (fig for _, fig in results)
    assert len(first.axes) == len(second.axes) == 10  # 2 rows of 5 axes
    assert first.get_suptitle() and second.get_suptitle().endswith("(cont.)")
    assert last.get_suptitle().endswith("(cont.)")
    assert len(p.metrics_records) == 5  # every comparison, once


@pytest.mark.parametrize(
    "scale", [{"shared_limits": True}, {"panels": "difference"}], ids=["data", "diff"]
)
def test_a_real_grid_on_one_scale_is_one_figure_with_one_set_of_limits(lanes, scale):
    """Five rows, ``max_rows: 2``: all stay in the one figure that owns the scale."""
    lanes["model"], lanes["sat"] = _lane(0, time=False), _lane(1, time=False)
    page = _compare_page(plot={"max_rows": 2, **scale})
    page["compare"] = {
        "test": "model",
        "reference": ["sat"],
        "variables": _VARIABLES,
        "select": {"test": {}, "reference": {}},
    }
    (p,) = _expand(page)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = P.build(p)

    assert [s for s, _ in results] == [""]
    assert any("max_rows=2 was not applied to 5" in str(w.message) for w in caught)
    (_, fig), = results
    # every row's colorbars read the same limits: data and difference alike
    bars = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]
    assert bars
    limits = {tuple(np.round(ax.get_ylim(), 6)) for ax in bars}
    assert len(limits) <= 2  # test/reference share one, differences the other


# ======================================================================================
# pages: caption:
# ======================================================================================


def test_a_caption_is_templated_like_the_title_and_recorded():
    page = {
        "title": "Nitrate ({v})",
        "field": {"variables": ["{v}"], "select": {"depth": "surface"}},
        "caption": "Surface {v}, {month.name}.",
        "for_each": {"v": ["nitrate"], "month": {"run": {"last": 1}}},
    }
    pages = _expand(page)
    assert [p.caption for p in pages] == ["Surface nitrate, March."]
    assert pages[0].as_dict()["caption"] == "Surface nitrate, March."
    json.dumps(pages[0].as_dict())


def test_a_page_without_a_caption_has_no_caption_key():
    (page,) = _expand({"title": "x", "field": {"variables": ["temperature"]}})
    assert page.caption is None and "caption" not in page.as_dict()


@pytest.mark.parametrize(
    "page",
    [
        {"title": "x", "field": {"variables": ["temperature"]}},
        _compare_page(),
        {"title": "x", "summary": {}},
        {
            "title": "x",
            "XY": {"members": {"a": {}}, "x": "salinity", "y": "temperature"},
        },
        {"title": "x", "TS": {"members": {"a": {}}}},
    ],
)
def test_every_figure_page_kind_takes_a_caption(page):
    (expanded,) = _expand({**page, "caption": "a caption"})
    assert expanded.caption == "a caption"


def test_a_section_takes_no_caption():
    with pytest.raises(Exception, match="takes no caption"):
        _suite([{"title": "Part", "section": "", "caption": "no"}])


def test_a_caption_that_is_not_text_is_refused():
    with pytest.raises(ValueError, match="caption: must resolve to text"):
        _expand(
            {
                "title": "x",
                "field": {"variables": ["temperature"]},
                "caption": "{note}",
            },
            defaults={"test": "stub", "note": ["a", "b"]},
        )


@pytest.mark.usefixtures("stub_model")
def test_run_suite_puts_the_caption_under_a_pages_first_figure_only(
    tmp_path, monkeypatch
):
    suite = _model_only_suite(tmp_path)
    suite["pages"][0]["caption"] = "A caption"
    seen = []
    real_emit = PdfReport.emit
    monkeypatch.setattr(
        PdfReport,
        "emit",
        lambda self, fig, stem, caption=None: (
            seen.append((stem, caption)),
            real_emit(self, fig, stem, caption=caption),
        )[1],
    )
    real_build = P.build
    monkeypatch.setattr(
        P,
        "build",
        lambda page, **kw: (
            [("", f) for _, f in real_build(page, **kw)] + [("_second", _figure())]
        ),
    )
    result = run_suite(_write_suite(tmp_path, suite))

    assert [c for _, c in seen] == ["A caption", None]
    manifest = json.loads(result.manifest.read_text())
    assert manifest["pages"][0]["caption"] == "A caption"


# ======================================================================================
# pages: defaults.plot
# ======================================================================================


def test_defaults_plot_merges_under_a_pages_own_plot_and_remembers_what_it_added():
    suite = _suite(
        [
            {"title": "a", "field": {"variables": ["temperature"]}},
            {
                "title": "b",
                "field": {"variables": ["temperature"]},
                "plot": {"robust": False, "ncols": 2},
            },
        ],
        defaults={"test": "stub", "plot": {"robust": True, "cmap": "viridis"}},
        pdf=False,
    )
    a, b = P.expand(suite)
    assert a.plot == {"robust": True, "cmap": "viridis"}
    assert a.default_plot == ("robust", "cmap")
    assert b.plot == {"robust": False, "cmap": "viridis", "ncols": 2}
    assert b.default_plot == ("cmap",)  # the page's own robust won
    assert "default_plot" not in a.as_dict()


def _page_with(default_plot=(), plot=None):
    return P.ExpandedPage(
        title="x",
        kind="compare",
        kwargs={},
        plot=plot or {},
        cache=False,
        default_plot=tuple(default_plot),
    )


def test_an_option_only_defaults_set_is_dropped_when_the_draw_refuses_it():
    from ocean_skill.plot.matplotlib_renderer import _check_options

    def draw(**opts):
        _check_options(
            lambda *, size=None, title=None: None, opts, name="metrics_series"
        )
        return opts

    page = _page_with(default_plot=["robust"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # inherited options are skipped silently
        out = P._draw(page, draw, {"robust": True, "size": "page"})
    assert out == {"size": "page"}


def test_an_option_the_page_wrote_is_still_an_error():
    from ocean_skill.plot.matplotlib_renderer import _check_options

    def draw(**opts):
        _check_options(lambda *, size=None: None, opts, name="metrics_series")

    page = _page_with(default_plot=["cmap"])  # robust was the page's own
    with pytest.raises(TypeError, match="'robust' is not an option"):
        P._draw(page, draw, {"robust": True, "cmap": "x"})


def test_check_options_raises_a_type_error_that_names_the_options():
    from ocean_skill.plot.matplotlib_renderer import _check_options
    from ocean_skill.plot.spec import UnusableOptions

    with pytest.raises(UnusableOptions) as caught:
        _check_options(lambda *, size=None: None, {"robust": 1, "cmap": 2, "size": 3})
    assert isinstance(caught.value, TypeError)  # what a caller that never met it sees
    assert caught.value.options == ("cmap", "robust")
    assert "got 2 unusable options" in str(caught.value)  # the message users read


def test_only_the_options_check_s_own_error_counts_as_an_unusable_option():
    """A TypeError that merely says so is not one: defaults.plot options stay."""

    def draw(**opts):
        raise TypeError("got 1 unusable option:\n  'robust' is not an option of x()")

    page = _page_with(default_plot=["robust"])
    with pytest.raises(TypeError, match="unusable option"):
        P._draw(page, draw, {"robust": True})


def test_a_metrics_series_page_survives_a_suite_wide_robust():
    from tests.test_metrics_series import ComparisonSet, _fan

    class Comp(type(_fan()[0])):
        family = "series"

    comps = [Comp(c.label, **c.metrics()) for c in _fan()]
    import ocean_skill as osk

    page = _expand(
        _compare_page(draw="metrics_series"),
        defaults={"test": "stub", "plot": {"robust": True}},
    )[0]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(osk, "compare", lambda **kw: ComparisonSet(comps))
        mp.setattr(P, "_all_nan_reason", lambda c: None)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            ((suffix, fig),) = P.build(page)
    assert suffix == "" and fig is not None


def test_a_summary_page_never_sees_defaults_plot(monkeypatch):
    import ocean_skill as osk

    seen = {}
    monkeypatch.setattr(osk, "summary", lambda records, **kw: seen.update(kw) or "fig")
    (page,) = _expand(
        {"title": "s", "summary": {"kind": "both"}},
        defaults={"test": "stub", "plot": {"robust": True}},
    )
    P.build(page, pooled_records=[{"variable": "t", "bias": 1.0}])
    assert "robust" not in seen and seen["kind"] == "both"


# ======================================================================================
# pages: compare: over:
# ======================================================================================


def _over_page(**extra):
    return {
        "title": "Skill",
        "compare": {
            "reference": ["sat"],
            "variables": ["nitrate", "phosphate"],
            "over": "time",
        },
        **extra,
    }


def _suite_file(tmp_path, pages, **extra):
    return _write_suite(
        tmp_path,
        {
            "name": "skill",
            "output_dir": str(tmp_path / "out"),
            "cover": False,
            "defaults": {"test": "model"},
            "pages": pages,
            **extra,
        },
    )


def _model_index(monkeypatch, lanes):
    monkeypatch.setattr(
        "ocean_skill.extrema._native_time_index",
        lambda source: pd.DatetimeIndex(lanes["model"].time.values),
    )


def test_over_reaches_compare_and_draws_the_skill_map(tmp_path, lanes, monkeypatch):
    _model_index(monkeypatch, lanes)
    result = run_suite(_suite_file(tmp_path, [_over_page()]))

    assert result.exit_code == 0 and len(result.figures) == 1
    (page,) = result.pages
    assert page.kwargs["over"] == "time"
    assert [r["over"] for r in page.metrics_records] == ["time", "time"]
    manifest = json.loads(result.manifest.read_text())
    assert manifest["pages"][0]["kwargs"]["over"] == "time"


def test_a_skill_map_page_is_cut_into_figures_like_any_row_grid(lanes, monkeypatch):
    _model_index(monkeypatch, lanes)
    page = _over_page(plot={"max_rows": 1})
    (p,) = P.expand(_suite([page], defaults={"test": "model"}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = P.build(p)
    assert [s for s, _ in results] == ["", "_part2"]
    assert len(p.metrics_records) == 2


def test_self_check_of_a_skill_map_page_finds_identical_lanes(
    tmp_path, lanes, monkeypatch
):
    _model_index(monkeypatch, lanes)
    result = run_suite(_suite_file(tmp_path, [_over_page()]), self_check=True)
    assert result.exit_code == 0 and result.problems == []
    assert result.pages[0].title.startswith("[self-check]")
    assert all(r["rmse"] == pytest.approx(0.0) for r in result.pages[0].metrics_records)


def test_a_skill_map_page_whose_reference_is_all_nan_is_skipped_not_drawn(
    tmp_path, lanes, monkeypatch
):
    _model_index(monkeypatch, lanes)
    lanes["sat"] = _lane(1, nan=True)
    result = run_suite(_suite_file(tmp_path, [_over_page()]))
    assert result.figures == []
    (problem,) = result.problems
    assert problem.kind == "skipped" and "all-NaN" in problem.reason


# ======================================================================================
# pages: compare: selects:
# ======================================================================================


def _selects_page(select=None, **selects):
    compare = {
        "reference": ["woa"],
        "variables": ["temperature"],
        "selects": selects or {"Equator": {"transect": {"lat": 0}}},
    }
    if select is not None:
        compare["select"] = select
    return {"title": "x", "compare": compare}


def test_selects_with_a_transect_and_no_flat_select_keeps_the_run_window_on_test():
    # compare() merges the plain entry over the window pair-spec, which puts the
    # same transect on both lanes -- so the page is no longer refused for it.
    (page,) = _expand(_selects_page())
    assert page.kwargs["select"] == {
        "test": {"time": {"min": INDEX[0].isoformat(), "max": INDEX[-1].isoformat()}},
        "reference": {},
    }


def test_selects_with_a_transect_and_a_test_only_depth_is_refused_at_expand():
    page = _selects_page(select={"test": {"depth": [0, 50]}, "reference": {}})
    with pytest.raises(
        ValueError, match=r"page 'x': compare: selects: names a transect.*same depths"
    ):
        _expand(page)


def test_selects_with_a_transect_beside_a_pair_spec_select_sharing_depths_is_kept():
    select = {"test": {"depth": [0, 50]}, "reference": {"depth": [0, 50]}}
    (page,) = _expand(_selects_page(select=select))
    assert page.kwargs["select"]["reference"] == {"depth": [0, 50]}
    assert page.kwargs["selects"] == {"Equator": {"transect": {"lat": 0}}}


def test_selects_with_a_transect_and_a_flat_select_is_left_as_written():
    (page,) = _expand(_selects_page(select={"depth": [0, 50]}))
    assert page.kwargs["select"] == {"depth": [0, 50]}  # no window, still flat
    assert page.kwargs["selects"] == {"Equator": {"transect": {"lat": 0}}}


def test_a_one_sided_transect_entry_over_a_flat_select_is_refused_at_expand():
    # the flat page select used to skip the check, leaving it to compare() mid-run
    page = _selects_page(
        select={"depth": [0, 50]},
        Equator={"test": {"transect": {"lat": 0}}, "reference": {}},
    )
    with pytest.raises(ValueError, match=r"selects: names a transect.*one side only"):
        _expand(page)


def test_an_empty_selects_is_refused_at_expand():
    page = _selects_page()
    page["compare"]["selects"] = {}
    with pytest.raises(ValueError, match=r"selects: is empty"):
        _expand(page)


def test_selects_without_a_transect_still_get_the_run_window():
    (page,) = _expand(
        _selects_page(
            Shelf={"lat": {"min": 0, "max": 5}}, Gyre={"lat": {"min": 10, "max": 20}}
        )
    )
    assert page.kwargs["select"] == {
        "test": {"time": {"min": INDEX[0].isoformat(), "max": INDEX[-1].isoformat()}},
        "reference": {},
    }


def test_selects_must_be_a_mapping():
    page = _selects_page()
    page["compare"]["selects"] = ["Equator"]
    with pytest.raises(ValueError, match="selects: must be a mapping"):
        _expand(page)


def test_a_select_entry_with_its_own_time_decides_the_cache_flag():
    closed = _selects_page(A={"lat": 0, "time": "2010-01-12"})
    (page,) = _expand(closed)
    assert page.cache is True  # the window is pinned, the entry's instant is closed

    open_ended = _selects_page(A={"lat": 0, "time": "2010-03-09"})  # the last step
    (page,) = _expand(open_ended)
    assert page.cache is False

    mixed = _selects_page(A={"lat": 0}, B={"lat": 5, "time": "2010-03-09"})
    (page,) = _expand(mixed)
    assert page.cache is False


def test_a_flat_select_and_its_entries_times_are_checked_together():
    page = _selects_page(
        select={"time": "2010-01-12"}, A={"transect": {"lat": 0}, "time": "2010-03-09"}
    )
    (expanded,) = _expand(page)
    assert expanded.cache is False
    ok = _selects_page(
        select={"time": "2010-01-12"}, A={"transect": {"lat": 0}, "time": "2010-01-19"}
    )
    (expanded,) = _expand(ok)
    assert expanded.cache is True


def test_self_check_collapses_each_select_entry_to_its_test_side():
    page = _selects_page(
        select={"depth": 0},
        A={"test": {"lat": 0}, "reference": {"lat": 1}},
        B={"lat": 5},
    )
    (expanded,) = _expand(page)
    checked = P.self_check_page(expanded)
    assert checked.kwargs["selects"] == {"A": {"lat": 0}, "B": {"lat": 5}}
    assert expanded.kwargs["selects"]["A"] == {
        "test": {"lat": 0},
        "reference": {"lat": 1},
    }  # the page itself is untouched


# An entry may carry its own aggregate: {select: ..., aggregate: ...}. The checks above
# look inside it.


def test_a_transect_inside_an_entry_with_its_own_aggregate_is_still_checked():
    page = _selects_page(
        select={"test": {"depth": [0, 50]}, "reference": {}},
        Equator={"select": {"transect": {"lat": 0}}, "aggregate": {"time": "mean"}},
    )
    with pytest.raises(ValueError, match="names a transect"):
        _expand(page)


def test_an_entry_with_its_own_aggregate_has_its_time_checked():
    page = _selects_page(
        A={"select": {"lat": 0, "time": "2010-03-09"}, "aggregate": {"time": "mean"}}
    )
    (expanded,) = _expand(page)
    assert expanded.cache is False
    page = _selects_page(
        A={"select": {"lat": 0, "time": "2010-01-12"}, "aggregate": {"time": "mean"}}
    )
    (expanded,) = _expand(page)
    assert expanded.cache is True


def test_self_check_collapses_an_entrys_own_select_and_aggregate():
    page = _selects_page(
        select={"depth": 0},
        A={
            "select": {"test": {"lat": 0}, "reference": {"lat": 1}},
            "aggregate": {"test": {"time": "mean"}, "reference": {}},
        },
        B={"select": {"lat": 5}, "aggregate": [{"time": "mean"}]},
    )
    (expanded,) = _expand(page)
    checked = P.self_check_page(expanded)
    assert checked.kwargs["selects"] == {
        "A": {"select": {"lat": 0}, "aggregate": {"time": "mean"}},
        "B": {"select": {"lat": 5}, "aggregate": [{"time": "mean"}]},
    }


# ======================================================================================
# for_each items that are mappings
# ======================================================================================


def test_a_for_each_mapping_item_fills_dotted_placeholders():
    groups = [
        {"name": "physics", "variables": ["temperature", "salinity"]},
        {"name": "nutrients", "variables": ["nitrate"]},
    ]
    pages = _expand(
        {
            "title": "{group.name} maps",
            "for_each": {"group": "{groups}"},
            "compare": {"reference": ["woa"], "variables": "{group.variables}"},
        },
        defaults={"test": "stub", "groups": groups},
    )
    assert [p.title for p in pages] == ["physics maps", "nutrients maps"]
    assert [p.kwargs["variables"] for p in pages] == [
        ["temperature", "salinity"],
        ["nitrate"],
    ]


def test_a_for_each_mapping_key_named_like_a_dict_method_reads_the_key():
    groups = [
        {"name": "a", "values": "salt", "items": ["x"], "get": "g", "keys": "k"},
    ]
    pages = _expand(
        {
            "title": "{group.values} {group.get} {group.keys}",
            "for_each": {"group": "{groups}"},
            "compare": {"reference": ["woa"], "variables": "{group.items}"},
        },
        defaults={"test": "stub", "groups": groups},
    )
    assert [p.title for p in pages] == ["salt g k"]
    assert pages[0].kwargs["variables"] == ["x"]


def test_a_whole_for_each_mapping_value_is_a_plain_dict():
    import json

    groups = [{"name": "a", "items": ["x"], "values": 1}]
    pages = _expand(
        {
            "title": "t",
            "for_each": {"group": "{groups}"},
            "compare": {"reference": ["woa"], "variables": ["v"], "extra": "{group}"},
        },
        defaults={"test": "stub", "groups": groups},
    )
    extra = pages[0].kwargs["extra"]
    assert type(extra) is dict
    assert extra == groups[0]
    assert json.dumps(extra)
