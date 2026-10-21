"""Tests for :mod:`ocean_skill.suites.report`: assembling PNGs (+ a PDF)."""

from __future__ import annotations

import re
import shutil
import stat
import subprocess
import warnings
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ocean_skill.suites import report as _report
from ocean_skill.suites.report import (
    PAGE_H,
    PAGE_W,
    PNG_DPI,
    PdfReport,
    compress_pdf_images,
)


def _figure(value: float = 1.0, figsize=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.pcolormesh([[value]])
    return fig


def _small(value: float = 1.0):
    """A figure about 2.2in tall on a page: four of them (and no more) share one."""
    return _figure(value, figsize=(6.0, 2.0))


def _big(value: float = 1.0):
    """A figure about 6.2in tall on a page: two of them never fit together."""
    return _figure(value, figsize=(6.0, 6.0))


def _full_bleed_figure(figsize):
    """A figure whose tight bbox is (almost) exactly ``figsize`` -- no default margins."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.pcolormesh([[1.0]])
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


#: Twelve lines of notes: a divider header so tall that a 9.8in figure would have to
#: shrink below :data:`~ocean_skill.suites.report.MIN_FIT_SCALE` to fit under it.
_LONG_NOTES = "\n".join(["A line of notes."] * 12)


def _pagecount_by_bytes(pdf_path) -> int:
    data = pdf_path.read_bytes()
    return len(re.findall(rb"/Type\s*/Page[^s]", data))


def _mediaboxes(pdf_path) -> list[tuple[float, float, float, float]]:
    data = pdf_path.read_bytes()
    boxes = []
    for match in re.findall(rb"/MediaBox\s*\[([^\]]+)\]", data):
        x0, y0, x1, y1 = (float(v) for v in match.split())
        boxes.append((x0, y0, x1, y1))
    return boxes


def test_pages_all_land_in_the_pdf(tmp_path):
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        report.emit(_big(), "01_a")
        report.emit(_big(2.0), "02_b")

    pdf_path = tmp_path / "report.pdf"
    assert pdf_path.exists()
    assert report.get_pagecount() == 2
    assert _pagecount_by_bytes(pdf_path) == 2


def test_png_filenames_are_sequential_and_slugged(tmp_path):
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        report.emit(_figure(), "nutrients_vs_woa23")
        report.emit(_figure(), "glodap")

    names = sorted(p.name for p in (tmp_path / "figures").iterdir())
    assert names == [
        "01_nutrients_vs_woa23.png",
        "02_glodap.png",
    ]


def test_pdf_none_writes_pngs_only(tmp_path):
    with PdfReport(None, tmp_path / "figures") as report:
        report.emit(_figure(), "a")

    assert not (tmp_path / "report.pdf").exists()
    assert report.get_pagecount() is None
    assert len(list((tmp_path / "figures").iterdir())) == 1


def test_two_reports_never_collide(tmp_path):
    for i in range(2):
        d = tmp_path / f"run{i}"
        with PdfReport(d / "report.pdf", d / "figures") as report:
            report.emit(_figure(), "a")
    assert (tmp_path / "run0" / "report.pdf").exists()
    assert (tmp_path / "run1" / "report.pdf").exists()


def test_a_figure_missing_no_figure_is_written_when_build_raises(tmp_path):
    """A page's own build failure never corrupts the PDF that is already open."""
    with pytest.raises(RuntimeError):
        with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
            report.emit(_figure(), "a")
            raise RuntimeError("boom")
    # the page written before the raise is still on disk and in the PDF
    assert (tmp_path / "figures" / "01_a.png").exists()
    assert (tmp_path / "report.pdf").exists()


# -- letter-page pagination -------------------------------------------------------------

_PAGE_PT = (0.0, 0.0, PAGE_W * 72, PAGE_H * 72)


def test_every_pdf_page_is_a_fixed_letter_page_regardless_of_figure_shape(tmp_path):
    """Wide, tall, square, and default-shaped figures all land on the same letter page."""
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        report.emit(_figure(), "default")
        report.emit(_figure(figsize=(8.5, 2.7)), "wide")  # a map row / series shape
        report.emit(_figure(figsize=(4.0, 10.0)), "tall")  # a profile shape
        report.emit(_figure(figsize=(5.0, 5.0)), "square")  # a Taylor/target shape

    boxes = _mediaboxes(tmp_path / "report.pdf")
    assert len(boxes) == 3  # default + wide share one page; tall and square do not
    for box in boxes:
        assert box == pytest.approx(_PAGE_PT, abs=0.5)


def test_an_oversize_figure_grows_the_page_and_warns_instead_of_clipping(tmp_path):
    """Too tall even once scaled to the page width (12x20in -> 8x13in)."""
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        report.emit(_small(), "before")
        with pytest.warns(UserWarning, match="does not fit"):
            report.emit(_full_bleed_figure((12.0, 20.0)), "oversize")
        report.emit(_small(), "after")  # room on the first page, but not beside it

    boxes = _mediaboxes(tmp_path / "report.pdf")
    oversize_box = boxes[1]
    width_pt = oversize_box[2] - oversize_box[0]
    height_pt = oversize_box[3] - oversize_box[1]
    assert width_pt >= 12.0 * 72 - 1
    assert height_pt >= PAGE_H * 72 - 1
    # the letter-sized pages surrounding it are unaffected
    assert boxes[0] == pytest.approx(_PAGE_PT, abs=0.5)
    assert boxes[2] == pytest.approx(_PAGE_PT, abs=0.5)


def test_pngs_stay_tight_crops_not_full_letter_pages(tmp_path):
    """PNGs are for embedding elsewhere, so they keep the figure's own aspect ratio."""
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        report.emit(_figure(figsize=(8.5, 2.7)), "wide")

    png_path = next((tmp_path / "figures").glob("*_wide.png"))
    image = mpimg.imread(png_path)
    height_px, width_px, *_ = image.shape
    # a tight crop of an 8.5x2.7in figure is nowhere near a full 8.5x11in page at
    # the same dpi -- the point is that the PNG's own aspect ratio is preserved
    assert height_px < width_px
    assert height_px < PAGE_H * PNG_DPI * 0.6


# -- section dividers --------------------------------------------------------------


def _log_pdf_writes(report):
    """Record every ``PdfPages.savefig`` call as ``"figure"`` or ``"divider:<title>"``.

    A page with a divider (alone, or the header of figures) is the only one this module
    draws with a title on the page figure itself (the figures here are bare
    ``pcolormesh`` axes, and a caption is not a title), so the first of the figure's
    ``texts`` tells the two apart. ``log.texts`` keeps every page's whole text, for the
    captions.
    """
    log = _PageLog()
    real = report._pdf.savefig

    def savefig(fig, *args, **kwargs):
        texts = [t.get_text() for t in fig.texts]
        log.texts.append(texts)
        log.append(f"divider:{texts[0]}" if texts else "figure")
        return real(fig, *args, **kwargs)

    report._pdf.savefig = savefig
    return log


class _PageLog(list):
    """The page kinds :func:`_log_pdf_writes` saw, plus each page's texts."""

    def __init__(self):
        super().__init__()
        self.texts: list[list[str]] = []


def test_a_section_is_buffered_then_written_just_before_the_next_figure(tmp_path):
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        log = _log_pdf_writes(report)
        report.section("s1", "Part one", "Some notes.")
        assert log == []  # queued, not written
        assert report.get_pagecount() == 0
        report.emit(_small(), "a")
        report.emit(_small(2.0), "b")

    # one page: the divider's header with both figures under it
    assert log == ["divider:Part one"]
    assert report.written_sections == {"s1"}


def test_a_newer_section_replaces_one_still_pending(tmp_path):
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        log = _log_pdf_writes(report)
        report.section("old", "Empty part", "Every page of this one skipped.")
        report.section("new", "Real part", "")
        report.emit(_small(), "a")

    assert log == ["divider:Real part"]
    assert report.written_sections == {"new"}  # the older one was never written


def test_a_section_still_pending_at_exit_is_dropped(tmp_path):
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        report.emit(_small(), "a")
        report.section("trailing", "Nothing follows", "")

    assert report.written_sections == set()
    assert report.get_pagecount() == 1
    assert _pagecount_by_bytes(tmp_path / "report.pdf") == 1


def test_a_section_starts_a_new_page_even_when_the_last_one_has_room(tmp_path):
    """A compare page can emit several figures; the divider heads just the first."""
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        log = _log_pdf_writes(report)
        report.section(1, "Part", "")
        report.emit(_small(), "a")
        report.emit(_small(), "a_taylor")
        report.section(2, "Next part", "")
        report.emit(_small(), "b")

    assert log == ["divider:Part", "divider:Next part"]
    assert report.written_sections == {1, 2}


def test_a_tall_figure_is_shrunk_under_its_divider_down_to_the_minimum_scale(
    tmp_path,
):
    """Section text is never alone on a page while its figure fits at >= 0.6 scale."""
    assert _report.MIN_FIT_SCALE == 0.6
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        log = _log_pdf_writes(report)
        report.section(1, "Part one", _LONG_NOTES)
        report.emit(_full_bleed_figure((5.0, 9.8)), "tall")

    # one page: the divider is the figure's header, not a page of its own
    assert log == ["divider:Part one"]
    assert report.written_sections == {1}
    assert _pagecount_by_bytes(tmp_path / "report.pdf") == 1


def test_a_figure_too_tall_to_fit_under_its_divider_leaves_it_a_page_of_its_own(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(_report, "MIN_FIT_SCALE", 0.95)
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        log = _log_pdf_writes(report)
        report.section(1, "Part one", _LONG_NOTES)
        report.emit(_full_bleed_figure((5.0, 9.8)), "tall")

    assert log == ["divider:Part one", "figure"]
    assert report.written_sections == {1}
    assert _pagecount_by_bytes(tmp_path / "report.pdf") == 2


def test_a_text_table_is_never_shrunk_under_its_divider(tmp_path):
    """A table that fits only shrunk goes on the page after its divider's, full size."""
    from ocean_skill.plot.typography import TABLE_FIGURE

    fig = _full_bleed_figure((5.0, 9.8))
    fig.set_label(TABLE_FIGURE)
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        log = _log_pdf_writes(report)
        report.section(1, "Part one", _LONG_NOTES)
        report.emit(fig, "table")

    assert log == ["divider:Part one", "figure"]
    assert _pagecount_by_bytes(tmp_path / "report.pdf") == 2


def test_a_section_is_pdf_only_and_never_shifts_the_png_numbering(tmp_path):
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        report.section(1, "Part one", "Notes")
        report.emit(_figure(), "a")
        report.section(2, "Part two", "More notes")
        report.emit(_figure(), "b")

    assert sorted(p.name for p in (tmp_path / "figures").iterdir()) == [
        "01_a.png",
        "02_b.png",
    ]
    assert [p.name for p in report.png_paths] == ["01_a.png", "02_b.png"]


def test_get_pagecount_counts_the_pages_actually_written(tmp_path):
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        report.emit(_big(), "a")
        assert report.get_pagecount() == 0  # the open page is not written yet
        report.section(1, "Part", "")
        assert report.get_pagecount() == 1  # a new section closes it; still queued
        report.emit(_big(), "b")
        report.section(2, "Trailing", "")
        assert report.get_pagecount() == 2  # b closed by the section; divider queued

    assert report.get_pagecount() == 2
    assert _pagecount_by_bytes(tmp_path / "report.pdf") == 2


def test_a_divider_with_its_figure_is_an_exact_letter_page(tmp_path):
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        report.section(1, "Part one", "A paragraph of notes. " * 20)
        report.emit(_figure(figsize=(8.5, 2.7)), "wide")

    boxes = _mediaboxes(tmp_path / "report.pdf")
    assert len(boxes) == 1  # the figure fits under the divider's header
    assert boxes[0] == pytest.approx(_PAGE_PT, abs=0.5)


def test_a_divider_alone_is_an_exact_letter_page(tmp_path, monkeypatch):
    monkeypatch.setattr(_report, "MIN_FIT_SCALE", 0.95)  # force the divider alone
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        report.section(1, "Part one", _LONG_NOTES)
        report.emit(_full_bleed_figure((5.0, 9.8)), "tall")

    boxes = _mediaboxes(tmp_path / "report.pdf")
    assert len(boxes) == 2
    for box in boxes:  # the divider (no tight-bbox placement) and the figure alike
        assert box == pytest.approx(_PAGE_PT, abs=0.5)


def test_no_pdf_is_written_when_nothing_drew_even_if_sections_exist(tmp_path):
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        report.section(1, "Part one", "Notes")
        report.section(2, "Part two", "")

    assert not (tmp_path / "report.pdf").exists()
    assert report.written_sections == set()
    assert list((tmp_path / "figures").iterdir()) == []


def test_section_is_a_no_op_without_a_pdf(tmp_path):
    with PdfReport(None, tmp_path / "figures") as report:
        report.section(1, "Part one", "Notes")
        report.emit(_figure(), "a")

    assert report.get_pagecount() is None
    assert report.written_sections == set()
    assert [p.name for p in report.png_paths] == ["01_a.png"]
    assert not (tmp_path / "report.pdf").exists()


LONG_NOTES = "para one\n\npara two " + "word " * 60


@pytest.mark.parametrize("text", ["", "one line", LONG_NOTES])
def test_a_divider_draws_for_any_notes_shape(tmp_path, text):
    """Empty, short, and long multi-paragraph notes all write a page."""
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        report.section(1, "A fairly long section title that has to wrap " * 2, text)
        report.emit(_figure(), "a")

    assert report.written_sections == {1}
    assert _pagecount_by_bytes(tmp_path / "report.pdf") == 1


# -- compress_pdf_images -----------------------------------------------------------


def _map_like_figure(seed: int = 0):
    """Two blended, smooth ``pcolormesh`` fields: a rasterized RGB image to shrink.

    Two layers with different colormaps and ``alpha`` give far more than 256 distinct
    colours, so matplotlib writes a true RGB image (a single colormapped field comes out
    as a palette image, which Ghostscript leaves alone). Smooth with a little noise,
    like a model field: pure noise is the one thing JPEG does *worse* than Flate on.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:240, 0:360]
    a = np.sin(x / (25 + seed)) * np.cos(y / 18) + 0.05 * rng.standard_normal(x.shape)
    b = np.cos(x / 40) + np.sin(y / (30 + seed)) + 0.05 * rng.standard_normal(x.shape)
    ax.pcolormesh(a, cmap="viridis")
    ax.pcolormesh(b, cmap="magma", alpha=0.5)
    return fig


def _gs_pagecount(path) -> int:
    """Page count via Ghostscript itself (its output may hide ``/Type /Page``)."""
    out = subprocess.run(
        [
            shutil.which("gs"),
            "-q",
            "-dNODISPLAY",
            "-dNOSAFER",
            "-dBATCH",
            "-c",
            f"({path}) (r) file runpdfbegin pdfpagecount = quit",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip())


def _leftovers(directory) -> list[str]:
    """Everything in ``directory`` other than the report itself (temp files)."""
    return sorted(p.name for p in directory.iterdir() if p.name != "report.pdf")


@pytest.mark.skipif(shutil.which("gs") is None, reason="Ghostscript (gs) not installed")
def test_compress_pdf_images_shrinks_a_rasterized_report_and_keeps_its_pages(
    tmp_path, monkeypatch
):
    # unfilled pages: their mesh images are the file's bulk, which is what this is about
    monkeypatch.setattr(_report, "MAX_FILL_SCALE", 1.0)
    pdf = tmp_path / "report.pdf"
    with PdfReport(pdf, tmp_path / "figures") as report:
        report.emit(_map_like_figure(0), "a")
        report.section(1, "Part two", "Notes.")
        report.emit(_map_like_figure(1), "b")
    pages_before = _gs_pagecount(pdf)
    assert pages_before == 2  # a, then the divider with b under it
    size_before = pdf.stat().st_size

    sizes = compress_pdf_images(pdf)

    assert sizes is not None
    before, after = sizes
    assert before == size_before
    assert after == pdf.stat().st_size
    assert after < before * 0.7  # the Flate-coded RGB rasters are most of the file
    assert b"/DCTDecode" in pdf.read_bytes()
    assert _gs_pagecount(pdf) == pages_before
    assert _leftovers(tmp_path) == ["figures"]  # no temp file left behind


def test_compress_pdf_images_without_gs_warns_and_leaves_the_file_alone(
    tmp_path, monkeypatch
):
    pdf = tmp_path / "report.pdf"
    with PdfReport(pdf, tmp_path / "figures") as report:
        report.emit(_figure(), "a")
    original = pdf.read_bytes()

    monkeypatch.setattr(_report.shutil, "which", lambda name: None)
    with pytest.warns(UserWarning, match="Ghostscript"):
        assert compress_pdf_images(pdf) is None

    assert pdf.read_bytes() == original
    assert _leftovers(tmp_path) == ["figures"]


def _stub_gs(monkeypatch, run):
    """Make ``compress_pdf_images`` find a ``gs`` and hand its command to ``run``."""
    monkeypatch.setattr(_report.shutil, "which", lambda name: "/fake/gs")
    monkeypatch.setattr(_report.subprocess, "run", run)


def _output_path(cmd) -> str:
    return next(a for a in cmd if a.startswith("-sOutputFile=")).split("=", 1)[1]


def test_compress_pdf_images_nonzero_exit_warns_and_leaves_the_file_alone(
    tmp_path, monkeypatch
):
    pdf = tmp_path / "report.pdf"
    with PdfReport(pdf, tmp_path / "figures") as report:
        report.emit(_figure(), "a")
    original = pdf.read_bytes()

    def failing(cmd, **kwargs):
        # a partial output, as a crashed gs might leave
        Path(_output_path(cmd)).write_bytes(b"%PDF-1.5 truncated")
        return subprocess.CompletedProcess(cmd, 1, "", "Error: /undefined in foo")

    _stub_gs(monkeypatch, failing)
    with pytest.warns(UserWarning, match="exited with status 1.*undefined in foo"):
        assert compress_pdf_images(pdf) is None

    assert pdf.read_bytes() == original
    assert _leftovers(tmp_path) == ["figures"]  # the partial temp file is cleaned up


def test_compress_pdf_images_timeout_warns_and_leaves_the_file_alone(
    tmp_path, monkeypatch
):
    pdf = tmp_path / "report.pdf"
    with PdfReport(pdf, tmp_path / "figures") as report:
        report.emit(_figure(), "a")
    original = pdf.read_bytes()

    def hanging(cmd, **kwargs):
        assert kwargs["timeout"] == _report.GS_TIMEOUT_S == 600
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    _stub_gs(monkeypatch, hanging)
    with pytest.warns(UserWarning, match="did not finish within 600s"):
        assert compress_pdf_images(pdf) is None

    assert pdf.read_bytes() == original
    assert _leftovers(tmp_path) == ["figures"]


def test_compress_pdf_images_keeps_the_original_when_jpeg_is_not_smaller(
    tmp_path, monkeypatch
):
    pdf = tmp_path / "report.pdf"
    with PdfReport(pdf, tmp_path / "figures") as report:
        report.emit(_figure(), "a")
    original = pdf.read_bytes()

    def bigger(cmd, **kwargs):
        Path(_output_path(cmd)).write_bytes(b"x" * (len(original) + 100))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    _stub_gs(monkeypatch, bigger)
    with pytest.warns(UserWarning, match="did not make report.pdf smaller"):
        assert compress_pdf_images(pdf) is None

    assert pdf.read_bytes() == original
    assert _leftovers(tmp_path) == ["figures"]


def test_compress_pdf_images_replaces_atomically_and_asks_gs_for_lossless_masks(
    tmp_path, monkeypatch
):
    """A successful pass swaps the new file in, with the settings we rely on."""
    pdf = tmp_path / "report.pdf"
    with PdfReport(pdf, tmp_path / "figures") as report:
        report.emit(_figure(), "a")
    before = pdf.stat().st_size
    seen: dict = {}

    def smaller(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["input"] = cmd[cmd.index("-f") + 1]
        # the output sits beside the report (same filesystem, so os.replace is atomic)
        # and is not the report itself
        seen["out"] = _output_path(cmd)
        Path(seen["out"]).write_bytes(b"%PDF-1.5 tiny")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    _stub_gs(monkeypatch, smaller)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert compress_pdf_images(pdf) == (before, len(b"%PDF-1.5 tiny"))

    assert pdf.read_bytes() == b"%PDF-1.5 tiny"
    cmd = seen["cmd"]
    assert cmd[0] == "/fake/gs"
    assert seen["input"] == str(pdf.resolve())
    assert seen["out"] != str(pdf)
    assert Path(seen["out"]).parent == tmp_path
    for flag in (
        "-dSAFER",
        "-sDEVICE=pdfwrite",
        "-dColorImageFilter=/DCTEncode",
        "-dDownsampleColorImages=false",
        "-dGrayImageFilter=/FlateEncode",  # the soft masks stay lossless
        "-dDownsampleGrayImages=false",
    ):
        assert flag in cmd
    # JPEG quality has to come in through distiller params (-dJPEGQ is ignored)
    params = cmd[cmd.index("-c") + 1]
    assert f"/QFactor {_report.JPEG_QFACTOR}" in params
    assert "/HSamples [1 1 1 1]" in params and "/VSamples [1 1 1 1]" in params
    assert not any(a.startswith("-dJPEGQ") for a in cmd)
    assert _leftovers(tmp_path) == ["figures"]


@pytest.mark.parametrize("mode", [0o640, 0o644])
def test_compress_pdf_images_keeps_the_reports_permission_bits(
    tmp_path, monkeypatch, mode
):
    """The temp file is born 0600; a group-readable report must stay group-readable."""
    pdf = tmp_path / "report.pdf"
    with PdfReport(pdf, tmp_path / "figures") as report:
        report.emit(_figure(), "a")
    pdf.chmod(mode)

    def smaller(cmd, **kwargs):
        Path(_output_path(cmd)).write_bytes(b"%PDF-1.5 tiny")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    _stub_gs(monkeypatch, smaller)
    assert compress_pdf_images(pdf) is not None

    assert pdf.read_bytes() == b"%PDF-1.5 tiny"
    assert stat.S_IMODE(pdf.stat().st_mode) == mode


@pytest.mark.skipif(shutil.which("gs") is None, reason="Ghostscript (gs) not installed")
def test_compress_pdf_images_keeps_the_reports_permission_bits_with_real_gs(tmp_path):
    pdf = tmp_path / "report.pdf"
    with PdfReport(pdf, tmp_path / "figures") as report:
        report.emit(_map_like_figure(0), "a")
    pdf.chmod(0o640)

    assert compress_pdf_images(pdf) is not None

    assert stat.S_IMODE(pdf.stat().st_mode) == 0o640


def test_rasterize_skips_contour_sets_without_warning():
    # matplotlib cannot rasterize a ContourSet and warns when asked to; an XY/TS page's
    # density contours (and any section's contour lines) must not trigger that.
    fig, ax = plt.subplots()
    ax.scatter([0, 1], [0, 1])
    grid = np.arange(16.0).reshape(4, 4)
    contours = ax.contour(grid)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _report._rasterize(fig)
    assert ax.collections[0].get_rasterized()
    assert not contours.get_rasterized()
    plt.close(fig)


# -- __exit__ and the original exception ---------------------------------------------


def _fail_the_last_write(monkeypatch):
    def boom(self, *a, **k):
        raise OSError("disk full")

    # a lone figure that has room is enlarged and written packed, else alone (vector)
    monkeypatch.setattr(PdfReport, "_write_alone", boom)
    monkeypatch.setattr(PdfReport, "_write_packed", boom)


def _write_report_raising(tmp_path, error):
    """Run a report block that emits a figure and then raises ``error``."""
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        report.emit(_small(), "a")
        raise error


def test_a_flush_failure_does_not_replace_the_exception_already_in_flight(
    tmp_path, monkeypatch
):
    _fail_the_last_write(monkeypatch)
    with (
        pytest.warns(UserWarning, match=r"last page .*disk full.*while KeyError"),
        pytest.raises(KeyError, match="the real problem"),
    ):
        _write_report_raising(tmp_path, KeyError("the real problem"))


def test_the_pdf_is_still_closed_when_the_last_page_could_not_be_written(
    tmp_path, monkeypatch
):
    from matplotlib.backends.backend_pdf import PdfPages

    _fail_the_last_write(monkeypatch)
    closed = []
    real_exit = PdfPages.__exit__
    monkeypatch.setattr(
        PdfPages,
        "__exit__",
        lambda self, *a: (closed.append(a[0]), real_exit(self, *a))[1],
    )
    with pytest.warns(UserWarning, match="last page"), pytest.raises(KeyError):
        _write_report_raising(tmp_path, KeyError("x"))
    assert closed == [KeyError]  # closed with the original error, not the flush's


def test_a_flush_failure_with_nothing_in_flight_is_raised(tmp_path, monkeypatch):
    _fail_the_last_write(monkeypatch)
    with (
        pytest.raises(OSError, match="disk full"),
        PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report,
    ):
        report.emit(_small(), "a")


# -- one divider layout --------------------------------------------------------------


def _rules(fig):
    from matplotlib.lines import Line2D

    return [a for a in fig.artists if isinstance(a, Line2D)]


def test_a_stand_alone_divider_and_a_packed_pages_header_are_one_layout(monkeypatch):
    """Both go through ``_draw_header``: same title, rule and notes, same geometry."""
    title, notes = "Part one", "Some notes about the part."

    alone = _report._section_figure(title, notes)
    assert alone.get_size_inches() == pytest.approx((_report.PAGE_W, _report.PAGE_H))

    heading = plt.figure(figsize=(_report.PAGE_W, _report.PAGE_H))
    top = _report.PAGE_H - _report._top_margin()
    _report._draw_header(heading, _report.PAGE_W / 2, top, title, notes)

    def geometry(fig):
        return (
            [(t.get_text(), t.get_position(), t.get_fontsize()) for t in fig.texts],
            [ln.get_xydata().tolist() for ln in _rules(fig)],
        )

    assert len(_rules(alone)) == 1
    assert geometry(alone) == geometry(heading)
    plt.close(heading)


def test_a_stand_alone_divider_hangs_from_the_top_with_a_rule_and_notes():
    fig = _report._section_figure("Part one", "Some notes.")
    title, notes = fig.texts
    (rule,) = _rules(fig)
    assert title.get_text() == "Part one" and title.get_ha() == "center"
    # hung from the top of the page, never centred in a blank one
    assert title.get_position()[1] > 0.9
    # title, then its rule, then the notes below it
    assert title.get_position()[1] > rule.get_ydata()[0] > notes.get_position()[1]


def test_a_title_only_divider_has_a_title_and_a_rule_and_no_notes():
    fig = _report._section_figure("Part one", "")
    assert [t.get_text() for t in fig.texts] == ["Part one"] and len(_rules(fig)) == 1
