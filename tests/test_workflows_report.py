"""Tests for :mod:`ocean_skill.workflows.report`: assembling PNGs (+ a PDF)."""

from __future__ import annotations

import re

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pytest

from ocean_skill.workflows.report import PAGE_H, PAGE_W, PNG_DPI, PdfReport


def _figure(value: float = 1.0, figsize=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.pcolormesh([[value]])
    return fig


def _full_bleed_figure(figsize):
    """A figure whose tight bbox is (almost) exactly ``figsize`` -- no default margins."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.pcolormesh([[1.0]])
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


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


def test_title_pages_and_log_page_all_land_in_the_pdf(tmp_path):
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        report.title_page("title text")
        report.emit(_figure(), "01_a")
        report.emit(_figure(2.0), "02_b")
        report.log_page("log text")

    pdf_path = tmp_path / "report.pdf"
    assert pdf_path.exists()
    assert report.get_pagecount() == 4
    assert _pagecount_by_bytes(pdf_path) == 4


def test_pngs_are_named_in_order():
    pass  # covered by the numbering assertion below


def test_png_filenames_are_sequential_and_slugged(tmp_path):
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        report.title_page("t")
        report.emit(_figure(), "nutrients_vs_woa23")
        report.emit(_figure(), "glodap")
        report.log_page("l")

    names = sorted(p.name for p in (tmp_path / "figures").iterdir())
    assert names == [
        "01_title.png",
        "02_nutrients_vs_woa23.png",
        "03_glodap.png",
        "04_log.png",
    ]


def test_pdf_none_writes_pngs_only(tmp_path):
    with PdfReport(None, tmp_path / "figures") as report:
        report.title_page("t")
        report.emit(_figure(), "a")
        report.log_page("l")

    assert not (tmp_path / "report.pdf").exists()
    assert report.get_pagecount() is None
    assert len(list((tmp_path / "figures").iterdir())) == 3


def test_two_reports_never_collide(tmp_path):
    for i in range(2):
        d = tmp_path / f"run{i}"
        with PdfReport(d / "report.pdf", d / "figures") as report:
            report.title_page("t")
            report.emit(_figure(), "a")
            report.log_page("l")
    assert (tmp_path / "run0" / "report.pdf").exists()
    assert (tmp_path / "run1" / "report.pdf").exists()


def test_a_figure_missing_no_figure_is_written_when_build_raises(tmp_path):
    """A page's own build failure never corrupts the PDF that is already open."""
    with pytest.raises(RuntimeError):
        with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
            report.title_page("t")
            raise RuntimeError("boom")
    # the title page written before the raise is still on disk and in the PDF
    assert (tmp_path / "figures" / "01_title.png").exists()
    assert (tmp_path / "report.pdf").exists()


# -- letter-page pagination -------------------------------------------------------------

_PAGE_PT = (0.0, 0.0, PAGE_W * 72, PAGE_H * 72)


def test_every_pdf_page_is_a_fixed_letter_page_regardless_of_figure_shape(tmp_path):
    """Wide, tall, square, and text pages all land on the same 8.5x11in page."""
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        report.title_page("title text")
        report.emit(_figure(figsize=(8.5, 2.7)), "wide")  # a map row / series shape
        report.emit(_figure(figsize=(4.0, 10.0)), "tall")  # a profile shape
        report.emit(_figure(figsize=(5.0, 5.0)), "square")  # a Taylor/target shape
        report.log_page("log text")

    boxes = _mediaboxes(tmp_path / "report.pdf")
    assert len(boxes) == 5
    for box in boxes:
        assert box == pytest.approx(_PAGE_PT, abs=0.5)


def test_an_oversize_figure_grows_the_page_and_warns_instead_of_clipping(tmp_path):
    with PdfReport(tmp_path / "report.pdf", tmp_path / "figures") as report:
        report.title_page("t")
        with pytest.warns(UserWarning, match="does not fit"):
            report.emit(_full_bleed_figure((12.0, 3.0)), "oversize")
        report.log_page("l")

    boxes = _mediaboxes(tmp_path / "report.pdf")
    oversize_box = boxes[1]
    width_pt = oversize_box[2] - oversize_box[0]
    height_pt = oversize_box[3] - oversize_box[1]
    assert width_pt >= 12.0 * 72 - 1
    assert height_pt >= PAGE_H * 72 - 1
    # the letter-sized title/log pages are unaffected
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
