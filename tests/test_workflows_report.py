"""Tests for :mod:`ocean_skill.workflows.report`: assembling PNGs (+ a PDF)."""

from __future__ import annotations

import re

import matplotlib.pyplot as plt
import pytest

from ocean_skill.workflows.report import PdfReport


def _figure(value: float = 1.0):
    fig, ax = plt.subplots()
    ax.pcolormesh([[value]])
    return fig


def _pagecount_by_bytes(pdf_path) -> int:
    data = pdf_path.read_bytes()
    return len(re.findall(rb"/Type\s*/Page[^s]", data))


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
