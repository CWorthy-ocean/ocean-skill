"""Assemble a suite's drawn pages into a report: PNGs, a PDF, a manifest.

:class:`PdfReport` writes each page's figure to ``figures/NN_<slug>.png`` and,
unless the suite has ``pdf: false``, adds it as a page to one ``report.pdf`` --
matplotlib's own :class:`~matplotlib.backends.backend_pdf.PdfPages`, so there is
no new dependency. A title page (what was asked for, what the run covers) and a
closing log page (what actually happened) bracket the drawn pages, both plain
text so they carry no data of their own to get stale.

PDF pages are all US Letter portrait (8.5x11in, :data:`~ocean_skill.plot.typography.PAGE_W`
/ :data:`~ocean_skill.plot.typography.PAGE_H`) -- a figure's tight ink is placed at the
top of the page, horizontally centred, so the report paginates and prints like a
document regardless of how wide or tall any one figure is. PNGs are saved as tight
crops around the figure instead, since they are meant for embedding elsewhere, not for
printing.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from dataclasses import field as _dc_field
from pathlib import Path
from typing import Any

from ocean_skill.plot.typography import PAGE_H, PAGE_W

__all__ = ["PdfReport"]

#: Kept between the figure's own tight ink and the top of the page -- purely cosmetic
#: (a page is never rejected for lacking it; see :func:`_page_bbox`).
PAGE_MARGIN = 0.5

#: PNGs carry the detail a suite's now-ignored ``zoom=`` used to buy (see
#: ocean_skill.workflows.pages._pin_to_page): a plain dpi bump costs nothing extra
#: since the PDF copy is rasterized/downsampled separately at its own dpi.
PNG_DPI = 200
PDF_DPI = 150


def _rasterize(fig: Any) -> None:
    """Rasterize the heavy mesh/image artists so a PDF page stays small.

    Vector pcolormesh on a fine model grid can bloat a PDF by two orders of
    magnitude; text, axes, and colorbars stay vector. Renderer-level
    ``rasterize=`` is interactive-only (holoviews), so this is done here, once,
    on the returned figure -- after the PNG is written, so the PNG itself stays
    fully vector/raster as the renderer already chose.
    """
    for ax in fig.axes:
        for artist in (*ax.collections, *ax.images):
            artist.set_rasterized(True)


def _page_bbox(fig: Any, stem: str) -> Any:
    """Return the Bbox (inches) that places ``fig`` on one US Letter portrait page.

    The figure's own tight bounding box is anchored to the top of the page, centred
    left-right, with :data:`PAGE_MARGIN` of headroom above it. A figure that already
    fits comfortably inside 8.5x11 (the common case, since figures are drawn against
    that canvas by default -- see :mod:`ocean_skill.plot.typography`) gets an exact
    letter page; one that doesn't -- an explicit ``figsize=`` wider or taller than the
    page slipped past :func:`ocean_skill.workflows.pages._pin_to_page` -- gets a page
    grown just enough to hold it, with a warning, rather than a clipped page.
    """
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())

    width = max(tight.width, PAGE_W)
    height = max(tight.height + PAGE_MARGIN, PAGE_H)
    if width > PAGE_W or height > PAGE_H:
        warnings.warn(
            f"page {stem!r}: figure ({tight.width:.2f}x{tight.height:.2f}in) does not "
            f"fit an {PAGE_W}x{PAGE_H}in page even with pinning -- growing the page "
            "instead of clipping the figure",
            stacklevel=2,
        )

    x0 = tight.x0 - (width - tight.width) / 2
    y1 = tight.y1 + PAGE_MARGIN
    return Bbox([[x0, y1 - height], [x0 + width, y1]])


@dataclass
class PdfReport:
    """Write PNGs (always) and a PDF (unless ``pdf_path`` is ``None``) as pages arrive.

    Used as a context manager: pages are added with :meth:`title_page`,
    :meth:`emit`, and :meth:`log_page`, in that order; the PDF is opened before
    the first page and closed on exit even if a page's own drawing raised, so a
    partially-run report is never left with a corrupt or half-written PDF.
    """

    pdf_path: Path | None
    figures_dir: Path
    _pdf: Any = _dc_field(default=None, repr=False)
    _index: int = 0
    png_paths: list[Path] = _dc_field(default_factory=list)

    def __post_init__(self) -> None:
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> PdfReport:
        if self.pdf_path is not None:
            from matplotlib.backends.backend_pdf import PdfPages

            self._pdf = PdfPages(self.pdf_path)
            self._pdf.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._pdf is not None:
            self._pdf.__exit__(*exc)

    def _emit(self, fig: Any, stem: str) -> Path:
        self._index += 1
        png = self.figures_dir / f"{self._index:02d}_{stem}.png"
        fig.savefig(png, dpi=PNG_DPI, bbox_inches="tight")
        if self._pdf is not None:
            page = _page_bbox(fig, stem)
            _rasterize(fig)
            self._pdf.savefig(fig, dpi=PDF_DPI, bbox_inches=page)
        fig.clear()
        self.png_paths.append(png)
        return png

    def title_page(self, text: str) -> Path:
        return self._emit(_text_figure(text), "title")

    def emit(self, fig: Any, stem: str) -> Path:
        return self._emit(fig, stem)

    def log_page(self, text: str) -> Path:
        return self._emit(_text_figure(text), "log")

    def get_pagecount(self) -> int | None:
        # PdfPages.get_pagecount() reports 0 once the file is closed -- this
        # object's own running index is the reliable count either way.
        return self._index if self._pdf is not None else None


def _text_figure(text: str) -> Any:
    """Build a plain, one-page text figure -- report structure, not a data caveat."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.text(
        0.06, 0.94, text, va="top", ha="left", family="monospace", fontsize=9, wrap=True
    )
    from matplotlib._pylab_helpers import Gcf

    manager = next((m for m in Gcf.figs.values() if m.canvas.figure is fig), None)
    if manager is not None:
        Gcf.figs.pop(manager.num, None)
    return fig
