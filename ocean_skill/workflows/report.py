"""Assemble a suite's drawn pages into a report: PNGs, a PDF, a manifest.

:class:`PdfReport` writes each page's figure to ``figures/NN_<slug>.png`` and,
unless the suite has ``pdf: false``, adds it as a page to one ``report.pdf`` --
matplotlib's own :class:`~matplotlib.backends.backend_pdf.PdfPages`, so there is
no new dependency. A title page (what was asked for, what the run covers) and a
closing log page (what actually happened) bracket the drawn pages, both plain
text so they carry no data of their own to get stale.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as _dc_field
from pathlib import Path
from typing import Any

__all__ = ["PdfReport"]


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
        fig.savefig(png, dpi=150, bbox_inches="tight")
        if self._pdf is not None:
            _rasterize(fig)
            self._pdf.savefig(fig, dpi=150, bbox_inches="tight")
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

    fig = plt.figure(figsize=(8.5, 11))
    fig.text(
        0.06, 0.94, text, va="top", ha="left", family="monospace", fontsize=9, wrap=True
    )
    from matplotlib._pylab_helpers import Gcf

    manager = next((m for m in Gcf.figs.values() if m.canvas.figure is fig), None)
    if manager is not None:
        Gcf.figs.pop(manager.num, None)
    return fig
