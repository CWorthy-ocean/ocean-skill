"""Assemble a suite's drawn pages into a report: PNGs, a PDF, a manifest.

:class:`PdfReport` writes each page's figure to ``figures/NN_<slug>.png`` and,
unless the suite has ``pdf: false``, adds it as a page to one ``report.pdf`` --
matplotlib's own :class:`~matplotlib.backends.backend_pdf.PdfPages`, so there is
no new dependency.

PDF pages are all US Letter portrait (8.5x11in, :data:`~ocean_skill.plot.typography
.PAGE_W` / :data:`~ocean_skill.plot.typography.PAGE_H`) -- a figure's tight ink is
placed at the top of the page, horizontally centred, so the report paginates and prints
like a document regardless of how wide or tall any one figure is. PNGs are saved as
tight crops around the figure instead, since they are meant for embedding elsewhere,
not for printing.

:meth:`PdfReport.section` adds a text-only divider page (a title plus optional notes)
to the PDF -- and only the PDF: it never gets a PNG and never advances the ``NN_``
numbering of the figures, so a suite's ``figures/`` directory reads the same with or
without sections. It is buffered rather than written at once, so a section whose pages
all skipped leaves no orphan divider behind (see :meth:`PdfReport.section`).

:func:`compress_pdf_images` is an optional post-pass over a finished ``report.pdf``
(``pdf_images: jpeg`` in a suite) that re-encodes its rasterized map images as JPEG
with Ghostscript -- matplotlib's PDF backend only writes them losslessly.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import textwrap
import warnings
from dataclasses import dataclass
from dataclasses import field as _dc_field
from pathlib import Path
from typing import Any

from ocean_skill.plot.typography import PAGE_H, PAGE_W, STEP, reference_scale

__all__ = ["JPEG_QFACTOR", "PdfReport", "compress_pdf_images"]

#: Kept between the figure's own tight ink and the top of the page -- purely cosmetic
#: (a page is never rejected for lacking it; see :func:`_page_bbox`).
PAGE_MARGIN = 0.5

#: PNGs carry the detail a suite's now-ignored ``zoom=`` used to buy (see
#: ocean_skill.workflows.pages._pin_to_page): a plain dpi bump costs nothing extra
#: since the PDF copy is rasterized/downsampled separately at its own dpi.
PNG_DPI = 200
PDF_DPI = 150

#: A section divider's title is the suptitle role of the report's reference type scale
#: (:func:`~ocean_skill.plot.typography.reference_scale`; the largest type any figure
#: carries) raised this many steps of the modular scale
#: (:data:`~ocean_skill.plot.typography.STEP`), so it reads as a heading above the
#: figures' own text; the notes below it use the ``title`` role -- the size a figure's
#: panel titles are set in -- as body text.
SECTION_TITLE_STEPS = 4

#: Where a divider's title sits: the bottom of its text block is anchored a little above
#: the page's vertical middle (so a title that wraps onto a second line grows upward),
#: and the notes hang from just below the rule beneath it.
SECTION_TITLE_Y = 0.56
SECTION_NOTES_Y = 0.51

#: Wrap widths, in characters: the title at roughly 22pt and the notes at roughly 11pt
#: both fit comfortably inside a letter page's side margins at these counts.
SECTION_TITLE_WRAP = 34
SECTION_NOTES_WRAP = 78

#: JPEG quality for :func:`compress_pdf_images`, as a Ghostscript distiller ``QFactor``
#: (0 = best, larger = smaller and blockier). See that function for how it was chosen.
JPEG_QFACTOR = 0.25

#: Seconds :func:`compress_pdf_images` waits for Ghostscript before giving up.
GS_TIMEOUT_S = 600


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


def _wrap_lines(text: str, width: int) -> list[str]:
    """Wrap each line of ``text`` to ``width`` characters, keeping blank lines.

    Line structure is the author's: a blank line stays a paragraph break and a line
    that is already short stays as written, so a literal YAML block (``|``) lays out as
    typed. A folded one (``>-``) arrives as one long line per paragraph and so wraps
    here.
    """
    out: list[str] = []
    for line in text.splitlines():
        out.extend(textwrap.wrap(line, width=width) or [""])
    return out


def _section_figure(title: str, text: str) -> Any:
    """Draw one divider page: a large centred title, a rule, and wrapped notes below.

    A fresh full-page :class:`~matplotlib.figure.Figure` of exactly
    ``PAGE_W`` x ``PAGE_H`` -- the caller saves it *without* a tight bounding box (see
    :meth:`PdfReport.section`), so it stays exactly letter-sized like every other page.
    Built directly rather than through :mod:`matplotlib.pyplot`, so it is never
    registered with pyplot's figure manager and there is nothing to close afterwards.
    """
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    sizes = reference_scale()
    title_pt = sizes["suptitle"] * STEP**SECTION_TITLE_STEPS
    body_pt = sizes["title"]

    fig = Figure(figsize=(PAGE_W, PAGE_H))
    fig.text(
        0.5,
        SECTION_TITLE_Y,
        "\n".join(_wrap_lines(title, SECTION_TITLE_WRAP)),
        ha="center",
        va="bottom",
        fontsize=title_pt,
        fontweight="bold",
        multialignment="center",
    )
    # A short rule between title and notes: keeps a title-only divider from looking
    # like a page that failed to draw its body.
    fig.add_artist(
        Line2D(
            [0.38, 0.62],
            [SECTION_TITLE_Y - 0.012] * 2,
            color="0.55",
            linewidth=0.8,
            transform=fig.transFigure,
        )
    )
    if text.strip():
        fig.text(
            0.5,
            SECTION_NOTES_Y,
            "\n".join(_wrap_lines(text, SECTION_NOTES_WRAP)),
            ha="center",
            va="top",
            fontsize=body_pt,
            color="0.25",
            multialignment="left",
            linespacing=1.5,
        )
    return fig


def compress_pdf_images(path: str | Path) -> tuple[int, int] | None:
    """Re-encode ``path``'s rasterized map images as JPEG with Ghostscript, in place.

    About 95% of a report's bytes are the rasterized ``pcolormesh`` images that
    :func:`_rasterize` keeps the PDF small with, and matplotlib's PDF backend can only
    write those losslessly (Flate). This is the post-pass that does what it can't: it
    rewrites the finished PDF through Ghostscript's ``pdfwrite`` device with the colour
    images switched to ``DCTEncode``. Everything else is untouched in spirit -- the
    text stays vector, the page sizes stay letter, and the images' own resolution is
    kept (``-dDownsample*Images=false``) -- and the 8-bit grey images, which are the
    transparency soft masks that cut land and out-of-domain cells out of the maps, stay
    lossless so no mask edge gets JPEG ringing.

    ``-dJPEGQ`` has no effect on ``pdfwrite``; the JPEG quality goes in through
    distiller params instead (``/ColorImageDict``). Chroma subsampling is switched off
    there (``/HSamples``/``/VSamples`` all ``1``): Ghostscript's default 2x2 smears the
    sharp colour edges a map is made of -- a coastline, an eddy front -- into visibly
    mushy, desaturated patches, which is the one thing that is easy to see at print
    zoom.

    **Quality.** :data:`JPEG_QFACTOR` (``QFactor`` 0.25) was picked by rendering a real
    13-page report at 300 dpi and comparing it page by page against the same report
    re-written losslessly by Ghostscript, zoomed in on coastlines and eddy fields:
    0.15 and 0.25 are indistinguishable (99.9% of pixels within 12/255 of the lossless
    render, every page); 0.4 starts to show faint blocking in the darkest water; and
    Ghostscript's own default (2x2 chroma) was plainly worse than any of them. Measured
    on that report: 7.55 MB lossless -> 4.78 MB (-37%) at 0.25, against 5.23 MB at 0.15
    and 4.47 MB at 0.4.

    The result is written to a temporary file beside ``path`` and moved into place with
    :func:`os.replace`, so an interrupted run never leaves a half-written report. The
    temporary file is created owner-only (0600), so the original's permission bits are
    copied onto it first (:func:`shutil.copymode`): on a shared machine a report that
    group collaborators could read must not turn private by being compressed. If
    Ghostscript is not installed, exits non-zero, exceeds :data:`GS_TIMEOUT_S`, or the
    rewritten file would not actually be smaller, a warning names the reason, the
    lossless PDF is left exactly as it was, and ``None`` is returned. Otherwise returns
    ``(size_before, size_after)`` in bytes.
    """
    path = Path(path)
    gs = shutil.which("gs")
    if gs is None:
        warnings.warn(
            f"pdf_images: jpeg needs Ghostscript (gs), which is not on PATH -- "
            f"leaving {path.name} lossless",
            stacklevel=2,
        )
        return None

    before = path.stat().st_size
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".jpeg.pdf", dir=path.parent
    )
    os.close(fd)
    tmp = Path(tmp_name)
    cmd = [
        gs,
        "-dSAFER",
        "-dBATCH",
        "-dNOPAUSE",
        "-q",
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.5",
        "-dAutoFilterColorImages=false",
        "-dColorImageFilter=/DCTEncode",
        "-dDownsampleColorImages=false",
        "-dAutoFilterGrayImages=false",
        "-dGrayImageFilter=/FlateEncode",
        "-dDownsampleGrayImages=false",
        # "%" starts a page-number or device pattern in an output name: escape it
        f"-sOutputFile={os.fspath(tmp).replace('%', '%%')}",
        "-c",
        f"<< /ColorImageDict << /QFactor {JPEG_QFACTOR} /Blend 1 "
        "/HSamples [1 1 1 1] /VSamples [1 1 1 1] >> >> setdistillerparams",
        "-f",
        os.fspath(path.resolve()),
    ]
    try:
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=GS_TIMEOUT_S,
                check=False,
            )
        except subprocess.TimeoutExpired:
            warnings.warn(
                f"pdf_images: jpeg -- Ghostscript did not finish within "
                f"{GS_TIMEOUT_S}s; leaving {path.name} lossless",
                stacklevel=2,
            )
            return None
        except OSError as exc:
            warnings.warn(
                f"pdf_images: jpeg -- could not run Ghostscript ({exc}); leaving "
                f"{path.name} lossless",
                stacklevel=2,
            )
            return None
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            warnings.warn(
                f"pdf_images: jpeg -- Ghostscript exited with status "
                f"{proc.returncode}"
                + (f" ({detail[-1]})" if detail else "")
                + f"; leaving {path.name} lossless",
                stacklevel=2,
            )
            return None
        after = tmp.stat().st_size if tmp.exists() else 0
        if after <= 0 or after >= before:
            warnings.warn(
                f"pdf_images: jpeg -- the JPEG pass did not make {path.name} smaller "
                f"({before} -> {after} bytes); leaving it lossless",
                stacklevel=2,
            )
            return None
        # mkstemp made the temp file 0600: keep the report's own mode instead
        shutil.copymode(path, tmp)
        os.replace(tmp, path)
        return before, after
    finally:
        tmp.unlink(missing_ok=True)


@dataclass
class PdfReport:
    """Write PNGs (always) and a PDF (unless ``pdf_path`` is ``None``) as pages arrive.

    Used as a context manager: pages are added with :meth:`emit`; the PDF is
    opened before the first page and closed on exit even if a page's own drawing
    raised, so a partially-run report is never left with a corrupt or half-written
    PDF. :meth:`section` adds a PDF-only divider page between them.

    ``written_sections`` collects the key of every divider that actually reached the
    PDF, so a caller can tell a section it handed over from one that was written (see
    :meth:`section` for when one is dropped instead).
    """

    pdf_path: Path | None
    figures_dir: Path
    _pdf: Any = _dc_field(default=None, repr=False)
    _index: int = 0
    _pdf_pages: int = 0
    _pending: tuple[Any, str, str] | None = _dc_field(default=None, repr=False)
    png_paths: list[Path] = _dc_field(default_factory=list)
    written_sections: set[Any] = _dc_field(default_factory=set)

    def __post_init__(self) -> None:
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> PdfReport:
        if self.pdf_path is not None:
            from matplotlib.backends.backend_pdf import PdfPages

            self._pdf = PdfPages(self.pdf_path)
            self._pdf.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        # A divider still pending here had no figure after it: dropped, never written.
        self._pending = None
        if self._pdf is not None:
            self._pdf.__exit__(*exc)

    def _write_pending_section(self) -> None:
        """Write the buffered divider (if any) into the PDF, then clear it."""
        if self._pending is None or self._pdf is None:
            return
        key, title, text = self._pending
        self._pending = None
        fig = _section_figure(title, text)
        # No bbox_inches: a tight crop would shrink the page to its ink. The figure is
        # already exactly PAGE_W x PAGE_H, which is what a letter page needs.
        self._pdf.savefig(fig)
        self._pdf_pages += 1
        self.written_sections.add(key)

    def _emit(self, fig: Any, stem: str) -> Path:
        self._index += 1
        png = self.figures_dir / f"{self._index:02d}_{stem}.png"
        fig.savefig(png, dpi=PNG_DPI, bbox_inches="tight")
        if self._pdf is not None:
            page = _page_bbox(fig, stem)
            _rasterize(fig)
            self._write_pending_section()
            self._pdf.savefig(fig, dpi=PDF_DPI, bbox_inches=page)
            self._pdf_pages += 1
        fig.clear()
        self.png_paths.append(png)
        return png

    def emit(self, fig: Any, stem: str) -> Path:
        return self._emit(fig, stem)

    def section(self, key: Any, title: str, text: str = "") -> None:
        """Queue a text-only divider page: a large ``title`` and ``text`` notes below.

        **Buffered, not written.** The divider goes into the PDF immediately before the
        *next figure that is emitted*, and not otherwise: a newer :meth:`section` call
        replaces one still pending (the older divider is never written), and one still
        pending when the report closes is dropped. That is what keeps a section whose
        pages all skipped from leaving an orphan divider behind, and keeps a run where
        nothing drew from writing a PDF at all (``PdfPages`` creates no file until its
        first page is saved).

        PDF-only: no PNG, and it does not advance the ``NN_`` numbering of the figures.
        With ``pdf_path=None`` (a suite's ``pdf: false``) this is a no-op. ``key`` is
        opaque to this class -- it is recorded in :attr:`written_sections` when (and
        only when) the divider is actually written, so the caller can tell which of the
        sections it handed over made it.
        """
        if self._pdf is None:
            return
        self._pending = (key, title, text)

    def get_pagecount(self) -> int | None:
        # PdfPages.get_pagecount() reports 0 once the file is closed -- this
        # object's own running count (figures plus dividers actually written) is the
        # reliable one either way; the figure-only ``_index`` numbers the PNGs.
        return self._pdf_pages if self._pdf is not None else None
