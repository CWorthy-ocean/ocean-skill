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

**Packing.** Consecutive figures share a page when they fit: each is stacked under the
one before it (scaled down only if wider than the page) until the next would not fit,
which starts a new page (:meth:`PdfReport.emit`, greedy and in order). A figure wider
than the page is scaled to its width first (so only one too tall after that is kept off
shared pages, on a page grown to hold it). A figure alone on its page that the page has
no room to enlarge is written as before, vector text included; a page of several, or one
that is enlarged (below), is assembled from raster images of the figures, so its text is
raster at no less than :data:`PNG_DPI` on paper.

**Filling.** Once a page's contents are chosen (the packing above is unchanged: nothing
moves between pages for this), the figures on it are scaled up together, by one factor
for the page, so the page's stack -- header, figures, captions and the gaps between
them -- fills the usable page height (:func:`_fill_scale`). No figure may pass the
page's usable width, so a wide page is limited by its widest figure; a narrower one
stays centred. The factor is at most :data:`MAX_FILL_SCALE` (a small figure is not
blown up into a poster), and a gain under :data:`MIN_FILL_GAIN` is not taken.
Only figures scale: captions and the divider's header keep their type size. The
enlarged figure is drawn from a raster render made at :data:`PNG_DPI` times the factor
at :meth:`PdfReport.emit` (while the figure still exists), so its text stays as sharp
as the unscaled PNG's.

:meth:`PdfReport.section` adds a text-only divider (a title plus optional notes) to the
PDF -- and only the PDF: it never gets a PNG and never advances the ``NN_`` numbering of
the figures, so a suite's ``figures/`` directory reads the same with or without
sections. It is buffered rather than written at once, so a section whose pages all
skipped leaves no orphan divider behind (see :meth:`PdfReport.section`). The figure(s)
after it that fit are put on the same page under it; the first is scaled down to fit
when that keeps it at :data:`MIN_FIT_SCALE` of its size, and one that still does not fit
leaves the divider a page of its own, centred.

A figure may carry a caption (``emit(..., caption=)``): small wrapped text under it in
the PDF only.

:meth:`PdfReport.cover` queues the report's cover page(s) the same way: PDF-only,
unnumbered, written once before the first figure (and before any pending divider), and
dropped if nothing draws.

:func:`compress_pdf_images` is an optional post-pass over a finished ``report.pdf``
(``pdf_images: jpeg`` in a suite) that re-encodes its rasterized map images as JPEG
with Ghostscript -- matplotlib's PDF backend only writes them losslessly.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import tempfile
import textwrap
import warnings
from dataclasses import dataclass
from dataclasses import field as _dc_field
from pathlib import Path
from types import TracebackType
from typing import Any

import matplotlib

from ocean_skill.plot.typography import PAGE_H, PAGE_W, STEP, reference_scale

__all__ = [
    "JPEG_QFACTOR",
    "MAX_FILL_SCALE",
    "MIN_FILL_GAIN",
    "MIN_FIT_SCALE",
    "PdfReport",
    "compress_pdf_images",
]

#: Kept between the figure's own tight ink and the top of the page -- purely cosmetic
#: (a page is never rejected for lacking it; see :func:`_page_bbox`).
PAGE_MARGIN = 0.5

#: Packing: the least room kept below the last figure of a page, the space between
#: stacked figures, and the side margins a scaled-down figure keeps (inches). A packed
#: figure's own PNG already carries matplotlib's tight-crop padding (``pad``, below),
#: so the page's first ink sits :data:`PAGE_MARGIN` from the top like a lone figure's.
PAGE_BOTTOM = 0.5
PACK_GAP = 0.15
SIDE_MARGIN = 0.25

#: Packing: the smallest share of its natural size a figure may be scaled down to so
#: that it fits on a page under a divider's header (see :meth:`PdfReport._place`); a
#: figure that would need to shrink further leaves the divider a page of its own.
MIN_FIT_SCALE = 0.7

#: Filling: the most a page's figures are scaled up to use its spare room (see
#: :func:`_fill_scale`). 1.6 keeps a small figure from turning into a poster while a
#: 2/3-empty page still comes close to full; and it bounds the extra raster render
#: (:data:`PNG_DPI` x 1.6 = 320 dpi) a figure that may be enlarged is drawn at.
MAX_FILL_SCALE = 1.6

#: Filling: a page whose spare room would enlarge its figures by less than this is left
#: at their natural size, and a lone figure stays vector: a 10% gain is not worth
#: swapping its vector text for a raster, nor an extra high-resolution render.
MIN_FILL_GAIN = 1.1

#: A caption: type size in points, line pitch as a multiple of it, and the wrap width
#: in characters (about seven inches at that size).
CAPTION_PT = 8.5
CAPTION_SPACING = 1.25
CAPTION_WRAP = 110
CAPTION_GAP = 0.05

#: A divider used as the header of a packed page: the rule and notes sit this far below
#: the title (inches), and this much space is left under the whole block.
HEADER_RULE_GAP = 0.08
HEADER_NOTES_GAP = 0.1
HEADER_BOTTOM_GAP = 0.3

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

#: Where a stand-alone divider's header hangs from: the top of its title, as a fraction
#: of the page height up from the bottom (a little above the vertical middle). The block
#: below it is the same one that heads a packed page (:func:`_draw_header`).
SECTION_TOP_Y = 0.6

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
    fully vector/raster as the renderer already chose. Contour sets are left alone:
    matplotlib cannot rasterize them (it warns and ignores the request), and a few
    contour lines are light as vectors anyway.
    """
    from matplotlib.contour import ContourSet

    for ax in fig.axes:
        for artist in (*ax.collections, *ax.images):
            if not isinstance(artist, ContourSet):
                artist.set_rasterized(True)


def _width_scale(width: float) -> float:
    """Return the factor (at most 1) that fits a figure ``width`` inches wide to a page.

    A figure wider than the page is scaled to the page's usable width, side margins
    kept; a narrower one is left at its own size.
    """
    if width <= PAGE_W:
        return 1.0
    return (PAGE_W - 2 * SIDE_MARGIN) / width


def _fill_scale(items: list[_Item], header: tuple[Any, str, str] | None) -> float:
    """Return the one factor (1 to :data:`MAX_FILL_SCALE`) that fills a page's height.

    The page stack is the header (if any), then each figure at ``h`` x the factor with
    its caption (type size unscaled) under it, :data:`PACK_GAP` between figures, in the
    room between the first figure's top and :data:`PAGE_BOTTOM`. The factor is the
    least of what that room allows, what the widest figure's width allows
    (:data:`SIDE_MARGIN` each side), and :data:`MAX_FILL_SCALE`. A page with an
    oversize figure (already on a grown page of its own), or a factor under
    :data:`MIN_FILL_GAIN`, stays at 1.
    """
    if not items or any(it.oversize for it in items):
        return 1.0
    room = PAGE_H - _top_margin() - PAGE_BOTTOM
    fixed = PACK_GAP * (len(items) - 1)
    fixed += sum(_caption_height(it.caption) for it in items)
    if header is not None:
        fixed += _header_height(header[1], header[2])
    by_height = (room - fixed) / sum(it.h for it in items)
    by_width = (PAGE_W - 2 * SIDE_MARGIN) / max(it.w for it in items)
    scale = min(MAX_FILL_SCALE, by_height, by_width)
    return scale if scale >= MIN_FILL_GAIN else 1.0


def _overflows(tight: Any) -> bool:
    """Report whether ink ``tight`` is bigger than a page can hold on its own.

    Width does not count: a figure wider than the page is scaled down to it (see
    :func:`_width_scale`), so it only overflows when it is still too tall after that.
    """
    height = tight.height * _width_scale(tight.width)
    return height + PAGE_MARGIN > PAGE_H


def _tight_bbox(fig: Any, stem: str) -> Any:
    """Return ``fig``'s tight bounding box (inches), warning if it overflows a page.

    A figure that already fits comfortably inside 8.5x11 (the common case, since figures
    are drawn against that canvas by default -- see :mod:`ocean_skill.plot.typography`)
    is placed on an exact letter page by :func:`_page_bbox`; one that does not -- an
    explicit ``figsize=`` wider or taller than the page slipped past
    :func:`ocean_skill.workflows.pages._pin_to_page` -- gets a page grown just enough to
    hold it, with this warning, rather than a clipped page.
    """
    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    if _overflows(tight):
        warnings.warn(
            f"page {stem!r}: figure ({tight.width:.2f}x{tight.height:.2f}in) does not "
            f"fit an {PAGE_W}x{PAGE_H}in page even with pinning -- growing the page "
            "instead of clipping the figure",
            stacklevel=3,
        )
    return tight


def _page_bbox(tight: Any) -> Any:
    """Return the Bbox (inches) placing ink ``tight`` on one US Letter portrait page.

    The ink is anchored to the top of the page, centred left-right, with
    :data:`PAGE_MARGIN` of headroom above it; a page it overflows is grown to hold it.
    """
    from matplotlib.transforms import Bbox

    width = max(tight.width, PAGE_W)
    height = max(tight.height + PAGE_MARGIN, PAGE_H)
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
    The header itself is :func:`_draw_header`, the block that also heads a packed page,
    hung from :data:`SECTION_TOP_Y`. Built directly rather than through
    :mod:`matplotlib.pyplot`, so it is never registered with pyplot's figure manager and
    there is nothing to close afterwards.
    """
    from matplotlib.figure import Figure

    fig = Figure(figsize=(PAGE_W, PAGE_H))
    _draw_header(fig, PAGE_W / 2, SECTION_TOP_Y * PAGE_H, title, text)
    return fig


@contextlib.contextmanager
def _quiet():
    """Mute warnings while a page is written to the PDF.

    A page is written after later ones have been emitted, and drawing its figure again
    would repeat the layout warnings its PNG raised in :meth:`PdfReport.emit` -- where
    the runner can tell which page they belong to. Here they would be blamed on
    whichever page happens to be emitting.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _top_margin() -> float:
    """Inches from the top of a packed page to its first figure's PNG edge.

    The PNG already carries matplotlib's tight-crop padding, so this is
    :data:`PAGE_MARGIN` less that: the first ink lands where a lone figure's does.
    """
    return PAGE_MARGIN - matplotlib.rcParams["savefig.pad_inches"]


def _text_height(lines: list[str], size: float, spacing: float) -> float:
    """Inches ``lines`` of ``size`` points occupy at ``spacing`` times that pitch."""
    return len(lines) * size * spacing / 72


def _caption_height(lines: list[str]) -> float:
    """Inches a wrapped caption takes under a figure (0 for none)."""
    if not lines:
        return 0.0
    return CAPTION_GAP + _text_height(lines, CAPTION_PT, CAPTION_SPACING)


def _header_height(title: str, text: str) -> float:
    """Inches a divider takes as a page header, the space under it included."""
    title_pt = reference_scale()["suptitle"] * STEP**SECTION_TITLE_STEPS
    height = _text_height(_wrap_lines(title, SECTION_TITLE_WRAP), title_pt, 1.2)
    if text.strip():
        body_pt = reference_scale()["title"]
        height += HEADER_RULE_GAP + HEADER_NOTES_GAP
        height += _text_height(_wrap_lines(text, SECTION_NOTES_WRAP), body_pt, 1.5)
    return height + HEADER_BOTTOM_GAP


def _draw_caption(fig: Any, x: float, y_top: float, lines: list[str]) -> None:
    """Draw ``lines`` centred at ``x`` hanging from ``y_top`` (inches in ``fig``)."""
    w, h = fig.get_size_inches()
    fig.text(
        x / w,
        y_top / h,
        "\n".join(lines),
        transform=fig.transFigure,
        ha="center",
        va="top",
        fontsize=CAPTION_PT,
        color="0.25",
        linespacing=CAPTION_SPACING,
    )


def _draw_header(fig: Any, x: float, y_top: float, title: str, text: str) -> None:
    """Draw a divider's title, rule and notes at ``x`` hanging from ``y_top`` (inches).

    Positions are fractions of ``fig``'s own size through ``fig.transFigure``, which is
    what a tight ``bbox_inches`` shifts; the block takes :func:`_header_height`.
    """
    from matplotlib.lines import Line2D

    w, h = fig.get_size_inches()
    title_pt = reference_scale()["suptitle"] * STEP**SECTION_TITLE_STEPS
    lines = _wrap_lines(title, SECTION_TITLE_WRAP)
    fig.text(
        x / w,
        y_top / h,
        "\n".join(lines),
        transform=fig.transFigure,
        ha="center",
        va="top",
        fontsize=title_pt,
        fontweight="bold",
        multialignment="center",
        linespacing=1.2,
    )
    y = y_top - _text_height(lines, title_pt, 1.2) - HEADER_RULE_GAP
    fig.add_artist(
        Line2D(
            [(x - 1.1) / w, (x + 1.1) / w],
            [y / h] * 2,
            color="0.55",
            linewidth=0.8,
            transform=fig.transFigure,
        )
    )
    if text.strip():
        fig.text(
            x / w,
            (y - HEADER_NOTES_GAP) / h,
            "\n".join(_wrap_lines(text, SECTION_NOTES_WRAP)),
            transform=fig.transFigure,
            ha="center",
            va="top",
            fontsize=reference_scale()["title"],
            color="0.25",
            multialignment="left",
            linespacing=1.5,
        )


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
class _Item:
    """One figure waiting for its PDF page: its PNG, its size, and what goes with it."""

    png: Path
    #: Inches the figure takes on a packed page: its PNG's size, scaled to the width
    #: (or, under a divider's header, to fit; see :meth:`PdfReport._place`).
    w: float
    h: float
    #: The PNG's own size in inches, which :attr:`w` and :attr:`h` are scaled from.
    nat_w: float
    nat_h: float
    #: The caption, already wrapped (empty for none).
    caption: list[str]
    #: The figure's tight bounding box, for placing it alone (``None`` once packed).
    tight: Any
    #: The figure itself, kept only while it may still be written alone as vector.
    fig: Any
    #: Whether it, caption included, is too tall for a page on its own even scaled to
    #: the page's width: it is then written alone, on the page grown to hold it, never
    #: packed.
    oversize: bool = False
    #: Whether it is wider than the page, or was scaled to fit under a header: it is
    #: then always drawn from its PNG at :attr:`w` x :attr:`h`, never as vector.
    scaled: bool = False
    #: A PNG of the figure rendered at :data:`PNG_DPI` x the largest factor a page
    #: could enlarge it by, kept only for a figure that may be enlarged (see
    #: :func:`_fill_scale`); ``None`` otherwise.
    hires: bytes | None = None

    @property
    def need(self) -> float:
        """Inches of page height this item uses, caption included."""
        return self.h + _caption_height(self.caption)


@dataclass
class PdfReport:
    """Write PNGs (always) and a PDF (unless ``pdf_path`` is ``None``) as pages arrive.

    Used as a context manager: pages are added with :meth:`emit`; the PDF is
    opened before the first page and closed on exit even if a page's own drawing
    raised, so a partially-run report is never left with a corrupt or half-written
    PDF. :meth:`section` adds a PDF-only divider between them.

    A PDF page is filled greedily: figures (and a divider's header above them) collect
    in an open page until the next would not fit in what is left, and then the page is
    written. So nothing reaches the PDF until the page after it starts or the report
    closes; :meth:`get_pagecount` counts only pages already written.

    ``written_sections`` collects the key of every divider that actually reached the
    PDF, so a caller can tell a section it handed over from one that was written (see
    :meth:`section` for when one is dropped instead).
    """

    pdf_path: Path | None
    figures_dir: Path
    _pdf: Any = _dc_field(default=None, repr=False)
    _index: int = 0
    _pdf_pages: int = 0
    _header: tuple[Any, str, str] | None = _dc_field(default=None, repr=False)
    _items: list[_Item] = _dc_field(default_factory=list, repr=False)
    _used: float = 0.0
    _cover: list[Any] | None = _dc_field(default=None, repr=False)
    png_paths: list[Path] = _dc_field(default_factory=list)
    written_sections: set[Any] = _dc_field(default_factory=set)
    cover_written: bool = False

    def __post_init__(self) -> None:
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> PdfReport:
        if self.pdf_path is not None:
            from matplotlib.backends.backend_pdf import PdfPages

            self._pdf = PdfPages(self.pdf_path)
            self._pdf.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Write the open page and close the PDF, whatever happened inside the block.

        When an exception is already on its way out, a failure here is warned about
        rather than raised: a second error would replace the first, the one the caller
        needs to see.
        """
        try:
            try:
                # The open page has figures: write it. A divider (or cover) with no
                # figure after it is dropped, never written.
                self._flush()
            except Exception as flush_error:
                if exc_type is None:
                    raise
                warnings.warn(
                    f"the last page of the report could not be written "
                    f"({flush_error!r}) while {exc_type.__name__} was already being "
                    "raised; closing the PDF without it",
                    stacklevel=2,
                )
        finally:
            self._header = None
            self._cover = None
            if self._pdf is not None:
                try:
                    self._pdf.__exit__(exc_type, exc, tb)
                except Exception as close_error:
                    if exc_type is None:
                        raise
                    warnings.warn(
                        f"the report's PDF could not be closed cleanly "
                        f"({close_error!r}) while {exc_type.__name__} was already "
                        "being raised",
                        stacklevel=2,
                    )

    def _write_pending_cover(self) -> None:
        """Write the buffered cover page(s) (if any) into the PDF, then clear them."""
        if self._cover is None or self._pdf is None:
            return
        figs, self._cover = self._cover, None
        for fig in figs:
            self._pdf.savefig(fig)  # exact letter pages: no bbox_inches, like a divider
            self._pdf_pages += 1
        self.cover_written = bool(figs)

    def _flush(self) -> None:
        """Write the open page (if it holds a figure) and start a new one."""
        header, items = self._header, self._items
        self._header, self._items, self._used = None, [], 0.0
        if self._pdf is None or not items:
            return
        with _quiet():
            self._write_pending_cover()
            fill = _fill_scale(items, header)
            if len(items) == 1 and not items[0].scaled and fill == 1.0:
                self._write_alone(items[0], header)
            else:
                self._write_packed(items, header, fill)
        self._pdf_pages += 1
        if header is not None:
            self.written_sections.add(header[0])
        for item in items:
            item.fig.clear()
            item.hires = None

    def _write_alone(self, item: _Item, header: tuple[Any, str, str] | None) -> None:
        """Write one figure (and its header and caption) as a vector page of its own."""
        from matplotlib.transforms import Bbox

        fig, tight = item.fig, item.tight
        centre = tight.x0 + tight.width / 2
        top, bottom = tight.y1, tight.y0
        if header is not None:
            height = _header_height(header[1], header[2])
            top += height
            _draw_header(fig, centre, top, header[1], header[2])
        if item.caption:
            _draw_caption(fig, centre, bottom - CAPTION_GAP, item.caption)
            bottom -= _caption_height(item.caption)
        page = _page_bbox(Bbox([[tight.x0, bottom], [tight.x1, top]]))
        _rasterize(fig)
        self._pdf.savefig(fig, dpi=PDF_DPI, bbox_inches=page)

    def _write_packed(
        self,
        items: list[_Item],
        header: tuple[Any, str, str] | None,
        fill: float = 1.0,
    ) -> None:
        """Write figures stacked top down, as one page assembled from raster images.

        Each figure is drawn ``fill`` times its :attr:`_Item.w` x :attr:`_Item.h`, from
        its high-resolution render when ``fill`` > 1 (see :func:`_fill_scale`); the
        header and captions stay at their own size.
        """
        import io

        from matplotlib.figure import Figure
        from PIL import Image

        page = Figure(figsize=(PAGE_W, PAGE_H))
        y = PAGE_H - _top_margin()
        if header is not None:
            _draw_header(page, PAGE_W / 2, y, header[1], header[2])
            y -= _header_height(header[1], header[2])
        for i, item in enumerate(items):
            if i:
                y -= PACK_GAP
            w, h = item.w * fill, item.h * fill
            ax = page.add_axes(
                [(PAGE_W - w) / 2 / PAGE_W, (y - h) / PAGE_H, w / PAGE_W, h / PAGE_H]
            )
            source = io.BytesIO(item.hires) if fill > 1.0 and item.hires else item.png
            with Image.open(source) as img:
                # interpolation="none" makes matplotlib embed the pixels as they are
                ax.imshow(img.convert("RGB"), interpolation="none", aspect="auto")
            ax.set_axis_off()
            y -= h
            if item.caption:
                _draw_caption(page, PAGE_W / 2, y - CAPTION_GAP, item.caption)
                y -= _caption_height(item.caption)
        self._pdf.savefig(page)  # exact letter page: no bbox_inches, like a divider

    def _write_header_page(self, header: tuple[Any, str, str]) -> None:
        """Write a divider with no figure under it as a page of its own."""
        fig = _section_figure(header[1], header[2])
        # No bbox_inches: a tight crop would shrink the page to its ink. The figure is
        # already exactly PAGE_W x PAGE_H, which is what a letter page needs.
        self._pdf.savefig(fig)
        self._pdf_pages += 1
        self.written_sections.add(header[0])

    def emit(self, fig: Any, stem: str, caption: str | None = None) -> Path:
        """Save ``fig`` as ``figures/NN_<stem>.png`` and queue it for the PDF.

        ``caption`` is drawn under the figure in the PDF (wrapped to the page); the PNG
        is the bare figure. The figure is measured here, so a warning about its size
        belongs to the page that drew it, and is cleared once its PDF page is written.
        """
        self._index += 1
        png = self.figures_dir / f"{self._index:02d}_{stem}.png"
        fig.savefig(png, dpi=PNG_DPI, bbox_inches="tight")
        self.png_paths.append(png)
        if self._pdf is None:
            fig.clear()
            return png
        from PIL import Image

        tight = _tight_bbox(fig, stem)
        with Image.open(png) as img:
            w, h = (px / PNG_DPI for px in img.size)
        scale = min(1.0, (PAGE_W - 2 * SIDE_MARGIN) / w)
        wide = tight.width > PAGE_W
        item = _Item(
            png=png,
            w=w * scale,
            h=h * scale,
            nat_w=w,
            nat_h=h,
            caption=_wrap_lines(caption, CAPTION_WRAP) if caption else [],
            tight=tight,
            fig=fig,
        )
        # Too tall for an empty page, its caption counted: alone, on a page grown to it.
        item.oversize = item.need > PAGE_H - _top_margin() - PAGE_BOTTOM
        item.scaled = wide and not item.oversize
        # A figure that a page could enlarge (alone it would be) gets a render of the
        # size it would then need, now, while the figure still exists: a packed page is
        # made after the figure is cleared, and a page's factor is never above its
        # figures' own (more content only takes room).
        if (cap := _fill_scale([item], None)) > 1.0:
            item.hires = self._render_hires(fig, PNG_DPI * cap)
        self._place(item)
        return png

    @staticmethod
    def _render_hires(fig: Any, dpi: float) -> bytes:
        """Return ``fig`` as PNG bytes, tight-cropped like its PNG, at ``dpi``."""
        import io

        buf = io.BytesIO()
        with _quiet():  # its layout warnings were raised, and attributed, by the PNG
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        return buf.getvalue()

    def _place(self, item: _Item) -> None:
        """Add ``item`` to the open page, first writing it out if ``item`` won't fit."""
        gap = PACK_GAP if self._items else 0.0
        room = PAGE_H - _top_margin() - PAGE_BOTTOM
        if self._header and not self._items and self._used + item.need > room:
            self._fit_under_header(item, room - self._used)
        full = self._used + gap + item.need > room
        # An oversize figure takes a page to itself, and nothing joins it there.
        if item.oversize or any(it.oversize for it in self._items):
            full = True
        if (self._items or self._header) and full:
            if self._items:
                self._flush()
            else:  # a divider that has no room left for a figure under it
                with _quiet():
                    self._write_pending_cover()
                    self._write_header_page(self._header)
                self._header, self._used = None, 0.0
            gap = 0.0
        self._items.append(item)
        self._used += gap + item.need
        if len(self._items) > 1:  # packed pages are made from PNGs: free the figures
            for it in self._items:
                it.fig.clear()

    @staticmethod
    def _fit_under_header(item: _Item, avail: float) -> None:
        """Scale ``item`` down to fit ``avail`` inches under a divider, if slight.

        The figure and its caption must fit in ``avail`` (the caption is not scaled).
        That is done when it keeps the figure at :data:`MIN_FIT_SCALE` or more of its
        natural size; otherwise nothing changes and the divider gets a page to itself.
        A figure so shrunk is no longer oversize: it is packed, centred, below it.
        """
        scale = (avail - _caption_height(item.caption)) / item.nat_h
        if scale >= MIN_FIT_SCALE:
            item.w, item.h = item.nat_w * scale, item.nat_h * scale
            item.oversize, item.scaled = False, True

    def section(self, key: Any, title: str, text: str = "") -> None:
        """Queue a text-only divider: a large ``title`` and ``text`` notes below.

        **Buffered, not written.** The divider goes into the PDF with the *next figure
        that is emitted*, and not otherwise: above it on the same page when the figure
        fits under it, else on a page of its own just before. A newer :meth:`section`
        call replaces one still pending (the older divider is never written), and one
        still pending when the report closes is dropped. That is what keeps a section
        whose pages all skipped from leaving an orphan divider behind, and keeps a run
        where nothing drew from writing a PDF at all (``PdfPages`` creates no file until
        its first page is saved).

        PDF-only: no PNG, and it does not advance the ``NN_`` numbering of the figures.
        With ``pdf_path=None`` (a suite's ``pdf: false``) this is a no-op. ``key`` is
        opaque to this class -- it is recorded in :attr:`written_sections` when (and
        only when) the divider is actually written, so the caller can tell which of the
        sections it handed over made it.
        """
        if self._pdf is None:
            return
        self._flush()  # a new section starts a new page
        self._header = (key, title, text)
        self._used = _header_height(title, text)

    def cover(self, figs: list[Any]) -> None:
        """Queue the report's cover page(s): letter-size figures, written as page 1.

        Buffered like :meth:`section`: the pages go into the PDF immediately before the
        first page that holds a figure (ahead of any pending divider), and are dropped
        if the report closes without one -- so a run where nothing drew writes no PDF.
        PDF-only: no PNG, and it does not advance the ``NN_`` numbering. A no-op with
        ``pdf_path=None``; :attr:`cover_written` says whether the pages made it in.
        """
        if self._pdf is None:
            return
        self._cover = list(figs)

    def get_pagecount(self) -> int | None:
        # PdfPages.get_pagecount() reports 0 once the file is closed -- this
        # object's own running count (cover, dividers and figure pages actually
        # written; not the one still open) is the reliable one either way; the
        # figure-only ``_index`` numbers the PNGs.
        return self._pdf_pages if self._pdf is not None else None
