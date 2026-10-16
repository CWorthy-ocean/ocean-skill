"""A publishable table of selected metrics: :func:`ocean_skill.table` and its writers.

:meth:`ComparisonSet.metrics <ocean_skill.comparison.ComparisonSet.metrics>` returns
every metric and every identifying field for every comparison -- the right thing for
analysis, the wrong one for a paper, where a table wants *chosen* metrics, readable
names, units in the headers, a few columns broken out by season, a ``0.91 / 0.85`` cell
holding two depths, and the same cells merged across the rows they describe. This module
is "portrait without the picture": it pivots the same metric records (and refuses the
same duplicate cells,
:func:`ocean_skill.plot.portrait._grid`), formats numbers by the same rule
(:func:`ocean_skill.plot.portrait._fmt_value`), and returns a :class:`MetricsTable`.

**One layout, many writers.** Whether a cell spans three lines is decided once, in
:func:`build_table`, which produces a :class:`_Layout` -- header rows, a body grid with
``rowspan``/``colspan`` on its origin cells -- and every writer (notebook HTML, plain
text, Markdown, LaTeX) only renders it. The merge logic therefore cannot drift between
formats; a format that cannot merge (Markdown, text) leaves the covered cells blank.

**What "split" and "join" mean.** A metric is *split* by a record field when each level
of that field gets its own line (seasons MAM/JJA/SON for ``bias``) and *joined* when the
levels share one cell (``R²`` at surface / column). Records that share a row key but
differ only in the split/join field are one row; a record whose split field is the
"whole" value (no season label) feeds the metrics that are not split, and ``split_all=``
shows it for the split ones too, as one more line. Anything else sharing a cell is a
duplicate and raises, as it does for the portrait.
"""

from __future__ import annotations

import calendar
import html as _html
import itertools
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["MetricsTable", "build_table"]

#: How each metric is spelled in a header, unless ``headers=`` says otherwise.
DEFAULT_HEADERS: dict[str, str] = {
    "r2": "R²",
    "rmse": "RMSE",
    "bias": "bias",
    "corr": "corr",
    "crmsd": "CRMSD",
    "mae": "MAE",
    "sigma_ratio": "σ ratio",
    "n": "n",
}

#: The shown form of an en dash standing for "nothing here" in a joined cell.
_MISSING = "–"

#: The whole-record line of a split metric (``split_all=``), in place of a level.
_ALL = object()

#: The tallest a table's rows may stack in one figure (:meth:`MetricsTable.to_figures`),
#: in inches: what a Letter page holds under a title.
MAX_TABLE_IN = 8.6

#: The writers' file suffixes, for :meth:`MetricsTable.save`.
_SUFFIXES = (".tex", ".md", ".html", ".csv")


# --- the layout every writer renders ------------------------------------------------


@dataclass
class _Cell:
    """One drawn cell: its text, how far it spans, and how it reads."""

    text: str
    rowspan: int = 1
    colspan: int = 1
    align: str = "l"  # "l" | "c" | "r"
    numeric: bool = False  # a number (LaTeX gives its minus sign real math spacing)


@dataclass
class _Col:
    """One body column: its header path and alignment."""

    path: tuple[str, ...]
    align: str = "l"


@dataclass
class _Layout:
    """Header rows, body grid and column list: the one thing every writer reads.

    ``header[r]`` holds only the cells that *start* in header row ``r`` (their
    ``rowspan``/``colspan`` say what they cover). ``body[i][j]`` is the cell starting
    at body line ``i``, column ``j``, or ``None`` where an earlier cell's ``rowspan``
    covers it. ``row_starts`` are the body lines where a new row (a group of sub-rows)
    begins.
    """

    cols: list[_Col]
    header: list[list[_Cell]]
    body: list[list[_Cell | None]]
    row_starts: list[int]
    multiline: bool = False


def _header_rows(cols: list[_Col]) -> list[list[_Cell]]:
    """Stack the columns' header paths into header rows with merged spans.

    A path ``("bias", "MAM")`` puts ``bias`` on the first row, spanning every adjacent
    column that shares it, and ``MAM`` on the second. A shorter path (a column with no
    sub-header) is made one cell as tall as the header, so ``Notes`` sits beside the
    two-level groups rather than above an empty row.
    """
    depth = max((len(c.path) for c in cols), default=1)
    rows: list[list[_Cell]] = []
    for r in range(depth):
        row: list[_Cell] = []
        j = 0
        while j < len(cols):
            path = cols[j].path
            if len(path) <= r:  # covered by a taller cell on an earlier row
                j += 1
                continue
            if len(path) == r + 1:  # this column's last element: spans the rest
                row.append(_Cell(path[r], rowspan=depth - r, align="c"))
                j += 1
                continue
            end = j + 1  # a group: every adjacent column sharing the prefix so far
            while (
                end < len(cols)
                and len(cols[end].path) > r + 1
                and cols[end].path[: r + 1] == path[: r + 1]
            ):
                end += 1
            row.append(_Cell(path[r], colspan=end - j, align="c"))
            j = end
        rows.append(row)
    return rows


def _header_slots(header: list[list[_Cell]], ncols: int) -> list[list[Any]]:
    r"""Place the header cells on a ``rows x ncols`` grid.

    Each slot is the :class:`_Cell` that *starts* there, ``"col"`` where an earlier
    cell on the same row spans across it, or ``"row"`` where a cell from a row above
    spans down onto it -- the distinction a LaTeX writer needs (a ``\\multicolumn``
    swallows its ``&`` separators, a ``\\multirow`` leaves an empty cell).
    """
    slots: list[list[Any]] = [[None] * ncols for _ in header]
    for r, row in enumerate(header):
        pos = 0
        for c in row:
            while pos < ncols and slots[r][pos] is not None:
                pos += 1
            slots[r][pos] = c
            for k in range(1, c.colspan):
                slots[r][pos + k] = "col"
            for rr in range(r + 1, r + c.rowspan):
                for k in range(c.colspan):
                    slots[rr][pos + k] = "row"
            pos += c.colspan
    return slots


# --- text helpers --------------------------------------------------------------------

_TEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
_SUPERSCRIPT_RUN = re.compile("[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+")


def _tex(text: str, *, numeric: bool = False) -> str:
    """Escape ``text`` for LaTeX and spell its symbols (R², σ, °, µ, −, –) in math mode.

    A number's leading minus (ASCII or the Unicode ``−``) becomes ``$-$`` so it has the
    width of a real minus sign instead of a hyphen.
    """
    text = text.replace("−", "-")
    out = "".join(_TEX_SPECIALS.get(ch, ch) for ch in text)
    out = _SUPERSCRIPT_RUN.sub(
        lambda m: "$^{" + m.group(0).translate(_SUPERSCRIPTS) + "}$", out
    )
    out = out.replace("σ", r"$\sigma$").replace("°", r"$^\circ$")
    out = out.replace("µ", r"$\mu$").replace("μ", r"$\mu$")
    out = out.replace("–", "--").replace("—", "---")
    if numeric:
        out = re.sub(r"(?<![\w$])-(?=[\d.])", "$-$", out)
    return out


def _md(text: str) -> str:
    """Escape what would break a GitHub pipe-table cell."""
    return text.replace("|", r"\|").replace("\n", " ")


def _season_like(value: Any) -> bool:
    """Whether ``value`` reads as a season/months label rather than a time window.

    ``"MAM"``, ``"Mar,Apr"`` and ``"MAM 2024-01-01–2024-12-31"`` do; ``"2024-01"`` and a
    ``{"min", "max"}`` window do not.
    """
    from ocean_skill.operators import DEFAULT_SEASONS

    if not isinstance(value, str) or not value.strip():
        return False
    token = value.replace(",", " ").split()[0]
    return token.upper() in DEFAULT_SEASONS or token.title() in calendar.month_abbr[1:]


def _is_whole(field_name: str | None, value: Any) -> bool:
    """Whether ``value`` of the split field is the "whole record" one, not a level.

    ``None`` always is. For ``time`` so is anything that is not a season/months label
    -- a window, a single month's date -- since that is a whole-record request, not one
    slice of a seasonal fan.
    """
    if value is None:
        return True
    return field_name == "time" and not _season_like(value)


def _level_text(field_name: str, value: Any) -> str:
    """One level as a table reads it: depth bands as ``"0-5 m"``, windows as spans."""
    if value is None:
        return ""
    if field_name == "variable":
        from ocean_skill.plot.summary import pretty_level

        return pretty_level(field_name, value)  # "temperature", not the CF name
    if isinstance(value, str):
        return value
    if field_name == "depth":
        from ocean_skill.comparison import _depth_label

        if isinstance(value, int | float) and not isinstance(value, bool):
            return f"{value:g} m"
        return _depth_label(value)
    if field_name == "time" and isinstance(value, dict | list | tuple):
        from ocean_skill.comparison import _time_label

        return _time_label(value)
    if isinstance(value, dict | list | tuple):
        return str(value)
    from ocean_skill.plot.summary import pretty_level

    return pretty_level(field_name, value)


def _ordered_levels(field_name: str, values) -> list:
    """Distinct ``values`` in table order.

    First seen, except where a field has a natural order: seasons and months in
    calendar order, depths shallow to deep.
    """
    from ocean_skill.operators import DEFAULT_SEASONS
    from ocean_skill.plot.summary import _hashable

    seen: dict = {}
    for v in values:
        seen.setdefault(_hashable(v), v)
    levels = list(seen.values())
    if field_name == "time" and levels and all(_season_like(v) for v in levels):
        months = [m.title() for m in calendar.month_abbr[1:]]

        def rank(v):
            tok = v.replace(",", " ").split()[0]
            if tok.upper() in DEFAULT_SEASONS:
                return DEFAULT_SEASONS.index(tok.upper())
            return months.index(tok.title())

        levels.sort(key=rank)
    elif field_name == "depth":

        def depth_rank(v):
            text = _level_text("depth", v).lower()
            if text == "surface":
                return (0, 0.0)
            if "column" in text:
                return (2, 0.0)
            if isinstance(v, int | float):
                return (1, float(v))
            if isinstance(v, dict) and "min" in v:
                return (1, float(v["min"]))
            return (1, math.inf)

        levels.sort(key=depth_rank)  # stable: ties keep first-seen order
    return levels


def _format(name: str, value: Any, round_: bool) -> str:
    """One number as a cell reads it (blank when there is nothing to show)."""
    from ocean_skill.metrics import REGISTRY
    from ocean_skill.plot.portrait import _fmt_value

    if value is None or not np.isfinite(value):
        return ""
    if round_:
        return _fmt_value(name, value)
    metric = REGISTRY.get(name)
    if metric is not None and metric.units == "count":
        return f"{round(float(value))}"
    return f"{float(value):.10g}"


def _display_units(rec: dict[str, Any]) -> str:
    """A record's reference units as a plot prints them (``degC`` → ``°C``)."""
    from ocean_skill.units import display

    units = rec.get("units")
    # "1" is the CF spelling of dimensionless: nothing to print after a number
    return display(units) if units and str(units) != "1" else ""


# --- the table object ----------------------------------------------------------------


class MetricsTable:
    r"""A table of metrics that renders itself in a notebook and writes to any format.

    Built by :func:`ocean_skill.table` / :meth:`ComparisonSet.table
    <ocean_skill.comparison.ComparisonSet.table>`, not directly. Every output reads the
    one internal layout, so merged cells (a station's ``R²`` spanning its three seasonal
    ``bias`` lines) are decided once:

    * a notebook shows an HTML table with real ``rowspan``\\ s (``_repr_html_``);
    * ``str(table)`` is aligned plain text for the terminal;
    * :meth:`to_html`, :meth:`to_markdown`, :meth:`to_latex` return strings --
      Markdown leaves a spanned cell blank on the later lines, LaTeX uses
      ``\\multirow`` (add ``\\usepackage{booktabs, multirow}``);
    * :meth:`to_dataframe` returns the unformatted numbers, and :meth:`save` writes a
      ``.tex``/``.md``/``.html``/``.csv`` file by suffix.
    """

    def __init__(self, layout: _Layout, values: list[dict[str, Any]], spec: dict):
        self._layout = layout
        # one dict per displayed number: row, level, column, metric, slot, value, units
        self._values = values
        # how to_dataframe lays them out (see build_table)
        self._spec = spec

    # -- writers --------------------------------------------------------------------

    def to_html(self) -> str:
        """The table as an HTML ``<table>`` with real ``rowspan``/``colspan`` cells.

        Styled inline with ``currentColor`` rules, so it reads the same in a light or a
        dark notebook and pastes into a document without a stylesheet.
        """
        lay = self._layout
        rule = "1px solid currentColor"
        soft = "1px solid rgba(128,128,128,0.45)"
        esc = _html.escape
        out = [
            '<table style="border-collapse:collapse;font-variant-numeric:'
            'tabular-nums;margin:0.5em 0">',
            "<thead>",
        ]
        for r, row in enumerate(lay.header):
            top = "2px solid currentColor" if r == 0 else "none"
            bottom = rule if r == len(lay.header) - 1 else "none"
            out.append("<tr>")
            for c in row:
                span = (f' colspan="{c.colspan}"' if c.colspan > 1 else "") + (
                    f' rowspan="{c.rowspan}"' if c.rowspan > 1 else ""
                )
                # a group header's underline separates it from its sub-headers
                under = soft if c.colspan > 1 else bottom
                if c.rowspan > 1:
                    under = rule
                out.append(
                    f'<th{span} style="padding:3px 10px;text-align:center;'
                    f"border-top:{top};border-bottom:{under};"
                    f'font-weight:600">{esc(c.text)}</th>'
                )
            out.append("</tr>")
        out.append("</thead><tbody>")
        starts = set(lay.row_starts)
        last = len(lay.body) - 1
        for i, line in enumerate(lay.body):
            border = ""
            if i in starts and i != 0 and lay.multiline:
                border = f"border-top:{soft};"
            if i == last:
                border += "border-bottom:2px solid currentColor;"
            out.append("<tr>")
            for j, c in enumerate(line):
                if c is None:
                    continue
                align = {"l": "left", "c": "center", "r": "right"}[c.align]
                span = f' rowspan="{c.rowspan}"' if c.rowspan > 1 else ""
                valign = "middle" if c.rowspan > 1 else "baseline"
                out.append(
                    f'<td{span} style="padding:2px 10px;text-align:{align};'
                    f'vertical-align:{valign};{border}">{esc(c.text)}</td>'
                )
            out.append("</tr>")
        out.append("</tbody></table>")
        return "".join(out)

    def _repr_html_(self) -> str:
        return self.to_html()

    def _flat_header(self) -> list[str]:
        """One header line: each column's path joined (``"bias MAM"``)."""
        out = []
        for col in self._layout.cols:
            parts: list[str] = []
            for p in col.path:
                if p and (not parts or parts[-1] != p):
                    parts.append(p)
            out.append(" ".join(parts))
        return out

    def to_markdown(self) -> str:
        """Return the table as a GitHub pipe table.

        Cells a span covers are blank after the first line.

        Markdown has one header row, so a two-level header (``bias`` over ``MAM``) is
        flattened to ``"bias MAM"``.
        """
        lay = self._layout
        head = self._flat_header()
        lines = ["| " + " | ".join(_md(h) for h in head) + " |"]
        marks = {"l": ":---", "c": ":---:", "r": "---:"}
        lines.append("| " + " | ".join(marks[c.align] for c in lay.cols) + " |")
        for line in lay.body:
            cells = ["" if c is None else _md(c.text) for c in line]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"

    def to_latex(self) -> str:
        r"""Return the table as a booktabs ``tabular``.

        ``\multirow``/``\multicolumn`` draw the spans.

        Needs ``\usepackage{booktabs, multirow}`` (noted in a leading comment). LaTeX
        specials are escaped and symbols spelled in math mode (R² → ``R$^{2}$``,
        σ → ``$\sigma$``); a number's minus sign becomes ``$-$``. Multi-line rows are
        divided by a ``\cmidrule`` across the table, group headers are underlined with
        ``\cmidrule(lr)``.
        """
        lay = self._layout
        ncols = len(lay.cols)
        spec = "".join(c.align for c in lay.cols)
        out = [
            r"% requires \usepackage{booktabs, multirow}",
            r"\begin{tabular}{" + spec + "}",
            r"\toprule",
        ]
        slots = _header_slots(lay.header, ncols)
        for r, row_slots in enumerate(slots):
            cells: list[str] = []
            underlines: list[tuple[int, int]] = []
            for pos, slot in enumerate(row_slots):
                if slot == "col":
                    continue  # swallowed by the \\multicolumn before it
                if slot == "row":
                    cells.append("")  # the \\multirow above fills it
                    continue
                text = _tex(slot.text)
                if slot.colspan > 1:
                    cells.append(rf"\multicolumn{{{slot.colspan}}}{{c}}{{{text}}}")
                    underlines.append((pos + 1, pos + slot.colspan))
                elif slot.rowspan > 1:
                    cells.append(rf"\multirow{{{slot.rowspan}}}{{*}}{{{text}}}")
                else:
                    cells.append(text)
            out.append(" & ".join(cells) + r" \\")
            if r < len(slots) - 1:
                for lo, hi in underlines:
                    out.append(rf"\cmidrule(lr){{{lo}-{hi}}}")
        out.append(r"\midrule")
        starts = set(lay.row_starts)
        for i, line in enumerate(lay.body):
            if i in starts and i != 0 and lay.multiline:
                out.append(rf"\cmidrule{{1-{ncols}}}")
            cells = []
            for c in line:
                if c is None:
                    cells.append("")
                elif c.rowspan > 1 and c.text:
                    body = _tex(c.text, numeric=c.numeric)
                    cells.append(rf"\multirow{{{c.rowspan}}}{{*}}{{{body}}}")
                elif c.rowspan > 1:
                    cells.append("")  # an empty merged cell needs no \\multirow
                else:
                    cells.append(_tex(c.text, numeric=c.numeric))
            out.append(" & ".join(cells) + r" \\")
        out += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(out) + "\n"

    def __str__(self) -> str:
        """Aligned plain text; spanned cells are blank after their first line."""
        lay = self._layout
        ncols = len(lay.cols)
        # header grid: a spanned cell's text goes on its first column, and a cell
        # spanning several header rows reads on the last of them (nearest the data)
        hgrid = [[""] * ncols for _ in lay.header]
        for r, row_slots in enumerate(_header_slots(lay.header, ncols)):
            for pos, slot in enumerate(row_slots):
                if isinstance(slot, _Cell):
                    hgrid[r + slot.rowspan - 1][pos] = slot.text
        body = [["" if c is None else c.text for c in line] for line in lay.body]
        widths = [
            max([len(h[j]) for h in hgrid] + [len(b[j]) for b in body] + [1])
            for j in range(ncols)
        ]

        def fmt(cells: list[str]) -> str:
            parts = []
            for j, text in enumerate(cells):
                align = lay.cols[j].align
                parts.append(
                    text.rjust(widths[j])
                    if align == "r"
                    else text.center(widths[j])
                    if align == "c"
                    else text.ljust(widths[j])
                )
            return "  ".join(parts).rstrip()

        rule = "-" * (sum(widths) + 2 * (ncols - 1))
        out = [fmt(h) for h in hgrid] + [rule]
        starts = set(lay.row_starts)
        for i, line in enumerate(body):
            if i in starts and i != 0 and lay.multiline:
                out.append(rule)
            out.append(fmt(line))
        out.append(rule)
        return "\n".join(out)

    __repr__ = __str__

    # -- data -----------------------------------------------------------------------

    def to_dataframe(self, *, units: bool = False):
        """The numbers behind the table, unformatted, as a :class:`pandas.DataFrame`.

        ``units=True`` adds a leading ``units`` column: each row's reference units as a
        plot prints them (blank when dimensionless).

        Rows are ``(row, level)`` (``level`` is the split level -- a season -- when
        metrics are split into sub-rows) or just ``row``. Columns are metric names,
        extended to a MultiIndex ``(metric, column, slot)`` where there is a ``cols=``
        level, a joined field, or a split-into-columns level. A value spanning several
        lines (an unsplit metric) sits on the row's first line and is ``NaN`` on the
        rest, mirroring the merged cell -- or on its ``split_all=`` line, when the row
        has one, since the value is that whole-record line's.
        """
        import pandas as pd

        spec = self._spec
        index: list[tuple] = []
        seen: set[tuple] = set()
        data: dict[tuple, dict[tuple, float]] = {}
        for v in self._values:
            ikey = (v["row"], v["level"]) if spec["two_level_index"] else (v["row"],)
            if ikey not in seen:
                seen.add(ikey)
                index.append(ikey)
            path = [v["metric"]]
            if spec["has_cols"]:
                path.append(v["column"])
            if v["metric"] in spec["slotted"]:
                path.append(v["slot"])
            data.setdefault(tuple(path), {})[ikey] = v["value"]
        width = max((len(p) for p in data), default=1)
        paths = [p + ("",) * (width - len(p)) for p in data]
        frame = {
            padded: [cells.get(ikey, np.nan) for ikey in index]
            for padded, cells in zip(paths, data.values(), strict=True)
        }
        names = ["row", "level"] if spec["two_level_index"] else ["row"]
        idx = (
            pd.MultiIndex.from_tuples(index, names=names)
            if len(names) > 1
            else pd.Index([i[0] for i in index], name="row")
        )
        df = pd.DataFrame(frame, index=idx)
        if width == 1:
            df.columns = [p[0] for p in paths]
        else:
            df.columns = pd.MultiIndex.from_tuples(paths)
        if units:
            row_units = {v["row"]: v["units"] for v in self._values}
            key = ("units",) + ("",) * (width - 1) if width > 1 else "units"
            df.insert(0, key, [row_units[i[0]] for i in index])
        return df

    def to_tidy(self):
        """One line per number -- row, level, column, metric, slot, value, units."""
        import pandas as pd

        df = pd.DataFrame(self._values)
        # drop the columns that carry nothing for this table
        return df.loc[:, [c for c in df.columns if (df[c] != "").any()]]

    def to_figures(
        self, *, max_rows: int | None = 24, title: str | None = "Metrics"
    ) -> list:
        """Draw the table as matplotlib figures of at most ``max_rows`` rows each.

        Each figure is as wide as a Letter page, with the header repeated and the
        title of every one after the first ending in ``(cont.)``; text columns wrap
        so the table fits the width, and a long label makes its row taller rather
        than the table wider. A figure also ends early when its rows would no longer
        fit a Letter page. ``max_rows`` counts rows (a row that spans several lines
        counts once); ``None`` limits only by the page. Spanning
        cells (a table built with ``split=``) are drawn on their first line only.
        """
        import textwrap

        import matplotlib.pyplot as plt
        from matplotlib.table import Table

        from ocean_skill.plot.typography import MIN_PT, PAGE_W

        lay = self._layout
        header = self._flat_header()
        if header and not header[0]:
            header[0] = "Comparison"
        body = [[c.text if c is not None else "" for c in line] for line in lay.body]
        aligns = [c.align for c in lay.cols]
        starts = [*lay.row_starts, len(body)]
        rows = [body[a:b] for a, b in itertools.pairwise(starts)]
        if not rows:
            raise ValueError("table has no rows to draw")
        wrap = 34  # characters; wider text wraps onto more lines in its own row
        width_in = PAGE_W - 0.6
        wrapped = [
            [
                [
                    textwrap.wrap(t, wrap) or [""] if a == "l" else [t]
                    for t, a in zip(line, aligns, strict=True)
                ]
                for line in row
            ]
            for row in rows
        ]
        head = [textwrap.wrap(h, wrap) or [""] for h in header]
        # one type size and column widths for every figure, so continued ones match:
        # the size that fits the widest column contents into the page width
        every_line = [line for row in wrapped for line in row]
        size = 8.5
        while True:
            char_in = size * 0.62 / 72
            widths = [
                max(len(x) for cell in col for x in cell) + 3
                for col in zip(head, *every_line, strict=True)
            ]
            if sum(widths) * char_in <= width_in or size <= MIN_PT:
                break
            size -= 0.5
        col_in = [
            w * char_in * min(1.0, width_in / (sum(widths) * char_in)) for w in widths
        ]
        line_in = size * 1.9 / 72
        head_in = max(len(c) for c in head) * line_in + 0.06
        row_in = [
            [max(len(c) for c in line) * line_in + 0.06 for line in row]
            for row in wrapped
        ]
        # a figure takes rows until max_rows, or what a Letter page can hold under
        # the title, is reached (a long label makes a row tall)
        chunks: list[list[int]] = [[]]
        used = head_in
        for i, row in enumerate(row_in):
            if chunks[-1] and (
                (max_rows and len(chunks[-1]) >= max_rows)
                or used + sum(row) > MAX_TABLE_IN
            ):
                chunks.append([])
                used = head_in
            chunks[-1].append(i)
            used += sum(row)

        fig_list = []
        for part, chunk in enumerate(chunks):
            lines = [line for i in chunk for line in wrapped[i]]
            heights = [head_in] + [h for i in chunk for h in row_in[i]]
            title_in = 0.45 if title else 0.1
            fig_h = title_in + sum(heights) + 0.15
            fig = plt.figure(figsize=(PAGE_W, fig_h))
            if title:
                fig.suptitle(
                    f"{title} (cont.)" if part else title,
                    x=0.5,
                    y=1 - 0.1 / fig_h,
                    va="top",
                    fontsize=size + 3,
                )
            ax = fig.add_axes(
                (
                    (PAGE_W - sum(col_in)) / 2 / PAGE_W,
                    0.1 / fig_h,
                    sum(col_in) / PAGE_W,
                    sum(heights) / fig_h,
                )
            )
            ax.axis("off")
            tab = Table(ax, loc="upper left")
            tab.auto_set_font_size(False)
            loc = {"l": "left", "c": "center", "r": "right"}
            for r, cells in enumerate([head, *lines]):
                for c, text in enumerate(cells):
                    tab.add_cell(
                        r,
                        c,
                        col_in[c] / sum(col_in),
                        heights[r] / sum(heights),
                        text="\n".join(text),
                        loc="center" if r == 0 else loc[aligns[c]],
                        facecolor="0.9" if r == 0 else ("0.96" if r % 2 == 0 else "w"),
                        edgecolor="0.75",
                    )
                    cell = tab[r, c]
                    cell.set_linewidth(0.4)
                    cell.get_text().set_fontsize(size)
                    if r == 0:
                        cell.get_text().set_fontweight("bold")
            ax.add_table(tab)
            fig_list.append(fig)
        return fig_list

    def save(self, path: str | Path) -> Path:
        """Write the table to ``path``, choosing the format by suffix.

        ``.tex`` (:meth:`to_latex`), ``.md`` (:meth:`to_markdown`), ``.html``
        (:meth:`to_html`), ``.csv`` (:meth:`to_tidy`, one line per number). Missing
        directories are created; returns the path written.
        """
        path = Path(path).expanduser()
        suffix = path.suffix.lower()
        if suffix not in _SUFFIXES:
            raise ValueError(
                f"cannot save a table as {path.name!r}: the suffix picks the format, "
                f"and must be one of {', '.join(_SUFFIXES)}."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        if suffix == ".csv":
            self.to_tidy().to_csv(path, index=False)
        else:
            text = {
                ".tex": self.to_latex,
                ".md": self.to_markdown,
                ".html": self.to_html,
            }[suffix]()
            path.write_text(text, encoding="utf-8")
        return path


# --- building ------------------------------------------------------------------------


@dataclass
class _Row:
    """One table row (a group of sub-rows): its records and the lines it draws."""

    key: Any
    text: str
    recs: list[dict[str, Any]] = field(default_factory=list)
    by_col: dict[Any, list[dict[str, Any]]] = field(default_factory=dict)
    lines: list[Any] = field(default_factory=lambda: [None])  # split levels, or None
    rep: dict[str, Any] = field(default_factory=dict)  # the record that speaks for it


def _table_records(cs, skip: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """The set's metric records, labelled the way the set labels its own members.

    :func:`ocean_skill.plot.summary._records` plus what a table needs that a plot does
    not: the *set's* label for each member (``cs._label_for``: the pooled or dict name,
    the same one a Taylor point gets, not the comparison's own), the dict key as
    ``group``, and how the member was combined (``average``/``pool`` and over how many).

    ``skip`` names the split/join fields. A label that names one of their levels
    (``temperature 0-5 m MAM``) would give every season and depth its own row, so when
    the labels tell apart comparisons that differ only in those fields they are rebuilt
    by the pooling rule without them (``temperature``). Labels that already group them
    (your ``labels=``, or a dict's keys) are kept.
    """
    from ocean_skill.comparison import _row_labels
    from ocean_skill.plot.summary import _hashable, _records

    recs = _records(cs.comparisons)
    groups = getattr(cs, "_groups", None)
    labels = [str(cs._label_for(i)) for i in range(len(cs.comparisons))]
    if skip and groups is None:
        levels = [tuple(_hashable(r.get(f)) for f in skip) for r in recs]
        rebuilt = _row_labels(cs.comparisons, skip, levels)
        if len(set(rebuilt)) < len(set(labels)):
            labels = rebuilt
    for i, (c, rec) in enumerate(zip(cs.comparisons, recs, strict=True)):
        rec["label"] = labels[i]
        if groups is not None:
            rec["group"] = groups[i]
        combined = getattr(c, "_combined", None)
        if combined:
            rec["_combined"] = combined
        if getattr(c, "notes", None) and not rec.get("notes"):
            rec["notes"] = c.notes
    return recs


def _auto_note(rec: dict[str, Any], skip: set[str]) -> str:
    """Describe a record's selection in a few words: depth, time window, pooled, ...

    ``skip`` names fields the table already shows elsewhere (the row label, a split or
    joined field), so a note never repeats a header.
    """
    parts: list[str] = []
    depth = rec.get("depth")
    if "depth" not in skip and depth is not None:
        text = _level_text("depth", depth)
        if text and text != "no vertical axis":
            parts.append(text)
    time = rec.get("time")
    if "time" not in skip and time is not None:
        text = _level_text("time", time)
        if text:
            parts.append(text)
    combined = rec.get("_combined")
    if combined:
        how, n = combined
        parts.append(f"{'station' if how == 'average' else 'sample'} average ({n})")
    for key in ("demeaned", "detided"):
        if key not in skip and rec.get(key) not in (None, "", "raw"):
            parts.append(str(rec[key]))
    return "; ".join(parts)


def build_table(
    cs,
    *,
    metrics,
    rows: str = "label",
    cols: str | None = None,
    split: dict[str, str] | None = None,
    join: dict[str, str] | None = None,
    join_labels: dict[str, list[str]] | None = None,
    split_all: str | None = None,
    layout: str = "rows",
    headers: dict[str, str] | None = None,
    units: bool = True,
    round: bool = True,
    info: dict[str, dict[str, str]] | None = None,
    notes: dict[str, str] | None = None,
    auto_notes: Any = True,
    save: str | Path | None = None,
) -> MetricsTable:
    """Build a :class:`MetricsTable` from a ``ComparisonSet``.

    The implementation behind :func:`ocean_skill.table`; see there for the parameters.
    """
    from ocean_skill.metrics import REGISTRY
    from ocean_skill.plot.portrait import _resolve_metric_names
    from ocean_skill.plot.summary import _hashable

    recs = _table_records(
        cs, skip=tuple(dict.fromkeys([*(split or {}).values(), *(join or {}).values()]))
    )
    if not recs:
        raise ValueError("table needs at least one comparison, got none")
    names = _resolve_metric_names(recs, metrics)
    if layout not in ("rows", "columns"):
        raise ValueError(
            f"layout={layout!r}: choose 'rows' (split levels as sub-rows) or "
            "'columns' (split levels as column groups)."
        )
    headers = dict(headers or {})
    split = dict(split or {})
    join = dict(join or {})
    if split_all is not None:
        if not isinstance(split_all, str) or not split_all:
            raise ValueError(
                f"split_all= is the name of the whole-record line, a non-empty string "
                f"like 'All year', not {split_all!r}."
            )
        if not split:
            raise ValueError(
                f"split_all={split_all!r} adds a whole-record line to the split "
                "metrics, but nothing is split -- give split={'bias': 'time'} or drop "
                "split_all=."
            )
    for what, mapping in (("split", split), ("join", join)):
        stray = [m for m in mapping if m not in names]
        if stray:
            raise ValueError(
                f"{what}= names {stray} which are not among the table's metrics "
                f"{list(names)} -- add them to metrics= or drop them from {what}=."
            )
        for m, f in mapping.items():
            if not any(f in r for r in recs):
                raise ValueError(
                    f"{what}={{{m!r}: {f!r}}}: no record has a {f!r} field "
                    f"(fields: {sorted(set().union(*recs))})."
                )
    both = [m for m in split if join.get(m) == split[m]]
    if both:
        raise ValueError(
            f"metric(s) {both} split= and join= the same field: a field's levels are "
            "either separate lines or one cell, not both. (Splitting by one field and "
            "joining by another is fine: split={'bias': 'time'}, join={'bias': "
            "'depth'}.)"
        )
    if len(set(split.values())) > 1:
        raise ValueError(
            f"split= uses several fields {sorted(set(split.values()))}; a table can "
            "break metrics out by one field at a time."
        )
    for key, fld in (("rows", rows), ("cols", cols)):
        if fld is not None and fld != "label" and not any(fld in r for r in recs):
            raise ValueError(
                f"{key}={fld!r}: no record has that field "
                f"(fields: {sorted(set().union(*recs))})."
            )

    # -- the split field and its levels --------------------------------------------
    sfield = next(iter(split.values()), None)
    split_levels: list = []
    if sfield is not None:
        split_levels = _ordered_levels(
            sfield,
            (r.get(sfield) for r in recs if not _is_whole(sfield, r.get(sfield))),
        )
    split_metrics = [m for m in names if m in split] if split_levels else []

    def whole(r: dict[str, Any]) -> bool:
        return sfield is None or _is_whole(sfield, r.get(sfield))

    # -- rows (and column groups): first-seen order --------------------------------
    def keyof(r, fld):
        return _hashable(r.get("label") if fld == "label" else r.get(fld))

    row_objs: dict[Any, _Row] = {}
    col_keys: dict[Any, Any] = {}
    for r in recs:
        rk = keyof(r, rows)
        raw = r.get("label") if rows == "label" else r.get(rows)
        row = row_objs.setdefault(
            rk, _Row(rk, str(raw) if rows == "label" else _level_text(rows, raw))
        )
        row.recs.append(r)
        ck = keyof(r, cols) if cols else None
        col_keys.setdefault(ck, r.get(cols) if cols else None)
        row.by_col.setdefault(ck, []).append(r)
    if cols:
        ordered = _ordered_levels(cols, col_keys.values())
        col_order = [_hashable(v) for v in ordered]
    else:
        col_order = [None]
    col_text = {ck: _level_text(cols, col_keys[ck]) for ck in col_order} if cols else {}

    join_levels = {
        m: _ordered_levels(f, (r.get(f) for r in recs if r.get(f) is not None))
        for m, f in join.items()
    }
    join_names: dict[str, list[str]] = {}
    for m, f in join.items():
        given = (join_labels or {}).get(f)
        if given is not None and len(given) != len(join_levels[m]):
            raise ValueError(
                f"join_labels[{f!r}] has {len(given)} names for "
                f"{len(join_levels[m])} levels "
                f"{[_level_text(f, v) for v in join_levels[m]]}"
                " -- give one name per level, in level order."
            )
        join_names[m] = (
            list(given)
            if given is not None
            else [_level_text(f, v) for v in join_levels[m]]
        )

    def dup(row: _Row, what: str, found: list[dict[str, Any]]):
        raise ValueError(
            f"table: more than one comparison has {rows}={row.text!r}"
            + (f", {cols}={what!r}" if cols else "")
            + f" ({len(found)} candidates for one cell) -- a table cell is one number. "
            "Narrow the comparisons, join= the field that tells them apart (a split "
            "metric can be joined too), or choose a rows=/cols= that does."
        )

    def one_cell(row: _Row, ck, m: str, found, level: str, slot: str = ""):
        """The text of one cell from its candidate records, joined if ``m`` is."""
        if m not in join:
            if len(found) > 1:
                dup(row, col_text.get(ck, ""), found)
            val = number(found[0], m) if found else None
            if found:
                emit(row, level, col_text.get(ck, ""), m, slot, val)
            return _format(m, val, round)
        f = join[m]
        slots: dict = {}
        for r in found:
            k = _hashable(r.get(f))
            if k in slots:
                dup(row, col_text.get(ck, ""), found)
            slots[k] = r
        if not slots:
            return ""
        shown = []
        for lv, name in zip(join_levels[m], join_names[m], strict=True):
            r = slots.get(_hashable(lv))
            val = number(r, m) if r is not None else None
            text = _format(m, val, round)
            shown.append(text if text else _MISSING)
            if r is not None:
                emit(
                    row,
                    level,
                    col_text.get(ck, ""),
                    m,
                    f"{slot} {name}".strip(),
                    val,
                )
        return " / ".join(shown)

    for row in row_objs.values():
        present = (
            {_hashable(r.get(sfield)) for r in row.recs if not whole(r)}
            if sfield is not None
            else set()
        )
        lines = [lv for lv in split_levels if _hashable(lv) in present]
        row.lines = lines if (lines and split_metrics) else [None]
        wholes = [r for r in row.recs if whole(r)]
        if split_all is not None and split_metrics and wholes and layout == "rows":
            row.lines = [*(row.lines if row.lines != [None] else []), _ALL]
        row.rep = (wholes or row.recs)[0]

    def candidates(row: _Row, ck, *, level, level_given: bool):
        """The records a cell may read: the level's, or the whole-record ones."""
        found = row.by_col.get(ck, [])
        if level_given and level is not None:
            lk = _hashable(level)
            return [r for r in found if not whole(r) and _hashable(r.get(sfield)) == lk]
        return [r for r in found if whole(r)]

    def number(rec, m):
        v = rec.get(m)
        return None if v is None else float(v)

    # -- units ----------------------------------------------------------------------
    # any of the row's records: a pooled/averaged one can come back without units
    row_units = {
        row.key: next(
            (u for u in map(_display_units, [row.rep, *row.recs]) if u), ""
        )
        for row in row_objs.values()
    }
    unit_set = set(row_units.values())
    shared_units = (
        next(iter(unit_set))
        if units and len(unit_set) == 1 and "" not in unit_set
        else ""
    )
    units_column = units and len(unit_set) > 1 and unit_set != {""}

    def metric_header(m: str) -> str:
        base = headers.get(m, DEFAULT_HEADERS.get(m, m))
        extra = []
        metric = REGISTRY.get(m)
        if shared_units and metric is not None and metric.units == "same":
            extra.append(shared_units)
        if m in join:
            extra.append(" / ".join(join_names[m]))
        return f"{base} ({'; '.join(extra)})" if extra else base

    # -- notes / info ---------------------------------------------------------------
    def lookup(mapping, row: _Row):
        if not mapping:
            return None
        for k in (row.text, row.rep.get("label"), row.key):
            if k in mapping:
                return mapping[k]
        return None

    skip = {m for m in (rows, cols) if m} | set(join.values())
    if sfield is not None and split_levels:
        skip.add(sfield)

    def auto_for(row: _Row) -> str:
        if auto_notes is False:
            return ""
        if callable(auto_notes):
            return str(auto_notes(row.rep) or "")
        if isinstance(auto_notes, dict):
            given = lookup(auto_notes, row)
            if given is not None:
                return str(given)
        elif auto_notes is not True:
            raise ValueError(
                "auto_notes= is True, False, a callable record -> str, or a "
                f"{{row: text}} dict, not {auto_notes!r}."
            )
        return _auto_note(row.rep, skip)

    def user_for(row: _Row) -> str:
        given = lookup(notes, row)
        if given is not None:
            return str(given)
        return next((str(r["notes"]) for r in row.recs if r.get("notes")), "")

    info_cols: list[str] = []
    for per_row in (info or {}).values():
        for k in per_row:
            if k not in info_cols:
                info_cols.append(k)

    row_list = list(row_objs.values())
    auto_text = {r.key: auto_for(r) for r in row_list}
    user_text = {r.key: user_for(r) for r in row_list}

    # -- columns --------------------------------------------------------------------
    # each spec: header path, alignment, whether it spans a row's lines, and a function
    # (row, line index) -> (text, numeric?)
    specs: list[dict[str, Any]] = []
    values: list[dict[str, Any]] = []
    rows_header = headers.get(
        "label", "" if rows == "label" else rows.replace("_", " ").capitalize()
    )
    specs.append(
        {
            "path": (rows_header,),
            "align": "l",
            "span": True,
            "cell": lambda row, i: (row.text, False),
        }
    )

    def emit(row, level, column, metric, slot, value):
        values.append(
            {
                "row": row.text,
                "level": level,
                "column": column,
                "metric": metric,
                "slot": slot,
                "value": np.nan if value is None else value,
                "units": row_units[row.key],
            }
        )

    def level_name(row: _Row, i: int) -> str:
        lv = row.lines[i]
        if lv is _ALL:
            return split_all
        return "" if lv is None else _level_text(sfield, lv)

    def home_line(row: _Row) -> str:
        """Name the line the data writers file a merged (unsplit) value under.

        Its own whole-record line when ``split_all=`` gave the row one -- the value *is*
        that line's -- else the row's first line, like the merged cell.
        """
        return split_all if _ALL in row.lines else level_name(row, 0)

    def plain_spec(m: str, ck):
        hdr = (metric_header(m),) + ((col_text[ck],) if cols else ())

        def cell(row: _Row, i: int):
            found = candidates(row, ck, level=None, level_given=False)
            if len(found) > 1:
                dup(row, col_text.get(ck, ""), found)
            val = number(found[0], m) if found else None
            if i == 0 and found:
                emit(row, home_line(row), col_text.get(ck, ""), m, "", val)
            return _format(m, val, round), True

        return {"path": hdr, "align": "c", "span": True, "cell": cell}

    def join_spec(m: str, ck):
        hdr = (metric_header(m),) + ((col_text[ck],) if cols else ())

        def cell(row: _Row, i: int):
            found = candidates(row, ck, level=None, level_given=False)
            text = one_cell(row, ck, m, found, home_line(row))
            return (text or " / ".join([_MISSING] * len(join_levels[m]))), True

        return {"path": hdr, "align": "c", "span": True, "cell": cell}

    def split_rows_spec(m: str, ck):
        hdr = (metric_header(m),) + ((col_text[ck],) if cols else ())

        def cell(row: _Row, i: int):
            lv = row.lines[i]
            given = lv is not None and lv is not _ALL
            found = candidates(row, ck, level=lv if given else None, level_given=given)
            return one_cell(row, ck, m, found, level_name(row, i)), True

        return {"path": hdr, "align": "c", "span": False, "cell": cell}

    def split_columns_spec(m: str, ck, lv, label: str):
        hdr = (metric_header(m),) + ((col_text[ck],) if cols else ()) + (label,)

        def cell(row: _Row, i: int):
            has_splits = any(not whole(r) for r in row.recs)
            if lv is None:  # "all": the whole-record value (of a row with no levels)
                found = (
                    []
                    if has_splits and split_all is None
                    else candidates(row, ck, level=None, level_given=False)
                )
            else:
                found = candidates(row, ck, level=lv, level_given=True)
            return one_cell(row, ck, m, found, "", label), True

        return {"path": hdr, "align": "c", "span": True, "cell": cell}

    # does any row lack split records (so the columns layout needs an "all" column)?
    any_unsplit = any(row.lines == [None] for row in row_list)
    level_header_done = False
    for m in names:
        for ck in col_order:
            if m in split_metrics:
                if layout == "rows":
                    if not level_header_done:
                        level_header_done = True
                        text = headers.get(
                            "split",
                            "Season"
                            if sfield == "time"
                            and all(_season_like(v) for v in split_levels)
                            else sfield.replace("_", " ").capitalize(),
                        )
                        specs.append(
                            {
                                "path": (text,),
                                "align": "l",
                                "span": False,
                                "cell": lambda row, i: (level_name(row, i), False),
                            }
                        )
                    specs.append(split_rows_spec(m, ck))
                else:
                    for lv in split_levels:
                        specs.append(
                            split_columns_spec(m, ck, lv, _level_text(sfield, lv))
                        )
                    if split_all is not None:
                        specs.append(split_columns_spec(m, ck, None, split_all))
                    elif any_unsplit:
                        specs.append(split_columns_spec(m, ck, None, "all"))
            elif m in join:
                specs.append(join_spec(m, ck))
            else:
                specs.append(plain_spec(m, ck))
    if units_column:
        specs.append(
            {
                "path": (headers.get("units", "Units"),),
                "align": "l",
                "span": True,
                "cell": lambda row, i: (row_units[row.key], False),
            }
        )
    for k in info_cols:
        specs.append(
            {
                "path": (headers.get(k, k),),
                "align": "l",
                "span": True,
                "cell": lambda row, i, k=k: (
                    str((lookup(info, row) or {}).get(k, "")),
                    False,
                ),
            }
        )
    if any(auto_text.values()):
        specs.append(
            {
                "path": (headers.get("notes", "Notes"),),
                "align": "l",
                "span": True,
                "cell": lambda row, i: (auto_text[row.key], False),
            }
        )
    if any(user_text.values()):
        specs.append(
            {
                "path": (headers.get("user_notes", "User notes"),),
                "align": "l",
                "span": True,
                "cell": lambda row, i: (user_text[row.key], False),
            }
        )

    # -- body grid ------------------------------------------------------------------
    cols_list = [_Col(s["path"], s["align"]) for s in specs]
    body: list[list[_Cell | None]] = []
    row_starts: list[int] = []
    for row in row_list:
        n = len(row.lines) if layout == "rows" else 1
        row_starts.append(len(body))
        grid: list[list[_Cell | None]] = [[None] * len(specs) for _ in range(n)]
        for j, s in enumerate(specs):
            if s["span"]:
                text, numeric = s["cell"](row, 0)
                grid[0][j] = _Cell(text, rowspan=n, align=s["align"], numeric=numeric)
            else:
                for i in range(n):
                    text, numeric = s["cell"](row, i)
                    grid[i][j] = _Cell(text, align=s["align"], numeric=numeric)
        body.extend(grid)

    lay = _Layout(
        cols=cols_list,
        header=_header_rows(cols_list),
        body=body,
        row_starts=row_starts,
        multiline=len(body) > len(row_list),
    )
    spec = {
        "two_level_index": layout == "rows" and bool(split_metrics),
        "has_cols": bool(cols),
        "slotted": set(join) | (set(split_metrics) if layout == "columns" else set()),
    }
    table = MetricsTable(lay, values, spec)
    if save is not None:
        table.save(save)
    return table
