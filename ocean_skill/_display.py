r"""Return values that display as the text they contain.

``describe()`` and the ``info()`` helpers return text meant to be *read*, and they are
advertised as one-line interactive calls (``osk.describe("glodap")``). A plain ``str``
echoes its **repr** in a notebook or REPL, so the whole summary arrives on one line with
literal ``\\n`` between fields and every call has to be wrapped in ``print()``.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Self

__all__ = ["Description", "Text"]


class Text(str):
    """A ``str`` that shows itself rendered rather than escaped.

    Subclassing ``str`` rather than returning a bespoke object keeps every string
    operation working — slicing, ``in``, ``+``, ``.splitlines()``, passing it anywhere a
    ``str`` is expected — so nothing that already consumes these return values has to
    change. Only the *display* differs.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return str(self)

    def _repr_html_(self) -> str:
        """Notebook rendering: monospace, wrapped, and escaped.

        Escaped because the content is data — catalog metadata carries URLs, units like
        ``mmol m-3``, and occasionally angle brackets, none of which should be able to
        inject markup into the notebook.
        """
        return (
            "<pre style='white-space:pre-wrap; margin:0'>"
            f"{html.escape(str(self))}</pre>"
        )


class Description(Text):
    """``describe()``'s summary: renders as text, and carries what it rendered.

    Every field here is **transient** -- computed by
    :func:`ocean_skill.catalog.describe` from :func:`ocean_skill.catalog.discover`
    at call time, never written to (or read from) the catalog YAML itself.
    ``catalog_path``/``catalog_paths`` reflect this
    process's search path right now (cwd, ``$OCEAN_SKILL_CATALOGS``,
    :func:`ocean_skill.catalog.add_search_path`); the catalog file has no record of
    its own location, so catalogs stay portable -- copy one elsewhere and it still
    opens.

    Unlike :class:`Text`, instances get a ``__dict__`` (no ``__slots__``) so these
    attributes can be set after construction.
    """

    kind: str
    """``"source"`` or ``"catalog"`` -- which summary this is."""

    name: str
    """The bare name passed to ``describe()``: a source name, or a catalog name."""

    catalog: str
    """The catalog name (equal to ``name`` when ``kind == "catalog"``)."""

    catalog_path: Path
    """The file this name currently resolves from -- the highest-precedence file
    when :attr:`catalog_paths` has more than one entry."""

    catalog_paths: tuple[Path, ...]
    """Every catalog file contributing to this name, low precedence to high. Usually
    one entry; more than one only when two search-path tiers each hold a file with
    this catalog's name."""

    metadata: dict[str, Any]
    """Entry metadata (``kind == "source"``) or catalog-level metadata
    (``kind == "catalog"``), as already shown in the rendered text."""

    sources: tuple[str, ...]
    """The catalog's source names (``kind == "catalog"``), or this one source's own
    name as a single-element tuple (``kind == "source"``)."""

    def __new__(
        cls,
        text: str,
        /,
        *,
        kind: str,
        name: str,
        catalog: str,
        catalog_path: Path,
        catalog_paths: tuple[Path, ...],
        metadata: dict[str, Any],
        sources: tuple[str, ...],
    ) -> Self:
        self = super().__new__(cls, text)
        self.kind = kind
        self.name = name
        self.catalog = catalog
        self.catalog_path = catalog_path
        self.catalog_paths = catalog_paths
        self.metadata = metadata
        self.sources = sources
        return self
