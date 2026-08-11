r"""Return values that display as the text they contain.

``describe()`` and the ``info()`` helpers return text meant to be *read*, and they are
advertised as one-line interactive calls (``osk.describe("glodap")``). A plain ``str``
echoes its **repr** in a notebook or REPL, so the whole summary arrives on one line with
literal ``\\n`` between fields and every call has to be wrapped in ``print()``.
"""

from __future__ import annotations

import html

__all__ = ["Text"]


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
