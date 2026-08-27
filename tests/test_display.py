r"""``describe()`` and the ``info()`` helpers display as text, not as an escaped repr.

They are advertised as one-line interactive calls, so echoing them in a notebook or
REPL has to show the summary itself rather than one long line of ``\n`` escapes that
only reads properly once wrapped in ``print()``.
"""

from __future__ import annotations

import ocean_skill as osk
from ocean_skill._display import Text


def _a_source() -> str:
    return sorted(osk.catalogs)[0]


def test_describe_echoes_the_summary_not_an_escaped_repr():
    described = osk.describe(_a_source())

    assert repr(described) == str(described)
    assert "\\n" not in repr(described), "newlines must render, not escape"
    assert repr(described).count("\n") > 1, "a multi-line summary stays multi-line"


def test_describe_is_still_an_ordinary_string():
    """A str subclass, so nothing that already consumed the text has to change."""
    described = osk.describe(_a_source())

    assert isinstance(described, str)
    assert described.startswith("source:")
    assert len(described.splitlines()) > 1
    assert "path" in described
    assert described.upper() == str(described).upper()


def test_print_still_works_unchanged(capsys):
    print(osk.describe(_a_source()))

    out = capsys.readouterr().out
    assert out.startswith("source:")
    assert "\\n" not in out


def test_the_info_helpers_display_the_same_way():
    for text in (osk.cache.info(), osk.outputs.info()):
        assert isinstance(text, Text)
        assert repr(text) == str(text)


def test_match_report_displays_like_describe():
    """osk.match_report() is the same Text contract describe()'s section rides on."""
    report = osk.match_report(_a_source())

    assert isinstance(report, Text)
    assert repr(report) == str(report)
    assert "\\n" not in repr(report), "newlines must render, not escape"
    assert report.startswith(("source:", "catalog:"))


def test_vocabulary_match_report_object_displays_like_text():
    """The underlying vocabulary.MatchReport rides the same contract as Text."""
    from ocean_skill.vocabulary import match_report as vocab_match_report

    report = vocab_match_report(["sea_water_temperature", "Instrument_Type"])

    assert repr(report) == str(report)
    assert "<pre" in report._repr_html_()
    # a variable name carrying markup must not inject into the notebook rendering
    injected = vocab_match_report(["<script>x</script>"])
    assert "<script>" not in injected._repr_html_()
    assert "&lt;script&gt;" in injected._repr_html_()


def test_notebook_html_is_escaped():
    """The content is catalog metadata — data must not be able to inject markup."""
    assert Text("<script>x</script>")._repr_html_().count("&lt;script&gt;") == 1
    assert "<script>" not in Text("<script>x</script>")._repr_html_()


def test_html_preserves_line_structure():
    assert "<pre" in Text("a\nb")._repr_html_()
