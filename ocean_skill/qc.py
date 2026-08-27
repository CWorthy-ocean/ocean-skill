"""Provider QC flags: recognize them, pair them to their data column, apply them.

**Scope, deliberately narrow**: this module recognizes and applies flags a data
*provider* already computed and shipped in the file (WOCE bottle/CTD codes, Argo/
GTSPP/OceanSITES/Copernicus flags, SeaDataNet's L20 codes, ERDDAP's own QARTOD
companions, ...). It never *runs* a QC test itself -- there is no ioos_qc
dependency here and no test-running design; that is out of scope by user decision
(see the plan this module implements).

The shape it handles is one: **a flag column paired with a data column**, values
rejected in place (set to NaN) according to a policy. Every provider scheme maps
onto one canonical scale -- :data:`QARTOD_FLAGS` -- so a ``keep=`` policy ports
across datasets that used different provider conventions; the mapping is a lens,
not a conversion, so the provider's own flag column and its verbatim definitions
are always retained rather than discarded once translated.

**Architecture: the catalog declares, read time applies.** :func:`resolve_contract`
runs once, at probe/build time (see :mod:`ocean_skill.build`), and its result is
saved verbatim into the catalog entry's ``qc`` metadata -- flag columns detected,
paired to their data column, and (when a scheme is declared or the values are
unambiguous under every registered scheme that could apply -- the *consensus
rule*, below) mapped onto :data:`QARTOD_FLAGS`. :func:`apply` reads that saved
contract at read time (:func:`ocean_skill.sources.read`) and does the actual
masking, with a per-call ``qc=`` override (:func:`effective_policy`) so different
policies can be compared without rebuilding the catalog. An entry with no ``qc``
contract at all gets no QC behaviour, ever -- see :func:`effective_policy`.

**The consensus rule.** When the builder gives no scheme and no explicit mapping,
:func:`resolve_contract` checks which of the other registered schemes
(:func:`compatible_schemes`) define every value actually observed in the flag
column. If they all agree on what those values mean in :data:`QARTOD_FLAGS` terms,
that mapping is adopted automatically -- but loudly: a warning states exactly what
was assumed and how to pin it down instead (``qc={'scheme': ...}``), and the saved
contract records ``scheme: consensus`` plus the adopted mapping, so it is never
silently unclear what happened. If the compatible schemes disagree on any observed
value, nothing is adopted: the flag column is still recognized, paired, and kept in
the output, but not applied, with a warning naming the candidates and exactly how
they disagree.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

from ocean_skill import _stacklevel

__all__ = [
    "QARTOD_FLAGS",
    "SCHEMES",
    "apply",
    "compatible_schemes",
    "detect_flag_columns",
    "effective_policy",
    "expand_scheme",
    "mask_fill_values",
    "pair_flags",
    "resolve_contract",
]

#: The one canonical scale every provider scheme's ``flag_to_qartod`` maps onto,
#: after ioos_qc/QARTOD's own numbering (used verbatim by the ``"qartod"`` entry in
#: :data:`SCHEMES`, for a provider whose raw flags already are QARTOD codes).
QARTOD_FLAGS: dict[str, int] = {
    "GOOD": 1,
    "UNKNOWN": 2,
    "SUSPECT": 3,
    "FAIL": 4,
    "MISSING": 9,
}

#: Named provider flag schemes, each ``{"flag_definitions": {value: text},
#: "flag_to_qartod": {value: QARTOD_name}}``. :func:`expand_scheme` fills a
#: builder's ``qc={"scheme": ...}`` from here; explicit keys the builder gives win
#: over these defaults, value by value.
#:
#: Every mapping beyond the identity one on ``"qartod"`` itself is a judgment call
#: -- there is no universal answer for what e.g. "value changed" or "interpolated"
#: should mean on the GOOD/UNKNOWN/SUSPECT/FAIL/MISSING scale, only a reasonable
#: one. They are visible here (rather than hidden in code) exactly so a builder can
#: see and override one, via ``qc={"scheme": ..., "flag_to_qartod": {value:
#: "GOOD"}}``, the way ``PROFILE_QC`` overrides WOCE's ``6`` (replicate
#: measurements) to ``GOOD``.
SCHEMES: dict[str, dict[str, Any]] = {
    # ioos_qc/QARTOD's own numbering -- the identity mapping, for a provider whose
    # raw flags already are QARTOD codes and only need recognizing, not translating.
    "qartod": {
        "flag_definitions": {
            1: "pass",
            2: "not evaluated",
            3: "suspect or interesting",
            4: "fail",
            9: "missing data",
        },
        "flag_to_qartod": {1: "GOOD", 2: "UNKNOWN", 3: "SUSPECT", 4: "FAIL", 9: "MISSING"},
    },
    # GTSPP / OceanSITES / Copernicus Marine's shared numbering (the "Copernicus
    # standard" the SEANOE mooring record pages cite): 0 no-QC, 1 good, 2 probably
    # good, 3 potentially correctable bad, 4 bad, 5 value changed, 7 nominal,
    # 8 interpolated, 9 missing. No 6 (reserved, unused).
    "argo": {
        "flag_definitions": {
            0: "no quality control performed",
            1: "good data",
            2: "probably good data",
            3: "potentially correctable bad data",
            4: "bad data",
            5: "value changed",
            7: "nominal value",
            8: "interpolated value",
            9: "missing value",
        },
        "flag_to_qartod": {
            0: "UNKNOWN",
            1: "GOOD",
            2: "GOOD",  # judgment call: "probably good" is usable
            3: "SUSPECT",
            4: "FAIL",
            5: "SUSPECT",  # judgment call: provider-adjusted, but flag it for review
            7: "GOOD",  # a nominal (deployment-estimated) value, not itself bad data
            8: "SUSPECT",  # interpolated, not directly measured
            9: "MISSING",
        },
    },
    # WOCE bottle-sample flags, as printed in a file header comment row (e.g. "#
    # Flag scheme: 2- good, 3- questionable, 4 - bad, 6 - replicates, 9 - missing").
    "woce_bottle": {
        "flag_definitions": {
            2: "good",
            3: "questionable",
            4: "bad",
            6: "replicates",
            9: "missing",
        },
        "flag_to_qartod": {
            2: "GOOD",
            3: "SUSPECT",
            4: "FAIL",
            6: "GOOD",  # judgment call: a replicate measurement is still usable
            9: "MISSING",
        },
    },
    # WOCE CTD flags (WOCE Operations Manual / GO-SHIP exchange format) -- a
    # different scale from the bottle one above, despite sharing the "WOCE" name.
    "woce_ctd": {
        "flag_definitions": {
            1: "not calibrated",
            2: "acceptable measurement",
            3: "questionable measurement",
            4: "bad measurement",
            5: "not reported",
            6: "interpolated over a pressure interval larger than 2 dbar",
            7: "despiked",
            8: "not assigned",
            9: "not sampled",
        },
        "flag_to_qartod": {
            1: "SUSPECT",  # not calibrated: unverified, not simply "not evaluated"
            2: "GOOD",
            3: "SUSPECT",
            4: "FAIL",
            5: "MISSING",
            6: "GOOD",  # interpolated but still usable, mirrors woce_bottle's 6
            7: "SUSPECT",  # despiked: the value was altered
            8: "UNKNOWN",
            9: "MISSING",
        },
    },
    # SeaDataNet's L20 quality flag vocabulary: numeric 0-9 plus single-letter
    # codes (SeaDataNet's own "phenomenon uncertain" -- see detect_flag_columns'
    # letter-code allowance).
    "seadatanet": {
        "flag_definitions": {
            0: "no quality control",
            1: "good value",
            2: "probably good value",
            3: "probably bad value",
            4: "bad value",
            5: "changed value",
            6: "value below detection",
            7: "value in excess",
            8: "interpolated value",
            9: "missing value",
            "A": "value phenomenon uncertain",
        },
        "flag_to_qartod": {
            0: "UNKNOWN",
            1: "GOOD",
            2: "GOOD",
            3: "SUSPECT",
            4: "FAIL",
            5: "SUSPECT",
            6: "SUSPECT",
            7: "SUSPECT",
            8: "SUSPECT",
            9: "MISSING",
            "A": "SUSPECT",
        },
    },
}

#: ERDDAP's own QARTOD companion suffixes, checked against the raw (unstripped)
#: column name -- mirrors :data:`ocean_skill.tabular._QC_SUFFIXES`.
_ERDDAP_QC_SUFFIXES = ("_qc_agg", "_qc_tests")

#: Name tokens :func:`detect_flag_columns` looks for, as a whole word in the
#: units-stripped base -- broader than :func:`ocean_skill.tabular.is_qc_column`
#: (which only looks for "flag"): "qc"/"qartod" also count here, since detection
#: additionally *confirms* by value (see :func:`_looks_like_flag_values`), so the
#: broader name net does not by itself sweep in a real measurement.
_FLAG_NAME = re.compile(
    r"(?:^|[^0-9A-Za-z])(?:flags|flag|qartod|qc)(?:[^0-9A-Za-z]|$)", re.IGNORECASE
)

#: Tokens :func:`_strip_flag_token` removes from a flag column's base before
#: pairing it to a data column, in the same "whole word" sense as _FLAG_NAME.
_FLAG_TOKENS = frozenset({"flag", "flags", "qc", "qartod"})


def _normalize_flag_value(v: Any) -> Any:
    """Normalize one flag value so int/float/str/numpy spellings compare equal.

    ``2``, ``2.0``, ``np.float64(2.0)`` and ``"2"`` all normalize to the plain int
    ``2``; a single non-numeric character normalizes to its upper-case form (a
    SeaDataNet-style letter code, e.g. ``"a"``/``"A"``); ``None``/``NaN`` normalize
    to ``None`` (dropped by every caller before comparing).
    """
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            v = float(s)
        except ValueError:
            return s.upper()
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if f != f:  # NaN
        return None
    return int(f) if f.is_integer() else f


def _strip_flag_token(base: str) -> str:
    """Return ``base`` with its trailing flag/qc/qartod token removed.

    ``"TEMP_flag"`` -> ``"TEMP"``, ``"Salinity_CTD_flag"`` -> ``"Salinity_CTD"``,
    ``"x_qc_agg"``/``"x_qc_tests"`` -> ``"x"`` (the ERDDAP suffixes are stripped as
    a whole unit, not token by token, so the pairing lands on the variable's own
    name rather than a name with "_agg"/"_tests" still attached). Used by
    :func:`pair_flags`'s exact-base tier; a base with no such token returns
    unchanged.
    """
    for suffix in _ERDDAP_QC_SUFFIXES:
        if base.lower().endswith(suffix):
            return base[: -len(suffix)]
    parts = re.split(r"([^0-9A-Za-z]+)", base)
    idx = next(
        (i for i in range(len(parts) - 1, -1, -1) if parts[i].lower() in _FLAG_TOKENS),
        None,
    )
    if idx is None:
        return base
    del parts[idx]
    if idx - 1 >= 0 and parts[idx - 1] and not parts[idx - 1][0].isalnum():
        del parts[idx - 1]
    return "".join(parts).strip("_ ").strip()


def _looks_like_flag_values(series) -> bool:
    """Whether ``series``' own values look like flag codes, not a measurement.

    Every observed value must, on its own, look like a code: a small
    non-negative integer (0-9; float-typed integers and NaN allowed, so a flag
    column read as ``float64`` still passes) or a single-character letter code
    (SeaDataNet-style, e.g. ``"A"``). The two are allowed to mix in one column
    (SeaDataNet's own scale does: mostly digits, with a handful of letter
    codes) -- what fails is anything with a *fractional* value (a real physical
    reading, e.g. ``23.456`` degC) or a multi-character non-numeric value,
    which is what correctly rejects ``Temperature_qc[degree_C]``'s own values
    even though its *name* matches.
    """
    values = series.dropna()
    if values.empty:
        return False
    normalized = [_normalize_flag_value(v) for v in values.unique()]
    if any(n is None for n in normalized):
        return False

    def _looks_like_code(n: Any) -> bool:
        if isinstance(n, str):
            return len(n) == 1
        return isinstance(n, int) and 0 <= n <= 9

    return all(_looks_like_code(n) for n in normalized)


def detect_flag_columns(df) -> list[str]:
    """Return ``df``'s columns that look like QC flag companions.

    Two independent tells, both required (name proposes, values confirm): the
    column's own name (``flag``/``flags``/``qc``/``qartod`` as a whole word in
    its units-stripped base, via :func:`ocean_skill.tabular.split_units`, or the
    ERDDAP ``_qc_agg``/``_qc_tests`` suffixes on the raw name) and its values
    (see :func:`_looks_like_flag_values`). Requiring both is what correctly
    accepts ``TEMP_flag`` (2s and 9s) while rejecting ``Temperature_qc[degree_C]``
    (real, fractional degC values) even though its name alone would also match --
    and what leaves an ``station_id``-shaped integer column alone, since its name
    never reaches the value check at all.
    """
    from ocean_skill.tabular import split_units

    out: list[str] = []
    for col in df.columns:
        raw = str(col)
        base, _ = split_units(raw)
        name_hit = raw.endswith(_ERDDAP_QC_SUFFIXES) or bool(_FLAG_NAME.search(base))
        if name_hit and _looks_like_flag_values(df[col]):
            out.append(raw)
    return out


def pair_flags(
    df,
    flag_cols,
    pairs: dict[str, str] | None = None,
    *,
    subject: str = "this source",
) -> dict[str, str]:
    """Return ``{flag_column: data_column}`` for each of ``flag_cols``.

    Three tiers, in order:

    1. ``pairs`` (builder-supplied) -- an explicit override, always tried first.
    2. **Exact base match**: strip the flag token from the flag column's own
       base (:func:`_strip_flag_token`) and look for another column whose own
       units-stripped base matches, case-insensitively
       (``Salinity_CTD_flag`` -> ``Salinity_CTD``).
    3. **Unique case-insensitive prefix match**, when the exact match finds
       nothing: the stripped base as a prefix of exactly one other column's own
       base (``TEMP_flag``'s ``"temp"`` prefixes only ``Temperature_CTD``).
       Two or more candidates is an ambiguous prefix and is left unpaired
       rather than guessed at.

    A flag column matching none of the three is left out of the returned dict
    (not mapped to anything) and named in one warning suggesting ``pairs=``.
    """
    from ocean_skill.tabular import split_units

    pairs_in = {str(k): str(v) for k, v in (pairs or {}).items()}
    flag_set = {str(c) for c in flag_cols}
    others = [str(c) for c in df.columns if str(c) not in flag_set]
    other_bases = {c: split_units(c)[0] for c in others}

    result: dict[str, str] = {}
    unpaired: list[str] = []
    for flag in flag_cols:
        flag = str(flag)
        if flag in pairs_in:
            result[flag] = pairs_in[flag]
            continue
        base, _ = split_units(flag)
        stripped = _strip_flag_token(base).strip()
        low = stripped.casefold()
        exact = [c for c, b in other_bases.items() if b.strip().casefold() == low]
        if len(exact) == 1:
            result[flag] = exact[0]
            continue
        if low:
            prefixed = [
                c for c, b in other_bases.items() if b.strip().casefold().startswith(low)
            ]
            if len(prefixed) == 1:
                result[flag] = prefixed[0]
                continue
        unpaired.append(flag)

    if unpaired:
        warnings.warn(
            f"{subject}: could not pair flag column(s) {unpaired} to a data column "
            "(no exact-base or unique-prefix match) -- pass "
            "qc={'pairs': {" + repr(unpaired[0]) + ": 'the_data_column'}} to fix it.",
            stacklevel=_stacklevel.find(),
        )
    return result


def expand_scheme(spec: dict[str, Any]) -> dict[str, Any]:
    """Fill ``flag_definitions``/``flag_to_qartod`` for ``spec["scheme"]``.

    Looks ``spec["scheme"]`` up in :data:`SCHEMES` and merges its
    ``flag_definitions``/``flag_to_qartod`` under whatever ``spec`` already
    carries for those two keys -- **explicit keys in ``spec`` win**, value by
    value, so ``qc={"scheme": "woce_bottle", "flag_to_qartod": {6: "GOOD"}}``
    overrides only WOCE's own ``6`` and keeps every other registry default.

    An unregistered (or absent, e.g. ``"consensus"``, which names an *adopted*
    mapping rather than a registry entry) scheme leaves ``spec``'s own
    ``flag_definitions``/``flag_to_qartod`` untouched -- there is nothing to
    merge them with.
    """
    out = dict(spec)
    registry = SCHEMES.get(spec.get("scheme"))
    if registry is not None:
        out["flag_definitions"] = {
            **registry["flag_definitions"],
            **(spec.get("flag_definitions") or {}),
        }
        out["flag_to_qartod"] = {
            **registry["flag_to_qartod"],
            **(spec.get("flag_to_qartod") or {}),
        }
    else:
        out.setdefault("flag_definitions", spec.get("flag_definitions") or {})
        out.setdefault("flag_to_qartod", spec.get("flag_to_qartod") or {})
    return out


def compatible_schemes(values) -> list[str]:
    """Registered scheme names whose ``flag_definitions`` cover every value in
    ``values``.

    Deliberately excludes ``"qartod"`` itself: that entry is the canonical
    *output* scale (see the module docstring), not a provider convention to
    guess a dataset follows -- a provider whose raw flags already are QARTOD
    codes should say ``qc={"scheme": "qartod"}`` explicitly rather than being
    auto-detected as if QARTOD were just another candidate among equals. Sorted,
    for a stable, readable warning message.
    """
    normalized = {
        n for n in (_normalize_flag_value(v) for v in values) if n is not None
    }
    return sorted(
        name
        for name, spec in SCHEMES.items()
        if name != "qartod"
        and normalized
        <= {_normalize_flag_value(k) for k in spec["flag_definitions"]}
    )


def _normalized_scheme_map(name: str) -> dict[Any, str]:
    return {
        _normalize_flag_value(k): v for k, v in SCHEMES[name]["flag_to_qartod"].items()
    }


def _consensus(values: set) -> tuple[dict[Any, str] | None, list[str], str]:
    """Return ``(adopted_mapping_or_None, candidate_schemes, message)`` for
    ``values`` under the consensus rule -- see the module docstring.
    """
    candidates = compatible_schemes(values)
    if not candidates:
        return (
            None,
            [],
            f"observed flag value(s) {sorted(values, key=str)} are not fully "
            "covered by any registered scheme (see qc.SCHEMES) -- pass "
            "qc={'flag_to_qartod': {...}} yourself to apply them.",
        )
    per_scheme = {name: _normalized_scheme_map(name) for name in candidates}
    agree = all(len({per_scheme[name][v] for name in candidates}) == 1 for v in values)
    if agree:
        adopted = {v: per_scheme[candidates[0]][v] for v in values}
        readable = ", ".join(
            f"{v}: {m}" for v, m in sorted(adopted.items(), key=lambda kv: str(kv[0]))
        )
        message = (
            f"adopted flag mapping {{{readable}}} (consensus of "
            f"{', '.join(candidates)}, from observed values); pass "
            "qc={'scheme': '" + candidates[0] + "'} (or another name from "
            "qc.SCHEMES) to pin it down instead."
        )
        return adopted, candidates, message
    lines = [
        f"{name}: "
        + ", ".join(f"{v}->{per_scheme[name][v]}" for v in sorted(values, key=str))
        for name in candidates
    ]
    message = (
        f"observed flag value(s) {sorted(values, key=str)} are covered by more than "
        f"one registered scheme, and they disagree on what the values mean "
        f"({'; '.join(lines)}) -- nothing was auto-adopted, so these flags are "
        "recorded but NOT applied. Pass qc={'scheme': '" + candidates[0] + "'} (or "
        "another name from qc.SCHEMES) or qc={'flag_to_qartod': {...}} yourself."
    )
    return None, candidates, message


def _warn_declared_scheme_mismatch(
    scheme: str, flag_to_qartod: dict[Any, str], observed: set, subject: str
) -> None:
    mapped = {_normalize_flag_value(k) for k in flag_to_qartod}
    unknown = observed - mapped
    if unknown:
        warnings.warn(
            f"{subject}: observed flag value(s) {sorted(unknown, key=str)} are not "
            f"covered by scheme {scheme!r} -- left unmasked (neither kept nor "
            "rejected) since there is no declared meaning for them. Add them via "
            "qc={'flag_to_qartod': {...}} if they should be interpreted.",
            stacklevel=_stacklevel.find(),
        )


def _observed_values(df, flag_cols) -> set:
    out: set = set()
    for col in flag_cols:
        if col not in df.columns:
            continue
        for v in df[col].dropna().unique():
            n = _normalize_flag_value(v)
            if n is not None:
                out.add(n)
    return out


def resolve_contract(
    spec: dict[str, Any] | None, df, *, subject: str = "this source"
) -> dict[str, Any] | None:
    """Build the resolved ``qc`` contract for one frame, at probe time.

    ``spec`` is the builder's own ``qc=`` argument -- ``None``/``{}`` are both
    valid (see the module docstring's three-tier input table): flag-column
    detection and pairing always run regardless of what (if anything) ``spec``
    says, but scheme resolution (and therefore ``keep``/applying anything at
    read time) only happens when ``spec`` declares a scheme or an explicit
    ``flag_to_qartod``, or the observed values reach the consensus rule.

    ``spec["flags"]`` may override detection two ways: a **list** of column
    names (detection is skipped, pairing still runs on exactly those columns),
    or a **dict** (the resolved ``{flag_col: data_col}`` pairing itself, given
    outright -- pairing is skipped too).

    Returns ``None`` only when there is truly nothing to record: no flag
    columns detected or declared, and no other ``qc`` spec given at all (no
    ``fill_values``, no scheme, ...) -- the {} vs. ``None`` distinction
    :func:`ocean_skill.build._probe_dataframe` relies on to decide whether the
    entry gets a ``"qc"`` metadata key at all.
    """
    spec = dict(spec or {})
    flags_input = spec.get("flags")
    if isinstance(flags_input, dict):
        flag_cols = [str(c) for c in flags_input]
        pairs = {str(k): str(v) for k, v in flags_input.items()}
    else:
        flag_cols = (
            [str(c) for c in flags_input]
            if flags_input is not None
            else detect_flag_columns(df)
        )
        pairs = pair_flags(df, flag_cols, spec.get("pairs"), subject=subject)

    if not flag_cols and not spec:
        return None

    contract: dict[str, Any] = {"flags": pairs}
    if spec.get("scheme_source"):
        contract["scheme_source"] = str(spec["scheme_source"])
    if spec.get("fill_values"):
        contract["fill_values"] = [float(v) for v in spec["fill_values"]]

    observed = _observed_values(df, pairs) if pairs else set()
    scheme = spec.get("scheme")
    explicit_mapping = spec.get("flag_to_qartod")

    if scheme or explicit_mapping:
        expanded = expand_scheme(spec)
        contract["scheme"] = scheme or "custom"
        if expanded.get("flag_definitions"):
            contract["flag_definitions"] = expanded["flag_definitions"]
        contract["flag_to_qartod"] = expanded.get("flag_to_qartod", {})
        if scheme and observed:
            _warn_declared_scheme_mismatch(
                scheme, contract["flag_to_qartod"], observed, subject
            )
        contract["keep"] = list(spec.get("keep") or ["GOOD"])
        if "keep_provider" in spec:
            contract["keep_provider"] = list(spec["keep_provider"])
    elif observed and pairs:
        adopted, _candidates, message = _consensus(observed)
        warnings.warn(message, stacklevel=_stacklevel.find())
        if adopted is not None:
            contract["scheme"] = "consensus"
            contract["flag_to_qartod"] = adopted
            contract["keep"] = list(spec.get("keep") or ["GOOD"])
            if "keep_provider" in spec:
                contract["keep_provider"] = list(spec["keep_provider"])
    elif pairs:
        warnings.warn(
            f"{subject}: flag column(s) {sorted(pairs)} detected and paired, but no "
            "scheme or mapping was given or could be auto-adopted -- recorded but "
            "NOT applied at read time. Pass qc={'scheme': <name from qc.SCHEMES>} "
            "or qc={'flag_to_qartod': {...}} to enable masking.",
            stacklevel=_stacklevel.find(),
        )
    return contract


def mask_fill_values(df, fill_values, *, time_col: str | None = None):
    """Return a copy of ``df`` with ``fill_values`` replaced by NaN.

    Applied to every numeric column except ``time_col`` (the T axis is never
    what a fill-value convention describes, and masking it would turn a real
    timestamp that happens to collide with a fill number into a dropped row
    downstream). Non-numeric columns are left alone. This is unconditional --
    it runs whenever the contract declares ``fill_values``, independent of
    whether any flag column exists at all, which is what a mooring recipe with
    no scheme/flags (only ``fill_values``) needs.
    """
    import numpy as np
    import pandas as pd

    out = df.copy()
    targets = {float(v) for v in fill_values}
    for col in out.columns:
        if col == time_col:
            continue
        series = out[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        numeric = pd.to_numeric(series, errors="coerce")
        hit = numeric.isin(targets)
        if bool(hit.any()):
            out.loc[hit, col] = np.nan
    return out


def effective_policy(qc_arg: Any, meta: dict[str, Any]) -> dict[str, Any] | None:
    """Return the resolved policy :func:`apply` should use, or ``None``.

    ``None`` means exactly one thing: **the entry has no ``qc`` contract at
    all** (``meta.get("qc")`` is falsy) -- an entry never gets QC behaviour
    invented for it just because a caller passed ``qc=``, and this is also what
    lets a caller's cache key stay byte-identical for a non-QC source (see
    :func:`ocean_skill.comparison.prepare_source`).

    With a contract present, ``qc_arg``:

    - ``None`` -- use the contract's own policy (its ``keep``, default
      ``["GOOD"]`` when a scheme resolved, or no flag masking at all when none
      did) unchanged.
    - a **dict** -- overrides the contract's keys (typically ``keep=`` and/or
      ``keep_provider=``), keeping everything else (``flags``, ``fill_values``,
      ``flag_to_qartod``, ...) from the contract.
    - the string ``"off"`` -- flag masking is skipped entirely (provider values
      pass through untouched); fill masking still runs, since that is about
      recognizing a missing-value sentinel, not about the flag policy.
    """
    contract = meta.get("qc") if meta else None
    if not contract:
        return None
    policy = dict(contract)
    if qc_arg is None:
        return policy
    if isinstance(qc_arg, str):
        if qc_arg == "off":
            return {**policy, "keep": None, "keep_provider": None, "off": True}
        raise ValueError(
            f"qc={qc_arg!r} not recognized -- pass a dict override (e.g. "
            "qc={'keep': ['GOOD', 'SUSPECT']}) or the string 'off'."
        )
    if isinstance(qc_arg, dict):
        return {**policy, **qc_arg}
    raise TypeError(
        f"qc={qc_arg!r}: expected None, a dict override, or 'off', got "
        f"{type(qc_arg).__name__}."
    )


def apply(df, meta: dict[str, Any], policy: Any = None):
    """Apply ``meta``'s resolved ``qc`` contract to ``df``, at read time.

    1. **Fill masking**, unconditional whenever the contract declares
       ``fill_values`` -- every numeric column except the T axis (see
       :func:`mask_fill_values`) -- needed even for an entry with no flags at
       all (a mooring recipe that only declares ``fill_values``).
    2. **Flag masking**, in place, for each ``{flag_col: data_col}`` pair the
       contract names: ``data_col`` is set to NaN wherever the flag's QARTOD
       meaning is not in ``keep`` (or, when ``keep_provider`` is given instead,
       wherever the flag's *raw provider value* is not in it). A value with no
       known QARTOD meaning at all (out of the declared scheme's coverage) is
       left alone -- neither kept nor rejected -- rather than guessed at. No
       new columns are added and nothing is renamed; the flag columns
       themselves ride through unchanged.

    ``policy`` is the per-call override (see :func:`effective_policy`); pass
    ``"off"`` to skip flag masking while still masking fills. Records the
    resolved policy actually used on ``df.attrs["qc_applied"]``, for
    :func:`ocean_skill.tabular.to_dataset`'s provenance. Returns ``df``
    unchanged (the same object) when there is no contract at all -- a true
    no-op, not even a copy.
    """
    resolved = effective_policy(policy, meta)
    if resolved is None:
        return df

    out = df.copy()
    fill_values = resolved.get("fill_values")
    time_col = (meta.get("axes") or {}).get("T")
    if fill_values:
        out = mask_fill_values(out, fill_values, time_col=time_col)

    pairs = resolved.get("flags") or {}
    off = bool(resolved.get("off"))
    keep = resolved.get("keep")
    keep_provider = resolved.get("keep_provider")
    if not off and pairs and (keep is not None or keep_provider is not None):
        flag_to_qartod = {
            _normalize_flag_value(k): v
            for k, v in (resolved.get("flag_to_qartod") or {}).items()
        }
        for flag_col, data_col in pairs.items():
            if flag_col not in out.columns or data_col not in out.columns:
                continue
            values = out[flag_col]
            if keep_provider is not None:
                keep_norm = {_normalize_flag_value(v) for v in keep_provider}
                reject = ~values.map(
                    lambda v, _k=keep_norm: _normalize_flag_value(v) in _k
                )
            else:
                keep_set = set(keep or [])
                meanings = values.map(
                    lambda v, _m=flag_to_qartod: _m.get(_normalize_flag_value(v))
                )
                reject = meanings.notna() & ~meanings.isin(keep_set)
            out.loc[reject, data_col] = float("nan")

    out.attrs = {**getattr(out, "attrs", {}), "qc_applied": resolved}
    return out
