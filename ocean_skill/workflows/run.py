"""Run a comparison suite: ``python -m ocean_skill.workflows.run suite.yaml``.

A suite is a YAML file describing which sources to compare, over which variables and
depths, and what to draw — the same objects :func:`ocean_skill.compare` builds, just
declared instead of coded. That makes the regular-run case (cron, CI, a during-run hook)
a single command with no Python to edit, while ad-hoc work still uses the API directly.

Optionally the suite can refresh a model's kerchunk reference before comparing, which is
what makes it usable against a run that is still writing output.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

from ocean_skill.comparison import SURFACE

__all__ = ["main", "run_suite"]


def _refresh_sources(spec: list[dict[str, Any]], catalog_path: str | Path) -> None:
    """Rebuild kerchunk references and their catalog entries from the current files.

    Reopens the existing catalog (if any) and updates it in place, rather than
    building a fresh one from scratch: a run between output steps — or one that was
    restarted and has temporarily produced no files — must not wipe out every other
    entry the catalog already had. A source whose glob matches nothing simply keeps
    whatever entry it already had; only sources that actually rebuilt get touched.

    An entry's ``keep`` key (default ``"all"``) is forwarded to
    :func:`ocean_skill.build.make_kerchunk` — a restart stream that is still being
    refreshed against a live run declares ``keep: latest-per-file`` to drop each
    file's earlier, superseded record.
    """
    import glob as _glob

    import intake

    from ocean_skill.build import add_source, make_kerchunk, new_catalog, save

    catalog_path = Path(catalog_path).expanduser()
    cat = None
    if catalog_path.exists():
        try:
            cat = intake.from_yaml_file(str(catalog_path))
        except Exception as exc:
            print(
                f"  refresh: could not reopen existing {catalog_path} ({exc}); "
                "starting a fresh catalog"
            )
    if cat is None:
        cat = new_catalog(title="refreshed by ocean_skill.workflows.run")

    rebuilt, skipped = [], []
    for entry in spec:
        # glob.glob (not Path.glob) so absolute patterns work
        files = sorted(_glob.glob(str(Path(entry["files"]).expanduser())))
        if not files:
            skipped.append(entry["name"])
            continue
        ref = make_kerchunk(
            files,
            entry["ref"],
            grid=entry.get("grid"),
            keep=entry.get("keep", "all"),
        )
        add_source(cat, entry["name"], ref)
        rebuilt.append(entry["name"])
        print(f"  refresh: {entry['name']} <- {len(files)} files")

    if skipped:
        print(f"  refresh: no files matched for {skipped}; kept their existing entries")
    if not rebuilt and not skipped:
        return  # nothing declared to refresh; leave the file untouched
    save(cat, catalog_path)


def run_suite(suite_path: str | Path):
    """Load and execute a suite YAML; return the resulting :class:`ComparisonSet`."""
    import ocean_skill as osk

    suite = yaml.safe_load(Path(suite_path).expanduser().read_text())
    out_dir = Path(suite.get("output_dir", f"output/{suite.get('project', 'suite')}"))

    if refresh := suite.get("refresh"):
        _refresh_sources(refresh["sources"], refresh["catalog"])

    results = osk.compare(
        reference=suite["reference"],
        test=suite["test"],
        variables=suite["variables"],
        depths=tuple(suite.get("depths", (SURFACE,))),
        method=suite.get("regrid", "conservative_normed"),
        # A suite has to be able to say this, and until now could not: `aggregate` was
        # simply not forwarded, so every suite silently got the old implicit time mean.
        # With no default reduction a suite omitting it would fail on its own model
        # output, so it is both forwarded and required in the YAML — see
        # _require_reduced.
        aggregate=suite.get("aggregate"),
        select=suite.get("select"),
    )
    if not len(results):
        print("no comparisons produced; check the suite's sources and variables")
        return results

    plot = suite.get("plot", {})
    results.plot(
        title=plot.get("title", suite.get("name", "comparison")),
        save=out_dir / "figures" / plot.get("filename", "comparison.png"),
    )
    csv = results.write_metrics(out_dir, stem=suite.get("name", "metrics"))
    print(f"{len(results)} comparisons -> {out_dir}/figures/, {csv}")
    return results


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(
            "usage: python -m ocean_skill.workflows.run <suite.yaml>", file=sys.stderr
        )
        return 2
    run_suite(argv[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
