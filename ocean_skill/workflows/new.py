"""``ocean-skill-suite new mid-run-check``: fill the template's setup block from a run.

Starting a mid-run check by hand means copying ``templates/mid-run-check.yaml`` and
editing a few paths. This finds them in a C-Star run folder (``<run>/tasks/<task>/
{input/input_datasets,output,work}``) and rewrites the block between the template's
``# >>> setup`` and ``# <<< setup`` markers. The rest of the file (pages, comments) is
kept byte for byte: the block is spliced in as text, since loading and dumping the YAML
would drop the comments. ``final-report`` and ``model-comparison`` are copied and
edited by hand; guesses are written as ``# TODO check`` comments.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import warnings
from pathlib import Path

import yaml

TEMPLATE = Path(__file__).parent / "templates" / "mid-run-check.yaml"

#: The shared observation catalogs on Anvil; ``--shared-catalogs`` points elsewhere.
SHARED_CATALOGS = "/anvil/projects/x-ees250129/cstar-forge-data/source-data/catalogs"

#: Restart references are split into one-level chunks of this many rows (the largest
#: divisor of ``eta_rho`` in range) so a page matched to a depth reads whole columns.
SUBCHUNK_RANGE = (32, 128)

_STAMP = re.compile(r"output_rst\.(\d{4})(\d{2})(\d{2})")


class SuiteNewError(Exception):
    """Something the user has to fix before a suite can be written."""


def _abs(path: str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(path)))


def _q(text: object) -> str:
    """``text`` as a YAML scalar: plain when that reads back the same, else quoted."""
    s = str(text)
    plain = (
        re.fullmatch(r"[A-Za-z0-9_./~][A-Za-z0-9_./~+=*-]*", s)
        and yaml.safe_load(s) == s
    )
    return s if plain else json.dumps(s)


def find_restarts(roots: list[Path]) -> list[str]:
    """Return one restart glob per run folder.

    A folder of symlinks to every restart (``gathered_output/``, ``tasks/rst_files/``)
    is preferred where there is one: it de-duplicates restarts two tasks both wrote.
    """
    globs = []
    for root in roots:
        gathered = (root / "gathered_output", root / "tasks" / "rst_files")
        folders = [f for f in gathered if any(f.glob("output_rst.*.nc"))]
        globs.append(
            str(folders[0] / "output_rst.*.nc")
            if folders
            else str(root / "tasks" / "*" / "output" / "output_rst.*.nc")
        )
    if not any(glob.glob(g) for g in globs):
        raise SuiteNewError(
            f"no output_rst.*.nc under {', '.join(map(str, roots))}; "
            "pass --files GLOB to name them"
        )
    return globs


def find_grid(roots: list[Path]) -> Path:
    """Return the one ``*_grid.nc`` in the tasks' inputs, or raise naming them."""
    pattern = "tasks/*/input/input_datasets/*_grid.nc"
    grids = [Path(g) for r in roots for g in sorted(glob.glob(str(r / pattern)))]
    names = sorted({g.name for g in grids})
    if len(names) != 1:
        found = (
            f"{len(names)} different grids ({', '.join(names)})" if names else "no grid"
        )
        raise SuiteNewError(f"{found} in {pattern}; pass --grid FILE")
    return grids[0]


def pick_subchunk(grid: Path) -> tuple[dict[str, int], int]:
    """Return the restart ``subchunk:`` for ``grid`` and its ``eta_rho`` length.

    One level per chunk, and the largest divisor of ``eta_rho`` in
    :data:`SUBCHUNK_RANGE` rows so the chunks tile the grid. With none (674 = 2 x 337)
    the rows stay whole.
    """
    import xarray as xr

    with xr.open_dataset(grid) as ds:
        n = int(ds.sizes["eta_rho"])
    rows = max(
        (d for d in range(SUBCHUNK_RANGE[0], SUBCHUNK_RANGE[1] + 1) if n % d == 0),
        default=0,
    )
    return {"s_rho": 1, "s_w": 1, **({"eta_rho": rows} if rows else {})}, n


def render(
    args: argparse.Namespace, roots: list[Path], out: Path
) -> tuple[str, list[str], list[str]]:
    """Return the suite text, its ``# TODO check`` items and its warnings."""
    from ocean_skill.workflows import cover

    globs = [str(_abs(g)) for g in args.files] if args.files else find_restarts(roots)
    grid = _abs(args.grid) if args.grid else find_grid(roots)
    if not grid.is_file():
        raise SuiteNewError(f"grid {grid} not found")
    sub, n_eta = pick_subchunk(grid)
    stamps = sorted(
        m.groups() for g in globs for p in glob.glob(g) if (m := _STAMP.search(p))
    )
    tasks = sorted(t for r in roots for t in (r / "tasks").glob("*"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # several workplans: the user names one
        found = {
            "workplan": next(filter(None, map(cover.find_workplan, tasks)), None),
            "blueprint": next(filter(None, map(cover.newest_blueprint, tasks)), None),
        }

    todo, warn = [], []

    def check(text: str) -> str:
        todo.append(text)
        return f"# TODO check: {text}"

    name = re.sub(r"[^A-Za-z0-9_]+", "_", args.name).strip("_")
    output_dir = str(_abs(args.output_dir))
    catalogs = (
        str(_abs(args.catalog_dir)) if args.catalog_dir else f"{output_dir}/catalogs"
    )
    catalog = f"{catalogs}/{name.replace('_', '-')}.yaml"
    lines = [
        "name: mid-run-check",
        f"output_dir: {_q(output_dir)}",
        "pdf_images: jpeg",
        f"catalog_search_paths: [{_q(args.shared_catalogs)}, {_q(catalogs)}]",
        f"cache_dir: {_q(output_dir)}",
        "",
        "run_info:",
    ]
    for key, path in found.items():
        lines.append(
            f"  {key}: {_q(path)}"
            if path
            else f"  {check(f'run_info.{key} not found under {roots[0]}')}"
        )
    lines += [
        "",
        "refresh:",
        f"  catalog: {_q(catalog)}",
        "  sources:",
        f"    - name: {_q(name)}",
        f"      files: {json.dumps(globs) if len(globs) > 1 else _q(globs[0])}",
        f"      grid: {_q(grid)}",
        f"      ref: {_q(f'{output_dir}/refs/{name}.parquet')}",
        "      keep: latest-per-file  # restart files: keep each file's main time step",
    ]
    if "eta_rho" in sub:
        lines.append(
            f"      # {sub['eta_rho']}-row, 1-level chunks: depth pages read less."
        )
    else:
        why = f"eta_rho={n_eta} has no divisor in {SUBCHUNK_RANGE}, so rows stay whole"
        lines.append(f"      # {why}: depth pages over many restarts need more memory.")
        warn.append(f"{why}; depth pages over many restarts use more memory")
    first = "-".join(stamps[0]) if stamps else None
    period = f'  period: {{min: "{first or "2000-01-01"}"}}'
    lines += [
        f"      subchunk: {json.dumps(sub)}",
        "",
        "defaults:",
        f"  test: {_q(name)}",
        period
        if first
        else f"{period}  {check('restart dates not read from file names')}",
    ]

    text = TEMPLATE.read_text()
    opened = re.search(r"^# >>> setup.*\n", text, re.MULTILINE)
    closed = re.search(r"^# <<< setup.*\n", text, re.MULTILINE)
    if opened is None or closed is None:
        raise SuiteNewError(f"{TEMPLATE} has no setup markers")
    head = text[: opened.start()].replace(
        "ocean-skill-run mid-run-check.yaml", f"ocean-skill-run {out}"
    )
    note = (
        "# Generated by `ocean-skill-suite new mid-run-check`; the description below "
        "is the template's.\n"
    )
    block = opened.group(0) + "\n".join(lines) + "\n" + closed.group(0)
    return note + head + block + text[closed.end() :], todo, warn


def validate(path: Path) -> list[str]:
    """Return the problems with the written suite: it must load, its paths exist."""
    from ocean_skill.config import SuiteConfig

    try:
        suite = SuiteConfig.model_validate(yaml.safe_load(path.read_text()))
    except Exception as exc:
        return [f"the suite does not load: {exc}"]
    problems = []
    for src in suite.refresh.sources:
        files = [src.files] if isinstance(src.files, str) else src.files
        problems += [f"no files match {g}" for g in files if not glob.glob(g)]
        if src.grid and not Path(src.grid).is_file():
            problems.append(f"grid {src.grid} not found")
    for key in ("workplan", "blueprint"):
        value = getattr(suite.run_info, key, None)
        if value and not Path(value).is_file():
            problems.append(f"run_info.{key} {value} not found")
    return problems


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ocean-skill-suite")
    new = parser.add_subparsers(dest="command", required=True).add_parser(
        "new", help="write a mid-run-check suite from a C-Star run folder"
    )
    add = new.add_argument
    add("template", help="the suite to start from: mid-run-check")
    add(
        "--run",
        action="append",
        required=True,
        metavar="DIR",
        help="C-Star run folder (holds tasks/); repeat for segments",
    )
    add("--name", required=True, help="the model source's name")
    add("--output-dir", required=True, help="reports and cache go here")
    add("--files", nargs="+", metavar="GLOB", help="the restarts, if not found")
    add("--grid", metavar="FILE", help="the grid, if not found")
    add("--catalog-dir", help="this suite's catalog (default <output-dir>/catalogs)")
    add("--shared-catalogs", default=SHARED_CATALOGS)
    add("-o", "--out", default="mid-run-check.yaml", help="file to write")
    add("--force", action="store_true", help="overwrite an existing file")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run ``ocean-skill-suite new mid-run-check --run DIR --name N --output-dir DIR``.

    Returns 0 when the suite was written and checks out, 1 when it was written but the
    check found problems, 2 when nothing could be written.
    """
    try:
        args = _build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2
    try:
        if args.template != "mid-run-check":
            raise SuiteNewError(
                f"template {args.template!r} is not supported yet; copy "
                f"ocean_skill/workflows/templates/{args.template}.yaml and edit its "
                "setup block (between '# >>> setup' and '# <<< setup') by hand"
            )
        out, roots = _abs(args.out), [_abs(r) for r in args.run]
        if out.exists() and not args.force:
            raise SuiteNewError(f"{out} exists; pass --force to overwrite it")
        if missing := [r for r in roots if not r.is_dir()]:
            raise SuiteNewError(f"{missing[0]} is not a directory")
        text, todo, warn = render(args, roots, out)
    except SuiteNewError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)

    problems = validate(out)
    print(f"Wrote {out}" + (f" ({len(todo)} marked '# TODO check')." if todo else "."))
    for line in warn:
        print(f"warning: {line}", file=sys.stderr)
    for line in problems:
        print(f"problem: {line}", file=sys.stderr)
    print(f"Next:\n  ocean-skill-run {out} --list")
    print(f"  ocean-skill-run {out} --smoke 2 --self-check --strict")
    print(f"  ocean-skill-run {out}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
