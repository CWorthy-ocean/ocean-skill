"""ocean-skill: modular model–data validation and analysis for ocean models.

Read any two things through intake catalogs, label which is ``reference`` and which is
``test``, then align them in space/time, compute skill metrics, and plot — with a
static⇄interactive switch. Model-agnostic; ROMS specifics live behind
``ocean_skill.roms``.

The public API is intentionally small:

    import ocean_skill as osk
    osk.catalogs                 # discovered catalogs / sources
    osk.find(variable=...)       # search sources across catalogs
    osk.find(variable=...).map() # ... and map where the matches are
    osk.find_catalogs(name=...)  # search catalogs themselves (existence checks)
    osk.map_locations()          # map every discovered dataset (metadata only)
    osk.describe("glodap")       # metadata for one source, or one whole catalog --
                                 # including which declared variables the vocabulary
                                 # recognizes (and as what), and which it doesn't
    osk.describe("glodap").catalog_path  # ... the catalog file it was found in
    osk.match_report("glodap")   # ... that vocabulary section alone, any time
    osk.coord_report("glodap")   # ... and which of T/X/Y/Z it recognizes, as what
    osk.overlap("his", "glodap") # do these two sources even share space/time? (read-free)
    osk.read("glodap")           # -> standardized xr.Dataset / pandas.DataFrame
    osk.detide(osk.read("tide_gauge"))  # PL33 low-pass -- subtidal by default
    osk.compare(reference=..., test=..., variables=[...],
                aggregate={"time": "mean"}).plot()   # no default reduction: say so
    osk.compare(..., over="time").plot()             # score against time, cell by cell
    osk.compare(..., times={"resample": "1MS", "reduce": "mean"})  # one comparison
                                                     # per month, plots or plays as one
    osk.compare(..., aggregate={"time": [{"groupby": "month", ...}, "var"]})
                                                     # a list of steps on one axis, in
                                                     # order: seasonal-cycle variance
    osk.compare(..., aggregate=[{"time": "mean"}, {"time": [...]}])  # a list of specs:
                                                     # one member per statistic
    osk.field(source, variable, select=...)          # one source, no reference
    osk.field(...).extremum("max").plot()            # where the max is, and how it
                                                     # evolves around that snapshot
    osk.TS({"ROMS": roms, "WOA23": woa}).plot()      # T against S per source;
                                                     # osk.XY: any two variables
    osk.summary([set_a, set_b, one_comparison])      # comparisons you already have,
                                                     # pooled onto Taylor + target
    osk.plot([eq, band_a, band_b])                   # comparisons (or fields) built
                                                     # separately, one figure; a
                                                     # {label: item} dict names rows
    osk.map_metrics(mooring_set)                     # per-station metrics, interpolated
                                                     # onto a map, one panel per metric
    comparison.map_locations()                       # where a plotted selection sits

    osk.cache.info()             # processed intermediates are cached; where, how big
    osk.outputs.info()           # where figures + metrics get written

    osk.catalog.search_paths()              # where catalogs are discovered, in order
    osk.catalog.add_search_path("/shared")  # register a shared/team catalog dir in code
"""

import importlib as _importlib
import types as _types

from ocean_skill import cache, catalog, outputs, qc
from ocean_skill import mld as _mld  # noqa: F401  (registers CALCULATORS["mld"])
from ocean_skill import internal_tides as _internal_tides  # noqa: F401  (registers "baroclinic_pressure_flux")
from ocean_skill import tides as _tides  # noqa: F401  (registers "tidal_amplitude"/"tidal_phase")
from ocean_skill.catalog import (
    Overlap,
    catalogs,
    coord_report,
    describe,
    find,
    find_catalogs,
    match_report,
    overlap,
)
from ocean_skill.combine import plot as _plot_items
from ocean_skill.comparison import Comparison, ComparisonSet, compare, summary
from ocean_skill.detide import detide
from ocean_skill.extrema import Extrema, Extremum
from ocean_skill.field import Cross, Field, FieldSet, field
from ocean_skill.pick import pick_path
from ocean_skill.plot.map_locations import map_locations
from ocean_skill.plot.map_metrics import map_metrics
from ocean_skill.sources import read
from ocean_skill.xy import TS, XY

__version__ = "0.0.1"


# ``osk.plot(...)`` draws several comparisons or fields on one figure (see
# ocean_skill.combine), but ``ocean_skill.plot`` is also the plotting subpackage, and
# a plain ``from ocean_skill.combine import plot`` here would replace it as a
# *package attribute*: ``import ocean_skill.plot.registry as r``, and every
# ``monkeypatch.setattr("ocean_skill.plot.registry.render", ...)``, resolve through
# ``getattr(ocean_skill, "plot")`` and would land on a function with no submodules.
# So the subpackage itself is made callable instead -- one object that is both the
# module (``osk.plot.registry`` keeps working) and the function.
class _CallablePlotPackage(_types.ModuleType):
    def __call__(self, items, *, renderer="matplotlib", **plot_kwargs):
        return _plot_items(items, renderer=renderer, **plot_kwargs)

    __call__.__doc__ = _plot_items.__doc__


_importlib.import_module("ocean_skill.plot").__class__ = _CallablePlotPackage

__all__ = [
    "TS",
    "XY",
    "Comparison",
    "ComparisonSet",
    "Cross",
    "Extrema",
    "Extremum",
    "Field",
    "FieldSet",
    "Overlap",
    "__version__",
    "cache",
    "catalog",
    "catalogs",
    "compare",
    "coord_report",
    "describe",
    "detide",
    "field",
    "find",
    "find_catalogs",
    "map_locations",
    "map_metrics",
    "match_report",
    "outputs",
    "overlap",
    "pick_path",
    "plot",
    "qc",
    "read",
    "summary",
]
