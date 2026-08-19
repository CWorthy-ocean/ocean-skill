"""ocean-skill: modular model–data validation and analysis for ocean models.

Read any two things through intake catalogs, label which is ``reference`` and which is
``test``, then align them in space/time, compute skill metrics, and plot — with a
static⇄interactive switch. Model-agnostic; ROMS specifics live behind
``ocean_skill.roms``.

The public API is intentionally small:

    import ocean_skill as osk
    osk.catalogs                 # discovered catalogs / sources
    osk.find(variable=...)       # search sources across catalogs
    osk.describe("glodap")       # metadata for one source, or one whole catalog
    osk.read("glodap")           # -> standardized xr.Dataset / pandas.DataFrame
    osk.compare(reference=..., test=..., variables=[...],
                aggregate={"time": "mean"}).plot()   # no default reduction: say so
    osk.compare(..., over="time").plot()             # score against time, cell by cell
    osk.field(source, variable, select=...)          # one source, no reference
    osk.summary([set_a, set_b, one_comparison])      # comparisons you already have,
                                                     # pooled onto Taylor + target

    osk.cache.info()             # processed intermediates are cached; where, how big
    osk.outputs.info()           # where figures + metrics get written
"""

from ocean_skill import cache, outputs
from ocean_skill import mld as _mld  # noqa: F401  (registers CALCULATORS["mld"])
from ocean_skill.catalog import catalogs, describe, find
from ocean_skill.comparison import Comparison, ComparisonSet, compare, summary
from ocean_skill.field import Field, field
from ocean_skill.sources import read

__version__ = "0.0.1"

__all__ = [
    "Comparison",
    "ComparisonSet",
    "Field",
    "__version__",
    "cache",
    "catalogs",
    "compare",
    "describe",
    "field",
    "find",
    "outputs",
    "read",
    "summary",
]
