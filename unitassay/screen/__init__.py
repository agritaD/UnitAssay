"""unitassay.screen — per-frame isolation quality for sorted units.

Authors: Agrita Dubey, Qiao (Betty) Xu
Developed with Claude (Anthropic).

For each cluster, computes the L-ratio against a shape-dissimilar REFERENCE
channel and against each neighbouring cluster, separately in each drift frame,
and writes one SortData.h5 per cluster. `unitassay.curate` reads those files.

Also carries spike extraction and format conversion.

Previously developed under the internal name `mind_snag`; the module was renamed
on publication. The pipeline, config keys and file formats are unchanged.

Authors: Agrita Dubey, Qiao (Betty) Xu
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""

__version__ = "2.0.0"

from unitassay.screen.config import IsolateConfig
from unitassay.screen.pipeline import Pipeline

__all__ = ["IsolateConfig", "Pipeline", "__version__"]
