"""unitassay.screen — per-frame isolation quality for sorted units.

Authors: Agrita Dubey, Qiao (Betty) Xu

For each cluster, computes the L-ratio against a shape-dissimilar REFERENCE
channel and against each neighbouring cluster, separately in each drift frame,
and writes one SortData.h5 per cluster. `unitassay.curate` reads those files.

Also carries spike extraction and format conversion.

Authors: Agrita Dubey, Qiao (Betty) Xu
Part of unitassay. Licence: MIT.
"""

__version__ = "2.0.0"

from unitassay.screen.config import IsolateConfig
from unitassay.screen.pipeline import Pipeline

__all__ = ["IsolateConfig", "Pipeline", "__version__"]
