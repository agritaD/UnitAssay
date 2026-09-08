"""unitassay.link — match the same unit across recordings.

Authors: Agrita Dubey, Qiao (Betty) Xu
Developed with Claude (Anthropic).

NOT YET IMPLEMENTED. This layer is declared rather than omitted so the package
states its intended scope: `isolate` measures, `curate` judges, `link` follows a
unit across sessions.

Its starting point is `unitassay.isolate.stitching`, which already implements
cross-recording neuron stitching against the sorter's own output. What `link`
adds is the same treatment applied to CURATED units, so an identity carried
across recordings is one the curation layer has already vouched for.

The module imports cleanly and only the call raises -- a module that raises at
IMPORT breaks documentation builds and dependency checks for everyone, including
people who never touch this layer.
"""

__all__ = ['link_sessions']


def link_sessions(*args, **kwargs):
    """Placeholder. Raises, with a pointer to the code this will build on."""
    raise NotImplementedError(
        'unitassay.link is not implemented yet. The cross-recording matching '
        'this layer will build on is in unitassay.isolate.stitching.')
