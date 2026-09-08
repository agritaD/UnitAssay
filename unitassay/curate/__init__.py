"""unitassay.curate — decide, then name what is wrong.

Author: Agrita Dubey
Developed with Claude (Anthropic).

    >>> from unitassay import curate
    >>> rows = curate.curate_session(ks4_dir, mindsnag_subdir='isolate_out')

Each returned row carries three layers, which answer different questions and
must not be collapsed:

    decision / rule / tag        WHICH TEST decided, and what it said
    category / sub_type          the MECHANISM, named for the test
    outcome / findings           the ERROR CLASS, named for the cluster

`findings` is the parallel view: every rule's test evaluated independently, so a
row reports EVERY finding it carries rather than the first rule to fire. 36% of
clusters carry more than one.

Author: Agrita Dubey
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""
from .curation import curate_session, cluster_inputs      # noqa: F401
from .decision_flow import curate, rule_label, rule_id     # noqa: F401
from .findings import findings, outcome                    # noqa: F401
from .taxonomy import classify, classify_all, curate_and_classify   # noqa: F401

__all__ = ['curate_session', 'cluster_inputs', 'curate', 'rule_label',
           'rule_id', 'findings', 'outcome', 'classify', 'classify_all',
           'curate_and_classify']
