"""The between-cluster verdict gives the L-ratio precedence on a conflict.

The within-cluster test resolves dip/L-ratio conflicts in favour of the
L-ratio; the between-cluster test used to return None instead, which sent 2,518 real clusters to rule 5d as "evidence
incomplete". None must now mean only "the test could not run".
"""
import math

import pytest

from unitassay.curate.constants import DIP_PVAL_THRESH, L_RATIO_THRESH
from unitassay.curate.curation import between_separated

LO_DIP, HI_DIP = DIP_PVAL_THRESH / 10, min(0.99, DIP_PVAL_THRESH * 10)
LO_LR, HI_LR = L_RATIO_THRESH / 10, L_RATIO_THRESH * 10


def test_both_agree_separated():
    assert between_separated(LO_DIP, LO_LR) is True


def test_both_agree_not_separated():
    assert between_separated(HI_DIP, HI_LR) is False


def test_conflict_dip_separated_lratio_not_follows_lratio():
    assert between_separated(LO_DIP, HI_LR) is False


def test_conflict_lratio_separated_dip_not_follows_lratio():
    assert between_separated(HI_DIP, LO_LR) is True


@pytest.mark.parametrize('dp,lr', [(None, LO_LR), (LO_DIP, None),
                                   (math.nan, LO_LR), (LO_DIP, math.inf)])
def test_could_not_run_is_none_never_a_verdict(dp, lr):
    assert between_separated(dp, lr) is None


def test_threshold_is_strict_less_than():
    assert between_separated(LO_DIP, L_RATIO_THRESH) is False
