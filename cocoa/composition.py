"""Criteria for judging the dataset composition of a candidate cluster.

`extract_homogeneous_clusters` walks a dendrogram from the top down and asks a
criterion whether each candidate cluster contains an acceptable mix of neurons
from each dataset, stopping the split when it doesn't.

The built-in `is_good` compares the datasets against *each other*, which
implicitly expects all of them to contribute equally. That is rarely true: a
dataset covering both hemispheres contributes about twice as many neurons as one
covering a single hemisphere, so a perfectly matched cell type shows up as e.g.
8 vs 4 and gets rejected for being "lopsided".

`ProportionalBalance` instead compares each dataset against how much it is
*expected* to contribute, where the expectation is simply the share of the pool
that dataset makes up. Nothing has to be declared about the datasets - a dataset
covering two hemispheres already accounts for twice as much of the pool, so the
expectation falls out of the data.

`EqualBalance` is the `is_good` rule expressed in the same framework, with its
hard-coded numbers exposed as parameters. It is mostly here so the two rules can
be compared like for like.

Criteria are opt-in; the plain `is_good` function remains the default:

>>> from cocoa.composition import ProportionalBalance
>>> cl.extract_homogeneous_clusters(eval_func=ProportionalBalance())

"""

import functools

import numpy as np

from scipy.stats import entropy


__all__ = [
    "CompositionCriterion",
    "EqualBalance",
    "ProportionalBalance",
    "ClusterContext",
]

# Defaults for `ProportionalBalance`. `MAX_FOLD_RANGE`/`MAX_KL` and `MIN_Z` were
# picked by sweeping against `is_good` over the 20x20 grid of two-dataset counts:
# with an evenly sized pool these values reproduce `is_good` on 396 of 400 cells,
# disagreeing only on [1, 4], [2, 5] and their mirrors - the cases where its
# absolute "counts must differ by more than 3" gate waved a 4:1 (resp. 2.5:1)
# imbalance through.
MAX_FOLD_RANGE = 2.0
MAX_KL = 0.05
MIN_Z = 0.75
ALPHA_PRESENCE = 0.01

# Defaults for `EqualBalance` - the thresholds `is_good` hard-codes.
MAX_DIFF = 3
MAX_RATIO = 2


class ClusterContext:
    """What a composition criterion gets to see about a candidate cluster.

    One of these is built for every subtree the walk evaluates, so it is kept
    deliberately cheap: `n` and `expected` are derived on first access rather
    than up front, because the default criterion never asks for them.

    Attributes
    ----------
    counts :        (n_datasets, ) array
                    Neurons per dataset in this cluster. Unlike the vector handed
                    to the legacy `is_good`, absent datasets are included as
                    zeros - so a criterion can tell *which* dataset is missing.
    pool_counts :   (n_datasets, ) array
                    Neurons per dataset across the whole pool being clustered.
    proportions :   (n_datasets, ) array
                    `pool_counts` normalised to sum to 1, i.e. the share of the
                    pool each dataset makes up.
    expected :      (n_datasets, ) array
                    Neurons per dataset we would expect in a cluster this size,
                    i.e. `counts.sum() * proportions`.
    dataset_labels : (n_datasets, ) array
                    Dataset labels, sorted, aligned with all of the above.
    n :             int
                    Number of neurons in this cluster.
    pool_size :     int
                    Number of neurons in the whole pool.
    node :          int
                    ID of the dendrogram node this cluster corresponds to.
    dist :          float
                    Height of the hinge this cluster was formed at; 0 for a leaf.
    parent_counts : (n_datasets, ) array
                    `counts` for the node one level up.

    """

    __slots__ = (
        "counts",
        "pool_counts",
        "proportions",
        "dataset_labels",
        "pool_size",
        "node",
        "dist",
        "parent_counts",
        "_n",
        "_expected",
    )

    def __init__(
        self,
        counts,
        pool_counts,
        proportions,
        dataset_labels,
        pool_size,
        node,
        dist,
        parent_counts,
    ):
        self.counts = counts
        self.pool_counts = pool_counts
        self.proportions = proportions
        self.dataset_labels = dataset_labels
        self.pool_size = pool_size
        self.node = node
        self.dist = dist
        self.parent_counts = parent_counts
        self._n = None
        self._expected = None

    @property
    def n(self):
        if self._n is None:
            self._n = int(self.counts.sum())
        return self._n

    @property
    def expected(self):
        if self._expected is None:
            self._expected = self.n * self.proportions
        return self._expected

    def __repr__(self):
        return (
            f"ClusterContext(node={self.node}, counts={list(self.counts)}, "
            f"expected={[round(float(e), 2) for e in self.expected]})"
        )


class CompositionCriterion:
    """Base class for context-aware composition criteria.

    Subclasses implement ``__call__(counts, ctx)`` and are handed the full count
    vector - including zeros for absent datasets - plus a `ClusterContext`.

    The `needs_context` attribute is what tells `extract_homogeneous_clusters` to
    use this richer calling convention. Plain functions are called the legacy
    way, as ``eval_func(counts_without_zeros, n_datasets)``.

    """

    needs_context = True

    def __call__(self, counts, ctx):
        raise NotImplementedError


class EqualBalance(CompositionCriterion):
    """The rule `is_good` implements, as a criterion with the numbers exposed.

    A cluster is acceptable when every dataset is represented and the largest
    count is not both `max_diff` *more* than the smallest and `max_ratio` *times*
    the smallest. With the default thresholds this reproduces `is_good` exactly.

    Note what it does **not** look at: `ctx.proportions`. This criterion compares
    the datasets against each other, which amounts to expecting all of them to
    contribute equally. That is what makes it misbehave when they don't - with a
    pool of 200 FlyWire and 100 hemibrain neurons a perfectly matched cell type
    of ``[8, 4]`` is exactly on expectation, but this rejects it for being
    "lopsided". Use `ProportionalBalance` in that situation.

    It is included mainly so the two rules can be compared side by side, and so
    the thresholds `is_good` hard-codes can be tuned without writing a new
    function.

    Parameters
    ----------
    max_diff :      int | float
                    The largest and smallest count must differ by more than this
                    before the cluster can be rejected. Acts as a floor that
                    keeps small clusters from being judged on noise.
    max_ratio :     int | float
                    The largest count must be at least this many times the
                    smallest before the cluster can be rejected.
    require_all :   bool
                    If True (default), a cluster missing any dataset is rejected.

    Examples
    --------
    >>> from cocoa.composition import EqualBalance
    >>> # Same as the default `is_good`
    >>> cl.extract_homogeneous_clusters(eval_func=EqualBalance())
    >>> # Same rule, but stricter about lopsidedness
    >>> cl.extract_homogeneous_clusters(eval_func=EqualBalance(max_diff=2))

    """

    needs_context = True

    def __init__(self, max_diff=MAX_DIFF, max_ratio=MAX_RATIO, require_all=True):
        self.max_diff = max_diff
        self.max_ratio = max_ratio
        self.require_all = require_all

    def __repr__(self):
        return (
            f"EqualBalance(max_diff={self.max_diff}, max_ratio={self.max_ratio}, "
            f"require_all={self.require_all})"
        )

    def __call__(self, counts, ctx):
        counts = np.asarray(counts, dtype=np.float64)
        present = counts > 0

        # A missing dataset is what `is_good` catches via `len(v) < n_unique_ds`
        if self.require_all and not present.all():
            return False

        counts = counts[present]
        if not counts.size:
            return False

        mn, mx = counts.min(), counts.max()
        return not ((mx - mn) > self.max_diff and (mx / mn) >= self.max_ratio)


class ProportionalBalance(CompositionCriterion):
    """Judge a cluster against the composition of the pool it came from.

    Where `is_good` asks "are the datasets represented roughly equally?", this
    asks "is each dataset represented roughly as much as we would expect?" - the
    expectation being that dataset's share of the pool. With a pool of 200
    FlyWire and 100 hemibrain neurons a cluster of 12 is expected to be 8 and 4,
    so ``[8, 4]`` is spot on (and `is_good` rejects it) while ``[4, 8]`` is
    backwards (and both reject it).

    Two things have to be true before a cluster is rejected, mirroring the
    structure `is_good` already uses:

      1. The composition is off expectation by at least `max_fold_range`.
      2. It is off by more than counts of this size naturally wobble.

    Requiring both matters. On its own the first is far too twitchy for small
    clusters, where a count of 1 vs 3 is three-fold off but means nothing; on its
    own the second becomes hair-trigger for large clusters, where even a slight
    imbalance is many times the noise - and since a rejection *halts* the split,
    that would collapse the whole pool into a single cluster.

    Parameters
    ----------
    max_fold_range : float
                    How far off expectation a cluster may be, as a fold change.
                    The statistic is ``max(observed/expected) /
                    min(observed/expected)`` across the datasets present, so 2
                    means "no dataset may be twice as over-represented as
                    another is under-represented". With an evenly sized pool this
                    reduces exactly to the ``max/min`` that `is_good` uses.
    min_z :         float
                    How far off expectation a cluster must be, in units of
                    sampling noise, before the fold change is held against it.
                    Counts wobble by roughly the square root of their size, so
                    this is `is_good`'s fixed "differ by more than 3" gate made
                    to scale with cluster size.
    require_all :   bool
                    If True (default), a cluster missing any dataset is always
                    rejected - the same rule `is_good` applies. If False, a
                    missing dataset only counts against a cluster when its
                    absence is surprising, i.e. when the chance of seeing none of
                    it is below `alpha_presence`. Setting this to False makes the
                    walk descend further and produces smaller clusters.
    alpha_presence : float
                    Only used when ``require_all=False``; see above.
    statistic :     "fold_range" | "kl"
                    How to measure being off expectation. ``"fold_range"``
                    (default) compares the most over- and under-represented
                    datasets, and is easy to reason about. ``"kl"`` uses the
                    Kullback-Leibler divergence of the observed composition from
                    the expected one, which accounts for every dataset rather
                    than just the extremes. Note that plain entropy is *not* a
                    substitute: it treats an even split as ideal and so carries
                    the same assumption this class exists to remove.
    max_kl :        float
                    Threshold for ``statistic="kl"``, in nats.

    Examples
    --------
    >>> from cocoa.composition import ProportionalBalance
    >>> # Default: same presence rule as `is_good`, expectation-aware balance
    >>> cl.extract_homogeneous_clusters(eval_func=ProportionalBalance())
    >>> # Let rare datasets be absent from small clusters
    >>> cl.extract_homogeneous_clusters(
    ...     eval_func=ProportionalBalance(require_all=False)
    ... )

    """

    needs_context = True

    def __init__(
        self,
        max_fold_range=MAX_FOLD_RANGE,
        min_z=MIN_Z,
        require_all=True,
        alpha_presence=ALPHA_PRESENCE,
        statistic="fold_range",
        max_kl=MAX_KL,
    ):
        if statistic not in ("fold_range", "kl"):
            raise ValueError(
                f'`statistic` must be "fold_range" or "kl", got "{statistic}"'
            )
        self.max_fold_range = max_fold_range
        self.min_z = min_z
        self.require_all = require_all
        self.alpha_presence = alpha_presence
        self.statistic = statistic
        self.max_kl = max_kl

    def __repr__(self):
        if self.statistic == "fold_range":
            stat = f"max_fold_range={self.max_fold_range}"
        else:
            stat = f'statistic="kl", max_kl={self.max_kl}'
        return (
            f"ProportionalBalance({stat}, min_z={self.min_z}, "
            f"require_all={self.require_all})"
        )

    def __call__(self, counts, ctx):
        counts = np.asarray(counts, dtype=np.float64)
        n = counts.sum()

        # Nothing to judge
        if n <= 0:
            return False

        present = counts > 0

        # Is anything missing, and if so does that matter?
        if not present.all():
            if self.require_all:
                return False
            # Only hold an absence against a dataset if we would have expected to
            # see it at all: in a small cluster a small dataset is easily missed
            p_zero = (1.0 - ctx.proportions[~present]) ** n
            if (p_zero < self.alpha_presence).any():
                return False

        expected = ctx.expected

        # How far off expectation are we? Absent datasets are excluded - they are
        # the presence gate's business, and including them would make
        # `fold_range` infinite as soon as anything is missing.
        if self.statistic == "fold_range":
            ratio = counts[present] / expected[present]
            effect = ratio.max() / ratio.min()
            off_expectation = effect >= self.max_fold_range
        else:
            effect = entropy(counts[present], qk=expected[present])
            off_expectation = effect >= self.max_kl

        # ... and is that more than counts this size wobble by anyway?
        usable = expected > 0
        if usable.any():
            z = np.max(
                np.abs(counts[usable] - expected[usable]) / np.sqrt(expected[usable])
            )
        else:
            z = 0.0
        above_noise = z > self.min_z

        return not (off_expectation and above_noise)


def _wants_context(eval_func):
    """Check whether `eval_func` expects the context-aware calling convention.

    `functools.partial` does not forward attributes, so a wrapped criterion would
    otherwise fall back to the legacy convention and be handed a zero-stripped
    vector and an integer - wrong answers rather than an error.
    """
    while isinstance(eval_func, functools.partial):
        eval_func = eval_func.func
    return getattr(eval_func, "needs_context", False)
