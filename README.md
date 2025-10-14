![cocoa](docs/_static/cocoa.svg)

# cocoa

`cocoa` is a Python library for **co**mparative **co**nnectomics **a**nalyses.

It implements various dataset-agnostic as well as dataset-specific methods
for matching, connectivity, co-clustering and cell typing.

Currently implemented are:

1. [FlyWire](https://flywire.ai)
2. [hemibrain](https://neuprint.janelia.org/?dataset=hemibrain%3Av1.2.1&qt=findneurons)
3. [MANC](https://neuprint.janelia.org/?dataset=manc%3Av1.2.3&qt=findneurons)
4. [male CNS](https://neuprint.janelia.org/?dataset=male-cns%3Av0.9&qt=findneurons)

On the TO-DO list:
- female adult nerve cord (FANC)
- brain and nerve cord (BANC)

Feel free to open an Issue or a PR if you want a specific dataset added.

## Install

```bash
pip3 install git+https://github.com/flyconnectome/cocoa.git -U
```

### Other requirements

All dependencies should be installed automatically. However, to use the
pre-define datasets you will need to set a couple environment variables and
secrets:
1. To use the neuPrint datasets (hemibrain, MANC and maleCNS) you need to set your
   API token as `NEUPRINT_APPLICATION_CREDENTIALS`
   (see [neuprint-python](https://github.com/connectome-neuprint/neuprint-python))
2. To use the CAVE/chunkedgraph datasets (FlyWire, FANC) you need to have your
   CAVE token set (see [fafbseg](https://fafbseg-py.readthedocs.io/en/latest/source/tutorials/flywire_setup.html))
3. For internal use only: if you want to use the live annotations from flytable
   make sure to set the `SEATABLE_SERVER` and `SEATABLE_TOKEN` environment variables
   (see [sea-serpent](https://github.com/schlegelp/sea-serpent))

## Concepts

The main concept in `cocoa` is that of a `DataSet`. A `DataSet` represents
a collection of neurons from a specific source (e.g. FlyWire or hemibrain),
and provides methods to fetch annotations and connectivity"

While you can use `cocoa` to run clusterings on just a single dataset,
its real power lies in co-clustering neurons from multiple datasets. To do
this, it aut-magically computes mappings between neurons from different
datasets based on available labels. These labels are then used to
generate a joint connectivity vector from which we can compute pairwise
distances.

## Examples

```Python
>>> import cocoa as cc
>>> # Define the sets of neurons to co-cluster
>>> hb = cc.Hemibrain(label='hemibrain',
...                   ).add_neurons(['SLP001', 'SLP003'])
>>> fwl = cc.FlyWire(label='FlyWire_left',
...                  materialization=783,
...                  ).add_neurons(['SLP001', 'SLP003'], sides='left')
>>> fwr = cc.FlyWire(label='FlyWire_right',
...                  materialization=783,
...                  ).add_neurons(['SLP001', 'SLP003'], sides='right')
>>> # Combine into a clustering and co-cluster
>>> cl = cc.Clustering([hb, fwl, fwr]).compile()
>>> # The clustering `cl` contains the results of the clustering.
>>> # The joint connectivity vector:
>>> cl.vect_
                   downstream                          ... upstream
                      LHAV1b1 LHPV4g1 LHAV5e1 LHAV1b3  ...    CL018 CL077 SLP202 LC9
294437347                   0       0       1       0  ...        0     0      0   0
543692985                   0       0       0       4  ...        0     6      0   1
720575940617091414          0       0       1       0  ...        0     0      0   0
720575940623050334          0       0       0       2  ...        1     1      0   0
720575940627960442          0       0       1       0  ...        0     0      1   0
720575940628895750          1       4       0       3  ...        0     5      0   0
>>> # The pairwise (cosine) distances:
>>> cl.dists_
                    SLP001_hemibrain  ...  SLP003_FlyWire_right
294437347                   0.000000  ...              0.990616
543692985                   0.988929  ...              0.092726
720575940617091414          0.141363  ...              0.994823
720575940623050334          0.993146  ...              0.046200
720575940627960442          0.218134  ...              0.992618
720575940628895750          0.990616  ...              0.000000
>>> # It also provides some useful methods to work with the data
>>> table = cl.to_table(clusters=cl.extract_homogeneous_clusters())
>>> table
                   id   label        dataset  cn_frac_used  dend_ix  cluster
0           543692985  SLP003      hemibrain      0.503151        0        0
1  720575940623050334  SLP003   FlyWire_left      0.541004        1        0
2  720575940628895750  SLP003  FlyWire_right      0.545074        2        0
3           294437347  SLP001      hemibrain      0.308048        3        1
4  720575940617091414  SLP001   FlyWire_left      0.375770        4        1
5  720575940627960442  SLP001  FlyWire_right      0.328080        5        1
>>> # See also `cl.plot_clustermap` for a quick visualization
```

Alternatively, you can also use the `generate_clustering` helper function.
That may be enough in cases where you don't need fine-grained control.

```Python
>>> cl = cc.generate_clustering(
...            fw=['SLP001', 'SLP002'],
...            hb=['SLP001', 'SLP002']
...         ).compile()
```

## Documentation

`cocoa` does not yet have a dedicated documentation but we provide a number of
[examples/](examples/) that show how to use the library for various tasks:

- `0_flywire_hemibrain_FC1-3.ipynb`: demonstrates co-clustering for a small group of neurons, including visualization of the results
- `1_malecns_flywire_mapping.ipynb`: show how to use `cocoa` to generate mappings between neurons from different datasets
- `2_malecns_flywire_optic_lobes.ipynb`: demonstrates a large-scale (~160k neurons) co-clustering between two datasets

In addition,all functions/classes have extensive docstrings:

```python
>>> help(cc.Clustering.compile)
cc.Clustering.compile(
    self,
    join='outer',
    metric='cosine',
    mapper=<class 'cocoa.mappers.GraphMapper'>,
    force_recompile=False,
    exclude_labels=None,
    include_labels=None,
    ignore_unlabeled=True,
    cn_frac_threshold=None,
    augment=None,
    n_batches='auto',
    verbose=True,
)
Docstring:
Compile combined connectivity vector and calculate distance matrix.

Parameters
----------
join :      "inner" | "outer" | "existing"
            How to combine the dataset connectivity vectors:
              - "existing" (default) will check if a label exists in
                theory and use it even if it's not present in the
                connectivity vectors of all datasets
              - "inner" will get the intersection of all labels across
                the connectivity vectors
              - "outer" will use all available labels
            Note: if you are using a GraphMapper, you should use "outer"
            as the mapper will already have filtered out non-matching
            labels.
metric :    "cosine" | "Euclidean"
            Metric to use for distance calculations.
mapper :    cocoa.Mapper | dict
            The mapper used to match neuron labels across datasets.
            Examples are `cocoa.GraphMapper` and `cocoa.SimpleMapper`.
            See the mapper's documentation for more information.
            Alternatively, you can also provide a dictionary that maps
            IDs to labels.
exclude_labels : str | list of str, optional
            If provided will exclude given labels from the observation
            vector. This uses regex!
[...]
```