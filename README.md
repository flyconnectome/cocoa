![cocoa](docs/_static/cocoa.svg)

# cocoa

`cocoa` is a Python library for **co**mparative **co**nnectomics **a**nalyses.

It implements various dataset-agnostic as well as dataset-specific methods
for matching, co-clustering and cell typing.

Currently implemented are:

1. FlyWire
2. hemibrain
3. maleCNS (not public yet)

On the TODO list:
- female adult nerve cord (FANC)
- male adult never cord (MANC)

Feel free to open an Issue or a PR if you want another dataset added.

## Install

```bash
pip3 install git+https://github.com/flyconnectome/cocoa.git
```

### Other requirements

All dependencies should be installed automatically. However, to use the
pre-define datasets you will need to set a couple environment variables and
secrets:
1. If you want to use the live annotations from flytable make sure to set the
   `SEATABLE_SERVER` and `SEATABLE_TOKEN` environment variables (see
   [sea-serpent](https://github.com/schlegelp/sea-serpent))
2. To use neuPrint datasets (hemibrain, MANC and maleCNS) you need to set your
   API token as `NEUPRINT_APPLICATION_CREDENTIALS`
   (see [neuprint-python](https://github.com/connectome-neuprint/neuprint-python))
3. To use the CAVE/chunkedgraph datasets (FlyWire, FANC) you need to have your
   CAVE token set (see [fafbseg](https://fafbseg-py.readthedocs.io/en/latest/source/tutorials/flywire_setup.html))

## Examples

```Python
>>> import cocoa as cc
>>> # Define the sets of neurons to co-cluster
>>> hb = cc.Hemibrain(label='hemibrain',
...                   live_annot=True  # this make sure we use data from flytable
...                   ).add_neurons(['SLP001', 'SLP003'], sides='right')
>>> fwl = cc.FlyWire(label='FlyWire_left',
...                  materialization=783,
...                  live_annot=True  # this make sure we use data from flytable
...                  ).add_neurons(['SLP001', 'SLP003'], sides='left')
>>> fwr = cc.FlyWire(label='FlyWire_right',
...                  materialization=783,
...                  live_annot=True  # this make sure we use data from flytable
...                  ).add_neurons(['SLP001', 'SLP003'], sides='right')
>>> # Combine into a clustering and co-cluster
>>> cl = cc.Clustering([hb, fwl, fwr]).compile()
>>> # The clustering `cl` contains the results of the clustering...
>>> cl.dists_
                    SLP001_hemibrain  ...  SLP003_FlyWire_right
294437347                   0.000000  ...              0.990616
543692985                   0.988929  ...              0.092726
720575940617091414          0.141363  ...              0.994823
720575940623050334          0.993146  ...              0.046200
720575940627960442          0.218134  ...              0.992618
720575940628895750          0.990616  ...              0.000000
>>> # ... and provides some useful methods to work with the data
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