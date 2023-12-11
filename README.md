![cocoa](docs/_static/cocoa.svg)

A Python library for comparative connectomics analyses.

`cocoa` implements various dataset-agnostic as well as dataset-specific methods
for matching, co-clustering and cell typing.

Currently implemented are:

1. FlyWire
2. hemibrain
3. maleCNS

On the TODO list: FANC, MANC

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

