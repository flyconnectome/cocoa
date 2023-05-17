import clio

import seaserpent as ss
import neuprint as neu
import numpy as np

from functools import lru_cache
from fafbseg import flywire


@lru_cache
def _get_table(which="info"):
    """Initialize connection to annotation table."""
    if which == "info":
        return ss.Table("info", base="main", read_only=True)
    elif which in ("optic", "optic_lobes"):
        return ss.Table("optic", base="optic_lobes", read_only=True)
    else:
        raise ValueError(f"Unknown table {which}")


@lru_cache
def _get_hemibrain_meta():
    return ss.Table("hb_info", "hemibrain")[
        ["bodyId", "side", "type", "morphology_type"]
    ]


@lru_cache
def _get_mcns_meta():
    client = _get_clio_client("CNS")
    return clio.fetch_annotations(None, client=client)


@lru_cache
def _get_neuprint_client():
    return neu.Client("https://neuprint.janelia.org", dataset="hemibrain:v1.2.1")

@lru_cache
def _get_neuprint_mcns_client():
    return neu.Client("https://neuprint-cns.janelia.org", dataset="cns")


@lru_cache
def _get_clio_client(dataset):
    return clio.Client(dataset=dataset)


@lru_cache
def _get_flywire_types(mat, add_side=False):
    """Fetch types from `info`.

    - cached
    - uses `hemibrain_type` first and then falls back to `cell_type`
    - maps to given materialization version

    """
    print(f"Caching FlyWire `type` annotations for mat {mat}... ", end="", flush=True)
    if mat != "live":
        timestamp = f"mat_{mat}"
    else:
        timestamp = None
    table = _get_table(which="info")
    has_hb_type = table.hemibrain_type.notnull() & table.hemibrain_type_source.notnull()
    has_cell_type = table.cell_type.notnull()
    cols = ["supervoxel_id", "hemibrain_type", "cell_type"]
    if add_side:
        cols += ["side"]
    typed = table.loc[has_hb_type | has_cell_type, cols]
    typed[f"root_{mat}"] = flywire.supervoxels_to_roots(
        typed.supervoxel_id.values, timestamp=timestamp, progress=False
    )

    if add_side:
        has_ct = typed.cell_type.notnull()
        typed.loc[has_ct, "cell_type"] = [
            f"{t}_{s}"
            for t, s in zip(typed.cell_type.values[has_ct], typed.side.values[has_ct])
        ]
        has_ht = typed.hemibrain_type.notnull()
        typed.loc[has_ht, "hemibrain_type"] = [
            f"{t}_{s}"
            for t, s in zip(
                typed.hemibrain_type.values[has_ht], typed.side.values[has_ht]
            )
        ]

    type_dict = (
        typed[typed.cell_type.notnull()].set_index(f"root_{mat}").cell_type.to_dict()
    )
    type_dict.update(
        typed[typed.hemibrain_type.notnull()]
        .set_index(f"root_{mat}")
        .hemibrain_type.to_dict()
    )
    print("Done!")
    return type_dict


@lru_cache
def _get_hemibrain_types(add_side=False, use_morphology_type=False):
    """Fetch hemibrain types from flytable."""
    print("Caching hemibrain `type` annotations... ", end="", flush=True)
    meta = _get_hemibrain_meta()
    meta["bodyId"] = meta.bodyId.astype(int)

    # Overwrite cell type
    if use_morphology_type:
        meta = meta.copy()
        meta["type"] = meta.morphology_type

    # Drop untyped
    meta = meta[meta.type.notnull()]

    if add_side:
        meta["type"] = [f"{t}_{s}" for t, s in zip(meta.type.values, meta.side.values)]
    print("Done!")
    return meta.set_index("bodyId").type.to_dict()


@lru_cache
def _get_mcns_types(add_side=False):
    """Fetch male CNS types from clio."""
    print("Caching male CNS `type` annotations... ", end="", flush=True)
    meta = _get_mcns_meta()

    # Drop untyped
    meta = meta[meta.type.notnull()]

    if add_side:
        meta["type"] = [
            f"{t}_{s}" for t, s in zip(meta.type.values, meta.soma_side.values)
        ]
    print("Done!")
    return meta.set_index("bodyid").type.to_dict()


@lru_cache
def _get_fw_sides(mat):
    """Fetch sides from `info`.

    - cached
    - maps to given materialization version

    """
    print(f"Caching FlyWire `side` annotations for mat {mat}... ", end="", flush=True)
    if mat != "live":
        timestamp = f"mat_{mat}"
    else:
        timestamp = None
    table = _get_table(which="info")
    cols = ["supervoxel_id", "side"]

    sides = table[cols]
    sides[f"root_{mat}"] = flywire.supervoxels_to_roots(
        sides.supervoxel_id.values, progress=False, timestamp=timestamp
    )
    print("Done!")
    return sides.set_index(f"root_{mat}").side.to_dict()


@lru_cache
def _get_hb_sides():
    """Fetch hemibrain sides from flytable."""
    print("Caching hemibrain `side` annotations... ", end="", flush=True)
    meta = _get_hemibrain_meta()
    meta["bodyId"] = meta.bodyId.astype(int)

    # Drop neurons without a side
    meta = meta[meta.side.notnull()]

    print("Done!")
    return meta.set_index("bodyId").side.to_dict()


def _is_int(x):
    """Check if `x` is integer."""
    try:
        int(x)
        return True
    except BaseException:
        return False


def _add_types(
    edges,
    types,
    col,
    drop_untyped=False,
    sides=None,
    sides_rel=False,
    expand_morphology_types=False,
    ignore_cn_types=False,
    inplace=True,
):
    """Add types to edge list.

    Parameters
    ----------
    edges :     pd.DataFrame
                Edge list. Must contains columns 'pre', 'post' and 'weight'.
    types :     dict
                Dictionary mapping ID -> type.
    col :       'pre' | 'post'
                Which column to modify.
    expand_morphology_types : bool
                If True, will expand morphology types into connectivity types
                (e.g. "AVLP524" -> "AVLP524_a,AVLP524_b"). This makes aligning
                with the hemibrain much easier.
    ignore_cn_type :  bool
                If True, will collapse connectivity types.
    sides :     dict, optional
                Dictionary mapping ID -> side (left/right/center).
                If provided, will add side to type - e.g. as "SAD003_right".
    sides_rel : bool
                If True, will record `side` relative to the other neuron, i.e.
                e.g. as "SAD003_ipsi" or "SAD003_contra".

    Returns
    -------
    edges

    """
    assert col in ("pre", "post")
    other = {"pre": "post", "post": "pre"}[col]

    # This removes the connectivity suffixes
    if ignore_cn_types:
        types = _collapse_connectivity_types(types)

    # Turn morphology types into connectivity compound types
    if expand_morphology_types:
        keys, values = list(types.keys()), list(types.values())
        types = dict(zip(keys, _morphology_to_connectivity_types(values)))

    if not inplace:
        edges = edges.copy()

    # Add type
    edges["type"] = edges[col].map(types)
    has_type = edges.type.notnull()

    # Add side
    if sides:
        edges["side"] = edges[col].map(sides)

        has_side = edges.side.notnull()
        not_center = edges.side != "center"

        # Make sides relative to the neuron on the other side of the edge
        if sides_rel:
            edges["side_other"] = edges[other].map(sides)
            has_side_both = edges.side.notnull() & edges.side_other.notnull()
            not_center_both = (edges.side_other != "center") & (edges.side != "center")
            same_side = edges.side == edges.side_other
            edges.loc[
                has_type & has_side_both & not_center_both & same_side, "side"
            ] = "ipsi"
            edges.loc[
                has_type & has_side_both & not_center_both & ~same_side, "side"
            ] = "contra"

        # Add side to the type
        to_mod = has_side & has_type & not_center
        edges.loc[to_mod, "type"] = edges.loc[to_mod, ["type", "side"]].apply(
            lambda x: f"{x[0]}_{x[1]}", axis=1
        )

    # Replace `col` with type
    edges.loc[has_type, col] = edges.loc[has_type, "type"]
    edges.drop(["type", "side", "side_other"], errors="ignore", inplace=True, axis=1)

    if drop_untyped:
        edges = edges.loc[has_type]

    return edges


def _collapse_connectivity_types(type_dict):
    """Remove connectivity type suffixes from {ID: type} dictionary."""
    type_dict = type_dict.copy()
    hb_meta = _get_hemibrain_meta()
    cn2morph = hb_meta.set_index("type").morphology_type.to_dict()
    for k, v in type_dict.items():
        new_v = ",".join([cn2morph.get(t, t) for t in v.split(",")])
        type_dict[k] = new_v
    return type_dict


def _morphology_to_connectivity_types(x):
    """Translate morphology to connectivity types."""
    hb_meta = _get_hemibrain_meta().drop_duplicates("type")
    morph2cn = (
        hb_meta[hb_meta.morphology_type.notnull()]
        .groupby("morphology_type")
        .type.apply(lambda x: ",".join(x))
        .to_dict()
    )
    new_labels = []
    for label in x:
        if isinstance(x, str):
            new_labels.append(",".join([morph2cn.get(l, l) for l in label.split(",")]))
        else:
            new_labels.append(morph2cn.get(label, label))

    return np.array(new_labels)
