import clio
import requests

import datetime as dt
import seaserpent as ss
import neuprint as neu
import numpy as np
import pandas as pd

from urllib.parse import urlparse
from functools import lru_cache
from fafbseg import flywire
from pathlib import Path


CACHE_DIR = "~/.cocoa/cache/"
FLYWIRE_ANNOT_URL = "https://github.com/flyconnectome/flywire_annotations/raw/main/supplemental_files/Supplemental_file1_annotations.tsv"
HEMIBRAIN_ANNOT_URL = "https://github.com/flyconnectome/flywire_annotations/raw/main/supplemental_files/Supplemental_file4_hemibrain_meta.csv"
ANNOT_REPO_URL = "https://api.github.com/repos/flyconnectome/flywire_annotations"


def download_cache_file(url, force_reload="auto", verbose=True):
    """Load file from URL and cache locally.

    Parameters
    ----------
    url :           str
                    URL to file.
    force_reload :  bool
                    If True, will force downloading file again even if it
                    already exists locally.
    verbose :       bool

    Returns
    -------
    path :          pathlib.Path
                    Path to the locally stored file.

    """
    if not isinstance(url, str):
        raise TypeError(f"Expected `url` of type str, got {type(url)}")

    # Make sure the cache dir exists
    cache_dir = Path(CACHE_DIR).expanduser().absolute()
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)

    fp = cache_dir / Path(url).name

    if not fp.exists() or force_reload:
        if verbose:
            print(
                f"Caching {fp.name} from {urlparse(url).netloc}... ", end="", flush=True
            )
        r = requests.get(url, allow_redirects=True)
        r.raise_for_status()
        content_type = r.headers.get("content-type", "").lower()
        is_text = "text" in content_type or "html" in content_type
        with open(fp, mode="w" if is_text else "w") as f:
            f.write(r.content.decode())
        if verbose:
            print("Done.")
    return fp


@lru_cache
def _load_static_flywire_annotations(mat=None, force_reload=False):
    """Download and cache FlyWire annotations from Github repo."""
    print(
        f"Caching FlyWire annotations for mat '{mat}'... ",
        end="",
        flush=True,
    )
    fp = Path(CACHE_DIR).expanduser().absolute() / Path(FLYWIRE_ANNOT_URL).name

    # If file already exists, check if we need to refresh the cache
    if fp.exists():
        r = requests.get(ANNOT_REPO_URL)
        try:
            r.raise_for_status()
        except BaseException:
            print("Failed to check annotation repo for updates")
        # Last time anything was committed to the repo
        last_upd = dt.datetime.fromisoformat(r.json()["pushed_at"][:-1])
        # Last time the local file was modified
        last_mod = dt.datetime.fromtimestamp(fp.stat().st_mtime)
        if last_mod < last_upd:
            force_reload = True

    fp = download_cache_file(
        FLYWIRE_ANNOT_URL, force_reload=force_reload, verbose=False
    )
    table = pd.read_csv(fp, sep="\t", low_memory=False).astype(
        {"root_id": np.int64, "supervoxel_id": np.int64}
    )

    if mat not in ("630", 630, None):
        if mat in ("live", "current"):
            timestamp = None
        else:
            timestamp = f"mat_{mat}"
        to_update = ~flywire.is_latest_root(
            table.root_id, timestamp=timestamp, progress=False
        )
        table.loc[to_update, "root_id"] = flywire.supervoxels_to_roots(
            table.supervoxel_id.values[to_update], timestamp=timestamp, progress=False
        )
    print("Done.")
    return table


@lru_cache
def _load_live_flywire_annotations(mat=None):
    """Load live FlyWire annotations from SeaTable."""
    print(
        f"Caching live FlyWire annotations for mat '{mat}'... ",
        end="",
        flush=True,
    )

    cols = [
        "root_id",
        "supervoxel_id",
        "super_class",
        "cell_class",
        "cell_type",
        "hemibrain_type",
        "ito_lee_hemilineage",
        "side",
    ]
    info = _get_table(which="info")
    optic = _get_table(which="optic")
    table = pd.concat(
        (info.loc[info.flow.notnull(), cols], optic.loc[optic.flow.notnull(), cols]),
        axis=0,
    ).astype({"root_id": np.int64, "supervoxel_id": np.int64})

    if mat not in ("live", "current", None):
        timestamp = f"mat_{mat}"
        to_update = ~flywire.is_latest_root(
            table.root_id, timestamp=timestamp, progress=False
        )
        table.loc[to_update, "root_id"] = flywire.supervoxels_to_roots(
            table.supervoxel_id.values[to_update], timestamp=timestamp, progress=False
        )
    print("Done.")

    return table


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
def _load_static_hemibrain_annotations(force_reload=False):
    """Download and cache hemibrain annotations from Github repo."""
    print(
        "Caching hemibrain annotations... ",
        end="",
        flush=True,
    )
    fp = Path(CACHE_DIR).expanduser().absolute() / Path(HEMIBRAIN_ANNOT_URL).name

    # If file already exists, check if we need to refresh the cache
    if fp.exists():
        r = requests.get(ANNOT_REPO_URL)
        try:
            r.raise_for_status()
        except BaseException:
            print("Failed to check annotation repo for updates")
        # Last time anything was committed to the repo
        last_upd = dt.datetime.fromisoformat(r.json()["pushed_at"][:-1])
        # Last time the local file was modified
        last_mod = dt.datetime.fromtimestamp(fp.stat().st_mtime)
        if last_mod < last_upd:
            force_reload = True

    fp = download_cache_file(
        HEMIBRAIN_ANNOT_URL, force_reload=force_reload, verbose=False
    )
    table = pd.read_csv(fp)
    print("Done.")
    return table


@lru_cache
def _load_live_hemibrain_annotations():
    """Load live hemibrain annotations from SeaTable."""
    print(
        "Caching live hemibrain annotations... ",
        end="",
        flush=True,
    )

    table = ss.Table("hb_info", "hemibrain")[
        ["bodyId", "side", "type", "morphology_type"]
    ]
    print("Done.")

    return table


@lru_cache
def _get_hemibrain_meta(live=False):
    if live:
        meta = _load_live_hemibrain_annotations()
    else:
        meta = _load_static_hemibrain_annotations()
    return meta


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
def _get_fw_types(mat, add_side=False, live=False):
    """Fetch types from `info`.

    - cached
    - uses `cell_type` first and then falls back to `hemibrain_type`
    - maps to given materialization version

    """
    cols = ["supervoxel_id", "hemibrain_type", "cell_type"]
    if add_side:
        cols += ["side"]

    if not live:
        table = _load_static_flywire_annotations(mat=mat)
    else:
        table = _load_live_flywire_annotations(mat=mat)
    typed = table[table.hemibrain_type.notnull() | table.cell_type.notnull()]

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
        typed[typed.hemibrain_type.notnull()]
        .set_index("root_id")
        .hemibrain_type.to_dict()
    )
    type_dict.update(
        typed[typed.cell_type.notnull()].set_index("root_id").cell_type.to_dict()
    )
    return type_dict


@lru_cache
def _get_hemibrain_types(add_side=False, use_morphology_type=False, live=False):
    """Fetch hemibrain types from flytable."""
    meta = _get_hemibrain_meta(live=live)
    meta["bodyId"] = meta.bodyId.astype(int)

    # Overwrite cell type
    if use_morphology_type:
        meta = meta.copy()
        meta["type"] = meta.morphology_type

    # Drop untyped
    meta = meta[meta.type.notnull()]

    if add_side:
        meta["type"] = [f"{t}_{s}" for t, s in zip(meta.type.values, meta.side.values)]
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
    print("Done.")
    return meta.set_index("bodyid").type.to_dict()


@lru_cache
def _get_fw_sides(mat, live=False):
    """Fetch side annotations for Flywire."""
    if not live:
        sides = _load_static_flywire_annotations(mat=mat)
    else:
        sides = _load_live_flywire_annotations(mat=mat)

    return sides.set_index("root_id").side.to_dict()


@lru_cache
def _get_hb_sides(live=False):
    """Fetch hemibrain sides from flytable."""
    meta = _get_hemibrain_meta(live=live)
    meta["bodyId"] = meta.bodyId.astype(int)

    # Drop neurons without a side
    meta = meta[meta.side.notnull()]

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


def _collapse_connectivity_types(type_dict, live=False):
    """Remove connectivity type suffixes from {ID: type} dictionary."""
    type_dict = type_dict.copy()
    hb_meta = _get_hemibrain_meta(live=live)
    cn2morph = hb_meta.set_index("type").morphology_type.to_dict()
    for k, v in type_dict.items():
        new_v = ",".join([cn2morph.get(t, t) for t in v.split(",")])
        type_dict[k] = new_v
    return type_dict


def _morphology_to_connectivity_types(x, live=False):
    """Translate morphology to connectivity types."""
    hb_meta = _get_hemibrain_meta(live=live).drop_duplicates("type")
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
