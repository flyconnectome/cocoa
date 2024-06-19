from .ds_utils import _get_clio_client


FLYWIRE_MINIMAL_SCENE = {
    "layers": [
        {
            "source": "graphene://https://prod.flywire-daf.com/segmentation/1.0/fly_v31",
            "type": "segmentation_with_graph",
            "selectedAlpha": 0.14,
            "segments": [],
            "name": "flywire",
        }
    ],
    "layout": "3d",
}


FLYWIRE_FLAT_MINIMAL_SCENE = {
    "layers": [
        {
            "source": "precomputed://gs://flywire_v141_m783",
            "type": "segmentation",
            "selectedAlpha": 0.14,
            "segments": [],
            "name": "flywire_783",
        }
    ],
    "layout": "3d",
}


HEMIBRAIN_MINIMAL_SCENE = {
    "layers": [
        {
            "type": "segmentation",
            "source": "precomputed://https://spine.itanna.io/files/data/hemibrain2flywire/precomputed/neuronmeshes/mesh#type=mesh",
            "tab": "source",
            "segments": [],
            "colorSeed": 2407669673,
            "name": "hemibrain_meshes",
            "visible": True,
        }
    ],
    "showAxisLines": False,
    "layout": "3d",
}

MCNS_MINIMAL_SCENE = {
    "layers": [
        {
            "type": "segmentation",
            "source": [
                {
                    "url": "",
                    "transform": {
                        "matrix": [
                            [0.9939, -0.0292, 0.0346, 34642.4292],
                            [0.0522, 0.8457, -0.0324, 9300.882],
                            [-0.0565, -0.0305, 0.9484, -982.3336],
                        ],
                        "outputDimensions": {
                            "x": [4e-9, "m"],
                            "y": [4e-9, "m"],
                            "z": [4e-8, "m"],
                        },
                    },
                    "subsources": {"default": True, "meshes": True},
                    "enableDefaultSubsources": False,
                }                
            ],
            "tab": "source",
            "segments": [],
            "colorSeed": 2407669673,
            "name": "hemibrain_meshes",
            "visible": True,
        }
    ],
    "showAxisLines": False,
    "layout": "3d",
}


def _get_mcns_scene():
    """Parse the minimal scene for male CNS."""
    global MCNS_MINIMAL_SCENE
    
    client = _get_clio_client('CNS')

    # Extract data source from the client 
    url = client.meta['versions'][0]['neuroglancer']['layers'][0]['source']['url']
    MCNS_MINIMAL_SCENE['layers'][0]['source'][0]['url'] = url

    return MCNS_MINIMAL_SCENE

