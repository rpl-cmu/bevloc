# Third Party
import json
from copy import deepcopy
import warnings

# Default settings
settings = \
{
    "encoding": {
        "feature_size" : 64,
        "strategy": "push"
    },
    "grid_size": [200, 200, 8],
    "voxel_size": [0.5, 0.5, 0.5]
}

def read_settings_file(file_path):
    """
    Read the settings file and return the dictionary
    
    \param[in] file_path: Path to the settings file
    """
    out_settings  = deepcopy(settings)
    with open(file_path, "r") as f:
        file_settings = json.load(f)
        for key in file_settings:
            if key not in out_settings.keys():
                warnings.warn(f"Not in default settings, {key} may not be used...")
            out_settings[key] = file_settings.get(key)

        # Do any input or default checking here.
        if not out_settings.get("grid_size") or not len(out_settings["grid_size"]) == 3:
            raise ValueError("Grid size not specified or not 3 dimensional")
        if not out_settings.get("voxel_size") or not len(out_settings["voxel_size"]) == 3:
            raise ValueError("Voxel size not specified or not 3 dimensional")
        if not out_settings["encoding"].get("feature_size"):
            raise ValueError("Encoding size not specified")
        if not out_settings["encoding"].get("strategy") or out_settings["encoding"].get("strategy") not in ["push", "pull"]:
            raise ValueError("Strategy not specified or not push or pull")
    
    return out_settings