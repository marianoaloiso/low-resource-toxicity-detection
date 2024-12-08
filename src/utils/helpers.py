from pathlib import Path
import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

def save_json_with_numpy_conversion(data: dict, file_path: Path):
    with open(file_path, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder)

