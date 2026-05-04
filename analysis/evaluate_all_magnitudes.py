import json
from pathlib import Path
from collections import defaultdict
import numpy as np

def load_raw_pcm_left(filepath):
    raw = np.fromfile(filepath, dtype=np.int16)
    if raw.size % 2 != 0:
        raw = raw[:-1]
    stereo = raw.reshape(-1, 2)
    return stereo[:, 0].astype(np.float32) / 32768.0

sdsu_dir = Path("raw_data/Participant-1-02232026")
pcm_files = list(sdsu_dir.glob("*.pcm"))

class_stats = defaultdict(lambda: {"rms": [], "peak": []})
for f in pcm_files:
    class_name = f.stem.split('_')[1].upper()
    sig = load_raw_pcm_left(f)
    class_stats[class_name]["rms"].append(float(np.sqrt(np.mean(sig**2))))
    class_stats[class_name]["peak"].append(float(np.max(np.abs(sig))))

output = {}
for cls_name, vals in class_stats.items():
    output[cls_name] = {
        "files": len(vals["rms"]),
        "avg_peak": round(np.mean(vals["peak"]), 5),
        "avg_rms": round(np.mean(vals["rms"]), 5)
    }

with open('sdsu_mags_clean.json', 'w') as f:
    json.dump(output, f, indent=2)
