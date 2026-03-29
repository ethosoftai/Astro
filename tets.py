import json
import numpy as np

NPZ_PATH = "single_sample.npz"
OUT_JSON = "heightmap.json"

data = np.load(NPZ_PATH)
target = data["target"].astype(np.float32)
target = np.clip(target, -1.0, 1.0)

payload = {
    "width": int(target.shape[1]),
    "height": int(target.shape[0]),
    "min": float(target.min()),
    "max": float(target.max()),
    "values": target.tolist()
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(payload, f)

print(f"[OK] {OUT_JSON} yazıldı. Shape={target.shape}, min={target.min():.4f}, max={target.max():.4f}")