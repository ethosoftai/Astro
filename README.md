# Astro

A*-RL tabanlı global-local path planning and following projesi.

## Klasör Yapısı

- `src/python/`: Python kodları (`createAstarPath.py`, `trainLlmModel.py`, `tets.py`)
- `src/web/`: Web/görselleştirme kodları (`rover_viewer.html`)
- `data/`: Veri çıktıları (`path_coords.npy`, `scale_factor.npy`, `rover_episode_data.json`, `heightmap.json`)
- `models/`: Model dosyaları (`sac_rover.zip`, `vec_normalize.pkl`)
- `assets/images/`: Görseller
- `logs/sac_rover_tensorboard/`: TensorBoard logları

## Çalıştırma

Global rota üretimi:

```bash
python src/python/createAstarPath.py
```

Local RL eğitimi:

```bash
python src/python/trainLlmModel.py
```

Heightmap üretimi (`single_sample.npz` `data/` altında olmalı):

```bash
python src/python/tets.py
```

Viewer:

- `src/web/rover_viewer.html` dosyasını aç.
- Viewer veriyi `../../data/` altından okur.
