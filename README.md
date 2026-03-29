# Astro

A*-RL tabanlı **global-local path planning and following** projesi.

Bu repo, Ay topografyası üzerinde iki aşamalı bir yaklaşım kullanır:

1. **Global Planlama (A\*)**: 3D mesh üzerinde eğim kısıtlı küresel rota üretimi
2. **Local Takip (SAC RL)**: Üretilen rotanın ilk waypoint segmentlerinde engellerden kaçarak takip eğitimi

## İçerik

- `createAstarPath.py`: GLB mesh'ten global A* rota üretir
- `trainLlmModel.py`: Local çevre + SAC eğitimi + çıktı dosyaları
- `rover_viewer.html`: Bölüm verilerini 3D görselleştirme
- `tets.py`: `.npz` örnekten `heightmap.json` üretir
- `path_coords.npy`, `scale_factor.npy`: Global planlama çıktıları
- `sac_rover.zip`, `vec_normalize.pkl`: Eğitilmiş model ve normalize istatistikleri

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib trimesh gymnasium stable-baselines3
```

## Çalıştırma Sırası

1. Global rota üret:

```bash
python createAstarPath.py
```

2. Local RL eğitimi başlat:

```bash
python trainLlmModel.py
```

3. Viewer aç:

- `rover_viewer.html` dosyasını tarayıcıda aç
- `rover_episode_data.json` ve `heightmap.json` aynı dizinde olmalı

## Notlar

- `trainLlmModel.py` içinde engel yoğunluğu ve patikadan sapma aralığı artırılmıştır.
- JSON çıktı yazımı atomik yapılır; bu sayede yarım JSON okuma hataları azaltılır.

## Lisans

Bu depoda lisans dosyası bulunmuyor. Gerekirse `LICENSE` ekleyin.
