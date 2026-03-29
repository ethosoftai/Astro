import numpy as np
import trimesh
import heapq
import time
from pathlib import Path
import matplotlib.pyplot as plt
from collections import deque
from scipy.interpolate import griddata
from matplotlib.colors import LightSource

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"

# =================================================================
# 1. LUNAR ROVER ENVIRONMENT
# =================================================================
class LunarRoverEnv:
    def __init__(self, glb_path):
        print(f"[ENV] Orijinal Mesh yükleniyor: {glb_path}")
        scene = trimesh.load(glb_path, force='mesh')

        if isinstance(scene, trimesh.Scene):
            self.mesh = trimesh.util.concatenate([
                g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)
            ])
        else:
            self.mesh = scene

        print("[ENV] Harita onarılıyor (Kopuk noktalar birleştiriliyor)...")
        self.mesh.merge_vertices()

        self.vertices = np.array(self.mesh.vertices, dtype=np.float64)
        self.faces = np.array(self.mesh.faces)

        print("[ENV] Normaller ve Eğimler hesaplanıyor...")
        up_vector = np.median(self.mesh.vertex_normals, axis=0)
        up_vector /= np.linalg.norm(up_vector)

        dot = np.dot(self.mesh.vertex_normals, up_vector)
        self.slopes = np.degrees(np.arccos(np.abs(np.clip(dot, -1.0, 1.0))))

        print("[ENV] Komşuluk grafı (Adjacency) oluşturuluyor...")
        self.adjacency = self.mesh.vertex_neighbors
        print("[ENV] Hazır!")

    def find_guaranteed_target(self, start_idx, max_slope, min_steps=500):
        visited = {start_idx}
        queue = deque([(start_idx, 0)])

        while queue:
            current, steps = queue.popleft()
            if steps >= min_steps:
                return current
            for neighbor in self.adjacency[current]:
                if neighbor not in visited and self.slopes[neighbor] <= max_slope:
                    visited.add(neighbor)
                    queue.append((neighbor, steps + 1))
        return None

# =================================================================
# 2. GLOBAL PATHFINDING (A* 3D)
# =================================================================
def a_star_3d(env, start_idx, end_idx, max_slope=40.0, max_iter=2000000):
    print(f"\n[A*] Global Rota aranıyor: {start_idx} -> {end_idx}")
    t0 = time.time()
    goal_pos = env.vertices[end_idx]

    open_set = [(0.0, start_idx)]
    came_from = {}
    g_score = {start_idx: 0.0}
    closed_set = set()
    iterations = 0

    while open_set and iterations < max_iter:
        iterations += 1
        current_f, current = heapq.heappop(open_set)

        if current == end_idx:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_idx)
            print(f"[A*] BAŞARILI! Süre: {time.time()-t0:.2f}sn | Adım Sayısı (Waypoint): {len(path):,}")
            return path[::-1]

        closed_set.add(current)

        for neighbor in env.adjacency[current]:
            if neighbor in closed_set or env.slopes[neighbor] > max_slope:
                continue

            dist = np.linalg.norm(env.vertices[neighbor] - env.vertices[current])
            slope_penalty = (env.slopes[neighbor] / max_slope) ** 2
            tentative_g = g_score[current] + (dist * (1.0 + 3.0 * slope_penalty))

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h = np.linalg.norm(env.vertices[neighbor] - goal_pos)
                heapq.heappush(open_set, (tentative_g + h, neighbor))

    return None

# =================================================================
# 3. ANA ÇALIŞTIRMA & 2D/3D HARİTALAMA
# =================================================================
try:
    glb_dosyasi = str(ROOT_DIR / "Moon_NASA_LRO_8k_Topo.glb")
    env = LunarRoverEnv(glb_dosyasi)

    valid_starts = np.where(env.slopes < 10.0)[0]
    start_node, end_node = None, None

    print("\n[TARAMA] Geçerli ve geniş bir harita bölgesi aranıyor...")
    for attempt in range(1, 15):
        s_node = np.random.choice(valid_starts)
        e_node = env.find_guaranteed_target(s_node, max_slope=40.0, min_steps=1200)

        if e_node is not None:
            start_node = s_node
            end_node = e_node
            print(f"[OK] Ana kıta bulundu! (Deneme {attempt})")
            break

    path = a_star_3d(env, start_node, end_node, max_slope=40.0)

    if path:
        path_coords = env.vertices[path]

        # 1. ÖLÇEKLENDİRME HESAPLAMASI
        model_radius = np.max(np.linalg.norm(env.vertices, axis=1))
        scale_factor = 1737.4 / model_radius

        diffs = np.diff(path_coords, axis=0)
        total_distance_km = np.sum(np.linalg.norm(diffs, axis=1)) * scale_factor

        print(f"\n🚀 [BİLGİ] Global A* Mesafesi: {total_distance_km:.1f} km")

        # Save the parameters to disk
        np.save(DATA_DIR / "path_coords.npy", path_coords)
        np.save(DATA_DIR / "scale_factor.npy", np.array(scale_factor))
        print("\n[OK] path_coords ve scale_factor dizine kaydedildi.")

        # =================================================================
        # 3A. 2D TOPOGRAFİK HARİTA GÖRSELLEŞTİRME
        # =================================================================
        print("\n[GÖRSEL] 2D Topografik Harita Hazırlanıyor...")

        distances = np.linalg.norm(env.vertices, axis=1)
        longitudes_deg = np.degrees(np.arctan2(env.vertices[:, 1], env.vertices[:, 0]))
        latitudes_deg = np.degrees(np.arcsin(env.vertices[:, 2] / distances))
        elevation_km = (distances * scale_factor) - 1737.4

        grid_x = np.linspace(-180, 180, 2000)
        grid_y = np.linspace(-90, 90, 1000)
        grid_x_mesh, grid_y_mesh = np.meshgrid(grid_x, grid_y)

        elevation_grid = griddata((longitudes_deg, latitudes_deg), elevation_km,
                                  (grid_x_mesh, grid_y_mesh), method='nearest')

        def convert_path_to_2d_lats_lons(coords):
            path_dists = np.linalg.norm(coords, axis=1)
            path_lons = np.degrees(np.arctan2(coords[:, 1], coords[:, 0]))
            path_lats = np.degrees(np.arcsin(coords[:, 2] / path_dists))
            for i in range(1, len(path_lons)):
                if abs(path_lons[i] - path_lons[i-1]) > 180:
                    path_lons[i] = np.nan
            return path_lons, path_lats

        global_path_lons, global_path_lats = convert_path_to_2d_lats_lons(path_coords)

        fig, ax = plt.subplots(figsize=(18, 10), dpi=100)
        cmap = plt.get_cmap('gist_earth')

        ls = LightSource(azdeg=315, altdeg=45)
        rgb_shaded = ls.shade(elevation_grid, cmap=cmap, blend_mode='soft', vert_exag=0.08)

        img = ax.imshow(rgb_shaded, extent=[-180, 180, -90, 90], origin='lower', aspect='equal')

        # A* Rotası (Kırmızı)
        ax.plot(global_path_lons, global_path_lats, color='red', linestyle='-', linewidth=2.5, label=f'Global A* Rotası ({total_distance_km:.1f} km)')

        start_lon, start_lat = global_path_lons[0], global_path_lats[0]
        end_lon, end_lat = global_path_lons[-1], global_path_lats[-1]
        ax.scatter(start_lon, start_lat, color='cyan', s=150, marker='o', edgecolors='white', label='Başlangıç', zorder=10)
        ax.scatter(end_lon, end_lat, color='gold', s=250, marker='*', edgecolors='white', label='Hedef', zorder=10)

        img_for_cbar = ax.imshow(elevation_grid, cmap=cmap, extent=[-180, 180, -90, 90], origin='lower', aspect='equal', alpha=0)
        cbar = fig.colorbar(img_for_cbar, ax=ax, shrink=0.75, pad=0.03, aspect=25)
        cbar.set_label('Yükseklik (km)', fontsize=14, color='white', labelpad=10)
        cbar.ax.tick_params(colors='white', labelsize=11)

        plt.title('Ay Küresel Navigasyon: A* 2D Projeksiyon', fontsize=18, fontweight='bold', color='white', pad=15)
        plt.xlabel('Boylam', fontsize=12, color='white')
        plt.ylabel('Enlem', fontsize=12, color='white')

        legend = plt.legend(loc='upper right', fontsize=11, facecolor='black', edgecolor='white', labelcolor='white')
        legend.get_frame().set_alpha(0.8)

        ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.3)
        ax.set_xticks(np.arange(-180, 181, 30))
        ax.set_yticks(np.arange(-90, 91, 30))

        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        plt.show()

        # =================================================================
        # 3B. 3D AY YÜZEYİ VE ROTA GÖRSELLEŞTİRME
        # =================================================================
        print("\n[GÖRSEL] 3D Model Haritası Hazırlanıyor...")
        
        fig3d = plt.figure(figsize=(12, 10), dpi=100)
        ax3d = fig3d.add_subplot(111, projection='3d')
        fig3d.patch.set_facecolor('black')
        ax3d.set_facecolor('black')

        # Ay küresini belirginleştirmek için rastgele 15.000 nokta seçiyoruz (Performans için)
        sample_size = min(15000, len(env.vertices))
        sample_indices = np.random.choice(len(env.vertices), size=sample_size, replace=False)
        moon_sample = env.vertices[sample_indices]

        # Arka plan Ay yüzeyi (Hafif gri/mavi tonlarında nokta bulutu)
        ax3d.scatter(moon_sample[:, 0], moon_sample[:, 1], moon_sample[:, 2], 
                     color='#4d5a68', s=1, alpha=0.15, label='Ay Yüzeyi (Örneklem)')

        # A* 3D Rotası (Kalın Kırmızı Çizgi)
        ax3d.plot(path_coords[:, 0], path_coords[:, 1], path_coords[:, 2], 
                  color='red', linewidth=3, label='A* 3D Rotası')

        # Başlangıç ve Bitiş Noktaları
        ax3d.scatter(path_coords[0, 0], path_coords[0, 1], path_coords[0, 2], 
                     color='cyan', s=100, marker='o', edgecolors='white', label='Başlangıç', zorder=5)
        ax3d.scatter(path_coords[-1, 0], path_coords[-1, 1], path_coords[-1, 2], 
                     color='gold', s=150, marker='*', edgecolors='white', label='Hedef', zorder=5)

        # 3D Eksen Ayarları
        ax3d.set_title('Ay Küresel Navigasyon: 3D A* Rotası', fontsize=16, fontweight='bold', color='white', pad=20)
        
        # Siyah arkaplan ve görünmez eksenler uzay hissi verir
        ax3d.grid(False)
        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.zaxis.pane.fill = False
        ax3d.xaxis.pane.set_edgecolor('black')
        ax3d.yaxis.pane.set_edgecolor('black')
        ax3d.zaxis.pane.set_edgecolor('black')
        ax3d.tick_params(colors='white')

        legend3d = ax3d.legend(loc='upper right', fontsize=10, facecolor='black', edgecolor='white', labelcolor='white')
        legend3d.get_frame().set_alpha(0.8)

        print("[OK] 3D Harita hazır!")
        plt.show()

except Exception as e: 
    print(f"Hata detayı: {e}")
