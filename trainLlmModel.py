import json
import os
import sys
import tempfile
import webbrowser

import gymnasium as gym
import matplotlib
import numpy as np
from gymnasium import spaces
from matplotlib.patches import Polygon
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Qt plugin çökmesini engellemek için TkAgg; yoksa dosyaya kaydeden Agg kullan
try:
    import tkinter  # noqa: F401
    matplotlib.use("TkAgg")
except ImportError:
    matplotlib.use("Agg")
    print("[UYARI] Tkinter bulunamadı. Grafikler ekranda gösterilmek yerine .png olarak kaydedilecek.")

import matplotlib.pyplot as plt


class LunarRoverLocalEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        path_subset,
        scale_factor,
        hazard_radius_m=3.5,
        detection_radius_m=80.0,
        lookahead_dist_m=15.0,
        max_speed_kmh=40.0,
        max_steps=30000,
    ):
        super().__init__()

        self.scale_factor = float(scale_factor)

        path_subset = np.asarray(path_subset, dtype=np.float32)
        if path_subset.ndim != 2 or path_subset.shape[1] < 2:
            raise ValueError("path_subset en az iki sütunlu (x, y) koordinatlar içermelidir.")
        if len(path_subset) < 2:
            raise ValueError("En az 2 waypoint gerekli.")

        self.path_coords = path_subset[:, :2].astype(np.float32)
        self.final_goal = self.path_coords[-1]

        # scale_factor: km / birim
        self.m2u = 1.0 / (self.scale_factor * 1000.0)

        self.hazard_size_mu = float(hazard_radius_m) * self.m2u
        self.detection_radius = float(detection_radius_m) * self.m2u
        self.lookahead_dist = float(lookahead_dist_m) * self.m2u
        self.max_v = (float(max_speed_kmh) * 1000.0 / 3600.0) * self.m2u
        self.max_steps = int(max_steps)

        self.episode_count = 0
        self.history = []
        self.hazards = []
        self.hazards_np = np.zeros((0, 3), dtype=np.float32)

        self.num_lidar_rays = 8
        self.fov_rad = np.deg2rad(120.0)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        obs_dim = 3 + self.num_lidar_rays + 1
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.max_battery = 100.0
        self.battery = self.max_battery

        self.current_pos = np.copy(self.path_coords[0])
        p0, p1 = self.path_coords[0], self.path_coords[1]
        self.heading = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
        self.current_step = 0
        self._last_closest_idx = 0
        self.current_carrot = np.copy(self.final_goal)

        self._spawn_hazards_along_path()

    def _get_pure_pursuit_target(self):
        min_dist = float("inf")
        closest_pt = None
        closest_idx = 0
        cte = 0.0

        for i in range(len(self.path_coords) - 1):
            p1 = self.path_coords[i]
            p2 = self.path_coords[i + 1]
            line_vec = p2 - p1
            line_len = np.linalg.norm(line_vec)
            if line_len <= 1e-8:
                continue

            line_unit = line_vec / line_len
            p_vec = self.current_pos - p1
            proj_len = np.dot(p_vec, line_unit)
            proj_len_clamped = np.clip(proj_len, 0.0, line_len)
            proj_pt = p1 + proj_len_clamped * line_unit

            dist = np.linalg.norm(self.current_pos - proj_pt)
            if dist < min_dist:
                min_dist = dist
                closest_pt = proj_pt
                closest_idx = i

                cross_prod = line_vec[0] * p_vec[1] - line_vec[1] * p_vec[0]
                cte = dist * np.sign(cross_prod)

        if closest_pt is None:
            closest_pt = np.copy(self.path_coords[-1])
            self._last_closest_idx = max(0, len(self.path_coords) - 2)
            return closest_pt, cte

        closest_idx = max(closest_idx, getattr(self, "_last_closest_idx", 0))
        self._last_closest_idx = closest_idx

        lookahead_pt = np.copy(closest_pt)
        remaining_lookahead = self.lookahead_dist

        for curr_idx in range(closest_idx, len(self.path_coords) - 1):
            seg_start = closest_pt if curr_idx == closest_idx else self.path_coords[curr_idx]
            seg_end = self.path_coords[curr_idx + 1]
            seg_vec = seg_end - seg_start
            seg_len = np.linalg.norm(seg_vec)

            if seg_len <= 1e-8:
                lookahead_pt = np.copy(seg_end)
                continue

            if remaining_lookahead <= seg_len:
                dir_vec = seg_vec / seg_len
                lookahead_pt = seg_start + dir_vec * remaining_lookahead
                break

            remaining_lookahead -= seg_len
            lookahead_pt = np.copy(seg_end)

        return lookahead_pt, cte

    def _spawn_hazards_along_path(self):
        self.hazards = []

        for i in range(len(self.path_coords) - 1):
            p1 = self.path_coords[i]
            p2 = self.path_coords[i + 1]

            segment_vec = p2 - p1
            segment_len = np.linalg.norm(segment_vec)
            if segment_len <= 1e-8:
                continue

            segment_unit = segment_vec / segment_len
            perp_vec = np.array([-segment_unit[1], segment_unit[0]], dtype=np.float32)

            num_hazards_for_segment = int((segment_len * self.scale_factor * 1000.0) / 10.0)

            for _ in range(num_hazards_for_segment):
                for _retry in range(15):
                    t = np.random.uniform(0.1, 0.9)
                    base_point = p1 + t * segment_vec

                    offset_dist = np.random.uniform(-self.hazard_size_mu * 15.0, self.hazard_size_mu * 15.0)
                    new_hazard_pos = base_point + perp_vec * offset_dist

                    new_hazard_radius = np.random.uniform(self.hazard_size_mu * 0.5, self.hazard_size_mu * 2.5)
                    hz_data = np.array(
                        [new_hazard_pos[0], new_hazard_pos[1], new_hazard_radius],
                        dtype=np.float32,
                    )

                    too_close_to_other = False
                    if self.hazards:
                        hz_arr = np.array(self.hazards, dtype=np.float32)
                        dists = np.linalg.norm(hz_arr[:, :2] - new_hazard_pos, axis=1)
                        min_allowed_dists = (hz_arr[:, 2] + new_hazard_radius) * 1.5
                        if np.any(dists < min_allowed_dists):
                            too_close_to_other = True

                    if not too_close_to_other:
                        far_from_start = np.linalg.norm(new_hazard_pos - self.path_coords[0]) > self.hazard_size_mu * 3.0
                        far_from_goal = np.linalg.norm(new_hazard_pos - self.final_goal) > self.hazard_size_mu * 3.0
                        if far_from_start and far_from_goal:
                            self.hazards.append(hz_data)
                            break

        self.hazards_np = np.array(self.hazards, dtype=np.float32) if self.hazards else np.zeros((0, 3), dtype=np.float32)

    def _get_lidar(self):
        angles = np.linspace(-self.fov_rad / 2, self.fov_rad / 2, self.num_lidar_rays) + self.heading
        lidar_dists = np.zeros(self.num_lidar_rays, dtype=np.float32)

        if len(self.hazards_np) == 0:
            return lidar_dists

        max_hz_radius = np.max(self.hazards_np[:, 2]) if len(self.hazards_np) > 0 else 0.0
        close_x = np.abs(self.hazards_np[:, 0] - self.current_pos[0]) < (self.detection_radius + max_hz_radius)
        close_y = np.abs(self.hazards_np[:, 1] - self.current_pos[1]) < (self.detection_radius + max_hz_radius)
        nearby_hz = self.hazards_np[close_x & close_y]

        if len(nearby_hz) == 0:
            return lidar_dists

        r2 = nearby_hz[:, 2] ** 2
        ray_dirs = np.column_stack((np.cos(angles), np.sin(angles)))
        L = nearby_hz[:, :2] - self.current_pos

        tca = np.dot(ray_dirs, L.T)
        L_sq = np.sum(L ** 2, axis=1)
        d2 = L_sq - tca ** 2

        valid = (tca >= 0) & (d2 <= r2)
        d2_clipped = np.minimum(d2, r2)
        thc = np.sqrt(np.maximum(0.0, r2 - d2_clipped))

        t0 = tca - thc
        t0[~valid] = np.inf
        t0[t0 <= 0] = np.inf

        min_dists = np.min(t0, axis=1)
        norm_dists = 1.0 - (np.clip(min_dists, 0.0, self.detection_radius) / self.detection_radius)
        return norm_dists.astype(np.float32)

    def _get_obs(self):
        lookahead_pt, cte = self._get_pure_pursuit_target()
        self.current_carrot = lookahead_pt

        dist_to_carrot = np.linalg.norm(lookahead_pt - self.current_pos)
        norm_dist = np.clip(dist_to_carrot / max(self.lookahead_dist * 2.0, 1e-8), 0.0, 1.0)

        target_theta = np.arctan2(
            lookahead_pt[1] - self.current_pos[1],
            lookahead_pt[0] - self.current_pos[0],
        )
        heading_error = target_theta - self.heading
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
        norm_heading = heading_error / np.pi

        cte_limit = max(self.hazard_size_mu * 5.0, 1e-8)
        norm_cte = np.clip(cte / cte_limit, -1.0, 1.0)

        lidar_obs = self._get_lidar()
        norm_battery = np.clip(self.battery / self.max_battery, 0.0, 1.0)

        return np.concatenate([[norm_dist, norm_heading, norm_cte], lidar_obs, [norm_battery]]).astype(np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.current_step += 1
        prev_pos = np.copy(self.current_pos)
        prev_carrot = np.copy(getattr(self, "current_carrot", self.final_goal))

        forward_action = (action[0] + 1.0) / 2.0
        v = forward_action * self.max_v
        omega = float(action[1]) * (np.pi / 12.0)

        self.heading += omega
        self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi

        centrifugal_force = v * abs(omega)
        lunar_grip = 0.5 * self.m2u

        slip_vector = np.zeros(2, dtype=np.float32)
        slip_penalty = 0.0
        if centrifugal_force > lunar_grip:
            slip_amount = (centrifugal_force - lunar_grip) * 1.5
            slip_dir = self.heading - np.sign(omega) * (np.pi / 2.0)
            slip_vector[0] = slip_amount * np.cos(slip_dir)
            slip_vector[1] = slip_amount * np.sin(slip_dir)
            slip_penalty = (slip_amount / max(self.hazard_size_mu, 1e-8)) * 3.0

        self.current_pos[0] += v * np.cos(self.heading) + slip_vector[0]
        self.current_pos[1] += v * np.sin(self.heading) + slip_vector[1]
        self.history.append(np.copy(self.current_pos))

        drain = 0.05 + (forward_action * 0.15) + (abs(float(action[1])) * 0.1)
        base_regen = 0.025
        critical_regen = 0.06 if self.battery < (self.max_battery * 0.2) else 0.0
        solar_regen = base_regen + critical_regen
        self.battery = float(np.clip(self.battery + solar_regen - drain, 0.0, self.max_battery))

        obs = self._get_obs()
        _, cte = self._get_pure_pursuit_target()

        reward = -0.05
        reward -= slip_penalty

        terminated = False
        truncated = False

        if self.battery <= 0.0:
            reward -= 20.0
            terminated = True

        prev_dist_to_carrot = np.linalg.norm(prev_carrot - prev_pos)
        curr_dist_to_carrot = np.linalg.norm(self.current_carrot - self.current_pos)
        reward += ((prev_dist_to_carrot - curr_dist_to_carrot) / max(self.hazard_size_mu, 1e-8)) * 5.0

        prev_dist_to_goal = np.linalg.norm(self.final_goal - prev_pos)
        curr_dist_to_goal = np.linalg.norm(self.final_goal - self.current_pos)
        goal_progress = prev_dist_to_goal - curr_dist_to_goal
        if goal_progress > 0:
            reward += (goal_progress / max(self.hazard_size_mu, 1e-8)) * 2.0

        heading_err = abs(float(obs[1]))
        max_lidar_danger = float(np.max(obs[3:3 + self.num_lidar_rays]))
        heading_weight = np.clip(1.0 - (max_lidar_danger / 0.4), 0.0, 1.0)
        reward -= heading_err * 0.3 * heading_weight
        if heading_err < 0.2:
            reward += float(action[0]) * 0.2 * heading_weight

        collided = False
        if len(self.hazards_np) > 0:
            max_r = np.max(self.hazards_np[:, 2])
            close_x = np.abs(self.hazards_np[:, 0] - self.current_pos[0]) < (max_r + self.hazard_size_mu)
            close_y = np.abs(self.hazards_np[:, 1] - self.current_pos[1]) < (max_r + self.hazard_size_mu)
            nearby_collision_hz = self.hazards_np[close_x & close_y]

            if len(nearby_collision_hz) > 0:
                dists_sq = np.sum((nearby_collision_hz[:, :2] - self.current_pos) ** 2, axis=1)
                r2 = nearby_collision_hz[:, 2] ** 2
                if np.any(dists_sq < r2):
                    reward -= 5.0
                    terminated = True
                    collided = True

        if not collided:
            cte_abs = abs(cte)
            free_corridor = self.hazard_size_mu * 1.5
            if cte_abs > free_corridor:
                excess = cte_abs - free_corridor
                reward -= (excess / max(self.hazard_size_mu, 1e-8)) * 0.1

            if cte_abs > self.hazard_size_mu * 3.5:
                reward -= 2.0

        if curr_dist_to_goal < self.hazard_size_mu * 1.5:
            reward += 20.0
            terminated = True

        if self.current_step >= self.max_steps:
            truncated = True

        return obs, float(reward), terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1

        if self.episode_count % 10 == 0 and len(self.history) > 10:
            self.render_plot()

        self.current_pos = np.copy(self.path_coords[0])

        p0 = self.path_coords[0]
        p1 = self.path_coords[1]
        self.heading = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])

        self.current_step = 0
        self._last_closest_idx = 0
        self.battery = self.max_battery
        self.history = [np.copy(self.current_pos)]
        self.current_carrot = np.copy(self.path_coords[1])

        if self.episode_count % 20 == 1 or len(self.hazards_np) == 0:
            self._spawn_hazards_along_path()

        return self._get_obs(), {}

    def render_plot(self):
        plt.figure(figsize=(9, 9))
        ax = plt.gca()

        plt.plot(self.path_coords[:, 0], self.path_coords[:, 1], "b--", alpha=0.5, linewidth=2, label="6-WP Path")
        plt.scatter(self.path_coords[:, 0], self.path_coords[:, 1], c="cyan", s=30, zorder=5)

        for hz in self.hazards:
            circle = plt.Circle((hz[0], hz[1]), hz[2], color="orange", alpha=0.6)
            ax.add_patch(circle)

        hist = np.array(self.history, dtype=np.float32)
        if len(hist) > 0:
            plt.plot(hist[:, 0], hist[:, 1], "g-", linewidth=2, alpha=0.8, label="Rover Trajectory")

        if hasattr(self, "current_carrot"):
            plt.scatter(
                self.current_carrot[0],
                self.current_carrot[1],
                c="gold",
                marker="*",
                s=300,
                edgecolors="black",
                zorder=15,
            )
            plt.plot(
                [self.current_pos[0], self.current_carrot[0]],
                [self.current_pos[1], self.current_carrot[1]],
                "y:",
                linewidth=1.5,
            )

        tri_size = self.hazard_size_mu * 1.2
        triangle_pts = np.array(
            [[tri_size, 0], [-tri_size / 2, tri_size / 2], [-tri_size / 2, -tri_size / 2]],
            dtype=np.float32,
        )
        R = np.array(
            [[np.cos(self.heading), -np.sin(self.heading)], [np.sin(self.heading), np.cos(self.heading)]],
            dtype=np.float32,
        )
        rotated_tri = np.dot(triangle_pts, R.T) + self.current_pos
        ax.add_patch(Polygon(rotated_tri, facecolor="red", edgecolor="white", zorder=10, label="Rover"))

        plt.title(f"Optimized Path (6 Waypoints)\nEpisode: {self.episode_count}")
        ax.set_facecolor("#111111")
        plt.grid(color="white", alpha=0.1)
        plt.legend(loc="upper right", facecolor="black", labelcolor="white")

        if matplotlib.get_backend().lower() == "agg":
            plt.savefig(f"sac_episode_{self.episode_count}.png", bbox_inches="tight")

        try:
            render_data = {
                "episode": int(self.episode_count),
                "scale_factor": float(self.scale_factor),
                "m2u": float(self.m2u),
                "path": self.path_coords[:, :2].tolist(),
                "hazards": self.hazards_np.tolist() if len(self.hazards_np) > 0 else [],
                "trajectory": [p.tolist() for p in self.history],
            }
            fd, tmp_path = tempfile.mkstemp(prefix="rover_episode_data_", suffix=".json", dir=".")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(render_data, f)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, "rover_episode_data.json")
            except Exception:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise

            viewer_path = os.path.abspath("rover_viewer.html")
            print(f"[3D] 3D Görev İzleyici Hazır: {viewer_path}")
            # webbrowser.open(f"file://{viewer_path}")
        except Exception as e:
            print(f"[ERROR] 3D veri dışa aktarma hatası: {e}")

        if matplotlib.get_backend().lower() != "agg":
            try:
                print(
                    f"Bölüm {self.episode_count} haritası açıldı. "
                    f"3D izleyici için rover_viewer.html dosyasını da açabilirsiniz."
                )
                plt.show()
            except Exception as e:
                print(f"Plot gösterme hatası: {e}")
        plt.close()


def train_first_6_waypoints(path_coords, scale_factor, total_timesteps=350000):
    print("\n[RL] İlk 6 waypoint için çoklu segment SAC eğitimi başlıyor...")

    path_coords = np.asarray(path_coords, dtype=np.float32)
    if len(path_coords) < 2:
        raise ValueError("Eğitim için en az 2 waypoint gerekli.")
    path_subset = path_coords[: min(6, len(path_coords))]

    raw_env = LunarRoverLocalEnv(path_subset, scale_factor)
    vec_env = DummyVecEnv([lambda: raw_env])

    if os.path.exists("vec_normalize.pkl"):
        print("[YÜKLENİYOR] VecNormalize ayarları bulundu. Yükleniyor...")
        env = VecNormalize.load("vec_normalize.pkl", vec_env)
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model_path = "sac_rover.zip"
    tensorboard_dir = "./sac_rover_tensorboard/"

    if os.path.exists(model_path):
        print("[YÜKLENİYOR] Önceki model bulundu. Eğitim kaldığı yerden devam ediyor.")
        model = SAC.load(model_path, env=env)
        model.tensorboard_log = tensorboard_dir
    else:
        print("[YENİ MODEL] Sıfırdan SAC modeli oluşturuluyor...")
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            batch_size=256,
            tensorboard_log=tensorboard_dir,
        )

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="./", name_prefix="sac_yedek")

    interrupted = False
    try:
        step_size = 5000
        steps_done = 0
        while steps_done < total_timesteps:
            batch = min(step_size, total_timesteps - steps_done)
            model.learn(
                total_timesteps=batch,
                callback=checkpoint_callback,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            steps_done += batch
            env.save("vec_normalize.pkl")
        print("[OK] Eğitim başarıyla tamamlandı.")
    except KeyboardInterrupt:
        interrupted = True
        print("\n[DURDURULDU] Eğitim kullanıcı tarafından kesildi.")
    finally:
        print("[KAYDEDİLİYOR] Model ve çevre istatistikleri diske kaydediliyor...")
        env.save("vec_normalize.pkl")
        model.save("sac_rover")
        print("[BAŞARILI] Kayıt işlemi tamamlandı.")

    return model, env, interrupted


if __name__ == "__main__":
    try:
        print("[INIT] path_coords ve scale_factor diskten yükleniyor...")
        path_coords = np.load("path_coords.npy")
        scale_factor = float(np.load("scale_factor.npy"))

        train_first_6_waypoints(path_coords, scale_factor, total_timesteps=350000)
    except KeyboardInterrupt:
        print("\n[ÇIKIŞ] Program güvenle sonlandırıldı.")
        sys.exit(0)