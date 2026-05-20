#!/usr/bin/env python3

import math
import heapq
import yaml
import rospy
import numpy as np
import matplotlib.pyplot as plt
import cv2

from sensor_msgs.msg import Range, CompressedImage
from duckietown_msgs.msg import Twist2DStamped


class AStarDWAPlanner:
    def __init__(self):
        rospy.init_node("astar_dwa_node", anonymous=False)

        config_path = rospy.get_param("~config_path", "")
        if config_path == "":
            rospy.logerr("No config_path provided. Use _config_path:=/path/to/planner.yaml")
            raise RuntimeError("Missing config_path")

        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.vehicle_name = self.cfg["ros"]["vehicle_name"]
        self.wheels_topic = f"/{self.vehicle_name}/car_cmd_switch_node/cmd"
        self.tof_topic = f"/{self.vehicle_name}/front_center_tof_driver_node/range"

        self.cmd_pub = rospy.Publisher(
            self.wheels_topic,
            Twist2DStamped,
            queue_size=1
        )

        self.robot_radius = self.cfg["robot"]["radius"]
        self.safety_margin = self.cfg["robot"]["safety_margin"]
        self.sensing_radius = self.cfg["robot"]["sensing_radius"]

        self.use_unknown_obstacle = self.cfg["bonus"]["unknown_obstacle"]
        self.include_static_obstacle = self.cfg["bonus"].get(
            "include_static_obstacle_when_bonus",
            False
        )
        
        # DÜZELTME 1: Dinamik Engelleri hafızada tutacak liste
        self.dynamic_obstacles = []  
        self.latest_range = None

        tof_cfg = self.cfg.get("tof", {})
        self.tof_min_range = tof_cfg.get("min_range", 0.05)
        self.tof_max_range = tof_cfg.get("max_range", self.sensing_radius)
        self.tof_stable_readings = tof_cfg.get("stable_readings", 2)
        self.tof_counter = 0

        if self.use_unknown_obstacle:
            rospy.Subscriber(self.tof_topic, Range, self.tof_callback)

        self.map_width = self.cfg["map"]["width"]
        self.map_height = self.cfg["map"]["height"]
        self.resolution = self.cfg["map"]["resolution"]

        self.grid_w = int(self.map_width / self.resolution)
        self.grid_h = int(self.map_height / self.resolution)

        self.start = (
            self.cfg["start"]["x"],
            self.cfg["start"]["y"],
            self.cfg["start"]["theta"]
        )

        self.goal = (
            self.cfg["goal"]["x"],
            self.cfg["goal"]["y"]
        )

        self.state = np.array([
            self.start[0],
            self.start[1],
            self.start[2]
        ], dtype=float)

        self.current_v = 0.0
        self.current_w = 0.0

        self.obstacle = (
            self.cfg["obstacle"]["x"],
            self.cfg["obstacle"]["y"],
            self.cfg["obstacle"]["radius"]
        )

        self.inflated_radius = (
            self.obstacle[2] + self.robot_radius + self.safety_margin
        )

        self.global_path = self.run_astar()
        if len(self.global_path) == 0:
            rospy.logerr("A* could not find a path.")
            raise RuntimeError("No path found")

        self.goal_tolerance = self.cfg["control"]["goal_tolerance"]
        self.loop_rate = self.cfg["control"]["loop_rate"]

        self.enable_visualization = self.cfg["visualization"]["enabled"]
        self.viz_topic = f"/{self.vehicle_name}/astar_dwa_viz/compressed"

        if self.enable_visualization:
            self.viz_pub = rospy.Publisher(
                self.viz_topic,
                CompressedImage,
                queue_size=1
            )
            rospy.loginfo(f"Publishing visualization to: {self.viz_topic}")
        else:
            self.viz_pub = None

        rospy.loginfo("A* + DWA planner initialized.")
        rospy.loginfo(f"Publishing twist command to: {self.wheels_topic}")

    def tof_callback(self, msg):
        r = float(msg.range)

        if not math.isfinite(r):
            self.latest_range = None
            self.tof_counter = 0
            return

        if r < self.tof_min_range or r > self.tof_max_range:
            self.latest_range = None
            self.tof_counter = 0
            return

        self.latest_range = r
        self.tof_counter += 1

    def world_to_grid(self, x, y):
        gx = int(x / self.resolution)
        gy = int(y / self.resolution)
        gx = max(0, min(self.grid_w - 1, gx))
        gy = max(0, min(self.grid_h - 1, gy))
        return gx, gy

    def grid_to_world(self, gx, gy):
        x = (gx + 0.5) * self.resolution
        y = (gy + 0.5) * self.resolution
        return x, y

    def heuristic(self, a, b):
        # DÜZELTME 2: Math Domain Error (karekök çökmesi) düzeltildi
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def run_astar(self, start_xy=None):
        if start_xy is None:
            start_grid = self.world_to_grid(self.start[0], self.start[1])
        else:
            start_grid = self.world_to_grid(start_xy[0], start_xy[1])

        goal_grid = self.world_to_grid(self.goal[0], self.goal[1])

        obstacles = self.get_active_obstacles()

        moves = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
            (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
        ]

        open_set = []
        heapq.heappush(open_set, (0.0, start_grid))
        came_from = {}
        g_score = {start_grid: 0.0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_grid:
                return self.reconstruct_path(came_from, current)

            for dx, dy, move_cost in moves:
                nx = current[0] + dx
                ny = current[1] + dy

                if nx < 0 or ny < 0 or nx >= self.grid_w or ny >= self.grid_h:
                    continue

                wx, wy = self.grid_to_world(nx, ny)

                blocked = False
                for ox, oy, radius in obstacles:
                    inflated = radius + self.robot_radius + self.safety_margin
                    if math.hypot(wx - ox, wy - oy) <= inflated:
                        blocked = True
                        break

                if blocked:
                    continue

                neighbor = (nx, ny)
                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score, neighbor))

        return []

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return [self.grid_to_world(gx, gy) for gx, gy in path]

    def normalize_angle(self, angle):
        while angle > math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle

    def distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    # DÜZELTME 3: "Engel Körlüğü" çözüldü. Artık engelleri UNUTMUYOR.
    def get_active_obstacles(self):
        obstacles = []

        # Bonus kapalıysa veya özellikle izin verdiysek YAML obstacle kullanılır.
        if (not self.use_unknown_obstacle) or self.include_static_obstacle:
            obstacles.append(self.obstacle)

        obstacles.extend(self.dynamic_obstacles)
        return obstacles

    def detect_new_obstacle_from_tof(self):
        if not self.use_unknown_obstacle:
            return False

        if self.latest_range is None:
            return False

        if self.tof_counter < self.tof_stable_readings:
            return False

        if self.latest_range >= self.sensing_radius:
            return False

        ox = self.state[0] + self.latest_range * math.cos(self.state[2])
        oy = self.state[1] + self.latest_range * math.sin(self.state[2])

        if not (0.0 <= ox <= self.map_width and 0.0 <= oy <= self.map_height):
            return False

        # Aynı obstacle tekrar tekrar eklenmesin.
        for mx, my, _ in self.dynamic_obstacles:
            if self.distance((ox, oy), (mx, my)) < 0.15:
                return False

        obs_radius = self.cfg["obstacle"]["radius"]
        self.dynamic_obstacles.append((ox, oy, obs_radius))

        rospy.loginfo(
            f"NEW obstacle mapped at ({ox:.2f}, {oy:.2f}). Replanning A* path..."
        )

        return True

    def simulate_trajectory(self, state, v, w):
        dt = self.cfg["dwa"]["dt"]
        horizon = self.cfg["dwa"]["horizon"]
        trajectory = []
        x, y, theta = state.copy()
        steps = int(horizon / dt)

        for _ in range(steps):
            x += v * math.cos(theta) * dt
            y += v * math.sin(theta) * dt
            theta += w * dt
            theta = self.normalize_angle(theta)
            trajectory.append((x, y, theta))
        return trajectory

    def trajectory_collision(self, trajectory, obstacles):
        for obs in obstacles:
            ox, oy, radius = obs
            inflated = radius + self.robot_radius + self.safety_margin
            for x, y, _ in trajectory:
                if x < 0 or y < 0 or x > self.map_width or y > self.map_height:
                    return True
                if self.distance((x, y), (ox, oy)) <= inflated:
                    return True
        return False

    def path_cost(self, trajectory):
        end = trajectory[-1]
        end_xy = (end[0], end[1])
        min_dist = min([self.distance(end_xy, p) for p in self.global_path])
        return min_dist

    def goal_cost(self, trajectory):
        end = trajectory[-1]
        return self.distance((end[0], end[1]), self.goal)

    # DÜZELTME 4: Look-ahead (İleriye bakma) eklendi
    def get_lookahead_point(self, x, y):
        closest_idx = 0
        min_d = float('inf')
        for i, p in enumerate(self.global_path):
            d = self.distance((x, y), p)
            if d < min_d:
                min_d = d
                closest_idx = i
        
        # Yaklaşık 6 nokta ilerisine bak (yumuşak dönüş sağlar)
        lookahead_distance = 2 
        target_idx = min(len(self.global_path) - 1, closest_idx + lookahead_distance)
        return self.global_path[target_idx]

    def heading_cost(self, trajectory):
        end = trajectory[-1]
        x, y, theta = end

        # Robot artık nihai B noktasına değil, rotasındaki sıradaki noktaya bakıyor
        target_pt = self.get_lookahead_point(x, y)
        desired_theta = math.atan2(target_pt[1] - y, target_pt[0] - x)
        
        error = self.normalize_angle(desired_theta - theta)
        return abs(error)

    def obstacle_cost(self, trajectory, obstacles):
        min_dist_to_any_obs = float("inf")
        inflated_thresh = 0.0

        for obs in obstacles:
            ox, oy, radius = obs
            inflated = radius + self.robot_radius + self.safety_margin
            for x, y, _ in trajectory:
                d = self.distance((x, y), (ox, oy))
                if d < min_dist_to_any_obs:
                    min_dist_to_any_obs = d
                    inflated_thresh = inflated

        if min_dist_to_any_obs <= inflated_thresh:
            return float("inf")
        return 1.0 / (min_dist_to_any_obs - inflated_thresh + 1e-6)

    def dynamic_window(self):
        dwa = self.cfg["dwa"]
        dt = dwa["dt"]
        min_v = max(dwa["min_v"], self.current_v - dwa["max_accel_v"] * dt)
        max_v = min(dwa["max_v"], self.current_v + dwa["max_accel_v"] * dt)
        min_w = max(dwa["min_w"], self.current_w - dwa["max_accel_w"] * dt)
        max_w = min(dwa["max_w"], self.current_w + dwa["max_accel_w"] * dt)
        return min_v, max_v, min_w, max_w

    def dwa_control(self):
        dwa = self.cfg["dwa"]
        cost_cfg = self.cfg["cost"]

        # Artık tek engel değil, engeller LİSTESİ çekiliyor
        obstacles = self.get_active_obstacles()

        min_v, max_v, min_w, max_w = self.dynamic_window()
        v_samples = np.linspace(min_v, max_v, dwa["v_samples"])
        w_samples = np.linspace(min_w, max_w, dwa["w_samples"])

        best_cost = float("inf")
        best_control = (0.0, 0.0)
        best_trajectory = []
        all_trajectories = []

        for v in v_samples:
            for w in w_samples:
                trajectory = self.simulate_trajectory(self.state, v, w)
                all_trajectories.append(trajectory)

                # Tüm engellere karşı çarpışma kontrolü
                if self.trajectory_collision(trajectory, obstacles):
                    continue

                p_cost = self.path_cost(trajectory)
                g_cost = self.goal_cost(trajectory)
                o_cost = self.obstacle_cost(trajectory, obstacles)
                h_cost = self.heading_cost(trajectory)

                total_cost = (
                    cost_cfg["path_weight"] * p_cost +
                    cost_cfg["goal_weight"] * g_cost +
                    cost_cfg["obstacle_weight"] * o_cost +
                    cost_cfg["heading_weight"] * h_cost
                )

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_control = (v, w)
                    best_trajectory = trajectory

        return best_control, best_trajectory, all_trajectories

    def publish_cmd(self, v, w):
            robot_cfg = self.cfg["robot"]
            max_v_cmd = robot_cfg["max_v_cmd"]
            max_w_cmd = robot_cfg["max_w_cmd"]
            
            # Buradaki -1.0 çarpanı robotu sağa yerine sola döndürecektir.
            # Simülasyonda test et, eğer düzelirse YAML'dan omega_sign'ı yönet.
            omega_correction = -1.0 

            v_out = max(-max_v_cmd, min(max_v_cmd, v))
            w_out = max(-max_w_cmd, min(max_w_cmd, w))

            msg = Twist2DStamped()
            msg.header.stamp = rospy.Time.now()
            msg.v = v_out
            msg.omega = omega_correction * w_out # YÖN DÜZELTME
            self.cmd_pub.publish(msg)

    def update_internal_pose(self, v, w, dt):
        self.state[0] += v * math.cos(self.state[2]) * dt
        self.state[1] += v * math.sin(self.state[2]) * dt
        self.state[2] += w * dt
        self.state[2] = self.normalize_angle(self.state[2])

    def stop_robot(self):
        self.publish_cmd(0.0, 0.0)
        self.current_v = 0.0
        self.current_w = 0.0
        rospy.loginfo("Motors stopped.")

    def reached_goal(self):
        d = self.distance((self.state[0], self.state[1]), self.goal)
        return d <= self.goal_tolerance

    def world_to_canvas(self, x, y):
        viz_cfg = self.cfg.get("visualization", {})
        canvas_w = viz_cfg.get("canvas_width", 700)
        canvas_h = viz_cfg.get("canvas_height", 700)
        margin = viz_cfg.get("margin", 60)

        usable_w = canvas_w - 2 * margin
        usable_h = canvas_h - 2 * margin

        px = int(margin + (x / self.map_width) * usable_w)
        py = int(canvas_h - margin - (y / self.map_height) * usable_h)
        return px, py

    def draw_circle_world(self, canvas, x, y, radius_m, color, thickness):
        cx, cy = self.world_to_canvas(x, y)
        viz_cfg = self.cfg.get("visualization", {})
        canvas_w = viz_cfg.get("canvas_width", 700)
        margin = viz_cfg.get("margin", 60)
        usable_w = canvas_w - 2 * margin

        radius_px = int((radius_m / self.map_width) * usable_w)
        cv2.circle(canvas, (cx, cy), radius_px, color, thickness)

    def publish_visualization_image(self, canvas):
        if not self.enable_visualization or self.viz_pub is None:
            return
        success, encoded = cv2.imencode(".jpg", canvas)
        if not success:
            return
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = encoded.tobytes()
        self.viz_pub.publish(msg)

    def visualize(self, best_trajectory, all_trajectories):
        if not self.enable_visualization:
            return

        viz_cfg = self.cfg.get("visualization", {})
        canvas_w = viz_cfg.get("canvas_width", 700)
        canvas_h = viz_cfg.get("canvas_height", 700)
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

        # Harita gridleri ve arka plan çizimi
        cv2.putText(canvas, "A* + DWA Local Obstacle Avoidance", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        p00 = self.world_to_canvas(0.0, 0.0)
        p11 = self.world_to_canvas(self.map_width, self.map_height)
        cv2.rectangle(canvas, p11, p00, (0, 0, 0), 2)

        grid_step = self.resolution * 5
        x = 0.0
        while x <= self.map_width + 1e-6:
            p1 = self.world_to_canvas(x, 0.0)
            p2 = self.world_to_canvas(x, self.map_height)
            cv2.line(canvas, p1, p2, (230, 230, 230), 1)
            x += grid_step

        y = 0.0
        while y <= self.map_height + 1e-6:
            p1 = self.world_to_canvas(0.0, y)
            p2 = self.world_to_canvas(self.map_width, y)
            cv2.line(canvas, p1, p2, (230, 230, 230), 1)
            y += grid_step

        # A* Rota Çizimi
        if len(self.global_path) > 1:
            for i in range(len(self.global_path) - 1):
                p1 = self.world_to_canvas(self.global_path[i][0], self.global_path[i][1])
                p2 = self.world_to_canvas(self.global_path[i + 1][0], self.global_path[i + 1][1])
                cv2.line(canvas, p1, p2, (255, 0, 0), 3)

        # DWA Olası Yayları
        for traj in all_trajectories:
            if len(traj) < 2: continue
            for i in range(len(traj) - 1):
                p1 = self.world_to_canvas(traj[i][0], traj[i][1])
                p2 = self.world_to_canvas(traj[i + 1][0], traj[i + 1][1])
                cv2.line(canvas, p1, p2, (180, 180, 180), 1)

        # Seçilen DWA Yayı
        if len(best_trajectory) > 1:
            for i in range(len(best_trajectory) - 1):
                p1 = self.world_to_canvas(best_trajectory[i][0], best_trajectory[i][1])
                p2 = self.world_to_canvas(best_trajectory[i + 1][0], best_trajectory[i + 1][1])
                cv2.line(canvas, p1, p2, (0, 180, 0), 3)

        # Engelleri (TÜMÜNÜ) Çiz
        # Engelleri çiz
        obstacles = self.get_active_obstacles()

        # Eğer bonus kapalıysa veya bonus açıkken statik obstacle özellikle dahil edildiyse,
        # obstacles[0] statik obstacle kabul edilir.
        is_static_visible = (not self.use_unknown_obstacle) or self.include_static_obstacle

        for i, obs in enumerate(obstacles):
            ox, oy, radius = obs
            inflated = radius + self.robot_radius + self.safety_margin

            if is_static_visible and i == 0:
                # YAML'dan gelen statik obstacle
                border_color = (0, 0, 255)      # red
                fill_color = (0, 0, 255)        # red
                lbl = "Static"
            else:
                # ToF ile sonradan algılanan obstacle
                border_color = (0, 165, 255)    # orange
                fill_color = (0, 165, 255)      # orange
                lbl = "ToF Mapped"

            self.draw_circle_world(canvas, ox, oy, inflated, border_color, 2)
            self.draw_circle_world(canvas, ox, oy, radius, fill_color, -1)

            op = self.world_to_canvas(ox, oy)
            cv2.putText(
                canvas,
                lbl,
                (op[0] + 10, op[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                border_color,
                1,
                cv2.LINE_AA
            )

        # Robot Çizimi
        rx, ry, rtheta = self.state
        robot_px = self.world_to_canvas(rx, ry)
        self.draw_circle_world(canvas, rx, ry, self.robot_radius, (0, 0, 0), -1)
        self.draw_circle_world(canvas, rx, ry, self.sensing_radius, (0, 165, 255), 2)

        arrow_len = 0.15
        arrow_end = self.world_to_canvas(
            rx + arrow_len * math.cos(rtheta),
            ry + arrow_len * math.sin(rtheta)
        )
        cv2.arrowedLine(canvas, robot_px, arrow_end, (0, 0, 0), 2, tipLength=0.3)
        cv2.putText(canvas, "Robot", (robot_px[0] + 10, robot_px[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # Start & Goal
        start_px = self.world_to_canvas(self.start[0], self.start[1])
        goal_px = self.world_to_canvas(self.goal[0], self.goal[1])
        cv2.circle(canvas, start_px, 8, (0, 180, 0), -1)
        cv2.putText(canvas, "Start", (start_px[0] + 10, start_px[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 150, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, goal_px, 8, (0, 0, 255), -1)
        cv2.putText(canvas, "Goal", (goal_px[0] + 10, goal_px[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        # Durum Metinleri ve Lejant
        cv2.putText(canvas, f"x={rx:.2f}, y={ry:.2f}, theta={math.degrees(rtheta):.1f} deg",
                    (20, canvas_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        
        obs_text = f"Total Obstacles: {len(obstacles)}"
        cv2.putText(canvas, obs_text, (20, canvas_h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(
            canvas,
            "Blue: current A* path",
            (canvas_w - 230, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1
        )

        cv2.putText(
            canvas,
            "Gray: DWA samples",
            (canvas_w - 230, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (120, 120, 120),
            1
        )

        cv2.putText(
            canvas,
            "Green: chosen DWA",
            (canvas_w - 230, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 160, 0),
            1
        )

        if is_static_visible:
            cv2.putText(
                canvas,
                "Red: static obstacle",
                (canvas_w - 230, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1
            )
            cv2.putText(
                canvas,
                "Orange: ToF mapped obstacle",
                (canvas_w - 230, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 165, 255),
                1
            )
        else:
            cv2.putText(
                canvas,
                "Orange: ToF mapped obstacle",
                (canvas_w - 230, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 165, 255),
                1
            )

        self.publish_visualization_image(canvas)

    def run(self):
        rate = rospy.Rate(self.loop_rate)
        dt = 1.0 / self.loop_rate

        while not rospy.is_shutdown():
            if self.reached_goal():
                rospy.loginfo("GOAL REACHED! Assignment 4 Complete.")
                self.stop_robot()
                break

            new_obstacle_detected = self.detect_new_obstacle_from_tof()

            if new_obstacle_detected:
                self.stop_robot()

                new_path = self.run_astar(
                    start_xy=(self.state[0], self.state[1])
                )

                if len(new_path) > 0:
                    self.global_path = new_path
                    rospy.loginfo(
                        f"A* replanned path with {len(self.global_path)} waypoints."
                    )
                else:
                    rospy.logwarn(
                        "A* could not find new path after obstacle detection. Keeping old path."
                    )

            control, best_trajectory, all_trajectories = self.dwa_control()
            v, w = control

            self.current_v = v
            self.current_w = w

            sent_v, sent_w = self.publish_cmd(v, w)

            if self.cfg.get("simulation", {}).get("use_internal_pose", True):
                self.update_internal_pose(sent_v, sent_w, dt)

            self.visualize(best_trajectory, all_trajectories)

            rate.sleep()

        self.stop_robot()


# DÜZELTME 5: Python'un çalışması için gereken dunder formatı eklendi
if __name__ == "__main__":
    try:
        node = AStarDWAPlanner()
        node.run()
    except rospy.ROSInterruptException:
        pass