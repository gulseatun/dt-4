#!/usr/bin/env python3

import math
import heapq
import yaml
import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Range, CompressedImage
from duckietown_msgs.msg import Twist2DStamped

# ==========================================
# DEFAULT CONFIGURATION
# ==========================================
DEFAULT_CONFIG = {
    "ros": {"vehicle_name": "bear"},
    "map": {"width": 1.5, "height": 1.5, "resolution": 0.05},
    "start": {"x": 0.1, "y": 0.1, "theta": math.pi/4},
    "goal": {"x": 1.3, "y": 1.3},
    "obstacle": {"x": 0.75, "y": 0.75, "radius": 0.05},
    "robot": {
        "radius": 0.1, 
        "safety_margin": 0.05, 
        "sensing_radius": 0.4,
        "wheel_base": 0.108,
        "wheel_cmd_gain": 1.0,
        "max_wheel_cmd": 1.0
    },
    "dwa": {
        "dt": 0.1,
        "horizon": 2.0,
        "v_samples": 11,
        "w_samples": 21,
        "min_v": 0.0,
        "max_v": 0.22,
        "min_w": -2.0,
        "max_w": 2.0,
        "max_accel_v": 0.2,
        "max_accel_w": 1.5
    },
    "cost": {
        "path_weight": 1.0,
        "goal_weight": 1.0, 
        "obstacle_weight": 2.5, 
        "heading_weight": 0.8
    },
    "bonus": {"unknown_obstacle": True},
    "control": {"goal_tolerance": 0.1, "loop_rate": 10},
    "simulation": {"use_internal_pose": True}
}

class AStarDWAPlanner:
    def __init__(self):
        rospy.init_node("astar_dwa_node", anonymous=False)
        rospy.on_shutdown(self.stop_robot)

        config_path = rospy.get_param("~config_path", "")
        if config_path == "":
            rospy.logwarn("No config_path provided. Using internal DEFAULT_CONFIG.")
            self.cfg = DEFAULT_CONFIG
        else:
            try:
                with open(config_path, "r") as f:
                    self.cfg = yaml.safe_load(f)
            except Exception as e:
                rospy.logerr(f"Failed to load YAML. Using defaults. Error: {e}")
                self.cfg = DEFAULT_CONFIG

        self.vehicle_name = self.cfg["ros"]["vehicle_name"]
        self.wheels_topic = f"/{self.vehicle_name}/car_cmd_switch_node/cmd"
        self.tof_topic = f"/{self.vehicle_name}/front_center_tof_driver_node/range"
        self.viz_topic = f"/{self.vehicle_name}/assignment4_viz/compressed"

        self.cmd_pub = rospy.Publisher(self.wheels_topic, Twist2DStamped, queue_size=1)
        self.viz_pub = rospy.Publisher(self.viz_topic, CompressedImage, queue_size=1)

        self.use_unknown_obstacle = self.cfg["bonus"]["unknown_obstacle"]
        self.latest_range = None
        
        self.mapped_dynamic_obstacles = []

        if self.use_unknown_obstacle:
            rospy.Subscriber(self.tof_topic, Range, self.tof_callback)

        self.map_width = self.cfg["map"]["width"]
        self.map_height = self.cfg["map"]["height"]
        self.resolution = self.cfg["map"]["resolution"]

        self.grid_w = int(self.map_width / self.resolution)
        self.grid_h = int(self.map_height / self.resolution)

        self.start = (self.cfg["start"]["x"], self.cfg["start"]["y"], self.cfg["start"]["theta"])
        self.goal = (self.cfg["goal"]["x"], self.cfg["goal"]["y"])
        
        self.state = np.array([self.start[0], self.start[1], self.start[2]], dtype=float)
        self.current_v = 0.0
        self.current_w = 0.0

        self.static_obstacle = (self.cfg["obstacle"]["x"], self.cfg["obstacle"]["y"], self.cfg["obstacle"]["radius"])
        self.robot_radius = self.cfg.get("robot", {}).get("radius", 0.1)
        self.safety_margin = self.cfg.get("robot", {}).get("safety_margin", 0.05)
        self.sensing_radius = self.cfg.get("robot", {}).get("sensing_radius", 0.4)

        rospy.loginfo("Calculating Initial Global Path with A*...")
        self.global_path = self.run_astar()
        if len(self.global_path) == 0:
            rospy.logerr("A* could not find a path.")
            raise RuntimeError("No path found")

        self.goal_tolerance = self.cfg.get("control", {}).get("goal_tolerance", 0.1)
        self.loop_rate = self.cfg.get("control", {}).get("loop_rate", 10)

    def tof_callback(self, msg):
        self.latest_range = msg.range

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

    def distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def get_all_obstacles(self):
        """Döndürülen listede hem YAML'dan gelen statik hem de ToF ile bulunan dinamik engeller vardır."""
        return self.mapped_dynamic_obstacles

    def update_dynamic_obstacles(self):
        """Yeni bir engel tespit edilirse True döndürerek A* Replan mekanizmasını tetikler."""
        replan_needed = False
        if self.use_unknown_obstacle and self.latest_range is not None:
            r = self.latest_range
            if 0.02 < r < self.sensing_radius:
                ox = self.state[0] + r * math.cos(self.state[2])
                oy = self.state[1] + r * math.sin(self.state[2])
                
                is_new = True
                for mx, my, _ in self.mapped_dynamic_obstacles:
                    if self.distance((ox, oy), (mx, my)) < 0.15: 
                        is_new = False
                        break
                        
                if is_new:
                    self.mapped_dynamic_obstacles.append((ox, oy, self.cfg["obstacle"]["radius"]))
                    rospy.logwarn(f"!!! YENI ENGEL TESPIT EDILDI at: {ox:.2f}, {oy:.2f} !!!")
                    replan_needed = True
            
            self.latest_range = None # Sensör verisini işledikten sonra temizle
            
        return replan_needed

    def run_astar(self, start_pos=None):
        """A* Algoritması artık engelleri dinamik olarak hesaba katarak rotayı baştan çizer."""
        if start_pos is None:
            start_pos = (self.start[0], self.start[1])

        start_grid = self.world_to_grid(start_pos[0], start_pos[1])
        goal_grid = self.world_to_grid(self.goal[0], self.goal[1])

        moves = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
            (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
        ]

        open_set = []
        heapq.heappush(open_set, (0.0, start_grid))
        came_from = {}
        g_score = {start_grid: 0.0}

        # Güncel tüm engelleri al (Statik + Dinamik)
        all_obs = self.get_all_obstacles()

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_grid:
                return self.reconstruct_path(came_from, current)

            for dx, dy, move_cost in moves:
                nx, ny = current[0] + dx, current[1] + dy

                if nx < 0 or ny < 0 or nx >= self.grid_w or ny >= self.grid_h:
                    continue
                
                wx, wy = self.grid_to_world(nx, ny)
                collision = False
                
                # A* rotayı çizerken tüm engellerden güvenli mesafede uzak durur
                for ox, oy, orad in all_obs:
                    obs_inflated = orad + self.robot_radius + self.safety_margin
                    if self.distance((wx, wy), (ox, oy)) <= obs_inflated:
                        collision = True
                        break
                
                if collision:
                    continue

                neighbor = (nx, ny)
                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.distance(self.grid_to_world(nx, ny), self.goal)
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

    def get_lookahead_point(self, x, y):
        closest_idx = 0
        min_d = float('inf')
        for i, p in enumerate(self.global_path):
            d = self.distance((x, y), p)
            if d < min_d:
                min_d = d
                closest_idx = i
                
        lookahead_distance = 6 
        target_idx = min(len(self.global_path) - 1, closest_idx + lookahead_distance)
        return self.global_path[target_idx]

    def heading_cost(self, trajectory):
        end = trajectory[-1]
        x, y, theta = end
        
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
        obstacles = self.get_all_obstacles() # Tüm engeller hesaba katılır

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

    def publish_wheels(self, v, w):
        robot_cfg = self.cfg.get("robot", {})
        max_cmd = robot_cfg.get("max_wheel_cmd", 1.0)
        v_clipped = max(-max_cmd, min(max_cmd, v))
        
        msg = Twist2DStamped()
        msg.header.stamp = rospy.Time.now()
        msg.v = v_clipped
        msg.omega = w
        
        self.cmd_pub.publish(msg)

    def update_internal_pose(self, v, w, dt):
        self.state[0] += v * math.cos(self.state[2]) * dt
        self.state[1] += v * math.sin(self.state[2]) * dt
        self.state[2] += w * dt
        self.state[2] = self.normalize_angle(self.state[2])

    def stop_robot(self):
        self.publish_wheels(0.0, 0.0)
        rospy.loginfo("Motors stopped.")

    def reached_goal(self):
        return self.distance((self.state[0], self.state[1]), self.goal) <= self.goal_tolerance

    def publish_cv2_visualization(self, best_traj, all_trajs):
        w_px, h_px = 600, 600
        canvas = np.ones((h_px, w_px, 3), dtype=np.uint8) * 255

        def to_px(x, y):
            px = int((x / self.map_width) * w_px)
            py = int(h_px - (y / self.map_height) * h_px)
            return px, py

        # Draw Grid
        grid_step = int((self.resolution / self.map_width) * w_px)
        for i in range(0, w_px, grid_step):
            cv2.line(canvas, (i, 0), (i, h_px), (240, 240, 240), 1)
            cv2.line(canvas, (0, i), (w_px, i), (240, 240, 240), 1)

        # Draw Global Path
        for i in range(len(self.global_path) - 1):
            p1 = to_px(self.global_path[i][0], self.global_path[i][1])
            p2 = to_px(self.global_path[i+1][0], self.global_path[i+1][1])
            cv2.line(canvas, p1, p2, (255, 0, 0), 2)

        # Draw Obstacles 
        obstacles = self.get_all_obstacles()
        for i, obs in enumerate(obstacles):
            ox, oy, rad = obs
            px, py = to_px(ox, oy)
            inflated_px = int(((rad + self.robot_radius + self.safety_margin) / self.map_width) * w_px)
            rad_px = int((rad / self.map_width) * w_px)
            
            color = (0, 0, 255) if i == 0 else (0, 165, 255) 
            cv2.circle(canvas, (px, py), inflated_px, (200, 200, 255), 1) 
            cv2.circle(canvas, (px, py), rad_px, color, -1) 

        # Draw Sensing Area
        rx, ry = to_px(self.state[0], self.state[1])
        sense_px = int((self.sensing_radius / self.map_width) * w_px)
        cv2.circle(canvas, (rx, ry), sense_px, (0, 200, 255), 1, lineType=cv2.LINE_AA)

        # Draw DWA Trajectories
        for traj in all_trajs:
            if not traj: continue
            for i in range(len(traj) - 1):
                p1 = to_px(traj[i][0], traj[i][1])
                p2 = to_px(traj[i+1][0], traj[i+1][1])
                cv2.line(canvas, p1, p2, (200, 200, 200), 1)

        if best_traj:
            for i in range(len(best_traj) - 1):
                p1 = to_px(best_traj[i][0], best_traj[i][1])
                p2 = to_px(best_traj[i+1][0], best_traj[i+1][1])
                cv2.line(canvas, p1, p2, (0, 150, 0), 2)

        # Draw Robot
        robot_px = int((self.robot_radius / self.map_width) * w_px)
        cv2.circle(canvas, (rx, ry), robot_px, (0, 0, 0), 2)
        hx = rx + int(math.cos(self.state[2]) * robot_px)
        hy = ry - int(math.sin(self.state[2]) * robot_px)
        cv2.line(canvas, (rx, ry), (hx, hy), (0, 0, 255), 2)

        # Draw Goal
        gx, gy = to_px(self.goal[0], self.goal[1])
        cv2.drawMarker(canvas, (gx, gy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)

        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode(".jpg", canvas)[1]).tobytes()
        self.viz_pub.publish(msg)

    def run(self):
        rate = rospy.Rate(self.loop_rate)
        dt = 1.0 / self.loop_rate

        while not rospy.is_shutdown():
            if self.reached_goal():
                rospy.loginfo("GOAL REACHED! Assignment 4 Complete.")
                self.stop_robot()
                break

            # 1. ToF ile yeni engel tespiti yapıldı mı kontrol et
            if self.update_dynamic_obstacles():
                rospy.logwarn("A* Re-planning: Yeni engel etrafindan optimal rota ciziliyor...")
                new_path = self.run_astar(start_pos=(self.state[0], self.state[1]))
                if len(new_path) > 0:
                    self.global_path = new_path
                else:
                    rospy.logerr("A* engelin etrafindan kacacak bir rota bulamadi!")

            # 2. DWA'yı yeni (veya mevcut) rotaya göre çalıştır
            control, best_trajectory, all_trajectories = self.dwa_control()
            v, w = control

            self.current_v = v
            self.current_w = w

            self.publish_wheels(v, w)

            if self.cfg["simulation"]["use_internal_pose"]:
                self.update_internal_pose(v, w, dt)

            self.publish_cv2_visualization(best_trajectory, all_trajectories)

            rate.sleep()

        self.stop_robot()

if __name__ == "__main__":
    try:
        node = AStarDWAPlanner()
        node.run()
    except rospy.ROSInterruptException:
        pass