#!/usr/bin/env python3

import math
import heapq
import yaml
import rospy
import numpy as np
import matplotlib.pyplot as plt

from sensor_msgs.msg import Range
from duckietown_msgs.msg import WheelsCmdStamped


class AStarDWAPlanner:
    def __init__(self):
        rospy.init_node("astar_dwa_node", anonymous=False)

        config_path = rospy.get_param("~config_path", "")
        if config_path == "":
            rospy.logerr("No config_path provided. Use _config_path:=/path/to/planner.yaml")
            raise RuntimeError("Missing config_path")

        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        #CMD_TOPIC = f"/{ROBOT_NAME}/car_cmd_switch_node/cmd"
        self.vehicle_name = self.cfg["ros"]["vehicle_name"]
        self.wheels_topic = f"/{self.vehicle_name}/car_cmd_switch_node/cmd"
        self.tof_topic = f"/{self.vehicle_name}/front_center_tof_driver_node/range"

        self.cmd_pub = rospy.Publisher(
            self.wheels_topic,
            WheelsCmdStamped,
            queue_size=1
        )

        self.use_unknown_obstacle = self.cfg["bonus"]["unknown_obstacle"]
        self.detected_obstacle = None
        self.latest_range = None

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

        self.robot_radius = self.cfg["robot"]["radius"]
        self.safety_margin = self.cfg["robot"]["safety_margin"]
        self.sensing_radius = self.cfg["robot"]["sensing_radius"]

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

        if self.enable_visualization:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(7, 7))

        rospy.loginfo("A* + DWA planner initialized.")
        rospy.loginfo(f"Publishing wheels command to: {self.wheels_topic}")

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

    def heuristic(self, a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def run_astar(self):
        start_grid = self.world_to_grid(self.start[0], self.start[1])
        goal_grid = self.world_to_grid(self.goal[0], self.goal[1])

        moves = [
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
            (1, 1, math.sqrt(2)),
            (1, -1, math.sqrt(2)),
            (-1, 1, math.sqrt(2)),
            (-1, -1, math.sqrt(2)),
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

        world_path = []
        for gx, gy in path:
            world_path.append(self.grid_to_world(gx, gy))

        return world_path

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def distance(self, p1, p2):
        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2
        )

    def get_active_obstacle(self):
        if self.use_unknown_obstacle:
            if self.latest_range is not None:
                if self.latest_range < self.sensing_radius:
                    ox = self.state[0] + self.latest_range * math.cos(self.state[2])
                    oy = self.state[1] + self.latest_range * math.sin(self.state[2])
                    self.detected_obstacle = (
                        ox,
                        oy,
                        self.cfg["obstacle"]["radius"]
                    )

            if self.detected_obstacle is not None:
                return self.detected_obstacle

            return None

        return self.obstacle

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

    def trajectory_collision(self, trajectory, obstacle):
        if obstacle is None:
            return False

        ox, oy, radius = obstacle
        inflated = radius + self.robot_radius + self.safety_margin

        for x, y, _ in trajectory:
            if x < 0 or y < 0 or x > self.map_width or y > self.map_height:
                return True

            d = self.distance((x, y), (ox, oy))
            if d <= inflated:
                return True

        return False

    def path_cost(self, trajectory):
        end = trajectory[-1]
        end_xy = (end[0], end[1])

        min_dist = float("inf")

        for p in self.global_path:
            d = self.distance(end_xy, p)
            if d < min_dist:
                min_dist = d

        return min_dist

    def goal_cost(self, trajectory):
        end = trajectory[-1]
        return self.distance((end[0], end[1]), self.goal)

    def obstacle_cost(self, trajectory, obstacle):
        if obstacle is None:
            return 0.0

        ox, oy, radius = obstacle
        inflated = radius + self.robot_radius + self.safety_margin

        min_dist = float("inf")

        for x, y, _ in trajectory:
            d = self.distance((x, y), (ox, oy))
            if d < min_dist:
                min_dist = d

        if min_dist <= inflated:
            return float("inf")

        return 1.0 / (min_dist - inflated + 1e-6)

    def heading_cost(self, trajectory):
        end = trajectory[-1]
        x, y, theta = end

        desired_theta = math.atan2(
            self.goal[1] - y,
            self.goal[0] - x
        )

        error = self.normalize_angle(desired_theta - theta)
        return abs(error)

    def dynamic_window(self):
        dwa = self.cfg["dwa"]

        dt = dwa["dt"]

        min_v = max(
            dwa["min_v"],
            self.current_v - dwa["max_accel_v"] * dt
        )

        max_v = min(
            dwa["max_v"],
            self.current_v + dwa["max_accel_v"] * dt
        )

        min_w = max(
            dwa["min_w"],
            self.current_w - dwa["max_accel_w"] * dt
        )

        max_w = min(
            dwa["max_w"],
            self.current_w + dwa["max_accel_w"] * dt
        )

        return min_v, max_v, min_w, max_w

    def dwa_control(self):
        dwa = self.cfg["dwa"]
        cost_cfg = self.cfg["cost"]

        obstacle = self.get_active_obstacle()

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

                if self.trajectory_collision(trajectory, obstacle):
                    continue

                p_cost = self.path_cost(trajectory)
                g_cost = self.goal_cost(trajectory)
                o_cost = self.obstacle_cost(trajectory, obstacle)
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
        robot_cfg = self.cfg["robot"]

        wheel_base = robot_cfg["wheel_base"]
        wheel_gain = robot_cfg["wheel_cmd_gain"]
        max_cmd = robot_cfg["max_wheel_cmd"]

        vel_left = v - (w * wheel_base / 2.0)
        vel_right = v + (w * wheel_base / 2.0)

        cmd_left = wheel_gain * vel_left
        cmd_right = wheel_gain * vel_right

        cmd_left = max(-max_cmd, min(max_cmd, cmd_left))
        cmd_right = max(-max_cmd, min(max_cmd, cmd_right))

        msg = WheelsCmdStamped()
        msg.header.stamp = rospy.Time.now()
        msg.vel_left = cmd_left
        msg.vel_right = cmd_right

        self.cmd_pub.publish(msg)

    def update_internal_pose(self, v, w, dt):
        self.state[0] += v * math.cos(self.state[2]) * dt
        self.state[1] += v * math.sin(self.state[2]) * dt
        self.state[2] += w * dt
        self.state[2] = self.normalize_angle(self.state[2])

    def stop_robot(self):
        self.publish_wheels(0.0, 0.0)

    def reached_goal(self):
        d = self.distance((self.state[0], self.state[1]), self.goal)
        return d <= self.goal_tolerance

    def visualize(self, best_trajectory, all_trajectories):
        if not self.enable_visualization:
            return

        self.ax.clear()
        self.ax.set_xlim(0, self.map_width)
        self.ax.set_ylim(0, self.map_height)
        self.ax.set_aspect("equal")
        self.ax.grid(True)

        path_x = [p[0] for p in self.global_path]
        path_y = [p[1] for p in self.global_path]
        self.ax.plot(path_x, path_y, "b-", linewidth=2, label="A* Global Path")

        for traj in all_trajectories:
            if len(traj) == 0:
                continue
            tx = [p[0] for p in traj]
            ty = [p[1] for p in traj]
            self.ax.plot(tx, ty, color="gray", alpha=0.25, linewidth=0.8)

        if len(best_trajectory) > 0:
            bx = [p[0] for p in best_trajectory]
            by = [p[1] for p in best_trajectory]
            self.ax.plot(bx, by, "g-", linewidth=3, label="Chosen DWA Trajectory")

        obstacle = self.get_active_obstacle()

        if obstacle is not None:
            ox, oy, radius = obstacle
            inflated = radius + self.robot_radius + self.safety_margin

            obstacle_circle = plt.Circle(
                (ox, oy),
                radius,
                color="red",
                alpha=0.8,
                label="Obstacle"
            )
            self.ax.add_patch(obstacle_circle)

            inflated_circle = plt.Circle(
                (ox, oy),
                inflated,
                color="red",
                fill=False,
                linestyle="--",
                linewidth=2,
                label="Inflated Boundary"
            )
            self.ax.add_patch(inflated_circle)

        robot_circle = plt.Circle(
            (self.state[0], self.state[1]),
            self.robot_radius,
            color="black",
            alpha=0.8,
            label="Robot"
        )
        self.ax.add_patch(robot_circle)

        sensing_circle = plt.Circle(
            (self.state[0], self.state[1]),
            self.sensing_radius,
            color="orange",
            fill=False,
            linestyle=":",
            linewidth=2,
            label="Sensing Area"
        )
        self.ax.add_patch(sensing_circle)

        arrow_len = 0.12
        self.ax.arrow(
            self.state[0],
            self.state[1],
            arrow_len * math.cos(self.state[2]),
            arrow_len * math.sin(self.state[2]),
            head_width=0.04,
            color="black"
        )

        self.ax.plot(self.start[0], self.start[1], "go", markersize=8, label="Start")
        self.ax.plot(self.goal[0], self.goal[1], "rx", markersize=10, label="Goal")

        self.ax.set_title("A* Global Path + DWA Local Obstacle Avoidance")
        self.ax.legend(loc="upper right", fontsize=8)

        plt.pause(0.001)

    def run(self):
        rate = rospy.Rate(self.loop_rate)
        dt = 1.0 / self.loop_rate

        while not rospy.is_shutdown():
            if self.reached_goal():
                rospy.loginfo("Goal reached.")
                self.stop_robot()
                break

            control, best_trajectory, all_trajectories = self.dwa_control()
            v, w = control

            self.current_v = v
            self.current_w = w

            self.publish_wheels(v, w)

            if self.cfg["simulation"]["use_internal_pose"]:
                self.update_internal_pose(v, w, dt)

            self.visualize(best_trajectory, all_trajectories)

            rate.sleep()

        self.stop_robot()

        if self.enable_visualization:
            plt.ioff()
            plt.show()


if __name__ == "__main__":
    try:
        node = AStarDWAPlanner()
        node.run()
    except rospy.ROSInterruptException:
        pass