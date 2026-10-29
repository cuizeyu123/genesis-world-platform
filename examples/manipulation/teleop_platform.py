"""Real-time web teleoperation platform for the Franka Panda.

A local HTTP server exposes a browser console with a live camera stream and a control
panel. The simulation owns the main thread: it integrates the latest command, steps the
scene, renders the camera and publishes a JPEG frame. The HTTP threads only move data, so
Genesis is never called from a worker thread.

Features
--------
- Cartesian keyboard teleoperation in the world frame (WASD, R/F for height, Q/E for yaw)
  with speed scaling and a workspace clamp.
- Joint-space mode with one slider per arm joint.
- Gripper open/close, home pose, and scene reset.
- Session recording to MP4, MCAP, or LeRobot v2.1 datasets.
- Live MJPEG video, joint/end-effector state, FPS and command round-trip time.
- A 0.7 s command watchdog stops the arm when the browser disconnects.
- Optional Genesis viewer next to the browser console.

Usage
-----
    python examples/manipulation/teleop_platform.py
    python examples/manipulation/teleop_platform.py --port 8010 --vis
    python examples/manipulation/teleop_platform.py --host 0.0.0.0

Then open http://127.0.0.1:8010.
"""

import argparse
import asyncio
import base64
from collections import deque
from dataclasses import dataclass, replace
import importlib.util
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import threading
import time
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel, Field
import uvicorn
if __package__:
    from .teleop_recording import create_recording_session
    from .teleop_skills import PROGRAMS, interpret_natural_program
    from .teleop_validation import validate_agent_plan, validate_move_delta
else:
    from teleop_recording import create_recording_session
    from teleop_skills import PROGRAMS, interpret_natural_program
    from teleop_validation import validate_agent_plan, validate_move_delta
import genesis as gs
from genesis.utils.geom import inv_T, pos_lookat_up_to_T, trans_quat_to_T, transform_quat_by_quat
from genesis.utils.misc import tensor_to_array


ARM_DOFS = np.arange(7)
FINGER_DOF_IDX = np.arange(7, 9)
GRIPPER_OPEN = 0.04
HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, GRIPPER_OPEN, GRIPPER_OPEN])
CAMERA_POS = np.array([2.0, -1.15, 1.05])
CAMERA_LOOKAT = np.array([0.5, -0.05, 0.18])
CAMERA_FOV = 33.0
WORKSPACE_LOW = np.array([0.24, -0.44, 0.03])
WORKSPACE_HIGH = np.array([0.86, 0.44, 0.76])
# The shoulder sits at this height; keeping the hand inside a sphere around it avoids the
# fully-extended configurations where IK has no solution and gravity torque is largest.
SHOULDER = np.array([0.0, 0.0, 0.333])
REACH_RADIUS = 0.70
CONTROL_HZ = 30.0
SERVO_DT = 1.0 / CONTROL_HZ
FIRST_PERSON_RES = (320, 240)
FIRST_PERSON_HZ = 15.0
HOME_DURATION_S = 2.4
AGENT_MOVE_MAX_M = 0.03
AGENT_MOVE_DURATION_S = 0.9
COLLISION_FORCE_THRESHOLD_N = 18.0
VLM_URL = os.environ.get("GENESIS_TELEOP_VLM_URL", "http://127.0.0.1:11434/api/chat")
VLM_MODEL = os.environ.get("GENESIS_TELEOP_VLM_MODEL", "").strip()
try:
    VLM_TIMEOUT_S = float(os.environ.get("GENESIS_TELEOP_VLM_TIMEOUT", "90"))
except ValueError:
    VLM_TIMEOUT_S = 90.0
if not math.isfinite(VLM_TIMEOUT_S) or VLM_TIMEOUT_S <= 0.0:
    VLM_TIMEOUT_S = 90.0
ROBOT_KP = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100])
ROBOT_KV = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10])
ROBOT_FORCE_MIN = np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100])
ROBOT_FORCE_MAX = np.array([87, 87, 87, 87, 12, 12, 12, 100, 100])
LOG_PATH = Path(
    os.environ.get(
        "GENESIS_TELEOP_LOG_PATH",
        Path(__file__).resolve().parents[2] / "out" / "teleop" / "teleop.log",
    )
)
LOGGER = logging.getLogger("teleop")


def configure_logging(level="INFO"):
    """Write detailed rotating logs to disk and concise operational logs to the console."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    for handler in LOGGER.handlers[:]:
        LOGGER.removeHandler(handler)
        handler.close()
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(threadName)s] %(name)s:%(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = RotatingFileHandler(
        LOG_PATH,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(console_handler)
    uvicorn_logger = logging.getLogger("uvicorn.error")
    uvicorn_logger.handlers.clear()
    uvicorn_logger.setLevel(logging.INFO)
    uvicorn_logger.propagate = False
    uvicorn_logger.addHandler(file_handler)
    uvicorn_logger.addHandler(console_handler)
    return LOG_PATH


@dataclass
class TeleopCommand:
    keys: tuple[str, ...] = ()
    gripper: float = GRIPPER_OPEN
    speed: float = 0.15
    yaw_speed: float = 0.7
    mode: str = "cartesian"
    joints: tuple[float, ...] = ()
    action: str = ""
    action_params: dict | None = None
    enabled: bool = True


class CommandPayload(BaseModel):
    keys: list[str] = Field(default_factory=list, max_length=8)
    gripper: float = Field(default=GRIPPER_OPEN, ge=0.0, le=GRIPPER_OPEN)
    speed: float = Field(default=0.15, ge=0.01, le=0.5)
    yaw_speed: float = Field(default=0.7, ge=0.05, le=2.5)
    mode: Literal["cartesian", "joint"] = "cartesian"
    joints: list[float] = Field(default_factory=list, max_length=7)
    enabled: bool = True


class ActionPayload(BaseModel):
    action: Literal["home", "reset_scene", "toggle_recording"]


class RecordingStartPayload(BaseModel):
    format: Literal["mp4", "mcap", "lerobot"]
    task: str = Field(default="Teleoperation session", min_length=1, max_length=256)


class SensorModePayload(BaseModel):
    mode: Literal["rgb", "depth", "pointcloud", "segmentation"]


class CameraControlPayload(BaseModel):
    orbit_x: float = Field(default=0.0, ge=-2.0, le=2.0)
    orbit_y: float = Field(default=0.0, ge=-2.0, le=2.0)
    pan_x: float = Field(default=0.0, ge=-2.0, le=2.0)
    pan_y: float = Field(default=0.0, ge=-2.0, le=2.0)
    zoom: float = Field(default=0.0, ge=-2.0, le=2.0)
    reset: bool = False


class AgentAnalyzePayload(BaseModel):
    task: str = Field(min_length=1, max_length=500)


class AgentPlan(BaseModel):
    summary: str = Field(min_length=1, max_length=120)
    target_object: str = Field(default="", max_length=80)
    action: Literal["move", "open_gripper", "close_gripper", "observe", "complete"]
    delta_m: list[float] = Field(default_factory=list, max_length=3)
    reason: str = Field(min_length=1, max_length=180)
    confidence: float = Field(ge=0.0, le=1.0)


class AgentExecutePayload(BaseModel):
    suggestion_id: str = Field(min_length=1, max_length=64)


class FrankaTeleopPlatform:
    def __init__(self, args):
        self._args = args
        self._running = True
        self._lock = threading.Lock()
        self._command = TeleopCommand()
        self._pending_actions = deque()
        self._api_server = None
        self._api_thread = None
        self._frame = None
        self._camera_bgr = None
        self._state = {}
        self._recorder = None
        self._recording_status = "idle"
        self._recording_format = ""
        self._recording_task = ""
        self._recording_path = ""
        self._recording_frames = 0
        self._recording_error = ""
        self._last_action_joints = HOME_QPOS[:7].copy()
        self._last_action_gripper = GRIPPER_OPEN
        self._clamped = False
        self._control_enabled = True
        self._agent_trajectory = None
        self._agent_gripper_target = None
        self._agent_move_failed = False
        self._program_task = None
        self._program_status = {"program": "", "label": "", "state": "idle", "step": 0, "total": 0, "message": ""}
        camera_offset = CAMERA_POS - CAMERA_LOOKAT
        self._camera_target = CAMERA_LOOKAT.copy()
        self._camera_distance = float(np.linalg.norm(camera_offset))
        self._camera_azimuth = math.atan2(camera_offset[1], camera_offset[0])
        self._camera_elevation = math.asin(camera_offset[2] / self._camera_distance)
        self._pending_camera = np.zeros(5, dtype=np.float64)
        self._pending_camera_reset = False
        self._camera_snapshot = {
            "position": CAMERA_POS.round(4).tolist(),
            "target": CAMERA_LOOKAT.round(4).tolist(),
            "distance": round(self._camera_distance, 4),
        }
        self._last_frame_at = None
        self._jpeg_failures = 0
        self._last_jpeg_warning_at = 0.0
        self._last_camera_command_at = None
        self._last_camera_applied_at = None
        self._camera_updates_since_log = 0
        self._last_camera_log_at = 0.0
        self._last_runtime_log_at = time.perf_counter()
        self._simulation_loop_active = False
        self._sensor_mode = "rgb"
        self._first_person_frame = None
        self._first_person_frame_at = None
        self._last_first_person_render_at = 0.0
        self._last_first_person_error_log_at = 0.0
        self._home_trajectory = None
        self._collision_mode = False
        self._collision_reset_pending = False
        self._collision_force_n = 0.0
        self._collision_position = None
        self._collision_link = ""
        self._last_command_time = time.monotonic()
        self._start_api_server()

        gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01),
            viewer_options=gs.options.ViewerOptions(
                res=(1280, 960),
                camera_pos=CAMERA_POS,
                camera_lookat=CAMERA_LOOKAT,
                camera_fov=CAMERA_FOV,
                realtime_factor=1.0,
            ),
            show_viewer=args.vis,
        )
        self.scene.add_entity(
            morph=gs.morphs.Plane(
                pos=(0.0, 0.0, -0.04),
            ),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Rough(
                color=(0.16, 0.17, 0.19),
            ),
        )
        self.scene.add_entity(
            morph=gs.morphs.Box(
                size=(1.3, 1.0, 0.04),
                pos=(0.45, 0.0, -0.02),
                fixed=True,
            ),
            material=gs.materials.Rigid(
                friction=1.0,
            ),
            surface=gs.surfaces.Rough(
                color=(0.44, 0.33, 0.24),
            ),
        )
        self.cubes = [
            self.scene.add_entity(
                morph=gs.morphs.Box(
                    size=(0.05, 0.05, 0.05),
                    pos=(0.55, -0.12, 0.025),
                ),
                material=gs.materials.Rigid(
                    rho=400,
                    friction=1.0,
                ),
                surface=gs.surfaces.Rough(
                    color=(0.20, 0.45, 0.85),
                ),
            ),
            self.scene.add_entity(
                morph=gs.morphs.Box(
                    size=(0.04, 0.04, 0.04),
                    pos=(0.62, 0.12, 0.02),
                ),
                material=gs.materials.Rigid(
                    rho=400,
                    friction=1.0,
                ),
                surface=gs.surfaces.Rough(
                    color=(0.30, 0.70, 0.35),
                ),
            ),
        ]
        self.tray = self.scene.add_entity(
            morph=gs.morphs.Box(
                size=(0.24, 0.24, 0.01),
                pos=(0.36, -0.34, 0.005),
                fixed=True,
            ),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Rough(
                color=(0.80, 0.80, 0.82),
            ),
        )
        self.franka = self.scene.add_entity(
            morph=gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda.xml",
            ),
            # Gravity compensation turns the arm into a near-weightless teleoperation device:
            # the hand tracks the commanded pose without the ~1 cm sag that gravity causes.
            material=gs.materials.Rigid(
                gravity_compensation=1.0,
            ),
        )
        self.camera = self.scene.add_camera(
            res=(640, 480),
            pos=CAMERA_POS,
            lookat=CAMERA_LOOKAT,
            fov=CAMERA_FOV,
            GUI=False,
        )
        self.first_person_camera = self.scene.add_camera(
            res=FIRST_PERSON_RES,
            pos=CAMERA_POS,
            lookat=CAMERA_LOOKAT,
            fov=78.0,
            GUI=False,
            near=0.03,
            far=8.0,
        )
        self.scene.build()

        self.hand = self.franka.get_link("hand")
        self._setup_robot()
        hand_pos = tensor_to_array(self.hand.get_pos(relative=False)).astype(np.float64)
        hand_quat = tensor_to_array(self.hand.get_quat(relative=False)).astype(np.float64)
        view_direction = np.array([0.52, 0.0, -0.42], dtype=np.float64)
        view_direction /= np.linalg.norm(view_direction)
        first_person_pos = hand_pos + view_direction * 0.045
        first_person_lookat = first_person_pos + view_direction * 0.85
        hand_transform = trans_quat_to_T(hand_pos, hand_quat)
        camera_transform = pos_lookat_up_to_T(
            first_person_pos,
            first_person_lookat,
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )
        self.first_person_camera.attach(self.hand, inv_T(hand_transform) @ camera_transform)
        self._cube_qpos = [cube.get_qpos().clone() for cube in self.cubes]
        self._sync_target()
        self._gripper = GRIPPER_OPEN
        self._q_limit = tensor_to_array(self.franka.q_limit)[:, :7]
        self._fps = CONTROL_HZ
        self._last_time = time.perf_counter()
        self._last_command_time = time.monotonic()

    def _setup_robot(self):
        self.franka.set_dofs_kp(ROBOT_KP)
        self.franka.set_dofs_kv(ROBOT_KV)
        self.franka.set_dofs_force_range(ROBOT_FORCE_MIN, ROBOT_FORCE_MAX)
        self.franka.set_qpos(HOME_QPOS, zero_velocity=True)
        for _ in range(20):
            self.franka.control_dofs_position(HOME_QPOS)
            self.scene.step()

    def _sync_target(self):
        self._target_pos = tensor_to_array(self.hand.get_pos(relative=False)).astype(np.float64)
        self._target_quat = tensor_to_array(self.hand.get_quat(relative=False)).astype(np.float64)

    def _clamp_target(self, target):
        boxed = np.clip(target, WORKSPACE_LOW, WORKSPACE_HIGH)
        is_clamped = not np.allclose(boxed, target)
        if np.linalg.norm(boxed - SHOULDER) > REACH_RADIUS:
            # Stop at the reach boundary instead of sliding along it, so motion stays intuitive.
            return None, True
        return boxed, is_clamped

    def _joint_positions(self):
        return tensor_to_array(self.franka.get_qpos())

    def _apply_action(self, action, params=None):
        if action == "home":
            if self._collision_mode:
                LOGGER.warning("home request ignored while collision zero-force mode is active")
                return
            self._agent_trajectory = None
            current = self._joint_positions()[:9].copy()
            self._home_trajectory = {
                "start": current,
                "target": HOME_QPOS.copy(),
                "started_at": float(self.scene.t) * float(self.scene.dt),
                "duration": HOME_DURATION_S,
                "last_log_sim_time": float(self.scene.t) * float(self.scene.dt),
            }
            LOGGER.info("smooth home trajectory started: duration=%.2fs start=%s", HOME_DURATION_S, current.round(3).tolist())
        elif action == "reset_scene":
            for cube, qpos in zip(self.cubes, self._cube_qpos):
                cube.set_qpos(qpos, zero_velocity=True)
        elif action == "toggle_recording":
            if self._recorder is not None:
                self._stop_recording()
            else:
                self._start_recording({"format": "mp4", "task": "Teleoperation session"})
        elif action == "recording_start":
            self._start_recording(params or {})
        elif action == "recording_stop":
            self._stop_recording()
        elif action == "agent_move":
            if self._collision_mode or self._home_trajectory is not None:
                LOGGER.warning("VLM motion ignored: collision or home trajectory is active")
                return
            try:
                delta = np.asarray((params or {}).get("delta_m", []), dtype=np.float64)
            except (TypeError, ValueError):
                LOGGER.warning("VLM motion rejected: invalid delta")
                return
            if delta.shape != (3,) or not np.all(np.isfinite(delta)) or np.max(np.abs(delta)) > AGENT_MOVE_MAX_M + 1e-9:
                LOGGER.warning("VLM motion rejected: delta outside per-axis limit: %s", delta.tolist())
                return
            target, clamped = self._clamp_target(self._target_pos + delta)
            if target is None or clamped:
                LOGGER.warning("VLM motion rejected: target outside Panda workspace: target=%s", (self._target_pos + delta).round(4).tolist())
                return
            self._agent_trajectory = {
                "start": self._target_pos.copy(),
                "target": target.copy(),
                "started_at": float(self.scene.t) * float(self.scene.dt),
                "duration": AGENT_MOVE_DURATION_S,
            }
            self._agent_move_failed = False
            self._home_trajectory = None
            LOGGER.info("VLM Cartesian step started: delta_m=%s duration=%.2fs", delta.round(4).tolist(), AGENT_MOVE_DURATION_S)
        elif action == "agent_gripper":
            if self._collision_mode or self._home_trajectory is not None:
                LOGGER.warning("VLM gripper action ignored: collision or home trajectory is active")
                return
            target = float((params or {}).get("target", GRIPPER_OPEN))
            if target not in (0.0, GRIPPER_OPEN):
                LOGGER.warning("VLM gripper action rejected: target must be fully open or closed")
                return
            self._agent_gripper_target = target
            LOGGER.info("VLM gripper action started: target_m=%.3f", target)
        elif action == "agent_stop":
            self._agent_trajectory = None
            self._agent_gripper_target = None
            if self._program_status.get("program") == "home":
                self._home_trajectory = None
            self._sync_target()
            if self._program_task is not None or self._program_status.get("state") in ("queued", "running"):
                self._finish_program("stopped", "已由操作员停止")
            LOGGER.info("VLM task stopped by operator")
        elif action == "agent_program":
            try:
                self._start_program(params or {})
            except Exception as exc:
                LOGGER.exception("natural-language program setup failed: %s", exc)
                self._program_status = {
                    "program": (params or {}).get("program", ""),
                    "label": (params or {}).get("label", "操作程序"),
                    "state": "failed",
                    "step": 0,
                    "total": 0,
                    "message": f"程序准备失败：{type(exc).__name__}",
                }

    def _program_move_steps(self, target, label):
        destination = np.asarray(target, dtype=np.float64)
        start = np.asarray(getattr(self, "_program_planning_position", self._target_pos), dtype=np.float64).copy()
        distance = float(np.linalg.norm(destination - start))
        segments = max(1, int(math.ceil(distance / 0.16)))
        steps = []
        for index in range(1, segments + 1):
            waypoint = start + (destination - start) * (index / segments)
            bounded, clamped = self._clamp_target(waypoint)
            if bounded is None or clamped:
                raise ValueError(f"{label}超出机械臂安全工作范围")
            segment_distance = float(np.linalg.norm(waypoint - (start if index == 1 else steps[-1]["target"])))
            steps.append({
                "kind": "move",
                "target": waypoint,
                "duration": max(0.75, segment_distance / 0.16 * 1.6),
                "label": label,
            })
        self._program_planning_position = destination.copy()
        return steps

    def _start_program(self, params):
        if self._collision_mode or self._home_trajectory is not None or not self._command.enabled:
            self._program_status = {**self._program_status, "state": "failed", "message": "碰撞保护、回零或控制停止期间不能运行程序"}
            return
        program = params.get("program")
        color = params.get("object_color")
        label = params.get("label", PROGRAMS.get(program, {}).get("label", program or ""))
        if program == "home":
            self._program_status = {"program": program, "label": label, "state": "running", "step": 1, "total": 1, "message": "平滑回到初始姿态"}
            self._apply_action("home")
            return
        if program == "reset_scene":
            self._apply_action("reset_scene")
            self._program_status = {"program": program, "label": label, "state": "completed", "step": 1, "total": 1, "message": "场景物体已复位"}
            return

        steps = []
        self._program_planning_position = self._target_pos.copy()
        if program in ("pick", "pick_place_tray"):
            object_candidates = [
                (index, cube, color_name)
                for index, (cube, color_name) in enumerate(zip(self.cubes, ("blue", "green")))
                if color is None or color == color_name
            ]
            if not object_candidates:
                self._program_status = {"program": program, "label": label, "state": "failed", "step": 0, "total": 0, "message": "找不到指定颜色的方块"}
                return
            _, cube, color_name = min(
                object_candidates,
                key=lambda item: float(np.linalg.norm(tensor_to_array(item[1].get_pos(relative=False))[:2] - self._target_pos[:2])),
            )
            object_position = tensor_to_array(cube.get_pos(relative=False)).astype(np.float64)
            grasp_z = max(float(object_position[2]) + 0.11, 0.12)
            approach = np.array([object_position[0], object_position[1], grasp_z + 0.14], dtype=np.float64)
            grasp = np.array([object_position[0], object_position[1], grasp_z], dtype=np.float64)
            lift = grasp + np.array([0.0, 0.0, 0.16])
            steps.append({"kind": "gripper", "target": GRIPPER_OPEN, "label": "打开夹爪"})
            steps.extend(self._program_move_steps(approach, f"移动到{color_name}方块上方"))
            steps.extend(self._program_move_steps(grasp, "下降到抓取位置"))
            steps.append({"kind": "gripper", "target": 0.0, "label": "夹住方块"})
            steps.append({"kind": "wait", "duration": 0.25, "label": "确认夹持"})
            steps.extend(self._program_move_steps(lift, "抬起方块"))
            if program == "pick_place_tray":
                tray_position = tensor_to_array(self.tray.get_pos(relative=False)).astype(np.float64)
                tray_hover = np.array([tray_position[0], tray_position[1], 0.34], dtype=np.float64)
                tray_drop = np.array([tray_position[0], tray_position[1], max(float(tray_position[2]) + 0.17, 0.17)], dtype=np.float64)
                steps.extend(self._program_move_steps(tray_hover, "移动到托盘上方"))
                steps.extend(self._program_move_steps(tray_drop, "下降到托盘"))
                steps.append({"kind": "gripper", "target": GRIPPER_OPEN, "label": "放开方块"})
                steps.append({"kind": "wait", "duration": 0.3, "label": "等待方块落稳"})
                steps.extend(self._program_move_steps(tray_hover, "抬离托盘"))
        elif program == "place_held_tray":
            tray_position = tensor_to_array(self.tray.get_pos(relative=False)).astype(np.float64)
            tray_hover = np.array([tray_position[0], tray_position[1], 0.34], dtype=np.float64)
            tray_drop = np.array([tray_position[0], tray_position[1], max(float(tray_position[2]) + 0.17, 0.17)], dtype=np.float64)
            steps.extend(self._program_move_steps(tray_hover, "移动到托盘上方"))
            steps.extend(self._program_move_steps(tray_drop, "下降到托盘"))
            steps.append({"kind": "gripper", "target": GRIPPER_OPEN, "label": "放开物体"})
            steps.append({"kind": "wait", "duration": 0.3, "label": "等待物体落稳"})
            steps.extend(self._program_move_steps(tray_hover, "抬离托盘"))
        elif program in ("release", "close_gripper"):
            steps.append({"kind": "gripper", "target": GRIPPER_OPEN if program == "release" else 0.0, "label": label})
        else:
            self._program_status = {"program": program or "", "label": label, "state": "failed", "step": 0, "total": 0, "message": "未知的内置程序"}
            return

        self._program_task = {"program": program, "label": label, "steps": steps, "index": 0, "phase": "starting"}
        self._program_status = {"program": program, "label": label, "state": "running", "step": 0, "total": len(steps), "message": "准备执行"}
        LOGGER.info("natural-language program started: program=%s color=%s steps=%d", program, color_name if program in ("pick", "pick_place_tray") else "", len(steps))
        self._start_program_step()

    def _start_program_step(self):
        task = self._program_task
        if task is None:
            return
        if task["index"] >= len(task["steps"]):
            self._finish_program("completed", "操作程序已完成")
            return
        step = task["steps"][task["index"]]
        self._program_status = {
            "program": task["program"],
            "label": task["label"],
            "state": "running",
            "step": task["index"] + 1,
            "total": len(task["steps"]),
            "message": step["label"],
        }
        if step["kind"] == "move":
            self._agent_move_failed = False
            self._agent_trajectory = {
                "start": self._target_pos.copy(),
                "target": np.asarray(step["target"], dtype=np.float64).copy(),
                "started_at": float(self.scene.t) * float(self.scene.dt),
                "duration": step["duration"],
            }
            task["phase"] = "move"
        elif step["kind"] == "gripper":
            self._agent_gripper_target = float(step["target"])
            task["phase"] = "gripper"
        else:
            task["phase"] = "wait"
            task["started_at"] = float(self.scene.t) * float(self.scene.dt)

    def _finish_program(self, state, message):
        task = self._program_task
        if task is not None:
            self._program_status = {
                "program": task["program"],
                "label": task["label"],
                "state": state,
                "step": self._program_status.get("step", 0),
                "total": len(task["steps"]),
                "message": message,
            }
        elif self._program_status.get("state") in ("queued", "running"):
            self._program_status = {**self._program_status, "state": state, "message": message}
        self._program_task = None
        self._agent_trajectory = None
        self._agent_gripper_target = None
        LOGGER.info("natural-language program %s: %s", state, message)

    def _advance_program_step(self):
        if self._program_task is None:
            return
        self._program_task["index"] += 1
        self._start_program_step()

    def _advance_program_command(self):
        task = self._program_task
        if task is None:
            return
        if task["phase"] == "move":
            self._advance_agent_trajectory()
            if self._agent_move_failed:
                self._finish_program("failed", "目标不可达或动作超出工作范围，程序已停止")
            elif self._agent_trajectory is None:
                self._advance_program_step()
            return
        if task["phase"] == "gripper":
            target = self._agent_gripper_target
            if target is None:
                self._advance_program_step()
                return
            self._gripper = float(np.clip(self._gripper + np.clip(target - self._gripper, -0.002, 0.002), 0.0, GRIPPER_OPEN))
            self._last_action_joints = self._joint_positions()[:7].copy()
            self._last_action_gripper = self._gripper
            self.franka.control_dofs_position(np.array([self._gripper, self._gripper]), FINGER_DOF_IDX)
            for _ in range(2):
                self.scene.step()
            if abs(self._gripper - target) < 0.001:
                self._agent_gripper_target = None
                self._advance_program_step()
            return
        self.franka.control_dofs_position(np.array([self._gripper, self._gripper]), FINGER_DOF_IDX)
        for _ in range(2):
            self.scene.step()
        sim_time = float(self.scene.t) * float(self.scene.dt)
        if sim_time - task["started_at"] >= task["steps"][task["index"]]["duration"]:
            self._advance_program_step()

    def _advance_agent_trajectory(self):
        trajectory = self._agent_trajectory
        if trajectory is None:
            return False
        sim_time = float(self.scene.t) * float(self.scene.dt)
        elapsed = max(0.0, sim_time - trajectory["started_at"])
        t = float(np.clip(elapsed / trajectory["duration"], 0.0, 1.0))
        blend = t * t * t * (10.0 + t * (-15.0 + 6.0 * t))
        candidate = trajectory["start"] + blend * (trajectory["target"] - trajectory["start"])
        target, clamped = self._clamp_target(candidate)
        if target is None or clamped:
            LOGGER.warning("VLM motion stopped at workspace boundary: target=%s", candidate.round(4).tolist())
            self._agent_trajectory = None
            self._agent_move_failed = True
            self._sync_target()
            return False
        previous_target = self._target_pos.copy()
        qpos, err = self.franka.inverse_kinematics(
            link=self.hand,
            pos=target,
            quat=self._target_quat,
            init_qpos=self._joint_positions(),
            dofs_idx_local=ARM_DOFS,
            return_error=True,
            max_samples=1,
        )
        if np.linalg.norm(tensor_to_array(err)[:3]) > 0.012:
            LOGGER.warning("VLM motion stopped: IK error exceeded 12 mm at target=%s", target.round(4).tolist())
            self._agent_trajectory = None
            self._agent_move_failed = True
            self._target_pos = previous_target
        else:
            self._target_pos = target
            self._clamped = False
            self._last_action_joints = tensor_to_array(qpos)[:7].copy()
            self._last_action_gripper = self._gripper
            self.franka.control_dofs_position(self._last_action_joints, ARM_DOFS)
        self.franka.control_dofs_position(np.array([self._gripper, self._gripper]), FINGER_DOF_IDX)
        for _ in range(2):
            self.scene.step()
        if t >= 1.0:
            self._agent_trajectory = None
            LOGGER.info("VLM Cartesian step completed: ee_target=%s", self._target_pos.round(4).tolist())
        return True

    def _advance_home_trajectory(self):
        trajectory = self._home_trajectory
        if trajectory is None:
            return False
        sim_time = float(self.scene.t) * float(self.scene.dt)
        elapsed = max(0.0, sim_time - trajectory["started_at"])
        t = float(np.clip(elapsed / trajectory["duration"], 0.0, 1.0))
        blend = t * t * t * (10.0 + t * (-15.0 + 6.0 * t))
        qpos = trajectory["start"] + blend * (trajectory["target"] - trajectory["start"])
        self.franka.control_dofs_position(qpos, np.arange(9))
        self._last_action_joints = qpos[:7].copy()
        self._last_action_gripper = float(np.mean(qpos[7:9]))
        if sim_time - trajectory["last_log_sim_time"] >= 0.5 or t >= 1.0:
            trajectory["last_log_sim_time"] = sim_time
            LOGGER.debug(
                "smooth home progress: elapsed=%.2f/%.2fs progress=%.0f%% commanded_joints=%s",
                elapsed,
                trajectory["duration"],
                100.0 * t,
                qpos[:7].round(3).tolist(),
            )
        for _ in range(2):
            self.scene.step()
        if t >= 1.0:
            self._home_trajectory = None
            self._gripper = GRIPPER_OPEN
            self._sync_target()
            LOGGER.info("smooth home trajectory completed")
            if self._program_status.get("program") == "home" and self._program_status.get("state") == "running":
                self._program_status = {**self._program_status, "state": "completed", "step": 1, "message": "已回到初始姿态"}
        return True

    def _step_zero_force_mode(self):
        current = self._joint_positions()[:9].copy()
        self._last_action_joints = current[:7].copy()
        self._last_action_gripper = float(np.mean(current[7:9]))
        self.franka.control_dofs_force(np.zeros(9), np.arange(9))
        for _ in range(2):
            self.scene.step()

    def _enter_collision_mode(self, force_n, position, link_name):
        if self._collision_mode:
            return
        self._collision_mode = True
        self._agent_trajectory = None
        self._agent_gripper_target = None
        self._home_trajectory = None
        if self._program_task is not None or self._program_status.get("state") in ("queued", "running"):
            self._finish_program("failed", "碰撞保护已触发，操作程序已停止")
        self._collision_force_n = float(force_n)
        self._collision_position = np.asarray(position, dtype=np.float64).round(4).tolist()
        self._collision_link = link_name
        self.franka.set_dofs_kp(np.zeros(9))
        self.franka.set_dofs_kv(np.zeros(9))
        self._control_enabled = False
        LOGGER.error(
            "collision protection engaged: force=%.2fN threshold=%.2fN link=%s position=%s; arm is now in zero-force mode",
            force_n,
            COLLISION_FORCE_THRESHOLD_N,
            link_name or "unknown",
            self._collision_position,
        )

    def _check_collision(self):
        if self._collision_mode:
            return
        contacts = self.franka.get_contacts(exclude_self_contact=True)
        force_a = tensor_to_array(contacts["force_a"])
        force_b = tensor_to_array(contacts["force_b"])
        if force_a.size == 0 and force_b.size == 0:
            return
        force_a = np.asarray(force_a).reshape((-1, 3))
        force_b = np.asarray(force_b).reshape((-1, 3))
        magnitudes = np.maximum(np.linalg.norm(force_a, axis=1), np.linalg.norm(force_b, axis=1))
        if magnitudes.size == 0:
            return
        geom_a = tensor_to_array(contacts["geom_a"]).reshape(-1)
        geom_b = tensor_to_array(contacts["geom_b"]).reshape(-1)
        link_a = tensor_to_array(contacts["link_a"]).reshape(-1)
        link_b = tensor_to_array(contacts["link_b"]).reshape(-1)
        if self._program_task is not None and self._program_task.get("phase") == "gripper":
            movable_geometry_ranges = [(cube.geom_start, cube.geom_end) for cube in self.cubes]
            movable_geometry_ranges.append((self.tray.geom_start, self.tray.geom_end))
            for index in range(min(len(magnitudes), len(geom_a), len(geom_b), len(link_a), len(link_b))):
                try:
                    robot_contact = self.franka.geom_start <= geom_a[index] < self.franka.geom_end
                    link_index = link_a[index] if robot_contact else link_b[index]
                    other_geom = geom_b[index] if robot_contact else geom_a[index]
                    link_name = getattr(self.scene.rigid_solver.links[int(link_index)], "name", "")
                except (IndexError, TypeError, ValueError):
                    continue
                touches_program_object = any(start <= other_geom < end for start, end in movable_geometry_ranges)
                if link_name in ("left_finger", "right_finger") and touches_program_object:
                    magnitudes[index] = 0.0
        contact_index = int(np.argmax(magnitudes))
        force_n = float(magnitudes[contact_index])
        if force_n < COLLISION_FORCE_THRESHOLD_N:
            return
        positions = tensor_to_array(contacts["position"]).reshape((-1, 3))
        link_name = ""
        if contact_index < len(geom_a) and contact_index < len(geom_b):
            try:
                link_index = link_a[contact_index] if self.franka.geom_start <= geom_a[contact_index] < self.franka.geom_end else link_b[contact_index]
                link = self.scene.rigid_solver.links[int(link_index)]
                link_name = getattr(link, "name", "")
            except (IndexError, TypeError, ValueError):
                pass
        self._enter_collision_mode(force_n, positions[contact_index], link_name)

    def queue_collision_reset(self):
        with self._lock:
            if not self._collision_mode:
                return False
            self._collision_reset_pending = True
        LOGGER.info("operator requested exit from collision zero-force mode")
        return True

    def _apply_collision_reset_request(self):
        with self._lock:
            requested = self._collision_reset_pending
            self._collision_reset_pending = False
        if requested:
            self._reset_collision_mode()

    def _reset_collision_mode(self):
        if not self._collision_mode:
            return False
        qpos = self._joint_positions()[:9].copy()
        self.franka.set_dofs_kp(ROBOT_KP)
        self.franka.set_dofs_kv(ROBOT_KV)
        self.franka.control_dofs_position(qpos, np.arange(9))
        self._collision_mode = False
        self._collision_force_n = 0.0
        self._collision_position = None
        self._collision_link = ""
        self._home_trajectory = None
        self._control_enabled = True
        self._gripper = float(np.mean(qpos[7:9]))
        self._sync_target()
        LOGGER.warning("collision zero-force mode cleared by operator; position control restored at current pose")
        return True

    def _apply_command(self, command):
        if command.action:
            # Recording and object reset remain available while paused. Robot motion actions
            # such as homing are accepted only when control is enabled.
            if command.action in ("toggle_recording", "reset_scene", "recording_start", "recording_stop", "agent_stop", "agent_program") or command.enabled:
                self._apply_action(command.action, command.action_params)

        if self._collision_mode:
            self._agent_trajectory = None
            self._agent_gripper_target = None
            self._step_zero_force_mode()
            self._sync_target()
            self._clamped = False
            return

        if not command.enabled:
            self._home_trajectory = None
            self._agent_trajectory = None
            self._agent_gripper_target = None
            if self._program_task is not None or self._program_status.get("state") in ("queued", "running"):
                self._finish_program("stopped", "控制已停止或连接中断")
            if self._control_enabled:
                # Capture the current pose once and hold it, so Stop doesn't keep integrating
                # an old key or slider target.
                qpos = self._joint_positions()
                self._gripper = float(np.mean(qpos[7:9]))
                self._last_action_joints = qpos[:7].copy()
                self._last_action_gripper = self._gripper
                self.franka.control_dofs_position(qpos[:9], np.arange(9))
                self._control_enabled = False
            for _ in range(2):
                self.scene.step()
            self._sync_target()
            self._clamped = False
            return

        self._control_enabled = True

        if self._program_task is not None:
            if command.keys:
                self._finish_program("stopped", "收到人工移动输入，程序已交还控制权")
                self._sync_target()
            else:
                self._advance_program_command()
                return

        if self._agent_trajectory is not None and command.keys:
            LOGGER.info("VLM Cartesian step interrupted by direct teleoperation input")
            self._agent_trajectory = None
            self._sync_target()

        if self._advance_home_trajectory():
            self._clamped = False
            return

        if self._advance_agent_trajectory():
            return

        speed = float(np.clip(command.speed, 0.01, 0.5))
        yaw_speed = float(np.clip(command.yaw_speed, 0.05, 2.5))
        gripper_goal = command.gripper if self._agent_gripper_target is None else self._agent_gripper_target
        self._gripper = float(
            np.clip(
                self._gripper + np.clip(gripper_goal - self._gripper, -0.002, 0.002),
                0.0,
                GRIPPER_OPEN,
            )
        )
        if self._agent_gripper_target is not None and abs(self._gripper - self._agent_gripper_target) < 0.001:
            self._agent_gripper_target = None

        if command.mode == "joint" and len(command.joints) == 7:
            joints = np.clip(np.array(command.joints), self._q_limit[0], self._q_limit[1])
            self._last_action_joints = joints.copy()
            self._last_action_gripper = self._gripper
            self.franka.control_dofs_position(joints, ARM_DOFS)
            self.franka.control_dofs_position(np.array([self._gripper, self._gripper]), FINGER_DOF_IDX)
            for _ in range(2):
                self.scene.step()
            self._sync_target()
            self._clamped = False
            return

        keys = set(command.keys)
        delta = np.zeros(3)
        delta[0] = (("KeyW" in keys) - ("KeyS" in keys)) * speed
        delta[1] = (("KeyA" in keys) - ("KeyD" in keys)) * speed
        delta[2] = (("KeyR" in keys) - ("KeyF" in keys)) * speed
        yaw = (("KeyQ" in keys) - ("KeyE" in keys)) * yaw_speed

        if not np.any(delta) and abs(yaw) < 1e-9:
            # Keep the last arm target while idle. Re-solving IK at 30 Hz with an unchanged
            # Cartesian target wastes time and delays the first camera frame.
            self.franka.control_dofs_position(np.array([self._gripper, self._gripper]), FINGER_DOF_IDX)
            self._last_action_joints = self._joint_positions()[:7].copy()
            self._last_action_gripper = self._gripper
            for _ in range(2):
                self.scene.step()
            self._clamped = False
            return

        previous_target = self._target_pos.copy()
        target, is_clamped = self._clamp_target(self._target_pos + delta * SERVO_DT)
        if target is None:
            self._target_pos = previous_target
            self._clamped = True
        else:
            self._target_pos = target

        if abs(yaw) > 1e-9:
            half = 0.5 * yaw * SERVO_DT
            self._target_quat = transform_quat_by_quat(
                self._target_quat, np.array([math.cos(half), 0.0, 0.0, math.sin(half)])
            )

        qpos, err = self.franka.inverse_kinematics(
            link=self.hand,
            pos=self._target_pos,
            quat=self._target_quat,
            init_qpos=self._joint_positions(),
            dofs_idx_local=ARM_DOFS,
            return_error=True,
            max_samples=1,
        )
        if np.linalg.norm(tensor_to_array(err)[:3]) > 0.012:
            # Hold the last reachable target instead of chasing a pose the arm cannot reach.
            self._target_pos = previous_target
            self._clamped = True
        else:
            self._clamped = is_clamped
            action_joints = tensor_to_array(qpos)[:7]
            self._last_action_joints = action_joints.copy()
            self._last_action_gripper = self._gripper
            self.franka.control_dofs_position(action_joints, ARM_DOFS)
        self.franka.control_dofs_position(np.array([self._gripper, self._gripper]), FINGER_DOF_IDX)
        for _ in range(2):
            self.scene.step()

    def _start_recording(self, params):
        recording_format = params.get("format", "mp4")
        task = params.get("task", "Teleoperation session").strip() or "Teleoperation session"
        try:
            recorder = create_recording_session(
                recording_format,
                task,
                CONTROL_HZ,
                (640, 480),
            )
        except Exception as exc:
            with self._lock:
                self._recording_status = "error"
                self._recording_error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("recording start failed: format=%s task=%s", recording_format, task)
            return
        self._recorder = recorder
        with self._lock:
            self._recording_status = "recording"
            self._recording_format = recording_format
            self._recording_task = task
            self._recording_path = str(recorder.path.resolve())
            self._recording_frames = 0
            self._recording_error = ""
        LOGGER.info("recording started: format=%s task=%s path=%s", recording_format, task, recorder.path.resolve())

    def _stop_recording(self):
        recorder = self._recorder
        if recorder is None:
            with self._lock:
                self._recording_status = "idle"
            return
        self._recorder = None
        try:
            recorder.finish()
        except Exception as exc:
            with self._lock:
                self._recording_status = "error"
                self._recording_error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("recording finalize failed: format=%s path=%s", recorder.recording_format, recorder.path)
            return
        with self._lock:
            self._recording_status = "idle"
            self._recording_frames = recorder.frame_count
            self._recording_error = ""
        LOGGER.info(
            "recording saved: format=%s frames=%d path=%s",
            recorder.recording_format,
            recorder.frame_count,
            recorder.path.resolve(),
        )

    def queue_recording_start(self, recording_format: str, task: str):
        with self._lock:
            if self._recording_status in ("starting", "recording", "stopping"):
                return False
            self._recording_status = "starting"
            self._pending_actions.append(("recording_start", {"format": recording_format, "task": task}))
        LOGGER.info("recording start queued: format=%s task=%s", recording_format, task)
        return True

    def queue_recording_stop(self):
        with self._lock:
            if self._recording_status not in ("recording", "starting"):
                return False
            self._recording_status = "stopping"
            self._pending_actions.append(("recording_stop", None))
        LOGGER.info("recording stop queued")
        return True

    def _write_recording_frame(self):
        recorder = self._recorder
        if recorder is None:
            return
        with self._lock:
            state = dict(self._state)
            main_jpeg = self._frame
            main_bgr = self._camera_bgr
            first_person_jpeg = self._first_person_frame
            command = self._command
        if main_jpeg is None or main_bgr is None:
            return
        observation = {
            "sim_time": state.get("sim_time", 0.0),
            "joints": state.get("joints", []),
            "gripper": state.get("gripper", 0.0),
            "ee": state.get("ee", []),
            "quat": state.get("quat", []),
            "collision_mode": state.get("collision_mode", False),
            "collision_force_n": state.get("collision_force_n", 0.0),
            "sensor_mode": state.get("sensor_mode", "rgb"),
        }
        action = {
            "joint_target": self._last_action_joints.round(6).tolist(),
            "gripper_target": float(self._last_action_gripper),
            "control_mode": command.mode,
        }
        timestamp_ns = time.time_ns()
        try:
            recorder.write_frame(
                observation,
                action,
                main_bgr,
                main_jpeg,
                first_person_jpeg,
                timestamp_ns,
            )
            with self._lock:
                self._recording_frames = recorder.frame_count
        except Exception as exc:
            LOGGER.exception("recording frame write failed: format=%s", recorder.recording_format)
            self._recorder = None
            try:
                recorder.finish()
            except Exception:
                LOGGER.exception("recording cleanup failed after write error")
            with self._lock:
                self._recording_status = "error"
                self._recording_error = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _to_bgr(image):
        image = np.asarray(tensor_to_array(image))
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
            image = np.moveaxis(image, 0, -1)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if image.ndim == 3 and image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        if image.shape[-1] == 4:
            image = image[..., :3]
        if image.dtype != np.uint8:
            scale = 255.0 if image.size and np.nanmax(image) <= 1.0 else 1.0
            image = np.clip(np.nan_to_num(image) * scale, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image[..., ::-1])

    def _render_first_person_image(self, mode):
        self.first_person_camera.move_to_attach()
        if mode == "rgb":
            return self._to_bgr(self.first_person_camera.render(rgb=True)[0])
        if mode == "depth":
            depth = np.asarray(tensor_to_array(self.first_person_camera.render(rgb=False, depth=True)[1])).squeeze()
            valid = np.isfinite(depth) & (depth > 0.0) & (depth < self.first_person_camera.far)
            depth_norm = np.clip((np.nan_to_num(depth, nan=2.5) - 0.1) / 2.4, 0.0, 1.0)
            depth_u8 = np.ascontiguousarray((depth_norm * 255.0).astype(np.uint8))
            color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
            color[~valid] = 0
            return color
        if mode == "segmentation":
            seg = self.first_person_camera.render(
                rgb=False,
                segmentation=True,
                colorize_seg=True,
            )[2]
            return self._to_bgr(seg)

        pointcloud, mask = self.first_person_camera.render_pointcloud(world_frame=True)
        pointcloud = np.asarray(pointcloud)
        mask = np.asarray(mask)
        if pointcloud.ndim == 4 and pointcloud.shape[0] == 1:
            pointcloud = pointcloud[0]
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        # Encode world X/Y/Z as R/G/B with fixed bounds for comparable frames.
        lower = np.array([0.0, -0.5, -0.05], dtype=np.float32)
        upper = np.array([1.0, 0.5, 1.0], dtype=np.float32)
        xyz_rgb = np.clip((pointcloud - lower) / (upper - lower), 0.0, 1.0)
        xyz_rgb = np.nan_to_num(xyz_rgb)
        xyz_rgb = (xyz_rgb * 255.0).astype(np.uint8)
        xyz_rgb[~mask.astype(bool)] = 0
        return np.ascontiguousarray(xyz_rgb[..., ::-1])

    def _publish_first_person_frame(self):
        now = time.monotonic()
        with self._lock:
            if now - self._last_first_person_render_at < 1.0 / FIRST_PERSON_HZ:
                return
            self._last_first_person_render_at = now
            mode = self._sensor_mode
        try:
            bgr = self._render_first_person_image(mode)
            ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
            if not ok:
                raise RuntimeError("first-person JPEG encoding failed")
            with self._lock:
                self._first_person_frame = encoded.tobytes()
                self._first_person_frame_at = time.monotonic()
        except Exception:
            if now - self._last_first_person_error_log_at >= 5.0:
                LOGGER.exception("first-person camera render failed: mode=%s", mode)
                self._last_first_person_error_log_at = now

    def _publish_frame(self):
        rgb = self.camera.render(rgb=True)[0]
        # OpenCV's encoder and VideoWriter require a contiguous image. Channel reversal by
        # slicing alone creates a negative-stride view and can terminate the simulation loop.
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if ok:
            with self._lock:
                first_frame = self._frame is None
                self._frame = encoded.tobytes()
                self._camera_bgr = bgr
                self._last_frame_at = time.monotonic()
            if first_frame:
                LOGGER.info("first camera frame published: resolution=%sx%s", bgr.shape[1], bgr.shape[0])
        else:
            self._jpeg_failures += 1
            now = time.monotonic()
            if now - self._last_jpeg_warning_at >= 10.0:
                LOGGER.warning("JPEG encoding failed: total_failures=%d", self._jpeg_failures)
                self._last_jpeg_warning_at = now

    def _publish_state(self):
        joints = self._joint_positions()
        state = {
            "ee": tensor_to_array(self.hand.get_pos(relative=False)).round(4).tolist(),
            "quat": tensor_to_array(self.hand.get_quat(relative=False)).round(4).tolist(),
            "joints": joints[:7].round(4).tolist(),
            "joint_limits": self._q_limit[:, :7].round(4).T.tolist(),
            "gripper": round(float(joints[7]), 4),
            "fps": round(self._fps, 1),
            "sim_time": round(float(self.scene.t) * self.scene.dt, 2),
            "clamped": self._clamped,
            "recording": self._recording_status in ("starting", "recording", "stopping"),
            "recording_status": self._recording_status,
            "recording_format": self._recording_format,
            "recording_task": self._recording_task,
            "recording_path": self._recording_path,
            "recording_frames": self._recording_frames,
            "recording_error": self._recording_error,
            "enabled": self._control_enabled,
            "homing": self._home_trajectory is not None,
            "collision_mode": self._collision_mode,
            "collision_force_n": round(self._collision_force_n, 2),
            "collision_threshold_n": COLLISION_FORCE_THRESHOLD_N,
            "collision_position": self._collision_position,
            "collision_link": self._collision_link,
            "sensor_mode": self._sensor_mode,
            "camera": dict(self._camera_snapshot),
            "objects": [
                {"name": "blue cube", "color": "blue", "position": tensor_to_array(self.cubes[0].get_pos(relative=False)).round(4).tolist()},
                {"name": "green cube", "color": "green", "position": tensor_to_array(self.cubes[1].get_pos(relative=False)).round(4).tolist()},
                {"name": "gray tray", "color": "gray", "position": tensor_to_array(self.tray.get_pos(relative=False)).round(4).tolist()},
            ],
            "agent_busy": (
                self._agent_trajectory is not None
                or self._agent_gripper_target is not None
                or self._program_status.get("state") in ("queued", "running")
            ),
            "program_status": dict(self._program_status),
        }
        with self._lock:
            self._state = state

    def set_command(self, payload: CommandPayload):
        with self._lock:
            self._command = TeleopCommand(
                keys=tuple(payload.keys),
                gripper=float(payload.gripper),
                speed=float(payload.speed),
                yaw_speed=float(payload.yaw_speed),
                mode=payload.mode,
                joints=tuple(payload.joints[:7]),
                action="",
                action_params=None,
                enabled=payload.enabled,
            )
            self._last_command_time = time.monotonic()

    def queue_action(self, action: str):
        with self._lock:
            self._pending_actions.append((action, None))
        LOGGER.info("action queued: %s", action)

    def queue_agent_action(self, action: str, params: dict | None = None):
        with self._lock:
            state = self._state
            if state.get("collision_mode"):
                raise ValueError("碰撞保护生效时不能执行 VLM 动作")
            if state.get("homing"):
                raise ValueError("机械臂正在回到初始姿态")
            if not self._command.enabled:
                raise ValueError("请先恢复机械臂控制")
            if state.get("agent_busy"):
                raise ValueError("上一条 VLM 动作尚未完成")
            if action not in ("move", "open_gripper", "close_gripper"):
                raise ValueError("不支持的 VLM 动作")
            if action == "move":
                delta = validate_move_delta((params or {}).get("delta_m", []))
                ee = np.asarray(state.get("ee") or [], dtype=np.float64)
                target = ee + delta if ee.shape == (3,) else np.full(3, np.nan)
                if not np.all(np.isfinite(target)) or np.any(target < WORKSPACE_LOW) or np.any(target > WORKSPACE_HIGH):
                    raise ValueError("VLM 移动目标超出工作空间，请重新观察")
                if np.linalg.norm(target - SHOULDER) > REACH_RADIUS:
                    raise ValueError("VLM 移动目标超出机械臂可达范围，请重新观察")
                self._pending_actions.append(("agent_move", {"delta_m": delta.tolist()}))
            else:
                target = 0.0 if action == "close_gripper" else GRIPPER_OPEN
                self._pending_actions.append(("agent_gripper", {"target": target}))
        LOGGER.info("VLM action queued by teleop UI: action=%s params=%s", action, params)

    def queue_agent_stop(self):
        with self._lock:
            self._pending_actions.append(("agent_stop", None))
        LOGGER.info("VLM task stop requested by operator")

    def queue_natural_program(self, text):
        request = interpret_natural_program(text)
        with self._lock:
            state = self._state
            if not state.get("ee"):
                raise ValueError("Genesis 仿真尚未就绪")
            if state.get("collision_mode"):
                raise ValueError("碰撞保护生效时不能执行操作程序")
            if state.get("homing"):
                raise ValueError("机械臂正在回到初始姿态")
            if not self._command.enabled or time.monotonic() - self._last_command_time > 0.7:
                raise ValueError("请先连接工作台并恢复机械臂控制")
            if self._command.mode == "joint":
                raise ValueError("自然语言操作需要先切换到末端控制模式")
            pending_program = any(action == "agent_program" for action, _ in self._pending_actions)
            if state.get("agent_busy") or pending_program:
                raise ValueError("当前已有动作或程序正在执行")
            self._program_status = {
                "program": request["program"],
                "label": request["label"],
                "state": "queued",
                "step": 0,
                "total": 0,
                "message": "已匹配内置程序，等待仿真执行",
            }
            self._pending_actions.append(("agent_program", request))
        LOGGER.info("natural-language request matched built-in program: request=%r match=%s", text, request)
        return request

    def set_sensor_mode(self, mode: str):
        with self._lock:
            self._sensor_mode = mode
            self._first_person_frame = None
            self._first_person_frame_at = None
            self._last_first_person_render_at = 0.0
        LOGGER.info("first-person sensor mode selected: %s", mode)

    def get_command(self):
        with self._lock:
            command = self._command
            if self._pending_actions:
                action, params = self._pending_actions.popleft()
                command = replace(command, action=action, action_params=params)
            # A closed tab or a dropped connection must stop the arm rather than replay the last
            # command forever.
            if time.monotonic() - self._last_command_time > 0.7:
                return replace(command, keys=(), enabled=False)
            return command

    def get_frame(self):
        with self._lock:
            return self._frame

    def get_first_person_frame(self):
        with self._lock:
            return self._first_person_frame

    def get_state(self):
        with self._lock:
            return dict(self._state)

    def get_diagnostics(self):
        now = time.monotonic()
        with self._lock:
            state = dict(self._state)
            last_frame_at = self._last_frame_at
            last_camera_command_at = self._last_camera_command_at
            last_camera_applied_at = self._last_camera_applied_at
            camera = dict(self._camera_snapshot)
            pending_camera = self._pending_camera.round(5).tolist()
            last_command_time = self._last_command_time
            sensor_mode = self._sensor_mode
            first_person_frame_at = self._first_person_frame_at
            collision_mode = self._collision_mode
            collision_force_n = self._collision_force_n
            recording_status = self._recording_status
            recording_format = self._recording_format
            recording_frames = self._recording_frames
            recording_path = self._recording_path
            recording_error = self._recording_error
        age = lambda timestamp: None if timestamp is None else round(max(0.0, now - timestamp), 3)
        return {
            "running": self._running,
            "simulation_loop_active": self._simulation_loop_active,
            "simulation_ready": bool(state.get("ee")) and not state.get("error"),
            "frame_ready": last_frame_at is not None,
            "frame_age_s": age(last_frame_at),
            "command_age_s": age(last_command_time),
            "camera_command_age_s": age(last_camera_command_at),
            "camera_applied_age_s": age(last_camera_applied_at),
            "camera": camera,
            "camera_pending": pending_camera,
            "first_person_mode": sensor_mode,
            "first_person_frame_age_s": age(first_person_frame_at),
            "collision_mode": collision_mode,
            "collision_force_n": round(collision_force_n, 2),
            "collision_threshold_n": COLLISION_FORCE_THRESHOLD_N,
            "recording": {
                "status": recording_status,
                "format": recording_format,
                "frames": recording_frames,
                "path": recording_path,
                "error": recording_error,
            },
            "last_error": state.get("error", ""),
            "log_file": str(LOG_PATH),
        }

    def queue_camera_control(self, payload: CameraControlPayload):
        with self._lock:
            self._last_camera_command_at = time.monotonic()
            if payload.reset:
                self._pending_camera_reset = True
                self._pending_camera.fill(0.0)
            else:
                self._pending_camera += np.array(
                    [payload.orbit_x, payload.orbit_y, payload.pan_x, payload.pan_y, payload.zoom],
                    dtype=np.float64,
                )
        LOGGER.debug(
            "camera input queued: orbit=(%.4f, %.4f) pan=(%.4f, %.4f) zoom=%.4f reset=%s",
            payload.orbit_x,
            payload.orbit_y,
            payload.pan_x,
            payload.pan_y,
            payload.zoom,
            payload.reset,
        )

    def _apply_camera_control(self):
        with self._lock:
            control = self._pending_camera.copy()
            self._pending_camera.fill(0.0)
            reset = self._pending_camera_reset
            self._pending_camera_reset = False

        if reset:
            camera_offset = CAMERA_POS - CAMERA_LOOKAT
            self._camera_target = CAMERA_LOOKAT.copy()
            self._camera_distance = float(np.linalg.norm(camera_offset))
            self._camera_azimuth = math.atan2(camera_offset[1], camera_offset[0])
            self._camera_elevation = math.asin(camera_offset[2] / self._camera_distance)
        elif np.any(control):
            orbit_x, orbit_y, pan_x, pan_y, zoom = control
            self._camera_azimuth -= orbit_x * 2.8
            self._camera_azimuth = math.atan2(math.sin(self._camera_azimuth), math.cos(self._camera_azimuth))
            self._camera_elevation = float(np.clip(self._camera_elevation + orbit_y * 2.0, -1.45, 1.45))
            self._camera_distance = float(np.clip(self._camera_distance * math.exp(zoom * 0.25), 0.45, 6.0))

            horizontal = self._camera_distance * math.cos(self._camera_elevation)
            offset = np.array(
                [
                    horizontal * math.cos(self._camera_azimuth),
                    horizontal * math.sin(self._camera_azimuth),
                    self._camera_distance * math.sin(self._camera_elevation),
                ]
            )
            forward = -offset / self._camera_distance
            world_up = np.array([0.0, 0.0, 1.0])
            right = np.cross(forward, world_up)
            right /= max(np.linalg.norm(right), 1e-8)
            camera_up = np.cross(right, forward)
            world_per_viewport = 2.0 * self._camera_distance * math.tan(math.radians(CAMERA_FOV) / 2.0)
            self._camera_target += world_per_viewport * (-pan_x * right + pan_y * camera_up)

        if reset or np.any(control):
            horizontal = self._camera_distance * math.cos(self._camera_elevation)
            position = self._camera_target + np.array(
                [
                    horizontal * math.cos(self._camera_azimuth),
                    horizontal * math.sin(self._camera_azimuth),
                    self._camera_distance * math.sin(self._camera_elevation),
                ]
            )
            self.camera.set_pose(pos=position, lookat=self._camera_target, up=(0.0, 0.0, 1.0))
            now = time.monotonic()
            with self._lock:
                self._last_camera_applied_at = now
                self._camera_snapshot = {
                    "position": position.round(4).tolist(),
                    "target": self._camera_target.round(4).tolist(),
                    "distance": round(self._camera_distance, 4),
                }
                self._camera_updates_since_log += 1
                log_update_count = self._camera_updates_since_log
                should_log = reset or now - self._last_camera_log_at >= 1.0
                if should_log:
                    self._camera_updates_since_log = 0
                    self._last_camera_log_at = now
            if should_log:
                LOGGER.info(
                    "camera pose applied: updates=%d reset=%s distance=%.3f position=%s target=%s",
                    log_update_count,
                    reset,
                    self._camera_distance,
                    np.round(position, 3).tolist(),
                    np.round(self._camera_target, 3).tolist(),
                )

    def _start_api_server(self):
        app = create_app(self)
        config = uvicorn.Config(
            app,
            host=self._args.host,
            port=self._args.port,
            log_level="warning",
            access_log=False,
            log_config=None,
        )
        self._api_server = uvicorn.Server(config)
        self._api_thread = threading.Thread(
            target=lambda: asyncio.run(self._api_server.serve()),
            name="teleop-api",
            daemon=True,
        )
        self._api_thread.start()
        display_host = "127.0.0.1" if self._args.host in ("0.0.0.0", "::") else self._args.host
        LOGGER.info("FastAPI console starting at http://%s:%s; log_file=%s", display_host, self._args.port, LOG_PATH)
        if self._args.host in ("0.0.0.0", "::"):
            LOGGER.warning("listening on all network interfaces; LAN control has no authentication")

    def run(self):
        LOGGER.info("simulation loop started: control_hz=%.1f", CONTROL_HZ)
        self._simulation_loop_active = True

        period = 1.0 / CONTROL_HZ
        next_time = time.perf_counter()
        last_time = time.perf_counter()
        try:
            while self._running:
                self._apply_collision_reset_request()
                command = self.get_command()
                self._apply_command(command)
                self._check_collision()
                self._apply_camera_control()
                self._publish_frame()
                self._publish_first_person_frame()
                self._publish_state()
                self._write_recording_frame()

                now = time.perf_counter()
                self._fps = 0.9 * self._fps + 0.1 / max(now - last_time, 1e-4)
                last_time = now
                if now - self._last_runtime_log_at >= 15.0:
                    LOGGER.info(
                        "simulation heartbeat: time=%.2fs fps=%.1f camera_distance=%.3f command_age=%.3fs frame_ready=%s",
                        float(self.scene.t) * self.scene.dt,
                        self._fps,
                        self._camera_distance,
                        time.monotonic() - self._last_command_time,
                        self.get_frame() is not None,
                    )
                    self._last_runtime_log_at = now

                next_time += period
                sleep = next_time - time.perf_counter()
                if sleep > 0.0:
                    time.sleep(sleep)
                else:
                    next_time = time.perf_counter()
        except KeyboardInterrupt:
            LOGGER.info("shutdown requested by keyboard interrupt")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("simulation loop failed: %s", error)
            with self._lock:
                self._state = {**self._state, "error": error, "enabled": False}
            self._running = False
            self._simulation_loop_active = False
            LOGGER.error("diagnostics remain available at /api/diagnostics; press Ctrl+C to exit")
            try:
                while True:
                    time.sleep(0.25)
            except KeyboardInterrupt:
                LOGGER.info("shutdown requested after simulation error")
        finally:
            self._running = False
            self._simulation_loop_active = False
            if self._recorder is not None:
                self._stop_recording()
            self._api_server.should_exit = True
            if self._api_thread is not None:
                self._api_thread.join(timeout=5)


def create_app(platform: FrankaTeleopPlatform) -> FastAPI:
    """Build the HTTP API and serve the compiled React workbench."""
    app = FastAPI(title="Genesis Teleoperation API", version="1.2.0")
    agent_plan_lock = threading.Lock()
    agent_plans = {}
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def log_http_problems(request, call_next):
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            LOGGER.exception("HTTP request crashed: %s %s", request.method, request.url.path)
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if response.status_code >= 400 or elapsed_ms >= 1000.0:
            LOGGER.warning(
                "HTTP request issue: %s %s status=%d elapsed_ms=%.1f",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
            )
        return response

    @app.get("/api/health")
    async def health():
        state = platform.get_state()
        recording_formats = ["mp4"]
        if importlib.util.find_spec("mcap"):
            recording_formats.append("mcap")
        if importlib.util.find_spec("pyarrow"):
            recording_formats.append("lerobot")
        return {
            "running": platform._running,
            "ready": bool(state.get("ee")) and not state.get("error"),
            "error": state.get("error", ""),
            "features": {
                "smooth_home": True,
                "first_person_modes": ["rgb", "depth", "pointcloud", "segmentation"],
                "collision_zero_force": True,
                "recording_formats": recording_formats,
                "embodied_agent": True,
                "closed_loop_agent": True,
                "natural_programs": True,
                "vlm_configured": bool(VLM_MODEL),
            },
        }

    @app.get("/api/diagnostics")
    async def diagnostics():
        return platform.get_diagnostics()

    @app.get("/api/state")
    async def state():
        return platform.get_state()

    @app.get("/api/agent/status")
    async def agent_status():
        return {
            "provider": "ollama",
            "configured": bool(VLM_MODEL),
            "model": VLM_MODEL,
            "endpoint": VLM_URL if VLM_MODEL else "",
            "max_move_m": AGENT_MOVE_MAX_M,
        }

    @app.get("/api/agent/programs")
    async def agent_programs():
        return {"programs": PROGRAMS, "execution": "local_whitelist"}

    @app.post("/api/agent/program")
    async def agent_program(payload: AgentAnalyzePayload):
        try:
            request = platform.queue_natural_program(payload.task)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"ok": True, "queued": True, **request}

    @app.post("/api/agent/analyze")
    async def agent_analyze(payload: AgentAnalyzePayload):
        if not payload.task.strip():
            raise HTTPException(status_code=422, detail="任务描述不能为空")
        if not VLM_MODEL:
            raise HTTPException(
                status_code=503,
                detail="未配置 VLM。安装并启动 Ollama，再设置 GENESIS_TELEOP_VLM_MODEL 为已下载的视觉模型名称。",
            )
        state_snapshot = platform.get_state()
        if not state_snapshot.get("ee"):
            raise HTTPException(status_code=503, detail="Genesis 仿真尚未就绪")
        frame = platform.get_frame()
        if frame is None:
            raise HTTPException(status_code=503, detail="相机画面尚未就绪")
        state_for_model = {
            "ee_position_world_m": state_snapshot.get("ee"),
            "ee_orientation_wxyz": state_snapshot.get("quat"),
            "gripper_opening_m": state_snapshot.get("gripper"),
            "visible_scene_objects_from_sim": state_snapshot.get("objects", []),
            "workspace_bounds_m": {"low": WORKSPACE_LOW.tolist(), "high": WORKSPACE_HIGH.tolist()},
            "collision_protection_active": state_snapshot.get("collision_mode", False),
        }
        schema = AgentPlan.model_json_schema() if hasattr(AgentPlan, "model_json_schema") else AgentPlan.schema()
        prompt = (
            "You are the visual task planner for a Franka Panda robot in a Genesis simulation. "
            "Inspect the attached overhead RGB camera image and the measured state. Return exactly one next step, "
            "not a whole open-loop plan. The operator will review and confirm it. "
            f"Requested task: {payload.task.strip()}\n"
            f"Measured state: {json.dumps(state_for_model, ensure_ascii=False)}\n"
            "Actions: move (delta_m is a relative XYZ world-frame displacement), open_gripper, close_gripper, observe, or complete. "
            f"For move, every axis must be between -{AGENT_MOVE_MAX_M:.2f} and +{AGENT_MOVE_MAX_M:.2f} meters and the total displacement must be at most 0.04 m. "
            "If the scene or target is unclear, choose observe with an empty delta_m. Choose complete only when the requested task is clearly finished. Do not output joint angles, forces, "
            "absolute coordinates, or claims that an object was grasped unless the image clearly shows it. "
            "Use Chinese for summary and reason. Keep summary within 60 Chinese characters and reason within 100 Chinese characters. "
            "Return one compact JSON object only, with no analysis, markdown, or extra text. The required JSON shape is: "
            + json.dumps(schema, ensure_ascii=False)
        )
        request_body = {
            "model": VLM_MODEL,
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": [base64.b64encode(frame).decode("ascii")],
            }],
            "stream": False,
            # Some Ollama reasoning models put their output in `thinking` and
            # leave `message.content` empty unless thinking is explicitly off.
            "think": False,
            "format": schema,
            "options": {"temperature": 0, "num_predict": 512},
        }

        def call_vlm():
            request = Request(
                VLM_URL,
                data=json.dumps(request_body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=VLM_TIMEOUT_S) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                detail = exc.read(500).decode("utf-8", errors="replace")
                raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
            except (URLError, TimeoutError, OSError) as exc:
                raise RuntimeError(f"无法连接 Ollama：{exc}") from exc

        started = time.perf_counter()
        try:
            response = await asyncio.to_thread(call_vlm)
        except RuntimeError as exc:
            LOGGER.warning("VLM request failed: model=%s error=%s", VLM_MODEL, exc)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("VLM response was invalid: model=%s", VLM_MODEL)
            raise HTTPException(status_code=502, detail=f"VLM 返回内容无效：{type(exc).__name__}") from exc

        message = response.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            done_reason = response.get("done_reason", "unknown")
            eval_count = response.get("eval_count", "unknown")
            thinking_chars = len(message.get("thinking") or "")
            LOGGER.warning(
                "VLM returned empty content: model=%s done_reason=%s eval_count=%s thinking_chars=%d",
                VLM_MODEL,
                done_reason,
                eval_count,
                thinking_chars,
            )
            if thinking_chars and done_reason == "length":
                hint = (
                    "模型把输出长度用在思考内容上，没有生成可执行的 JSON。"
                    "请切换到视觉模型的 Instruct 版本；若使用 Qwen3-VL 2B，请执行 `ollama pull qwen3-vl:2b-instruct`，"
                    "并将 GENESIS_TELEOP_VLM_MODEL 设为 qwen3-vl:2b-instruct 后重启后端。"
                )
            else:
                hint = "模型没有返回可用正文，请检查 Ollama 日志和模型版本。"
            raise HTTPException(
                status_code=502,
                detail=f"VLM 没有返回可用正文（done_reason={done_reason}, eval_count={eval_count}, thinking_chars={thinking_chars}）。{hint}",
            )

        try:
            plan = AgentPlan.model_validate_json(content) if hasattr(AgentPlan, "model_validate_json") else AgentPlan.parse_raw(content)
        except Exception as exc:
            done_reason = response.get("done_reason", "unknown")
            eval_count = response.get("eval_count", "unknown")
            LOGGER.warning(
                "VLM response failed schema validation: model=%s done_reason=%s eval_count=%s error=%s content_chars=%d",
                VLM_MODEL,
                done_reason,
                eval_count,
                exc,
                len(content),
            )
            if done_reason == "length":
                raise HTTPException(
                    status_code=502,
                    detail=f"VLM 输出达到长度上限后仍未完成 JSON（已生成 {eval_count} tokens）。请重试；若持续发生，请切换到更适合结构化输出的视觉 Instruct 模型。",
                ) from exc
            raise HTTPException(status_code=502, detail=f"VLM 返回内容无效：{type(exc).__name__}") from exc

        try:
            validate_agent_plan(plan.action, plan.delta_m)
        except ValueError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        LOGGER.info(
            "VLM scene analysis completed: model=%s elapsed_s=%.2f action=%s target=%s confidence=%.2f",
            VLM_MODEL,
            time.perf_counter() - started,
            plan.action,
            plan.target_object,
            plan.confidence,
        )
        suggestion_id = uuid.uuid4().hex
        now = time.monotonic()
        state_copy = json.loads(json.dumps(state_snapshot))
        with agent_plan_lock:
            for stale_id in [key for key, value in agent_plans.items() if now - value["created_at"] > 45.0]:
                agent_plans.pop(stale_id, None)
            agent_plans[suggestion_id] = {
                "created_at": now,
                "ee": state_copy.get("ee"),
                "objects": state_copy.get("objects", []),
                "plan": plan.model_dump() if hasattr(plan, "model_dump") else plan.dict(),
            }
        return {**agent_plans[suggestion_id]["plan"], "suggestion_id": suggestion_id, "expires_in_s": 45}

    @app.post("/api/agent/execute")
    async def agent_execute(payload: AgentExecutePayload):
        with agent_plan_lock:
            suggestion = agent_plans.get(payload.suggestion_id)
            if suggestion is None:
                raise HTTPException(status_code=409, detail="此建议已失效，请重新观察场景")
            if time.monotonic() - suggestion["created_at"] > 45.0:
                agent_plans.pop(payload.suggestion_id, None)
                raise HTTPException(status_code=409, detail="此建议已过期，请重新观察场景")
            agent_plans.pop(payload.suggestion_id, None)

        plan = suggestion["plan"]
        if plan["action"] in ("observe", "complete"):
            raise HTTPException(status_code=409, detail="模型建议观察或已判定任务完成，没有可执行动作")
        current_state = platform.get_state()
        old_ee = np.asarray(suggestion.get("ee") or [], dtype=np.float64)
        current_ee = np.asarray(current_state.get("ee") or [], dtype=np.float64)
        if old_ee.shape != (3,) or current_ee.shape != (3,) or np.linalg.norm(current_ee - old_ee) > 0.02:
            raise HTTPException(status_code=409, detail="分析后末端位置已变化，请重新观察后再执行")
        old_objects = suggestion.get("objects", [])
        current_objects = current_state.get("objects", [])
        if len(old_objects) != len(current_objects) or any(
            np.linalg.norm(np.asarray(previous.get("position", []), dtype=np.float64) - np.asarray(current.get("position", []), dtype=np.float64)) > 0.01
            for previous, current in zip(old_objects, current_objects)
        ):
            raise HTTPException(status_code=409, detail="分析后场景物体位置已变化，请重新观察后再执行")

        action = plan["action"]
        params = {"delta_m": plan["delta_m"]} if action == "move" else None
        try:
            platform.queue_agent_action(action, params)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"ok": True, "queued": True, "action": action}

    @app.post("/api/agent/stop")
    async def agent_stop():
        platform.queue_agent_stop()
        return {"ok": True, "queued": True}

    @app.post("/api/command")
    async def command(payload: CommandPayload):
        platform.set_command(payload)
        return {"ok": True}

    @app.post("/api/action")
    async def action(payload: ActionPayload):
        platform.queue_action(payload.action)
        return {"ok": True}

    @app.post("/api/recording/start")
    async def recording_start(payload: RecordingStartPayload):
        if payload.format == "mcap":
            available = importlib.util.find_spec("mcap") is not None
            if not available:
                raise HTTPException(status_code=501, detail="Install the teleop recording dependencies first")
        if payload.format == "lerobot":
            available = importlib.util.find_spec("pyarrow") is not None
            if not available:
                raise HTTPException(status_code=501, detail="Install the teleop recording dependencies first")
        accepted = platform.queue_recording_start(payload.format, payload.task)
        if not accepted:
            raise HTTPException(status_code=409, detail="A recording is already active or starting")
        return {"ok": True, "queued": True}

    @app.post("/api/recording/stop")
    async def recording_stop():
        if not platform.queue_recording_stop():
            raise HTTPException(status_code=409, detail="No recording is active")
        return {"ok": True, "queued": True}

    @app.post("/api/sensor/mode")
    async def sensor_mode(payload: SensorModePayload):
        platform.set_sensor_mode(payload.mode)
        return {"ok": True, "mode": payload.mode}

    @app.post("/api/collision/reset")
    async def collision_reset():
        accepted = platform.queue_collision_reset()
        if not accepted:
            raise HTTPException(status_code=409, detail="Collision zero-force mode is not active")
        return {"ok": True, "queued": True}

    @app.post("/api/camera")
    async def camera_control(payload: CameraControlPayload):
        platform.queue_camera_control(payload)
        return {"ok": True}

    @app.get("/api/snapshot.jpg")
    async def snapshot():
        frame = platform.get_frame()
        if frame is None:
            raise HTTPException(status_code=503, detail="Camera frame is not ready")
        return StreamingResponse(iter([frame]), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    async def frames():
        while platform._running:
            frame = platform.get_frame()
            if frame is None:
                await asyncio.sleep(0.02)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n"
            yield f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
            yield frame + b"\r\n"
            await asyncio.sleep(1.0 / CONTROL_HZ)

    @app.get("/api/stream.mjpg")
    async def stream():
        return StreamingResponse(
            frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    async def first_person_frames():
        while platform._running:
            frame = platform.get_first_person_frame()
            if frame is None:
                await asyncio.sleep(0.04)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n"
            yield f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
            yield frame + b"\r\n"
            await asyncio.sleep(1.0 / CONTROL_HZ)

    @app.get("/api/first-person.mjpg")
    async def first_person_stream():
        return StreamingResponse(
            first_person_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    web_root = Path(__file__).resolve().parent.parent / "teleop" / "web"
    dist_root = web_root / "dist"
    if dist_root.is_dir():
        app.mount("/assets", StaticFiles(directory=dist_root / "assets"), name="frontend-assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def frontend(path: str):
            target = (dist_root / path).resolve()
            if dist_root.resolve() not in target.parents and target != dist_root.resolve():
                raise HTTPException(status_code=404, detail="Not found")
            if target.is_file():
                return FileResponse(target)
            return FileResponse(dist_root / "index.html")
    else:
        @app.get("/", include_in_schema=False)
        async def frontend_missing():
            return {
                "detail": "React frontend is not built. Run `npm install` and `npm run build` in examples/teleop/web.",
                "api_docs": "/docs",
            }

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8010, help="HTTP port of the console")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address; use 0.0.0.0 for LAN access")
    parser.add_argument("--cpu", action="store_true", help="Run on the CPU backend")
    parser.add_argument("--vis", action="store_true", help="Also show the Genesis viewer")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help=f"Console log level; detailed rotating logs are always written to {LOG_PATH}",
    )
    args = parser.parse_args()

    configure_logging(args.log_level)
    LOGGER.info(
        "teleop startup: host=%s port=%d backend=%s viewer=%s log_level=%s log_file=%s",
        args.host,
        args.port,
        "cpu" if args.cpu else "gpu",
        args.vis,
        args.log_level,
        LOG_PATH,
    )
    try:
        platform = FrankaTeleopPlatform(args)
        platform.run()
    except KeyboardInterrupt:
        LOGGER.info("shutdown requested during startup")
    except Exception:
        LOGGER.exception("fatal teleoperation startup error")
        raise


if __name__ == "__main__":
    main()
