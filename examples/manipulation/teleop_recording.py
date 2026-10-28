"""Session writers for the Genesis teleoperation workbench."""

from __future__ import annotations

import base64
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "out" / "teleop"
JOINT_NAMES = [f"joint_{index}.pos" for index in range(1, 8)] + ["gripper.pos"]


def _session_stem() -> str:
    return datetime.now().strftime("teleop_%Y%m%d_%H%M%S_%f")


def create_recording_session(
    recording_format: str,
    task: str,
    fps: float,
    resolution: tuple[int, int],
):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stem = _session_stem()
    if recording_format == "mp4":
        return Mp4Session(OUTPUT_ROOT / f"{stem}.mp4", fps, resolution)
    if recording_format == "mcap":
        return McapSession(OUTPUT_ROOT / f"{stem}.mcap", task)
    if recording_format == "lerobot":
        return LeRobotV21Session(OUTPUT_ROOT / f"{stem}_lerobot", task, fps, resolution)
    raise ValueError(f"Unsupported recording format: {recording_format}")


class Mp4Session:
    recording_format = "mp4"

    def __init__(self, path: Path, fps: float, resolution: tuple[int, int]):
        self.path = path
        self.frame_count = 0
        self._writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, resolution
        )
        if not self._writer.isOpened():
            self._writer.release()
            raise RuntimeError(f"Could not open MP4 writer: {path}")

    def write_frame(self, observation, action, main_bgr, main_jpeg, first_person_jpeg, timestamp_ns):
        self._writer.write(main_bgr)
        self.frame_count += 1

    def finish(self):
        self._writer.release()


class McapSession:
    recording_format = "mcap"

    _STATE_SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "TeleopState",
        "type": "object",
        "properties": {"timestamp_ns": {"type": "integer"}},
        "required": ["timestamp_ns"],
        "additionalProperties": True,
    }
    _ACTION_SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "TeleopAction",
        "type": "object",
        "properties": {"timestamp_ns": {"type": "integer"}},
        "required": ["timestamp_ns"],
        "additionalProperties": True,
    }
    _IMAGE_SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "foxglove.CompressedImage",
        "type": "object",
        "properties": {
            "timestamp": {
                "type": "object",
                "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}},
                "required": ["sec", "nsec"],
            },
            "frame_id": {"type": "string"},
            "data": {"type": "string", "contentEncoding": "base64"},
            "format": {"type": "string"},
        },
        "required": ["timestamp", "frame_id", "data", "format"],
    }

    def __init__(self, path: Path, task: str):
        try:
            from mcap.writer import CompressionType, Writer
        except ImportError as exc:
            raise RuntimeError("MCAP recording requires the 'mcap' package") from exc

        self.path = path
        self.frame_count = 0
        self._stream = path.open("wb")
        self._writer = Writer(self._stream, compression=CompressionType.NONE)
        self._writer.start(library="genesis-teleop")
        self._writer.add_metadata(
            "teleop_session",
            {"robot": "franka_panda", "task": task, "format": "mcap-json"},
        )
        self._state_channel = self._register_json_channel("/teleop/state", "TeleopState", self._STATE_SCHEMA)
        self._action_channel = self._register_json_channel("/teleop/action", "TeleopAction", self._ACTION_SCHEMA)
        self._image_channel = self._register_json_channel(
            "/camera/top/compressed", "foxglove.CompressedImage", self._IMAGE_SCHEMA
        )
        self._wrist_channel = self._register_json_channel(
            "/camera/first_person/compressed", "foxglove.CompressedImage", self._IMAGE_SCHEMA
        )

    def _register_json_channel(self, topic: str, schema_name: str, schema: dict[str, Any]) -> int:
        schema_id = self._writer.register_schema(
            name=schema_name,
            encoding="jsonschema",
            data=json.dumps(schema, separators=(",", ":")).encode("utf-8"),
        )
        return self._writer.register_channel(
            topic=topic,
            message_encoding="json",
            schema_id=schema_id,
        )

    @staticmethod
    def _write_json(writer, channel_id: int, timestamp_ns: int, message: dict[str, Any]):
        writer.add_message(
            channel_id=channel_id,
            log_time=timestamp_ns,
            publish_time=timestamp_ns,
            data=json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
        )

    def _write_image(self, channel_id: int, jpeg: bytes | None, timestamp_ns: int, frame_id: str):
        if not jpeg:
            return
        sec, nsec = divmod(timestamp_ns, 1_000_000_000)
        message = {
            "timestamp": {"sec": sec, "nsec": nsec},
            "frame_id": frame_id,
            "data": base64.b64encode(jpeg).decode("ascii"),
            "format": "jpeg",
        }
        self._write_json(self._writer, channel_id, timestamp_ns, message)

    def write_frame(self, observation, action, main_bgr, main_jpeg, first_person_jpeg, timestamp_ns):
        self._write_json(self._writer, self._state_channel, timestamp_ns, {"timestamp_ns": timestamp_ns, **observation})
        self._write_json(self._writer, self._action_channel, timestamp_ns, {"timestamp_ns": timestamp_ns, **action})
        self._write_image(self._image_channel, main_jpeg, timestamp_ns, "world")
        self._write_image(self._wrist_channel, first_person_jpeg, timestamp_ns, "hand")
        self.frame_count += 1

    def finish(self):
        try:
            self._writer.finish()
        finally:
            self._stream.close()


class LeRobotV21Session:
    """Write one standards-compatible LeRobot v2.1 episode (Parquet + MP4 + meta)."""

    recording_format = "lerobot"
    camera_key = "observation.images.top"

    def __init__(self, path: Path, task: str, fps: float, resolution: tuple[int, int]):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("LeRobot recording requires the 'pyarrow' package") from exc

        self.path = path
        self.task = task.strip() or "Teleoperation session"
        self.fps = int(round(fps))
        self.width, self.height = resolution
        self.frame_count = 0
        self._pa = pa
        self._pq = pq
        self._rows: list[dict[str, Any]] = []
        self._video_path = path / "videos" / "chunk-000" / self.camera_key / "episode_000000.mp4"
        self._video_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(
            str(self._video_path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, resolution
        )
        if not self._writer.isOpened():
            self._writer.release()
            raise RuntimeError(f"Could not open LeRobot video writer: {self._video_path}")

    def write_frame(self, observation, action, main_bgr, main_jpeg, first_person_jpeg, timestamp_ns):
        frame_index = self.frame_count
        state = list(observation["joints"]) + [float(observation["gripper"])]
        target = list(action["joint_target"]) + [float(action["gripper_target"])]
        self._writer.write(main_bgr)
        self._rows.append(
            {
                "action": target,
                "observation.state": state,
                "observation.ee_pose": list(observation["ee"]) + list(observation["quat"]),
                "timestamp": frame_index / self.fps,
                "frame_index": frame_index,
                "episode_index": 0,
                "index": frame_index,
                "task_index": 0,
            }
        )
        self.frame_count += 1

    @staticmethod
    def _feature(dtype: str, shape: list[int], names=None, info=None):
        result = {"dtype": dtype, "shape": shape, "names": names}
        if info is not None:
            result["info"] = info
        return result

    @staticmethod
    def _stats(rows: list[dict[str, Any]], key: str):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        return {
            "min": np.min(values, axis=0).tolist(),
            "max": np.max(values, axis=0).tolist(),
            "mean": np.mean(values, axis=0).tolist(),
            "std": np.std(values, axis=0).tolist(),
            "count": [len(rows)],
        }

    def _write_metadata(self):
        if not self._rows:
            raise RuntimeError("No frames were captured; the LeRobot dataset is empty")

        root = self.path
        data_path = root / "data" / "chunk-000" / "episode_000000.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pa = self._pa
        columns = {
            "action": pa.array([row["action"] for row in self._rows], type=pa.list_(pa.float32())),
            "observation.state": pa.array(
                [row["observation.state"] for row in self._rows], type=pa.list_(pa.float32())
            ),
            "observation.ee_pose": pa.array(
                [row["observation.ee_pose"] for row in self._rows], type=pa.list_(pa.float32())
            ),
            "timestamp": pa.array([row["timestamp"] for row in self._rows], type=pa.float32()),
            "frame_index": pa.array([row["frame_index"] for row in self._rows], type=pa.int64()),
            "episode_index": pa.array([row["episode_index"] for row in self._rows], type=pa.int64()),
            "index": pa.array([row["index"] for row in self._rows], type=pa.int64()),
            "task_index": pa.array([row["task_index"] for row in self._rows], type=pa.int64()),
        }
        self._pq.write_table(pa.table(columns), data_path, compression="zstd")

        stats_keys = [
            "action",
            "observation.state",
            "observation.ee_pose",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ]
        stats = {key: self._stats(self._rows, key) for key in stats_keys}
        metadata = root / "meta"
        metadata.mkdir(parents=True, exist_ok=True)
        (metadata / "tasks.jsonl").write_text(
            json.dumps({"task_index": 0, "task": self.task}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (metadata / "episodes.jsonl").write_text(
            json.dumps(
                {"episode_index": 0, "tasks": [self.task], "length": self.frame_count},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (metadata / "episodes_stats.jsonl").write_text(
            json.dumps({"episode_index": 0, "stats": stats}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (metadata / "stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        video_info = {
            "video.fps": self.fps,
            "video.height": self.height,
            "video.width": self.width,
            "video.channels": 3,
            "video.codec": "mp4v",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        }
        features = {
            "action": self._feature("float32", [8], JOINT_NAMES),
            "observation.state": self._feature("float32", [8], JOINT_NAMES),
            "observation.ee_pose": self._feature(
                "float32", [7], ["x", "y", "z", "qx", "qy", "qz", "qw"]
            ),
            self.camera_key: self._feature(
                "video",
                [self.height, self.width, 3],
                ["height", "width", "channels"],
                video_info,
            ),
            "timestamp": self._feature("float32", [1]),
            "frame_index": self._feature("int64", [1]),
            "episode_index": self._feature("int64", [1]),
            "index": self._feature("int64", [1]),
            "task_index": self._feature("int64", [1]),
        }
        info = {
            "codebase_version": "v2.1",
            "robot_type": "franka_panda",
            "total_episodes": 1,
            "total_frames": self.frame_count,
            "total_tasks": 1,
            "total_videos": 1,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": self.fps,
            "splits": {"train": "0:1"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
        }
        (metadata / "info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def finish(self):
        self._writer.release()
        self._write_metadata()
