"""Autonomous tabletop pick-and-place with a Franka Panda.

The demo runs a complete planning pipeline on a scene with three objects, without any
hard-coded grasp pose:

1. Perception: a fixed camera renders RGB-D. FastSAM, a class-agnostic instance segmenter,
   proposes masks on the RGB image; every mask is back-projected through the depth map into a
   world-frame point cloud, clipped to a calibrated tabletop workspace, and deduplicated against
   the larger masks that already cover the same object. The support plane is the dominant depth
   histogram peak, and projecting the visible surface onto it completes the occluded lower half.
   The yaw reference of each object comes from a principal component analysis of its horizontal
   footprint.
2. Candidate generation: parallel-jaw grasps are sampled over five approach directions and
   a yaw sweep of the closing axis. Each candidate is the 6-DoF gripper pose whose finger
   pads straddle the object at the center of its projection onto the grasp frame.
3. Geometric filtering: gripper stroke, finger clearance above the support surface, wrist
   clearance, and free space along the approach corridor.
4. Validation: inverse kinematics for the pregrasp and grasp poses, plus collision queries
   for both poses and for the joint-space approach between them. The robot state is
   restored after every query, so validation is side-effect free.
5. Ranking: IK accuracy, grasp width, approach direction, face alignment and joint-space travel.
6. Execution: RRTConnect to the pregrasp, Cartesian approach, force-closed grip, lift, a
   fixed-height transfer to the drop zone, and release.
7. Verification: the object must rise with the gripper and land inside the drop zone.

Usage
-----
    python examples/manipulation/auto_grasp_planning.py --vis
    python examples/manipulation/auto_grasp_planning.py --num-picks 3 --seed 7
    python examples/manipulation/auto_grasp_planning.py --segmentation sim
    python examples/manipulation/auto_grasp_planning.py --perception ground_truth

The run records 'out/auto_grasp.mp4' unless '--vis' or '--no-video' is given. FastSAM requires
the 'ultralytics' package; its weights are downloaded on first use, or passed with '--weights'.
"""

import argparse
from dataclasses import dataclass
import math
import os
import time
from typing import Any

import numpy as np
import torch

import genesis as gs
from genesis.utils.geom import R_to_quat, quat_to_R
from genesis.utils.misc import tensor_to_array


ARM_DOFS = np.arange(7)
FINGER_DOFS = np.arange(7, 9)
GRIPPER_OPEN = 0.04
# Distance from the hand link origin to the center of the finger pads, along the hand +z axis.
TCP_OFFSET = 0.103
# Distance from the pad center to the fingertip, along the hand +z axis.
FINGER_REACH = 0.0054
# Usable stroke, kept inside the 0.08 m mechanical opening to leave the pads some margin.
GRIPPER_MIN_WIDTH = 0.016
GRIPPER_MAX_WIDTH = 0.072
PREGRASP_DISTANCE = 0.12
# Force applied by each finger once the object is straddled.
GRIP_FORCE = -8.0

TOP_YAW_SAMPLES = 12
SIDE_YAW_SAMPLES = 6
MIN_OBJECT_POINTS = 400
# Calibrated tabletop region in which the objects to pick are expected.
WORKSPACE_X = (0.32, 0.92)
WORKSPACE_Y = (-0.46, 0.46)
OBJECT_MAX_HEIGHT = 0.14
# Drop targets are spaced well beyond the gripper footprint so that placing one object cannot
# disturb an object that is already on the pad.
PLACE_TARGETS = ((0.44, -0.38), (0.44, -0.18), (0.44, 0.02))
APPROACH_DIRECTIONS = (
    np.array([0.0, 0.0, -1.0]),
    np.array([1.0, 0.0, 0.0]),
    np.array([-1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, -1.0, 0.0]),
)


@dataclass
class SceneObject:
    name: str
    entity: Any


@dataclass
class MaskedInstance:
    points: np.ndarray
    score: float
    mask: np.ndarray | None = None


@dataclass
class PerceivedObject:
    obj: SceneObject
    points: np.ndarray
    reference: np.ndarray
    top_z: float
    mask: np.ndarray | None = None


@dataclass
class DropZone:
    center_xy: np.ndarray
    top_z: float
    half_extents: np.ndarray


@dataclass
class Grasp:
    obj: SceneObject
    approach: np.ndarray
    closing: np.ndarray
    tcp: np.ndarray
    width: float
    approach_score: float
    width_score: float
    face_score: float = 0.0
    qpos_grasp: np.ndarray | None = None
    qpos_pregrasp: np.ndarray | None = None
    pos_err: float = math.inf
    rot_err: float = math.inf
    score: float = -math.inf

    @property
    def hand_pos(self):
        return self.tcp - TCP_OFFSET * self.approach

    @property
    def quat(self):
        x_axis = np.cross(self.closing, self.approach)
        return R_to_quat(np.stack((x_axis, self.closing, self.approach), axis=1))

    @property
    def transform(self):
        transform = np.eye(4)
        transform[:3, :3] = np.stack((np.cross(self.closing, self.approach), self.closing, self.approach), axis=1)
        transform[:3, 3] = self.tcp
        return transform


@dataclass
class PickResult:
    obj_name: str
    is_success: bool
    lift_dz: float
    place_err: float
    duration: float


def to_tensor(array):
    return torch.as_tensor(array, dtype=gs.tc_float, device=gs.device)


def build_scene(args):
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        vis_options=gs.options.VisOptions(
            segmentation_level="entity",
        ),
        viewer_options=gs.options.ViewerOptions(
            res=(960, 720),
            camera_pos=(2.0, -1.15, 1.05),
            camera_lookat=(0.5, -0.05, 0.18),
            camera_fov=33,
            realtime_factor=1.0,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(
        morph=gs.morphs.Plane(
            pos=(0.0, 0.0, -0.04),
        ),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Rough(
            color=(0.16, 0.17, 0.19),
        ),
    )
    scene.add_entity(
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

    rng = np.random.default_rng(args.seed)
    build_specs = (
        (
            "box_a",
            gs.morphs.Box(size=(0.055, 0.04, 0.07), pos=(0.60, -0.17, 0.035), euler=(0, 0, rng.uniform(0, 180))),
            (0.20, 0.45, 0.85),
        ),
        ("cylinder", gs.morphs.Cylinder(radius=0.026, height=0.07, pos=(0.63, 0.03, 0.035)), (0.85, 0.42, 0.20)),
        (
            "box_b",
            gs.morphs.Box(size=(0.07, 0.035, 0.045), pos=(0.55, 0.22, 0.0225), euler=(0, 0, rng.uniform(0, 180))),
            (0.30, 0.70, 0.35),
        ),
    )

    objects = []
    for name, morph, color in build_specs:
        entity = scene.add_entity(
            morph=morph,
            material=gs.materials.Rigid(
                rho=400,
                friction=1.2,
            ),
            surface=gs.surfaces.Rough(
                color=color,
            ),
        )
        objects.append(SceneObject(name=name, entity=entity))

    drop_zone = DropZone(
        center_xy=np.array([0.44, -0.18]),
        top_z=0.01,
        half_extents=np.array([0.10, 0.30]),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(2.0 * drop_zone.half_extents[0], 2.0 * drop_zone.half_extents[1], drop_zone.top_z),
            pos=(drop_zone.center_xy[0], drop_zone.center_xy[1], 0.5 * drop_zone.top_z),
            fixed=True,
        ),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Rough(
            color=(0.80, 0.80, 0.82),
        ),
    )

    franka = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )

    cam = None
    if not args.vis and not args.no_video:
        cam = scene.add_camera(
            res=(960, 720),
            pos=(2.0, -1.15, 1.05),
            lookat=(0.5, -0.05, 0.18),
            fov=33,
            GUI=False,
        )
    perception_cam = scene.add_camera(
        res=(960, 720),
        pos=(0.95, -0.75, 1.25),
        lookat=(0.56, 0.0, 0.0),
        fov=40,
        GUI=False,
    )

    scene.build()
    return scene, franka, objects, drop_zone, cam, perception_cam


def setup_robot(franka):
    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
    franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )
    home = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, GRIPPER_OPEN, GRIPPER_OPEN])
    franka.set_qpos(to_tensor(home), zero_velocity=True)


def get_world_verts(entity):
    verts = [tensor_to_array(geom.get_verts()) for link in entity.links for geom in link.geoms]
    return np.concatenate(verts, axis=0)


class FastSamSegmenter:
    """Class-agnostic instance segmentation of an RGB image with FastSAM."""

    def __init__(self, weights):
        try:
            from ultralytics import FastSAM
        except ImportError as error:
            raise ImportError(
                "FastSAM segmentation requires 'ultralytics'; install it with 'pip install ultralytics'."
            ) from error
        self._model = FastSAM(weights)
        self._device = "cuda:0" if gs.device.type == "cuda" else "cpu"

    def __call__(self, rgb):
        result = self._model(
            rgb[..., ::-1],
            device=self._device,
            retina_masks=True,
            imgsz=1024,
            conf=0.2,
            iou=0.8,
            verbose=False,
        )[0]
        if result.masks is None:
            return np.zeros((0, *rgb.shape[:2]), dtype=bool), np.zeros((0,))
        return tensor_to_array(result.masks.data) > 0.5, tensor_to_array(result.boxes.conf)


def estimate_support_z(points_world, valid_mask):
    x = points_world[..., 0][valid_mask]
    y = points_world[..., 1][valid_mask]
    z = points_world[..., 2][valid_mask]
    table_roi = np.logical_and.reduce((x > 0.20, x < 0.95, np.abs(y) < 0.50, z > -0.05, z < 0.15))
    hist, edges = np.histogram(z[table_roi], bins=np.arange(-0.05, 0.151, 0.004))
    peak = int(np.argmax(hist))
    return float(0.5 * (edges[peak] + edges[peak + 1]))


def mask_instances(points_world, valid_mask, masks, scores, support_z):
    instances = []
    for idx in np.argsort(scores)[::-1]:
        points = points_world[np.logical_and(masks[idx], valid_mask)]
        points = points[
            (points[:, 0] > WORKSPACE_X[0])
            & (points[:, 0] < WORKSPACE_X[1])
            & (np.abs(points[:, 1]) < WORKSPACE_Y[1])
            & (points[:, 2] > support_z + 0.004)
            & (points[:, 2] < support_z + OBJECT_MAX_HEIGHT)
        ]
        if len(points) < MIN_OBJECT_POINTS:
            continue
        if points[:, 2].max() - support_z < 0.02:
            continue
        extent = points[:, :2].max(axis=0) - points[:, :2].min(axis=0)
        if extent.max() > 0.12:
            continue
        # Masks are kept from best to worst, so anything already contained in a kept object is a
        # fragment of that object (a part, or object plus its shadow).
        is_duplicate = False
        for instance in instances:
            inside = np.logical_and(
                points[:, :2] > instance.points[:, :2].min(axis=0),
                points[:, :2] < instance.points[:, :2].max(axis=0),
            ).all(axis=1)
            if inside.mean() > 0.5:
                is_duplicate = True
                break
        if not is_duplicate:
            instances.append(MaskedInstance(points=points, score=float(scores[idx]), mask=masks[idx]))
    return instances


def perceive_objects(scene, cam, objects, drop_zone, args, segmenter):
    if args.perception == "ground_truth":
        perceived = []
        for obj in objects:
            points = get_world_verts(obj.entity)
            obj_rot = quat_to_R(tensor_to_array(obj.entity.get_quat(relative=False)))
            perceived.append(
                PerceivedObject(obj=obj, points=points, reference=obj_rot[:, 0], top_z=float(points[:, 2].max()))
            )
        return perceived

    points_world, valid_mask = cam.render_pointcloud(world_frame=True)
    support_z = estimate_support_z(points_world, valid_mask)
    if args.segmentation == "sim":
        _, _, seg, _ = cam.render(rgb=False, depth=False, segmentation=True)
        idxc_by_entity = {seg_key: idxc for idxc, seg_key in scene.visualizer.segmentation_idx_dict.items()}
        instances = []
        for obj in objects:
            mask = np.logical_and(seg == idxc_by_entity[obj.entity.idx], valid_mask)
            instances.append(MaskedInstance(points=points_world[mask], score=1.0, mask=mask))
    else:
        rgb = cam.render(rgb=True)[0]
        masks, scores = segmenter(rgb)
        instances = mask_instances(points_world, valid_mask, masks, scores, support_z)
        print(f"[perception] FastSAM: {len(instances)} candidate masks")

    perceived = []
    for instance in instances:
        points = instance.points
        if len(points) < MIN_OBJECT_POINTS:
            continue
        # A mask sometimes includes a few pixels of a neighbouring surface; drop points that sit
        # far from the bulk of the instance before fitting the grasp frame.
        median = np.median(points, axis=0)
        points = points[np.linalg.norm(points - median, axis=1) < 0.08]
        if len(points) < MIN_OBJECT_POINTS:
            continue
        support_points = points.copy()
        support_points[:, 2] = support_z
        cloud = np.concatenate((points, support_points), axis=0)

        xy = points[:, :2] - points[:, :2].mean(axis=0)
        _, eigenvectors = np.linalg.eigh(xy.T @ xy)
        reference = np.array([eigenvectors[0, -1], eigenvectors[1, -1], 0.0])
        if np.linalg.norm(reference) < 1e-6:
            reference = np.array([1.0, 0.0, 0.0])
        reference /= np.linalg.norm(reference)

        center = 0.5 * (cloud.min(axis=0) + cloud.max(axis=0))
        # Objects already resting in the drop area are picked objects, not candidates. The margin
        # keeps the gripper's pixels from ever being mistaken for an object either.
        if np.all(np.abs(center[:2] - drop_zone.center_xy) < drop_zone.half_extents + 0.02):
            continue
        # The entity handle is only used to score the outcome of a grasp (lift, place); planning
        # uses the masked cloud and never reads it.
        obj = min(
            objects,
            key=lambda candidate: np.linalg.norm(
                tensor_to_array(candidate.entity.get_pos(relative=False))[:2] - center[:2]
            ),
        )
        top_z = float(points[:, 2].max())
        perceived.append(PerceivedObject(obj=obj, points=cloud, reference=reference, top_z=top_z, mask=instance.mask))
        gt_center = tensor_to_array(obj.entity.get_pos(relative=False))
        print(
            f"[perception] {obj.name:<9} {len(points):5d} points | "
            f"center error {np.linalg.norm(center - gt_center) * 1000:4.1f} mm | top z {top_z:.3f} m"
        )
    return perceived


def tangent_basis(axis, reference):
    basis_u = reference - reference.dot(axis) * axis
    if np.linalg.norm(basis_u) < 1e-6:
        reference = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        basis_u = reference - reference.dot(axis) * axis
    basis_u /= np.linalg.norm(basis_u)
    return basis_u, np.cross(axis, basis_u)


def make_grasp(obj, verts, approach, closing):
    tangent = np.cross(closing, approach)
    closing_extent = verts @ closing
    approach_extent = verts @ approach
    tangent_extent = verts @ tangent
    tcp = (
        0.5 * (closing_extent.min() + closing_extent.max()) * closing
        + 0.5 * (approach_extent.min() + approach_extent.max()) * approach
        + 0.5 * (tangent_extent.min() + tangent_extent.max()) * tangent
    )
    width = float(closing_extent.max() - closing_extent.min())
    approach_score = 1.0 if approach[2] < -0.99 else 0.35
    width_score = 1.0 if width <= 0.05 else max(0.0, 1.0 - (width - 0.05) / (GRIPPER_MAX_WIDTH - 0.05))
    return Grasp(
        obj=obj,
        approach=approach,
        closing=closing,
        tcp=tcp,
        width=width,
        approach_score=approach_score,
        width_score=width_score,
    )


def is_geometrically_valid(grasp, other_verts):
    if not GRIPPER_MIN_WIDTH <= grasp.width <= GRIPPER_MAX_WIDTH:
        return False
    tcp, approach, closing = grasp.tcp, grasp.approach, grasp.closing
    # The lower pad and the fingertips must stay clear of the support plane at z = 0.
    if tcp[2] - 0.5 * grasp.width * abs(closing[2]) - 0.008 < 0.0:
        return False
    if (tcp + FINGER_REACH * approach)[2] < 0.01:
        return False
    hand_pos = grasp.hand_pos
    # A wrist that is too close to the table leaves no room for the hand body.
    if hand_pos[2] < 0.06:
        return False
    for s in (0.3, 0.6, 1.0):
        corridor_pos = hand_pos + s * (tcp - hand_pos)
        for verts in other_verts:
            if np.linalg.norm(verts - corridor_pos, axis=1).min() < 0.055:
                return False
    return True


def generate_grasps(obj, verts, reference_dir, other_verts):
    grasps = []
    for approach in APPROACH_DIRECTIONS:
        reference = reference_dir if approach[2] < -0.5 else np.array([0.0, 0.0, 1.0])
        basis_u, basis_v = tangent_basis(approach, reference)
        num_yaw = TOP_YAW_SAMPLES if approach[2] < -0.5 else SIDE_YAW_SAMPLES
        candidates = []
        for i_yaw in range(num_yaw):
            angle = math.pi * i_yaw / num_yaw
            closing = math.cos(angle) * basis_u + math.sin(angle) * basis_v
            closing = closing - closing.dot(approach) * approach
            closing /= np.linalg.norm(closing)
            grasp = make_grasp(obj, verts, approach, closing)
            if is_geometrically_valid(grasp, other_verts):
                candidates.append((angle, grasp))
        yaw_step = math.pi / num_yaw
        for i_candidate, (angle, grasp) in enumerate(candidates):
            # A face-aligned grasp is a local minimum of the opening width over the yaw sweep. A
            # grasp straddling two edges needs a wider opening than both of its neighbors, and its
            # line contacts make it prone to slipping while the object is carried, so it is kept
            # only as a lower-ranked fallback.
            width_prev = candidates[i_candidate - 1][1].width
            width_next = candidates[(i_candidate + 1) % len(candidates)][1].width
            is_face_aligned = grasp.width <= min(width_prev, width_next) + 0.003
            if is_face_aligned:
                # The point-cloud principal axis is a few degrees off the true face normal, so
                # refine the closing axis around the minimum to seat both pads flat on the object.
                for delta in np.linspace(-yaw_step, yaw_step, 9):
                    closing = math.cos(angle + delta) * basis_u + math.sin(angle + delta) * basis_v
                    closing = closing - closing.dot(approach) * approach
                    closing /= np.linalg.norm(closing)
                    candidate = make_grasp(obj, verts, approach, closing)
                    if is_geometrically_valid(candidate, other_verts) and candidate.width < grasp.width:
                        grasp = candidate
            grasp.face_score = 1.0 if is_face_aligned else 0.4
            grasp.score = 0.5 * grasp.approach_score + 0.3 * grasp.width_score + 0.2 * grasp.face_score
            grasps.append(grasp)
    return grasps


def solve_ik(franka, hand, pos, quat, init_qpos, max_samples=8):
    qpos, err = franka.inverse_kinematics(
        link=hand,
        pos=pos,
        quat=quat,
        init_qpos=init_qpos,
        dofs_idx_local=ARM_DOFS,
        return_error=True,
        max_samples=max_samples,
    )
    return tensor_to_array(qpos), tensor_to_array(err)


def collides_with_environment(franka, qpos_arm):
    qpos = np.concatenate((qpos_arm, [GRIPPER_OPEN, GRIPPER_OPEN]))
    franka.set_qpos(to_tensor(qpos), zero_velocity=True)
    pairs = franka.detect_collision(env_idx=0)
    if len(pairs) == 0:
        return False
    is_robot_a = np.logical_and(pairs[:, 0] >= franka.geom_start, pairs[:, 0] < franka.geom_end)
    is_robot_b = np.logical_and(pairs[:, 1] >= franka.geom_start, pairs[:, 1] < franka.geom_end)
    # Pairs with both geoms on the robot are self-collisions, which the default home pose already has.
    return bool(np.any(np.logical_not(np.logical_and(is_robot_a, is_robot_b))))


def validate_grasp(franka, hand, grasp, init_qpos):
    qpos_grasp, err_grasp = solve_ik(franka, hand, grasp.hand_pos, grasp.quat, init_qpos)
    grasp.pos_err = float(np.linalg.norm(err_grasp[:3]))
    grasp.rot_err = float(np.linalg.norm(err_grasp[3:]))
    if grasp.pos_err > 0.004 or grasp.rot_err > 0.03:
        return False

    pregrasp_pos = grasp.hand_pos - PREGRASP_DISTANCE * grasp.approach
    qpos_pregrasp, err_pregrasp = solve_ik(
        franka, hand, pregrasp_pos, grasp.quat, np.concatenate((qpos_grasp[:7], [GRIPPER_OPEN, GRIPPER_OPEN]))
    )
    if np.linalg.norm(err_pregrasp[:3]) > 0.004 or np.linalg.norm(err_pregrasp[3:]) > 0.03:
        return False

    arm_grasp, arm_pregrasp = qpos_grasp[:7], qpos_pregrasp[:7]
    for s in (0.0, 0.5, 1.0):
        if collides_with_environment(franka, arm_pregrasp + s * (arm_grasp - arm_pregrasp)):
            return False

    grasp.qpos_grasp = arm_grasp
    grasp.qpos_pregrasp = arm_pregrasp
    return True


def plan_grasps(franka, hand, perceived_objects, max_ik_checks):
    qpos_backup = franka.get_qpos().clone()
    qpos_nominal = tensor_to_array(qpos_backup)
    all_points = [perceived.points for perceived in perceived_objects]
    validated = []
    stats = []

    for i_obj, perceived in enumerate(perceived_objects):
        other_points = [points for j, points in enumerate(all_points) if j != i_obj]
        grasps = generate_grasps(perceived.obj, perceived.points, perceived.reference, other_points)
        grasps.sort(key=lambda grasp: grasp.score, reverse=True)
        num_valid = 0
        for grasp in grasps[:max_ik_checks]:
            if not validate_grasp(franka, hand, grasp, qpos_nominal):
                continue
            ik_score = 1.0 - max(grasp.pos_err / 0.004, grasp.rot_err / 0.03)
            joint_score = math.exp(-float(np.linalg.norm(grasp.qpos_grasp - qpos_nominal[:7])) / 2.5)
            grasp.score = (
                0.40 * ik_score
                + 0.20 * grasp.width_score
                + 0.15 * grasp.approach_score
                + 0.15 * grasp.face_score
                + 0.10 * joint_score
            )
            validated.append(grasp)
            num_valid += 1
        stats.append((perceived.obj.name, len(grasps), num_valid))

    franka.set_qpos(qpos_backup, zero_velocity=True)
    for name, num_candidates, num_valid in stats:
        print(f"[plan] {name:<9} {num_candidates:3d} candidates, {num_valid} validated")
    validated.sort(key=lambda grasp: grasp.score, reverse=True)
    return validated


def move_hand(franka, hand, grasp, qpos_fingers, target_pos, num_points, scene, grip_force=GRIP_FORCE):
    """Move the hand to a task-space target along a straight line in small IK steps.

    Each waypoint is solved from the previous joint configuration with joint-limit resampling
    disabled, so the solver cannot flip to another elbow branch. A waypoint that would need an
    implausible joint change is rejected instead of whipping the carried object around.
    """
    qpos_arm = tensor_to_array(franka.get_qpos())[:7]
    start_pos = tensor_to_array(hand.get_pos(relative=False))
    for s in np.linspace(0.0, 1.0, num_points + 1)[1:]:
        pos = start_pos + s * (target_pos - start_pos)
        qpos_target, err = solve_ik(
            franka, hand, pos, grasp.quat, np.concatenate((qpos_arm, qpos_fingers)), max_samples=1
        )
        if np.linalg.norm(err[:3]) > 0.008:
            print(f"[exec] transfer IK failed at s={s:.2f}: {np.linalg.norm(err[:3]):.4f} m")
            return None
        joint_step = float(np.linalg.norm(qpos_target[:7] - qpos_arm))
        if joint_step > 0.6:
            print(f"[exec] transfer rejected a {joint_step:.2f} rad joint jump at s={s:.2f}")
            return None
        qpos_arm = qpos_target[:7]
        for _ in range(6):
            franka.control_dofs_position(qpos_arm, ARM_DOFS)
            if grip_force is None:
                franka.control_dofs_position(np.array(qpos_fingers), FINGER_DOFS)
            else:
                franka.control_dofs_force(np.array([grip_force, grip_force]), FINGER_DOFS)
            scene.step()
    return qpos_arm


def execute_grasp(scene, franka, hand, grasp, drop_zone, place_xy, args):
    time_start = time.perf_counter()
    obj = grasp.obj
    obj_pos_before = tensor_to_array(obj.entity.get_pos(relative=False))

    pregrasp_goal = np.concatenate((grasp.qpos_pregrasp, [GRIPPER_OPEN, GRIPPER_OPEN]))
    path, is_valid = franka.plan_path(
        qpos_goal=pregrasp_goal,
        num_waypoints=120,
        max_retry=2,
        timeout=15.0,
        return_valid_mask=True,
    )
    if not bool(is_valid):
        print("[exec] motion planning to the pregrasp failed")
        return None

    for waypoint in path:
        franka.control_dofs_position(waypoint)
        scene.step()
    for _ in range(30):
        franka.control_dofs_position(pregrasp_goal)
        scene.step()

    grasp_goal = np.concatenate((grasp.qpos_grasp, [GRIPPER_OPEN, GRIPPER_OPEN]))
    approach_start = grasp.hand_pos - PREGRASP_DISTANCE * grasp.approach
    qpos_arm = grasp.qpos_pregrasp
    for s in (0.34, 0.67, 1.0):
        target = approach_start + s * (grasp.hand_pos - approach_start)
        qpos_target, err = solve_ik(
            franka, hand, target, grasp.quat, np.concatenate((qpos_arm, [GRIPPER_OPEN, GRIPPER_OPEN]))
        )
        if np.linalg.norm(err[:3]) > 0.01:
            print(f"[exec] approach IK failed at s={s:.2f}")
            return None
        qpos_arm = qpos_target[:7]
        for _ in range(10):
            franka.control_dofs_position(np.concatenate((qpos_arm, [GRIPPER_OPEN, GRIPPER_OPEN])))
            scene.step()
    for _ in range(20):
        franka.control_dofs_position(grasp_goal)
        scene.step()

    for _ in range(100):
        franka.control_dofs_position(grasp_goal)
        franka.control_dofs_force(np.array([GRIP_FORCE, GRIP_FORCE]), FINGER_DOFS)
        scene.step()

    # Let the contact impulses decay before the grip is loaded by the lift.
    for _ in range(40):
        franka.control_dofs_position(grasp_goal)
        franka.control_dofs_force(np.array([GRIP_FORCE, GRIP_FORCE]), FINGER_DOFS)
        scene.step()

    contacts = franka.get_contacts(with_entity=obj.entity)
    num_contacts = int(contacts["geom_a"].shape[0])
    qpos_fingers = tensor_to_array(franka.get_qpos())[7:]
    print(f"[exec] grip closed: {num_contacts} contacts, finger qpos {np.round(qpos_fingers, 4)}")

    lift_arm = move_hand(franka, hand, grasp, qpos_fingers, grasp.hand_pos + np.array([0.0, 0.0, 0.18]), 6, scene)
    if lift_arm is None:
        print("[exec] lift aborted")
        return None
    lift_dz = float(tensor_to_array(obj.entity.get_pos(relative=False))[2] - obj_pos_before[2])
    is_grasped = lift_dz > 0.05
    print(f"[exec] lift: {lift_dz:+.3f} m ({'grasped' if is_grasped else 'dropped'})")

    place_err = math.inf
    if is_grasped and args.place:
        transfer_arm = move_hand(
            franka, hand, grasp, qpos_fingers, np.array([place_xy[0], place_xy[1], 0.30]), 40, scene
        )
        if transfer_arm is not None:
            transfer_arm = move_hand(
                franka, hand, grasp, qpos_fingers, np.array([place_xy[0], place_xy[1], 0.16]), 20, scene
            )
        if transfer_arm is not None:
            release_goal = np.concatenate((transfer_arm, [GRIPPER_OPEN, GRIPPER_OPEN]))
            for _ in range(160):
                franka.control_dofs_position(release_goal)
                scene.step()
            move_hand(
                franka,
                hand,
                grasp,
                [GRIPPER_OPEN, GRIPPER_OPEN],
                np.array([place_xy[0], place_xy[1], 0.28]),
                20,
                scene,
                grip_force=None,
            )

            obj_pos = tensor_to_array(obj.entity.get_pos(relative=False))
            place_err = float(np.linalg.norm(obj_pos[:2] - place_xy))
            is_placed = (
                place_err < 0.10
                and np.all(np.abs(obj_pos[:2] - drop_zone.center_xy) < drop_zone.half_extents)
                and obj_pos[2] < drop_zone.top_z + 0.08
            )
            print(f"[exec] place: {place_err:.3f} m from the target ({'placed' if is_placed else 'missed'})")
            is_grasped = is_grasped and is_placed
        else:
            print("[exec] transfer failed, the object is still held")
            is_grasped = False

    return PickResult(
        obj_name=obj.name,
        is_success=is_grasped,
        lift_dz=lift_dz,
        place_err=place_err,
        duration=time.perf_counter() - time_start,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("--cpu", action="store_true", help="Run on the CPU backend")
    parser.add_argument("--num-picks", type=int, default=2, help="Number of objects to pick and place")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the object yaw angles")
    parser.add_argument("--no-place", dest="place", action="store_false", help="Stop after lifting")
    parser.add_argument("--no-video", action="store_true", help="Do not record a video")
    parser.add_argument(
        "--perception",
        choices=("camera", "ground_truth"),
        default="camera",
        help="Object source for planning",
    )
    parser.add_argument(
        "--segmentation",
        choices=("fastsam", "sim"),
        default="fastsam",
        help="Instance segmenter used by camera perception",
    )
    parser.add_argument("--weights", default="FastSAM-s.pt", help="FastSAM checkpoint")
    args = parser.parse_args()

    scene, franka, objects, drop_zone, cam, perception_cam = build_scene(args)
    setup_robot(franka)
    hand = franka.get_link("hand")
    segmenter = None
    if args.perception == "camera" and args.segmentation == "fastsam":
        segmenter = FastSamSegmenter(args.weights)

    if cam is not None:
        os.makedirs("out", exist_ok=True)
        cam.start_recording(save_to_filename=os.path.join("out", "auto_grasp.mp4"), fps=20)

    for _ in range(30):
        franka.control_dofs_position(tensor_to_array(franka.get_qpos()))
        scene.step()

    print(f"[scene] objects: {', '.join(obj.name for obj in objects)}")
    picked_names = set()
    results = []
    for i_pick in range(args.num_picks):
        remaining = [obj for obj in objects if obj.name not in picked_names]
        if not remaining:
            break
        print(f"\n[pick {i_pick + 1}] planning grasp")
        time_plan = time.perf_counter()
        perceived = perceive_objects(scene, perception_cam, remaining, drop_zone, args, segmenter)
        grasps = plan_grasps(franka, hand, perceived, max_ik_checks=8)
        print(f"[pick {i_pick + 1}] {len(grasps)} validated grasps in {time.perf_counter() - time_plan:.2f} s")
        if not grasps:
            print("[pick] no feasible grasp found")
            break

        best = grasps[0]
        print(
            f"[pick {i_pick + 1}] best: {best.obj.name} approach={np.round(best.approach, 2)} "
            f"closing={np.round(best.closing, 2)} width={best.width:.3f} m "
            f"tcp={np.round(best.tcp, 3)} score={best.score:.3f}"
        )
        debug_frames = [grasp.transform for grasp in grasps[:3] if grasp.obj is best.obj]
        scene.draw_debug_frames(debug_frames, axis_length=0.08, color=(0.7, 0.7, 0.7, 0.6))
        scene.draw_debug_frame(
            best.transform, axis_length=0.12, origin_size=0.012, axis_radius=0.004, color=(0.1, 1.0, 0.2, 0.9)
        )

        place_xy = np.array(PLACE_TARGETS[i_pick % len(PLACE_TARGETS)])
        result = None
        for grasp in grasps[:3]:
            result = execute_grasp(scene, franka, hand, grasp, drop_zone, place_xy, args)
            if result is not None:
                break
        scene.clear_debug_objects()

        if result is None:
            print("[pick] execution failed")
            break
        results.append(result)
        print(f"[pick {i_pick + 1}] {result.obj_name}: success={result.is_success} in {result.duration:.1f} s")
        if result.is_success:
            picked_names.add(result.obj_name)
        else:
            break

    if cam is not None:
        cam.stop_recording()
        print(f"\n[video] {os.path.abspath(os.path.join('out', 'auto_grasp.mp4'))}")

    print("\n[summary]")
    for result in results:
        place_text = f"{result.place_err:.3f} m" if math.isfinite(result.place_err) else "n/a"
        print(
            f"  {result.obj_name:<9} success={str(result.is_success):<5} "
            f"lift={result.lift_dz:+.3f} m place_err={place_text}"
        )
    is_all_success = len(results) == args.num_picks and all(result.is_success for result in results)
    print(f"[summary] {sum(result.is_success for result in results)}/{args.num_picks} successful picks")
    return 0 if is_all_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
