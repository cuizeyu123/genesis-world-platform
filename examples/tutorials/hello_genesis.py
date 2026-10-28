# import genesis as gs
# import numpy as np

# gs.init(backend=gs.cpu)

# scene = gs.Scene(
#     show_viewer=True,
#     viewer_options=gs.options.ViewerOptions(
#         camera_pos=(3.5, 0.0, 2.5),
#         camera_lookat=(0.0, 0.0, 0.5)
#     )
# )

# plane = scene.add_entity(gs.morphs.Plane())
# franka = scene.add_entity(
#     gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
# )

# scene.build()

# # ✅ 获取关节名称（字符串列表）
# joint_names = franka.get_joint_names()
# print(f"关节名称: {joint_names}")
# print(f"共有 {len(joint_names)} 个关节")

# # 或者获取自由度索引（另一种方式）
# # dof_indices = franka.get_dofs_idx()

# # 设置目标关节角度（注意：需要传入所有 9 个关节的值）
# target_qpos = np.array([0.0, 0.5, 0.0, -1.0, 0.0, 1.0, 0.0, 0.04, 0.04])

# # 控制循环
# for i in range(2000):
#     if i < 1000:
#         franka.set_dofs_target_position(
#             target_qpos * (i / 1000),    # 逐步移动到目标
#             joints=joint_names            # 传入关节名称列表
#         )
#     scene.step()
# import genesis as gs
# import numpy as np

# # gs.init(backend=gs.cpu)
# gs.init(backend=gs.gpu)
# scene = gs.Scene(
#     viewer_options=gs.options.ViewerOptions(
#         camera_pos=(0, -3.5, 2.5),
#         camera_lookat=(0.0, 0.0, 0.5),
#         camera_fov=30,
#         res           = (960, 640),
#         max_FPS       = 60,
#     ),
#     sim_options=gs.options.SimOptions(
#         dt=0.01,
#     ),
#     show_viewer=True,
# )
# plane = scene.add_entity(
#     gs.morphs.Plane(),
# )
# franka = scene.add_entity(
#     gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
# )
# scene.build()



# jnt_names = [
#     'joint1',
#     'joint2',
#     'joint3',
#     'joint4',
#     'joint5',
#     'joint6',
#     'joint7',
#     'finger_joint1',
#     'finger_joint2',
# ]
# dofs_idx = [franka.get_joint(name).dof_idx_local for name in jnt_names]

# ############ Optional: set control gains ############
# # 设置位置增益
# franka.set_dofs_kp(
#     np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
# )
# # 设置速度增益
# franka.set_dofs_kv(
#     np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
# )
# # 为安全设置力的范围
# franka.set_dofs_force_range(
#     np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
#     np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
# )
# # 硬重置
# for i in range(150):
#     if i < 50:
#         franka.set_dofs_position(np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]), dofs_idx)
#     elif i < 100:
#         franka.set_dofs_position(np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]), dofs_idx)
#     else:
#         franka.set_dofs_position(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), dofs_idx)

#     scene.step()

# # PD 控制
# for i in range(1250):
#     if i == 0:
#         franka.control_dofs_position(
#             np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]),
#             dofs_idx,
#         )
#     elif i == 250:
#         franka.control_dofs_position(
#             np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]),
#             dofs_idx,
#         )
#     elif i == 500:
#         franka.control_dofs_position(
#             np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
#             dofs_idx,
#         )
#     elif i == 750:
#         # 用速度控制第一个自由度，其余的用位置控制
#         franka.control_dofs_position(
#             np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])[1:],
#             dofs_idx[1:],
#         )
#         franka.control_dofs_velocity(
#             np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0])[:1],
#             dofs_idx[:1],
#         )
#     elif i == 1000:
#         franka.control_dofs_force(
#             np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
#             dofs_idx,
#         )
#     # 这是基于给定控制命令计算的控制力
#     # 如果使用力控制，它与给定的控制命令相同
#     print('control force:', franka.get_dofs_control_force(dofs_idx))

#     # 这是自由度实际受到的力
#     print('internal force:', franka.get_dofs_force(dofs_idx))

#     scene.step()



import numpy as np
import genesis as gs

########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################
scene = gs.Scene(
    sim_options = gs.options.SimOptions(
        dt = 0.01,
    ),
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (3, -1, 1.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 30,
        max_FPS       = 60,
    ),
    show_viewer = True,
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)
cube = scene.add_entity(
    gs.morphs.Box(
        size = (0.04, 0.04, 0.04),
        pos  = (0.65, 0.0, 0.02),
    )
)
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)
########################## build ##########################
scene.build()

motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)

# 设置控制增益
# 注意：以下值是针对 Franka 实现最佳行为而调整的
# 通常，每个新机器人都会有一组不同的参数。
# 有时高质量的 URDF 或 XML 文件也会提供这些参数并会被解析。
franka.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
)
franka.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
)
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
)


# 获取末端执行器连杆
end_effector = franka.get_link('hand')

# 移动到预抓取姿态
qpos = franka.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.65, 0.0, 0.25]),
    quat = np.array([0, 1, 0, 0]),
)
# 夹爪打开位置
qpos[-2:] = 0.04
path = franka.plan_path(
    qpos_goal     = qpos,
    num_waypoints = 200, # 2 秒持续时间
)
# 执行规划的路径
for waypoint in path:
    franka.control_dofs_position(waypoint)
    scene.step()

# 允许机器人到达最后一个路径点
for i in range(100):
    scene.step()


# 到达
qpos = franka.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.65, 0.0, 0.130]),
    quat = np.array([0, 1, 0, 0]),
)
franka.control_dofs_position(qpos[:-2], motors_dof)
for i in range(100):
    scene.step()

# 抓取
franka.control_dofs_position(qpos[:-2], motors_dof)
franka.control_dofs_force(np.array([-0.5, -0.5]), fingers_dof)

for i in range(100):
    scene.step()

# 抬起
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.28]),
    quat=np.array([0, 1, 0, 0]),
)
franka.control_dofs_position(qpos[:-2], motors_dof)
for i in range(200):
    scene.step()


