"""Small, explicit natural-language dispatcher for built-in teleop programs."""

import re


PROGRAMS = {
    "pick": {
        "label": "抓取物体",
        "description": "抓取蓝色或绿色方块；未指定颜色时选择离机械臂最近的方块。",
        "examples": ["抓取蓝色方块", "拿起绿色方块", "把物体夹起来"],
    },
    "pick_place_tray": {
        "label": "抓取并放到托盘",
        "description": "抓取指定方块并放到灰色托盘中央。",
        "examples": ["把蓝色方块放到托盘", "抓起绿色方块并放进灰色托盘"],
    },
    "place_held_tray": {
        "label": "把手中物体放到托盘",
        "description": "将夹爪当前夹持的物体放到灰色托盘中央。",
        "examples": ["把手里的物体放到托盘", "放到灰色托盘上"],
    },
    "release": {
        "label": "松开夹爪",
        "description": "平滑打开夹爪。",
        "examples": ["松开夹爪", "释放物体"],
    },
    "close_gripper": {
        "label": "闭合夹爪",
        "description": "平滑闭合夹爪。",
        "examples": ["夹紧夹爪", "闭合夹爪"],
    },
    "home": {
        "label": "回到初始姿态",
        "description": "沿连续轨迹回到 Panda 初始姿态。",
        "examples": ["回到初始姿态", "机械臂回零"],
    },
    "reset_scene": {
        "label": "重置场景物体",
        "description": "将方块放回场景初始位置。",
        "examples": ["重置场景", "重置物体"],
    },
}


def interpret_natural_program(text):
    """Map a user request to one named program; never returns executable code."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("请输入要执行的自然语言指令")

    normalized = re.sub(r"[\s，。！？、,.!?；;：:]", "", text.lower())
    has_tray = any(term in normalized for term in ("托盘", "tray"))
    place_verbs = ("放", "放入", "放进", "移到", "搬到", "送到", "place", "put", "move")
    pick_verbs = ("抓", "夹", "拿起", "拾取", "抓取", "pick", "grasp", "grab")
    has_object_target = any(term in normalized for term in ("蓝", "绿", "blue", "green", "方块", "cube"))

    if any(term in normalized for term in ("重置场景", "重置物体", "resetscene", "resetobjects")):
        program = "reset_scene"
    elif any(term in normalized for term in ("初始姿态", "回零", "回到原位", "回到初始", "home")):
        program = "home"
    elif has_tray and any(term in normalized for term in place_verbs):
        program = "pick_place_tray" if any(term in normalized for term in pick_verbs) or has_object_target else "place_held_tray"
    elif any(term in normalized for term in ("松开", "打开夹爪", "张开夹爪", "夹爪张开", "夹爪打开", "释放", "放下", "opengripper", "release", "drop")):
        program = "release"
    elif any(term in normalized for term in ("闭合夹爪", "夹爪闭合", "夹紧", "合上夹爪", "closegripper", "closethegripper")):
        program = "close_gripper"
    elif any(term in normalized for term in pick_verbs):
        program = "pick"
    else:
        examples = "、".join(item for entry in PROGRAMS.values() for item in entry["examples"][:1])
        raise ValueError(f"暂不支持这条指令。可以试试：{examples}")

    color = None
    if any(term in normalized for term in ("蓝色", "蓝方块", "蓝色方块", "blue")):
        color = "blue"
    elif any(term in normalized for term in ("绿色", "绿方块", "绿色方块", "green")):
        color = "green"
    elif any(term in normalized for term in ("红色", "红方块", "red")):
        raise ValueError("当前仿真场景只有蓝色和绿色方块，请改用其中一种")

    if color is not None and program not in ("pick", "pick_place_tray"):
        raise ValueError("颜色目标只支持抓取方块或抓取后放入托盘")

    return {"program": program, "label": PROGRAMS[program]["label"], "object_color": color}
