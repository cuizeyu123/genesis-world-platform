<div align="center">

# Genesis 实时遥控工作台

**在浏览器中遥控仿真机械臂，用自然语言执行抓取任务，并录制可复用的机器人数据。**

[![GitHub stars](https://img.shields.io/github/stars/cuizeyu123/genesis-world-platform?style=social)](https://github.com/cuizeyu123/genesis-world-platform/stargazers)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)
[![Genesis World](https://img.shields.io/badge/Simulator-Genesis%20World-7357e8)](https://github.com/Genesis-Embodied-AI/genesis-world)

[快速开始](#快速开始) · [工作台功能](#工作台功能) · [自然语言操作](#自然语言操作) · [VLM 配置](#可选配置-vlm-视觉任务) · [问题排查](#问题排查)

</div>

<div align="center">
<video src="https://github.com/cuizeyu123/genesis-world-platform/raw/refs/heads/main/examples/teleop/assets/teleop-demo.mp4" controls playsinline preload="metadata" width="100%">
  <a href="examples/teleop/assets/teleop-demo.mp4">播放工作台演示视频</a>
</video>
</div>

这是一个基于 [Genesis World](https://github.com/Genesis-Embodied-AI/genesis-world) 的实时仿真遥控项目。它把 Genesis 中的 Franka Panda 仿真、相机和安全检查接入浏览器工作台，支持手动遥控、固定程序化抓取、可选 VLM 视觉任务，以及 MP4、MCAP 和 LeRobot 数据录制。

> 当前版本运行在 Genesis 仿真中，不连接真实机械臂。碰撞检测和零力模式是仿真功能，不构成硬件安全认证。

## 工作台功能

- **实时仿真遥控**：末端笛卡尔移动、J1–J7 关节控制、夹爪控制、平滑回零和场景重置。
- **仿真画面与传感器**：浏览器实时画面、可旋转缩放的视角，以及 RGB、深度、点云和分割图第一视角窗口。
- **自然语言操作**：输入“抓取蓝色方块并放到托盘”，工作台匹配项目内置操作程序并显示执行进度。
- **可选 VLM**：连接本机 Ollama 视觉模型，分析画面并给出单步建议或逐步重观察任务。
- **数据录制**：保存 MP4 视频、MCAP 状态与图像，或 LeRobot v2.1 episode 数据集。
- **日志与保护**：工作空间与逆运动学检查、碰撞停机、日志轮转和仿真诊断接口。

## 快速开始

### 环境要求

- Windows 10/11、Python 3.10–3.13 和 Node.js LTS。
- 先根据本机显卡或 CPU 环境安装 [PyTorch](https://pytorch.org/get-started/locally/)；Genesis 支持的后端和显卡要求请参阅 [Genesis World 文档](https://genesis-world.readthedocs.io/en/latest/)。
- 首次安装 Genesis 及其依赖需要下载较多组件。建议使用独立虚拟环境。

### 安装

在 PowerShell 中运行：

```powershell
git clone https://github.com/cuizeyu123/genesis-world-platform.git
cd genesis-world-platform

py -3.11 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# PyTorch 请按本机硬件在 pytorch.org 选择对应安装命令
python -m pip install -e .
python -m pip install -r examples\teleop\requirements.txt

cd examples\teleop\web
npm ci
npm run build
cd ..\..\..
```

如果 PowerShell 尚未允许执行本地虚拟环境脚本，可在当前终端运行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& .\.venv\Scripts\Activate.ps1
```

### 启动

在仓库根目录启动仿真后端：

```powershell
& .\.venv\Scripts\Activate.ps1
python examples\manipulation\teleop_platform.py --port 8010
```

浏览器打开 <http://127.0.0.1:8010>。没有可用 GPU 时，使用 `--cpu` 启动 CPU 后端：

```powershell
python examples\manipulation\teleop_platform.py --port 8010 --cpu
```

首次运行可能需要等待 Genesis 初始化或编译内核。仿真状态可通过 <http://127.0.0.1:8010/api/health> 查看；诊断信息位于 <http://127.0.0.1:8010/api/diagnostics>。

## 使用工作台

### 手动遥控

| 输入 | 操作 |
| --- | --- |
| `W/A/S/D` 或方向键 | 移动末端 |
| `R/F` | 上升 / 下降 |
| `Q/E` | 绕 Z 轴旋转 |
| 空格键 | 张开 / 闭合夹爪 |
| 鼠标拖动 / 滚轮 | 旋转、平移 / 缩放仿真相机 |

也可切换到关节模式，拖动 J1–J7 控件。执行回零前，工作台会从当前关节位置连续移动到初始姿态。

### 自然语言操作

在“自然语言操作”输入指令，然后点击“执行指令”。支持的例子：

```text
抓取蓝色方块
拿起绿色方块
把蓝色方块放到托盘
把手里的物体放到托盘
松开夹爪
回到初始姿态
重置场景
```

未指定颜色时，抓取程序会选择离当前末端最近的蓝色或绿色方块。运行中可以停止程序，或用人工移动输入接管。自然语言入口只调用项目内的固定程序，不会执行模型生成的任意代码。目标位姿会经过工作空间和逆运动学检查；碰撞保护持续监测机械臂与场景的接触。

### 数据录制

工作台右侧可选择 MP4、MCAP 或 LeRobot v2.1 格式并填写任务说明。数据默认保存在仓库根目录 `out/teleop/`。MP4 可直接使用；MCAP 和 LeRobot 所需依赖随工作台依赖一同安装，重启后端后即可选择。

## 可选配置：VLM 视觉任务

自然语言内置程序不依赖 VLM。若要让视觉模型读取仿真画面并建议动作，可安装并启动 [Ollama](https://ollama.com/)，下载视觉 Instruct 模型：

```powershell
ollama pull qwen3-vl:2b-instruct
$env:GENESIS_TELEOP_VLM_MODEL = "qwen3-vl:2b-instruct"
$env:GENESIS_TELEOP_VLM_URL = "http://127.0.0.1:11434/api/chat"
python examples\manipulation\teleop_platform.py --port 8010
```

环境变量需要在启动后端的同一终端中设置。VLM 面板支持单步分析与自主闭环执行；每步动作会重新观察状态，置信度不足时暂停。未配置 Ollama 时，手动遥控和内置自然语言程序仍可使用。

## 诊断与问题排查

- **页面没有画面或显示等待场景**：等待仿真初始化，再检查 `/api/health` 的 `ready` 字段和 `/api/diagnostics`。
- **界面没有最新功能**：停止旧后端，在仓库根目录重新启动 `teleop_platform.py`，然后刷新浏览器。
- **VLM 显示未配置**：设置 `GENESIS_TELEOP_VLM_MODEL` 后重新启动后端；VLM 是可选项。
- **自然语言指令不支持**：使用上面的示例，当前程序支持场景中的蓝色、绿色方块和灰色托盘。
- **查找后端日志**：默认路径为 `out/teleop/teleop.log`，日志文件会轮转保留。
- **检查摄像头、控制延迟和碰撞状态**：访问 `http://127.0.0.1:8010/api/diagnostics`。

## 项目结构

```text
examples/
├── manipulation/teleop_platform.py  # FastAPI、Genesis 仿真与控制程序
├── manipulation/teleop_skills.py    # 自然语言到内置操作的匹配
└── teleop/
    ├── web/                          # React + TypeScript 工作台
    ├── README.md                     # 工作台详细说明
    └── TUTORIAL.md                   # 分步上手教程
```

## 相关文档与致谢

- [工作台上手教程](examples/teleop/TUTORIAL.md)
- [遥控工作台详细说明](examples/teleop/README.md)
- [Genesis World 文档](https://genesis-world.readthedocs.io/en/latest/)
- [上游 Genesis World 项目](https://github.com/Genesis-Embodied-AI/genesis-world)

本仓库基于 Genesis World，保留其仿真引擎与示例，并在此基础上维护实时遥控工作台。上游项目及本仓库代码采用 [Apache-2.0 许可证](LICENSE)。
