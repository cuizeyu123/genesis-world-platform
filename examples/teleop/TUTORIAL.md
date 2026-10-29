# 实时遥控工作台上手教程

本教程从启动 Genesis 仿真开始，带你完成一次手动移动、查看第一视角、录制数据和提交 VLM 单步建议。命令以 Windows PowerShell 和仓库根目录为例。

## 1. 准备环境

激活项目环境并安装工作台依赖：

```powershell
& "E:\A综合项目\具身\.venv\Scripts\Activate.ps1"
python -m pip install -r examples\teleop\requirements.txt
```

首次运行前构建浏览器界面：

```powershell
cd examples\teleop\web
npm install
npm run build
cd ..\..\..
```

## 2. 启动并检查仿真

```powershell
python examples\manipulation\teleop_platform.py --port 8010
```

打开 <http://127.0.0.1:8010>。Genesis 初始化或首次编译内核时，页面会先显示等待状态；看到 Panda 和实时状态后即可操作。若没有可用 GPU，可停止后端并用 `--cpu` 重启。

也可以先检查后端状态：

```powershell
Invoke-RestMethod http://127.0.0.1:8010/api/health
```

当 `ready` 为 `true` 后，仿真和相机帧已就绪。访问 <http://127.0.0.1:8010/api/diagnostics> 可查看仿真循环、相机帧和控制延迟。

## 3. 做一次安全的手动遥控

1. 确认顶部显示控制已启用，碰撞保护未触发。
2. 用 `W/A/S/D` 或方向键移动末端，`R/F` 升降，`Q/E` 旋转；先短按观察方向和速度。
3. 用视口右上角切换第一视角的 RGB、深度、点云或分割图。主仿真视口支持拖动旋转、平移和滚轮缩放。
4. 点击“回到初始姿态”观察连续回零；按“停止控制”可立即停止接收操作输入。
5. 如果发生仿真碰撞，控制会进入锁存的零力保护状态。先检查并移开碰撞源，再点击恢复位置控制。

按关节模式可分别调整 J1–J7。笛卡尔和关节目标都会经过工作空间和控制限幅检查。

## 4. 用自然语言调用内置程序

在右侧“自然语言操作”输入一句命令，点击“执行指令”。例如：

- `抓取蓝色方块并放到托盘`
- `拿起绿色方块`
- `把手里的物体放到托盘`
- `松开夹爪`、`回到初始姿态`、`重置场景`

未指定蓝/绿色时，抓取程序会选择离当前末端最近的示例方块。程序使用仿真内的物体坐标生成平滑轨迹；页面显示当前步骤，运行中可以点击“停止程序”或按人工移动键接管。此入口只会调用项目里列出的固定程序，不会执行任意模型生成代码。碰撞保护、工作空间限制和逆运动学检查仍会生效。

如果描述不在支持范围内，页面会返回可识别的指令示例。自然语言程序使用已知仿真物体状态，不需要配置 Ollama；需要基于相机判断陌生物体或开放式任务时，再使用下方 VLM 视觉任务。

## 5. 录制一次操作

在“数据录制”区域填写任务描述并选格式：

- MP4 保存主视角视频。
- MCAP 保存状态、控制目标和相机图像 topic。
- LeRobot v2.1 保存 episode 表格、视频和数据集元信息。

点击开始录制，完成一段短操作后停止。文件写入仓库根目录 `out/teleop/`。MCAP 和 LeRobot 依赖在 `examples/teleop/requirements.txt` 中；安装依赖后需重启后端，界面才会启用相应格式。

## 6. 配置 VLM 视觉任务

VLM 是可选功能，不影响普通遥控。先安装并启动 Ollama，然后下载图像模型：

```powershell
ollama pull qwen3-vl:2b-instruct
```

在**启动后端的同一个 PowerShell 窗口**设置模型并重启服务：

```powershell
$env:GENESIS_TELEOP_VLM_MODEL = "qwen3-vl:2b-instruct"
$env:GENESIS_TELEOP_VLM_URL = "http://127.0.0.1:11434/api/chat"
python examples\manipulation\teleop_platform.py --port 8010
```

页面右侧应显示模型名称。输入一个简单任务，例如“把红色方块移到托盘上”，先点击分析并检查建议，再手动确认执行。VLM 当前输出的是经过限幅的单步 Panda 动作建议，不是直接控制硬件的 VLA 策略。自主闭环模式会重复观察和规划，运行期间仍可随时停止。

## 7. 排查常见问题

| 现象 | 检查方式 |
| --- | --- |
| 页面显示等待场景 | 查看 `/api/health` 的 `ready` 和 `/api/diagnostics` 中的仿真错误、首帧状态；首次内核编译可能需要等待。 |
| VLM 显示“未配置” | 在启动后端的终端设置 `GENESIS_TELEOP_VLM_MODEL`，然后重启后端。 |
| VLM 返回空正文或 JSON 截断 | 确认模型是图像输入的 `instruct` 版本；查看项目根目录 `out/teleop/teleop.log` 中的 `done_reason` 和 `eval_count`。 |
| 录制格式不可选 | 确认已安装 `examples/teleop/requirements.txt`，并重启后端。 |
| 视角拖动或缩放异常 | 查看 `/api/diagnostics` 中 `camera_command_age_s`、`camera_applied_age_s` 和相机位置；查看后端日志与浏览器控制台。 |

## 安全边界

本示例只驱动 Genesis 仿真中的 Franka Panda。VLM 动作会经过单步位移限幅、工作空间、逆运动学和碰撞检查；这些仿真保护不等于真实机械臂安全认证。默认服务只监听本机回环地址，局域网模式应只用于可信网络。
