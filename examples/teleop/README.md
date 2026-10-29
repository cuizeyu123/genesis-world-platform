# 实时遥控工作台

这是一个运行在浏览器中的 Genesis 实时仿真遥控示例。FastAPI 提供状态、控制、操作和 MJPEG 相机接口；React + TypeScript + Vite 提供工作台界面。页面支持末端笛卡尔控制、七轴关节控制、夹爪、场景重置、工作空间限制提示，以及 MP4、MCAP 和 LeRobot v2.1 数据录制。

第一次运行可按[上手教程](TUTORIAL.md)逐步启动仿真、执行自然语言操作、配置 VLM 并录制演示数据。

工作视口右上角有一个安装在 Panda 末端的第一视角小窗，可切换 RGB、深度、世界坐标伪彩色点云和分割图。回零采用从当前关节位置出发的五次平滑插值，按仿真时间运行。后端每步检查机器人与环境的接触力；超过 18 N 时锁存碰撞状态、停止位置指令并进入零力模式，界面会显示接触位置和恢复入口。恢复前应先检查并移开碰撞源。

当前示例使用仓库内完整可加载的 Franka Panda 模型。截图中的 G1 双臂模型在仓库中没有完整 URDF/MJCF，因此这里保留截图的工作台布局，并用 Panda 仿真展示实际可执行的控制链路。

## 启动

首次使用时，在 PowerShell 中激活项目环境并安装工作台依赖：

```powershell
& "E:\A综合项目\具身\.venv\Scripts\Activate.ps1"
python -m pip install -r examples\teleop\requirements.txt
cd examples\teleop\web
npm install
npm run build
cd ../../..
```

启动遥控后端：

```powershell
& "E:\A综合项目\具身\.venv\Scripts\Activate.ps1"
python examples\manipulation\teleop_platform.py --port 8010
```

打开 <http://127.0.0.1:8010>。FastAPI 会先启动，Genesis 随后创建场景；首次运行若需编译内核，页面会显示等待首帧状态，编译结束后相机会自动出现。没有可用 GPU 时，可以加 `--cpu`；想同时显示 Genesis 原生窗口时，可以加 `--vis`。
请通过 Python 服务地址打开页面，不要直接双击 `web/index.html`；相机帧和状态都由仿真后端提供。

## 自然语言操作与 VLM 视觉任务

右侧操作面板可直接用自然语言调用本项目内置程序，不需要配置 VLM。输入“抓取蓝色方块并放到托盘”“拿起绿色方块”“把手里的物体放到托盘”“松开夹爪”“回到初始姿态”或“重置场景”，然后点击“执行指令”。抓取未指定颜色时会选择距离当前末端最近的蓝色或绿色方块。程序按当前仿真物体位置生成抓取、抬升和放置轨迹，前端显示步骤进度，也可以随时停止。

自然语言入口只匹配有限的内置程序名称和参数，不会把模型生成的代码直接运行。抓取与放置使用平滑分段轨迹，每个目标点都会检查 Panda 工作范围和逆运动学误差；已有的碰撞保护会在运行中持续生效。程序张合夹爪时，夹指与物体的预期接触不触发碰撞停机，机械臂其余连杆的碰撞保护保持开启。当前支持抓取方块、抓取后放入托盘、放下已持有物体、开合夹爪、平滑回零和重置场景。它操作的是 Genesis 仿真中的 Panda 与示例方块。

同一输入框还可以交给本机 Ollama 视觉模型分析当前 RGB 画面。单步模式会返回一条结构化建议，由操作者确认后执行；“自主闭环执行”会在每一步后重新读取相机和机器人状态，再规划下一步，最多执行 20 步或运行 3 分钟，可随时停止。自动模式在模型置信度低于 55% 或要求继续观察时暂停。末端移动每轴限制在 3 cm、合位移限制在 4 cm，并继续经过工作空间、逆运动学和碰撞保护检查；只有模型判定任务完成才会结束闭环。

需要先安装 Ollama 并下载支持图像输入的模型。轻量配置建议使用 `qwen3-vl:2b-instruct`（模型约 1.9 GB，需 Ollama 0.12.7 或更新版本）；如果观察和任务推理效果不足、机器资源允许，可换成 `qwen3-vl:4b-instruct`（约 3.3 GB）。模型文件大小不等于运行时所需内存，实际占用还会受上下文和 GPU/CPU 推理方式影响。建议使用 `instruct` 变体生成工作台所需的结构化动作 JSON；`qwen3-vl:2b` 当前对应 thinking 变体，可能将输出长度耗在思考内容上而不返回动作 JSON。下面以 `qwen3-vl:2b-instruct` 为例：

```powershell
ollama pull qwen3-vl:2b-instruct
$env:GENESIS_TELEOP_VLM_MODEL = "qwen3-vl:2b-instruct"
$env:GENESIS_TELEOP_VLM_URL = "http://127.0.0.1:11434/api/chat"
python examples\manipulation\teleop_platform.py --port 8010
```

Ollama 服务需保持运行。模型及地址也可用 `GENESIS_TELEOP_VLM_MODEL`、`GENESIS_TELEOP_VLM_URL` 配置；单次请求超时默认 90 秒，可通过 `GENESIS_TELEOP_VLM_TIMEOUT` 调整。未配置模型时，普通遥控仍可使用，VLM 面板会提示缺少配置。这个 VLM 执行器面向 Genesis 仿真示例，不连接真实机械臂。

## 日志与诊断

- 后端详细日志：项目根目录 `out\teleop\teleop.log`，按 5 MB 轮转并保留最近 5 个文件。
- 可通过环境变量 `GENESIS_TELEOP_LOG_PATH` 指定自定义日志文件，便于同时运行独立仿真实例或诊断副本。
- 控制台默认显示 `INFO` 及以上日志；使用 `--log-level DEBUG` 可记录每条视角输入，适合排查拖拽/滚轮没有效果的问题。
- 打开 <http://127.0.0.1:8010/api/diagnostics> 查看仿真循环、首帧、最近指令、相机位姿和待处理视角增量。拖拽时 `camera_command_age_s` 与 `camera_applied_age_s` 应持续更新，且 `camera.position` / `camera.target` 应变化。
- 诊断接口同时给出第一视角模式与帧龄，以及碰撞零力模式和接触力；碰撞触发、模式解除和相机渲染异常会写入同一日志。
- 启动错误、仿真循环异常、接口 4xx/5xx 和超过 1 秒的请求会写入日志；浏览器端视角请求失败会在页面提示，并写入浏览器开发者工具控制台。
- `--log-level WARNING` 可减少控制台输出；文件仍保留 DEBUG 明细。

开发前端时，可在一个终端运行仿真后端，在另一个终端进入 `examples\teleop\web` 并运行 `npm run dev`。Vite 开发服务器会把 `/api` 请求代理到 `http://127.0.0.1:8010`。

如需从同一局域网的另一台设备打开页面，可用 `--host 0.0.0.0` 启动，再通过运行主机的局域网地址访问。

## 操作

- `W/A/S/D` 或方向键：移动末端；`R/F`：升降；`Q/E`：绕 Z 轴旋转。
- 空格键：切换夹爪；也可用页面上的夹爪滑块和张开/闭合按钮。
- 选择“关节控制”后，拖动 J1–J7 滑块直接设定目标关节角。
- 在仿真视口内左键拖动可旋转视角；按住 Shift 或用鼠标右键拖动可平移；滚轮缩放，视口工具栏可复位视角。
- “停止控制”会清空输入并将机械臂保持在当前姿态；“恢复控制”重新接受操作。
- 点击“回到初始姿态”会连续平滑回零，不会瞬间写入初始关节角。
- 碰撞保护触发后，先检查场景并排除碰撞，再点击“检查后恢复位置控制”。保护状态锁存，普通遥控指令不会自动解除。
- 页面可保存相机当前帧；选择 MP4 时写入项目根目录 `out/teleop/*.mp4`。
- 在“数据录制”面板选择 MP4、MCAP 或 LeRobot v2.1 并填写任务描述。每次录制生成独立文件/数据集到项目根目录 `out/teleop/`；MCAP 包含状态、关节目标、主视角 RGB 和第一视角图像 topic，LeRobot 每次生成一个 episode，包含 30 FPS 主视角视频、关节状态、关节目标、末端位姿、任务和时间索引。
- LeRobot 输出遵循 v2.1 的 episode 文件布局（Parquet + MP4 + `meta/`），支持后续用 LeRobot 官方转换工具转换为 v3.0。录制依赖 `mcap` 和 `pyarrow` 已包含在 `examples/teleop/requirements.txt`；未安装时对应格式会在界面禁用，MP4 不受影响。
- LeRobot `observation.state` 和 `action` 使用 Panda 的 7 个关节弧度和 1 个夹爪开度；`observation.ee_pose` 为世界坐标 XYZ + XYZW 四元数。MCAP 使用 JSON schema channel，并将压缩 JPEG 图像编码为 `foxglove.CompressedImage` 消息。

碰撞阈值和零力动作目前仅依据 Genesis 仿真接触数据实现，适用于此仿真工作台；项目尚未接入真实机械臂驱动或硬件安全控制器。

Web 服务默认只监听本机回环地址；局域网模式面向可信网络，控制接口没有账号认证。
