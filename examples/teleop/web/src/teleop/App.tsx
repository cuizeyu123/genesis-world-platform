import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type Mode = "cartesian" | "joint";
type SensorMode = "rgb" | "depth" | "pointcloud" | "segmentation";
type RecordingFormat = "mp4" | "mcap" | "lerobot";
type ActionName = "home" | "reset_scene";
type AgentAction = "move" | "open_gripper" | "close_gripper" | "observe" | "complete";
type AgentPlan = {
  suggestion_id: string;
  expires_in_s: number;
  summary: string;
  target_object: string;
  action: AgentAction;
  delta_m: number[];
  reason: string;
  confidence: number;
};
type AgentStatus = { configured: boolean; model: string; max_move_m: number };
type ProgramStatus = { program: string; label: string; state: "idle" | "queued" | "running" | "completed" | "failed" | "stopped"; step: number; total: number; message: string };
type TeleopState = {
  ee: number[];
  quat: number[];
  joints: number[];
  joint_limits: number[][];
  gripper: number;
  fps: number;
  sim_time: number;
  clamped: boolean;
  recording: boolean;
  recording_status?: string;
  recording_format?: RecordingFormat;
  recording_task?: string;
  recording_path?: string;
  recording_frames?: number;
  recording_error?: string;
  enabled: boolean;
  homing: boolean;
  collision_mode: boolean;
  collision_force_n: number;
  collision_threshold_n: number;
  collision_position: number[] | null;
  collision_link: string;
  sensor_mode: SensorMode;
  objects?: { name: string; color: string; position: number[] }[];
  agent_busy?: boolean;
  program_status?: ProgramStatus;
  error?: string;
};
type Command = {
  keys: string[];
  gripper: number;
  speed: number;
  yaw_speed: number;
  mode: Mode;
  joints: number[];
  enabled: boolean;
};
type CameraControl = {
  orbit_x?: number;
  orbit_y?: number;
  pan_x?: number;
  pan_y?: number;
  zoom?: number;
  reset?: boolean;
};

const JOINT_HOME = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853];
const KEY_MAP: Record<string, string> = {
  ArrowUp: "KeyW",
  ArrowDown: "KeyS",
  ArrowLeft: "KeyA",
  ArrowRight: "KeyD",
};
const MOTION_KEYS = new Set([
  "KeyW", "KeyA", "KeyS", "KeyD", "KeyR", "KeyF", "KeyQ", "KeyE",
]);

function round(value: number | undefined, digits = 2) {
  return value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}

function App() {
  const [keys, setKeys] = useState<string[]>([]);
  const [enabled, setEnabled] = useState(true);
  const [gripper, setGripper] = useState(0.04);
  const [speed, setSpeed] = useState(0.15);
  const [mode, setMode] = useState<Mode>("cartesian");
  const [joints, setJoints] = useState(JOINT_HOME);
  const [touchedJoints, setTouchedJoints] = useState<boolean[]>(Array(7).fill(false));
  const [teleopState, setTeleopState] = useState<TeleopState | null>(null);
  const [online, setOnline] = useState(false);
  const [backendCompatibilityError, setBackendCompatibilityError] = useState("");
  const [frameLoaded, setFrameLoaded] = useState(false);
  const [firstPersonStreamError, setFirstPersonStreamError] = useState("");
  const [streamRetry, setStreamRetry] = useState(0);
  const [sensorMode, setSensorMode] = useState<SensorMode>("rgb");
  const [sensorRetry, setSensorRetry] = useState(0);
  const [recordingFormat, setRecordingFormat] = useState<RecordingFormat>("mp4");
  const [recordingTask, setRecordingTask] = useState("pick and place objects");
  const [availableRecordingFormats, setAvailableRecordingFormats] = useState<RecordingFormat[]>(["mp4"]);
  const [agentTask, setAgentTask] = useState("把蓝色方块移动到灰色托盘上");
  const [agentStatus, setAgentStatus] = useState<AgentStatus | null>(null);
  const [agentPlan, setAgentPlan] = useState<AgentPlan | null>(null);
  const [agentLoading, setAgentLoading] = useState(false);
  const [agentAutoRunning, setAgentAutoRunning] = useState(false);
  const [agentError, setAgentError] = useState("");
  const [programSubmitting, setProgramSubmitting] = useState(false);
  const [programError, setProgramError] = useState("");
  const [toast, setToast] = useState("");
  const [clock, setClock] = useState(() => new Date().toLocaleTimeString("zh-CN", { hour12: false }));
  const [rtt, setRtt] = useState<number | null>(null);
  const commandRef = useRef<Command>({ keys: [], gripper: 0.04, speed: 0.15, yaw_speed: 0.7, mode: "cartesian", joints: JOINT_HOME, enabled: true });
  const toastTimer = useRef<number | undefined>(undefined);
  const cameraErrorLoggedAt = useRef(0);
  const latestJoints = useRef<number[] | null>(null);
  const agentStopRequested = useRef(false);
  const agentAnalyzeController = useRef<AbortController | null>(null);
  const lastProgramStatus = useRef<ProgramStatus | null>(null);
  const viewDrag = useRef<{ pointerId: number; x: number; y: number; height: number; mode: "orbit" | "pan" } | null>(null);

  useEffect(() => {
    commandRef.current = { keys, gripper, speed, yaw_speed: 0.7, mode, joints, enabled };
  }, [keys, gripper, speed, mode, joints, enabled]);

  const notify = useCallback((message: string) => {
    setToast(message);
    window.clearTimeout(toastTimer.current);
    toastTimer.current = window.setTimeout(() => setToast(""), 2200);
  }, []);

  useEffect(() => {
    let active = true;
    async function refreshAgentStatus() {
      try {
        const response = await fetch("/api/agent/status", { cache: "no-store" });
        if (!response.ok) return;
        const status = await response.json() as AgentStatus;
        if (active) setAgentStatus(status);
      } catch {
        if (active) setAgentStatus(null);
      }
    }
    void refreshAgentStatus();
    const timer = window.setInterval(refreshAgentStatus, 10000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);

  const updateKeys = useCallback((key: string, pressed: boolean) => {
    setKeys((current) => {
      const next = new Set(current);
      if (pressed) next.add(key);
      else next.delete(key);
      return [...next];
    });
  }, []);

  useEffect(() => {
    let active = true;
    async function checkBackendFeatures() {
      try {
        const response = await fetch("/api/health", { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const health = await response.json() as {
          features?: { smooth_home?: boolean; first_person_modes?: string[]; collision_zero_force?: boolean; recording_formats?: RecordingFormat[]; natural_programs?: boolean };
        };
        const features = health.features;
        setAvailableRecordingFormats(features?.recording_formats ?? ["mp4"]);
        const current = !!features?.smooth_home
          && !!features?.collision_zero_force
          && !!features?.natural_programs
          && ["rgb", "depth", "pointcloud", "segmentation"].every((mode) => features.first_person_modes?.includes(mode));
        if (active) {
          setBackendCompatibilityError(current ? "" : "当前遥控后端版本较旧。请关闭正在运行的遥控服务，再从当前项目重新启动 teleop_platform.py，以启用当前工作台功能。");
        }
      } catch {
        if (active) setBackendCompatibilityError("");
      }
    }
    void checkBackendFeatures();
    const timer = window.setInterval(checkBackendFeatures, 3000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);

  useEffect(() => {
    let active = true;
    const delay = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));
    async function sendLoop() {
      while (active) {
        const started = performance.now();
        try {
          const response = await fetch("/api/command", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(commandRef.current),
          });
          if (!response.ok) throw new Error("控制指令未被接受");
          if (active) setRtt(performance.now() - started);
        } catch {
          if (active) setRtt(null);
        }
        await delay(33);
      }
    }
    void sendLoop();
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    async function poll() {
      try {
        const response = await fetch("/api/state", { cache: "no-store" });
        if (!response.ok) throw new Error("仿真状态不可用");
        const next = await response.json() as TeleopState;
        if (!active) return;
        setOnline(true);
        setTeleopState(next);
        const previousProgram = lastProgramStatus.current;
        const currentProgram = next.program_status;
        if (previousProgram && ["queued", "running"].includes(previousProgram.state) && currentProgram && ["completed", "failed", "stopped"].includes(currentProgram.state)) {
          setGripper(next.gripper);
          commandRef.current = { ...commandRef.current, gripper: next.gripper };
        }
        lastProgramStatus.current = currentProgram ?? null;
        if (next.sensor_mode) setSensorMode(next.sensor_mode);
        if (next.joints?.length === 7) {
          latestJoints.current = next.joints;
          setJoints((current) => current.map((value, i) => mode === "joint" && touchedJoints[i] ? value : next.joints[i]));
        }
        if (next.enabled !== undefined) setEnabled(next.enabled);
      } catch {
        if (!active) return;
        setOnline(false);
        setTeleopState(null);
        setFrameLoaded(false);
      }
    }
    void poll();
    const timer = window.setInterval(poll, 120);
    return () => { active = false; window.clearInterval(timer); };
  }, [mode, touchedJoints]);

  useEffect(() => {
    const timer = window.setInterval(() => setClock(new Date().toLocaleTimeString("zh-CN", { hour12: false })), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    const keyDown = (event: KeyboardEvent) => {
      if (["INPUT", "TEXTAREA", "SELECT"].includes((event.target as HTMLElement)?.tagName ?? "")) return;
      const translated = KEY_MAP[event.code] ?? event.code;
      if (MOTION_KEYS.has(translated)) {
        event.preventDefault();
        if (enabled) updateKeys(translated, true);
      } else if (event.code === "Space" && !event.repeat) {
        event.preventDefault();
        setGripper((value) => value > 0.02 ? 0 : 0.04);
      }
    };
    const keyUp = (event: KeyboardEvent) => updateKeys(KEY_MAP[event.code] ?? event.code, false);
    const blur = () => setKeys([]);
    window.addEventListener("keydown", keyDown);
    window.addEventListener("keyup", keyUp);
    window.addEventListener("blur", blur);
    return () => {
      window.removeEventListener("keydown", keyDown);
      window.removeEventListener("keyup", keyUp);
      window.removeEventListener("blur", blur);
    };
  }, [enabled, updateKeys]);

  useEffect(() => {
    if (online && !teleopState?.error) return;
    const timer = window.setTimeout(() => {
      setFrameLoaded(false);
      setStreamRetry((attempt) => attempt + 1);
    }, 2500);
    return () => window.clearTimeout(timer);
  }, [online, teleopState?.error, streamRetry]);

  const jointLimits = teleopState?.joint_limits;
  const activeKeys = useMemo(() => new Set(keys), [keys]);
  const programBusy = teleopState?.program_status?.state === "queued" || teleopState?.program_status?.state === "running";

  const toggleControl = () => {
    const next = !enabled;
    setEnabled(next);
    if (!next) setKeys([]);
  };

  const action = async (name: ActionName) => {
    if (name === "home" && teleopState?.homing) {
      notify("机械臂正在平滑回零");
      return;
    }
    if (name === "home" && !enabled) {
      notify("请先恢复控制，再回到初始姿态");
      return;
    }
    try {
      const response = await fetch("/api/action", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: name }),
      });
      if (!response.ok) throw new Error("操作失败");
      if (name === "home") {
        setTouchedJoints(Array(7).fill(false));
        setGripper(0.04);
        notify("正在平滑回到初始姿态");
      } else notify("场景物体已重置");
    } catch {
      notify("仿真服务未连接，操作没有发送");
    }
  };

  const toggleRecording = async () => {
    const isRecording = !!teleopState?.recording;
    try {
      const response = await fetch(isRecording ? "/api/recording/stop" : "/api/recording/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        ...(isRecording ? {} : { body: JSON.stringify({ format: recordingFormat, task: recordingTask.trim() || "Teleoperation session" }) }),
      });
      if (!response.ok) {
        const result = await response.json().catch(() => ({})) as { detail?: string };
        throw new Error(result.detail ?? `HTTP ${response.status}`);
      }
      notify(isRecording ? "正在结束并保存录制" : `开始记录 ${recordingFormat.toUpperCase()} 数据`);
    } catch (error) {
      console.error("[teleop] recording request failed", error);
      notify(error instanceof Error ? error.message : "录制操作失败，请检查仿真后端");
    }
  };

  const requestAgentPlan = async (task: string, signal?: AbortSignal) => {
    const response = await fetch("/api/agent/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task }),
      signal,
    });
    const result = await response.json().catch(() => ({})) as AgentPlan & { detail?: string };
    if (!response.ok) throw new Error(result.detail ?? `HTTP ${response.status}`);
    return result;
  };

  const submitAgentPlan = async (plan: AgentPlan) => {
    const response = await fetch("/api/agent/execute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ suggestion_id: plan.suggestion_id }),
    });
    const result = await response.json().catch(() => ({})) as { detail?: string };
    if (!response.ok) throw new Error(result.detail ?? `HTTP ${response.status}`);
    if (plan.action === "open_gripper" || plan.action === "close_gripper") {
      const target = plan.action === "open_gripper" ? 0.04 : 0;
      commandRef.current = { ...commandRef.current, gripper: target };
      setGripper(target);
    }
  };

  const analyzeAgentTask = async () => {
    const task = agentTask.trim();
    if (!task || agentLoading || agentAutoRunning || teleopState?.agent_busy) return;
    setAgentLoading(true);
    setAgentError("");
    setAgentPlan(null);
    try {
      const plan = await requestAgentPlan(task);
      setAgentPlan(plan);
      notify("视觉分析完成，请检查并确认下一步动作");
    } catch (error) {
      const message = error instanceof Error ? error.message : "VLM 分析失败";
      setAgentError(message);
      console.error("[teleop] VLM analysis failed", error);
    } finally {
      setAgentLoading(false);
    }
  };

  const executeAgentPlan = async () => {
    if (!agentPlan || ["observe", "complete"].includes(agentPlan.action) || !enabled || collisionMode) return;
    try {
      await submitAgentPlan(agentPlan);
      setAgentPlan(null);
      notify("已确认，仿真正在执行单步动作");
    } catch (error) {
      const message = error instanceof Error ? error.message : "VLM 动作执行失败";
      setAgentError(message);
      console.error("[teleop] VLM action failed", error);
    }
  };

  const stopAgentTask = async () => {
    agentStopRequested.current = true;
    agentAnalyzeController.current?.abort();
    try {
      await fetch("/api/agent/stop", { method: "POST" });
    } catch (error) {
      console.error("[teleop] VLM stop request failed", error);
    }
    notify("已请求停止当前自然语言任务");
  };

  const runNaturalProgram = async () => {
    const task = agentTask.trim();
    if (!task || programSubmitting || !enabled || collisionMode || teleopState?.agent_busy || keys.length) return;
    setProgramSubmitting(true);
    setProgramError("");
    setAgentPlan(null);
    try {
      const programCommand = { ...commandRef.current, mode: "cartesian" as const, keys: [] };
      commandRef.current = programCommand;
      setMode("cartesian");
      const commandResponse = await fetch("/api/command", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(programCommand),
      });
      if (!commandResponse.ok) throw new Error("工作台控制状态更新失败，请重试");
      const response = await fetch("/api/agent/program", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task }),
      });
      const result = await response.json().catch(() => ({})) as { detail?: string; label?: string; queued?: boolean };
      if (!response.ok) throw new Error(result.detail ?? `HTTP ${response.status}`);
      notify(`已调用程序：${result.label ?? "操作"}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "自然语言程序调用失败";
      setProgramError(message);
      console.error("[teleop] natural-language program request failed", error);
    } finally {
      setProgramSubmitting(false);
    }
  };

  const runAgentTask = async () => {
    const task = agentTask.trim();
    if (!task || agentAutoRunning || agentLoading || !enabled || collisionMode || keys.length || teleopState?.agent_busy) return;
    agentStopRequested.current = false;
    setAgentAutoRunning(true);
    setAgentError("");
    setAgentPlan(null);
    const startedAt = Date.now();
    const delay = (ms: number) => new Promise<void>((resolve) => window.setTimeout(resolve, ms));
    try {
      for (let step = 0; step < 20 && Date.now() - startedAt < 180000; step += 1) {
        if (agentStopRequested.current) break;
        if (!commandRef.current.enabled || commandRef.current.keys.length) throw new Error("控制已停止或收到人工移动输入，VLM 任务已中止");
        const controller = new AbortController();
        agentAnalyzeController.current = controller;
        const plan = await requestAgentPlan(task, controller.signal);
        agentAnalyzeController.current = null;
        if (agentStopRequested.current) break;
        setAgentPlan(plan);
        if (plan.action === "complete") {
          notify("VLM 判断任务已完成");
          return;
        }
        if (plan.action === "observe") {
          notify("VLM 需要更多观察，自动任务已暂停");
          return;
        }
        if (plan.confidence < 0.55) {
          setAgentError("VLM 对当前场景的置信度不足 55%，自动任务已暂停，请使用单步建议复核。");
          return;
        }
        await submitAgentPlan(plan);
        const queuedAt = Date.now();
        let finished = false;
        while (Date.now() - queuedAt < 10000) {
          if (agentStopRequested.current) break;
          if (!commandRef.current.enabled || commandRef.current.keys.length) {
            await fetch("/api/agent/stop", { method: "POST" }).catch(() => undefined);
            throw new Error("控制已停止或收到人工移动输入，VLM 任务已中止");
          }
          const stateResponse = await fetch("/api/state", { cache: "no-store" });
          if (!stateResponse.ok) throw new Error("仿真状态不可用，VLM 任务已中止");
          const state = await stateResponse.json() as TeleopState;
          if (state.collision_mode) throw new Error("碰撞保护已触发，VLM 任务已中止");
          if (!state.enabled) throw new Error("机械臂控制已停止，VLM 任务已中止");
          if (Date.now() - queuedAt > 300 && !state.agent_busy) {
            finished = true;
            break;
          }
          await delay(120);
        }
        if (agentStopRequested.current) break;
        if (!finished) throw new Error("单步动作等待超时，VLM 任务已中止");
      }
      if (!agentStopRequested.current && Date.now() - startedAt >= 180000) {
        setAgentError("闭环任务达到 3 分钟时限，已停止。可重新分析当前场景后继续。");
      } else if (!agentStopRequested.current) {
        setAgentError("闭环任务已执行 20 步，模型尚未确认完成。请检查现场后重新启动。");
      }
    } catch (error) {
      if (!agentStopRequested.current) {
        const message = error instanceof Error ? error.message : "VLM 闭环任务失败";
        setAgentError(message);
        console.error("[teleop] VLM task failed", error);
      }
    } finally {
      agentAnalyzeController.current = null;
      setAgentAutoRunning(false);
      if (agentStopRequested.current) setAgentPlan(null);
    }
  };

  const selectMode = (next: Mode) => {
    if (next === "joint" && latestJoints.current) {
      setJoints([...latestJoints.current]);
      setTouchedJoints(Array(7).fill(false));
    }
    setMode(next);
  };

  const setJoint = (index: number, value: number) => {
    setJoints((current) => current.map((joint, i) => i === index ? value : joint));
    setTouchedJoints((current) => current.map((touched, i) => i === index ? true : touched));
  };

  const changeSensorMode = async (next: SensorMode) => {
    try {
      const response = await fetch("/api/sensor/mode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: next }),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      setSensorMode(next);
      setFirstPersonStreamError("");
      setSensorRetry((value) => value + 1);
    } catch (error) {
      console.error("[teleop] first-person sensor mode update failed", error);
      setFirstPersonStreamError("后端没有响应第一视角模式接口。请重启当前项目的遥控后端。");
      notify("第一视角模式切换失败，请检查仿真后端");
    }
  };

  const resetCollision = async () => {
    try {
      const response = await fetch("/api/collision/reset", { method: "POST" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const result = await response.json() as { ok: boolean };
      if (!result.ok) throw new Error("当前没有待解除的碰撞保护");
      setEnabled(true);
      notify("正在退出零力模式，位置控制将从当前姿态恢复");
    } catch (error) {
      console.error("[teleop] collision recovery request failed", error);
      notify("碰撞模式恢复失败，请查看后端日志");
    }
  };

  const captureSnapshot = () => {
    const link = document.createElement("a");
    link.href = "/api/snapshot.jpg";
    link.download = `genesis-frame-${new Date().toISOString().replaceAll(":", "-")}.jpg`;
    link.click();
  };

  const sendCameraControl = useCallback((control: CameraControl) => {
    void fetch("/api/camera", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(control),
    }).then(async (response) => {
      if (response.ok) return;
      const detail = await response.text();
      throw new Error(`HTTP ${response.status}: ${detail}`);
    }).catch((error: unknown) => {
      const now = Date.now();
      if (now - cameraErrorLoggedAt.current >= 2000) {
        console.error("[teleop] camera control request failed", error);
        notify("视角指令发送失败；请查看后端日志或诊断信息");
        cameraErrorLoggedAt.current = now;
      }
    });
  }, [notify]);

  const resetCamera = () => sendCameraControl({ reset: true });

  const streamUrl = `/api/stream.mjpg?attempt=${streamRetry}`;
  const firstPersonStreamUrl = `/api/first-person.mjpg?attempt=${sensorRetry}`;
  const collisionMode = !!teleopState?.collision_mode;
  const firstPersonError = firstPersonStreamError || backendCompatibilityError;
  const simulationError = teleopState?.error;
  const simulationReady = !!teleopState?.ee && !simulationError;
  const frameMessage = simulationError
    ? "仿真循环发生错误"
    : !online
      ? "未连接到仿真服务"
      : teleopState?.ee
        ? "仿真画面正在恢复"
        : "正在等待 Genesis 仿真首帧";
  const frameDetail = simulationError
    || (!online
      ? "请先启动 FastAPI 遥控后端，再刷新或稍候重连。"
      : "后端已响应，Genesis 正在初始化场景并生成相机画面。画面就绪后会自动显示。");
  const connectionLabel = simulationError ? "仿真异常" : !online ? "连接中断" : simulationReady ? "仿真运行中" : "后端已连接";
  const motionButton = (label: string, key: string, className = "") => (
    <button
      key={key}
      className={`motion-button ${className} ${activeKeys.has(key) ? "held" : ""}`}
      type="button"
      aria-label={label}
      onPointerDown={(event) => {
        event.preventDefault();
        if (!enabled) return;
        updateKeys(key, true);
        try { event.currentTarget.setPointerCapture(event.pointerId); } catch { /* no-op */ }
      }}
      onPointerUp={() => updateKeys(key, false)}
      onPointerCancel={() => updateKeys(key, false)}
      onLostPointerCapture={() => updateKeys(key, false)}
      onPointerLeave={() => updateKeys(key, false)}
      disabled={!enabled || collisionMode}
    >{label}</button>
  );

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark">G</span><span>Genesis 遥控工作台</span></div>
        <nav className="nav" aria-label="主导航">
          <button className="active" type="button">实时控制</button>
          <button type="button">任务记录</button>
          <button type="button">设备管理</button>
        </nav>
        <div className="top-status">
          <span className="robot-tag">Franka Panda · 7 DoF</span>
          <span className={`connection ${online && !simulationError ? "" : "offline"}`}><i className="live-dot" />{connectionLabel}</span>
          <time>{clock}</time>
        </div>
      </header>

      <div className="toolbar">
        <div className="tool-group">
          <button className="tool-icon" type="button" title="回到初始姿态" onClick={() => action("home")}>⌂</button>
          <button className="tool-icon" type="button" title="重置场景物体" onClick={() => action("reset_scene")}>⟲</button>
          <span className="toolbar-note">快捷键：W A S D · R/F · Q/E</span>
        </div>
        <button className={`stop-button ${enabled ? "" : "stopped"}`} type="button" onClick={toggleControl} disabled={collisionMode}>{collisionMode ? "零力拖动中" : enabled ? "停止控制" : "恢复控制"}</button>
        <div className="tool-group right">
          <button className="tool-icon" type="button" title="网格" onClick={() => document.querySelector(".stage")?.classList.toggle("without-grid")}>▦</button>
          <button className="tool-icon" type="button" title="缩小" onClick={() => sendCameraControl({ zoom: 0.55 })}>−</button>
          <button className="tool-icon" type="button" title="放大" onClick={() => sendCameraControl({ zoom: -0.55 })}>＋</button>
          <button className="tool-icon" type="button" title="诊断信息" onClick={() => window.open("/api/diagnostics", "_blank", "noopener,noreferrer")}>ⓘ</button>
          <button className="tool-icon" type="button" title="截取当前画面" onClick={captureSnapshot}>⇩</button>
        </div>
      </div>

      <main className="workspace">
        <aside className="side left" aria-label="机械臂控制">
          <div className="side-head"><h2>关节控制</h2><span className="pill">deg</span></div>
          <div className="mode-switch" role="group" aria-label="控制模式">
            <button className={mode === "cartesian" ? "active" : ""} type="button" onClick={() => selectMode("cartesian")} disabled={programBusy}>末端控制</button>
            <button className={mode === "joint" ? "active" : ""} type="button" onClick={() => selectMode("joint")} disabled={programBusy}>关节控制</button>
          </div>
          <div className="slider-row"><label htmlFor="speed">速度</label><input id="speed" type="range" min="0.02" max="0.35" step="0.01" value={speed} onChange={(event) => setSpeed(Number(event.target.value))} disabled={!enabled || collisionMode || programBusy} /><output>{Math.round(speed / 0.35 * 100)}%</output></div>
          <div className="section-rule" />
          <div className={`joint-list ${mode === "joint" ? "joint-active" : ""}`} aria-label="七个关节目标角">
            {joints.map((value, index) => (
              <div className="joint-row" key={index}>
                <span className="joint-name">J{index + 1}</span>
                <input
                  type="range"
                  min={jointLimits?.[index]?.[0] ?? -2.9}
                  max={jointLimits?.[index]?.[1] ?? 2.9}
                  step="0.01"
                  value={value}
                  aria-label={`关节 J${index + 1} 目标位置`}
                  onChange={(event) => setJoint(index, Number(event.target.value))}
                  disabled={!enabled || collisionMode || !!teleopState?.homing || programBusy}
                />
                <span className="joint-value">{(value * 180 / Math.PI).toFixed(1)}°</span>
              </div>
            ))}
          </div>
          <div className="section-rule" />
          <div className="slider-row"><label htmlFor="gripper">夹爪</label><input id="gripper" type="range" min="0" max="0.04" step="0.001" value={gripper} onChange={(event) => setGripper(Number(event.target.value))} disabled={!enabled || collisionMode || programBusy} /><output>{Math.round(gripper * 1000)} mm</output></div>
          <div className="button-row"><button className="action-button" type="button" onClick={() => setGripper(0.04)} disabled={!enabled || collisionMode || programBusy}>张开</button><button className="action-button" type="button" onClick={() => setGripper(0)} disabled={!enabled || collisionMode || programBusy}>闭合</button></div>
        </aside>

        <section className={`stage ${frameLoaded ? "has-frame" : ""}`} aria-label="Genesis 仿真实时画面">
          <div className="stage-heading"><div><strong>Franka Panda 仿真实时相机画面</strong><span className="stage-breadcrumb"><i className="live-dot" />仿真画面&nbsp; / &nbsp;Genesis Camera</span></div><div className="stage-actions"><button type="button" title="复位视角" onClick={resetCamera}>⌖</button><button type="button" title="重载画面" onClick={() => { setFrameLoaded(false); setStreamRetry((value) => value + 1); }}>↻</button><button type="button" title="保存截图" onClick={captureSnapshot}>⇩</button><button type="button" title="全屏" onClick={() => document.querySelector(".stage")?.requestFullscreen?.()}>⛶</button></div></div>
          <div
            className={`camera-viewport ${viewDrag.current ? "dragging" : ""}`}
            onPointerDown={(event) => {
              if (![0, 1, 2].includes(event.button)) return;
              event.preventDefault();
              viewDrag.current = {
                pointerId: event.pointerId,
                x: event.clientX,
                y: event.clientY,
                height: Math.max(event.currentTarget.clientHeight, 1),
                mode: event.button === 1 || event.button === 2 || event.shiftKey ? "pan" : "orbit",
              };
              event.currentTarget.classList.add("dragging");
              try { event.currentTarget.setPointerCapture(event.pointerId); } catch { /* no-op */ }
            }}
            onPointerMove={(event) => {
              const drag = viewDrag.current;
              if (!drag || drag.pointerId !== event.pointerId) return;
              const dx = event.clientX - drag.x;
              const dy = event.clientY - drag.y;
              drag.x = event.clientX;
              drag.y = event.clientY;
              if (drag.mode === "orbit") {
                sendCameraControl({ orbit_x: dx / drag.height, orbit_y: dy / drag.height });
              } else {
                sendCameraControl({ pan_x: dx / drag.height, pan_y: dy / drag.height });
              }
            }}
            onPointerUp={(event) => {
              if (viewDrag.current?.pointerId === event.pointerId) viewDrag.current = null;
              event.currentTarget.classList.remove("dragging");
            }}
            onPointerCancel={(event) => {
              if (viewDrag.current?.pointerId === event.pointerId) viewDrag.current = null;
              event.currentTarget.classList.remove("dragging");
            }}
            onLostPointerCapture={(event) => {
              if (viewDrag.current?.pointerId === event.pointerId) viewDrag.current = null;
              event.currentTarget.classList.remove("dragging");
            }}
            onContextMenu={(event) => event.preventDefault()}
            onWheel={(event) => {
              event.preventDefault();
              sendCameraControl({ zoom: Math.max(-2, Math.min(2, event.deltaY / 300)) });
            }}
          >
            <div className="camera-grid" />
            <img
              className="camera-stream"
              src={streamUrl}
              alt="Franka Panda 仿真实时相机画面"
              onLoad={() => setFrameLoaded(true)}
              onError={() => {
                setFrameLoaded(false);
                window.setTimeout(() => setStreamRetry((attempt) => attempt + 1), 1800);
              }}
            />
            <div
              className="first-person-window"
              aria-label="机械臂末端第一视角传感器画面"
              onPointerDown={(event) => event.stopPropagation()}
              onWheel={(event) => { event.preventDefault(); event.stopPropagation(); }}
              onContextMenu={(event) => event.preventDefault()}
            >
              <div className="first-person-heading">
                <span><i className="live-dot" />第一视角</span>
                <select
                  aria-label="切换第一视角传感器画面"
                  value={sensorMode}
                  onChange={(event) => void changeSensorMode(event.target.value as SensorMode)}
                >
                  <option value="rgb">RGB</option>
                  <option value="depth">深度</option>
                  <option value="pointcloud">3D 坐标</option>
                  <option value="segmentation">分割</option>
                </select>
              </div>
              <img
                src={firstPersonStreamUrl}
                alt={`末端第一视角 ${sensorMode}`}
                onLoad={() => setFirstPersonStreamError("")}
                onError={() => setFirstPersonStreamError("第一视角视频流不可用，请检查当前遥控后端是否已更新并重启。")}
              />
              {firstPersonError && <div className="first-person-error" role="status">{firstPersonError}</div>}
              <div className="first-person-caption">
                {sensorMode === "pointcloud" ? "R:X · G:Y · B:Z（世界坐标）" : sensorMode === "depth" ? "颜色表示相机距离" : sensorMode === "segmentation" ? "按物体 / 链接着色" : "末端相机实时图像"}
              </div>
            </div>
            {!frameLoaded && <div className="frame-placeholder">
              <span className="spinner" />
              <strong>{frameMessage}</strong>
              <p>{frameDetail}</p>
              {simulationError && <code>{simulationError}</code>}
              <small>画面由 Genesis 后端渲染并通过 MJPEG 实时传输</small>
            </div>}
            <div className="camera-hint">左键拖动旋转 · Shift/右键拖动平移 · 滚轮缩放</div>
          </div>
          <div className="stage-footer"><span>{teleopState ? `仿真时间 ${round(teleopState.sim_time, 1)} s · ${round(teleopState.fps, 0)} FPS` : "Genesis Camera"}</span><span>{teleopState?.clamped ? "已到达工作空间边界" : "笛卡尔工作空间限制已启用"}</span></div>
        </section>

        <aside className="side right" aria-label="机器人状态">
          <div className="side-head"><h2>机器人状态</h2><span className={`pill ${simulationReady ? "pill-live" : ""}`}>{simulationReady ? "LIVE" : "WAIT"}</span></div>
          <div className="metric-grid">
            <div className="metric"><span>仿真帧率</span><strong>{teleopState ? `${round(teleopState.fps, 0)} FPS` : "-- FPS"}</strong></div>
            <div className="metric"><span>指令延迟</span><strong>{rtt === null ? "-- ms" : `${rtt.toFixed(1)} ms`}</strong></div>
            <div className="metric"><span>仿真时间</span><strong>{teleopState ? `${round(teleopState.sim_time, 1)} s` : "-- s"}</strong></div>
            <div className="metric"><span>运行状态</span><strong>{simulationError ? "异常" : simulationReady ? "运行中" : online ? "场景初始化" : "等待连接"}</strong></div>
          </div>
            {collisionMode && <div className="collision-alert" role="alert">
              <div className="collision-alert-title"><strong>碰撞触发 · 零力拖动</strong><span>{round(teleopState?.collision_force_n, 1)} N</span></div>
              <p>检测到机械臂外部接触力达到 {round(teleopState?.collision_threshold_n, 0)} N 阈值。仿真关节驱动力已关闭，等待人工检查。</p>
              {teleopState?.collision_link && <small>受力连杆：{teleopState.collision_link}</small>}
              {teleopState?.collision_position && <small>接触位置：{teleopState.collision_position.map((value) => round(value, 3)).join(", ")} m</small>}
              <button className="action-button primary" type="button" onClick={() => void resetCollision()}>检查后恢复位置控制</button>
            </div>}
          <div className="section-rule" />
          <div className="side-head"><h2>末端位姿</h2><span className="pill">m / quat</span></div>
          <div className="state-list">
            {(["X", "Y", "Z"] as const).map((axis, index) => <div key={axis}><span>位置 {axis}</span><b>{teleopState?.ee ? `${round(teleopState.ee[index], 3)} m` : "—"}</b></div>)}
            <div><span>姿态 WXYZ</span><b>{teleopState?.quat?.map((value) => round(value, 2)).join("  ") ?? "—"}</b></div>
            <div><span>夹爪开度</span><b>{teleopState ? `${Math.round(teleopState.gripper * 1000)} mm` : "—"}</b></div>
          </div>
          <div className="section-rule" />
          <div className="side-head"><h2>自然语言操作</h2><span className={`pill ${agentStatus?.configured ? "pill-live" : ""}`}>{agentStatus?.configured ? agentStatus.model : "程序模式"}</span></div>
          <div className="agent-panel">
            <label htmlFor="agent-task">输入指令，例如“抓取蓝色方块并放到托盘”</label>
            <textarea id="agent-task" value={agentTask} maxLength={500} onChange={(event) => setAgentTask(event.target.value)} placeholder="抓取蓝色方块并放到托盘" disabled={agentLoading || agentAutoRunning || programSubmitting || !!teleopState?.agent_busy} />
            <button className="action-button primary" type="button" onClick={() => void runNaturalProgram()} disabled={!simulationReady || !enabled || collisionMode || !!keys.length || agentLoading || agentAutoRunning || programSubmitting || !!teleopState?.homing || !!teleopState?.agent_busy || !agentTask.trim()}>
              {programSubmitting ? "正在调用程序…" : "自然语言调用程序"}
            </button>
            {teleopState?.program_status && teleopState.program_status.state !== "idle" && <div className="agent-plan" aria-live="polite">
              <strong>{teleopState.program_status.label} · {teleopState.program_status.state === "queued" ? "排队中" : teleopState.program_status.state === "running" ? "执行中" : teleopState.program_status.state === "completed" ? "已完成" : teleopState.program_status.state === "stopped" ? "已停止" : "未完成"}</strong>
              <span>{teleopState.program_status.message}{teleopState.program_status.state === "running" && teleopState.program_status.total > 0 ? `（${teleopState.program_status.step}/${teleopState.program_status.total}）` : ""}</span>
              {(["queued", "running"] as const).includes(teleopState.program_status.state as "queued" | "running") && <button className="action-button agent-stop-button" type="button" onClick={() => void stopAgentTask()}>停止程序</button>}
            </div>}
            {programError && <div className="recording-error" role="status">操作：{programError}</div>}
            <small className="agent-footnote">内置程序支持抓取蓝/绿方块、放入托盘、松开夹爪、回到初始姿态和重置场景。程序逐段移动并持续检查工作范围与碰撞保护。</small>
            <div className="button-row agent-controls">
              <button className="action-button" type="button" onClick={() => void analyzeAgentTask()} disabled={!agentStatus?.configured || !simulationReady || agentLoading || agentAutoRunning || !!keys.length || !!teleopState?.agent_busy || !agentTask.trim()}>
                {agentLoading ? "正在看图与分析…" : "分析并建议一步"}
              </button>
              <button className={`action-button ${agentAutoRunning ? "agent-stop-button" : "primary"}`} type="button" onClick={() => void (agentAutoRunning ? stopAgentTask() : runAgentTask())} disabled={!agentAutoRunning && (!agentStatus?.configured || !simulationReady || !enabled || collisionMode || !!keys.length || agentLoading || !!teleopState?.agent_busy || !agentTask.trim())}>
                {agentAutoRunning ? "停止任务" : "自主闭环执行"}
              </button>
            </div>
            {!agentStatus?.configured && <div className="recording-hint">配置本机 Ollama 视觉模型后可用。启动前设置 <code>GENESIS_TELEOP_VLM_MODEL</code>，详情见工作台 README。</div>}
            {agentPlan && <div className="agent-plan" aria-live="polite">
              <strong>{agentPlan.summary}</strong>
              {agentPlan.target_object && <span>目标：{agentPlan.target_object}</span>}
              <span>{agentPlan.reason}</span>
              {agentPlan.action === "move" && <code>ΔXYZ [{agentPlan.delta_m.map((value) => value.toFixed(3)).join(", ")}] m</code>}
              <small>置信度 {Math.round(agentPlan.confidence * 100)}% · 单步位移上限 3 cm/轴</small>
              <button className="action-button" type="button" onClick={() => void executeAgentPlan()} disabled={["observe", "complete"].includes(agentPlan.action) || agentAutoRunning || !enabled || collisionMode || !!keys.length || !!teleopState?.homing || !!teleopState?.agent_busy}>
                {agentPlan.action === "observe" ? "建议继续观察" : agentPlan.action === "complete" ? "任务已完成" : "确认并执行此步"}
              </button>
              <small>建议在 {agentPlan.expires_in_s} 秒内确认；若位置或场景改变，需重新分析。</small>
            </div>}
            {agentError && <div className="recording-error" role="status">VLM：{agentError}</div>}
            {agentAutoRunning && <small className="agent-running-note" role="status">闭环任务进行中：每步完成后重新观察，最多 20 步 / 3 分钟；可随时停止。</small>}
            <small className="agent-footnote">模型读取 RGB 仿真画面与末端状态。自主模式逐步重观察，每步限幅并经工作空间、逆运动学和碰撞保护；不连接真实机械臂。</small>
          </div>
          <div className="section-rule" />
          <div className="side-head"><h2>数据录制</h2><span className="pill">{teleopState?.recording_status === "starting" ? "启动中" : teleopState?.recording_status === "stopping" ? "保存中" : teleopState?.recording ? `录制中 · ${teleopState.recording_frames ?? 0} 帧` : "待机"}</span></div>
          <div className="recording-config">
            <label htmlFor="recording-format">数据格式</label>
            <select id="recording-format" value={recordingFormat} onChange={(event) => setRecordingFormat(event.target.value as RecordingFormat)} disabled={!!teleopState?.recording}>
              <option value="mp4">MP4 视频</option>
              <option value="mcap" disabled={!availableRecordingFormats.includes("mcap")}>MCAP（状态 + 图像）</option>
              <option value="lerobot" disabled={!availableRecordingFormats.includes("lerobot")}>LeRobot v2.1</option>
            </select>
            <label htmlFor="recording-task">任务描述</label>
            <input id="recording-task" value={recordingTask} maxLength={256} onChange={(event) => setRecordingTask(event.target.value)} disabled={!!teleopState?.recording} />
          </div>
          <div className="button-row"><button className={`action-button primary ${teleopState?.recording ? "recording" : ""}`} type="button" onClick={() => void toggleRecording()} disabled={teleopState?.recording_status === "starting" || teleopState?.recording_status === "stopping"}>{teleopState?.recording_status === "starting" ? "准备录制…" : teleopState?.recording_status === "stopping" ? "正在保存…" : teleopState?.recording ? "停止并保存" : "开始录制"}</button><button className="action-button" type="button" onClick={() => action("reset_scene")}>重置物体</button></div>
          {teleopState?.recording_error && <div className="recording-error" role="status">录制失败：{teleopState.recording_error}</div>}
          {!availableRecordingFormats.includes(recordingFormat) && <div className="recording-hint">未安装此格式依赖。请按工作台 README 安装记录依赖并重启后端。</div>}
          {!teleopState?.recording && teleopState?.recording_path && <div className="recording-hint">最近文件：{teleopState.recording_path}</div>}
          <div className="right-note">演示机器人：Franka Panda。控制目标由本地 Genesis 仿真执行。</div>
        </aside>

        <section className="dock" aria-label="底盘和末端快捷控制">
          <div className="dock-cell drive-cell">
            <div className="dock-title">末端平移 <span className="pill">WASD</span></div>
            <div className="drive-layout">
              <div className="direction-pad">
                {motionButton("前移", "KeyW", "north")}
                {motionButton("左移", "KeyA", "west")}
                {motionButton("后移", "KeyS", "south")}
                {motionButton("右移", "KeyD", "east")}
              </div>
              <div className="dock-info"><span>平移速度 <b>{speed.toFixed(2)} m/s</b></span><span>末端位置 <b>{teleopState?.ee?.map((v) => round(v, 2)).join(", ") ?? "—"}</b></span><span>控制模式 <b>{mode === "joint" ? "关节" : "笛卡尔"}</b></span></div>
              <div className="height-pad"><span>升降 R/F</span>{motionButton("上升", "KeyR", "north")}{motionButton("下降", "KeyF", "south")}</div>
            </div>
          </div>
          <div className="dock-cell rotation-cell">
            <div className="dock-title">高度与旋转 <span className="pill">Q / E</span></div>
            <div className="rotation-controls"><button className={`action-button ${activeKeys.has("KeyQ") ? "held" : ""}`} type="button" onPointerDown={() => enabled && updateKeys("KeyQ", true)} onPointerUp={() => updateKeys("KeyQ", false)} onPointerLeave={() => updateKeys("KeyQ", false)}>逆时针旋转 Q</button><button className={`action-button ${activeKeys.has("KeyE") ? "held" : ""}`} type="button" onPointerDown={() => enabled && updateKeys("KeyE", true)} onPointerUp={() => updateKeys("KeyE", false)} onPointerLeave={() => updateKeys("KeyE", false)}>顺时针旋转 E</button></div>
            <p>方向键或 WASD 平移，R/F 调整高度，Q/E 绕 Z 轴旋转。按住方向按钮持续移动。</p>
          </div>
          <div className="dock-cell speed-cell">
            <div className="dock-title">末端速度</div>
            <div className="slider-row"><label htmlFor="dock-speed">线速度</label><input id="dock-speed" type="range" min="0.02" max="0.35" step="0.01" value={speed} onChange={(event) => setSpeed(Number(event.target.value))} disabled={programBusy} /><output>{speed.toFixed(2)} m/s</output></div>
            <div className="dock-state"><span>当前末端</span><b>{teleopState?.ee?.map((v) => round(v, 3)).join(", ") ?? "—"}</b></div>
            <div className="dock-state"><span>关节模式</span><b>J1—J7</b></div>
          </div>
          <div className="dock-cell work-cell">
            <div className="dock-title">工作状态</div>
            <div className={`connection ${online && !simulationError ? "" : "offline"}`}><i className="live-dot" /><strong>{simulationError ? "仿真异常" : simulationReady ? "仿真运行中" : online ? "场景初始化中" : "等待仿真连接"}</strong></div>
            <div className="help-text">{simulationError || "Franka Panda 由本地 Genesis 仿真执行控制。"}</div>
            <div className="button-row"><button className="action-button" type="button" onClick={() => action("home")} disabled={!enabled || !!teleopState?.homing || collisionMode || programBusy}>{teleopState?.homing ? "平滑回零中…" : "回到初始姿态"}</button><button className="action-button" type="button" onClick={toggleControl} disabled={collisionMode}>{enabled ? "停止控制" : "恢复控制"}</button></div>
          </div>
        </section>
      </main>
      {toast && <div className="toast" role="status" aria-live="polite">{toast}</div>}
      {backendCompatibilityError && <div className="backend-version-alert" role="alert">{backendCompatibilityError}</div>}
    </div>
  );
}

export default App;
