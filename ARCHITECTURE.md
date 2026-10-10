# Architecture

Low-latency GPU inference delivered to a web browser. The browser keeps a stable frame rate regardless of backend inference speed.

## Contents

- [Overview](#overview)
- [Data flow](#data-flow)
- [Components](#components)
- [Backend](#backend)
- [Inference pipeline](#inference-pipeline)
- [Triton](#triton)
- [Backpressure](#backpressure)
- [On-demand GPU lifecycle](#on-demand-gpu-lifecycle)
- [Monitoring](#monitoring)

---

## Overview

Four layers:

1. **Browser client** — control UI and engine runtime
2. **AWS orchestration** — start/stop of the GPU instance
3. **GPU host (EC2)** — Nginx, Docker Compose
4. **Inference runtime** — FastAPI backend and Triton

Principle: **frontend FPS is decoupled from inference FPS.** The backend may run at 5–20 FPS depending on GPU load while the browser renders at 30 FPS or more and interpolates between results.

---

## Data flow

```mermaid
flowchart TB
    subgraph Client
        UI["Control UI"] <-- "postMessage" --> ENG["Engine (iframe)<br/>getUserMedia · JPEG · canvas"]
    end
    UI -- "start request" --> LAM["API Gateway + Lambda"] --> EC2
    subgraph EC2["EC2 GPU instance"]
        NG["Nginx<br/>TLS · WS proxy"] --> BE["FastAPI backend"]
        BE -- "gRPC" --> TR["Triton"] --> GPU[("T4")]
    end
    ENG -- "WSS: JPEG frames" --> NG
    BE -- "JSON results" --> ENG
```

Results travel back as GPU → Triton → FastAPI → WebSocket → engine → canvas overlay.

---

## Components

### Control UI

Session start, model selection, status display, stop. It does not transmit video frames.

### Engine

Runs in an iframe and communicates with the UI through `window.postMessage`.

- camera capture with `getUserMedia`
- frames resized to 320×240 and JPEG-encoded
- WebSocket transport
- overlay rendering at 30–60 FPS, interpolating between backend results

The iframe provides an independent lifecycle, crash isolation and a restartable runtime.

### Nginx

Runs on the host, outside Docker. Terminates TLS, proxies the WebSocket endpoint `wss://api.<domain>/ws/inference` and exposes `/health`. Reference configs are in [`nginx/`](nginx/).

---

## Backend

Asynchronous WebSocket server built with FastAPI.

- receives binary JPEG frames
- JPEG decoding in a thread pool, off the event loop
- per-session frame queue
- concurrent inference sessions limited by an `asyncio.Semaphore`
- result streaming, heartbeat ping every 5 s
- no video is stored or rendered on the server

---

## Inference pipeline

```mermaid
flowchart LR
    F["JPEG frame"] --> D["Decode<br/>(thread pool)"]
    D --> G{{"asyncio.gather"}}
    G --> Y["YOLO26s detection"] --> BT["ByteTrack"]
    G --> PO["YOLO26s-pose"]
    G --> OC["PaddleOCR<br/>(throttled)"]
    G --> OF["Farneback flow"]
    BT --> R["JSON result"]
    PO --> R
    OC --> R
    OF --> R
    R --> WS["WebSocket"]
    R -. "batched summaries" .-> LLM["LangChain · GPT-4o-mini"]
```

Modules are enabled per session through the `init` message.

---

## Triton

All TensorRT models are served by Triton over async gRPC. Model configs live in [`triton/model_repository/`](triton/model_repository/):

| Model | Platform | Max batch | Input | Output |
|---|---|---|---|---|
| `yolo` | `tensorrt_plan` | 1 | FP16 `[3, 640, 640]` | FP32 `[300, 6]` |
| `yolo_pose` | `tensorrt_plan` | 1 | FP16 `[3, 640, 640]` | FP32 `[300, 57]` |

GPU contention is bounded by limiting the number of concurrently running inference sessions.

---

## Backpressure

Each session owns a frame queue of size 2. When a new frame arrives and the queue is full, the oldest frame is discarded.

- latency stays bounded by the model, not by the sender
- memory use is constant
- the UI always shows the most recent result

The system is tuned for low latency rather than maximum throughput.

---

## On-demand GPU lifecycle

```mermaid
sequenceDiagram
    participant UI as Frontend
    participant API as API Gateway / Lambda
    participant EC2 as EC2 g4dn
    participant EB as EventBridge
    UI->>API: start request
    API->>EC2: start (if stopped)
    API->>EB: schedule auto-shutdown
    EC2->>EC2: systemd → Docker Compose up
    UI->>EC2: poll /health until ready
    EB->>API: stop Lambda after inactivity
    API->>EC2: stop
```

Boot chain: EC2 boot → systemd → Docker Compose → Triton → Backend → Prometheus.

---

## Monitoring

Prometheus scrapes the backend and Triton.

| Source | Examples |
|---|---|
| Backend | `inference_active_sessions`, `inference_duration_seconds`, `gpu_semaphore_in_use`, `pipeline_fps`, `session_frames_total` |
| Triton | inference counts, queue and compute durations (port 8002) |
