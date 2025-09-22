```mermaid
flowchart LR

subgraph "Operator and Client System"
  UI[Operator UI]
  CLIENT[Client System]
end

NET[WiFi or Ethernet]

subgraph "SPOT Robot"
  direction TB
  CAMS[Cameras]
  IMG[Image service]
  TAG[Fiducial detection]
  AUTO[Autonomy]
  TASK[Task manager]
  MAN[Manual override]
  MOT[Motion control]
  SAFE[Safety]
  LOG[Data logger]
  TELE[Telemetry]
end

STORE[Storage or BIM server]

UI --> NET
CLIENT <--> NET
NET --> MAN
NET --> AUTO

CAMS --> IMG --> TAG --> AUTO --> TASK --> MOT
MAN --> MOT
SAFE --> MOT

TASK --> LOG --> STORE
TELE --> CLIENT
