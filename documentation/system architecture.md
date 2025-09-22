```mermaid
flowchart LR

subgraph OP[Operator & Client System]
  UI[Operator UI<br/>(Keyboard / Xbox / GUI)]
  CLIENT[Client System<br/>(Task Console & Viewer)]
end

NET[(Wi-Fi / Ethernet)]

subgraph SPOT[SPOT Robot]
  direction TB

  subgraph SENS[Perception]
    CAMS[Cameras<br/>(Fisheye + Depth)]
    IMG[Image Service]
  end

  subgraph CTRL[Control & Autonomy]
    TAG[AprilTag / Fiducial Detection]
    AUTO[Autonomy<br/>(GraphNav / Navigation)]
    TASK[Task Manager<br/>(Image capture, move,<br/>send digital signal)]
    MAN[Manual Override Module]
    MOT[Motion Control<br/>(Robot Command)]
    SAFE[Safety<br/>(E-Stop, Lease, Auth)]
  end

  LOG[Data Logger]
  TELE[Telemetry / Status]
end

STORE[(Storage / BIM Server)]

%% Connections
UI -->|Teleop commands| NET
CLIENT <-->|RPC / APIs| NET
NET -->|gRPC| MAN
NET -->|gRPC| AUTO

CAMS --> IMG --> TAG --> AUTO --> TASK --> MOT
MAN --> MOT
SAFE -. supervises .-> MOT

TASK -->|Images / Signals| LOG --> STORE
TELE --> CLIENT
