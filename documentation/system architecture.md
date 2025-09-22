```mermaid
flowchart LR
%% Layout direction
%% LR = left-to-right for wide reports


subgraph OP[Operator & Client System]
UI[Operator UI\n(Keyboard / X‑Box / GUI)]
CLIENT[Client System\n(Task Console & Data Viewer)]
end


NET[(Wi‑Fi / Ethernet)]


subgraph SPOT[SPOT Robot]
direction TB
subgraph SENS[Perception]
CAMS[Cameras\n(Fisheye + Depth)]
IMG[Image Service]
end


subgraph CTRL[Control & Autonomy]
TAG[AprilTag / Fiducial Detection]
AUTO[Autonomy\n(GraphNav / Navigation)]
TASK[Task Manager\n(Image capture, move,\nsend digital signal)]
MAN[Manual Override Module]
MOT[Motion Control\n(Robot Command)]
SAFE[Safety\n(E‑Stop, Lease, Auth)]
end


LOG[Data Logger]
end


STORE[(Optional: Storage / BIM / Server)]


%% Connections
UI -->|Teleop commands| NET -->|RPC (gRPC)| MAN
UI -- Feedback (status, video) --> CLIENT
CLIENT <-->|RPC / APIs| NET
NET --> SPOT


CAMS --> IMG --> TAG
TAG --> AUTO
AUTO --> TASK
MAN --> MOT
AUTO --> MOT
SAFE -. supervises .-> MOT


%% Outputs / telemetry
TASK -->|Images / Signals| LOG
LOG -->|Uploads| STORE
SPOT -->|State, health| CLIENT


%% Notes
classDef dim fill:#f8f9fa,stroke:#c8c8c8,color:#111;
classDef strong fill:#e8f5e9,stroke:#29b06f,color:#0b4127;
class SENS,CTRL,strong
