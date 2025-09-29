```mermaid
flowchart TD
  A[run_program.py launcher] --> B[basic_program.py Spot control UI]
  B --> C[KyeboardSpotManager spot_control_manager.py]
  C --> D[FollowFiducial fiducial_follow.py]
  D --> E[DisplayImagesAsync]

  subgraph SDK_Clients
    RC[RobotCommandClient]
    RS[RobotStateClient]
    IM[ImageClient]
    WO[WorldObjectClient]
    PW[PowerClient]
    LS[LeaseClient]
    TS[TimeSync]
    ES[EstopClient]
  end

  C --> RC
  C --> RS
  C --> PW
  C --> LS
  C --> TS
  C --> ES

  D --> RC
  D --> RS
  D --> IM
  D --> WO
  D --> PW

  subgraph Spot_Robot
    R[Spot robot services]
  end

  RC --> R
  RS --> R
  IM --> R
  WO --> R
  PW --> R
  LS --> R
  TS --> R
  ES --> R
