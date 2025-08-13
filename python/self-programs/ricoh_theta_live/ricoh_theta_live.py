# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""
Live preview from Ricoh THETA using the Theta helper class (no Qt deps).

Requirements:
- opencv-python
- numpy

Press 'q' in the preview window to exit.
"""

import time
import cv2
import numpy as np

from ricoh_theta import Theta

SSID = "THETAYN30103903"  # <-- your camera SSID

def main():
    # You said you're in client mode now:
    camera = Theta(theta_ssid=SSID, client_mode=True, show_state_at_init=False)

    # Optional: sanity check the camera state
    try:
        camera.showState()
    except Exception as e:
        print(f"[WARN] showState failed: {e}")

    print("[INFO] Starting live preview… Press 'q' to quit.")
    gen = None
    frame_count = 0
    t0 = time.time()

    # For FPS overlay smoothing
    last_fps = 0.0
    last_fps_update = time.time()

    try:
        gen = camera.yieldLivePreview()  # generator of JPEG bytes

        # Optional warm-up
        _ = next(gen, None)

        while True:
            jpg = next(gen, None)
            if jpg is None:
                print("[ERROR] Preview generator ended.")
                break

            # Decode JPEG → BGR frame
            arr = np.frombuffer(jpg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                # Skip occasional partial frame
                continue

            frame_count += 1

            # Update FPS every 0.5s to avoid flicker
            now = time.time()
            if now - last_fps_update >= 0.5:
                elapsed = now - t0
                if elapsed > 0:
                    last_fps = frame_count / elapsed
                last_fps_update = now

            # Draw FPS on the frame (top-left corner)
            cv2.putText(
                frame,
                f"FPS: {last_fps:.1f}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Optionally resize for display
            # frame = cv2.resize(frame, (960, 480))

            cv2.imshow("Ricoh THETA Live Preview", frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except StopIteration:
        print("[INFO] Preview stopped (generator exhausted).")
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as e:
        print(f"[ERROR] Live preview error: {e}")
    finally:
        try:
            if gen is not None:
                gen.close()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()