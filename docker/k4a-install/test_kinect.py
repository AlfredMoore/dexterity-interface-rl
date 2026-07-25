import cv2
import numpy as np

from pyk4a import (
    PyK4A,
    Config,
    FPS,
    ColorResolution,
    DepthMode,
    ImageFormat,
)


camera = PyK4A(
    Config(
        color_resolution=ColorResolution.RES_720P,
        color_format=ImageFormat.COLOR_BGRA32,
        depth_mode=DepthMode.NFOV_2X2BINNED,
        camera_fps=FPS.FPS_30,
        synchronized_images_only=True,
    )
)

camera.start()
cv2.namedWindow("Azure Kinect RGB-D", cv2.WINDOW_NORMAL)

try:
    while True:
        capture = camera.get_capture()

        if capture.color is None or capture.depth is None:
            continue

        color = cv2.cvtColor(
            capture.color,
            cv2.COLOR_BGRA2BGR,
        )

        depth = capture.transformed_depth
        if depth is None:
            continue

        valid = depth > 0
        depth_clipped = np.clip(depth, 250, 2500)

        depth_8bit = (
            (depth_clipped.astype(np.float32) - 250)
            / (2500 - 250)
            * 255
        ).astype(np.uint8)

        depth_8bit = 255 - depth_8bit
        depth_color = cv2.applyColorMap(
            depth_8bit,
            cv2.COLORMAP_TURBO,
        )
        depth_color[~valid] = 0

        display = np.hstack((color, depth_color))
        display = cv2.resize(
            display,
            (1600, 450),
            interpolation=cv2.INTER_AREA,
        )

        cv2.imshow("Azure Kinect RGB-D", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
finally:
    camera.stop()
    cv2.destroyAllWindows()