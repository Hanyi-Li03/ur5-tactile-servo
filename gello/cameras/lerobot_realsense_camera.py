from typing import Optional, Tuple

import numpy as np
from PIL import Image

from gello.cameras.camera import CameraDriver


class LeRobotRealSenseCamera(CameraDriver):
    def __repr__(self) -> str:
        return f"LeRobotRealSenseCamera(device_id={self._device_id})"

    def __init__(
        self,
        device_id: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        flip: bool = False,
    ):
        from lerobot.common.cameras.configs import ColorMode, Cv2Rotation
        from lerobot.common.cameras.realsense.camera_realsense import RealSenseCamera
        from lerobot.common.cameras.realsense.configuration_realsense import (
            RealSenseCameraConfig,
        )

        self._device_id = device_id
        self._width = width
        self._height = height
        self._fps = fps
        self._flip = flip
        self._last_image = None

        config = RealSenseCameraConfig(
            serial_number_or_name=device_id,
            fps=fps,
            width=width,
            height=height,
            color_mode=ColorMode.RGB,
            use_depth=False,
            rotation=Cv2Rotation.ROTATE_180 if flip else Cv2Rotation.NO_ROTATION,
        )
        self._camera = RealSenseCamera(config)
        self._camera.connect()

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        image = self._camera.read()
        image = np.asarray(image, dtype=np.uint8)

        if img_size is not None and image.shape[:2] != (img_size[1], img_size[0]):
            image = np.asarray(Image.fromarray(image).resize(img_size), dtype=np.uint8)

        depth = np.zeros(image.shape[:2] + (1,), dtype=np.uint16)
        self._last_image = image
        return image, depth

    def close(self) -> None:
        self._camera.disconnect()
