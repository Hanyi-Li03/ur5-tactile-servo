import os
import time
from typing import List, Optional, Tuple

import numpy as np

from gello.cameras.camera import CameraDriver


def get_device_ids() -> List[str]:
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = ctx.query_devices()
    device_ids = []
    for dev in devices:
        dev.hardware_reset()
        device_ids.append(dev.get_info(rs.camera_info.serial_number))
    time.sleep(2)
    return device_ids


class RealSenseCamera(CameraDriver):
    def __repr__(self) -> str:
        return f"RealSenseCamera(device_id={self._device_id})"

    def __init__(
        self,
        device_id: Optional[str] = None,
        flip: bool = False,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_depth: bool = True,
    ):
        import pyrealsense2 as rs

        self._device_id = device_id
        self._width = width
        self._height = height
        self._fps = fps
        self._enable_depth = enable_depth
        self._last_image = None
        self._last_depth = None

        if device_id is None:
            ctx = rs.context()
            devices = ctx.query_devices()
            for dev in devices:
                dev.hardware_reset()
            time.sleep(2)
            self._pipeline = rs.pipeline()
            config = rs.config()
        else:
            self._pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(device_id)

        if enable_depth:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._pipeline.start(config)
        self._flip = flip

    def read(self, img_size=None):
        return self._read(img_size)

    def read_with_timestamp(self, img_size=None):
        return self._read(img_size, with_timestamp=True)

    def read_rgb(self, img_size=None):
        return self.read_rgb_with_timestamp(img_size)["image"]

    def read_rgb_with_timestamp(self, img_size=None):
        return self._read(img_size, with_timestamp=True, include_depth=False)

    def _read(
        self,
        img_size: Optional[Tuple[int, int]] = None,  # farthest: float = 0.12
        with_timestamp: bool = False,
        include_depth: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read a frame from the camera.

        Args:
            img_size: The size of the image to return. If None, the original size is returned.
            farthest: The farthest distance to map to 255.

        Returns:
            np.ndarray: The color image, shape=(H, W, 3)
            np.ndarray: The depth image, shape=(H, W, 1)
        """
        import cv2

        frames = None
        last_error = None
        for _ in range(1 if with_timestamp else 3):
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=200 if with_timestamp else 5000)
                break
            except RuntimeError as exc:
                last_error = exc
                time.sleep(0.02)
        if frames is None:
            if not with_timestamp and self._last_image is not None and self._last_depth is not None:
                print(f"[{self}] frame timeout, reusing last frame")
                return self._last_image.copy(), self._last_depth.copy()
            raise RuntimeError(f"{self} failed to read frames: {last_error}")

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame() if include_depth and self._enable_depth else None
        if not color_frame or (include_depth and self._enable_depth and not depth_frame):
            if not with_timestamp and self._last_image is not None and self._last_depth is not None:
                print(f"[{self}] incomplete frame, reusing last frame")
                return self._last_image.copy(), self._last_depth.copy()
            raise RuntimeError(f"{self} returned an incomplete frame")

        color_image = np.asanyarray(color_frame.get_data())
        host_timestamp = time.time()
        depth_image = None
        if depth_frame:
            depth_image = np.asanyarray(depth_frame.get_data())
        elif include_depth:
            depth_image = np.zeros((self._height, self._width), dtype=np.uint16)
        # depth_image = cv2.convertScaleAbs(depth_image, alpha=0.03)
        if img_size is None:
            image = color_image[:, :, ::-1]
            depth = depth_image
        else:
            image = cv2.resize(color_image, img_size)[:, :, ::-1]
            depth = cv2.resize(depth_image, img_size) if include_depth else None

        # rotate 180 degree's because everything is upside down in order to center the camera
        if self._flip:
            image = cv2.rotate(image, cv2.ROTATE_180)
            if include_depth:
                depth = cv2.rotate(depth, cv2.ROTATE_180)[:, :, None]
        else:
            if include_depth:
                depth = depth[:, :, None]

        if include_depth:
            # Keep the legacy fallback cache as a matching RGB/depth pair.
            self._last_image = image
            self._last_depth = depth
        if with_timestamp:
            frame = {"image": image.copy(),
                     "capture_timestamp": color_frame.get_timestamp()/1000.0,
                     "host_timestamp": host_timestamp,
                     "timestamp_domain": str(color_frame.get_frame_timestamp_domain()),
                     "frame_number": color_frame.get_frame_number()}
            if include_depth:
                frame["depth"] = depth.copy()
            return frame
        return image, depth

    def close(self):
        self._pipeline.stop()


def _debug_read(camera, save_datastream=False):
    import cv2

    cv2.namedWindow("image")
    cv2.namedWindow("depth")
    counter = 0
    if not os.path.exists("images"):
        os.makedirs("images")
    if save_datastream and not os.path.exists("stream"):
        os.makedirs("stream")
    while True:
        time.sleep(0.1)
        image, depth = camera.read()
        depth = np.concatenate([depth, depth, depth], axis=-1)
        key = cv2.waitKey(1)
        cv2.imshow("image", image[:, :, ::-1])
        cv2.imshow("depth", depth)
        if key == ord("s"):
            cv2.imwrite(f"images/image_{counter}.png", image[:, :, ::-1])
            cv2.imwrite(f"images/depth_{counter}.png", depth)
        if save_datastream:
            cv2.imwrite(f"stream/image_{counter}.png", image[:, :, ::-1])
            cv2.imwrite(f"stream/depth_{counter}.png", depth)
        counter += 1
        if key == 27:
            break


if __name__ == "__main__":
    device_ids = get_device_ids()
    print(f"Found {len(device_ids)} devices")
    print(device_ids)
    rs = RealSenseCamera(flip=True, device_id=device_ids[0])
    im, depth = rs.read()
    _debug_read(rs, save_datastream=True)
