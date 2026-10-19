"""Start the real UR/RG2, RealSense, and Xense ZMQ nodes."""

import threading
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Literal

import tyro

from gello.robots.robot import PrintRobot
from gello.tactile.xense import (
    AsyncXenseTactileReader,
    XENSE_DEFAULT_FORCE_HEIGHT,
    XENSE_DEFAULT_FORCE_WIDTH,
    XENSE_DEFAULT_LEFT_SERIAL,
    XENSE_DEFAULT_RIGHT_SERIAL,
    XenseTactileReader,
)
from gello.tactile.xense_process import ProcessXenseTactileReader
from gello.zmq_core.camera_node import ZMQServerCamera
from gello.zmq_core.robot_node import ZMQServerRobot
from gello.zmq_core.tactile_node import ZMQServerTactile


@dataclass
class Args:
    robot: Literal["ur", "none", "print"] = "ur"
    robot_port: int = 6001
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    hostname: str = "127.0.0.1"
    robot_ip: str = "192.168.1.102"
    enable_camera: bool = True
    enable_wrist_camera: bool = True
    enable_base_camera: bool = True
    wrist_camera_device_id: str = "412622271117"
    base_camera_device_id: str = "335522073597"
    wrist_camera_flip: bool = False
    base_camera_flip: bool = False
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    enable_depth: bool = False
    camera_backend: Literal["lerobot", "gello", "realsense"] = "lerobot"
    enable_tactile: bool = True
    tactile_source: Literal["xense"] = "xense"
    left_tactile_port: int = 5100
    right_tactile_port: int = 5101
    xense_left_serial: str = XENSE_DEFAULT_LEFT_SERIAL
    xense_right_serial: str = XENSE_DEFAULT_RIGHT_SERIAL
    xense_left_cam_id: int | None = None
    xense_right_cam_id: int | None = None
    xense_force_height: int = XENSE_DEFAULT_FORCE_HEIGHT
    xense_force_width: int = XENSE_DEFAULT_FORCE_WIDTH
    xense_backend: Literal["process", "thread"] = "process"
    xense_poll_interval_s: float = 0.0
    xense_warmup_timeout_s: float = 2.0
    xense_max_age_s: float = 0.15


def make_robot(args: Args):
    if args.robot == "ur":
        from gello.robots.ur import URRobot

        return URRobot(robot_ip=args.robot_ip, no_gripper=False)
    if args.robot in ("none", "print"):
        return PrintRobot(7)
    raise NotImplementedError("Choose one of: ur, none, print")


def launch_robot_server(args: Args) -> None:
    robot = make_robot(args)
    try:
        _launch_servers(args, robot)
    finally:
        close = getattr(robot, "close", None)
        if callable(close):
            close()


def _launch_servers(args: Args, robot) -> None:
    robot_server = ZMQServerRobot(robot, port=args.robot_port, host=args.hostname)
    print(f"Starting robot server on port {args.robot_port}")

    threads = [threading.Thread(target=robot_server.serve, daemon=True)]
    servers = [robot_server]
    cameras = []
    tactiles = []

    def make_camera(device_id: str, flip: bool):
        if args.camera_backend == "lerobot":
            from gello.cameras.lerobot_realsense_camera import LeRobotRealSenseCamera

            return LeRobotRealSenseCamera(
                device_id=device_id,
                flip=flip,
                width=args.camera_width,
                height=args.camera_height,
                fps=args.camera_fps,
            )
        if args.camera_backend in ("gello", "realsense"):
            from gello.cameras.realsense_camera import RealSenseCamera

            return RealSenseCamera(
                device_id=device_id,
                flip=flip,
                width=args.camera_width,
                height=args.camera_height,
                fps=args.camera_fps,
                enable_depth=args.enable_depth,
            )
        raise ValueError("camera_backend must be 'lerobot', 'gello', or 'realsense'")

    def make_xense(
        serial_number: str,
        *,
        cam_id: int | None,
        name: str,
    ) -> AsyncXenseTactileReader:
        process_backend = args.xense_backend == "process"
        reader_type = ProcessXenseTactileReader if process_backend else XenseTactileReader
        reader = reader_type(
            None if cam_id is not None else serial_number,
            cam_id=cam_id,
            force_height=args.xense_force_height,
            force_width=args.xense_force_width,
            name=name,
            **({"poll_interval_s": args.xense_poll_interval_s} if process_backend else {}),
        )
        try:
            async_reader = AsyncXenseTactileReader(
                reader,
                force_height=args.xense_force_height,
                force_width=args.xense_force_width,
                poll_interval_s=0.0 if process_backend else args.xense_poll_interval_s,
                name=name,
                max_age_s=args.xense_max_age_s,
            )
        except BaseException:
            reader.close()
            raise
        tactiles.append(async_reader)
        if not async_reader.wait_for_frame(args.xense_warmup_timeout_s):
            print(
                f"Warning: no {name} frame arrived during async warmup; "
                "ZMQ server will return zero Xense frames until data arrives."
            )
        return async_reader

    try:
        if args.enable_camera and args.enable_wrist_camera:
            wrist_camera = make_camera(
                device_id=args.wrist_camera_device_id,
                flip=args.wrist_camera_flip,
            )
            cameras.append(wrist_camera)
            wrist_camera_server = ZMQServerCamera(
                camera=wrist_camera,
                port=args.wrist_camera_port,
                host=args.hostname,
            )
            servers.append(wrist_camera_server)
            print(f"Starting wrist camera server on port {args.wrist_camera_port}")
            threads.append(threading.Thread(target=wrist_camera_server.serve, daemon=True))

        if args.enable_camera and args.enable_base_camera:
            base_camera = make_camera(
                device_id=args.base_camera_device_id,
                flip=args.base_camera_flip,
            )
            cameras.append(base_camera)
            base_camera_server = ZMQServerCamera(
                camera=base_camera,
                port=args.base_camera_port,
                host=args.hostname,
            )
            servers.append(base_camera_server)
            print(f"Starting base camera server on port {args.base_camera_port}")
            threads.append(threading.Thread(target=base_camera_server.serve, daemon=True))

        if args.enable_tactile:
            if args.tactile_source != "xense":
                raise ValueError("--tactile-source must be 'xense'")

            left_name = (
                f"left Xense tactile cam_id={args.xense_left_cam_id}"
                if args.xense_left_cam_id is not None
                else f"left Xense tactile {args.xense_left_serial}"
            )
            right_name = (
                f"right Xense tactile cam_id={args.xense_right_cam_id}"
                if args.xense_right_cam_id is not None
                else f"right Xense tactile {args.xense_right_serial}"
            )
            left_tactile = make_xense(
                args.xense_left_serial,
                cam_id=args.xense_left_cam_id,
                name=left_name,
            )
            left_tactile_server = ZMQServerTactile(
                tactile=left_tactile,
                port=args.left_tactile_port,
                host=args.hostname,
            )
            servers.append(left_tactile_server)
            print(f"Starting left Xense tactile ZMQ server on port {args.left_tactile_port}")
            threads.append(threading.Thread(target=left_tactile_server.serve, daemon=True))

            right_tactile = make_xense(
                args.xense_right_serial,
                cam_id=args.xense_right_cam_id,
                name=right_name,
            )
            right_tactile_server = ZMQServerTactile(
                tactile=right_tactile,
                port=args.right_tactile_port,
                host=args.hostname,
            )
            servers.append(right_tactile_server)
            print(f"Starting right Xense tactile ZMQ server on port {args.right_tactile_port}")
            threads.append(threading.Thread(target=right_tactile_server.serve, daemon=True))

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        for server in reversed(servers):
            server.stop()
        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        with ExitStack() as cleanup:
            for device in [*cameras, *tactiles]:
                close = getattr(device, "close", None)
                if callable(close):
                    cleanup.callback(close)
            for server, thread in zip(servers, threads):
                if not thread.is_alive():
                    cleanup.callback(server.close)


def main(args: Args) -> None:
    launch_robot_server(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
