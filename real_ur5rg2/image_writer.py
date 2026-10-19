"""Lossless PNG recording with less CPU spent on compression."""
import multiprocessing
import signal
import threading
from pathlib import Path

import numpy as np
from PIL import Image
from lerobot.common.datasets.image_writer import AsyncImageWriter, image_array_to_pil_image


def _png_worker(queue, errors, failed, num_threads: int, compress_level: int) -> None:
    # Let the collector handle Ctrl+C and drain its queue in finally. Killing
    # encoders with the same terminal interrupt can strand the queue feeder.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    def run() -> None:
        while True:
            item = queue.get()
            try:
                if item is None:
                    return
                array, path = item
                image = image_array_to_pil_image(array) if isinstance(array, np.ndarray) else array
                if not isinstance(image, Image.Image):
                    raise TypeError(f"Unsupported image type: {type(array)}")
                image.save(path, format="PNG", compress_level=compress_level)
            except Exception as exc:
                if not failed.is_set():
                    failed.set()
                    errors.put(f"Cannot write PNG {item[1]}: {exc}")
            finally:
                queue.task_done()

    threads = [threading.Thread(target=run) for _ in range(num_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


class RecordingImageWriter(AsyncImageWriter):
    """Keep LeRobot's writer interface and queue, with explicit PNG compression.

    Worker targets are module-level functions so both fork and spawn work.
    Compression changes encoded size and CPU cost, not decoded pixel values.
    """

    def __init__(self, *, compress_level: int = 1, num_processes: int = 2, num_threads: int = 4):
        if not 0 <= compress_level <= 9:
            raise ValueError("PNG compression level must be between 0 and 9")
        if num_processes < 1 or num_threads < 1:
            raise ValueError("PNG writer requires positive process and thread counts")
        self.compress_level = int(compress_level)
        self.num_processes = int(num_processes)
        self.num_threads = int(num_threads)
        self.threads = []
        self.processes = []
        self._stopped = False
        self._error = None
        self._errors_closed = False
        context = multiprocessing.get_context()
        self.queue = context.JoinableQueue()
        self._errors = context.SimpleQueue()
        self._failed = context.Event()
        try:
            for _ in range(self.num_processes):
                process = context.Process(
                    target=_png_worker,
                    args=(self.queue, self._errors, self._failed, self.num_threads, self.compress_level),
                    daemon=True,
                )
                process.start()
                self.processes.append(process)
        except BaseException:
            self.stop()
            raise

    def _raise_error(self) -> None:
        if not self._errors_closed and not self._errors.empty():
            self._error = self._errors.get()
        if self._error is not None:
            raise RuntimeError(self._error)

    def save_image(self, image, fpath: Path) -> None:
        self._raise_error()
        super().save_image(image, fpath)

    def wait_until_done(self) -> None:
        super().wait_until_done()
        self._raise_error()

    def stop(self) -> None:
        try:
            super().stop()
            self._raise_error()
        finally:
            if not self._errors_closed:
                self._errors.close()
                self._errors_closed = True
