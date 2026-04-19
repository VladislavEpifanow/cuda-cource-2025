import os

import cv2
import numpy as np
import tensorrt as trt
import torch

from .config import INPUT_HEIGHT, INPUT_WIDTH
from .video_utils import preprocess_to_chw


def _iter_video_samples(video_path, num_samples):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open calibration video: {video_path}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            sample_count = min(num_samples, frame_count)
            target_indices = np.unique(
                np.linspace(0, frame_count - 1, num=sample_count, dtype=np.int64)
            )
            target_pos = 0
            frame_idx = 0

            while target_pos < len(target_indices):
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx == int(target_indices[target_pos]):
                    yield preprocess_to_chw(frame)
                    target_pos += 1

                frame_idx += 1
        else:
            produced = 0
            while produced < num_samples:
                ret, frame = cap.read()
                if not ret:
                    break
                yield preprocess_to_chw(frame)
                produced += 1
    finally:
        cap.release()


class VideoEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """
    INT8 calibrator that samples frames directly from video.
    No manual frame slicing is required.
    """

    def __init__(self, video_path, num_samples, cache_file):
        super().__init__()
        self.video_path = video_path
        self.num_samples = max(1, int(num_samples))
        self.cache_file = cache_file
        self.batch_size = 1

        self.sample_iter = _iter_video_samples(video_path=self.video_path, num_samples=self.num_samples)
        self.next_sample = self._pull_next_sample(required=True)

        self.device_input = torch.empty(
            (self.batch_size, 3, INPUT_HEIGHT, INPUT_WIDTH), dtype=torch.float32, device="cuda"
        )

    def get_batch_size(self):
        return self.batch_size

    def _pull_next_sample(self, required=False):
        try:
            return next(self.sample_iter)
        except StopIteration:
            if required:
                raise RuntimeError(
                    f"Failed to read any frames from calibration video: {self.video_path}"
                )
            return None

    def get_batch(self, names):
        if self.next_sample is None:
            return None

        batch_np = np.expand_dims(self.next_sample, axis=0)
        self.next_sample = self._pull_next_sample(required=False)

        self.device_input.copy_(torch.from_numpy(batch_np), non_blocking=False)
        return [int(self.device_input.data_ptr())]

    def read_calibration_cache(self):
        if self.cache_file and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                cache = f.read()
            if cache:
                print(f"   Using calibration cache: {self.cache_file}")
                return cache
        return None

    def write_calibration_cache(self, cache):
        if not self.cache_file:
            return
        cache_dir = os.path.dirname(self.cache_file)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"   Calibration cache saved: {self.cache_file}")

    def close(self):
        self.next_sample = None
        self.sample_iter = None
        self.device_input = None
