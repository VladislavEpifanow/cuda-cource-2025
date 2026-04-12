import os

import cv2
import numpy as np
import tensorrt as trt
import torch

from .config import INPUT_HEIGHT, INPUT_WIDTH


def _preprocess_for_calibration(frame):
    resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    return np.ascontiguousarray(chw)


def _sample_video_frames(video_path, num_samples):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open calibration video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    samples = []

    if frame_count > 0:
        sample_count = min(num_samples, frame_count)
        indices = np.linspace(0, frame_count - 1, num=sample_count, dtype=np.int32)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                samples.append(_preprocess_for_calibration(frame))
    else:
        # Fallback for streams/codecs with unknown frame count.
        while len(samples) < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            samples.append(_preprocess_for_calibration(frame))

    cap.release()
    if not samples:
        raise RuntimeError(f"Failed to read any frames from calibration video: {video_path}")
    return samples


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

        self.samples = _sample_video_frames(video_path=self.video_path, num_samples=self.num_samples)
        self.cursor = 0

        self.device_input = torch.empty(
            (self.batch_size, 3, INPUT_HEIGHT, INPUT_WIDTH), dtype=torch.float32, device="cuda"
        )

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.cursor >= len(self.samples):
            return None

        batch_np = np.expand_dims(self.samples[self.cursor], axis=0)
        self.cursor += 1

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
