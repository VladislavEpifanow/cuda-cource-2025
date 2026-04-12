from collections import OrderedDict
import os

import tensorrt as trt
import torch
import torchvision

from .calibration import VideoEntropyCalibrator
from .config import (
    CALIBRATION_CACHE_PATH,
    DEFAULT_CALIBRATION_FRAMES,
    DEFAULT_VIDEO_INPUT,
    ENGINE_PATH,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    ONNX_PATH,
)

INPUT_SHAPE = (1, 3, INPUT_HEIGHT, INPUT_WIDTH)


class RetinaNetHeadExport(torch.nn.Module):
    """
    Export only backbone + head from RetinaNet.
    This avoids unstable ONNX/TensorRT behavior in torchvision postprocessing.
    """

    def __init__(self):
        super().__init__()
        weights = torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT
        self.model = torchvision.models.detection.retinanet_resnet50_fpn(weights=weights)
        self.model.eval()

        mean = torch.tensor(self.model.transform.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(self.model.transform.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, images):
        images = (images - self.mean) / self.std
        features = self.model.backbone(images)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        features = list(features.values())
        head_outputs = self.model.head(features)
        return head_outputs["cls_logits"], head_outputs["bbox_regression"]


def export_onnx(onnx_path=ONNX_PATH):
    print("1. Loading RetinaNet backbone+head...")
    model = RetinaNetHeadExport().cpu().eval()
    dummy_input = torch.rand(INPUT_SHAPE, dtype=torch.float32)
    onnx_dir = os.path.dirname(onnx_path)
    if onnx_dir:
        os.makedirs(onnx_dir, exist_ok=True)

    print("2. Exporting to ONNX...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["cls_logits", "bbox_regression"],
            dynamo=False,
            verbose=False,
        )
    print(f"ONNX saved: {onnx_path}")


def build_trt_engine(
    onnx_path=ONNX_PATH,
    engine_path=ENGINE_PATH,
    precision="int8",
    calibration_video=DEFAULT_VIDEO_INPUT,
    calibration_frames=DEFAULT_CALIBRATION_FRAMES,
    calibration_cache=CALIBRATION_CACHE_PATH,
):
    print("3. Building TensorRT Engine...")
    engine_dir = os.path.dirname(engine_path)
    if engine_dir:
        os.makedirs(engine_dir, exist_ok=True)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse ONNX")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise SystemExit(1)

    input_tensor = network.get_input(0)
    profile = builder.create_optimization_profile()
    profile.set_shape(input_tensor.name, INPUT_SHAPE, INPUT_SHAPE, INPUT_SHAPE)
    config.add_optimization_profile(profile)

    precision = precision.lower().strip()
    calibrator = None

    if precision == "int8":
        if not builder.platform_has_fast_int8:
            print("   WARNING: Fast INT8 is not supported on this GPU, falling back to FP16.")
            precision = "fp16"
        else:
            config.set_flag(trt.BuilderFlag.INT8)
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            calibrator = VideoEntropyCalibrator(
                video_path=calibration_video,
                num_samples=calibration_frames,
                cache_file=calibration_cache,
            )
            config.int8_calibrator = calibrator
            config.set_calibration_profile(profile)
            print(
                f"   INT8 Enabled (video calibration: {calibration_video}, "
                f"samples: {calibration_frames})"
            )

    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("   FP16 Enabled")
        else:
            print("   WARNING: Fast FP16 is not supported on this GPU, using FP32.")

    if precision not in {"int8", "fp16"}:
        raise ValueError(f"Unsupported precision: {precision}. Use 'int8' or 'fp16'.")

    print("   Compiling engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("   ERROR: Build failed.")
        raise SystemExit(2)

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"   SUCCESS! Engine saved: {engine_path}")
