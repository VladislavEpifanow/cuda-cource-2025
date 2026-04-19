from collections import OrderedDict
import os

import onnx
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
        cls_logits = head_outputs["cls_logits"]
        bbox_regression = head_outputs["bbox_regression"]

        # Keep ONNX export contract stable across torchvision/exporter versions.
        if isinstance(cls_logits, (list, tuple)):
            cls_logits = torch.cat(cls_logits, dim=1)
        if isinstance(bbox_regression, (list, tuple)):
            bbox_regression = torch.cat(bbox_regression, dim=1)

        return cls_logits, bbox_regression


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
    print("3. Validating ONNX...")
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        # Remove invalid ONNX file to avoid reusing a corrupted artifact.
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        raise RuntimeError(f"ONNX validation failed for '{onnx_path}': {e}") from e
    print("   ONNX validation passed.")
    print(f"ONNX saved: {onnx_path}")


def build_trt_engine(
    onnx_path=ONNX_PATH,
    engine_path=ENGINE_PATH,
    precision="int8",
    calibration_video=DEFAULT_VIDEO_INPUT,
    calibration_frames=DEFAULT_CALIBRATION_FRAMES,
    calibration_cache=CALIBRATION_CACHE_PATH,
):
    print("4. Building TensorRT Engine...")
    engine_dir = os.path.dirname(engine_path)
    if engine_dir:
        os.makedirs(engine_dir, exist_ok=True)

    requested_precision = precision.lower().strip()
    if requested_precision not in {"int8", "fp16", "fp32"}:
        raise ValueError(f"Unsupported precision: {precision}. Use 'int8', 'fp16' or 'fp32'.")

    logger = None
    builder = None
    network = None
    parser = None
    config = None
    profile = None
    calibrator = None
    serialized_engine = None
    try:
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        effective_precision = requested_precision
        if requested_precision == "int8":
            if not builder.platform_has_fast_int8:
                if builder.platform_has_fast_fp16:
                    print("   WARNING: Fast INT8 is not supported on this GPU, falling back to FP16.")
                    effective_precision = "fp16"
                else:
                    print("   WARNING: Fast INT8/FP16 are not supported on this GPU, using FP32.")
                    effective_precision = "fp32"
        elif requested_precision == "fp16":
            if not builder.platform_has_fast_fp16:
                print("   WARNING: Fast FP16 is not supported on this GPU, using FP32.")
                effective_precision = "fp32"

        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        config = builder.create_builder_config()

        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                print("ERROR: Failed to parse ONNX")
                errors = []
                for i in range(parser.num_errors):
                    err = str(parser.get_error(i))
                    errors.append(err)
                    print(err)
                raise RuntimeError(
                    "Failed to parse ONNX model with TensorRT parser."
                    + (f" Parser errors: {' | '.join(errors)}" if errors else "")
                )

        input_tensor = network.get_input(0)
        profile = builder.create_optimization_profile()
        profile.set_shape(input_tensor.name, INPUT_SHAPE, INPUT_SHAPE, INPUT_SHAPE)
        config.add_optimization_profile(profile)

        if effective_precision == "int8":
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
        elif effective_precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
            print("   FP16 Enabled")
        else:
            print("   FP32 Enabled")

        print("   Compiling engine...")
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("   ERROR: Build failed.")
            raise RuntimeError("TensorRT engine build failed: build_serialized_network returned None.")

        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
        print(f"   SUCCESS! Engine saved: {engine_path}")
    finally:
        if config is not None:
            try:
                config.int8_calibrator = None
            except Exception:
                pass
        if calibrator is not None:
            calibrator.close()
        del serialized_engine
        del profile
        del config
        del parser
        del network
        del builder
        del logger
        del calibrator
