import argparse

from retinanet_trt.config import (
    CALIBRATION_CACHE_PATH,
    DEFAULT_CALIBRATION_FRAMES,
    DEFAULT_VIDEO_INPUT,
    ENGINE_PATH,
    ONNX_PATH,
)
from retinanet_trt.engine_builder import build_trt_engine, export_onnx


def parse_args():
    parser = argparse.ArgumentParser(description="Build RetinaNet TensorRT engine")
    parser.add_argument("--onnx", default=ONNX_PATH, help="Path to ONNX file")
    parser.add_argument("--engine", default=ENGINE_PATH, help="Path to output TensorRT engine")
    parser.add_argument(
        "--precision",
        default="int8",
        choices=["int8", "fp16", "fp32"],
        help="Engine precision: int8 | fp16 | fp32 (default: int8)",
    )
    parser.add_argument(
        "--calib-video",
        default=DEFAULT_VIDEO_INPUT,
        help="Calibration video path for INT8 mode",
    )
    parser.add_argument(
        "--calib-frames",
        type=int,
        default=DEFAULT_CALIBRATION_FRAMES,
        help="Number of sampled frames for INT8 calibration",
    )
    parser.add_argument(
        "--calib-cache",
        default=CALIBRATION_CACHE_PATH,
        help="Path to INT8 calibration cache",
    )
    parser.add_argument(
        "--skip-onnx-export",
        action="store_true",
        help="Use existing ONNX file without re-export",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.skip_onnx_export:
        export_onnx(onnx_path=args.onnx)

    build_trt_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        precision=args.precision,
        calibration_video=args.calib_video,
        calibration_frames=args.calib_frames,
        calibration_cache=args.calib_cache,
    )


if __name__ == "__main__":
    main()
