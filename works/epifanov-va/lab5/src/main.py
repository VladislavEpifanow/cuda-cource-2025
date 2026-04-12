import argparse
import os
import time

import cv2

from retinanet_trt.config import (
    DEFAULT_CONF_THRESH,
    DEFAULT_TOPK_CANDIDATES,
    DEFAULT_VIDEO_INPUT,
    DEFAULT_VIDEO_OUTPUT,
    ENGINE_PATH,
)
from retinanet_trt.postprocess import RetinaPostprocessor
from retinanet_trt.trt_runtime import TrtModel
from retinanet_trt.video_utils import draw_detections, preprocess_frame


def parse_args():
    parser = argparse.ArgumentParser(description="TensorRT RetinaNet video object detection")
    parser.add_argument("--input", default=DEFAULT_VIDEO_INPUT, help="Input video path")
    parser.add_argument("--output", default=DEFAULT_VIDEO_OUTPUT, help="Output video path")
    parser.add_argument("--engine", default=ENGINE_PATH, help="TensorRT engine path")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESH, help="Confidence threshold")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Process only first N frames (0 means full video)",
    )
    return parser.parse_args()


def run_video_detection(
    engine_path,
    video_input,
    video_output,
    conf_thresh,
    max_frames=0,
):
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"Engine file not found: {engine_path}. Run build_engine.py first.")
    if not os.path.exists(video_input):
        raise FileNotFoundError(f"Input video not found: {video_input}")

    print("Loading TensorRT engine...")
    trt_model = TrtModel(engine_path)
    postprocessor = RetinaPostprocessor(
        conf_thresh=conf_thresh,
        topk_candidates=DEFAULT_TOPK_CANDIDATES,
        pre_nms_score_thresh=conf_thresh,
    )

    expected_outputs = {"cls_logits", "bbox_regression"}
    if not expected_outputs.issubset(set(trt_model.output_names)):
        raise RuntimeError(
            f"Unexpected engine outputs: {trt_model.output_names}. "
            f"Expected at least: {sorted(expected_outputs)}"
        )

    cap = cv2.VideoCapture(video_input)
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps <= 0:
        input_fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir = os.path.dirname(video_output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*"mp4v"), input_fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video for writing: {video_output}")

    print(f"Processing {video_input} ({width}x{height})...")
    start = time.time()
    frame_count = 0
    total_detections = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        inp, orig_h, orig_w = preprocess_frame(frame)
        outputs = trt_model.infer(inp)
        boxes, scores, labels = postprocessor.decode(
            outputs["cls_logits"], outputs["bbox_regression"], orig_h, orig_w
        )

        total_detections += draw_detections(frame, boxes, scores, labels)
        out.write(frame)

        if frame_count % 30 == 0:
            elapsed = time.time() - start
            avg_fps = frame_count / elapsed if elapsed > 0 else 0.0
            print(f"Frame {frame_count}, Avg FPS: {avg_fps:.2f}, Total detections: {total_detections}")

        if max_frames > 0 and frame_count >= max_frames:
            break

    cap.release()
    out.release()

    total_time = time.time() - start
    avg_fps = frame_count / total_time if total_time > 0 else 0.0
    video_duration = frame_count / input_fps if input_fps > 0 else 0.0
    realtime_factor = avg_fps / input_fps if input_fps > 0 else 0.0

    print(f"Done! Saved to {video_output}")
    print(f"Total frames: {frame_count}, Time: {total_time:.2f}s, Avg FPS: {avg_fps:.2f}")
    print(f"Detected objects: {total_detections}")
    print(f"Input video FPS: {input_fps:.2f}")
    print(f"Video duration (processed): {video_duration:.2f}s")
    print(f"Speed vs real-time: {realtime_factor:.2f}x")
    if realtime_factor >= 1.0:
        print("Result: faster than real-time")
    else:
        print("Result: slower than real-time")


def main():
    args = parse_args()
    run_video_detection(
        engine_path=args.engine,
        video_input=args.input,
        video_output=args.output,
        conf_thresh=args.conf,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
