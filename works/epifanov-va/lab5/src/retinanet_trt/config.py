from pathlib import Path

import torchvision

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENGINE_PATH = str(PROJECT_ROOT / "artifacts" / "retinanet.engine")
ONNX_PATH = str(PROJECT_ROOT / "artifacts" / "retinanet.onnx")
CALIBRATION_CACHE_PATH = str(PROJECT_ROOT / "artifacts" / "retinanet_int8.cache")

DEFAULT_VIDEO_INPUT = str(PROJECT_ROOT / "data" / "videos" / "test.mp4")
DEFAULT_VIDEO_OUTPUT = str(PROJECT_ROOT / "data" / "videos" / "output.mp4")

INPUT_WIDTH = 640
INPUT_HEIGHT = 640

DEFAULT_CONF_THRESH = 0.5
DEFAULT_NMS_THRESH = 0.5
DEFAULT_MAX_DET = 300
DEFAULT_TOPK_CANDIDATES = 300
DEFAULT_CALIBRATION_FRAMES = 128

WEIGHTS = torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT
COCO_CLASSES = WEIGHTS.meta["categories"]
