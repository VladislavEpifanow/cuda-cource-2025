from pathlib import Path

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

COCO_CLASSES = (
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A",
    "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)
