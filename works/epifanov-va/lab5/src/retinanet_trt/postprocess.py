from collections import OrderedDict

import torch
import torchvision
from torchvision.models.detection.image_list import ImageList

from .config import (
    COCO_CLASSES,
    DEFAULT_MAX_DET,
    DEFAULT_NMS_THRESH,
    DEFAULT_TOPK_CANDIDATES,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    WEIGHTS,
)


class RetinaPostprocessor:
    """
    Decode RetinaNet head outputs and apply NMS using torchvision internals.
    """

    def __init__(
        self,
        conf_thresh,
        nms_thresh=DEFAULT_NMS_THRESH,
        max_det=DEFAULT_MAX_DET,
        topk_candidates=DEFAULT_TOPK_CANDIDATES,
        pre_nms_score_thresh=None,
    ):
        self.conf_thresh = conf_thresh
        self.model = torchvision.models.detection.retinanet_resnet50_fpn(weights=WEIGHTS).eval().cuda()
        if pre_nms_score_thresh is None:
            pre_nms_score_thresh = conf_thresh
        self.model.score_thresh = float(max(0.01, min(pre_nms_score_thresh, 0.99)))
        self.model.nms_thresh = nms_thresh
        self.model.detections_per_img = max_det
        self.model.topk_candidates = int(max(50, topk_candidates))

        self.mean = torch.tensor(
            self.model.transform.image_mean, device="cuda", dtype=torch.float32
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            self.model.transform.image_std, device="cuda", dtype=torch.float32
        ).view(1, 3, 1, 1)

        # Precompute anchor split sizes for the fixed 640x640 input.
        with torch.no_grad():
            dummy = torch.zeros((1, 3, INPUT_HEIGHT, INPUT_WIDTH), device="cuda", dtype=torch.float32)
            norm = (dummy - self.mean) / self.std
            features = self.model.backbone(norm)
            if isinstance(features, torch.Tensor):
                features = OrderedDict([("0", features)])
            features = list(features.values())

            head = self.model.head(features)
            hw_per_level = [f.shape[2] * f.shape[3] for f in features]
            total_hw = sum(hw_per_level)
            anchors_per_location = head["cls_logits"].shape[1] // total_hw
            self.num_anchors_per_level = [hw * anchors_per_location for hw in hw_per_level]

            image_list = ImageList(dummy, [(INPUT_HEIGHT, INPUT_WIDTH)])
            anchors = self.model.anchor_generator(image_list, features)[0]
            self.split_anchors = list(anchors.split(self.num_anchors_per_level))

    @torch.no_grad()
    def decode(self, cls_logits, bbox_regression, orig_h, orig_w):
        cls_logits = cls_logits.float()
        bbox_regression = bbox_regression.float()

        split_head = {
            "cls_logits": list(cls_logits.split(self.num_anchors_per_level, dim=1)),
            "bbox_regression": list(bbox_regression.split(self.num_anchors_per_level, dim=1)),
        }

        detections = self.model.postprocess_detections(
            split_head, [self.split_anchors], [(INPUT_HEIGHT, INPUT_WIDTH)]
        )[0]

        if detections["scores"].numel() == 0:
            return [], [], []

        keep = detections["scores"] >= self.conf_thresh
        boxes = detections["boxes"][keep]
        scores = detections["scores"][keep]
        labels = detections["labels"][keep]

        if scores.numel() == 0:
            return [], [], []

        boxes = boxes.clone()
        boxes[:, [0, 2]] *= float(orig_w) / float(INPUT_WIDTH)
        boxes[:, [1, 3]] *= float(orig_h) / float(INPUT_HEIGHT)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, max(orig_w - 1, 0))
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, max(orig_h - 1, 0))

        return boxes.int().cpu().numpy(), scores.cpu().numpy(), labels.int().cpu().numpy()


def get_class_name(label_idx):
    if label_idx < 0 or label_idx >= len(COCO_CLASSES):
        return None
    class_name = COCO_CLASSES[label_idx]
    if class_name in ("N/A", "__background__"):
        return None
    return class_name
