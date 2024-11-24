from torch import nn

from .transform import YOLOTransform
from .yolo_util import Yolov3Body, Yolov7Body
from .losses import YOLOLoss


class YOLOv3(nn.Module):
    def __init__(self,
            num_classes=2,
            input_size=(416, 416),
            conf_thresh=0.05,
            nms_thresh=0.1,
            max_detections=100,
            anchors=[[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]]):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.max_detections = max_detections
        self.anchors = anchors
        #  transform
        self.transform = YOLOTransform(self.input_size, self.conf_thresh, self.nms_thresh, self.max_detections)
        #  model body
        self.model = Yolov3Body(anchors_mask=self.anchors, num_classes=self.num_classes, pretrained=True)
        #  losses
        self.losses = nn.ModuleList([YOLOLoss(self.num_classes, self.input_size, self.anchors[i]) for i in range(3)])

    def forward(self, images, targets=None):
        or_images = images.copy()
        images, gt = self.transform(images, targets)
        out0, out1, out2 = self.model(images)
        outputs = [out0, out1, out2]
        pred = [loss(outputs[i], gt) for i, loss in enumerate(self.losses)]
        
        if self.training:
            losses = [sum(loss) for loss in zip(*pred)]

            loss_dict = {
                "loss_box_x": losses[0],
                "loss_box_y": losses[1],
                "loss_box_width": losses[2],
                "loss_box_height": losses[3],
                "loss_objectness": losses[4],
                "loss_classifier": losses[5]
            }

            return loss_dict
        else:
            img_sizes = [img.shape[1:] for img in or_images]
            return self.transform.postprocess(pred, img_sizes)
        

class YOLOv7(nn.Module):
    def __init__(self,
            num_classes=2,
            input_size=(640, 640),
            conf_thresh=0.05,
            nms_thresh=0.1,
            max_detections=100,
            anchors=[[[142, 110], [192, 243], [459, 401]],
                    [[36, 75], [76, 55], [72, 146]],
                    [[12, 16], [19, 36], [40, 28]]]):
        super(YOLOv7, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.max_detections = max_detections
        self.anchors = anchors
        #  transform
        self.transform = YOLOTransform(self.input_size, self.conf_thresh, self.nms_thresh, self.max_detections)
        #  model body
        self.model = Yolov7Body(anchors_mask=self.anchors, num_classes=self.num_classes, pretrained=True)
        #  losses
        self.losses = nn.ModuleList([YOLOLoss(self.num_classes, self.input_size, self.anchors[i]) for i in range(3)])

    def forward(self, images, targets=None):
        or_images = images.copy()
        images, gt = self.transform(images, targets)
        out0, out1, out2 = self.model(images)
        outputs = [out0, out1, out2]
        pred = [loss(outputs[i], gt) for i, loss in enumerate(self.losses)]
        
        if self.training:
            losses = [sum(loss) for loss in zip(*pred)]

            loss_dict = {
                "loss_box_x": losses[0],
                "loss_box_y": losses[1],
                "loss_box_width": losses[2],
                "loss_box_height": losses[3],
                "loss_objectness": losses[4],
                "loss_classifier": losses[5]
            }

            return loss_dict
        else:
            img_sizes = [img.shape[1:] for img in or_images]
            return self.transform.postprocess(pred, img_sizes)