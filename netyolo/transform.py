import torch
from torch import nn

import torchvision
from torchvision.models.detection.transform import resize_boxes
import torchvision.transforms.functional as F


def resize_image(image, size):
    return nn.functional.interpolate(image[None], size, mode="bilinear")[0]


def clip_boxes_to_image(boxes, size):
    dim = boxes.dim() 
    boxes_x = boxes[..., 0::2]  # x1, x2
    boxes_y = boxes[..., 1::2]  # y1, y2
    height, width = size

    boxes_x = boxes_x.clamp(min=0, max=width)   
    boxes_y = boxes_y.clamp(min=0, max=height)  
    
    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)



class YOLOTransform(nn.Module):
    def __init__(self, input_size, conf_thresh, nms_thresh, max_detections, image_mean = [0.485, 0.456, 0.406], image_std = [0.229, 0.224, 0.225]):
        super(YOLOTransform, self).__init__()
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.max_detections = max_detections

        self.image_mean = image_mean
        self.image_std = image_std

    def forward(self, images, targets):
        image_list, target_list = [], []

        for i in range(len(images)):
            image = images[i]
            resized_image = F.normalize(resize_image(image, self.input_size), self.image_mean, self.image_std)

            if self.training:
                target = targets[i]
                boxes = target["boxes"]
                labels = target["labels"]

                h, w = image.shape[1:]
                out_boxes = torch.zeros(boxes.shape, dtype=torch.float32, device=boxes.device)

                out_boxes[:, 0] = torch.div(boxes[:, 0] + boxes[:, 2], 2 * w)
                out_boxes[:, 1] = torch.div(boxes[:, 1] + boxes[:, 3], 2 * h)
                out_boxes[:, 2] = torch.div(boxes[:, 2] - boxes[:, 0], w)
                out_boxes[:, 3] = torch.div(boxes[:, 3] - boxes[:, 1], h)
                
                out_labels = torch.transpose(labels.unsqueeze(0) - 1, 0, 1).float()

                boxes_with_cls = torch.cat([out_boxes, out_labels], 1)
                target_list.append(boxes_with_cls)

            image_list.append(resized_image)

        out_images = torch.stack(image_list)
        
        if self.training:
            # need padding boxes, while zero will be ignored in loss
            max_targets_per_image = max([tar.size(0) for tar in target_list])
            padded_target = []
            for tar in target_list:
                padding = torch.zeros(max_targets_per_image - tar.size(0), 5, device=tar.device)
                padded_target.append(torch.cat([tar, padding], dim=0))
            out_targets = torch.stack(padded_target) 
        else:
            out_targets = None
        
        return out_images, out_targets
    
    def postprocess(self, predictions, image_sizes):
        preds = torch.cat(predictions, 1)
        _, max_ids = torch.max(preds[:, :, 5:], dim=2)

        detections = []

        for i in range(preds.shape[0]):
            conf_mask = preds[i, :, 4] > self.conf_thresh
            conf_ids = torch.nonzero(conf_mask).flatten()

            x = preds[i, conf_ids, 0]
            y = preds[i, conf_ids, 1]
            w = preds[i, conf_ids, 2]
            h = preds[i, conf_ids, 3]

            xmin = x - w * 0.5
            ymin = y - h * 0.5
            xmax = xmin + w
            ymax = ymin + h

            relative_boxes = torch.stack([xmin, ymin, xmax, ymax], 1)
            boxes = resize_boxes(relative_boxes, self.input_size, image_sizes[i])
            boxes = clip_boxes_to_image(boxes ,image_sizes[i])
            
            labels = max_ids[i, conf_ids].long() + 1
            scores = preds[i, conf_ids, 4]

            keep = torchvision.ops.nms(boxes, scores, self.nms_thresh)[:self.max_detections]

            detection = {
                "boxes": boxes[keep],
                "labels": labels[keep],
                "scores": scores[keep]
            }
            
            detections.append(detection)

        return detections