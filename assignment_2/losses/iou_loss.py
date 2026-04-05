"""Custom IoU loss 
"""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

        tgt_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        tgt_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        tgt_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        tgt_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        tgt_area = (tgt_x2 - tgt_x1) * (tgt_y2 - tgt_y1)
        union_area = pred_area + tgt_area - inter_area + self.eps   

        iou = inter_area / union_area
        loss = 1 - iou
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss
        
if __name__ == "__main__":
    loss_fn = IoULoss()
    boxes = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
    print("Perfect IoU loss:", loss_fn(boxes, boxes).item())  # ~0.0

    pred   = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
    target = torch.tensor([[0.9, 0.9, 0.1, 0.1]])
    print("No-overlap IoU loss:", loss_fn(pred, target).item())  # ~1.0

    pred_g = torch.tensor([[0.5, 0.5, 0.3, 0.3]], requires_grad=True)
    tgt_g  = torch.tensor([[0.6, 0.6, 0.4, 0.4]])
    l = loss_fn(pred_g, tgt_g)
    l.backward()
    print("Gradient:", pred_g.grad)  # Should be non-None
