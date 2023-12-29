import torch
import torch.nn as nn
from utils import intersection_over_union
from hparam import hparam as hp

class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()
        # TODO: again using torch functions, is this possible on the MAXIM board?
        self.mse = nn.MSELoss(reduction="sum")
        self.S = hp['S']
        self.B = hp['B']
        self.lambda_noobj = hp['lambda_noobj']
        self.lambda_coord = hp['lambda_coord']

    def forward(self,predictions,target):
        # predictions are shaped (batch_size, S*S(C+B*5))
        # this is why we always need the ... to apply the functions to only the last dimension
        predictions = predictions.reshape(-1,self.S,self.S,self.B*5)

        # Calculate IoU for the two bounding boxes
        # Each prediction has shape: [0-19 classes, 20 confidence, 21-24 box1, 25 confidence, 26-29 box2]
        iou_b1 = intersection_over_union(predictions[...,1:5],target[...,1:5])
        iou_b2 = intersection_over_union(predictions[...,6:10],target[...,1:5])
        # ious is a tensor of shape (2, batch_size, S, S)
        ious = torch.cat([iou_b1.unsqueeze(0),iou_b2.unsqueeze(0)],dim=0)
        # bestbox is the index of ious with the highest IoU
        # with dim=0, we get the max IoU for each batch because we look only at the first dimension which has size of number of boxes
        iou_maxes, best_box = torch.max(ious,dim=0)
        # unsqueeze(3) adds a dimension to the end of the tensor which is the class probability label
        exists_box = target[...,0].unsqueeze(3) # identity function of object i

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # check box center coordinates
        box_predictions = exists_box * (
            (
                best_box * predictions[...,6:10]
                + (1-best_box) * predictions[...,1:5]
            )
        )
        box_targets = exists_box * target[...,1:5]

        # check box width and height
        # we abs the value in the sqrt but later put it back to negative if it was negative with the sign function
        box_predictions[...,2:4] = torch.sign(box_predictions[...,2:4]) * torch.sqrt(
            torch.abs(box_predictions[...,2:4] + 1e-6)
        )
        box_targets[...,2:4] = torch.sqrt(box_targets[...,2:4])

        # this is flatten from (batch_size, S, S, 4) to (batch_size*S*S,4) because this is how the mse loss works
        box_loss = self.mse(
            torch.flatten(box_predictions,end_dim=-2),
            torch.flatten(box_targets,end_dim=-2),
        )

        # ======================== #
        #   FOR OBJECT LOSS        #
        # ======================== #

        pred_box = (
            best_box * predictions[...,5:6] + (1-best_box) * predictions[...,0:1]
        )
        # (N*S*S,1)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[...,0:1]),
        )

        # ======================== #
        #   FOR NO OBJECT LOSS     #
        # ======================== #

        # (N,S,S,1) -> (N, S*S)
        no_object_loss = self.mse(
            torch.flatten((1-exists_box) * predictions[...,0:1],start_dim=1),
            torch.flatten((1-exists_box) * target[...,0:1],start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1-exists_box) * predictions[...,5:6],start_dim=1),
            torch.flatten((1-exists_box) * target[...,0:1],start_dim=1),
        )
        # ======================== #
        #   FINAL LOSS             #
        # ======================== #

        loss = (
            self.lambda_coord * box_loss # first two terms
            + object_loss # third term
            + self.lambda_noobj * no_object_loss # fourth term
        )

        return loss