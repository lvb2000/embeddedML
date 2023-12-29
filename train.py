import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import YOLOv1
from dataset import COCODataset
from hparam import hparam as hp
from utils import (
    intersection_over_union,
    non_max_suppression,
    mean_average_precision,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss
import pandas as pd

seed = 123
torch.manual_seed(seed)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes

transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for (x, y) in loop:
        x, y = x.to(hp["device"]), y.to(hp["device"])
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

def main():
    #------------------- Init model -------------------#
    model = YOLOv1().to(hp["device"])
    optimizer = optim.Adam(
        model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]
    )
    loss_fn = YoloLoss()

    #------------------- Init training Dataset -------------------#
    train_dataset = COCODataset(transform=transform)
    train_dataset.load_dataset()
    print(f"Training samples: {len(train_dataset)}")
    train_loader = DataLoader(dataset=train_dataset, batch_size=hp["batch_size"], num_workers=hp["num_worker"], pin_memory=hp["Pin_memory"], shuffle=True, drop_last=False)

    #------------------- Init validation Dataset -------------------#
    val_dataset = COCODataset(transform=transform)
    val_dataset.load_dataset("validation")
    print(f"Validation samples: {len(val_dataset)}")
    val_loader = DataLoader(dataset=val_dataset, batch_size=hp["batch_size"], num_workers=hp["num_worker"],
                              pin_memory=hp["Pin_memory"], shuffle=True, drop_last=False)

    best_map = 0

    # init df to store training and validation mAP
    df = pd.DataFrame(columns=['train_mAP', 'val_mAP'])

    for epoch in range(hp["num_epochs"]):

        #------------------- Training Mean Average Precision -------------------#
        train_pred_boxes, train_target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4,device=hp["device"]
        )
        train_mean_avg_prec = mean_average_precision(
            train_pred_boxes, train_target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {train_mean_avg_prec}")

        #------------------- Validation Mean Average Precision -------------------#
        val_pred_boxes, val_target_boxes = get_bboxes(
            val_loader, model, iou_threshold=0.5, threshold=0.4, device=hp["device"]
        )
        val_mean_avg_prec = mean_average_precision(
            val_pred_boxes, val_target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Validation mAP: {val_mean_avg_prec}")

        #------------------- Store mAP -------------------#
        df.loc[len(df)] = [train_mean_avg_prec,val_mean_avg_prec]
        df.to_csv('mAP.csv', index=False)

        #------------------- Checkpointing -------------------#
        if train_mean_avg_prec >= best_map:
            best_map = train_mean_avg_prec
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=hp["load_model_file"])

        #------------------- Training -------------------#
        train_fn(train_loader, model, optimizer, loss_fn)

if __name__ == "__main__":
    main()
