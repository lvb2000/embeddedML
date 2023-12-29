import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from model import YOLOv1
from dataset import COCODataset
from hparam import hparam as hp
from utils import (
    non_max_suppression,
    mean_average_precision,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    load_checkpoint,
)
from train import Compose

seed = 123
torch.manual_seed(seed)

transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

def test():
    model = YOLOv1().to(hp["device"])
    optimizer = optim.Adam(
        model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]
    )
    load_checkpoint(torch.load(hp["load_model_file"],map_location=torch.device('cpu')), model, optimizer)

    test_dataset = COCODataset(transform=transform)
    test_dataset.load_dataset("test")
    print(f"Test samples: {len(test_dataset)}")
    test_loader = DataLoader(dataset=test_dataset, batch_size=hp["batch_size"], num_workers=hp["num_worker"], pin_memory=hp["Pin_memory"], shuffle=True, drop_last=False)


    for x,y in test_loader:
        x = x.to(hp["device"])
        for idx in range(8):
            bboxes = cellboxes_to_boxes(model(x))
            real_boxes = cellboxes_to_boxes(y.flatten(start_dim=1))
            bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
            plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes,real_boxes[idx])
        break

    pred_boxes, target_boxes = get_bboxes(
        test_loader, model, iou_threshold=0.5, threshold=0.4, device=hp["device"]
    )
    mean_avg_prec = mean_average_precision(
        pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
    )
    print(f"Train mAP: {mean_avg_prec}")