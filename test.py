import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from model import YOLOv1
from dataset import COCOTestSet
from hparam import hparam as hp
from utils import (
    non_max_suppression,
    cellboxes_to_boxes,
    plot_image,
    load_checkpoint,
)
from train import Compose

seed = 123
torch.manual_seed(seed)

transform = Compose([transforms.Resize((hp['image_size'], hp['image_size'])), transforms.ToTensor()])

def test():
    model = YOLOv1().to(hp["device"])
    optimizer = optim.Adam(
        model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]
    )
    load_checkpoint(torch.load(hp["load_model_file"]), model, optimizer)

    test_dataset = COCOTestSet(transform=transform)
    test_dataset.load_dataset()
    print(f"Test samples: {len(test_dataset)}")
    test_loader = DataLoader(dataset=test_dataset, batch_size=hp["batch_size"], num_workers=hp["num_worker"], pin_memory=hp["Pin_memory"], shuffle=True, drop_last=False)


    for x in test_loader:
        x = x.to(hp["device"])
        for idx in range(16):
            bboxes = cellboxes_to_boxes(model(x),S=hp["S"] )
            bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
            plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)
        break

if __name__ == "__main__":
    test()