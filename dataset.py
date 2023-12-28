import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import fiftyone as fo
import fiftyone.zoo as foz
from hparam import hparam as hp

class COCODataset(Dataset):
    def get_index_dict(self):
        index_dict = {}
        for index,sample in enumerate(self.dataset):
            index_dict[index] = sample.id
        return index_dict
    def __init__(self,transform=None):
        self.dataset = fo.zoo.load_zoo_dataset(
            "coco-2017",
            split="train",
            label_types=["detections"],
            classes=["person"],
            max_samples=hp["max_training_samples"],
        )
        self.transform = transform
        self.idx_to_id = self.get_index_dict()

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        # get image and label from dataset
        id = self.idx_to_id[idx]
        sample = self.dataset[id]
        image_path = sample["filepath"]
        img = Image.open(image_path)
        # img_array = np.array(img)
        labels = sample["ground_truth"]["detections"]
        # convert image

        # convert labels ratios to absolute box coordinates
        boxes = []
        count = 0
        for label in labels:
            if count > 1:
                break
            count += 1
            box = label["bounding_box"]
            x,y = box[0]+box[2]/2, box[1]+box[3]/2
            w,h = box[2], box[3]
            boxes.append([1,x,y,w,h])
        boxes = torch.tensor(boxes)
        if self.transform:
            img, boxes = self.transform(img,boxes)
        label_matrix = torch.zeros((hp["S"],hp["S"],hp["C"]+5*hp["B"]))
        for box in boxes:
            class_label,x,y,width,height = box.tolist()
            class_label = int(class_label)
            i,j = int(hp["S"]*x), int(hp["S"]*y)
            x_cell,y_cell = hp["S"]*x-i, hp["S"]*y-j
            width_cell,height_cell = width*hp["S"], height*hp["S"]
            if label_matrix[i,j,20] == 0:
                label_matrix[i,j,20] = 1
                box_coordinates = torch.tensor([x_cell,y_cell,width_cell,height_cell])
                label_matrix[i,j,21:25] = box_coordinates
                label_matrix[i,j,class_label] = 1
        # if first dimenstion == 1 (grey scale image) then copy it into 3 channels
        if img.shape[0] == 1:
            img = img.repeat((3,1,1))
        return img, label_matrix