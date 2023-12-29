import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import fiftyone as fo
import fiftyone.zoo as foz
from hparam import hparam as hp
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class COCODataset(Dataset):
    def print_sample(self,img,boxes):
        im = np.array(img)
        if len(im.shape) == 2:
            im = np.repeat(im[:,:,np.newaxis],3,axis=2)
        height, width, _ = im.shape

        # Create figure and axes
        fig, ax = plt.subplots(1)
        # Display the image
        ax.imshow(im)

        # box[0] is x midpoint, box[2] is width
        # box[1] is y midpoint, box[3] is height

        # Create a Rectangle potch
        for box in boxes:
            box = box[1:]
            assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
            upper_left_x = box[0] - box[2] / 2
            upper_left_y = box[1] - box[3] / 2
            rect = patches.Rectangle(
                (upper_left_x * width, upper_left_y * height),
                box[2] * width,
                box[3] * height,
                linewidth=1,
                edgecolor="g",
                facecolor="none",
            )
            # Add the patch to the Axes
            ax.add_patch(rect)
        plt.show()

    def get_index_dict(self):
        index_dict = {}
        for index,sample in enumerate(self.dataset):
            index_dict[index] = sample.id
        return index_dict
    def __init__(self,transform=None):

        self.transform = transform

    def load_dataset(self,kind="train"):
        self.dataset = fo.zoo.load_zoo_dataset(
            "coco-2017",
            split=kind,
            label_types=["detections"],
            classes=["person"],
            max_samples=hp["max_training_samples"],
        )
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
        #count = 0
        for label in labels:
            if label["label"] != "person":
                continue
            box = label["bounding_box"]
            if box[2] <= 0.25 or box[3] == 0.25:
                continue
            x,y = box[0]+box[2]/2, box[1]+box[3]/2
            w,h = box[2], box[3]
            boxes.append([x,y,w,h])
        #self.print_sample(img, boxes)
        boxes = torch.tensor(boxes)
        if self.transform:
            img, boxes = self.transform(img,boxes)
        label_matrix = torch.zeros((hp["S"],hp["S"],5*hp["B"]))
        for box in boxes:
            x,y,width,height = box.tolist()
            i,j = int(hp["S"]*y),int(hp["S"]*x)
            x_cell,y_cell = hp["S"]*x-j, hp["S"]*y-i
            width_cell,height_cell = width*hp["S"], height*hp["S"]
            if label_matrix[i,j,0] == 0:
                label_matrix[i,j,0] = 1
                box_coordinates = torch.tensor([x_cell,y_cell,width_cell,height_cell])
                label_matrix[i,j,1:5] = box_coordinates
        # if first dimenstion == 1 (grey scale image) then copy it into 3 channels
        if img.shape[0] == 1:
            img = img.repeat((3,1,1))
        return img, label_matrix

class COCOTestSet(Dataset):
    def get_index_dict(self):
        index_dict = {}
        for index,sample in enumerate(self.dataset):
            index_dict[index] = sample.id
        return index_dict
    def __init__(self,transform=None):

        self.transform = transform

    def load_dataset(self):
        self.dataset = fo.zoo.load_zoo_dataset(
            "coco-2017",
            split="test",
            label_types=["detections"],
            classes=["person"],
            max_samples=hp["max_training_samples"],
        )
        self.idx_to_id = self.get_index_dict()
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        # get image and label from dataset
        id = self.idx_to_id[idx]
        sample = self.dataset[id]
        image_path = sample["filepath"]
        img = Image.open(image_path)

        # convert labels ratios to absolute box coordinates
        boxes = []
        if self.transform:
            img, boxes = self.transform(img,boxes)

        # if first dimenstion == 1 (grey scale image) then copy it into 3 channels
        if img.shape[0] == 1:
            img = img.repeat((3,1,1))
        return img