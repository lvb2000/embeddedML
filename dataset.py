import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
#from skimage import io


from pycocotools.coco import COCO

class COCODataset(Dataset):
    def __int__(self):
        print("COCO Dataset init")
        super(COCODataset, self).__init__()
        dataDir = '..'
        dataType = 'val2017'
        annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
        self.coco = COCO(annFile)
        category_ids = self.coco.getCatIds(catNms=['person'])
        self.image_ids = self.coco.getImgIds(catIds=category_ids)

    def __len__(self):
        return len(self.image_ids)
    def __getitem__(self, idx):
        img = self.coco.loadImgs(self.image_ids[idx])[0]
        I = io.imread(img['coco_url'])
        plt.axis('off')
        plt.imshow(I)
        plt.show()

def test():
    print("Testing dataset.py")
    dataset = COCODataset()
    print(len(dataset))
    dataset[0]

test()