import fiftyone as fo

def download_train():
    fo.zoo.load_zoo_dataset(
        "coco-2017",
        split="train",
    )

def download_test():
    fo.zoo.load_zoo_dataset(
        "coco-2017",
        split="test",
    )

if __name__ == "__main__":
    download_test()