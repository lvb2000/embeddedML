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

def download_val():
    fo.zoo.load_zoo_dataset(
        "coco-2017",
        split="validation",
    )

if __name__ == "__main__":
    download_val()