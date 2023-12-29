import fiftyone as fo

def download():
    fo.zoo.load_zoo_dataset(
        "coco-2017",
        split="train",
    )

if __name__ == "__main__":
    download()