from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import os
import csv
import PIL
from collections import namedtuple
from utils import kaggle_setup
from zipfile import ZipFile

FROM_KAGGLE = False
CSV = namedtuple("CSV", ["header", "index", "data"])


def crop_celeba(img):
    return transforms.functional.crop(img, top=40, left=15, height=148, width=148)


class CelebA(datasets.VisionDataset):
    """
    Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>
    """
    base_folder = "celeba-kaggle"

    def __init__(
            self,
            root,
            split,
            download=False,
            transform=transforms.ToTensor()
    ):
        super().__init__(root, transform=transform)
        self.split = split
        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        if download:
            kaggle_setup()
            download_folder = os.path.join(self.root, self.base_folder)
            kaggle_ref = "jessicali9530/celeba-dataset"
            os.system(f"kaggle datasets download -p {download_folder} {kaggle_ref}")
            print("Decompressing the downloaded file...")
            file = os.path.join(download_folder, kaggle_ref.split("/")[-1] + ".zip")
            # os.system(f"unzip -q -d {download_folder} {file}")  # too slow
            with ZipFile(file, "r") as zf:
                zf.extractall(path=download_folder)

        split_ = split_map[split.lower()]
        splits = self._load_csv("list_eval_partition.csv", header=0)
        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()
        if mask == slice(None):  # if split == "all"
            self.filename = splits.index
        else:
            self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]
        self.download = download

    def _load_csv(
            self,
            filename,
            header=None,
    ):
        with open(os.path.join(self.root, self.base_folder, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=",", skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1:]
        else:
            headers = []

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))

    def __getitem__(self, index):
        X = PIL.Image.open(os.path.join(
            self.root, self.base_folder, "img_align_celeba", "img_align_celeba", self.filename[index]))

        if self.transform is not None:
            X = self.transform(X)

        return X, 0

    def __len__(self):
        return len(self.filename)

    def extra_repr(self):
        lines = ["Split: {split}", ]
        return "\n".join(lines).format(**self.__dict__)


transform_mnist = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

transform_cifar10 = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

transform_celeba = transforms.Compose([
    crop_celeba,
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])


DATASETS = {
    "mnist": datasets.MNIST,
    "cifar10": datasets.CIFAR10,
    "celeba": CelebA if FROM_KAGGLE else datasets.CelebA
}


def get_data(root, dataset, download, batch_size):
    dataset = DATASETS[dataset]
    if dataset == "mnist":
        train_data = dataset(root=root, download=download, train=True, transform=transform_mnist)
    elif dataset == "cifar10":
        train_data = dataset(root=root, download=download, train=True, transform=transform_cifar10)
    elif dataset == "celeba":
        train_data = dataset(root=root, download=download, split="all", transform=transform_celeba)
    else:
        raise NotImplementedError
    train_loader = DataLoader(train_data, batch_size, num_workers=os.cpu_count(), shuffle=True, pin_memory=True)
    return train_loader
