from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def crop_celeba(img):
    return transforms.functional.crop(img, top=40, left=15, height=148, width=148)


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


def get_data(root, dataset, batch_size, num_workers):
    if dataset == "mnist":
        train_data = datasets.MNIST(root=root, download=False, train=True, transform=transform_mnist)
    elif dataset == "cifar10":
        train_data = datasets.CIFAR10(root=root, download=False, train=True, transform=transform_cifar10)
    elif dataset == "celeba":
        train_data = datasets.CelebA(root=root, download=False, split="train", transform=transform_celeba)
    else:
        raise NotImplementedError
    train_loader = DataLoader(train_data, batch_size, num_workers=num_workers, shuffle=True)
    return train_loader
