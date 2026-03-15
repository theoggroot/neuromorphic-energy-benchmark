# torchvision is a PyTorch library that provides datasets,
# model architectures and image transformations for computer vision tasks
import torchvision

# transforms are used to modify images before feeding them into the neural network
import torchvision.transforms as transforms

# DataLoader is used to efficiently load data in batches during training
from torch.utils.data import DataLoader


def get_dataloaders(batch_size=64):
    """
    This function loads the MNIST dataset and returns two dataloaders:

    1. train_loader -> used to train the neural network
    2. test_loader -> used to evaluate the model performance

    Batch size = number of images processed at once.
    """

    # ---------------------------------------------------
    # Why do we need transforms?
    # ---------------------------------------------------
    #
    # Neural networks cannot process raw images directly.
    #
    # Images are usually stored as pixel values (0–255).
    #
    # Neural networks work better if values are normalized
    # between 0 and 1.
    #
    # ToTensor() converts:
    #
    # image → PyTorch tensor
    # pixel range: 0-255 → 0-1
    #
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # ---------------------------------------------------
    # Load training dataset
    # ---------------------------------------------------
    #
    # MNIST contains handwritten digits (0-9)
    #
    # Each image:
    # size = 28 x 28 pixels
    #
    # train=True means we are loading the training split
    #
    train_dataset = torchvision.datasets.MNIST(
        root="./data",          # where dataset will be stored
        train=True,             # training dataset
        download=True,          # download automatically
        transform=transform     # apply the transform defined above
    )

    # ---------------------------------------------------
    # Load testing dataset
    # ---------------------------------------------------
    #
    # test dataset is used to evaluate how well
    # the model generalizes to unseen data
    #
    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    # ---------------------------------------------------
    # DataLoader explanation
    # ---------------------------------------------------
    #
    # DataLoader improves training speed by:
    #
    # 1. loading batches instead of single images
    # 2. shuffling the dataset
    # 3. managing memory efficiently
    #
    # batch_size=64 means:
    #
    # model trains on 64 images simultaneously
    #
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True   # shuffle prevents model from learning dataset order
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False  # no need to shuffle during evaluation
    )

    return train_loader, test_loader