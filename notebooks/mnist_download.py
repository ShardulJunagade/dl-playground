# download mnist dataset
from torchvision.datasets import MNIST
from torchvision import transforms
import os
import shutil
import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
import random
import matplotlib.pyplot as plt
from torchvision import datasets

def download_mnist(root, train=True, transform=None):
    """
    Download the MNIST dataset and save it to the specified root directory.
    
    Args:
        root (str): Root directory where the dataset will be saved.
        train (bool): If True, download the training set; otherwise, download the test set.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
    
    Returns:
        None
    """
    mnist = MNIST(root=root, train=train, download=True, transform=transform)
    return mnist
def save_mnist_images(mnist, save_dir):
    """
    Save MNIST images to the specified directory.

    Args:
        mnist (MNIST): The MNIST dataset object.
        save_dir (str): Directory where the images will be saved.

    Returns:
        None
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in tqdm(range(len(mnist))):
        image, label = mnist[i]
        image = transforms.ToPILImage()(image)
        label_dir = os.path.join(save_dir, str(label))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        image.save(os.path.join(label_dir, f"{i}.png"))

def main():
    parser = argparse.ArgumentParser(description="Download and save MNIST dataset")
    parser.add_argument("--root", type=str, default="./mnist_data", help="Root directory to save the dataset")
    parser.add_argument("--save_dir", type=str, default="./mnist_images", help="Directory to save the images")
    args = parser.parse_args()

    # Define a transform to convert the images to tensors
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download the MNIST dataset
    mnist_train = download_mnist(args.root, train=True, transform=transform)
    mnist_test = download_mnist(args.root, train=False, transform=transform)

    # Save the training and test images
    save_mnist_images(mnist_train, os.path.join(args.save_dir, "train"))
    save_mnist_images(mnist_test, os.path.join(args.save_dir, "test"))

if __name__ == "__main__":
    main()