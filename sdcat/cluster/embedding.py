# sdcat, Apache-2.0 license
# Filename: sdcat/cluster/embedding.py
# Description:  Miscellaneous functions for computing VIT embeddings and caching them.

import os
import multiprocessing
from pathlib import Path

import re
from PIL import Image, ImageFilter
from numpy import save, load
import numpy as np
from sahi.utils.torch import torch
from torchvision import transforms as pth_transforms
import cv2
from transformers import ViTModel, ViTImageProcessor

from sdcat.logger import info, err


class ViTWrapper:
    MODEL_NAME = "google/vit-base-patch16-224"
    VECTOR_DIMENSIONS = 768

    def __init__(self, device: str = "cpu", reset: bool = False, batch_size: int = 32):
        self.batch_size = batch_size

        self.model = ViTModel.from_pretrained(self.MODEL_NAME)
        self.processor = ViTImageProcessor.from_pretrained(self.MODEL_NAME)

        # Load the model and processor
        if 'cuda' in device and torch.cuda.is_available():
            device_num = int(device.split(":")[-1])
            info(f"Using GPU device {device_num}")
            torch.cuda.set_device(device_num)
            self.device = "cuda"
            self.model.to("cuda")
        else:
            self.device = "cpu"


def cache_embedding(embedding, model_name: str, filename: str):
    model_machine_friendly_name = model_name.replace("/", "_")
    # save numpy array as npy file
    save(f'{filename}_{model_machine_friendly_name}.npy', embedding)


def fetch_embedding(model_name: str, filename: str) -> np.array:
    model_machine_friendly_name = model_name.replace("/", "_")
    # if the npy file exists, return it
    if os.path.exists(f'{filename}_{model_machine_friendly_name}.npy'):
        data = load(f'{filename}_{model_machine_friendly_name}.npy')
        return data
    else:
        info(f'No embedding found for {filename}')
    return []


def has_cached_embedding(model_name: str, filename: str) -> int:
    """
    Check if the given filename has a cached embedding
    :param model_name: Name of the model
    :param filename: Name of the file
    :return: 1 if the image has a cached embedding, otherwise 0
    """
    model_machine_friendly_name = model_name.replace("/", "_")
    if os.path.exists(f'{filename}_{model_machine_friendly_name}.npy'):
        return 1
    return 0


def encode_image(filename):
    img = Image.open(filename)
    keep = img.copy()
    img.close()
    return keep


def compute_embedding_vits(images: list, model_name: str, device: str = "cpu"):
    """
    Compute the embedding for the given images using the given model
    :param images: List of image filenames
    :param model_name: Name of the model (i.e. google/vit-base-patch16-224, dinov2_vits16, etc.)
    :param device: Device to use for the computation (cpu or cuda:0, cuda:1, etc.)
    """
    batch_size = 8
    vit_model = ViTModel.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)

    if 'cuda' in device and torch.cuda.is_available():
        device_num = int(device.split(":")[-1])
        info(f"Using GPU device {device_num}")
        torch.cuda.set_device(device_num)
        vit_model.to("cuda")
        device = "cuda"
    else:
        device = "cpu"

    # Batch process the images
    batches = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]
    for batch in batches:
        try:
            # Skip running the model if the embeddings already exist
            if all([has_cached_embedding(model_name, filename) for filename in batch]):
                continue

            images = [Image.open(filename).convert("RGB") for filename in batch]
            inputs = processor(images=images, return_tensors="pt").to(device)

            with torch.no_grad():
                embeddings = vit_model(**inputs)

            batch_embeddings = embeddings.last_hidden_state[:, 0, :].cpu().numpy()

            # Save the embeddings
            for emb, filename in zip(batch_embeddings, batch):
                emb = emb.astype(np.float32)
                cache_embedding(emb, model_name, filename)
        except Exception as e:
            err(f'Error processing {batch}: {e}')


def compute_norm_embedding(model_name: str, images: list):
    """
    Compute the embedding for a list of images and save them to disk.
    Args:
    :param images:  List of image paths
    :param model_name: Name of the model to use for the embedding generation
    Returns:

    """
    # Calculate the mean and standard deviation of the images for normalization
    # This did not work well, but it might be worth revisiting, passing the mean
    # and std to the compute_embedding function
    # mean, std = calc_mean_std(images)

    # If using a GPU, set then skip the parallel CPU processing
    if torch.cuda.is_available():
        compute_embedding_vits(images, model_name)
    else:
        # Use a pool of processes to speed up the embedding generation 20 images at a time on each process
        num_processes = min(multiprocessing.cpu_count(), len(images) // 20)
        num_processes = max(1, num_processes)
        info(f'Using {num_processes} processes to compute {len(images)} embeddings 20 at a time ...')
        with multiprocessing.Pool(num_processes) as pool:
            args = [(images[i:i + 20], model_name) for i in range(0, len(images), 20)]
            pool.starmap(compute_embedding_vits, args)


def calc_mean_std(image_files: list) -> tuple:
    """
    Calculate the mean and standard deviation of a list of images.
    :param image_files: List of absolute image paths
    :return: mean and standard deviation of the images
    """
    mean = np.zeros((224, 224, 3), dtype=np.float32)
    std = np.zeros((224, 224, 3), dtype=np.float32)

    # Loop over all image files in a directory and process them in batches
    for i, file_path in enumerate(image_files):
        # Load the image and compute the mean and standard deviation for each color channel
        img = cv2.imread(file_path)
        # convert from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize the image to 224x224
        img = cv2.resize(img, (224, 224))

        # Compute the mean and standard deviation of the image
        mean += np.mean(img, axis=(0, 1))
        std += np.std(img, axis=(0, 1))

    mean_img = mean.astype(np.uint8) / len(image_files)
    std_img = std.astype(np.uint8) / len(image_files)
    return mean_img, std_img
