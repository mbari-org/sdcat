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
import torch.nn as nn
import cv2
from sdcat.logger import info, err


def cache_embedding(embedding, model_name: str, filename: str):
    # save numpy array as npy file
    save(f'{filename}_{model_name}.npy', embedding)


def cache_attention(attention, model_name: str, filename: str):
    # save numpy array as npy file
    save(f'{filename}_{model_name}_a.npy', attention)


def fetch_embedding(model_name: str, filename: str) -> np.array:
    # if the npy file exists, return it
    if os.path.exists(f'{filename}_{model_name}.npy'):
        data = load(f'{filename}_{model_name}.npy')
        return data
    else:
        info(f'No embedding found for {filename}')
    return []


def fetch_attention(model_name: str, filename: str) -> np.array:
    """
    Fetch the attention map for the given filename and model name
    :param model_name: Name of the model
    :param filename: Name of the file
    :return: Numpy array of the attention map
    """
    # if the npy file exists, return it
    if os.path.exists(f'{filename}_{model_name}_a.npy'):
        data = load(f'{filename}_{model_name}_a.npy')
        return data
    else:
        info(f'No attention map found for {filename}')
    return []


def has_cached_embedding(model_name: str, filename: str) -> int:
    """
    Check if the given filename has a cached embedding
    :param model_name: Name of the model
    :param filename: Name of the file
    :return: 1 if the image has a cached embedding, otherwise 0
    """
    if os.path.exists(f'{filename}_{model_name}.npy'):
        return 1
    return 0


def encode_image(filename):
    img = Image.open(filename)
    keep = img.copy()
    img.close()
    return keep


def compute_embedding(images: list, model_name: str):
    """
    Compute the embedding for the given images using the given model
    :param images: List of image filenames
    :param model_name: Name of the model
    """

    # Load the model
    if 'dinov2' in model_name:
        info(f'Loading model {model_name} from facebookresearch/dinov2...')
        model = torch.hub.load('facebookresearch/dinov2', model_name)
    elif 'dino' in model_name:
        info(f'Loading model {model_name} from facebookresearch/dino:main...')
        model = torch.hub.load('facebookresearch/dino:main', model_name)
    else:
        # TODO: Add more models
        err(f'Unknown model {model_name}!')
        return

    # The patch size is in the model name, e.g. dino_vits16 is a 16x16 patch size, dino_vits8 is a 8x8 patch size
    res = re.findall(r'\d+$', model_name)
    if len(res) > 0:
        patch_size = int(res[0])
    else:
        raise ValueError(f'Could not find patch size in model name {model_name}')
    info(f'Using patch size {patch_size} for model {model_name}')

    # Load images and generate embeddings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        # Set the cuda device
        if torch.cuda.is_available():
            model = model.to(device)

        for filename in images:
            # Skip if the embedding already exists
            if Path(f'{filename}_{model_name}.npy').exists():
                continue

            # Load the image
            square_img = Image.open(filename)

            # Do some image processing to reduce the noise in the image
            # Gaussian blur
            square_img = square_img.filter(ImageFilter.GaussianBlur(radius=1))

            image = np.array(square_img)

            norm_transform = pth_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            # Noramlize the tensor with the mean and std of the ImageNet dataset
            img_tensor = norm_transform(img_tensor)
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
            if 'cuda' in device:
                img_tensor = img_tensor.to(device)
            features = model(img_tensor)

            # TODO: add attention map cach as optional
            # attentions = model.get_last_selfattention(img_tensor)

            # nh = attentions.shape[1]  # number of head

            # w_featmap = 224 // patch_size
            # h_featmap = 224 // patch_size

            # Keep only the output patch attention
            # attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
            # attentions = attentions.reshape(nh, w_featmap, h_featmap)
            # attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[
            #     0].cpu().numpy()
            #
            # # Resize the attention map to the original image size
            # attentions = np.uint8(255 * attentions[0])

            # Get the feature embeddings
            embeddings = features.squeeze(dim=0)  # Remove batch dimension
            embeddings = embeddings.cpu().numpy()  # Convert to numpy array

            cache_embedding(embeddings, model_name, filename)  # save the embedding to disk
            #cache_attention(attentions, model_name, filename)  # save the attention map to disk


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
        compute_embedding(images, model_name)
    else:
        # Use a pool of processes to speed up the embedding generation 20 images at a time on each process
        num_processes = min(multiprocessing.cpu_count(), len(images) // 20)
        num_processes = max(1, num_processes)
        info(f'Using {num_processes} processes to compute {len(images)} embeddings 20 at a time ...')
        with multiprocessing.Pool(num_processes) as pool:
            args = [(images[i:i + 20], model_name) for i in range(0, len(images), 20)]
            pool.starmap(compute_embedding, args)


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
