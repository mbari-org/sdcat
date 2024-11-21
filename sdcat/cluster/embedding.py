# sdcat, Apache-2.0 license
# Filename: sdcat/cluster/embedding.py
# Description:  Miscellaneous functions for computing VIT embeddings and caching them.

import os
import multiprocessing
from PIL import Image
from numpy import save, load
import numpy as np
from sahi.utils.torch import torch
import cv2
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch.nn.functional as F

from sdcat.logger import info, err


class ViTWrapper:
    DEFAULT_MODEL_NAME = "google/vit-base-patch16-224"

    def __init__(self, device: str = "cpu", batch_size: int = 32, model_name: str = DEFAULT_MODEL_NAME):
        self.batch_size = batch_size
        self.name = model_name
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)

        # Load the model and processor
        if 'cuda' in device and torch.cuda.is_available():
            device_num = int(device.split(":")[-1])
            info(f"Using GPU device {device_num}")
            torch.cuda.set_device(device_num)
            self.device = "cuda"
            self.model.to("cuda")
        else:
            self.device = "cpu"

    @property
    def model_name(self) -> str:
        return self.name

    @property
    def vector_dimensions(self) -> int:
        return self.model.config.hidden_size

    def process_images(self, image_paths: list):
        info(f"Processing {len(image_paths)} images with {self.model_name}")

        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            embeddings = self.model.base_model(**inputs)
            batch_embeddings = embeddings.last_hidden_state[:, 0, :].cpu().numpy()
            # predicted_class_idx = logits.argmax(-1).cpu().numpy()
            # predicted_classes = [self.model.config.id2label[class_idx] for class_idx in predicted_class_idx]
            # predicted_scores = F.softmax(logits, dim=-1).cpu().numpy()
            # Get the top 3 classes and scores
            top_scores, top_classes = torch.topk(logits, 3)
            top_classes = top_classes.cpu().numpy()
            top_scores = F.softmax(top_scores, dim=-1).cpu().numpy()
            predicted_classes = [",".join([self.model.config.id2label[class_idx] for class_idx in class_list]) for class_list in top_classes]
            predicted_scores = [",".join([str(score) for score in score_list]) for score_list in top_scores]

        return batch_embeddings, predicted_classes, predicted_scores

def cache_embedding(embedding, pred, score, model_name: str, filename: str):
    model_machine_friendly_name = model_name.replace("/", "_")
    # save embeddings numpy array as npy file and the predictions as a txt file
    save(f'{filename}_{model_machine_friendly_name}.npy', embedding)
    with open(f'{filename}_{model_machine_friendly_name}_pred.txt', 'w') as f:
        predictions = pred.split(',')
        scores = score.split(',')
        for pred, score in zip(predictions, scores):
            f.write(f'{pred.strip()},{score.strip()}\n')


def fetch_embedding(model_name: str, filename: str):
    model_machine_friendly_name = model_name.replace("/", "_")
    # if the npy file exists, return it
    emb = []
    label = []
    score = []
    if os.path.exists(f'{filename}_{model_machine_friendly_name}.npy'):
        emb = load(f'{filename}_{model_machine_friendly_name}.npy')
    else:
        info(f'No embedding found for {filename}')
    if os.path.exists(f'{filename}_{model_machine_friendly_name}_pred.txt'):
        with open(f'{filename}_{model_machine_friendly_name}_pred.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(',')
                label.append(line[0])
                score.append(float(line[1]))
    else:
        info(f'No prediction found for {filename}')
    return emb, label, score


def has_cached_embedding(model_name: str, filename: str) -> int:
    """
    Check if the given filename has a cached embedding
    :param model_name: Name of the model
    :param filename: Name of the file
    :return: 1 if the image has a cached embedding, otherwise 0
    """
    model_machine_friendly_name = model_name.replace("/", "_")
    if os.path.exists(f'{filename}_{model_machine_friendly_name}.npy') and \
            os.path.exists(f'{filename}_{model_machine_friendly_name}_pred.txt'):
        return 1
    return 0


def encode_image(filename):
    img = Image.open(filename)
    keep = img.copy()
    img.close()
    return keep


def compute_embedding_vits(vit:ViTWrapper, images: list):
    """
    Compute the embedding for the given images using the given model
    :param vitwrapper: Wrapper for the ViT model
    :param images: List of image filenames
    :param model_name: Name of the model (i.e. google/vit-base-patch16-224, dinov2_vits16, etc.)
    :param device: Device to use for the computation (cpu or cuda:0, cuda:1, etc.)
    """
    batch_size = 32
    model_name = vit.model_name

    # Batch process the images
    batches = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]
    for batch in batches:
        try:
            # Skip running the model if the embeddings already exist
            if all([has_cached_embedding(model_name, filename) for filename in batch]):
                continue

            batch_embeddings, batch_labels, batch_scores = vit.process_images(batch)

            # Save the embeddings
            for emb, pred, score, filename in zip(batch_embeddings, batch_labels, batch_scores, batch):
                emb = emb.astype(np.float32)
                cache_embedding(emb, pred, score, model_name, filename)
        except Exception as e:
            err(f'Error processing {batch}: {e}')


def compute_norm_embedding(model_name: str, images: list, device: str = "cpu"):
    """
    Compute the embedding for a list of images and save them to disk.
    Args:
    :param images:  List of image paths
    :param model_name: Name of the model to use for the embedding generation
    :param device: Device to use for the computation (cpu or cuda:0, cuda:1, etc.)
    Returns:

    """
    # Calculate the mean and standard deviation of the images for normalization
    # This did not work well, but it might be worth revisiting, passing the mean
    # and std to the compute_embedding function
    # mean, std = calc_mean_std(images)
    vit_wrapper = ViTWrapper(device=device, model_name=model_name)

    # If using a GPU, set then skip the parallel CPU processing
    if torch.cuda.is_available():
        compute_embedding_vits(vit_wrapper, images)
    else:
        # Use a pool of processes to speed up the embedding generation 20 images at a time on each process
        num_processes = min(multiprocessing.cpu_count(), len(images) // 20)
        num_processes = max(1, num_processes)
        info(f'Using {num_processes} processes to compute {len(images)} embeddings 20 at a time ...')
        with multiprocessing.Pool(num_processes) as pool:
            args = [(vit_wrapper, images[i:i + 20]) for i in range(0, len(images), 20)]
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
