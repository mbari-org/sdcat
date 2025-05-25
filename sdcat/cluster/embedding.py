# sdcat, Apache-2.0 license
# Filename: sdcat/cluster/embedding.py
# Description:  Miscellaneous functions for computing VIT embeddings and caching them.

import os
from PIL import Image
from numpy import save, load
import numpy as np
from sahi.utils.torch import torch
import cv2
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch.nn.functional as F

from sdcat.cluster.utils import compute_embedding_multi_gpu
from sdcat.logger import info, err

class ViTWrapper:
    DEFAULT_MODEL_NAME = "google/vit-base-patch16-224"

    def __init__(self, device: str = "cpu", batch_size: int = 32, model_name: str = DEFAULT_MODEL_NAME):
        self.batch_size = batch_size
        self.name = model_name

        # Initialize device
        if "cuda" in device and torch.cuda.is_available():
            device_num = int(device.split(":")[-1])
            info(f"Using GPU device {device_num}")
            self.device = torch.device(f"cuda:{device_num}")
            torch.cuda.set_device(device_num)
        else:
            self.device = torch.device("cpu")

        # Load model & processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name).to(self.device)

    @property
    def model_name(self) -> str:
        return self.name

    @property
    def vector_dimensions(self) -> int:
        return self.model.config.hidden_size

    def process_images(self, image_paths: list[str]):
        info(f"Processing {len(image_paths)} images with {self.model_name}")

        # Load and preprocess images
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(images=images, return_tensors="pt")

        # Send tensors to device for speed-up
        for k in inputs:
            inputs[k] = inputs[k].to(device=self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            embeddings = self.model.base_model(**inputs)
            batch_embeddings = embeddings.last_hidden_state[:, 0, :].cpu().numpy()

            top_n = min(logits.shape[1], 2)
            top_scores, top_classes = torch.topk(logits, top_n)
            top_scores = F.softmax(top_scores, dim=-1).cpu().numpy()
            top_classes = top_classes.cpu().numpy()

            id2label = self.model.config.id2label
            predicted_classes = [",".join([id2label[idx] for idx in class_list]) for class_list in top_classes]
            predicted_scores = [",".join([f"{score:.4f}" for score in score_list]) for score_list in top_scores]

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
    emb_filename = f'{filename}_{model_machine_friendly_name}.npy'
    if os.path.exists(emb_filename):
        emb = load(emb_filename)
    else:
        info(f'No embedding found for {os.path.basename(emb_filename)}')
    if os.path.exists(f'{filename}_{model_machine_friendly_name}_pred.txt'):
        with open(f'{filename}_{model_machine_friendly_name}_pred.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(',')
                label.append(line[0])
                score.append(float(line[1]))
    else:
        info(f'No prediction found for {filename} for {model_name}')
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


def compute_embedding_vits(vit:ViTWrapper, images: list, batch_size:int=32):
    """
    Compute the embedding for the given images using the given model
    :param vitwrapper: Wrapper for the ViT model
    :param images: List of image filenames
    :param batch_size: Number of images to process in a batch
    """
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


def compute_norm_embedding(model_name: str, images: list, device: str = "cpu", batch_size: int = 32):
    """
    Compute the embedding for a list of images and save them to disk.
    Args:
    :param images:  List of image paths
    :param model_name: Name of the model to use for the embedding generation
    :param device: Device to use for the computation (cpu or cuda:0, cuda:1, etc.)
    :param batch_size: Number of images to process in a batch
    Returns:

    """
    # Calculate the mean and standard deviation of the images for normalization
    # This did not work well, but it might be worth revisiting, passing the mean
    # and std to the compute_embedding function
    # mean, std = calc_mean_std(images)

    # If using a GPU, set then skip the parallel CPU processing
    if torch.cuda.is_available() and 'cuda' in device:
        if torch.cuda.device_count() > 1 and device == "cuda":
            torch.cuda.empty_cache()
            compute_embedding_multi_gpu(model_name, images, batch_size)
        else:
            vit_wrapper = ViTWrapper(device=device, model_name=model_name)
            compute_embedding_vits(vit_wrapper, images, batch_size)
    else:
        # TODO: replace this - modin does not work well here
        vit_wrapper = ViTWrapper(device='cpu', model_name=model_name)
        import modin.pandas as pd
        df_args = pd.DataFrame([{
            "vit_wrapper": vit_wrapper,
            "images_batch": [batch],
            "batch_size": batch_size
        } for batch in images])

        def compute_embedding_wrapper(row):
            return compute_embedding_vits(
                row.vit_wrapper,
                row.images_batch,
                row.batch_size
            )
        info(f"Compute embeddings for {len(images)} images on CPU ...")
        df_args.apply(compute_embedding_wrapper, axis=1)

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
