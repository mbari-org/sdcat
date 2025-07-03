# sdcat, Apache-2.0 license
# Filename: sdcat/cluster/utils.py
# Description: Miscellaneous utility functions for cropping, clustering, and saving detections
import cv2
import numpy as np
import modin.pandas as pd
from typing import List

import pandas
from PIL import Image
from cleanvision import Imagelab
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pathlib import Path
from sdcat.logger import info, debug, warn, exception
from sdcat import __version__ as sdcat_version


def cluster_grid(
    prefix: str, cluster_sim: float, cluster_id: int, nb_images_display: int, images: List[str], output_path: Path
):
    """
    Cluster visualization; create a grid of images
    :
    :param cluster_sim: Cluster similarity
    :param cluster_size: Size of the cluster
    :param cluster_id: Cluster ID
    :param nb_images_display : Number of images to display in the grid
    :param images: list of images
    :param output_path: output path to save the visualization to
    :return:
    """
    cluster_size = len(images)
    debug(f"Cluster number {cluster_id} size {cluster_size} similarity {cluster_sim}")
    try:
        # Plot a grid for each group of images nb_images_display at a time (e.g. 8x8)
        for i in range(0, len(images), nb_images_display * nb_images_display):
            fig = plt.figure(figsize=(10.0, 10.0))
            grid = ImageGrid(
                fig,
                111,  # similar to subplot(111)
                nrows_ncols=(nb_images_display, nb_images_display),
                # creates nb_images_display x nb_images_display grid of axes
                axes_pad=0.025,
                share_all=True,
                cbar_pad=0.025,
            )
            images_display = images[i : i + nb_images_display * nb_images_display]
            page = i // (nb_images_display * nb_images_display)

            # If we have more than 3 pages, then only display the first 3 pages
            # There can be a large number of pages for detections in common classes
            if page > 3:
                break

            total_pages = len(images) // (nb_images_display * nb_images_display)
            for j, image in enumerate(images_display):
                try:
                    image_square = Image.open(image)
                    grid[j].imshow(image_square)
                except Exception as e:
                    exception(f"Error opening {image} {e}")
                    continue

                grid[j].axis("off")
                grid[j].set_xticklabels([])

            if cluster_id == -1:
                cluster_id = "noise"

            # Add a title to the figure
            if total_pages > 1:
                fig.suptitle(
                    f"{prefix}\nsdcat_version {sdcat_version}\nCluster {cluster_id}, Page: {page} of {total_pages}\nSimilarity: {cluster_sim:.2f}, Size: {cluster_size}",
                    fontsize=14,
                )
            else:
                fig.suptitle(
                    f"{prefix}\nsdcat_version {sdcat_version}\nCluster {cluster_id}\nSimilarity: {cluster_sim:.2f}, Size: {cluster_size}",
                    fontsize=14,
                )

            # Set the background color of the grid to white
            fig.set_facecolor("white")

            # Write the figure to a file
            out = output_path / f"{prefix}_cluster_{cluster_id}_p{page}.png"
            debug(f"Writing {out}")
            fig.savefig(out.as_posix())
            plt.close(fig)
    except Exception as e:
        exception(f"Error creating cluster grid {e}")


def clean_bad_images(filepaths: List[str]) -> List[str]:
    """Remove blurry and near duplicate images."""
    imagelab = Imagelab(filepaths=filepaths)

    # Override the default configurations
    config = {
        "visualize_num_images_per_row": 4,
        "report_num_images": 4,
        "report_max_prevalence": 1.0,  # Default is 0.5, but we want to see all issues
        "report_cell_size": (2, 2),
    }
    imagelab._config = config

    # Detect dark and blurry images
    issue_types = {
        "blurry": {"threshold": 0.52},  # Default is 0.5, but we want to be more strict
        "near_duplicates": {"hash_size": 4, "hash_types": ["whash", "phash"]},
    }
    imagelab.find_issues(issue_types)
    # imagelab.report() Disabling report = this is causing a segfault when run on large datasets

    bad_images = []

    # Remove all low information images
    low_info_images = imagelab.issues[imagelab.issues["is_low_information_issue"]].index
    if len(low_info_images) > 0:
        info(f"Removing {len(low_info_images)} low information images")
        bad_images.extend(low_info_images)

    # Remove all dark images
    dark_images = imagelab.issues[imagelab.issues["is_dark_issue"]].index
    if len(dark_images) > 0:
        info(f"Removing {len(dark_images)} dark images")
        bad_images.extend(dark_images)

    # Remove all blurry images
    blurry_images = imagelab.issues[imagelab.issues["is_blurry_issue"]].index
    if len(blurry_images) > 0:
        info(f"Removing {len(blurry_images)} blurry images")
        bad_images.extend(blurry_images)

    # Remove one image from each near duplicate set (keep the best blurry score)
    near_duplicates_image_sets = imagelab.info["near_duplicates"]["sets"]
    near_duplicates_scores = (
        imagelab.issues[imagelab.issues["is_near_duplicates_issue"]]
        .sort_values(by="blurry_score")
        .reset_index()[["index", "blurry_score"]]
        .values.tolist()
    )

    info(f"Removing {len(near_duplicates_image_sets)} duplicated image sets, keeping the least blurry image")
    for dup_set in near_duplicates_image_sets:
        if len(dup_set) <= 1:
            continue
        # Keep the image with the **highest** blurry_score (least blurry)
        best_image = max(
            (entry for entry in near_duplicates_scores if entry[0] in dup_set), key=lambda x: x[1], default=None
        )
        if best_image is None:
            continue
        to_remove = [img for img in dup_set if img != best_image[0]]
        bad_images.extend(to_remove)
    return list(bad_images)


def compute_square_coordinates(row) -> List[int]:
    """
    Compute square coordinates for cropping an image to a square, padding the shorter dimension.

    This also adjusts the crop to make sure the crop is fully in the frame, otherwise the crop that
    exceeds the frame is filled with black bars - these produce clusters of "edge" objects instead
    of the detection
    """
    x1 = int(row.image_width * row.x)
    y1 = int(row.image_height * row.y)
    x2 = int(row.image_width * row.xx)
    y2 = int(row.image_height * row.xy)
    width = x2 - x1
    height = y2 - y1
    shorter_side = min(height, width)
    longer_side = max(height, width)
    delta = abs(longer_side - shorter_side)

    # Divide the difference by 2 to determine how much padding is needed on each side
    padding = delta // 2

    # Add the padding to the shorter side of the image
    if width == shorter_side:
        x1 -= padding
        x2 += padding
    else:
        y1 -= padding
        y2 += padding

    # Make sure that the coordinates don't go outside the image
    # If they do, adjust by the overflow amount
    if y1 < 0:
        y1 = 0
        y2 += abs(y1)
        if y2 > row.image_height:
            y2 = row.image_height
    elif y2 > row.image_height:
        y2 = row.image_height
        y1 -= abs(y2 - row.image_height)
        if y1 < 0:
            y1 = 0
    if x1 < 0:
        x1 = 0
        x2 += abs(x1)
        if x2 > row.image_width:
            x2 = row.image_width
    elif x2 > row.image_width:
        x2 = row.image_width
        x1 -= abs(x2 - row.image_width)
        if x1 < 0:
            x1 = 0
    return [x1, y1, x2, y2]


def crop_rotated_fill(image, cx, cy, w, h, angle) -> np.ndarray:
    """
    Crop a rotated rectangle from an image and fill the background with the mean color of the image.
    """
    # Compute mean color of the image (across all pixels and channels)
    mean_color = image.mean(axis=(0, 1)).astype(np.uint8).tolist()

    # Create rotated rectangle
    rect = ((cx, cy), (w, h), angle)
    box = cv2.boxPoints(rect).astype(np.float32)

    # Destination points for upright box
    dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(box, dst_pts)

    # Warp with border filled by mean color
    warped = cv2.warpPerspective(image, M, (int(w), int(h)), borderMode=cv2.BORDER_CONSTANT, borderValue=mean_color)
    return warped


def crop_all_square_images(image_path: str, detections: pandas.DataFrame, square_dim: int, rotation: int):
    """
    Crop the image to a square padding the shortest dimension, then resize it to square_dim x square_dim
    Optimized for cropping all detections within an image
    :param image_path: Path to the image
    :param detections: Dataframe with the detections
    :param square_dim: dimension of the square image
    :param rotation: Rotation angle in degrees to apply to the image before cropping
    """
    try:
        if not Path(image_path).exists():
            warn(f"Skipping because the image {image_path} does not exist")
            return

        # If the number of crops that already exist is equal to the number of detections, return early
        detections["exists"] = detections["crop_path"].apply(lambda x: Path(x).exists())
        if detections["exists"].all():
            debug(f"All crops for {image_path} already exist, skipping cropping")
            return

        if rotation > 0:
            # Read the image converting to BGR format for OpenCV
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Iterate over the detections and crop each one
            for index, row in detections.iterrows():
                # If the crop already exists, skip it to save time
                if Path(row.crop_path).exists():
                    continue
                x1, y1, x2, y2 = compute_square_coordinates(row)
                img_cropped = crop_rotated_fill(
                    image_rgb, cx=int((x1 + x2) / 2), cy=int((y1 + y2) / 2), w=x2 - x1, h=y2 - y1, angle=int(rotation)
                )

                # Resize the image to square_dim x square_dim
                cropped = cv2.resize(img_cropped, (224, 224), interpolation=cv2.INTER_LANCZOS4)
                # Convert back to BGR format for saving
                cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
                cv2.imwrite(row.crop_path, cropped)
                debug(f"Cropping {image_path} to {row.crop_path} with coordinates ({x1}, {y1}, {x2}, {y2})")
        else:
            img = Image.open(image_path)

            # Iterate over the detections and crop each one
            for index, row in detections.iterrows():
                # If the crop already exists, skip it to save time
                if Path(row.crop_path).exists():
                    continue
                x1, y1, x2, y2 = compute_square_coordinates(row)
                img_cropped = img.crop((x1, y1, x2, y2))
                # Resize the image to square_dim x square_dim
                img_cropped = img_cropped.resize((square_dim, square_dim), Image.LANCZOS)
                debug(f"Cropping {image_path} to {row.crop_path} with coordinates ({x1}, {y1}, {x2}, {y2})")
                img_cropped.save(row.crop_path)
                img_cropped.close()

            img.close()
    except Exception as e:
        exception(f"Error cropping {image_path} {e}")
        raise e


def crop_square_image(row, square_dim: int):
    """
    Crop the image to a square padding the shortest dimension, then resize it to square_dim x square_dim
    :param row:
    :param square_dim: dimension of the square image
    :return:
    """
    try:
        if not Path(row.image_path).exists():
            warn(f"Skipping {row.crop_path} because the image {row.image_path} does not exist")
            return

        if Path(row.crop_path).exists():  # If the crop already exists, skip it
            return

        x1, y1, x2, y2 = compute_square_coordinates(row)

        # Crop the image
        img = Image.open(row.image_path)
        img = img.crop((x1, y1, x2, y2))

        # Resize the image to square_dim x square_dim
        img = img.resize((square_dim, square_dim), Image.LANCZOS)

        # Save the image
        img.save(row.crop_path)
        img.close()

    except Exception as e:
        exception(f"Error cropping {row.image_path} {e}")
        raise e


def rescale(img: np.ndarray, scale_percent: int = 75) -> np.ndarray:
    """
    Rescale an image
    :param img: Image to rescale
    :param scale_percent: Scale percentage
    :return: Rescaled image
    """
    width, height = img.shape[:2]
    width = int(width * scale_percent / 100)
    height = int(height * scale_percent / 100)
    dim = (height, width)

    # Resize the image to the new dimensions exactly
    img_rescaled = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img_rescaled


def filter_images(min_area: int, max_area: int, min_saliency: int, min_score: float, df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the dataframe to remove images that are too small or have low saliency
    :param min_area: Minimum area of the image
    :param max_area: Maximum area of the image
    :param min_saliency: Minimum saliency of the image
    :param df: Dataframe to filter
    :return: Filtered dataframe
    """
    # Filter by saliency, area, score or day/night
    size_before = len(df)
    if "saliency" in df.columns:
        df = df[(df["saliency"] > min_saliency) | (df["saliency"] == -1)]
    if "area" in df.columns:
        df = df[(df["area"] > min_area) & (df["area"] < max_area)]
    if "score" in df.columns:
        df = df[(df["score"] > min_score)]
    size_after = len(df)
    info(f"Removed {size_before - size_after} detections outside of area, saliency, or too low scoring")

    return df


def compute_embedding_multi_gpu(model_name: str, images: list, batch_size: int = 32):
    from concurrent.futures import ThreadPoolExecutor
    from sdcat.cluster.embedding import ViTWrapper, compute_embedding_vits
    import torch
    import math

    # Detect all CUDA devices
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

    # Split the images evenly across GPUs
    def split_batches(images, num_splits):
        batch_size = math.ceil(len(images) / num_splits)
        return [images[i * batch_size : (i + 1) * batch_size] for i in range(num_splits)]

    # Compute embeddings per GPU
    def compute_on_device(device, model_name, images, batch_size):
        vit_wrapper = ViTWrapper(device=device, model_name=model_name)
        compute_embedding_vits(vit_wrapper, images, batch_size)

    def multi_gpu_compute(model_name, images, batch_size):
        image_batches = split_batches(images, len(devices))
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = [
                executor.submit(compute_on_device, device, model_name, batch, batch_size)
                for device, batch in zip(devices, image_batches)
            ]
            # Wait for all tasks to complete
            for f in futures:
                f.result()

    multi_gpu_compute(model_name, images, batch_size)


def combine_csv(csv_files: List[Path], temp_path: Path, crop_path: str) -> str:
    from tqdm import tqdm

    output_file = temp_path / "combined.csv"

    info(f"Combining detection files to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as outfile:
        first_file = True
        for file in tqdm(csv_files, desc="Combining detection files", unit="file"):
            # Create a crop directory for each detection file
            crop_root = crop_path / file.stem
            crop_root.mkdir(parents=True, exist_ok=True)
            with open(file, "r", encoding="utf-8") as infile:
                lines = infile.readlines()
                if first_file:
                    header = lines[0].strip() + ",crop_root\n"
                    outfile.writelines(header)  # include header
                    out_text = [line.strip() + f",{crop_root}\n" for line in lines[1:]]
                    outfile.writelines(out_text)
                    first_file = False
                else:
                    out_text = [line.strip() + f",{crop_root}\n" for line in lines[1:]]
                    outfile.writelines(out_text)
    info(f"Combined detection files to {output_file}")
    return str(output_file)
