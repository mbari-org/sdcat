# sdcat, Apache-2.0 license
# Filename: sdcat/cluster/utils.py
# Description: Miscellaneous utility functions for cropping, clustering, and saving detections

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pathlib import Path

from sdcat.logger import debug, warn, exception


def cluster_grid(prefix: str, cluster_sim: float, cluster_id: int, cluster_size: int, nb_images_display: int,
                 images: list, output_path: Path):
    """
    Cluster visualization; create a grid of images
    :param cluster_sim: Cluster similarity
    :param cluster_size: Size of the cluster
    :param cluster_id: Cluster ID
    :param images: list of images
    :param output_path: output path to save the visualization to
    :return:
    """
    debug(f'Cluster number {cluster_id} size {len(cluster_size)} similarity {cluster_sim}\n')

    # Plot a grid for each group of images nb_images_display at a time (e.g. 8x8)
    for i in range(0, len(images), nb_images_display * nb_images_display):
        fig = plt.figure(figsize=(10., 10.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(nb_images_display, nb_images_display),
                         # creates nb_images_display x nb_images_display grid of axes
                         axes_pad=0.025,
                         share_all=True,
                         cbar_pad=0.025)
        images_display = images[i:i + nb_images_display * nb_images_display]
        page = i // (nb_images_display * nb_images_display)

        # If we have more than 3 pages, then only display the first 3 pages
        # There can be a large number of pages for detections in common classes
        if page > 3:
            break

        total_pages = len(images) // (nb_images_display * nb_images_display)
        # debug(f"{i} Image filename:", images[j])
        for j, image in enumerate(images_display):
            try:
                image_square = Image.open(image)
                grid[j].imshow(image_square)
            except Exception as e:
                exception(f'Error opening {image} {e}')
                continue

            grid[j].axis('off')
            # If the verified is in the image name, then add a label to the image in the top center corner
            if 'verified' in image:
                n = Path(image)
                title = f"{n.stem.split('_')[0]}"
                grid[j].text(30, 10, title, fontsize=8, color='white', ha='center', va='center')
            # clear the x and y-axis
            grid[j].set_xticklabels([])

        # Add a title to the figure
        if total_pages > 1:
            fig.suptitle(
                f"{prefix} Cluster {cluster_id}, Size: {len(cluster_size)}, Similarity: {cluster_sim:.2f}, Page: {page} of {total_pages}",
                fontsize=16)
        else:
            fig.suptitle(f"{prefix} Cluster {cluster_id}, Size: {len(cluster_size)}, Similarity: {cluster_sim:.2f}",
                         fontsize=16)

        # Set the background color of the grid to white
        fig.set_facecolor('white')

        # Write the figure to a file
        out = output_path / f'{prefix}_cluster_{cluster_id}_p{page}.png'
        debug(f'Writing {out}')
        fig.savefig(out.as_posix())
        plt.close(fig)



def square_image(row, square_dim: int):
    """
    Squares an image to the model dimension, filling it with black bars if necessary
    :param row:
    :param square_dim: dimension of the square image
    :return:
    """
    try:
        if not Path(row.image_path).exists():
            warn(f'Skipping {row.crop_path} because the image {row.image_path} does not exist')
            return

        if Path(row.crop_path).exists():  # If the crop already exists, skip it
            return

        # Determine the size of the new square
        max_side = max(row.image_width, row.image_height)

        # Create a new square image with a black background
        new_image = Image.new('RGB', (max_side, max_side), (0, 0, 0))

        img = Image.open(row.image_path)

        # Paste the original image onto the center of the new image
        new_image.paste(img, ((max_side - row.image_width) // 2, (max_side - row.image_height) // 2))

        # Resize the image to square_dim x square_dim
        img = img.resize((square_dim, square_dim), Image.LANCZOS)

        # Save the image
        img.save(row.crop_path)
        img.close()
    except Exception as e:
        exception(f'Error cropping {row.image_path} {e}')
        raise e


def crop_square_image(row, square_dim: int):
    """
    Crop the image to a square padding the shortest dimension, then resize it to square_dim x square_dim
    This also adjusts the crop to make sure the crop is fully in the frame, otherwise the crop that
    exceeds the frame is filled with black bars - these produce clusters of "edge" objects instead
    of the detection
    :param row:
    :param square_dim: dimension of the square image
    :return:
    """
    try:

        if not Path(row.image_path).exists():
            warn(f'Skipping {row.crop_path} because the image {row.image_path} does not exist')
            return

        if Path(row.crop_path).exists():  # If the crop already exists, skip it
            return

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

        # Crop the image
        img = Image.open(row.image_path)
        img = img.crop((x1, y1, x2, y2))

        # Resize the image to square_dim x square_dim
        img = img.resize((square_dim, square_dim), Image.LANCZOS)

        # Save the image
        img.save(row.crop_path)
        img.close()

    except Exception as e:
        exception(f'Error cropping {row.image_path} {e}')
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