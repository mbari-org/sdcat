# sdcat, Apache-2.0 license
# Filename: sdcat/ml/saliency.py
# Description:  Miscellaneous saliency functions for detecting targets in images using saliency maps
import os
import tempfile
import cv2
import numpy as np
import pandas as pd
from skimage.feature import blob_log

from sdcat.cluster.utils import rescale
from sdcat.reflectance.cleaner import specularity_removal
from sdcat.logger import debug, info
from pathlib import Path

# This is a global variable to control whether to save the algorithm results; useful for debugging or presentation
save = True


def compute_saliency(contour, min_std: float, img_saliency: np.ndarray, img_luminance: np.ndarray) -> int:
    """
    Calculate the saliency cost of a contour. Lower saliency contours are more likely to be reflections.
    :param contour: the contour
    :param min_std: minimum standard deviation of the contour to be considered
    :param img_saliency: saliency image
    :param img_luminance: luminance image
    :return: saliency cost (int)
    """
    # Calculate brightness and other statistics of the contour
    mask = np.zeros_like(img_saliency)
    cv2.drawContours(mask, [contour], 0, 255, thickness=cv2.FILLED)
    mean_intensity_l = cv2.mean(img_luminance, mask=mask)[0]
    mean_intensity_s = cv2.mean(img_saliency, mask=mask)[0]

    # Calculate area of the contour
    area = cv2.contourArea(contour)

    # Calculate variance of pixel intensities within the contour
    x, y, w, h = cv2.boundingRect(contour)
    roi = img_luminance[y : y + h, x : x + w]
    variance = np.var(roi)
    std = np.std(roi)

    # Create a factor to normalize the cost for different image sizes
    factor = img_luminance.size / 10000

    # The cost function penalizes smaller areas with low variance
    cost = (area * (mean_intensity_l + mean_intensity_s + 0.1 * area) - variance) / factor

    # If the std is too low, then the contour is not interesting; set the cost to 1
    if std < min_std:
        cost = 1

    return int(cost)


def process_contour(contours: np.ndarray, min_std: float, img_s: np.ndarray, img_l: np.ndarray) -> np.ndarray:
    """
    Process a single contour. Save the results to a csv file.
    This is a separate function, so it can be run in parallel.
    :param contours:  List of contours
    :param min_std: minimum standard deviation of the contour to be considered
    :param img_s: Saliency image
    :param img_l: Luminance image
    :return: dataframe of blobs
    """
    df = pd.DataFrame()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        area = cv2.contourArea(c)
        saliency = compute_saliency(c, min_std, img_s, img_l)

        debug(f"Found blob area: {area}, saliency: {saliency}, area: {area}")
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "image_path": "",
                        "class": "Unknown",
                        "score": 0.0,
                        "area": area,
                        "saliency": saliency,
                        "x": x,
                        "y": y,
                        "xx": x + w,
                        "xy": y + h,
                        "w": w,
                        "h": h,
                    },
                    index=[0],
                ),
            ]
        )

    return df




def extract_blobs(
    min_std: float, block_size: int, saliency_map: np.ndarray, img_color: np.ndarray, show=False, small=False
) -> (pd.DataFrame, np.ndarray):
    """
    Extract blobs from an image using Laplacian-of-Gaussian (LoG) blob detection.
    Uneven illumination / vignette is removed first via large-scale background subtraction.
    :param min_std: unused (kept for API compatibility)
    :param block_size: unused (kept for API compatibility)
    :param saliency_map: unused (kept for API compatibility)
    :param img_color: color image used for blob detection and visualisation
    :param show: True to show intermediate results
    :param small: unused (kept for API compatibility)
    :return: pandas dataframe of blobs, image with detections drawn
    """

    img_f = saliency_map.astype(np.float32) / 255.0

    # ── Pre-processing: remove vignette / uneven illumination ─────────────────
    bg = cv2.GaussianBlur(img_f, (201, 201), 0)   # large-scale background estimate
    fg = np.clip(img_f - bg, 0, 1)                # flat foreground

    if show:
        cv2.imshow("Foreground (vignette removed)", (fg * 255).astype(np.uint8))
        if save:
            cv2.imwrite("my_fg_vignette_removed.jpg", (fg * 255).astype(np.uint8))
        cv2.waitKey(0)

    # ── Blob detection (LoG) ──────────────────────────────────────────────────
    # Each row of `blobs`: [y, x, sigma]  →  radius ≈ sigma * sqrt(2)
    blobs = blob_log(
        fg,
        min_sigma  = 1.5,   # smallest particle ≈ 2 px radius
        max_sigma  = 20,    # largest  particle ≈ 28 px radius
        num_sigma  = 15,    # number of scale steps
        threshold  = 0.008, # lower → more sensitive (may add false positives)
        overlap    = 0.3,   # suppress overlapping detections
    )

    info(f"Found {len(blobs)} blobs via LoG. Building detections...")

    # ── Convert blob list to bounding-box dataframe ───────────────────────────
    rows = []
    for blob in blobs:
        by, bx, sigma = blob
        r = int(np.ceil(sigma * np.sqrt(2)))
        x  = max(0, int(bx) - r)
        y  = max(0, int(by) - r)
        xx = min(img_color.shape[1], int(bx) + r)
        xy = min(img_color.shape[0], int(by) + r)
        w  = xx - x
        h  = xy - y
        area = int(np.pi * r * r)
        rows.append(
            {
                "image_path": "",
                "class": "Unknown",
                "score": 0.0,
                "area": area,
                "saliency": float(sigma),   # use sigma as a proxy for saliency
                "x": x,
                "y": y,
                "xx": xx,
                "xy": xy,
                "w": w,
                "h": h,
            }
        )

    df = pd.DataFrame(rows)

    # ── Visualisation ─────────────────────────────────────────────────────────
    contour_img = img_color.copy()
    for _, row in df.iterrows():
        cv2.rectangle(
            contour_img,
            (int(row["x"]), int(row["y"])),
            (int(row["xx"]), int(row["xy"])),
            (81, 12, 51), 2,
        )

    if show:
        for _, row in df.iterrows():
            cv2.putText(
                contour_img,
                f"{row['saliency']:.1f}",
                (int(row["x"]), int(row["y"])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
            )
        cv2.imshow(f"{len(df)} Bounding Boxes (LoG)", contour_img)
        if save:
            cv2.imwrite("my_bounding_boxes.jpg", contour_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return df, contour_img


def fine_grained_saliency(image, clahe=False, show=False, small=False) -> np.ndarray:
    """
    Compute a fine-grained saliency map and normalize it
    :param image:  image
    :param clahe: True to run CLAHE on the luminance channel
    :param show:  True to show the results
    :return: normalized saliency map
    """

    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Blur the saturation channel to remove noise
    img_hsv[:, :, 1] = cv2.GaussianBlur(img_hsv[:, :, 1], (3, 3), 0)

    # Run CLAHE on the image luminance channel
    if clahe:
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6))
        filtered_lum = clahe.apply(img_lab[:, :, 0])
    else:
        filtered_lum = img_lab[:, :, 0]
    filtered_saturation = img_hsv[:, :, 1]

    img_sl = cv2.merge((filtered_saturation, filtered_lum))

    # Initialize an empty saliency map
    saliency_map = np.zeros_like(image[:, :, 0], dtype=np.float32)

    blurred_saturation = cv2.GaussianBlur(img_sl[:, :, 1], (5, 5), 0)
    blurred_lum = cv2.GaussianBlur(img_sl[:, :, 1], (5, 5), 0)

    # Calculate the center and surround regions for each channel
    center_lum = cv2.GaussianBlur(blurred_lum, (3, 3), 2)
    surround_lum = blurred_lum - center_lum

    center_saturation = cv2.GaussianBlur(blurred_saturation, (3, 3), 1)
    surround_saturation = blurred_saturation - center_saturation

    # Normalize the values to the range [0, 255]
    center_lum = cv2.normalize(center_lum, None, 0, 255, cv2.NORM_MINMAX)
    surround_lum = cv2.normalize(surround_lum, None, 0, 255, cv2.NORM_MINMAX)

    center_saturation = cv2.normalize(center_saturation, None, 0, 255, cv2.NORM_MINMAX)
    surround_saturation = cv2.normalize(surround_saturation, None, 0, 255, cv2.NORM_MINMAX)

    # Combine the center and surround regions for each channel
    # Reduce the weight of the saturation channel
    saliency_lum = 1.2 * (center_lum - surround_lum)
    saliency_saturation = (center_saturation - surround_saturation) / 4

    # Combine the saliency maps into the final saliency map
    saliency_level = (saliency_saturation + saliency_lum) / 2

    # Resize the saliency map to the original image size
    saliency_level = cv2.resize(saliency_level, (image.shape[1], image.shape[0]))

    # Accumulate the saliency map at each level
    saliency_map += saliency_level

    # Normalize the final saliency map
    saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX)

    # Blur the saliency map to remove noise
    saliency_map = cv2.GaussianBlur(saliency_map, (3, 3), 0)

    if show:
        # Display the original image and the saliency map
        cv2.imshow("Original Image", image)
        cv2.imshow("Fine-Grained Saliency Map", saliency_map.astype(np.uint8))
        if save:
            cv2.imwrite("my_fine_grained_saliency_map.jpg", saliency_map.astype(np.uint8))
            cv2.imwrite("my_original_image.jpg", image)

        cv2.waitKey(0)

    return saliency_map


def run_particle_detect_bulk(
    spec_removal: bool,
    scale_percent: int,
    images: list,
    out_path: Path,
    min_std: float = 8.0,
    block_size: int = 23,
    clahe: bool = False,
    show: bool = False,
):
    info(f"Processing {len(images)} images")
    for f in images:
        out_csv_file = out_path / f"{f.stem}.csv"
        out_image_file = out_path / f"{f.stem}.jpg" if show else None
        run_particle_detect(
            spec_removal,
            scale_percent,
            f.as_posix(),
            out_csv_file.as_posix(),
            out_image_file.as_posix() if out_image_file else None,
            min_std,
            block_size,
            clahe,
            show,
        )


def run_particle_detect(
    spec_removal: bool,
    scale_percent: int,
    in_image_file: str,
    out_csv_file: str,
    out_image_file: str = None,
    min_std: float = 8.0,
    block_size: int = 23,
    clahe: bool = False,
    show: bool = False,
    small: bool = False,
):
    """
    Run the detection algorithm on an saliency map image
    :param spec_removal: True to remove specular highlights
    :param scale_percent: Scale percentage to rescale the image before processing
    :param in_image_file: Input image filename
    :param clahe: True to run CLAHE on the luminance channel
    :param show: True to show the results
    :param small: True to target small objects (e.g. small fish, marine snow, etc)
    :param out_csv_file: The csv path to save the results
    :param out_image_file: (optional) The image filename to save the results
    :param min_std: minimum standard deviation of the contour to be considered
    :param block_size: block size for adaptive thresholding
    :return:
    """
    info(f"Processing {in_image_file}")
    image_path = Path(in_image_file)
    img_color = cv2.imread(image_path.as_posix())
    img_color_rescaled = rescale(img_color, scale_percent=scale_percent)

    if show:
        cv2.imshow("Original Rescaled", img_color_rescaled)
        if save:
            cv2.imwrite("my_original_rescaled.jpg", img_color_rescaled)
        cv2.waitKey(0)

    # Calculate the scale factor to rescale the detections back to the original image size
    scale_width, scale_height = (
        img_color.shape[0] / img_color_rescaled.shape[0],
        img_color.shape[1] / img_color_rescaled.shape[1],
    )

    if spec_removal:
        # Remove reflections
        img_spc = specularity_removal(img_color_rescaled)
        if show:
            cv2.imshow("Clean Rescaled", img_spc)
            if save:
                cv2.imwrite("my_clean_rescaled.jpg", img_spc)
            cv2.waitKey(0)
    else:
        img_spc = img_color_rescaled

    # Run the fine-grained saliency algorithm
    saliency_map = fine_grained_saliency(img_spc, clahe=clahe, show=show, small=small)

    # Extract blobs from the saliency map
    df, contour_img = extract_blobs(min_std, block_size, saliency_map, img_spc, show=show)

    info(f"Found {len(df)} blobs in {image_path}")

    if len(df) == 0:
        return

    if show or out_image_file:
        # Plot boxes on the input frame
        pred_list = df[["x", "y", "xx", "xy", "score", "saliency"]].values.tolist()
        for p in pred_list:
            # Draw a rectangle with yellow line borders of thickness of 3 px
            cv2.rectangle(img_spc, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (247, 108, 0), 3)
            # Add saliency score
            cv2.putText(img_spc, f"{int(p[5])}", (int(p[0]), int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        if show:
            cv2.imshow(f"Found {len(pred_list)} detections", img_spc)
            if save:
                cv2.imwrite("my_detections.jpg", img_spc)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if out_image_file:
            # Save the image with the contours drawn
            cv2.imwrite(out_image_file, img_spc)

    # Scale the detections back to the original image size
    df[["x", "xx", "w"]] *= scale_width
    df[["y", "xy", "h"]] *= scale_height
    df["image_path"] = image_path.as_posix()

    # Save the results to a csv file for later processing
    df.to_csv(out_csv_file, mode="w", header=True, index=False)


if __name__ == "__main__":
    # Read all images from the tests/data/kelpflow directory
    # Get the path of this file
    # test_path = Path(__file__).parent.parent.parent / 'tests' / 'data' / 'flirleft'
    # test_path = Path(__file__).parent.parent.parent / 'tests' / 'data' / 'isiis_single'
    # test_path = Path(__file__).parent.parent.parent / 'tests' / 'data' / 'bird'
    # test_path = Path(__file__).parent.parent.parent / 'tests' / 'data' / 'other'
    # test_path = Path(__file__).parent.parent.parent / 'tests' / 'data' / 'whale'
    # test_path = Path(__file__).parent.parent.parent / 'tests' / 'data' / 'front'
    # test_path = Path(__file__).parent.parent.parent / 'tests' / 'data' / 'kelpflow'
    # test_path = Path( __file__).parent.parent.parent / 'tests' / 'data' / 'glare'
    # test_path = Path(__file__).parent.parent.parent / 'tests' / 'data' / 'dolphin'
    # test_path = Path(__file__).parent.parent.parent / 'tests' / 'data' / 'all'
    # test_path = Path(__file__).parent.parent.parent / 'tests' / 'data' / 'otter'
    test_path = Path(__file__).parent.parent.parent / "tests" / "data" / "jelly"
    scale = 50  # Run at 1/2 scale
    out_path = Path(__file__).parent.parent.parent / "tests" / "data" / "out" / "jelly"
    out_path.mkdir(parents=True, exist_ok=True)

    # Count the images in the directory
    ext_glob = "*.JPG"
    num_images = len(list(test_path.glob(ext_glob)))

    # Do the work in a temporary directory
    tmp_dir = os.getenv("TMPDIR", tempfile.gettempdir())
    with tempfile.TemporaryDirectory(dir=tmp_dir) as temp_path:
        temp_path = Path(temp_path)

        for in_image_file in test_path.glob(ext_glob):
            run_saliency_detect(
                False,
                scale,
                in_image_file,
                (temp_path / f"{in_image_file.stem}.csv").as_posix(),
                (temp_path / f"{in_image_file.stem}.jpg").as_posix(),
                min_std=8.0,
                block_size=23,
                show=True,
            )

        # Copy the results to the output directory
        for csv in temp_path.glob("*.csv"):
            csv.rename(out_path / csv.name)
        for image in temp_path.glob("*.jpg"):
            image.rename(out_path / image.name)

    info(f"Processed {num_images} images. Results are in {out_path}")
