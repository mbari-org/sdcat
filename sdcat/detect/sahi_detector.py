# sdcat, Apache-2.0 license
# Filename: sdcat/ml/sahi.py
# Description:  SAHI detection models
import numpy as np
from pathlib import Path

import cv2
import pandas as pd
import torch
from sahi.predict import get_sliced_prediction

from sdcat.logger import info, exception
from sdcat.cluster.utils import rescale
from skimage.morphology import disk
from skimage.filters import rank


def local_equalize(image: np.ndarray, size: int = 3) -> np.ndarray:
    img = cv2.convertScaleAbs(image)  # convert to 8-bit
    footprint = disk(size)
    return rank.equalize(img, footprint=footprint)


def run_sahi_detect_bulk(scale_percent: int,
                         slice_width: int,
                         slice_height: int,
                         images: list,
                         csv_out_path: Path,
                         detection_model,
                         postprocess_match_metric: str,
                         overlap_width_ratio: float,
                         overlap_height_ratio: float,
                         allowable_classes: list = None,
                         class_agnostic: bool = False):
    info(f'Processing {len(images)} images')
    if not csv_out_path.is_dir():
        csv_out_path.mkdir(parents=True, exist_ok=True)
    for f in images:
        run_sahi_detect(scale_percent,
                        slice_width,
                        slice_height,
                        f,
                        (csv_out_path / f'{f.stem}.sahi.csv'),
                        detection_model,
                        postprocess_match_metric,
                        overlap_width_ratio,
                        overlap_height_ratio,
                        allowable_classes,
                        class_agnostic)


def run_sahi_detect(scale_percent: int,
                    slice_width: int,
                    slice_height: int,
                    image_path: Path,
                    csv_out_path: Path,
                    detection_model,
                    postprocess_match_metric: str = 'IOS',
                    overlap_width_ratio: float = 0.4,
                    overlap_height_ratio: float = 0.4,
                    allowable_classes: list = None,
                    class_agnostic: bool = False) -> pd.DataFrame:
    """
    Run the sahi detection model on an image
    :param scale_percent: percent to scale the image
    :param slice_width: slice size width
    :param slice_height: slice size height
    :param image_path: path to the image
    :param csv_out_path: output path for the detections
    :param image_color: color image
    :param detection_model: detection model
    :param postprocess_match_metric 'IOU' for intersection over union, 'IOS' for intersection over smaller area.
    :param overlap_width_ratio: overlap width ratio - amount of overlap between slices
    :param overlap_height_ratio: overlap height ratio - amount of overlap between slices
    :param allowable_classes: list of allowable classes
    :param class_agnostic: True if class agnostic
    :return: dataframe of detections
    """
    df = pd.DataFrame()
    # Abort if the image_path is not a file
    if not image_path.is_file():
        exception(f'Image path {image_path} is not a file')
        return df
    info(f'Processing {image_path}')
    img_color = cv2.imread(image_path.as_posix())
    img_color_rescaled = rescale(img_color, scale_percent=scale_percent)

    # Run image normalization on each color channel
    img_color_rescaled[:, :, 0] = cv2.equalizeHist(img_color_rescaled[:, :, 0])
    img_color_rescaled[:, :, 1] = cv2.equalizeHist(img_color_rescaled[:, :, 1])
    img_color_rescaled[:, :, 2] = cv2.equalizeHist(img_color_rescaled[:, :, 2])

    # Calculate the slice size
    if slice_width and slice_height:
        info(f'Using slice size width: {slice_width} and height: {slice_height}')
        slice_height = int(slice_height)
        slice_width = int(slice_width)
    else:
        # Let the auto-slice do its thing
        slice_width = None
        slice_height = None

    try:
        result = get_sliced_prediction(
            img_color_rescaled,
            detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            postprocess_type="NMM",
            postprocess_match_metric=postprocess_match_metric,
            perform_standard_pred=True,
            # Perform a standard prediction on top of sliced predictions to increase large object detection
            postprocess_class_agnostic=class_agnostic,  # If True, postprocess will ignore category ids.
        )

        predictions = []

        for obj in result.object_prediction_list:
            info(obj)
            if class_agnostic:
                obj.category.name = 'Unknown'
                obj.category.id = 0
                predictions.append(obj)
            if allowable_classes:
                if obj.category.name in allowable_classes:
                    predictions.append(obj)
            else:
                predictions.append(obj)

        if len(predictions) > 0:

            # Calculate the scale factor to rescale the detections back to the original image size
            scale_width, scale_height = img_color.shape[0] / img_color_rescaled.shape[0], img_color.shape[1] / \
                                        img_color_rescaled.shape[1]

            for p in predictions:
                box_width = p.bbox.maxx - p.bbox.minx
                box_height = p.bbox.maxy - p.bbox.miny
                area = box_width * box_height  # an approximation of the area
                score = p.score.value
                class_label = p.category.name
                # get the scores from the tensor
                if isinstance(score, torch.Tensor):
                    score = score.item()
                df = pd.concat([df, pd.DataFrame({
                    'image_path': image_path.as_posix(),
                    'class': class_label,
                    'score': score,
                    'area': area,
                    'saliency': -1,  # not computed - just a placeholder
                    'x': p.bbox.minx,
                    'y': p.bbox.miny,
                    'xx': p.bbox.maxx,
                    'xy': p.bbox.maxy,
                    'w': box_width,
                    'h': box_height},
                    index=[0])])

            # Remove duplicates, a duplicate has the same class and h and w
            df = df.drop_duplicates(subset=['class', 'h', 'w'])

            # Scale the detections back to the original image size
            df[['x', 'xx', 'w']] *= scale_width
            df[['y', 'xy', 'h']] *= scale_height

            # Write out the detections
            df.to_csv(csv_out_path.as_posix(), index=False)
    except Exception as e:
        exception(f"Error processing {image_path}: {e}")

    return df
