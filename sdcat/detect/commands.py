# sdcat, Apache-2.0 license
# Filename: sdcat/detect/commands.py
# Description:  Command line interface for running detection on images using SAHI and saliency detection algorithms.
import os
import shutil
from pathlib import Path

import click
from tqdm import tqdm
import modin.pandas as pd
import torch

from sdcat import common_args
from sdcat.config import config as cfg
from sdcat.config.config import default_config_ini
from sdcat.detect.filter_util import process_image
from sdcat.detect.model_util import create_model
from sdcat.detect.sahi_detector import run_sahi_detect_bulk, run_sahi_detect
from sdcat.detect.saliency_detector import run_saliency_detect, run_saliency_detect_bulk
from sdcat.logger import info, warn, create_logger_file

default_model = 'MBARI-org/megamidwater'


@click.command('detect',
               help=f'Detect objects in images.  By default, runs both SAHI and saliency blob detection and combines using NMS. '
                    f'Use --skip-sahi or --skip-saliency to exclude. '
                    f'See --config-ini to override detection defaults in {default_config_ini}.')
@common_args.config_ini
@common_args.start_image
@common_args.end_image
@click.option('--show', is_flag=True, help='Show algorithm steps.')
@click.option('--image-dir', required=True, help='Directory with images to run sliced detection.')
@click.option('--save-dir', required=True, help='Save detections to this directory.')
@click.option('--save-roi', is_flag=True, help='Save each region of interest/detection.')
@click.option('--roi-size', type=int, default=224, help='Rescale the region of interest.')
@click.option('--device', default='cpu', help='Device to use, e.g. cpu or cuda:0')
@click.option('--spec-remove', is_flag=True, help='Run specularity removal algorithm on the images before processing. '
                                                  '**CAUTION**this is slow. Set --scale-percent to < 100 to speed-up')
@click.option('--skip-sahi', is_flag=True, help='Skip sahi detection.')
@click.option('--skip-saliency', is_flag=True, help='Skip saliency detection.')
@click.option('--conf', default=0.1, type=float, help='Confidence threshold.')
@click.option('--scale-percent', default=80, type=int, help='Scaling factor to rescale the images before processing.')
@click.option('--model', default=default_model, help=f'Model to use. Defaults to {default_model}')
@click.option('--model-type', help=f'Type of model, e.g. yolov5, yolov8. Defaults to auto-detect.')
@click.option('--slice-size-width', type=int, help='Slice width size, leave blank for auto slicing')
@click.option('--slice-size-height', type=int, help='Slice height size, leave blank for auto slicing')
@click.option('--postprocess-match-metric', default='IOS', help='Postprocess match metric for NMS. postprocess_match_metric IOU for intersection over union, IOS for intersection over smaller area.')
@click.option('--overlap-width-ratio', type=float, default=0.4, help='Overlap width ratio for NMS')
@click.option('--overlap-height-ratio',type=float, default=0.4, help='Overlap height ratio for NMS')
@click.option('--clahe', is_flag=True, help='Run the CLAHE algorithm to contrast enhance before detection useful images with non-uniform lighting')

def run_detect(show: bool, image_dir: str, save_dir: str, save_roi:bool, roi_size: int, model: str, model_type:str,
               slice_size_width: int, slice_size_height: int, scale_percent: int,
               postprocess_match_metric: str, overlap_width_ratio: float, overlap_height_ratio: float,
               device: str, conf: float, skip_sahi: bool, skip_saliency: bool, spec_remove: bool,
               config_ini: str, clahe: bool, start_image: str, end_image: str):
    config = cfg.Config(config_ini)
    clahe = clahe if clahe else config('detect', 'clahe') == 'True'
    block_size = int(config('detect', 'block_size'))
    min_std = float(config('detect', 'min_std'))
    max_area = int(config('detect', 'max_area'))
    min_area = int(config('detect', 'min_area'))
    min_saliency = int(config('detect', 'min_saliency'))
    class_agnostic = config('detect', 'class_agnostic') == 'True'
    allowable_classes = []
    if not class_agnostic:
        allowable_classes = config('detect', 'allowable_classes')
        if len(allowable_classes) > 0:
            allowable_classes = allowable_classes.split(',')

    create_logger_file('detect')

    if not skip_sahi:
        if 'cuda' in device:
            num_devices = torch.cuda.device_count()
            info(f'{num_devices} cuda devices available')
            device_ = torch.device(device)
            torch.cuda.set_device(device_)
            device = 'cuda'
        detection_model = create_model(model, conf, device, model_type)

    if Path(model).is_dir():
        model = Path(model).name

    images_path = Path(image_dir)
    base_path = Path(save_dir) / model

    # clean-up any previous results
    if base_path.exists():
        info(f'Removing {base_path}')
        shutil.rmtree(base_path)

    if not skip_sahi:
        save_path_base = Path(save_dir) / model
    else:
        save_path_base = Path(save_dir)

    save_path_det_raw = save_path_base / 'det_raw' / 'csv'
    save_path_det_filtered = save_path_base / 'det_filtered' / 'csv'
    save_path_det_roi = save_path_base / 'det_filtered' / 'crops'
    save_path_viz = save_path_base / 'vizresults'

    if save_roi:
        save_path_det_roi.mkdir(parents=True, exist_ok=True)
        for f in save_path_det_roi.rglob('*'):
            os.remove(f)
    save_path_det_raw.mkdir(parents=True, exist_ok=True)
    save_path_det_filtered.mkdir(parents=True, exist_ok=True)
    save_path_viz.mkdir(parents=True, exist_ok=True)

    # Run on all images recursively
    images = [file for file in sorted(images_path.rglob('*'))
              if file.as_posix().endswith(('jpeg', 'png', 'jpg', 'JPEG', 'PNG', 'JPG', 'tif', 'tiff'))]

    if len(images) == 0:
        warn(f'No images found in {images_path}')
        return

    start_index = 0
    end_index = len(images)
    # If start_image is set, find the index of the start_image in the list of images
    if start_image:
        start_image = Path(start_image)
        start_index = next((i for i, image in enumerate(images) if start_image.name in image.name), None)
        if start_index is None:
            warn(f'Start image {start_image} not found in images')
            return

    # If end_image is set, find the index of the end_image in the list of images
    if end_image:
        end_image = Path(end_image)
        end_index = next((i for i, image in enumerate(images) if end_image.name in image.name), None)
        if end_index is None:
            warn(f'End image {end_image} not found in images')
            return

    images = images[start_index:end_index + 1]
    num_images = len(images)
    info(f'Found {num_images} images in {images_path}')

    if num_images == 0:
        return

    if not skip_saliency:
        # For development, run on a single image with
        # run_saliency_detect(spec_remove, scale_percent, images[0].as_posix(), (save_path_det_raw / f'{images[0].stem}.csv').as_posix(), clahe=clahe, show=True)
        df_args = pd.DataFrame([{
            "spec_remove": spec_remove,
            "scale_percent": scale_percent,
            "images": images[i:i + 1],  # a list with one image, like original
            "save_path_det_raw": save_path_det_raw,
            "min_std": min_std,
            "block_size": block_size,
            "clahe": clahe,
            "show": show
        } for i in range(0, num_images)])

        def run_saliency_wrapper(row):
            return run_saliency_detect_bulk(
                row.spec_remove,
                row.scale_percent,
                row.images,
                row.save_path_det_raw,
                row.min_std,
                row.block_size,
                row.clahe,
                row.show
            )
        info(f"Running saliency detection on {num_images} images...")
        df_args.apply(run_saliency_wrapper, axis=1)
    if device == 'cpu':
        if not skip_sahi:
            df_args = pd.DataFrame([{
                "scale_percent": scale_percent,
                "slice_size_width": slice_size_width,
                "slice_size_height": slice_size_height,
                "images": [image],  # wrapped in list to match original structure
                "save_path_det_raw": save_path_det_raw,
                "detection_model": detection_model,
                "postprocess_match_metric": postprocess_match_metric,
                "overlap_width_ratio": overlap_width_ratio,
                "overlap_height_ratio": overlap_height_ratio,
                "allowable_classes": allowable_classes,
                "class_agnostic": class_agnostic
            } for image in images])

            def run_sahi_wrapper(row):
                return run_sahi_detect_bulk(
                    row.scale_percent,
                    row.slice_size_width,
                    row.slice_size_height,
                    row.images,
                    row.save_path_det_raw,
                    row.detection_model,
                    row.postprocess_match_metric,
                    row.overlap_width_ratio,
                    row.overlap_height_ratio,
                    row.allowable_classes,
                    row.class_agnostic
                )

            info(f"Running SAHI detection on {len(images)} images...")
            df_args.apply(run_sahi_wrapper, axis=1)
    else:
        for f in tqdm(images):
            if not skip_saliency:
                run_saliency_detect(spec_remove,
                                    scale_percent,
                                    f.as_posix(),
                                    (save_path_det_raw / f'{f.stem}.csv').as_posix(),
                                    None,
                                    min_std,
                                    block_size,
                                    clahe,
                                    show)
            if not skip_sahi:
                run_sahi_detect(scale_percent,
                                slice_size_width,
                                slice_size_height,
                                f,
                                (save_path_det_raw / f'{f.stem}.sahi.csv'),
                                detection_model,
                                postprocess_match_metric,
                                overlap_width_ratio,
                                overlap_height_ratio,
                                allowable_classes,
                                class_agnostic)

    # Count the number of detections in all the images
    total_detections = sum(pd.read_csv(f).shape[0] for f in save_path_det_raw.rglob('*.csv'))
    total_images = len(images)

    df_args = pd.DataFrame([{
        "image": image,
        "save_path_base": save_path_base,
        "save_path_det_raw": save_path_det_raw,
        "save_path_det_filtered": save_path_det_filtered,
        "save_path_det_roi": save_path_det_roi,
        "save_path_viz": save_path_viz,
        "min_area": min_area,
        "max_area": max_area,
        "min_saliency": min_saliency,
        "class_agnostic": class_agnostic,
        "save_roi": save_roi,
        "roi_size": roi_size
    } for image in images])

    def process_image_wrapper(row):
        return process_image(
            row.image,
            row.save_path_base,
            row.save_path_det_raw,
            row.save_path_det_filtered,
            row.save_path_det_roi,
            row.save_path_viz,
            row.min_area,
            row.max_area,
            row.min_saliency,
            row.class_agnostic,
            row.save_roi,
            row.roi_size
        )

    info(f"Processing {len(images)} images")
    results = df_args.apply(process_image_wrapper, axis=1)
    total_filtered = sum(results)
    info(f'Found {total_detections} total localizations in {total_images} with {total_filtered} after NMS')
    info('Done')
