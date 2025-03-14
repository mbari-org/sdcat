import hashlib
import multiprocessing
import os
import shutil
import uuid
from pathlib import Path

import click
from tqdm import tqdm
import cv2
import pandas as pd
import torch
from sahi.postprocess.combine import nms

from sdcat import common_args
from sdcat.cluster.utils import crop_square_image
from sdcat.config import config as cfg
from sdcat.config.config import default_config_ini
from sdcat.detect.model_util import create_model
from sdcat.detect.sahi_detector import run_sahi_detect_bulk, run_sahi_detect
from sdcat.detect.saliency_detector import run_saliency_detect, run_saliency_detect_bulk
from sdcat.logger import exception, info, warn, create_logger_file

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

    num_processes = min(multiprocessing.cpu_count(), num_images)
    if not skip_saliency:
        # run_saliency_detect(spec_remove, scale_percent, images[0].as_posix(), (save_path_det_raw / f'{images[0].stem}.csv').as_posix(), clahe=clahe, show=True)
        # Do the work in parallel to speed up the processing on multicore machines
        info(f'Using {num_processes} processes to compute {num_images} images 10 at a time ...')
        # # Run multiple processes in parallel num_cpu images at a time
        with multiprocessing.Pool(num_processes) as pool:
            args = [(spec_remove,
                     scale_percent,
                     images[i:i + 1],
                     save_path_det_raw,
                     min_std,
                     block_size,
                     clahe,
                     show)
                    for i in range(0, num_images, 1)]
            pool.starmap(run_saliency_detect_bulk, args)
            pool.close()
    if device == 'cpu':
        with multiprocessing.Pool(num_processes) as pool:
            if not skip_sahi:
                # Run sahi detection on each image
                args = [(scale_percent,
                         slice_size_width,
                         slice_size_height,
                         [image],
                         save_path_det_raw,
                         detection_model,
                         postprocess_match_metric,
                         overlap_width_ratio,
                         overlap_height_ratio,
                         allowable_classes,
                         class_agnostic)
                        for image in images]
                pool.starmap(run_sahi_detect_bulk, args)
                pool.close()
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

    # Combine all the detections into a single dataframe per image
    for f in tqdm(images):
        # Get the width and height of the image
        img_color = cv2.imread(f.as_posix())
        height, width = img_color.shape[:2]

        # Dataframe to store all the csv files for this image
        df_combined = pd.DataFrame()

        # Path to save final csv file with the detections
        pred_out_csv = save_path_det_filtered / f'{f.stem}.csv'

        for csv_file in save_path_det_raw.rglob(f'{f.stem}*csv'):
            # Read in the csv file and add it to the combined dataframe
            df = pd.read_csv(csv_file, sep=',')
            df_combined = pd.concat([df_combined, df])

        if len(df_combined) == 0:
            warn(f'No detections found in {f}')
            continue

        size_before = len(df_combined)
        df_combined = df_combined[(df_combined['area'] > min_area) & (df_combined['area'] < max_area)]
        size_after = len(df_combined)
        info(f'Removed {size_before - size_after} detections that were too large or too small')

        size_before = len(df_combined)
        # allow negative saliency which means the saliency was not included - this is the case if running SAHI algorithm only and no saliency blob detect
        df_combined = df_combined[(df_combined['saliency'] > min_saliency) | (df_combined['saliency'] == -1)]
        size_after = len(df_combined)
        info(f'Removed {size_before - size_after} detections that were low saliency')

        if len(df_combined) == 0:
            warn(f'No detections found in {f}')
            continue

        # Convert the x, y, xx, xy, score columns to a list of Tensors that is Shape: [num_boxes,5].
        pred_list = df_combined[['x', 'y', 'xx', 'xy', 'score']].values.tolist()
        pred_list = torch.tensor(pred_list)  # convert the list of Tensors to a list of Tensors

        # Clean up the predictions using NMS
        info(f'Running NMS on {len(df_combined)} predictions')
        nms_pred_idx = nms(pred_list, 'IOU', 0.1)
        info(f'{len(nms_pred_idx)} predictions found after NMS')

        # Filter the original DataFrame based on the indices kept by NMS and keep the saliency and area columns
        df_final = df_combined.iloc[nms_pred_idx].reset_index(drop=True)
        df_final['saliency'] = df_combined['saliency'].iloc[nms_pred_idx].reset_index(drop=True)
        df_final['area'] = df_combined['area'].iloc[nms_pred_idx].reset_index(drop=True)

        # Plot boxes on the input frame
        pred_list = df_final[['x', 'y', 'xx', 'xy', 'score', 'class']].values.tolist()
        for p in pred_list:
            if class_agnostic:
                img_color = cv2.rectangle(img_color,
                                          (int(p[0]), int(p[1])),
                                          (int(p[2]), int(p[3])), (81, 12, 51), 3)
                img_color = cv2.putText(img_color, f'{p[5]} {p[4]:.2f}', (int(p[0]), int(p[1])),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                # Color based on the first 5 letters of the label so that the same label always gets the same color
                md5_hash = hashlib.md5(p[5].encode())
                hex_color = md5_hash.hexdigest()[:6]
                r, g, b = (int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:], 16))
                color = (r % 256, g % 256, b % 256)
                img_color = cv2.rectangle(img_color,
                                          (int(p[0]), int(p[1])),
                                          (int(p[2]), int(p[3])), color, 3)
                img_color = cv2.putText(img_color, p[5], (int(p[0]), int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2,
                                        cv2.LINE_AA)
        info(f'Saving visualization to {save_path_viz / f"{f.stem}.jpg"}')
        cv2.imwrite(f"{save_path_viz / f'{f.stem}.jpg'}", img_color)

        df_final['cluster'] = -1
        df_final['image_path'] = f.as_posix()
        df_final['image_width'] = width
        df_final['image_height'] = height

        # Normalize the predictions to 0-1.
        df_final['x'] = df_final['x'] / width
        df_final['y'] = df_final['y'] / height
        df_final['xx'] = df_final['xx'] / width
        df_final['xy'] = df_final['xy'] / height
        df_final['w'] = (df_final['xx'] - df_final['x'])
        df_final['h'] = (df_final['xy'] - df_final['y'])

        if save_roi:
            # Add in a column for the unique crop name for each detection with a unique id
            # create a unique uuid based on the md5 hash of the box in the row
            df_final['crop_path'] = df_final.apply(lambda
                                           row: f"{save_path_det_roi}/{uuid.uuid5(uuid.NAMESPACE_DNS, str(row['x']) + str(row['y']) + str(row['xx']) + str(row['xy']))}.png",
                                       axis=1)

            num_processes = min(multiprocessing.cpu_count(), len(df_final))
            # Crop and squaring the images in parallel using multiprocessing to speed up the processing
            info(f'Cropping {len(df_final)} detections in parallel using {num_processes} processes...')
            with multiprocessing.Pool(num_processes) as pool:
                args = [(row, roi_size) for index, row in df_final.iterrows()]
                pool.starmap(crop_square_image, args)

        info(f'Found {len(pred_list)} total localizations in {f} with {len(df_combined)} after NMS')
        info(f'Slice width: {slice_size_width} height: {slice_size_height}')

        # Save DataFrame to CSV file including image_width and image_height columns
        info(f'Detections saved to {pred_out_csv}')
        df_final.to_csv(pred_out_csv.as_posix(), index=False, header=True)
        if save_roi: info(f"ROI crops saved in {save_path_det_roi}")

        save_stats = save_path_base / 'stats.txt'
        with save_stats.open('w') as sf:
            sf.write(f"Statistics for {f}:\n")
            sf.write("----------------------------------\n")
            sf.write(f"Total number of bounding boxes: {df_combined.shape[0]}\n")
            sf.write(
                f"Total number of images with (bounding box) detections found: {df_combined['image_path'].nunique()}\n")
            sf.write(
                f"Average number of bounding boxes per image: {df_combined.shape[0] / df_combined['image_path'].nunique()}\n")
            sf.write(f"Average width of bounding boxes: {df_combined['w'].mean() * width}\n")
            sf.write(f"Average height of bounding boxes: {df_combined['h'].mean() * height}\n")
            sf.write(f"Average area of bounding boxes: {df_combined['area'].mean()}\n")
            sf.write(f"Average score of bounding boxes: {df_combined['score'].mean()}\n")
            sf.write(f"Average saliency of bounding boxes: {df_combined['saliency'].mean()}\n")
        info('Done')
