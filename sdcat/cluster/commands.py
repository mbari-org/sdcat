# sdcat, Apache-2.0 license
# Filename: cluster/commands.py
# Description:  Clustering commands
import json
import re
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import click
import ephem
import os
import modin.pandas as pd
import pytz
import torch
from PIL import Image

from sdcat import __version__ as sdcat_version
from sdcat.cluster.utils import filter_images, combine_csv
from sdcat import common_args
from sdcat.config import config as cfg
from sdcat.logger import info, err, warn
from sdcat.cluster.cluster import cluster_vits


@click.command('detections', help='Cluster detections. See cluster --config-ini to override cluster defaults.')
@common_args.config_ini
@common_args.start_image
@common_args.end_image
@common_args.use_tsne
@common_args.skip_visualization
@common_args.alpha
@common_args.cluster_selection_epsilon
@common_args.cluster_selection_method
@common_args.algorithm
@common_args.min_cluster_size
@common_args.min_sample_size
@common_args.batch_size
@click.option('--det-dir', help='Input folder(s) with raw detection results', multiple=True, required=True)
@click.option('--save-dir', help='Output directory to save clustered detection results', required=True)
@click.option('--device', help='Device to use, e.g. cpu or cuda:0', type=str, default='cpu')
@click.option('--use-vits', help='Set to using the predictions from the vits cluster model', is_flag=True)
@click.option('--weighted-score', help='Weigh for the score in the predictions from the vits model with the detection score', type=bool, default=False)
def run_cluster_det(det_dir, save_dir, device, use_vits, weighted_score, config_ini, alpha, cluster_selection_epsilon, cluster_selection_method, algorithm, min_cluster_size, min_sample_size, batch_size, start_image, end_image, use_tsne, skip_visualization):
    config = cfg.Config(config_ini)
    max_area = int(config('cluster', 'max_area'))
    min_area = int(config('cluster', 'min_area'))
    min_saliency = int(config('cluster', 'min_saliency'))
    alpha = alpha if alpha else float(config('cluster', 'alpha'))
    min_cluster_size = min_cluster_size if min_cluster_size else int(config('cluster', 'min_cluster_size'))
    min_samples = min_sample_size if min_sample_size else int(config('cluster', 'min_samples'))
    cluster_selection_epsilon = cluster_selection_epsilon if cluster_selection_epsilon else float(config('cluster','cluster_selection_epsilon'))
    cluster_selection_method = cluster_selection_method if cluster_selection_method else config('cluster', 'cluster_selection_method')
    algorithm = algorithm if algorithm else config('cluster', 'algorithm')
    remove_corners = config('cluster', 'remove_corners') == 'True'
    remove_bad_images = config('cluster', 'remove_bad_images') == 'True'
    latitude = float(config('cluster', 'latitude'))
    longitude = float(config('cluster', 'longitude'))
    min_score = float(config('cluster', 'min_score'))
    min_similarity = float(config('cluster', 'min_similarity'))
    model = config('cluster', 'model')

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_files = []
    for d in det_dir:
        info(f'Searching in {d} for detection csv files')
        d_path = Path(d)
        if not d_path.exists():
            err(f'Input path {d} does not exist.')
            return

        csv_files.extend(d_path.rglob('*.csv'))

    info(f'Found {len(csv_files)} detection files in {det_dir}')

    crop_path = save_dir / 'crops'
    crop_path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = combine_csv(csv_files, Path(temp_dir), crop_path, start_image, end_image)

        info('Loading detections')
        df = pd.read_csv(output_file, sep=',')

        info(f'Found {len(df)} detections in {det_dir}')

        if len(df) == 0:
            info(f'No detections found in {det_dir}')
            return

        # Check if the image_path column is empty
        if df['image_path'].isnull().values.any():
            err(f'Found {df["image_path"].isnull().sum()} detections with no image_path')
            return

        # Sort the dataframe by image_path to make sure the images are in order for start_image and end_image filtering
        df = df.sort_values(by='image_path')

        # Filter the dataframe to only include images in the start_image and end_image range
        df = filter_images(min_area, max_area, min_saliency, min_score, df)

        # Add in a column for the unique crop name for each detection with a unique id
        # create a unique uuid based on the md5 hash of the box in the row
        df['crop_path'] = df.apply(lambda
                                       row: f"{row['crop_root']}/{uuid.uuid5(uuid.NAMESPACE_DNS, str(row['x']) + str(row['y']) + str(row['xx']) + str(row['xy']))}.png",
                                   axis=1)

        # Add in a column for the unique crop name for each detection with a unique id
        df['cluster'] = -1  # -1 is the default value and means that the image is not in a cluster

        # Remove small or large detections before clustering
        size_before = len(df)
        info(f'Searching through {size_before} detections')
        df = df[(df['area'] > min_area) & (df['area'] < max_area)]
        size_after = len(df)
        info(f'Removed {size_before - size_after} detections that were too large or too small')

        def within_1_percent_of_corners(row):
            threshold = 0.01  # 1% threshold

            x, y, xx, yy = row['x'], row['y'], row['xx'], row['xy']

            # Check if any of the coordinates are within 1% of the image corners
            return (
                    (0 <= x <= threshold or 1 - threshold <= x <= 1) or
                    (0 <= y <= threshold or 1 - threshold <= y <= 1) or
                    (0 <= xx <= threshold or 1 - threshold <= xx <= 1) or
                    (0 <= yy <= threshold or 1 - threshold <= yy <= 1)
            )

        if remove_corners:
            # Remove any detections that are in any corner of the image
            size_before = len(df)
            df = df[~df.apply(within_1_percent_of_corners, axis=1)]
            size_after = len(df)
            info(f'Removed {size_before - size_after} detections that were in the corners of the image')

        # Replace any NaNs with 0
        df.fillna(0)

        # Print the first 5 rows of the dataframe
        info(df.head(5))

        if len(df) > 0:
            # Replace / with _ in the model name
            model_machine_friendly = model.replace('/', '_')

            # A prefix for the output files to make sure the output is unique for each execution
            prefix = f'{model_machine_friendly}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

            # Cluster the detections
            summary = cluster_vits(prefix, model, df, save_dir, alpha, cluster_selection_epsilon, cluster_selection_method, algorithm,
                                      min_similarity, min_cluster_size, min_samples, device, use_tsne=use_tsne,
                                      skip_visualization=skip_visualization, roi=False, weighted_score=weighted_score, use_vits=use_vits,
                                      remove_bad_images=remove_bad_images, batch_size=batch_size)

            if summary is None:
                err(f'No summary returned from clustering')
                return

            # Add more detail to the summary specific to detections
            summary['sdcat']['version'] = sdcat_version
            summary['dataset']['input'] = det_dir
            summary['dataset']['image_resolution'] = f"{df['image_width'].iloc[0]}x{df['image_height'].iloc[0]} pixels"
            summary['dataset']['image_count'] = len(df)

            with open(save_dir / f'{prefix}_summary.json', 'w') as f:
                json.dump(summary, f, indent=4)
            info(f"Summary saved to {save_dir / f'{prefix}_summary.json'}")

            # Save a copy of the config.ini file
            shutil.copy(Path(config_ini), save_dir / f'{prefix}_config.ini')
            info(f'Config saved to {save_dir / f"{prefix}_config.ini"}')
        else:
            warn(f'No detections found to cluster')

@click.command('roi', help='Cluster roi. See cluster --config-ini to override cluster defaults.')
@common_args.config_ini
@common_args.use_tsne
@common_args.skip_visualization
@common_args.alpha
@common_args.cluster_selection_epsilon
@common_args.cluster_selection_method
@common_args.algorithm
@common_args.min_cluster_size
@common_args.min_sample_size
@common_args.batch_size
@click.option('--roi-dir', help='Input folder(s) with raw ROI images', multiple=True, required=True)
@click.option('--save-dir', help='Output directory to save clustered detection results', required=True)
@click.option('--device', help='Device to use, e.g. cpu or cuda:0', type=str)
@click.option('--use-vits', help='Set to using the predictions from the vits cluster model', is_flag=True)
def run_cluster_roi(roi_dir, save_dir, device, use_vits, config_ini, alpha, cluster_selection_epsilon, cluster_selection_method, algorithm, min_cluster_size, min_sample_size, batch_size, use_tsne, skip_visualization):
    config = cfg.Config(config_ini)
    max_area = int(config('cluster', 'max_area'))
    min_area = int(config('cluster', 'min_area'))
    min_saliency = int(config('cluster', 'min_saliency'))
    alpha = alpha if alpha else float(config('cluster', 'alpha'))
    min_cluster_size = min_cluster_size if min_cluster_size else int(config('cluster', 'min_cluster_size'))
    min_samples = min_sample_size if min_sample_size else int(config('cluster', 'min_samples'))
    cluster_selection_epsilon = cluster_selection_epsilon if cluster_selection_epsilon else float(
        config('cluster', 'cluster_selection_epsilon'))
    cluster_selection_method = cluster_selection_method if cluster_selection_method else config('cluster',
                                                                                                'cluster_selection_method')
    algorithm = algorithm if algorithm else config('cluster', 'algorithm')
    remove_bad_images = config('cluster', 'remove_bad_images') == 'True'
    min_score = float(config('cluster', 'min_score'))
    min_similarity = float(config('cluster', 'min_similarity'))
    model = config('cluster', 'model')

    if device:
        num_devices = torch.cuda.device_count()
        info(f'{num_devices} cuda devices available')
        info(f'Using device {device}')
        if 'cuda' in device:
            device_num = device.split(':')[-1]
            info(f'Setting CUDA_VISIBLE_DEVICES to {device_num}')
            torch.cuda.set_device(device)
            os.environ['CUDA_VISIBLE_DEVICES'] = device_num

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Grab all images from the input directories
    supported_extensions = ['.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG']
    images = []

    for r in roi_dir:
        roi_path = Path(r)
        for ext in supported_extensions:
            images.extend(list(roi_path.rglob(f'*{ext}')))

    # Create a dataframe to store the combined data in an image_path column in sorted order
    df = pd.DataFrame({'image_path': [str(p) for p in images]})

    info(f'Found {len(df)} detections in {roi_dir}')

    if len(df) == 0:
        info(f'No detections found in {roi_dir}')
        return

    # Sort the dataframe by image_path to make sure the images are in order for start_image and end_image filtering
    df = df.sort_values(by='image_path')

    # Add the image_width and image_height columns to the dataframe
    # Assuming all images are the same size, we can just get the size of the first image and use that for all images
    im_size = Image.open(df['image_path'].iloc[0]).size
    df['image_height'] = im_size[1]
    df['image_width'] = im_size[0]

    # Create a unique crop name for each detection with a unique id
    crop_path = save_dir / 'crops'
    crop_path.mkdir(parents=True, exist_ok=True)
    df['crop_path'] = df.apply(lambda row:
                               f'{crop_path}/{Path(row["image_path"]).stem}.png',
                               axis=1)

    # Copy the images to the crop path directory. Images may be cleaned so we want to duplicate them here.
    for index, row in df.iterrows():
        shutil.copy(row['image_path'], row['crop_path'])

    df = filter_images(min_area, max_area, min_saliency, min_score, df)

    df['cluster'] = -1  # -1 is the default value and means that the image is not in a cluster
    df['class'] = 'Unknown'
    df['class_s'] = 'Unknown'
    df['score'] = 0.
    df['score_s'] = 0.

    # Replace any NaNs with 0
    df.fillna(0)

    # Print the first 5 rows of the dataframe
    info(df.head(5))

    if len(df) > 0:
        model_machine_friendly = model.replace('/', '_')

        # A prefix for the output files to make sure the output is unique for each execution
        prefix = f'{model_machine_friendly}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

        # Cluster the detections
        summary = cluster_vits(prefix, model, df, save_dir, alpha, cluster_selection_epsilon, cluster_selection_method, algorithm,
                                  min_similarity, min_cluster_size, min_samples, device,
                                  use_tsne=use_tsne, weighted_score=False, use_vits=use_vits,
                                  skip_visualization=skip_visualization,  roi=True,
                                  remove_bad_images=remove_bad_images, batch_size=batch_size)

        # Add more detail to the summary specific to ROIs
        summary['sdcat']['version'] = sdcat_version
        summary['dataset']['roi'] = True
        summary['dataset']['input'] = roi_dir
        summary['dataset']['image_count'] = len(df)

        with open(save_dir / f'{prefix}_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
        info(f'Summary saved to {save_dir / f"{prefix}_summary.json"}')

        # Save a copy of the config.ini file
        shutil.copy(Path(config_ini), save_dir / f'{prefix}_config.ini')
        info(f'Config saved to {save_dir / f"{prefix}_config.ini"}')
    else:
        warn(f'No detections found to cluster')
