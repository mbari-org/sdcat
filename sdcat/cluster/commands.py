# sdcat, Apache-2.0 license
# Filename: cluster/commands.py
# Description:  Clustering commands

import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import click
import ephem
import os
import pandas as pd
import pytz
import torch
from PIL import Image

from sdcat import common_args
from sdcat.config import config as cfg
from sdcat.logger import info, err, warn
from sdcat.cluster.cluster import cluster_vits


@click.command('detections', help='Cluster detections. See cluster --config-ini to override cluster defaults.')
@common_args.config_ini
@common_args.start_image
@common_args.end_image
@common_args.use_tsne
@common_args.alpha
@common_args.cluster_selection_epsilon
@common_args.cluster_selection_method
@common_args.min_cluster_size
@click.option('--det-dir', help='Input folder(s) with raw detection results', multiple=True, required=True)
@click.option('--save-dir', help='Output directory to save clustered detection results', required=True)
@click.option('--device', help='Device to use, e.g. cpu or cuda:0', type=str)
def run_cluster_det(det_dir, save_dir, device, config_ini, alpha, cluster_selection_epsilon, cluster_selection_method, min_cluster_size, start_image, end_image, use_tsne):
    config = cfg.Config(config_ini)
    max_area = int(config('cluster', 'max_area'))
    min_area = int(config('cluster', 'min_area'))
    min_saliency = int(config('cluster', 'min_saliency'))
    min_samples = int(config('cluster', 'min_samples'))
    alpha = alpha if alpha else float(config('cluster', 'alpha'))
    min_cluster_size = min_cluster_size if min_cluster_size else int(config('cluster', 'min_cluster_size'))
    cluster_selection_epsilon = cluster_selection_epsilon if cluster_selection_epsilon else float(config('cluster','cluster_selection_epsilon'))
    cluster_selection_method = cluster_selection_method if cluster_selection_method else config('cluster', 'cluster_selection_method')
    remove_corners = config('cluster', 'remove_corners')
    latitude = float(config('cluster', 'latitude'))
    longitude = float(config('cluster', 'longitude'))
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

    detections = []
    for d in det_dir:
        info(f'Searching in {d}')
        d_path = Path(d)
        if not d_path.exists():
            err(f'Input path {d} does not exist.')
            return

        detections.extend(d_path.rglob('*.csv'))

    # Combine all the detections into a single dataframe
    df = pd.DataFrame()

    crop_path = save_dir / 'crops'
    crop_path.mkdir(parents=True, exist_ok=True)

    for d in detections:
        df_new = pd.read_csv(d, sep=',')

        # concatenate to the df dataframe
        df = pd.concat([df, df_new], ignore_index=True)

    # Remove any duplicate rows; duplicates have the same .x, .y, .xx, .xy,
    df = df.drop_duplicates(subset=['x', 'y', 'xx', 'xy'])

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

    # If start_image is set, find the index of the start_image in the list of images
    if start_image:
        start_image = Path(start_image)
        start_image = start_image.resolve()
        start_image = start_image.stem
        start_image_index = df[df['image_path'].str.contains(start_image)].index[0]
        df = df.iloc[start_image_index:]

    # If end_image is set, find the index of the end_image in the list of images
    if end_image:
        end_image = Path(end_image)
        end_image = end_image.resolve()
        end_image = end_image.stem
        end_image_index = df[df['image_path'].str.contains(end_image)].index[0]
        df = df.iloc[:end_image_index]

    # Filter by saliency, area, score or day/night
    size_before = len(df)
    if 'saliency' in df.columns:
        df = df[(df['saliency'] > min_saliency) | (df['saliency'] == -1)]
    if 'area' in df.columns:
        df = df[(df['area'] > min_area) & (df['area'] < max_area)]
    if 'score' in df.columns:
        df = df[(df['score'] > min_score)]
    size_after = len(df)
    info(f'Removed {size_before - size_after} detections outside of area, saliency, or too low scoring')

    # Add in a column for the unique crop name for each detection with a unique id
    # create a unique uuid based on the md5 hash of the box in the row
    df['crop_path'] = df.apply(lambda
                                   row: f"{crop_path}/{uuid.uuid5(uuid.NAMESPACE_DNS, str(row['x']) + str(row['y']) + str(row['xx']) + str(row['xy']))}.png",
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

    pattern_date1 = re.compile(r'(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z')  # 20161025T184500Z
    pattern_date2 = re.compile(r'(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z\d*mF*')
    pattern_date3 = re.compile(r'(\d{2})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z')  # 161025T184500Z
    pattern_date4 = re.compile(r'(\d{2})-(\d{2})-(\d{2})T(\d{2})_(\d{2})_(\d{2})-')  # 16-06-06T16_04_54
    pattern_number = re.compile(r'^DSC(\d+)\.JPG')

    # Grab any additional metadata from the image name, e.g. depth, yearday, number, day/night
    depth = {}
    yearday = {}
    number = {}
    day_flag = {}
    observer = ephem.Observer()
    observer.lat = latitude
    observer.lon = longitude

    def is_day(utc_dt):
        observer.date = utc_dt
        sun = ephem.Sun(observer)
        if sun.alt > 0:
            return 1
        return 0

    for index, row in sorted(df.iterrows()):
        image_name = Path(row.image_path).name
        for depth_str in ['50m', '100m', '200m', '300m', '400m', '500m', '299m', '250m', '150m', '199m']:
            if depth_str in image_name:
                depth[index] = int(depth_str.split('m')[0])
        if pattern_date1.search(image_name):
            match = pattern_date1.search(image_name).groups()
            year, month, day, hour, minute, second = map(int, match)
            dt = datetime(year, month, day, hour, minute, second, tzinfo=pytz.utc)
            yearday[index] = int(dt.strftime("%j")) - 1
            day_flag[index] = is_day(dt)
        if pattern_date2.search(image_name):
            match = pattern_date2.search(image_name).groups()
            year, month, day, hour, minute, second = map(int, match)
            dt = datetime(year, month, day, hour, minute, second, tzinfo=pytz.utc)
            yearday[index] = int(dt.strftime("%j")) - 1
            day_flag[index] = is_day(dt)
        if pattern_date3.search(image_name):
            match = pattern_date3.search(image_name).groups()
            year, month, day, hour, minute, second = map(int, match)
            dt = datetime(year, month, day, hour, minute, second, tzinfo=pytz.utc)
            yearday[index] = int(dt.strftime("%j")) - 1
            day_flag[index] = is_day(dt)
        if pattern_date4.search(image_name):
            match = pattern_date4.search(image_name).groups()
            year, month, day, hour, minute, second = map(int, match)
            dt = datetime(year, month, day, hour, minute, second, tzinfo=pytz.utc)
            yearday[index] = int(dt.strftime("%j")) - 1
            day_flag[index] = is_day(dt)
        if pattern_number.search(image_name):
            match = pattern_number.match(image_name)
            numeric = int(match.group(1))
            number[index] = numeric

    # Add the depth, yearday, day, and night columns to the dataframe if they exist
    if len(depth) > 0:
        df['depth'] = depth
        df['depth'] = df['depth'].astype(int)
    if len(yearday) > 0:
        df['yearday'] = yearday
        df['yearday'] = df['yearday'].astype(int)
    if len(number) > 0:
        df['frame'] = number
    if len(day_flag) > 0:
        df['day'] = day_flag
        df['day'] = df['day'].astype(int)

    # Filter by day/night
    # size_before = len(df)
    # df = df[df['day'] == 1]
    # size_after = len(df)
    # info(f'Removed {size_before - size_after} detections that were at night')

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
        df_cluster = cluster_vits(prefix, model, df, save_dir, alpha, cluster_selection_epsilon, cluster_selection_method,
                                  min_similarity, min_cluster_size, min_samples, use_tsne)

        # Merge the results with the original DataFrame
        df.update(df_cluster)

        # Save the clustered detections to a csv file and a copy of the config.ini file
        df.to_csv(save_dir / f'{prefix}_cluster_detections.csv', index=False, header=True)
        shutil.copy(Path(config_ini), save_dir / f'{prefix}_config.ini')
    else:
        warn(f'No detections found to cluster')
 
@click.command('roi', help='Cluster roi. See cluster --config-ini to override cluster defaults.')
@common_args.config_ini
@common_args.use_tsne
@common_args.alpha
@common_args.cluster_selection_epsilon
@common_args.cluster_selection_method
@common_args.min_cluster_size
@click.option('--roi-dir', help='Input folder(s) with raw ROI images', multiple=True, required=True)
@click.option('--save-dir', help='Output directory to save clustered detection results', required=True)
@click.option('--device', help='Device to use, e.g. cpu or cuda:0', type=str)
def run_cluster_roi(roi_dir, save_dir, device, config_ini, alpha, cluster_selection_epsilon, cluster_selection_method, min_cluster_size, use_tsne):
    config = cfg.Config(config_ini)
    min_samples = int(config('cluster', 'min_samples'))
    alpha = alpha if alpha else float(config('cluster', 'alpha'))
    min_cluster_size = min_cluster_size if min_cluster_size else int(config('cluster', 'min_cluster_size'))
    cluster_selection_epsilon = cluster_selection_epsilon if cluster_selection_epsilon else float(config('cluster','cluster_selection_epsilon'))
    cluster_selection_method = cluster_selection_method if cluster_selection_method else config('cluster', 'cluster_selection_method')
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
    df = pd.DataFrame()
    df['image_path'] = images

    # Convert the image_path column to a string
    df['image_path'] = df['image_path'].astype(str)

    info(f'Found {len(df)} detections in {roi_dir}')

    if len(df) == 0:
        info(f'No detections found in {roi_dir}')
        return

    # Sort the dataframe by image_path to make sure the images are in order for start_image and end_image filtering
    df = df.sort_values(by='image_path')

    # Add the image_width and image_height columns to the dataframe
    for index, row in df.iterrows():
        im_size = Image.open(row['image_path']).size
        df.at[index, 'image_width'] = im_size[0]
        df.at[index, 'image_height'] = im_size[1]
    df['image_width'] = df['image_width'].astype(int)
    df['image_height'] = df['image_height'].astype(int)

    # Create a unique crop name for each detection with a unique id
    crop_path = save_dir / 'crops'
    crop_path.mkdir(parents=True, exist_ok=True)
    df['crop_path'] = df.apply(lambda row:
                               f'{crop_path}/{Path(row["image_path"]).stem}.png',
                               axis=1)

    # Add in a column for the unique crop name for each detection with a unique id
    df['cluster'] = -1  # -1 is the default value and means that the image is not in a cluster

    # Replace any NaNs with 0
    df.fillna(0)

    # Print the first 5 rows of the dataframe
    info(df.head(5))

    if len(df) > 0:
        model_machine_friendly = model.replace('/', '_')

        # A prefix for the output files to make sure the output is unique for each execution
        prefix = f'{model_machine_friendly}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

        # Cluster the detections
        df_cluster = cluster_vits(prefix, model, df, save_dir, alpha, cluster_selection_epsilon, cluster_selection_method,
                                  min_similarity, min_cluster_size, min_samples, use_tsne, roi=True)

        # Merge the results with the original DataFrame
        df.update(df_cluster)

        # Save the clustered detections to a csv file and a copy of the config.ini file
        df.to_csv(save_dir / f'{prefix}_cluster_detections.csv', index=False, header=True)
        shutil.copy(Path(config_ini), save_dir / f'{prefix}_config.ini')
    else:
        warn(f'No detections found to cluster')
