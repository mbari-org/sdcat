# sightwire, Apache-2.0 license
# Filename: cluster/commands.py
# Description:  Clustering commands

import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import click
import ephem
import pandas as pd
import pytz
import torch

import common_args
from config import config as cfg
from logger import info, err, warn
from sdcat.cluster.cluster import cluster_vits

@click.command('cluster', help='Cluster detections. See cluster --config-ini to override cluster defaults.')
@common_args.config_ini
@click.option('--det-dir', help='Input folder(s) with raw detection results', multiple=True)
@click.option('--save-dir',  help='Output directory to save clustered detection results')
@click.option('--device', help='Device to use.', type=int)
def run_cluster(det_dir, save_dir, device, config_ini):

    config = cfg.Config(config_ini)
    max_area = int(config('cluster', 'max_area'))
    min_area = int(config('cluster', 'min_area'))
    min_saliency = int(config('cluster', 'min_saliency'))
    min_samples = int(config('cluster', 'min_samples'))
    remove_corners = config('cluster', 'remove_corners')
    latitude = float(config('cluster', 'latitude'))
    longitude = float(config('cluster', 'longitude'))
    min_score = float(config('cluster', 'min_score'))
    min_similarity = float(config('cluster', 'min_similarity'))
    alpha = float(config('cluster', 'alpha'))
    cluster_selection_epsilon = float(config('cluster', 'cluster_selection_epsilon'))
    min_cluster_size = int(config('cluster', 'min_cluster_size'))
    model = config('cluster', 'model')

    if device:
        num_devices = torch.cuda.device_count()
        info(f'{num_devices} cuda devices available')
        info(f'Using device {device}')
        torch.cuda.set_device(device)

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
    df['cluster_id'] = -1  # -1 is the default value and means that the image is not in a cluster

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
        # A prefix for the output files to make sure the output is unique for each execution
        prefix = f'{model}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

        # Cluster the detections
        df_cluster = cluster_vits(prefix, model, df, save_dir, alpha, cluster_selection_epsilon, min_similarity, min_cluster_size, min_samples)

        # Merge the results with the original DataFrame
        df.update(df_cluster)

        # Save the clustered detections to a csv file and a copy of the config.ini file
        df.to_csv(save_dir / f'{prefix}_cluster_detections.csv', index=False, header=True)
        shutil.copy(Path(config_ini), save_dir / f'{prefix}_config.ini')
    else:
        warn(f'No detections found to cluster')