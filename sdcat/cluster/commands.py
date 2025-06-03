# sdcat, Apache-2.0 license
# Filename: cluster/commands.py
# Description:  Clustering commands to cluster both detections and ROIs
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
from PIL import Image
from tqdm import tqdm

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
@common_args.use_pca
@common_args.skip_visualization
@common_args.alpha
@common_args.cluster_selection_epsilon
@common_args.cluster_selection_method
@common_args.algorithm
@common_args.min_cluster_size
@common_args.min_sample_size
@common_args.vits_batch_size
@common_args.hdbscan_batch_size
@click.option('--det-dir', help='Input folder(s) with raw detection results', multiple=True, required=True)
@click.option('--save-dir', help='Output directory to save clustered detection results', required=True)
@click.option('--device', help='Device to use, e.g. cpu or cuda:0 or cuda to use all cuda devices', type=str, default='cpu')
@click.option('--use-vits', help='Set to using the predictions from the vits cluster model', is_flag=True)
def run_cluster_det(det_dir, save_dir, device, use_vits, config_ini, alpha, cluster_selection_epsilon, cluster_selection_method, algorithm, min_cluster_size, min_sample_size, vits_batch_size, hdbscan_batch_size, start_image, end_image, use_pca, skip_visualization):
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
    extract_metadata = config('cluster', 'extract_metadata') == 'True'
    remove_corners = config('cluster', 'remove_corners') == 'True'
    remove_bad_images = config('cluster', 'remove_bad_images') == 'True'
    latitude = float(config('cluster', 'latitude'))
    longitude = float(config('cluster', 'longitude'))
    min_score = float(config('cluster', 'min_score'))
    min_similarity = float(config('cluster', 'min_similarity'))
    model = config('cluster', 'model')
    allowable_classes = config('cluster', 'allowable_classes')
    if len(allowable_classes) > 0:
        allowable_classes = allowable_classes.split(',')

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
        csv_file = combine_csv(csv_files, Path(temp_dir), crop_path)

        info('Loading detections')
        df = pd.read_csv(csv_file, sep=',')

        if df.empty:
            info(f'No detections found in {det_dir}')
            return

        info(f'Found {len(df)} detections in {det_dir}')

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
            end_image_index = df[df['image_path'].str.contains(end_image)].index[-1]
            df = df.iloc[:end_image_index]

        # Filter the dataframe to only include images in the start_image and end_image range
        df = filter_images(min_area, max_area, min_saliency, min_score, df)

        if df.empty:
            info(f'No detections found after filtering')
            return

        # Add in a column for the unique crop name for each detection with a unique id
        # create a unique uuid based on the md5 hash of the box in the row
        df['crop_path'] = df.apply(lambda
                                       row: f"{row['crop_root']}/{uuid.uuid5(uuid.NAMESPACE_DNS, str(row['x']) + str(row['y']) + str(row['xx']) + str(row['xy']))}.png",
                                   axis=1)

        # Add in a column for the unique crop name for each detection with a unique id
        df['cluster'] = -1  # -1 is the default value and means that the image is not in a cluster
        df['class'] = 'Unknown'
        df['class_s'] = 'Unknown'
        df['score'] = 0.
        df['score_s'] = 0.

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

        # TODO: refactor this block for modin optimization
        if extract_metadata:
            info('Extracting metadata ...')
            pattern_date0 = re.compile(r'(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})\.(\d{6})Z')  # 20240923T225833.474105Z
            pattern_date1 = re.compile(r'(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z')  # 20161025T184500Z
            pattern_date2 = re.compile(r'(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z\d*mF*')
            pattern_date3 = re.compile(r'(\d{2})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z')  # 161025T184500Z
            pattern_date4 = re.compile(r'(\d{2})-(\d{2})-(\d{2})T(\d{2})_(\d{2})_(\d{2})-')  # 16-06-06T16_04_54
            pattern_depth = re.compile(r'_(\d+(?:\.\d+)?)m_') # 50m_100m_200m_ or _100.0m

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

            for index, row in tqdm(df.iterrows(), total=len(df), desc="Extracting metadata", unit="image"):
                image_name = Path(row.image_path).name
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
                if pattern_depth.search(image_name):
                    match = pattern_depth.match(image_name)
                    depth[index] = float(match.group(1))

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

        # Replace any NaNs with 0 and reindex
        df.fillna(0)
        df = df.reset_index(drop=True)

        # Print the first 5 rows of the dataframe
        info(df.head(5))

        if len(df) > 0:
            # Replace / with _ in the model name
            model_machine_friendly = model.replace('/', '_')

            # A prefix for the output files to make sure the output is unique for each execution
            prefix = f'{model_machine_friendly}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

            # Cluster the detections
            summary = cluster_vits(prefix, model, df, save_dir, alpha, cluster_selection_epsilon,
                                   cluster_selection_method, algorithm,
                                   min_similarity, min_cluster_size, min_samples, device,
                                   use_pca=use_pca, use_vits=use_vits,
                                   skip_visualization=skip_visualization, roi=False,
                                   remove_bad_images=remove_bad_images,
                                   vits_batch_size=vits_batch_size,
                                   hdbscan_batch_size=hdbscan_batch_size,
                                   allowable_classes=allowable_classes)

            if summary is None:
                err(f'No summary returned from clustering')
                return

            # Add more detail to the summary specific to detections
            summary['sdcat_version'] = sdcat_version
            summary['dataset']['input'] = det_dir
            summary['dataset']['image_resolution'] = f"{df['image_width'].iloc[0]}x{df['image_height'].iloc[0]} pixels"
            summary['dataset']['detection_count'] = len(df)

            with open(save_dir / f'{prefix}_summary.json', 'w') as f:
                json.dump(summary, f, indent=4)
            info(f"Summary saved to {save_dir / f'{prefix}_summary.json'}")
            info(f'Summary: {json.dumps(summary, indent=4)}')

            # Save a copy of the config.ini file
            shutil.copy(Path(config_ini), save_dir / f'{prefix}_config.ini')
            info(f'Config saved to {save_dir / f"{prefix}_config.ini"}')
        else:
            warn(f'No detections found to cluster')

@click.command('roi', help='Cluster roi. See cluster --config-ini to override cluster defaults.')
@common_args.config_ini
@common_args.use_pca
@common_args.skip_visualization
@common_args.alpha
@common_args.cluster_selection_epsilon
@common_args.cluster_selection_method
@common_args.algorithm
@common_args.min_cluster_size
@common_args.min_sample_size
@common_args.vits_batch_size
@common_args.hdbscan_batch_size
@click.option('--roi-dir', help='Input folder(s) with raw ROI images', multiple=True, required=True)
@click.option('--save-dir', help='Output directory to save clustered detection results', required=True)
@click.option('--device', help='Device to use, e.g. cpu or cuda:0 or cuda to use all cuda devices', type=str)
@click.option('--use-vits', help='Set to using the predictions from the vits cluster model', is_flag=True)
def run_cluster_roi(roi_dir, save_dir, device, use_vits, config_ini, alpha, cluster_selection_epsilon, cluster_selection_method, algorithm, min_cluster_size, min_sample_size, vits_batch_size, hdbscan_batch_size, use_pca, skip_visualization):
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
    allowable_classes = config('cluster', 'allowable_classes')
    if len(allowable_classes) > 0:
        allowable_classes = allowable_classes.split(',')

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

    if df.empty:
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
    # Only copy images that do not already exist in the crop path to save time
    to_copy = df[~df['crop_path'].apply(os.path.exists)]
    if to_copy.empty:
        info(f'No images to copy to {crop_path}')
    else:
        for src, dst in tqdm(zip(to_copy['image_path'], to_copy['crop_path']),
                             total=len(to_copy), desc="Copying images"):
            shutil.copy(src, dst)

    df = filter_images(min_area, max_area, min_saliency, min_score, df)
    if df.empty:
        info(f'No detections found after filtering')
        return

    df['cluster'] = -1  # -1 is the default value and means that the image is not in a cluster
    df['class'] = 'Unknown'
    df['class_s'] = 'Unknown'
    df['score'] = 0.
    df['score_s'] = 0.

    # Replace any NaNs with 0 and reindex
    df.fillna(0)
    df = df.reset_index(drop=True)

    # Print the first 5 rows of the dataframe
    info(df.head(5))

    if len(df) > 0:
        model_machine_friendly = model.replace('/', '_')

        # A prefix for the output files to make sure the output is unique for each execution
        prefix = f'{model_machine_friendly}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

        # Cluster the ROIs
        summary = cluster_vits(prefix, model, df, save_dir, alpha, cluster_selection_epsilon, cluster_selection_method, algorithm,
                               min_similarity, min_cluster_size, min_samples, device,
                               use_pca=use_pca, use_vits=use_vits,
                               skip_visualization=skip_visualization, roi=True,
                               remove_bad_images=remove_bad_images,
                               vits_batch_size=vits_batch_size,
                               hdbscan_batch_size=hdbscan_batch_size,
                               allowable_classes=allowable_classes)

        # Add more detail to the summary specific to ROIs
        summary['sdcat_version'] = sdcat_version
        summary['dataset']['roi'] = True
        summary['dataset']['input'] = roi_dir
        summary['dataset']['image_resolution'] = "224x224 pixels"
        summary['dataset']['detection_count'] = len(df)

        with open(save_dir / f'{prefix}_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
        info(f'Summary saved to {save_dir / f"{prefix}_summary.json"}')
        info(f'Summary: {json.dumps(summary, indent=4)}')

        # Save a copy of the config.ini file
        shutil.copy(Path(config_ini), save_dir / f'{prefix}_config.ini')
        info(f'Config saved to {save_dir / f"{prefix}_config.ini"}')
    else:
        warn(f'No detections found to cluster')
