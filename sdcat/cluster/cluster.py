# sdcat, Apache-2.0 license
# Filename: sdcat/cluster/cluster.py
# Description: Clustering using vision transformer features and HDBSCAN density-based clustering
import uuid
import warnings
from importlib.util import find_spec

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import os
from typing import List

import pandas
import seaborn as sns
import modin.pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from tqdm import tqdm
from umap import UMAP
from hdbscan import HDBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics.pairwise import cosine_similarity
from sdcat.logger import info, warn, debug
from sdcat.cluster.utils import cluster_grid, clean_bad_images, crop_all_square_images
from sdcat.cluster.embedding import fetch_embedding, has_cached_embedding, compute_norm_embedding
from sdcat import __version__ as sdcat_version

if find_spec("cuml"):
    info("=======> USING GPU for HDBSCAN AND UMAP <=========")
    from cuml.cluster import HDBSCAN as cuHDBSCAN  # pylint: disable=E0611, E0401
    from cuml.manifold.umap import UMAP as cuUMAP

    have_gpu = True
else:
    have_gpu = False

sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.set(rc={"figure.figsize": (12, 10)})


def _summarize_clusters(
    df: pd.DataFrame,
    output_path: Path,
    prefix: str,
    cluster_sim: list,
    model: str,
    skip_visualization: bool,
    hdbscan_params: dict[str],
) -> dict:
    """
    Summarize and optionally visualize the clusters using t-SNE or UMAP in grids
    """
    info("Summarizing clusters")
    num_samples = len(df)
    df = df._to_pandas()
    clustered_df = df[df["cluster"] != -1]
    noise_df = df[df["cluster"] == -1]
    labels = clustered_df["cluster"].values
    clusters = np.unique(labels)
    num_clusters = len(clusters)
    num_labels = len(labels)

    # Get the top 20 predicted classes in percent
    info("Getting top 20 predicted classes...")

    top20 = df["class"].value_counts(normalize=True).head(20) * 100
    top20 = top20.sort_index().round(2)

    summary = {
        "dataset": {
            "output": str(output_path),
            "clustering_algorithm": "HDBSCAN",
            "clustering_parameters": hdbscan_params,
            "feature_embedding_model": str(model),
        },
        "statistics": {
            "total_clusters": num_clusters,
            "cluster_coverage": f"{float(num_labels / num_samples):.2f} ({100*float(num_labels / num_samples):.2f}%)",
        },
    }

    image_paths = {c: df.loc[df["cluster"] == c, "crop_path"] for c in clusters}

    summary["statistics"]["top_predictions"] = [
        {"class": name, "percentage": f"{total:.2f}%"} for name, total in top20.items()
    ]

    if not skip_visualization:
        # Cannot use init='spectral' when n_components is >= num_samples - default to 'random' instead
        n_components = min(2, num_samples)
        if n_components >= num_samples:
            init = "random"
        else:
            init = "spectral"

        # Reduce the dimensionality of the embeddings using UMAP to 2 dimensions to visualize the clusters
        # Only use the exemplars and a random sample of 5000 images to speed up the visualization
        sampled_df = clustered_df.sample(n=min(num_labels, 5000), random_state=42, replace=False)
        sampled_emb = [fetch_embedding(model, filename)[0] for filename in sampled_df["crop_path"]]
        np_data = np.array(sampled_emb)

        n_neighbors = min(15, num_samples - 1)
        info(f"Using {n_neighbors} neighbors for dimensional reduction")
        if n_neighbors < 2:
            warn("Using PCA instead of UMAP to reduce for cluster 2D plot")
            from sklearn.decomposition import PCA

            pca = PCA(n_components=2)
            xx = pca.fit_transform(np_data)
        else:
            if have_gpu:
                info("Using GPU to reduce cluster 2D plot")
                xx = cuUMAP(
                    init=init,
                    n_components=2,
                    n_neighbors=n_neighbors,
                    min_dist=0.1,
                    metric="euclidean",
                    verbose=False,  # print statements from the verbose flag are forcing an exit in rapids so disable it
                ).fit_transform(np_data)
            else:
                info("Using UMAP to reduce cluster 2D plot")
                xx = UMAP(init=init, n_components=2, n_neighbors=n_neighbors, metric="cosine").fit_transform(np_data)

        df_joint = pandas.DataFrame({"x": xx[:, 0], "y": xx[:, 1], "labels": sampled_df["cluster"].values})
        p = sns.jointplot(data=df_joint, x="x", y="y", hue="labels")
        p.fig.suptitle(
            f"{prefix}\nsdcat_version {sdcat_version}\nClusters {num_clusters} with {num_samples} samples", fontsize=14
        )
        p.fig.subplots_adjust(top=0.80)
        p.savefig(f"{output_path}/{prefix}_summary.png")
        info(f"Saved {output_path}/{prefix}_summary.png")

        # Create a grid of the images to check the quality of the clustering results
        num_processes = min(os.cpu_count(), num_clusters)
        info(f"Using {num_processes} CPUs to visualize the {num_clusters} clusters")

        # Use a pool of processes to speed up the visualization of the clusters
        # Skip modin here because it does not offer much speedup
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for c in image_paths:
                grid_size = 4 if len(image_paths[c]) < 50 else 8  # grid size; larger clusters get larger grids
                futures.append(
                    executor.submit(cluster_grid, prefix, cluster_sim[c], c, grid_size, image_paths[c], output_path / prefix)
                )
            for future in as_completed(futures):
                future.result()

        # Create a grid of the noise images last with a different prefix
        if len(noise_df) > 0:
            grid_size = 4 if len(noise_df) < 50 else 8
            cluster_grid(prefix, 0, -1, grid_size, noise_df["crop_path"], output_path / prefix)

    return summary


def _reassign_noise(df: pd.DataFrame, exemplar_emb: np.ndarray, cluster: List[int], min_similarity: float, model: str):
    """
    Reassign the noise (-1) cluster to the nearest exemplar cluster based on cosine similarity.
    """
    noise_df = df[df["cluster"] == -1]
    noise_embeddings = np.array([fetch_embedding(model, path)[0] for path in noise_df["crop_path"]])
    similarities = cosine_similarity(noise_embeddings, exemplar_emb)
    max_scores = similarities.max(axis=1)
    best_sim_idx = similarities.argmax(axis=1)
    valid = max_scores > min_similarity
    valid_indices = noise_df.index[valid]
    df.loc[valid_indices, "cluster"] = cluster[best_sim_idx[valid]]
    return df


def _similarity_merge(
    df: pd.DataFrame, exemplar_emb: np.ndarray, cluster: List[int], min_similarity: float, model: str
) -> (pd.DataFrame, dict):
    """
    Merge clusters based on the linkage of the cosine similarity of their embeddings.
    """
    unique_clusters_before = df["cluster"].unique()

    info(f"Merging clusters with similarity threshold {min_similarity:.2f} ...")
    info(
        f"Maximum cluster id: {df['cluster'].values.max()} minimum cluster id: {df['cluster'].values.min()} unique clusters: {len(unique_clusters_before)}"
    )
    linkage_matrix = linkage(np.array(exemplar_emb), method="complete", metric="cosine")
    linkage_clusters = fcluster(linkage_matrix, 1 - min_similarity, criterion="distance")

    info(f"Unique clusters before merging: {len(unique_clusters_before)}")
    info(f"Linkage matrix size: {len(linkage_matrix)}")

    # If the cluster labels are all the same, then we have a single cluster and we can't merge
    if len(np.unique(linkage_clusters)) == 1:
        info("No clusters to merge")
    else:

        def map_to_new_cluster(old_cluster):
            if old_cluster == -1:
                debug(f"Cluster {old_cluster} is noise, skipping")
                return -1
            new_cluster = linkage_clusters[old_cluster]
            if new_cluster > len(cluster) - 1:
                return old_cluster
            actual_cluster = cluster[new_cluster]
            debug(f"Assigning cluster {old_cluster} to {actual_cluster}")
            return actual_cluster

        df["cluster"] = df["cluster"].map(map_to_new_cluster)
        unique_clusters_after = df["cluster"].unique()
        info(f"Unique clusters after merging: {len(unique_clusters_after)}")

    return df


def _compute_exemplars(df: pd.DataFrame, model: str, device: str, vits_batch_size: int) -> pd.DataFrame:
    non_noise_df = df[df["cluster"] != -1]
    max_score_idx = non_noise_df.sort_values("cluster", ascending=True).groupby("cluster")["HDBSCAN_probability"].idxmax()

    # Create a new dataframe with the exemplars for each cluster
    df_exemplars = df.loc[
        max_score_idx, ["cluster", "x", "y", "xx", "xy", "image_width", "image_height", "crop_path", "image_path"]
    ].copy()
    df_exemplars = df_exemplars.sort_values("cluster", ascending=True)
    df_exemplars["angle"] = 0

    # Add a new row for every row for every angle to be rotated
    df_exemplars_final = pd.DataFrame()
    for angle in [45, 135, 180, 225, 270, 315]:
        new_rows = df_exemplars.copy()
        new_rows["crop_path"] = new_rows["crop_path"].str.replace(".png", f"_rot{angle}.png")
        new_rows["angle"] = angle
        df_exemplars_final = pd.concat([df_exemplars_final, new_rows], ignore_index=True)

    # Add the original exemplars without rotation
    df_exemplars_final = pd.concat([df_exemplars_final, df_exemplars], ignore_index=True)

    # Crop all exemplar images at rotated angles, skipping those that already exist
    info(f"Cropping {len(df_exemplars_final)} exemplar images to square...")
    existing = df_exemplars_final["crop_path"].apply(lambda x: os.path.exists(x))
    rows_to_crop = df_exemplars_final[~existing]._to_pandas()
    image_angle_group = rows_to_crop.groupby(["image_path", "angle"])
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(crop_all_square_images, image_angle[0], df, 224, image_angle[1])
            for image_angle, df in image_angle_group
        ]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

    crop_paths = df_exemplars_final["crop_path"].values.tolist()
    compute_norm_embedding(model_name=model, images=crop_paths, device=device, batch_size=vits_batch_size)

    info("Computing mean embeddings for exemplars across all rotations ...")
    df_exemplar_group = df_exemplars_final.groupby("cluster")

    def calc_mean(group):
        embeddings = [fetch_embedding(model, path)[0] for path in group["crop_path"]]
        embeddings = [emb for emb in embeddings if len(emb) > 0]
        if not embeddings:
            raise RuntimeError(f"No embeddings found for {group['crop_path'].values}")
        return np.mean(np.array(embeddings), axis=0)

    # Apply the function to each group and get the mean embeddings array for the merge
    exemplar_means = df_exemplar_group.apply(calc_mean)
    exemplar_emb = np.stack(exemplar_means.values, axis=0)

    cluster = df_exemplar_group["cluster"].first().values
    return exemplar_emb, cluster


def _compute_avg_sim(df: pd.DataFrame, model: str) -> (pd.DataFrame, dict):
    def compute_cluster_avg_similarity(cluster_df):
        if len(cluster_df) == 0:
            return None

        cluster_emb = [fetch_embedding(model, f)[0] for f in cluster_df["crop_path"]]
        cluster_emb = np.array(cluster_emb)

        # This computes self-similarity within the cluster
        return np.mean(cosine_similarity(cluster_emb))

    # Report the number of items in each cluster
    cluster_counts = df.groupby("cluster").size()
    for cluster_id, count in cluster_counts.items():
        info(f"Cluster {cluster_id}: {count} items")

    # Group by cluster and apply the function in parallel
    info("Computing average similarity scores for each cluster ...")
    cluster_scores = df.groupby("cluster").apply(compute_cluster_avg_similarity)
    avg_sim_scores = cluster_scores.dropna().to_dict()
    return avg_sim_scores


def _run_hdbscan_assign(
    df: pandas.DataFrame,
    alpha: float,
    cluster_selection_epsilon: float,
    cluster_selection_method: str,
    algorithm: str,
    min_cluster_size: int,
    min_samples: int,
    use_pca: bool,
    use_cuhdbscan: bool,
    batch: int,
    max_batch: int,
    core_dist_n_jobs: int,
) -> pandas.DataFrame:
    """
    Cluster the features using HDBSCAN
    :param alpha:  The alpha parameter for HDBSCAN
    :param cluster_selection_epsilon:  The epsilon parameter for HDBSCAN
    :param algorithm:  The algorithm to use for clustering, 'best' or 'generic' or 'prims_kdtree' or 'boruvka_kdtree'
    :param cluster_selection_method:  The method to use for cluster selection, 'leaf' or 'eom'
    :param min_cluster_size:  The minimum number of samples in a cluster
    :param min_samples:   The number of samples in a neighborhood for a point
    :param use_pca:  Whether to use PCA for dimensionality reduction
    :param use_cuhdbscan:  Whether to use cuHDBSCAN for clustering; requires a GPU with CUDA support
    :param batch:  Batch number for logging
    :param max_batch:  Maximum batch number for logging
    :return: pandas.DataFrame with the cluster assignments
    """
    info(
        f"Clustering using HDBSCAN with: "
        f"sdcat version {sdcat_version},"
        f"core_dist_n_jobs: {core_dist_n_jobs},"
        f"batch {batch} of {max_batch},"
        f"alpha {alpha},"
        f"algorithm {algorithm},"
        f"cluster_selection_epsilon {cluster_selection_epsilon},"
        f"min_samples {min_samples},"
        f"min_cluster_size {min_cluster_size},"
        f"cluster_selection_method {cluster_selection_method},"
        f"use_pca {use_pca},"
        f"use_cuhdbscan {use_cuhdbscan}"
    )
    if use_cuhdbscan and not have_gpu:
        warn("cuHDBSCAN is requested but no GPU is available. Falling back to CPU HDBSCAN.")
        use_cuhdbscan = False

    # Get the number of samples which is the number of rows in the dataframe
    num_samples = df.shape[0]

    info(f"Stacking {len(df)} features ...")
    features = np.stack(df["embedding"].values)
    # Add in other features if present
    if df.shape[1] > 1:
        for col in df.columns:
            if col != "embedding":
                features = np.concatenate((features, df[col].values.reshape(-1, 1)), axis=1)

    if use_pca:
        info("Reducing to 100 dimensions using PCA...")
        pca = PCA(n_components=100, random_state=42)
        x = pca.fit_transform(features)
    else:
        x = features

    # Cluster the features using HDBSCAN
    if have_gpu and use_cuhdbscan:
        info(f"Running HDBSCAN on {num_samples} samples for batch {batch} of {max_batch} on GPU...")
        # L2 norm the data for cuHDBSCAN; used together with Euclidean distance as an approximation of cosine distance
        x_norm = normalize(x, norm="l2")
        scan = cuHDBSCAN(
            prediction_data=True,
            metric="euclidean",
            allow_single_cluster=True,
            algorithm=algorithm,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            alpha=alpha,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method=cluster_selection_method,
        )
        labels = scan.fit_predict(x_norm)
    else:
        info(
            f"Running HDBSCAN on {num_samples} samples for batch {batch} of {max_batch} on CPU core_dist_n_jobs {core_dist_n_jobs}..."
        )
        # Compute the cosine similarity matrix
        cosine_sim_matrix = cosine_similarity(x)
        distance_matrix = 1 - cosine_sim_matrix
        x = distance_matrix.astype(np.float64)
        scan = HDBSCAN(
            core_dist_n_jobs=core_dist_n_jobs,
            prediction_data=True,
            metric="precomputed",
            algorithm=algorithm,
            allow_single_cluster=True,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            alpha=alpha,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method=cluster_selection_method,
        )
        labels = scan.fit_predict(x)

    info(f"Number of clusters including unassigned -1 cluster: {len(np.unique(labels))}")

    df_ = df.copy()
    # Save the probabilities and cluster_batch for each cluster which is used for merging
    df_["HDBSCAN_probability"] = scan.probabilities_
    df_["cluster"] = labels
    # Set the cluster batch based on the batch number and cluster id,
    # Create a UUID for each cluster batch to ensure uniqueness
    # e.g. "9476e0d0-3ee3-492e-816c-050f02e07bdb_00000002" for batch 0 and cluster label 2
    # or "-1" for noise
    batch_uuid = str(uuid.uuid4())

    def set_cluster_batch(x):
        if x != -1:
            return f"{batch:08d}{x:08d}_{batch_uuid}"
        return "-1"

    df_["cluster_batch"] = df_["cluster"].apply(set_cluster_batch)
    unique_clusters = df_["cluster"].unique()
    info(f"Batch {batch} index: {df_.index[0]} to {df_.index[-1]}")
    info(
        f"Maximum cluster id: {labels.max()} minimum cluster id: {labels.min()} unique clusters: {len(unique_clusters)} noise: {np.sum(labels == -1)}"
    )
    return df_


def cluster_vits(
    prefix: str,
    model: str,
    df_dets: pd.DataFrame,
    output_path: Path,
    alpha: float,
    cluster_selection_epsilon: float,
    cluster_selection_method: str,
    algorithm: str,
    min_similarity: float,
    min_cluster_size: int,
    min_samples: int,
    device: str = "cpu",
    use_vits: bool = False,
    use_pca: bool = False,
    use_cuhdbscan=False,
    skip_visualization: bool = False,
    remove_bad_images: bool = False,
    roi: bool = False,
    vits_batch_size: int = 32,
    hdbscan_batch_size: int = 50000,
    allowable_classes: list = None,
) -> dict or None:
    """Cluster the crops using the VITS embeddings.
    :param prefix:  A unique prefix to save artifacts from clustering
    :param model: The model to use for clustering
    :param df_dets: The dataframe with the detections
    :param output_path: The output path to save the clustering artifacts to
    :param roi:  Whether the detections are already cropped to the ROI
    :param cluster_selection_epsilon: The epsilon parameter for HDBSCAN
    :param cluster_selection_method: The method to use for cluster selection, 'leaf' or 'eom'
    :param algorithm:  The algorithm to use for clustering, 'best' or 'generic' or 'prims_kdtree' or 'boruvka_kdtree'
    :param alpha: The alpha parameter for HDBSCAN
    :param min_similarity: The minimum similarity score to use for -1 cluster reassignment
    :param min_cluster_size: The minimum number of samples in a cluster
    :param min_samples:The number of samples in a neighborhood for a point
    :param device: The device to use for clustering, 'cpu' or 'cuda' or 'cuda:0' or 'cuda:1'
    :param use_vits: Set to using the predictions from the vits cluster model
    :param skip_visualization: Whether to skip the visualization of the clusters
    :param remove_bad_images: Whether to remove bad images from the clustering
    :param use_pca: Whether to use PCA for dimensionality reduction
    :param use_cuhdbscan: Whether to use cuHDBSCAN for clustering; requires a GPU with CUDA support
    :param vits_batch_size: The batch size to use for embedding extraction; maximize for speed for your GPU
    :param hdbscan_batch_size: The batch size to use for clustering with HDBSCAN; maximize for speed for your GPU
    :param allowable_classes: A list of classes to allow in the clustering; if None, all classes are allowed
    :return:  a dictionary with a summary."""
    warnings.filterwarnings("ignore")
    # If there are no detections, return an empty dataframe
    if df_dets.empty:
        warn("No detections found")
        return None

    # If the detections are not cropped, crop them to a square
    if not roi:
        # Only crop if needed
        existing = df_dets["crop_path"].apply(lambda x: os.path.exists(x))
        num_square = existing.sum()

        if num_square == len(df_dets):
            info(f"All {len(df_dets)} detections are already cropped to square")
        else:
            info(f"Cropping {len(df_dets)} detections...")
            # Filter rows that need cropping
            rows_to_crop = df_dets[~existing]._to_pandas()

            # Crop by grouped image_path which is more efficient
            grouped = rows_to_crop.groupby("image_path")
            info(f"Cropping {len(rows_to_crop)} detections...")

            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = [executor.submit(crop_all_square_images, group, df, 224, 0) for group, df in grouped]
                for _ in tqdm(as_completed(futures), total=len(futures)):
                    pass

    # Drop any rows with crop_path that have files that don't exist or are zero - sometimes the crops fail
    num_before = len(df_dets)
    df_dets = df_dets[df_dets["crop_path"].apply(lambda x: os.path.exists(x))]
    df_dets = df_dets[df_dets["crop_path"].apply(lambda x: os.path.getsize(x) > 0)]
    num_after = len(df_dets)
    if num_before != num_after:
        info(f"Dropped {num_before - num_after} detections with missing crop_path files")
    if df_dets.empty:
        warn("No detections found")
        return None

    df_dets = df_dets.sort_index()
    df_dets["exemplar"] = 0
    df_dets["cluster"] = -1  # -1 means noise or unassigned
    df_dets["cluster_batch"] = "-1"
    df_dets["HDBSCAN_probability"] = 0

    # Get the list of images to crop
    crop_paths = df_dets["crop_path"].values

    # Count how many files have the .npy extension
    num_cached = sum([has_cached_embedding(model, filename) for filename in crop_paths])
    info(f"Found {num_cached} cached embeddings for {len(crop_paths)} images")

    # Skip the embedding extraction if all the embeddings are cached
    if num_cached != len(crop_paths):
        debug(f"Extracted embeddings from {len(crop_paths)} images using model {model}...")
        compute_norm_embedding(model, crop_paths, device, vits_batch_size)

    def load_model_results(crop_path):
        try:
            _, labels, scores = fetch_embedding(model, crop_path)
            label = labels[0]
            label_s = labels[1]
            score = scores[0]
            score_s = scores[1]
            return pandas.Series({"class": label, "score": score, "class_s": label_s, "score_s": score_s})
        except IndexError:
            return pandas.Series({"class": "Unknown", "score": 0, "class_s": "Unknown", "score_s": 0})

    if use_vits:
        info(f"Loading ViTS model {model} results into dataframe ...")

        # Add in column if missing and apply the function to each row in the dataframe
        results_df = df_dets["crop_path"].apply(load_model_results).apply(pandas.Series)
        df_dets["class"] = results_df["class"]
        df_dets["score"] = results_df["score"]
        df_dets["class_s"] = results_df["class_s"]
        df_dets["score_s"] = results_df["score_s"]

        # If allowable_classes is provided, filter the dataframe to only include those classes and reset the index
        if allowable_classes:
            df_dets = df_dets[df_dets["class"].isin(allowable_classes)].reset_index(drop=True)

            if df_dets.empty:
                print(f"No detections left after filtering by allowable classes {allowable_classes}")
                return None

    if not (output_path / prefix).exists():
        (output_path / prefix).mkdir(parents=True)

    # Remove everything except ancillary data to include in clustering
    columns = [
        "x",
        "y",
        "xx",
        "xy",
        "w",
        "h",
        "image_width",
        "image_height",
        "cluster_id",
        "cluster",
        "score",
        "class",
        "score_s",
        "class_s",
        "image_path",
        "crop_path",
    ]
    # Check if the columns exist in the dataframe
    if all(col in df_dets.columns for col in columns):
        ancillary_df = df_dets.drop(
            columns=[
                "x",
                "y",
                "xx",
                "xy",
                "w",
                "h",
                "image_width",
                "image_height",
                "cluster_id",
                "cluster",
                "score",
                "cluster_s",
                "score_s",
                "class",
                "image_path",
                "crop_path",
            ]
        )
    else:
        ancillary_df = None

    if ancillary_df:
        # Add in any numerical ancillary data and replace NaNs with 0
        numerical_df = ancillary_df.select_dtypes(include=["float", "int"])
        numerical_df = numerical_df.fillna(0)
        # Normalize the numerical data from 0 to 1 - this is an important step!
        ancillary_df = (numerical_df - numerical_df.min()) / (numerical_df.max() - numerical_df.min())

    # Cluster
    num_batches = int(np.ceil(len(crop_paths) / hdbscan_batch_size))

    info(f"Remove any existing cluster grid images in the output_path in {(output_path / prefix)}")
    cluster_grids = (output_path / prefix).rglob(f"{prefix}_*cluster*.png")
    for c in cluster_grids:
        c.unlink()

    info(f"Processing images in batches of {hdbscan_batch_size} ...")

    def process_batch(i, n_jobs, df, df_ancillary=None):
        """
        Process a batch of images and return the clustered dataframe.
        Disable df_ancillary for now
        """
        start = df.index[0]
        end = df.index[-1]
        info(f"Processing batch {i + 1} of {num_batches} {start} to {end}...")
        filepaths = df["crop_path"].values.tolist()
        info(f"Fetching batch {i + 1}  embeddings for {len(filepaths)} images...")
        df["embedding"] = [fetch_embedding(model, filename)[0] for filename in filepaths]
        info(f"Done fetching batch {i + 1} embeddings for {len(filepaths)} images")

        # Drop any rows where the embedding is None or empty
        df = df[df["embedding"].apply(lambda x: x is not None and len(x) > 0)]

        if remove_bad_images:
            info(f"Cleaning bad images from {len(df)} images")
            bad_images = clean_bad_images(filepaths)
            bad_indexes = df[df["crop_path"].isin(bad_images)].index.tolist()
            info(
                f"Removed {len(bad_images)} of {len(filepaths)} images using cleanvision in batch {i + 1} of {num_batches}..."
            )
            df = df.drop(index=bad_indexes)
            # All detections in this batch are bad, so skip clustering
            if df.empty:
                warn(f"No detections left in batch {i + 1} of {num_batches} after cleaning bad images")
                return None

        # Drop unnecessary columns
        keep_columns = ["area", "saliency", "w", "h", "embedding", "crop_path"]
        df = df[[col for col in df.columns if col in keep_columns]]

        # Merge ancillary data
        if df_ancillary is not None:
            for filename in crop_paths[start:end]:
                ancillary_data_row = df_ancillary.loc[df_ancillary["crop_path"] == filename]
                if ancillary_data_row.empty:
                    ancillary_data = pd.Series([0] * len(df_ancillary.columns), index=df_ancillary.columns)
                else:
                    ancillary_data = ancillary_data_row.select_dtypes(include=["float", "int"]).iloc[0]
                df.loc[df_ancillary["crop_path"] == filename, ancillary_data.index] = ancillary_data

        # Drop the crop_path column if it exists as it is not numerical and not needed for clustering
        df = df.drop(columns=["crop_path"], errors="ignore")

        # Run clustering
        return _run_hdbscan_assign(
            df=df,
            alpha=alpha,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method=cluster_selection_method,
            algorithm=algorithm,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            use_pca=use_pca,
            batch=i,
            max_batch=num_batches,
            core_dist_n_jobs=n_jobs,
            use_cuhdbscan=use_cuhdbscan,
        )

    # Calculate the number of batches
    num_batches = int(np.ceil(len(df_dets) / hdbscan_batch_size))

    # If the number of batches is less than 2, set the core_dist_n_jobs to the number of CPU cores assuming the core
    # can handle the load of two batches at a time
    if num_batches > 1:
        core_dist_n_jobs = 1
        info(f"Using {core_dist_n_jobs} CPUs for HDBSCAN")
    else:
        # Set the number of processes to the number of batches
        core_dist_n_jobs = int(os.cpu_count())
        info(f"Using {core_dist_n_jobs} CPUs for HDBSCAN")

    # Run batches in parallel
    with ThreadPoolExecutor(max_workers=core_dist_n_jobs) as executor:
        futures = [
            executor.submit(
                process_batch,
                i,
                core_dist_n_jobs,
                df_dets.iloc[i * hdbscan_batch_size : min((i + 1) * hdbscan_batch_size, len(crop_paths))]._to_pandas(),
            )
            for i in range(num_batches)
        ]
        # Wait for all the futures to complete
        for result in tqdm(as_completed(futures), total=len(futures)):
            if result:
                df_batch = result.result()
                info(f"Batch {df_batch.index[0]} to {df_batch.index[-1]} completed")
                if df_batch is not None:
                    if df_batch.empty:
                        warn(f"Batch {df_batch.index[0]} to {df_batch.index[-1]} is empty after clustering")
                    else:
                        df_dets.update(df_batch["cluster_batch"])
                        df_dets.update(df_batch["cluster"])
                        df_dets.update(df_batch["HDBSCAN_probability"])

    info("All batch clusters processed")
    if df_dets.empty:
        warn("No detections left after clustering")
        return None

    # Drop any rows with NaN values
    df_dets = df_dets.dropna()
    if df_dets.empty:
        warn("No detections left after dropping NaN values")
        return None

    # Create an increasing array of integers based on the unique cluster_batch values, except for the noise
    info("Mapping clusters to unique integers ...")
    non_noise_df = df_dets[df_dets["cluster"] != -1]

    if non_noise_df.empty:
        warn("No clusters found")
        return None

    unique_cluster_batch = non_noise_df.groupby("cluster_batch")
    cluster_batch_group = {cg: i for i, cg in enumerate(unique_cluster_batch.groups)}
    cluster_batch_group["-1"] = -1  # Ensure noise cluster is mapped to -1

    info(f"Mapping {len(cluster_batch_group)} unique cluster batches to integers ...")
    df_dets["cluster_org"] = df_dets["cluster"]  # Keep the original cluster for reference
    values = df_dets["cluster_batch"].values
    df_dets["cluster"] = [cluster_batch_group.get(c, -1) for c in values]  # Map to contiguous integers

    unique_clusters = df_dets["cluster"].unique()
    info(f"Index: {df_dets.index[0]} to {df_dets.index[-1]}")
    info(
        f"Remapped maximum cluster id: {df_dets['cluster'].values.max()} minimum cluster id: {df_dets['cluster'].values.min()} unique clusters: {len(unique_clusters)}"
    )

    # Compute the exemplars for each cluster
    exemplar_emb, cluster = _compute_exemplars(df_dets, model, device, vits_batch_size)

    # Reassign noise (-1) cluster to the nearest exemplar cluster based on cosine similarity
    df_dets_re = _reassign_noise(df_dets, exemplar_emb, cluster, min_similarity, model)

    # Merge by similarity to the exemplar embeddings
    df_dets_final = _similarity_merge(df_dets_re, exemplar_emb, cluster, min_similarity, model)

    # We may have fewer clusters after merging, so reassign the cluster ids to contiguous integers
    non_noise_df = df_dets_final[df_dets_final["cluster"] != -1]
    unique_cluster = non_noise_df.groupby("cluster")
    cluster_group = {cg: i for i, cg in enumerate(unique_cluster.groups)}
    cluster_group["-1"] = -1  # Ensure noise cluster is mapped to -1

    info(f"Mapping {len(unique_clusters)} unique cluster to contiguous integers ...")
    values = df_dets_final["cluster"].values
    df_dets["cluster"] = [cluster_group.get(c, -1) for c in values]  # Map to contiguous integers

    # Get the average similarity within and across all clusters
    avg_sim_scores = _compute_avg_sim(df_dets_final, model)
    avg_similarity = np.mean(list(avg_sim_scores.values()))

    info(f"Average similarity: {avg_similarity:.2f} min {min_similarity:.2f}  ")
    unique_clusters = df_dets_final["cluster"].unique()

    # Drop the -1 value which is the noise cluster
    if -1 in unique_clusters:
        unique_clusters = unique_clusters[unique_clusters != -1]
    unique_clusters.sort()

    if len(unique_clusters) == 0:
        warn("No clusters found")

    info(f"Found {len(unique_clusters)} clusters with an average similarity of {avg_similarity:.2f} ")

    num_samples = df_dets.shape[0]
    clustered = df_dets["cluster"].values != -1
    coverage = np.sum(clustered) / num_samples
    info(f"Coverage: {coverage:.2f} ({np.sum(clustered)}/{num_samples})")

    hdbscan_params = {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "cluster_selection_method": "leaf",
        "metric": "precomputed",
        "algorithm": algorithm,
        "alpha": alpha,
        "cluster_selection_epsilon": cluster_selection_epsilon,
        "use_pca": use_pca,
    }
    if use_cuhdbscan and have_gpu:
        hdbscan_params["use_cuhdbscan"] = True
        hdbscan_params["metric"] = "l2"

    df_dets_final.to_csv(output_path / f"{prefix}_cluster_detections.csv")
    info(f'Saved {output_path / f"{prefix}_cluster_detections.csv"}')
    df_dets_final.to_parquet(output_path / f"{prefix}_cluster_detections.parquet")
    info(f'Saved {output_path / f"{prefix}_cluster_detections.parquet"}')

    # Return a summary of the clusters
    return _summarize_clusters(df_dets_final, output_path, prefix, avg_sim_scores, model, skip_visualization, hdbscan_params)
