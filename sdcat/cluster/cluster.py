# sdcat, Apache-2.0 license
# Filename: sdcat/cluster/cluster.py
# Description: Clustering using vision transformer features and HDBSCAN density-based clustering
import io
import multiprocessing
from collections import Counter
from importlib.util import find_spec

import pandas as pd
from pathlib import Path
import os
import json

import seaborn as sns
import numpy as np
from numpy.ma.core import indices
from umap import UMAP
from hdbscan import HDBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sdcat.logger import info, warn, debug, err
from sdcat.cluster.utils import cluster_grid, crop_square_image, square_image, clean_bad_images
from sdcat.cluster.embedding import fetch_embedding, has_cached_embedding, compute_norm_embedding

if find_spec("multicore_tsne"):
    from multicore_tsne import MulticoreTSNE as TSNE
else:
    from sklearn.manifold import TSNE

if find_spec("cuml"):
    info('=======> USING GPU for HDBSCAN AND UMAP <=========')
    from cuml.cluster import HDBSCAN as cuHDBSCAN  # pylint: disable=E0611, E0401
    from cuml.manifold.umap import UMAP as cuUMAP

    have_gpu = True
else:
    have_gpu = False

sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.set(rc={"figure.figsize": (12, 10)})


def _visualize_clusters(df: pd.DataFrame, clusters: list, output_path: Path, prefix: str, cluster_sim: list) -> None:
    """
    Visualize the clusters using t-SNE or UMAP.
    """
    # For each cluster  let's create a grid of the images to check the quality of the clustering results
    num_processes = min(multiprocessing.cpu_count(), len(clusters))
    info(f'Using {num_processes} processes to visualize the {len(clusters)} clusters')

    images = df['crop_path'].tolist()
    if num_processes == 0:
        err(f'No processes available to visualize the clusters')
        return None

    unique_clusters = {}
    for c in clusters:
        unique_clusters[int(c)] = list(df.loc[df['cluster_reindex'] == c].index)

    for c in unique_clusters.keys():
        if c == -1:
            continue
        cluster_grid(prefix,
                     cluster_sim[c],
                     c,
                     unique_clusters[c],
                     4 if len(unique_clusters[c]) < 50 else 8,
                     [images[idx] for idx in unique_clusters[c]],
                     output_path / prefix
                     )

    # Use a pool of processes to speed up the visualization of the clusters
    # with multiprocessing.Pool(num_processes) as pool:
    #     args = [(prefix,  # prefix
    #              cluster_sim[cluster_id],  # average similarity for the cluster
    #              cluster_id,  # cluster id
    #              unique_clusters[cluster_id],  # cluster indices
    #              4 if len(unique_clusters[cluster_id]) < 50 else 8,  # grid size; larger clusters get larger grids
    #              [images[idx] for idx in unique_clusters[cluster_id]],  # images in the cluster
    #              output_path / prefix) for cluster_id in
    #             range(0, len(unique_clusters))]
    #     pool.starmap(cluster_grid, args)

def _merge(
        df: pd.DataFrame,
        min_similarity: float,
        model: str
    ) -> (pd.DataFrame, dict):
    """
    Merge clusters based on the linkage of the cosine similarity of their embeddings.
    """
    # Add a cluster_reindex column to the dataframe with the reindexed cluster ids
    df['cluster_reindex'] = pd.NA
    valid_mask = df['cluster'] != -1 # mask all invalid clusters as NaN
    df.loc[valid_mask, 'cluster_reindex'] = pd.factorize(
        list(zip(df.loc[valid_mask, 'batch'], df.loc[valid_mask, 'cluster']))
    )[0]
    df['cluster_reindex'] = df['cluster_reindex'].astype('Int64')

    # Get the exemplar embeddings
    exemplar_df = df[df['exemplar'] == 1]
    exemplar_emb = []
    for filename in exemplar_df['crop_path']:
        emb, _, _ = fetch_embedding(model, filename)
        exemplar_emb.append(emb)



    info(f'Merging clusters with similarity threshold {min_similarity:.2f} ...')
    linkage_matrix = linkage(exemplar_emb, method='complete', metric='cosine')
    cluster_labels = fcluster(linkage_matrix, 1 - min_similarity, criterion='distance')

    unique_clusters = np.unique(cluster_labels)
    info(f'Unique clusters before merging: {len(unique_clusters)} clusters')
    # If the cluster labels are all the same, then we have a single cluster and we can't merge
    if len(np.unique(cluster_labels)) == 1:
        info(f'No clusters to merge')
    else:
        labels = df['cluster_reindex'].values
        # Assign the exemplar clusters to the original clusters based on the linkage matrix
        for i, j in enumerate(labels):
            if j is pd.NA:
                continue
            # debug(f'Label {labels[i]} is now {cluster_labels[labels[i]]}')
            labels[i] = cluster_labels[labels[i]]
        df['cluster_reindex'] = labels

        # Replace pd.NA in cluster_reindex column with -1
        df.fillna(-1, inplace=True)

        unique_clusters = df['cluster_reindex'].unique()
        # Drop the -1 value which are the noise
        unique_clusters = unique_clusters.tolist()
        unique_clusters.remove(-1)
        unique_clusters.sort()

        info(f'Unique clusters after merging: {len(unique_clusters)} clusters')
        for c in unique_clusters:
            rows = df[df['cluster_reindex'] == c]
            debug(f'Cluster {c} has {len(rows)} samples')

    # Compute the average similarity score for each cluster
    avg_sim_scores = {}
    for cluster in unique_clusters:
        cluster_df = df[df['cluster_reindex'] == cluster]
        if len(cluster_df) == 0:
            continue
        # Get the embeddings for the cluster
        cluster_emb = []
        for filename in cluster_df['crop_path']:
            emb, _, _ = fetch_embedding(model, filename)
            cluster_emb.append(emb)
        cluster_emb = np.array(cluster_emb)

        # Compute the average similarity score for the cluster
        sim = cosine_similarity(cluster_emb)
        avg_sim_scores[cluster] = np.mean(sim)

    return df, avg_sim_scores


def _run_hdbscan_assign(
        image_emb: np.ndarray,
        alpha: float,
        cluster_selection_epsilon: float,
        cluster_selection_method: str,
        algorithm: str,
        min_cluster_size: int,
        min_samples: int,
        use_tsne: bool,
        ancillary_df: pd.DataFrame,
        batch_id: int) -> tuple:
    """
    Cluster the embeddings using HDBSCAN
    :param image_emb:  The embeddings to cluster from the model
    :param alpha:  The alpha parameter for HDBSCAN
    :param cluster_selection_epsilon:  The epsilon parameter for HDBSCAN
    :param algorithm:  The algorithm to use for clustering, 'best' or 'generic' or 'prims_kdtree' or 'boruvka_kdtree'
    :param cluster_selection_method:  The method to use for cluster selection, 'leaf' or 'eom'
    :param min_cluster_size:  The minimum number of samples in a cluster
    :param min_samples:   The number of samples in a neighborhood for a point
    :param use_tsne:  Whether to use t-SNE for dimensionality reduction
    :param ancillary_df:  (optional) Ancillary data to include in the clustering
    :return: The average similarity score for each cluster, exemplar_df, cluster ids, cluster means, and coverage
    """
    info(f'Clustering using HDBSCAN with: \n'
        f'batch_id {batch_id} \n'
        f'alpha {alpha} \n'
        f'algorithm {algorithm} \n'
        f'cluster_selection_epsilon {cluster_selection_epsilon} \n'
        f'min_samples {min_samples} \n'        
        f'min_cluster_size {min_cluster_size} \n'
        f'cluster_selection_method {cluster_selection_method} \n'
        f'use_tsne {use_tsne} ...')

    # Add in any numerical ancillary data and replace NaNs with 0
    df = pd.DataFrame(image_emb)
    numerical = ancillary_df.select_dtypes(include=["float", "int"])
    if not numerical.empty:
        numerical = numerical.fillna(0)

        # Normalize the numerical data from 0 to 1 and add it to the dataframe
        numerical = (numerical - numerical.min()) / (numerical.max() - numerical.min())

        # Skip of the numerical data is all NaN
        if not np.all(np.isnan(numerical)):
            df = pd.merge(df, numerical, left_index=True, right_index=True, how='left')
            df = df.fillna(0)

    # Add a batch and cluster column
    df['batch_id'] = batch_id
    df['cluster'] = -1

    # Get the number of samples which is the number of rows in the dataframe - this is used mostly for calculating coverage
    num_samples = df.shape[0]

    # Perplexity must be less than the number of samples
    perplexity = min(30, num_samples - 1)

    # TSN-E does not work well when we have a few samples
    if num_samples > 100 and use_tsne:
        tsne = TSNE(n_components=2, perplexity=perplexity, metric="cosine", n_jobs=8, random_state=42, verbose=True)
        embedding = tsne.fit_transform(df.values)
    else:
        embedding = df.values
    x = MinMaxScaler().fit_transform(embedding)  # scale the embedding to 0-1

    # Cluster the embeddings using HDBSCAN
    if len(df) == 1:
        labels = np.array([0])
    else:
        if have_gpu:
            scan = cuHDBSCAN(
                metric='euclidean',  # 'precomputed' does not work with cuHDBSCAN
                allow_single_cluster=True,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                alpha=alpha,
                algorithm=algorithm,
                cluster_selection_epsilon=cluster_selection_epsilon,
                cluster_selection_method=cluster_selection_method)
            labels = scan.fit_predict(x)
        else:
            scan = HDBSCAN(
                    prediction_data=True,
                    metric='l2',
                    algorithm=algorithm,
                    allow_single_cluster=True,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    alpha=alpha,
                    cluster_selection_epsilon=cluster_selection_epsilon,
                    cluster_selection_method=cluster_selection_method)
            labels = scan.fit_predict(x)

    # Get the unique clusters and sort them; -1 are unassigned clusters
    cluster_df = pd.DataFrame(labels, columns=['cluster'])
    cluster_df['cluster'] = cluster_df['cluster'].astype('int') # Convert the cluster column to int
    unique_clusters = cluster_df['cluster'].unique().tolist()
    unique_clusters.sort()
    info(f"Number of clusters including unassigned -1 cluster: {len(unique_clusters)}")

    # Get the index of the highest scores for each unique cluster sorted in increasing order
    # and use this as a representative image for the cluster
    cluster_df['score'] = scan.probabilities_
    max_scores = cluster_df.sort_values('cluster', ascending=True).groupby('cluster')['score'].idxmax()

    # Remove the first element which is the -1 cluster
    max_scores = max_scores[1:]

    # Flag as an exemplar the max scoring exemplars for each cluster
    cluster_df['exemplar'] = 0
    cluster_df.loc[max_scores, 'exemplar'] = 1
    return cluster_df

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
        weighted_score: bool = False,
        use_vits: bool = False,
        use_tsne: bool = False,
        skip_visualization: bool = False,
        remove_bad_images: bool = False,
        roi: bool = False,
        batch_size: int = 32
) -> pd.DataFrame:
    """  Cluster the crops using the VITS embeddings.
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
    :param device: The device to use for clustering, 'cpu' or 'cuda'
    :param weighted_score: Whether to weight score for the prediction from vits model with detection weight
    :param use_vits: Set to using the predictions from the vits cluster model
    :param skip_visualization: Whether to skip the visualization of the clusters
    :param remove_bad_images: Whether to remove bad images from the clustering
    :param use_tsne: Whether to use t-SNE for dimensionality reduction
    :return:  a dataframe with the assigned cluster indexes, or -1 for non-assigned."""

    # If there are no detections, return an empty dataframe
    if df_dets.empty:
        warn('No detections found in {detections} ')
        return pd.DataFrame()

    # Count how many files exists
    num_crop = sum([os.path.exists(filename) for filename in df_dets['crop_path']])

    # Skip cropping if all the crops are already done
    if num_crop != len(df_dets):
        num_processes = min(multiprocessing.cpu_count(), len(df_dets))
        if roi is True:
            info(f'ROI crops already exist')
        else:
            # Crop and squaring the images in parallel using multiprocessing to speed up the processing
            info(f'Cropping {len(df_dets)} detections in parallel using {num_processes} processes...')
            with multiprocessing.Pool(num_processes) as pool:
                args = [(row, 224) for index, row in df_dets.iterrows()]
                pool.starmap(crop_square_image, args)

    if remove_bad_images:
        # Remove any detections that are in any corner of the image
        size_before = len(df_dets)
        df = clean_bad_images(df_dets)
        size_after = len(df)
        info(f'Removed {size_before - size_after} detections that were dark or blurry')

    # Drop any rows with crop_path that have files that don't exist - sometimes the crops fail
    df_dets = df_dets[df_dets['crop_path'].apply(lambda x: os.path.exists(x))]
    df_dets = df_dets.copy()

    # Get the list of images to crop
    images = df_dets['crop_path'].tolist()

    # Count how many files have the .npy extension
    num_cached = sum([has_cached_embedding(model, filename) for filename in images])

    # Skip the embedding extraction if all the embeddings are cached
    if num_cached != len(images):
        debug(f'Extracted embeddings from {len(images)} images using model {model}...')
        compute_norm_embedding(model, images, device, batch_size)

    if use_vits:
        debug('Compute weighted scores ...')
        for filename in images:
            _, label, score = fetch_embedding(model, filename)
            weight = 1
            if weighted_score:
                weight = df_dets.loc[df_dets['crop_path'] == filename, 'score'].values[0]
                # Weight cannot be zero or negative
                if weight <= 0:
                    weight = 1
            df_dets.loc[df_dets['crop_path'] == filename, 'class'] = label[0]
            df_dets.loc[df_dets['crop_path'] == filename, 'score'] =(score[0]+weight)/2.
            df_dets.loc[df_dets['crop_path'] == filename, 'class_s'] = label[1]
        df_dets.loc[df_dets['crop_path'] == filename, 'score_s'] = (score[1]+weight)/2.

    if not (output_path / prefix).exists():
        (output_path / prefix).mkdir(parents=True)

    # Remove everything except ancillary data to include in clustering
    columns = ['x', 'y', 'xx', 'xy', 'w', 'h', 'image_width', 'image_height', 'cluster_id',
               'cluster', 'score', 'class', 'score_s', 'class_s', 'image_path', 'crop_path']
    # Check if the columns exist in the dataframe
    if all(col in df_dets.columns for col in columns):
        ancillary_df = df_dets.drop(
            columns=['x', 'y', 'xx', 'xy', 'w', 'h', 'image_width', 'image_height', 'cluster_id',
                     'cluster', 'score', 'cluster_s', 'score_s',
                     'class', 'image_path', 'crop_path'])
    else:
        ancillary_df = df_dets

    # Cluster
    # Compute in batches of 300K
    batch_size = 300
    num_batches = int(np.ceil(len(images) / batch_size))

    batch_df = pd.DataFrame()
    batch_df['batch'] = -1

    # Remove any existing cluster images in the output_path
    for c in (output_path / prefix).parent.rglob(f'{prefix}_*cluster*.png'):
        c.unlink()

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(images))
        debug(f'Processing batch {i + 1} of {num_batches}...')

        # Get the embeddings for the batch
        image_emb = []
        for filename in images[start:end]:
            emb, label, score = fetch_embedding(model, filename)
            image_emb.append(emb)
        image_emb = np.array(image_emb)

        # Cluster the images
        df = _run_hdbscan_assign(image_emb,
                                 alpha,
                                 cluster_selection_epsilon,
                                 cluster_selection_method,
                                 algorithm,
                                 min_cluster_size,
                                 min_samples,
                                 use_tsne,
                                 ancillary_df,
                                 i)
        df['batch'] = i
        df['crop_path'] = images[start:end]
        batch_df = pd.concat([batch_df, df], ignore_index=True)

    # Merge
    final_df, avg_sim_scores = _merge(batch_df, min_similarity, model)

    # Get the average similarity across all clusters
    avg_similarity = np.mean(list(avg_sim_scores.values()))

    info(f'Average similarity: {avg_similarity:.2f} min {min_similarity:.2f}  ')
    unique_clusters = final_df['cluster_reindex'].unique()

    # Drop the -1 value which are the noise
    unique_clusters = unique_clusters.tolist()
    unique_clusters.remove(-1)
    unique_clusters.sort()

    if len(unique_clusters) == 0:
        warn('No clusters found')

    info(f'Found {len(unique_clusters)} clusters with an average similarity of {avg_similarity:.2f} ')

    # Save the exemplar embeddings to a dataframe with some metadata
    exemplar_df = final_df[final_df['exemplar'] == 1].copy()
    exemplar_df['model'] = model
    # Add the embedding to the exemplar dataframe
    exemplar_df['embedding'] = [fetch_embedding(model, filename)[0] for filename in exemplar_df['crop_path']]
    exemplar_df.to_csv(output_path / f'{prefix}_exemplars.csv', index=False)

    # Visualize the clusters
    if not skip_visualization:
        _visualize_clusters(final_df, unique_clusters, output_path, prefix, avg_sim_scores)

    num_samples = final_df.shape[0]
    clustered = final_df['cluster'].values >= 0
    coverage = np.sum(clustered) / num_samples
    info(f'Coverage: {coverage:.2f} ({np.sum(clustered)}/{num_samples})')

    hdbscan_params = {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "cluster_selection_method": "leaf",
        "metric": "precomputed",
        "algorithm": algorithm,
        "alpha": alpha,
        "cluster_selection_epsilon": cluster_selection_epsilon
    }
    params = {
        "coverage": coverage,
        "num_clusters": len(unique_clusters),
        "hdbscan_params": hdbscan_params,
    }
    info(f"Parameters {params}")

    with open(output_path / f'{prefix}_params.json', 'w') as f:
        json.dump(params, f, indent=4)

    exemplar_df.to_csv(output_path / f'{prefix}_exemplars.csv', index=False)
    final_df.to_csv(output_path / f'{prefix}_clusters.csv', index=False)
    info(f'Wrote {output_path / f"{prefix}_clusters.csv"}')
    info(f'Wrote {output_path / f"{prefix}_exemplars.csv"}')
    info(f'Wrote {output_path / f"{prefix}_params.json"}')

    return final_df

