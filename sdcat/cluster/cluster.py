# sdcat, Apache-2.0 license
# Filename: sdcat/cluster/cluster.py
# Description: Clustering using vision transformer features and HDBSCAN density-based clustering

import multiprocessing
from importlib.util import find_spec
import pandas as pd
from pathlib import Path
import os
import json
import seaborn as sns
import numpy as np
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sdcat.logger import info, warn, debug, err
from sdcat.cluster.utils import cluster_grid, crop_square_image, square_image
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


def _run_hdbscan_assign(
        prefix: str,
        image_emb: np.ndarray,
        alpha: float,
        cluster_selection_epsilon: float,
        cluster_selection_method: str,
        min_similarity: float,
        min_cluster_size: int,
        min_samples: int,
        use_tsne: bool,
        ancillary_df: pd.DataFrame,
        out_path: Path) -> tuple:
    """
    Cluster the embeddings using HDBSCAN and reassign unclustered to the nearest exemplars.
    :param prefix:  A unique prefix to save artifacts from clustering
    :param image_emb:  The embeddings to cluster from the model
    :param alpha:  The alpha parameter for HDBSCAN
    :param cluster_selection_epsilon:  The epsilon parameter for HDBSCAN
    :param cluster_selection_method:  The method to use for cluster selection, 'leaf' or 'eom'
    :param min_similarity:  The minimum similarity score to use for clustering reassignment
    :param min_cluster_size:  The minimum number of samples in a cluster
    :param min_samples:   The number of samples in a neighborhood for a point
    :param use_tsne:  Whether to use t-SNE for dimensionality reduction
    :param ancillary_df:  (optional) Ancillary data to include in the clustering
    :param out_path:  The output path to save the clustering artifacts to
    :return: The average similarity score for each cluster, exemplar_df, cluster ids, cluster means, and coverage
    """
    info(f'Clustering using HDBSCAN using alpha {alpha} cluster_selection_epsilon {cluster_selection_epsilon} '
         f'min_samples {min_samples} use_tsne {use_tsne} ...')

    # Remove any existing cluster images in the output_path
    for c in out_path.parent.rglob(f'{prefix}_*cluster*.png'):
        c.unlink()

    # Add in any numerical ancillary data and replace NaNs with 0
    df = pd.DataFrame(image_emb)
    numerical = ancillary_df.select_dtypes(include=["float", "int"])
    if not numerical.empty:
        numerical = numerical.fillna(0)

        # Normalize the numerical data from 0 to 1 and add it to the dataframe
        numerical = (numerical - numerical.min()) / (numerical.max() - numerical.min())

        df = pd.merge(df, numerical, left_index=True, right_index=True, how='left')
        df = df.fillna(0)

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
    x = MinMaxScaler().fit_transform(embedding) # scale the embedding to 0-1

    # Cluster the embeddings using HDBSCAN
    if have_gpu:
        scan = cuHDBSCAN(
            metric='euclidean',  # 'precomputed' does not work with cuHDBSCAN
            allow_single_cluster=True,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            alpha=alpha,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method=cluster_selection_method).fit_predict(x)
        labels = scan.fit_predict(x)
    else:
        scan = HDBSCAN(
                metric='l2',
                allow_single_cluster=True,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                alpha=alpha,
                cluster_selection_epsilon=cluster_selection_epsilon,
                cluster_selection_method=cluster_selection_method)
        labels = scan.fit_predict(x)

    # Get the unique clusters and sort them; -1 are unassigned clusters
    cluster_df = pd.DataFrame(labels, columns=['cluster'])
    unique_clusters = cluster_df['cluster'].unique().tolist()
    unique_clusters.sort()
    info(f"Number of clusters including unassigned -1 cluster: {len(unique_clusters)}")

    cluster_df['score'] = scan.probabilities_
    # Get the index of the highest scores for each unique cluster sorted in increasing order
    # and use this as a representative image for the cluster
    max_scores = cluster_df.sort_values('cluster', ascending=True).groupby('cluster')['score'].idxmax()
    # Remove the last index which is the -1 cluster
    max_scores = max_scores[:-1]

    # If all the clusters are unassigned, then use all the samples as exemplars,
    # and assign them to the unknown cluster. If embedding is empty, this is also the case (failed to extract embeddings)
    if len(unique_clusters) == 1 and unique_clusters[0] == -1:
        avg_sim_scores = []
        exemplar_df = pd.DataFrame()
        exemplar_df['cluster'] = len(x) * ['Unknown']
        exemplar_df['embedding'] = x.tolist()
        exemplar_df['image_path'] = ancillary_df['image_path'].tolist()
        clusters = []
        cluster_means = []
        coverage = 0.0
        return avg_sim_scores, exemplar_df, clusters, cluster_means, coverage

    # Get the representative embeddings for the max scoring examplars for each cluster and store them in a numpy array
    exemplar_emb = [image_emb[i] for i in max_scores]
    exemplar_emb = np.array(exemplar_emb)

    # Save the exemplar embeddings to a dataframe with some metadata
    exemplar_df = pd.DataFrame()
    exemplar_df['cluster'] = [f'Unknown C{i}' for i in range(0, len(max_scores))]
    if ancillary_df is not None and 'image_path' in ancillary_df.columns:
        exemplar_df['image_path'] = ancillary_df.iloc[max_scores]['image_path'].tolist()
    exemplar_df['embedding'] = exemplar_emb.tolist()

    # Reassign the unknowns to the closest cluster - this is only needed if the coverage is less than 1
    clustered = labels >= 0
    coverage = np.sum(clustered) / num_samples
    if coverage < 1.0:
        # Reassign based on the closest distance to exemplar
        for i, label in enumerate(labels):
            if label == -1:
                similarity_scores = cosine_similarity(image_emb[i].reshape(1, -1), exemplar_emb)
                closest_match_index = np.argmax(similarity_scores)
                # Only reassign if the similarity score is above the threshold
                if similarity_scores[0][closest_match_index] >= min_similarity:
                    labels[i] = closest_match_index

    clusters = [[] for _ in range(len(unique_clusters))]

    # Assign indices to the clusters
    for i in range(0, len(labels)):
        clusters[labels[i]].append(i)

    # Compute the average similarity score for each cluster
    avg_sim_scores = []
    for i, c in enumerate(clusters):
        debug(f'Computing similarity for cluster {i} with {len(c)} samples')
        if len(c) == 0:
            avg_sim_scores.append(0)
            continue
        cosine_sim_matrix = cosine_similarity(image_emb[c])
        avg_sim_scores.append(np.mean(cosine_sim_matrix))

    # Compute the cluster means
    cluster_means = []
    for c in clusters:
        cluster_means.append(np.mean(image_emb[c], axis=0))

    hdbscan_params = {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "cluster_selection_method": "leaf",
        "metric": "precomputed",
        "alpha": alpha,
        "cluster_selection_epsilon": cluster_selection_epsilon
    }
    clustered = labels >= 0
    coverage = np.sum(clustered) / num_samples
    params = {
        "coverage": coverage,
        "num_clusters": len(unique_clusters),
        "hdbscan_params": hdbscan_params,
    }
    info(f"Parameters {params}")

    # Cannot use init='spectral' when n_components is >= num_samples - default to 'random' instead
    n_components = min(2, num_samples)
    if n_components >= num_samples:
        init = 'random'
    else:
        init = 'spectral'

    # Reduce the dimensionality of the embeddings using UMAP to 2 dimensions to visualize the clusters
    if have_gpu:
        xx = cuUMAP(init=init,
                    n_components=2,
                    n_neighbors=3,
                    min_dist=0.1,
                    metric='euclidean').fit_transform(df.values)
    else:
        xx = UMAP(init=init,
                  n_components=2,
                  metric='cosine',
                  low_memory=True).fit_transform(df.values)

    df = pd.DataFrame({'x': xx[clustered, 0], 'y': xx[clustered, 1], 'labels': labels[clustered]})
    p = sns.jointplot(data=df, x='x', y='y', hue='labels')
    p.savefig(f"{out_path}/{prefix}_summary.png")
    info(f"Saved {out_path}/{prefix}_summary.png")

    with open(f'{out_path}/{prefix}_summary.json', 'w') as f:
        json.dump(params, f)

    return avg_sim_scores, exemplar_df, clusters, cluster_means, coverage


def cluster_vits(
        prefix: str,
        model: str,
        df_dets: pd.DataFrame,
        output_path: Path,
        alpha: float,
        cluster_selection_epsilon: float,
        cluster_selection_method: str,
        min_similarity: float,
        min_cluster_size: int,
        min_samples: int,
        use_tsne: bool = False,
        roi: bool = False) -> pd.DataFrame:
    """  Cluster the crops using the VITS embeddings.
    :param prefix:  A unique prefix to save artifacts from clustering
    :param model: The model to use for clustering
    :param df_dets: The dataframe with the detections
    :param output_path: The output path to save the clustering artifacts to
    :param roi:  Whether the detections are already cropped to the ROI
    :param cluster_selection_epsilon: The epsilon parameter for HDBSCAN
    :param cluster_selection_method: The method to use for cluster selection, 'leaf' or 'eom'
    :param alpha: The alpha parameter for HDBSCAN
    :param min_similarity: The minimum similarity score to use for -1 cluster reassignment
    :param min_cluster_size: The minimum number of samples in a cluster
    :param min_samples:The number of samples in a neighborhood for a point
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
            info(f'ROI crops already exist. Creating square crops in parallel using {num_processes} processes...')
            with multiprocessing.Pool(num_processes) as pool:
                args = [(row, 224) for index, row in df_dets.iterrows()]
                pool.starmap(square_image, args)
        else:
            # Crop and squaring the images in parallel using multiprocessing to speed up the processing
            info(f'Cropping {len(df_dets)} detections in parallel using {num_processes} processes...')
            with multiprocessing.Pool(num_processes) as pool:
                args = [(row, 224) for index, row in df_dets.iterrows()]
                pool.starmap(crop_square_image, args)

    # Drop any rows with crop_path that have files that don't exist - sometimes the crops fail
    df_dets = df_dets[df_dets['crop_path'].apply(lambda x: os.path.exists(x))]

    # Get the list of images to crop
    images = df_dets['crop_path'].tolist()

    # Count how many files have the .npy extension
    num_cached = sum([has_cached_embedding(model, filename) for filename in images])

    # Skip the embedding extraction if all the embeddings are cached
    if num_cached != len(images):
        debug(f'Extracted embeddings from {num_cached} images...')
        compute_norm_embedding(model, images)

    # Fetch the cached embeddings
    debug(f'Fetching embeddings ...')
    image_emb = []
    for filename in images:
        emb = fetch_embedding(model, filename)
        if len(emb) == 0:
            # If the embeddings are zero, then the extraction failed; add a zero array
            image_emb.append(np.zeros(384, dtype=np.float32))
        else:
            image_emb.append(emb)

    # If the embeddings are zero, then the extraction failed
    num_failed = [i for i, e in enumerate(image_emb) if np.all(e == 0)]
    if len(num_failed) == len(images):
        warn('Failed to extract embeddings from all images')
        return pd.DataFrame()

    image_emb = np.array(image_emb)

    if not (output_path / prefix).exists():
        (output_path / prefix).mkdir(parents=True)

    # Remove everything except ancillary data to include in clustering
    columns = ['x', 'y', 'xx', 'xy', 'w', 'h', 'image_width', 'image_height', 'cluster_id', 'cluster', 'score',
               'class', 'image_path', 'crop_path']
    # Check if the columns exist in the dataframe
    if all(col in df_dets.columns for col in columns):
        ancillary_df = df_dets.drop(
            columns=['x', 'y', 'xx', 'xy', 'w', 'h', 'image_width', 'image_height', 'cluster_id', 'cluster', 'score',
                     'class', 'image_path', 'crop_path'])
    else:
        ancillary_df = df_dets

    # Cluster the images
    cluster_sim, exemplar_df, unique_clusters, cluster_means, coverage = _run_hdbscan_assign(prefix,
                                                                                             image_emb,
                                                                                             alpha,
                                                                                             cluster_selection_epsilon,
                                                                                             cluster_selection_method,
                                                                                             min_similarity,
                                                                                             min_cluster_size,
                                                                                             min_samples,
                                                                                             use_tsne,
                                                                                             ancillary_df,
                                                                                             output_path / prefix)

    # Get the average similarity across all clusters
    avg_similarity = np.mean(cluster_sim)

    info(f'Average similarity: {avg_similarity:.2f} min {min_similarity:.2f}  ')

    if len(unique_clusters) == 0:
        warn('No clusters found')
        # Save the exemplar embeddings with the model type
        exemplar_df['model'] = model
        exemplar_df.to_csv(output_path / f'{prefix}_exemplars.csv', index=False)
        return None

    info(f'Found {len(unique_clusters)} clusters with an average similarity of {avg_similarity:.2f} ')

    # Assign the cluster ids to the dataframe
    for cluster_id, cluster in enumerate(unique_clusters):
        for idx in cluster:
            debug(f'Adding {images[idx]} to cluster id {cluster_id} ')
            df_dets.loc[df_dets['crop_path'] == images[idx], 'cluster'] = cluster_id

    # For each cluster  let's create a grid of the images to check the quality of the clustering results
    num_processes = min(multiprocessing.cpu_count(), len(unique_clusters))
    info(f'Using {num_processes} processes to visualize the {len(unique_clusters)} clusters')

    if num_processes == 0:
        err(f'No processes available to visualize the clusters')
        return None

    # Use a pool of processes to speed up the visualization of the clusters
    with multiprocessing.Pool(num_processes) as pool:
        args = [(prefix,  # prefix
                 cluster_sim[cluster_id],  # average similarity for the cluster
                 cluster_id,  # cluster id
                 unique_clusters[cluster_id],  # cluster indices
                 4 if len(unique_clusters[cluster_id]) < 50 else 8,  # grid size; larger clusters get larger grids
                 [images[idx] for idx in unique_clusters[cluster_id]],  # images in the cluster
                 output_path / prefix) for cluster_id in
                range(0, len(unique_clusters))]
        pool.starmap(cluster_grid, args)

    # Save the exemplar embeddings with the model type
    exemplar_df['model'] = model
    exemplar_df.to_csv(output_path / f'{prefix}_exemplars.csv', index=False)

    info(f"Number of images {len(images)}")
    info(f"Number of clusters {len(unique_clusters)}")
    info(f"Coverage {coverage:.2f}")

    return df_dets
