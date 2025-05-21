# sdcat, Apache-2.0 license
# Filename: sdcat/cluster/cluster.py
# Description: Clustering using vision transformer features and HDBSCAN density-based clustering
import warnings
from importlib.util import find_spec

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import os
import pandas
import seaborn as sns
import tqdm
import modin.pandas as pd
import numpy as np
from umap import UMAP
from hdbscan import HDBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics.pairwise import cosine_similarity
from sdcat.logger import info, warn, debug
from sdcat.cluster.utils import cluster_grid, crop_square_image, clean_bad_images
from sdcat.cluster.embedding import fetch_embedding, has_cached_embedding, compute_norm_embedding
from sdcat import __version__ as sdcat_version

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


def _summarize_clusters(df: pd.DataFrame, output_path: Path, prefix: str,
                            cluster_sim: list, model: str, skip_visualization: bool, hdbscan_params: dict[str]) -> dict:
    """
    Summarize and optionally visualize the clusters using t-SNE or UMAP in grids
    """
    clustered = df[df['cluster'] != -1]
    labels = clustered['cluster'].values
    num_samples = len(labels)
    clusters = np.unique(labels)
    num_clusters = len(clusters)

    # Get the top 20 predicted classes in percent
    top20 = df['class'].value_counts(normalize=True).head(5) * 100
    top20 = top20.sort_index().round(2)

    summary = {
        "dataset": {
            "output": str(output_path),
            "clustering_algorithm": "HDBSCAN",
            "clustering_parameters":  hdbscan_params,
            "feature_embedding_model": str(model),
        },
        "statistics": {
            "total_clusters": num_clusters,
            "cluster_coverage": f"{float(np.sum(labels) / num_samples):.2f} ({100*float(np.sum(labels) / num_samples):.2f}%)",
        }
    }

    image_paths = {c: df.loc[df['cluster'] == c, 'crop_path'].tolist()[:150] for c in clusters}

    for name, total in top20.items():
        summary["statistics"]["top_predictions"] = {
            "class": name,
            "percentage": f"{total:.2f}%",
        }

    if not skip_visualization:
        # Create a grid of the images to check the quality of the clustering results
        num_processes = min(os.cpu_count(), num_clusters)
        info(f'Using {num_processes} processes to visualize the {num_clusters} clusters')

        # Use a pool of processes to speed up the visualization of the clusters
        # Skip modin here because it does not offer much speedup
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for c in image_paths:
                grid_size = 4 if len(image_paths[c]) < 50 else 8  # grid size; larger clusters get larger grids
                futures.append(
                    executor.submit(
                        cluster_grid,
                        prefix,
                        cluster_sim[c],
                        c,
                        grid_size,
                        image_paths[c],
                        output_path / prefix
                    )
                )
            for future in as_completed(futures):
                future.result()

        # Cannot use init='spectral' when n_components is >= num_samples - default to 'random' instead
        n_components = min(2, num_samples)
        if n_components >= num_samples:
            init = 'random'
        else:
            init = 'spectral'

        # Reduce the dimensionality of the embeddings using UMAP to 2 dimensions to visualize the clusters
        # Only use the exemplars and a random sample of 5000 images to speed up the visualization
        sampled_df = clustered.sample(n=min(num_samples-1, 5000), random_state=42, replace=False)
        sampled_emb = [fetch_embedding(model, filename)[0] for filename in sampled_df['crop_path']]
        np_data = np.array(sampled_emb)

        n_neighbors = min(15, num_samples - 1)
        info(f'Using {n_neighbors} neighbors for dimensional reduction')
        if n_neighbors < 2:
            warn('Using PCA instead of UMAP to reduce for cluster 2D plot')
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            xx = pca.fit_transform(np_data)
        else:
            if have_gpu:
                info('Using GPU to reduce for cluster 2D plot')
                xx = cuUMAP(init=init,
                            n_components=2,
                            n_neighbors=n_neighbors,
                            min_dist=0.1,
                            metric='euclidean').fit_transform(np_data)
            else:
                info('Using UMAP to reduce for cluster 2D plot')
                xx = UMAP(init=init,
                          n_components=2,
                          n_neighbors=n_neighbors,
                          metric='cosine').fit_transform(np_data)

        df_joint = pd.DataFrame({'x': xx[:,0], 'y': xx[:,1], 'labels': sampled_df['cluster'].values})
        p = sns.jointplot(data=df_joint, x='x', y='y', hue='labels')
        p.fig.suptitle(f"{prefix}\nsdcat_version {sdcat_version}\nClusters {num_clusters} with {num_samples} samples", fontsize=14)
        p.fig.subplots_adjust(top=0.80)
        p.savefig(f"{output_path}/{prefix}_summary.png")
        info(f"Saved {output_path}/{prefix}_summary.png")

    return summary

def _similarity_merge(
        df: pd.DataFrame,
        exemplar_emb: np.ndarray,
        min_similarity: float,
        model: str
    ) -> (pd.DataFrame, dict):
    """
    Merge clusters based on the linkage of the cosine similarity of their embeddings.
    """
    unique_clusters_before = df['cluster'].unique()

    # Assign noise -1 cluster to the nearest exemplar
    noise_indices = df[df['cluster'] == -1].index
    info('Assigning noise clusters to nearest exemplar ...')
    for i in tqdm.tqdm(noise_indices):
        noise_emb, _, _ = fetch_embedding(model, df.iloc[i]['crop_path'])
        sim = cosine_similarity([noise_emb], exemplar_emb)
        cluster = np.argmax(sim)
        score = np.max(sim)
        if score > min_similarity:
            df.iloc[i]['cluster'] = cluster
            debug(f'Noise {i} is now {cluster} {score:.2f}')


    # Get the exemplar embeddings for the clusters again with noise clusters assigned to the nearest exemplar
    max_scores = df.sort_values('cluster', ascending=True).groupby('cluster')['HDBSCAN_probability'].idxmax()
    # Remove the first element which is the -1 cluster
    if -1 in max_scores.index:
        max_scores = max_scores.drop(-1)
    exemplar_emb = [fetch_embedding(model, filename)[0] for filename in df.loc[max_scores]['crop_path']]
    df.loc[max_scores, 'exemplar'] = 1

    info(f'Merging clusters with similarity threshold {min_similarity:.2f} ...')
    info(f"Maximum cluster id: {df['cluster'].values.max()} minimum cluster id: {df['cluster'].values.min()} unique clusters: {len(unique_clusters_before)}")
    linkage_matrix = linkage(exemplar_emb, method='complete', metric='cosine')
    linkage_clusters = fcluster(linkage_matrix, 1 - min_similarity, criterion='distance')

    info(f'Unique clusters before merging: {len(unique_clusters_before)}')
    info(f'Linkage matrix size: {len(linkage_matrix)}')

    # If the cluster labels are all the same, then we have a single cluster and we can't merge
    if len(np.unique(linkage_clusters)) == 1:
        info(f'No clusters to merge')
    else:
        # Assign the exemplar clusters to the original clusters based on the linkage matrix
        for i, old_cluster in enumerate(df['cluster'].values):
            if old_cluster == -1:
                continue
            if old_cluster > len(linkage_clusters) - 1 :
                warn(f'Cluster {old_cluster} is not in the linkage matrix')
                continue
            new_cluster = linkage_clusters[old_cluster]
            debug(f'Assigning cluster {old_cluster} to {new_cluster}')
            df.iloc[i]['cluster'] = new_cluster

        unique_clusters_after = df['cluster'].unique()
        info(f'Unique clusters after merging: {len(unique_clusters_after)}')

    def compute_cluster_avg_similarity(cluster_df):
        if len(cluster_df) == 0:
            return None

        cluster_emb = [fetch_embedding(model, f)[0] for f in cluster_df['crop_path']]
        cluster_emb = np.array(cluster_emb)

        return np.mean(cosine_similarity(cluster_emb))

    # Group by cluster and apply the function in parallel
    info("Computing average similarity scores for each cluster ...")
    cluster_scores = df.groupby("cluster").apply(compute_cluster_avg_similarity)
    avg_sim_scores = cluster_scores.dropna().to_dict()
    return df, avg_sim_scores


def _run_hdbscan_assign(
        df: pd.DataFrame,
        alpha: float,
        cluster_selection_epsilon: float,
        cluster_selection_method: str,
        algorithm: str,
        min_cluster_size: int,
        min_samples: int,
        use_tsne: bool,
        cluster_offset: int) -> pandas.DataFrame:
    """
    Cluster the features using HDBSCAN
    :param alpha:  The alpha parameter for HDBSCAN
    :param cluster_selection_epsilon:  The epsilon parameter for HDBSCAN
    :param algorithm:  The algorithm to use for clustering, 'best' or 'generic' or 'prims_kdtree' or 'boruvka_kdtree'
    :param cluster_selection_method:  The method to use for cluster selection, 'leaf' or 'eom'
    :param min_cluster_size:  The minimum number of samples in a cluster
    :param min_samples:   The number of samples in a neighborhood for a point
    :param use_tsne:  Whether to use t-SNE for dimensionality reduction
    :param cluster_offset:  Offset to add to the cluster IDs
    :return: pandas.DataFrame with the cluster assignments
    """
    info(f'Clustering using HDBSCAN with: '
        f'cluster_offset {cluster_offset},'
        f'alpha {alpha},'
        f'algorithm {algorithm},'
        f'cluster_selection_epsilon {cluster_selection_epsilon},'
        f'min_samples {min_samples},'        
        f'min_cluster_size {min_cluster_size},'
        f'cluster_selection_method {cluster_selection_method},'
        f'use_tsne {use_tsne} ...')

    # Get the number of samples which is the number of rows in the dataframe
    num_samples = df.shape[0]

    # Perplexity must be less than the number of samples
    perplexity = min(30, num_samples - 1)

    features = np.stack(df['embedding'].values)
    # Add in other features if present
    if df.shape[1] > 1:
        for col in df.columns:
            if col != 'embedding':
                features = np.concatenate((features, df[col].values.reshape(-1, 1)), axis=1)

    # TSN-E does not work well when we have a few samples
    if num_samples > 100 and use_tsne:
        tsne = TSNE(n_components=2, perplexity=perplexity, metric="cosine", n_jobs=8, random_state=42, verbose=True)
        x = tsne.fit_transform(features)
    else:
        x = features

    # Cluster the embeddings using HDBSCAN
    if have_gpu:
        scan = cuHDBSCAN(
            metric='euclidean',  # 'precomputed' does not work with cuHDBSCAN
            allow_single_cluster=True,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            alpha=alpha,
            algorithm=algorithm, # Should this be 'best' or 'generic'?
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

    info(f"Number of clusters including unassigned -1 cluster: {len(np.unique(labels))}")

    # Save the probabilities for each cluster which is used for merging
    df['HDBSCAN_probability'] = scan.probabilities_
    # Add the cluster_offset to the cluster labels, but only if they are not -1
    labels = np.where(labels == -1, -1, labels + cluster_offset)
    df['cluster'] = labels
    unique_clusters = df['cluster'].unique()
    info(f"Index: {df.index[0]} to {df.index[-1]}")
    info(f"Maximum cluster id: {df['cluster'].values.max()} minimum cluster id: {df['cluster'].values.min()} unique clusters: {len(unique_clusters)}")
    return df

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
) -> dict or None:
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
    :param device: The device to use for clustering, 'cpu' or 'cuda' or 'cuda:0' or 'cuda:1'
    :param weighted_score: Whether to weight score for the prediction from vits model with detection weight
    :param use_vits: Set to using the predictions from the vits cluster model
    :param skip_visualization: Whether to skip the visualization of the clusters
    :param remove_bad_images: Whether to remove bad images from the clustering
    :param use_tsne: Whether to use t-SNE for dimensionality reduction
    :param batch_size: The batch size to use for clustering; maximize for speed for your GPU
    :return:  a dictionary with a summary."""
    warnings.filterwarnings('ignore')
    # If there are no detections, return an empty dataframe
    if df_dets.empty:
        warn('No detections found in {detections} ')
        return None

    # If the detections are not cropped, crop them to a square
    if not roi:

        # Count how many crops are already square
        num_square = df_dets['crop_path'].apply(lambda x: os.path.exists(x)).sum()

        if num_square == len(df_dets):
            info(f'All {len(df_dets)} detections are already cropped to square')
        else:
            def crop_square_wrapper(row):
                return crop_square_image(row, 224)

            info(f'Cropping {len(df_dets)} detections...')
            df_dets.apply(crop_square_wrapper, axis=1)

    # Drop any rows with crop_path that have files that don't exist - sometimes the crops fail
    df_dets = df_dets[df_dets['crop_path'].apply(lambda x: os.path.exists(x))]
    if df_dets.empty:
        warn('No detections found in {detections} ')
        return None

    df_dets = df_dets.sort_index()
    df_dets['exemplar'] = 0
    df_dets['cluster'] = -1
    df_dets['HDBSCAN_probability'] = 0

    # Get the list of images to crop
    crop_paths = df_dets['crop_path'].values

    # Count how many files have the .npy extension
    num_cached = sum([has_cached_embedding(model, filename) for filename in crop_paths])
    info(f'Found {num_cached} cached embeddings for {len(crop_paths)} images')

    # Skip the embedding extraction if all the embeddings are cached
    if num_cached != len(crop_paths):
        debug(f'Extracted embeddings from {len(crop_paths)} images using model {model}...')
        compute_norm_embedding(model, crop_paths, device, batch_size)

    if use_vits:
        debug('Compute weighted scores ...')
        for filename in crop_paths:
            _, label, score = fetch_embedding(model, filename)
            weight = 1
            if weighted_score:
                weight = df_dets.loc[df_dets['crop_path'] == filename, 'score'].values[0]
                # Weight cannot be zero or negative
                if weight <= 0:
                    weight = .0001
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
        ancillary_df = None

    if ancillary_df:
        # Add in any numerical ancillary data and replace NaNs with 0
        numerical_df = ancillary_df.select_dtypes(include=["float", "int"])
        numerical_df = numerical_df.fillna(0)
        # Normalize the numerical data from 0 to 1 - this is an important step!
        ancillary_df = (numerical_df - numerical_df.min()) / (numerical_df.max() - numerical_df.min())

    # Cluster
    # Compute in batches of 50K; this works for the 8 block models on any GPU
    batch_size = 50000
    num_batches = int(np.ceil(len(crop_paths) / batch_size))

    info(f'Remove any existing cluster grid images in the output_path in {(output_path / prefix)}')
    cluster_grids = (output_path / prefix).rglob(f'{prefix}_*cluster*.png')
    for c in cluster_grids:
        c.unlink()

    cluster_offset = 0 # Start with 0 for the first batch
    info(f'Processing images in batches of {batch_size} ...')
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(crop_paths))
        info(f'Processing batch {i + 1} of {num_batches} {start} to {end}...')

        # Get the embeddings for the batch
        df_batch = df_dets.iloc[start:end].copy()
        df_batch["embedding"] = [fetch_embedding(model, filename)[0] for filename in crop_paths[start:end]]
        df_batch.index = df_dets.iloc[start:end].index

        if remove_bad_images:
            info(f'Cleaning bad images from {len(df_batch)} ')
            size_before = len(df_batch)
            filepaths = df_batch['crop_path'].values.tolist()
            bad_images = clean_bad_images(filepaths)
            df_batch = df_batch[~df_batch['crop_path'].isin(bad_images)]
            size_after = len(df_batch)
            info(f'Removed {size_before - size_after} detections using cleanvision in batch {i + 1} of {num_batches}...')
            if size_after == 0:
                warn(f'No detections left in batch {i + 1} of {num_batches} after cleaning bad images')
                continue

        # Only keep the columns needed for clustering
        keep_columns = ['area', 'saliency', 'w', 'h', 'embedding', 'crop_path']
        for col in df_batch.columns:
            if col not in keep_columns:
                df_batch = df_batch.drop(columns=[col], errors='ignore')

        # Add in the ancillary data if present. Assume keyed by crop_path
        if ancillary_df:
            ancillary_data = ancillary_data.select_dtypes(include=["float", "int"])
            for filename in crop_paths[start:end]:
                # Get the ancillary data for the image
                ancillary_data = ancillary_df.loc[ancillary_df['crop_path'] == filename]
                if ancillary_data.empty:
                    ancillary_data = pd.Series([0] * len(ancillary_df.columns), index=ancillary_df.columns)
                else:
                    ancillary_data = ancillary_data.iloc[0]
                df_batch.loc[df_batch['crop_path'] == filename, ancillary_df.columns] = ancillary_data

        df_batch = df_batch.drop(columns=['crop_path'], errors='ignore') # drop the crop_path column as only floats and ints are needed
        df_assign = _run_hdbscan_assign(df_batch,
                                 alpha,
                                 cluster_selection_epsilon,
                                 cluster_selection_method,
                                 algorithm,
                                 min_cluster_size,
                                 min_samples,
                                 use_tsne,
                                 cluster_offset)

        df_dets['cluster'].update(df_assign['cluster'])
        df_dets['HDBSCAN_probability'].update(df_assign['HDBSCAN_probability'])
        info(f'Unique clusters after {i + 1} of {num_batches}: {len(df_dets["cluster"].unique())}')
        cluster_offset = df_dets['cluster'].values.max() + 1

    max_scores = df_dets.sort_values('cluster', ascending=True).groupby('cluster')['HDBSCAN_probability'].idxmax()
    # Remove the first element which is the -1 cluster
    if -1 in max_scores.index:
        max_scores = max_scores.drop(-1)

    # Get the representative embeddings for the max scoring exemplars for each cluster and store them in a numpy array
    exemplar_emb = [fetch_embedding(model, filename)[0] for filename in df_dets.loc[max_scores]['crop_path']]
    exemplar_emb = np.array(exemplar_emb)

    # Merge by similarity
    df_dets_final, avg_sim_scores = _similarity_merge(df_dets, exemplar_emb, min_similarity, model)

    # Drop any rows with NaN values in the cluster column
    df_dets_final = df_dets_final.dropna(subset=['cluster'])

    # Get the average similarity across all clusters
    avg_similarity = np.mean(list(avg_sim_scores.values()))

    info(f'Average similarity: {avg_similarity:.2f} min {min_similarity:.2f}  ')
    unique_clusters = df_dets_final['cluster'].unique()

    # Drop the -1 value which is the noise cluster
    if -1 in unique_clusters:
        unique_clusters = unique_clusters[unique_clusters != -1]
    unique_clusters.sort()

    if len(unique_clusters) == 0:
        warn('No clusters found')

    info(f'Found {len(unique_clusters)} clusters with an average similarity of {avg_similarity:.2f} ')

    num_samples = df_dets.shape[0]
    clustered = df_dets['cluster'].values != -1
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

    df_dets_final.to_csv(output_path / f'{prefix}_cluster_detections.csv')
    info(f'Saved {output_path / f"{prefix}_cluster_detections.csv"}')
    df_dets_final.to_parquet(output_path / f'{prefix}_cluster_detections.parquet')
    info(f'Saved {output_path / f"{prefix}_cluster_detections.parquet"}')

    # Return a summary of the clusters
    return _summarize_clusters(df_dets_final,
                               output_path,
                               prefix,
                               avg_sim_scores,
                               model,
                               skip_visualization,
                               hdbscan_params)

